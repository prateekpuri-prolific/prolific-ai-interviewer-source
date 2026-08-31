#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GCP_PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
GCP_REGION="${GCP_REGION:-europe-west2}"
CLOUD_RUN_SERVICE="${CLOUD_RUN_SERVICE:-oasis-pilot}"
ARTIFACT_REPOSITORY="${ARTIFACT_REPOSITORY:-cloud-run-source-deploy}"
RUNTIME_SERVICE_ACCOUNT_EMAIL="${RUNTIME_SERVICE_ACCOUNT_EMAIL:-taskcrafter-staging-run@${GCP_PROJECT}.iam.gserviceaccount.com}"
SQL_INSTANCE="${SQL_INSTANCE:-oasis-postgres}"
BUILD_METHOD="${BUILD_METHOD:-cloud-build}"

if [[ -z "$GCP_PROJECT" || "$GCP_PROJECT" == "(unset)" ]]; then
  echo "Set GCP_PROJECT or select a gcloud project." >&2
  exit 1
fi
if ! gcloud auth list --filter=status:ACTIVE --format='value(account)' | grep -q .; then
  echo "No active gcloud account. Run: gcloud auth login" >&2
  exit 1
fi
if [[ -n "$(git -C "$ROOT_DIR" status --porcelain)" ]]; then
  echo "Refusing to deploy from a dirty working tree." >&2
  exit 1
fi

SOURCE_CODE_URL="$("$ROOT_DIR/deploy/source-release/publish.sh")"
SOURCE_SHA="$(git -C "$ROOT_DIR" rev-parse --short=12 HEAD)"
IMAGE_ROOT="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACT_REPOSITORY}"
BACKEND_IMAGE="${IMAGE_ROOT}/oasis-backend:${SOURCE_SHA}"
FRONTEND_IMAGE="${IMAGE_ROOT}/oasis-frontend:${SOURCE_SHA}"
CLOUD_SQL_CONNECTION_NAME="$(gcloud sql instances describe "$SQL_INSTANCE" --project "$GCP_PROJECT" --format='value(connectionName)')"

for secret in oasis-app-secret-key oasis-postgres-password oasis-admin-password oasis-openai-api-key; do
  gcloud secrets add-iam-policy-binding "$secret" \
    --project "$GCP_PROJECT" \
    --member="serviceAccount:${RUNTIME_SERVICE_ACCOUNT_EMAIL}" \
    --role=roles/secretmanager.secretAccessor \
    --condition=None \
    --quiet >/dev/null
done

if [[ "$BUILD_METHOD" == "local" ]]; then
  gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet
  docker build --platform linux/amd64 --tag "$BACKEND_IMAGE" "$ROOT_DIR/backend"
  docker push "$BACKEND_IMAGE"
  docker build --platform linux/amd64 --tag "$FRONTEND_IMAGE" "$ROOT_DIR/frontend"
  docker push "$FRONTEND_IMAGE"
elif [[ "$BUILD_METHOD" == "cloud-build" ]]; then
  gcloud builds submit "$ROOT_DIR/backend" \
    --project "$GCP_PROJECT" \
    --region "$GCP_REGION" \
    --tag "$BACKEND_IMAGE" \
    --quiet
  gcloud builds submit "$ROOT_DIR/frontend" \
    --project "$GCP_PROJECT" \
    --region "$GCP_REGION" \
    --tag "$FRONTEND_IMAGE" \
    --quiet
else
  echo "BUILD_METHOD must be cloud-build or local." >&2
  exit 1
fi

export GCP_PROJECT GCP_REGION CLOUD_RUN_SERVICE RUNTIME_SERVICE_ACCOUNT_EMAIL
export BACKEND_IMAGE FRONTEND_IMAGE SOURCE_CODE_URL CLOUD_SQL_CONNECTION_NAME
SERVICE_YAML="$(mktemp)"
trap 'rm -f "$SERVICE_YAML"' EXIT
envsubst < "$ROOT_DIR/deploy/gcp/cloud-run-service.yaml" > "$SERVICE_YAML"

gcloud run services replace "$SERVICE_YAML" \
  --project "$GCP_PROJECT" \
  --region "$GCP_REGION" \
  --quiet
gcloud run services add-iam-policy-binding "$CLOUD_RUN_SERVICE" \
  --project "$GCP_PROJECT" \
  --region "$GCP_REGION" \
  --member=allUsers \
  --role=roles/run.invoker \
  --quiet >/dev/null

SERVICE_URL="$(gcloud run services describe "$CLOUD_RUN_SERVICE" \
  --project "$GCP_PROJECT" \
  --region "$GCP_REGION" \
  --format='value(status.url)')"
for attempt in {1..30}; do
  if curl -fsS "${SERVICE_URL}/api/health" | grep -q '"healthy":true'; then
    printf '%s\n' "$SERVICE_URL"
    exit 0
  fi
  sleep 10
done

echo "Cloud Run deployed, but the health check did not become healthy." >&2
echo "Inspect: gcloud run services logs read ${CLOUD_RUN_SERVICE} --region ${GCP_REGION}" >&2
exit 1

#!/usr/bin/env bash
set -euo pipefail

# Provision or update the single-VM OASIS production pilot on Google Cloud.
# This deliberately stops short of publishing a Prolific study. It does create
# billable GCP resources, so run it only after reviewing deploy/gcp/README.md.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GCP_PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
GCP_REGION="${GCP_REGION:-europe-west2}"
GCP_ZONE="${GCP_ZONE:-${GCP_REGION}-a}"
VM_NAME="${VM_NAME:-oasis-pilot}"
SQL_INSTANCE="${SQL_INSTANCE:-oasis-postgres}"
ADDRESS_NAME="${ADDRESS_NAME:-oasis-pilot-ip}"
SERVICE_ACCOUNT_NAME="${SERVICE_ACCOUNT_NAME:-oasis-pilot-runtime}"
RUNTIME_SERVICE_ACCOUNT_EMAIL="${RUNTIME_SERVICE_ACCOUNT_EMAIL:-}"
DB_NAME="${DB_NAME:-oasis}"
DB_USER="${DB_USER:-oasis}"
SOURCE_CODE_URL="${SOURCE_CODE_URL:-}"

if [[ -z "$GCP_PROJECT" || "$GCP_PROJECT" == "(unset)" ]]; then
  echo "Set GCP_PROJECT or run: gcloud config set project YOUR_PROJECT_ID" >&2
  exit 1
fi

if [[ ! "$SOURCE_CODE_URL" =~ ^https:// ]]; then
  echo "Set SOURCE_CODE_URL to an HTTPS page where every participant can obtain the complete source for this AGPL deployment." >&2
  exit 1
fi

if ! gcloud auth list --filter=status:ACTIVE --format='value(account)' | grep -q .; then
  echo "No active gcloud account. Run: gcloud auth login" >&2
  exit 1
fi

if [[ ! -f "$ROOT_DIR/.env" ]]; then
  echo "Missing $ROOT_DIR/.env" >&2
  exit 1
fi

read_env_value() {
  local key="$1"
  sed -n "s/^${key}=//p" "$ROOT_DIR/.env" | tail -n 1
}

OPENAI_KEY="$(read_env_value OPENAI_API_KEY)"
if [[ -z "$OPENAI_KEY" ]]; then
  echo "OPENAI_API_KEY is not set in the local .env" >&2
  exit 1
fi

gcloud config set project "$GCP_PROJECT" >/dev/null
gcloud services enable \
  compute.googleapis.com \
  sqladmin.googleapis.com \
  secretmanager.googleapis.com \
  serviceusage.googleapis.com \
  --project "$GCP_PROJECT"

if [[ -n "$RUNTIME_SERVICE_ACCOUNT_EMAIL" ]]; then
  SERVICE_ACCOUNT_EMAIL="$RUNTIME_SERVICE_ACCOUNT_EMAIL"
  gcloud iam service-accounts describe "$SERVICE_ACCOUNT_EMAIL" \
    --project "$GCP_PROJECT" >/dev/null
else
  SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${GCP_PROJECT}.iam.gserviceaccount.com"
  if ! gcloud iam service-accounts describe "$SERVICE_ACCOUNT_EMAIL" --project "$GCP_PROJECT" >/dev/null 2>&1; then
    gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
      --display-name="OASIS pilot runtime" \
      --project "$GCP_PROJECT"
  fi

  for role in roles/cloudsql.client roles/secretmanager.secretAccessor roles/logging.logWriter roles/monitoring.metricWriter; do
    gcloud projects add-iam-policy-binding "$GCP_PROJECT" \
      --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
      --role="$role" \
      --condition=None \
      --quiet >/dev/null
  done
fi

ensure_generated_secret() {
  local name="$1"
  local bytes="$2"
  if ! gcloud secrets describe "$name" --project "$GCP_PROJECT" >/dev/null 2>&1; then
    gcloud secrets create "$name" --replication-policy=automatic --project "$GCP_PROJECT"
    local value
    value="$(openssl rand -hex "$bytes")"
    printf '%s' "$value" | gcloud secrets versions add "$name" --data-file=- --project "$GCP_PROJECT" >/dev/null
  fi
}

ensure_generated_secret oasis-postgres-password 24
ensure_generated_secret oasis-app-secret-key 32
ensure_generated_secret oasis-admin-password 18

if ! gcloud secrets describe oasis-openai-api-key --project "$GCP_PROJECT" >/dev/null 2>&1; then
  gcloud secrets create oasis-openai-api-key --replication-policy=automatic --project "$GCP_PROJECT"
fi
printf '%s' "$OPENAI_KEY" | gcloud secrets versions add oasis-openai-api-key \
  --data-file=- --project "$GCP_PROJECT" >/dev/null

DB_PASSWORD="$(gcloud secrets versions access latest --secret=oasis-postgres-password --project "$GCP_PROJECT")"

if ! gcloud sql instances describe "$SQL_INSTANCE" --project "$GCP_PROJECT" >/dev/null 2>&1; then
  gcloud sql instances create "$SQL_INSTANCE" \
    --project "$GCP_PROJECT" \
    --database-version=POSTGRES_16 \
    --region="$GCP_REGION" \
    --tier=db-g1-small \
    --storage-type=SSD \
    --storage-size=20 \
    --storage-auto-increase \
    --availability-type=zonal \
    --backup-start-time=02:00 \
    --enable-point-in-time-recovery
fi

if ! gcloud sql databases describe "$DB_NAME" --instance "$SQL_INSTANCE" --project "$GCP_PROJECT" >/dev/null 2>&1; then
  gcloud sql databases create "$DB_NAME" --instance "$SQL_INSTANCE" --project "$GCP_PROJECT"
fi

if gcloud sql users list --instance "$SQL_INSTANCE" --project "$GCP_PROJECT" --format='value(name)' | grep -qx "$DB_USER"; then
  gcloud sql users set-password "$DB_USER" --instance "$SQL_INSTANCE" \
    --password="$DB_PASSWORD" --project "$GCP_PROJECT" --quiet
else
  gcloud sql users create "$DB_USER" --instance "$SQL_INSTANCE" \
    --password="$DB_PASSWORD" --project "$GCP_PROJECT"
fi

if ! gcloud compute addresses describe "$ADDRESS_NAME" --region "$GCP_REGION" --project "$GCP_PROJECT" >/dev/null 2>&1; then
  gcloud compute addresses create "$ADDRESS_NAME" --region "$GCP_REGION" --project "$GCP_PROJECT"
fi
EXTERNAL_IP="$(gcloud compute addresses describe "$ADDRESS_NAME" --region "$GCP_REGION" --project "$GCP_PROJECT" --format='value(address)')"
DOMAIN="${DOMAIN:-${EXTERNAL_IP//./-}.sslip.io}"

if ! gcloud compute firewall-rules describe oasis-pilot-web --project "$GCP_PROJECT" >/dev/null 2>&1; then
  gcloud compute firewall-rules create oasis-pilot-web \
    --project "$GCP_PROJECT" \
    --network=default \
    --allow=tcp:80,tcp:443 \
    --target-tags=oasis-web \
    --description="Public HTTPS for OASIS participant and researcher UI"
fi

if ! gcloud compute instances describe "$VM_NAME" --zone "$GCP_ZONE" --project "$GCP_PROJECT" >/dev/null 2>&1; then
  gcloud compute instances create "$VM_NAME" \
    --project "$GCP_PROJECT" \
    --zone "$GCP_ZONE" \
    --machine-type=e2-standard-2 \
    --image-family=ubuntu-2404-lts-amd64 \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=30GB \
    --boot-disk-type=pd-balanced \
    --address="$EXTERNAL_IP" \
    --service-account="$SERVICE_ACCOUNT_EMAIL" \
    --scopes=cloud-platform \
    --tags=oasis-web
fi

gcloud compute ssh "$VM_NAME" --zone "$GCP_ZONE" --project "$GCP_PROJECT" --command \
  "sudo apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io docker-compose-v2 ca-certificates curl && sudo systemctl enable --now docker"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
mkdir -p "$TMP_DIR/oasis"

rsync -a \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='tests' \
  --exclude='evals' \
  --exclude='research' \
  "$ROOT_DIR/backend" "$TMP_DIR/oasis/"
rsync -a \
  --exclude='node_modules' \
  --exclude='dist' \
  "$ROOT_DIR/frontend" "$TMP_DIR/oasis/"
rsync -a "$ROOT_DIR/docker" "$TMP_DIR/oasis/"
cp "$ROOT_DIR/docker-compose.gcp.yml" "$TMP_DIR/oasis/"

APP_SECRET="$(gcloud secrets versions access latest --secret=oasis-app-secret-key --project "$GCP_PROJECT")"
ADMIN_PASSWORD="$(gcloud secrets versions access latest --secret=oasis-admin-password --project "$GCP_PROJECT")"
OPENAI_SECRET="$(gcloud secrets versions access latest --secret=oasis-openai-api-key --project "$GCP_PROJECT")"
SQL_CONNECTION_NAME="$(gcloud sql instances describe "$SQL_INSTANCE" --project "$GCP_PROJECT" --format='value(connectionName)')"

umask 077
{
  printf 'APP_NAME=OASIS\nAPP_ENV=production\nDEBUG=false\n'
  printf 'SECRET_KEY=%s\n' "$APP_SECRET"
  printf 'POSTGRES_USER=%s\nPOSTGRES_PASSWORD=%s\nPOSTGRES_DB=%s\n' "$DB_USER" "$DB_PASSWORD" "$DB_NAME"
  printf 'POSTGRES_HOST=cloud-sql-proxy\nPOSTGRES_PORT=5432\n'
  printf 'DATABASE_POOL_SIZE=5\nDATABASE_MAX_OVERFLOW=5\nDATABASE_POOL_TIMEOUT_SECONDS=30\nWEB_CONCURRENCY=2\n'
  printf 'REDIS_URL=redis://redis:6379/0\n'
  printf 'CLOUD_SQL_INSTANCE=%s\nDOMAIN=%s\n' "$SQL_CONNECTION_NAME" "$DOMAIN"
  printf 'SOURCE_CODE_URL=%s\n' "$SOURCE_CODE_URL"
  printf 'OPENAI_API_KEY=%s\nOPENAI_USE_EU=false\n' "$OPENAI_SECRET"
  printf 'AUTH_ENABLED=true\nAUTH_USERNAME=admin\nAUTH_PASSWORD=%s\n' "$ADMIN_PASSWORD"
  printf 'AUDIO_STORAGE_BACKEND=local\nAUDIO_STORAGE_LOCAL_PATH=/data/oasis-recordings\n'
} > "$TMP_DIR/oasis/.env.production"

tar -C "$TMP_DIR" -czf "$TMP_DIR/oasis-release.tgz" oasis
gcloud compute scp "$TMP_DIR/oasis-release.tgz" "$VM_NAME:/tmp/oasis-release.tgz" \
  --zone "$GCP_ZONE" --project "$GCP_PROJECT"

gcloud compute ssh "$VM_NAME" --zone "$GCP_ZONE" --project "$GCP_PROJECT" --command \
  "sudo mkdir -p /opt/oasis && sudo tar -xzf /tmp/oasis-release.tgz -C /opt/oasis --strip-components=1 && sudo chmod 600 /opt/oasis/.env.production && cd /opt/oasis && sudo docker compose --env-file .env.production -f docker-compose.gcp.yml up -d --build --remove-orphans"

for attempt in {1..30}; do
  if curl -fsS "https://${DOMAIN}/api/health" | grep -q '"healthy":true'; then
    echo
    echo "OASIS is healthy: https://${DOMAIN}"
    echo "Participant base URL: https://${DOMAIN}/interview/"
    echo "Researcher password remains in Secret Manager. Retrieve it with:"
    echo "  gcloud secrets versions access latest --secret=oasis-admin-password --project=$GCP_PROJECT"
    exit 0
  fi
  sleep 10
done

echo "Deployment finished, but the public health check is not healthy yet." >&2
echo "Inspect: gcloud compute ssh $VM_NAME --zone $GCP_ZONE --command='cd /opt/oasis && sudo docker compose -f docker-compose.gcp.yml logs --tail=200'" >&2
exit 1

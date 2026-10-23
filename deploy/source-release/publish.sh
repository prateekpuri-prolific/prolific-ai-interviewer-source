#!/usr/bin/env bash
set -euo pipefail

# Publish a clean, immutable corresponding-source snapshot for a production
# deployment. The working repository stays private; only committed deployable
# source is copied to the public release repository.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PUBLIC_SOURCE_REPO="${PUBLIC_SOURCE_REPO:-prateekpuri-prolific/prolific-ai-interviewer-source}"
SOURCE_REF="${SOURCE_REF:-HEAD}"

for command in git gh rsync tar rg; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Missing required command: $command" >&2
    exit 1
  fi
done

if [[ -n "$(git -C "$ROOT_DIR" status --porcelain)" ]]; then
  echo "Refusing to publish from a dirty working tree. Commit or stash changes first." >&2
  exit 1
fi

SOURCE_SHA="$(git -C "$ROOT_DIR" rev-parse --verify "${SOURCE_REF}^{commit}")"
SHORT_SHA="${SOURCE_SHA:0:12}"
SOURCE_TAG="source-${SHORT_SHA}"

VISIBILITY="$(gh repo view "$PUBLIC_SOURCE_REPO" --json visibility --jq .visibility 2>/dev/null || true)"
if [[ "$VISIBILITY" != "PUBLIC" ]]; then
  echo "Public source repository $PUBLIC_SOURCE_REPO does not exist or is not public." >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
ARCHIVE_DIR="$TMP_DIR/archive"
EXPORT_DIR="$TMP_DIR/export"
PUBLISH_DIR="$TMP_DIR/public-repository"
mkdir -p "$ARCHIVE_DIR" "$EXPORT_DIR"

# Export only tracked files from the exact commit. Internal evaluation runs and
# research outputs are not deployed and are therefore excluded from this source
# release; all application, build, migration, test, and operations source stays.
git -C "$ROOT_DIR" archive "$SOURCE_SHA" -- \
  .env.example \
  .env.gcp.example \
  .gitignore \
  CHANGELOG.md \
  CITATION.cff \
  CONTRIBUTING.md \
  FAQ.md \
  LICENSE \
  NOTICE \
  README.md \
  SOURCE_RELEASE.md \
  backend \
  deploy \
  docker \
  docker-compose.gcp.yml \
  docker-compose.yml \
  docs/ADAPTIVE_BEHAVIOR.md \
  docs/AUDIO_RECORDING.md \
  docs/DEPLOYMENT.md \
  docs/ENGAGEMENT_METRICS.md \
  docs/RESEARCH_PROTOCOL.md \
  frontend \
  misc \
  scripts | tar -x -C "$ARCHIVE_DIR"

rsync -a \
  --exclude='backend/evals/' \
  --exclude='backend/research/' \
  "$ARCHIVE_DIR/" "$EXPORT_DIR/"

for required_path in \
  LICENSE \
  backend/app \
  backend/alembic \
  backend/requirements.txt \
  frontend/src \
  frontend/package-lock.json \
  docker-compose.gcp.yml \
  deploy/gcp/provision.sh; do
  if [[ ! -e "$EXPORT_DIR/$required_path" ]]; then
    echo "Source export is incomplete; missing $required_path" >&2
    exit 1
  fi
done

BLOCKED_PATH="$(find "$EXPORT_DIR" -type f \( \
  -name '.env' -o \
  -name '.env.production' -o \
  -name '*.pem' -o \
  -name '*.key' -o \
  -name '*.p12' -o \
  -name '*.sqlite' -o \
  -name '*.sqlite3' -o \
  -name 'credentials.json' \
  \) -print -quit)"
if [[ -n "$BLOCKED_PATH" ]]; then
  echo "Source export contains a blocked credential or runtime-data file: $BLOCKED_PATH" >&2
  exit 1
fi

SECRET_PATTERN='(sk-proj-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----|postgres(ql)?://[^[:space:]:]+:[^[:space:]@]+@)'
if rg -n --hidden \
  --glob '!deploy/source-release/publish.sh' \
  --glob '!.env.example' \
  --glob '!.env.gcp.example' \
  "$SECRET_PATTERN" "$EXPORT_DIR" >&2; then
  echo "Potential secret detected. Nothing was published." >&2
  exit 1
fi

EXISTING_RELEASE_URL="$(gh release view "$SOURCE_TAG" --repo "$PUBLIC_SOURCE_REPO" --json url --jq .url 2>/dev/null || true)"
if [[ -n "$EXISTING_RELEASE_URL" ]]; then
  printf '%s\n' "$EXISTING_RELEASE_URL"
  exit 0
fi

# Publication only needs the current public snapshot, not every prior export.
gh repo clone "$PUBLIC_SOURCE_REPO" "$PUBLISH_DIR" -- --quiet --depth=1 >&2
rsync -a --delete --exclude='.git/' "$EXPORT_DIR/" "$PUBLISH_DIR/"

git -C "$PUBLISH_DIR" config user.name "Prolific Source Release"
git -C "$PUBLISH_DIR" config user.email "312557960+prateekpuri-prolific@users.noreply.github.com"
git -C "$PUBLISH_DIR" add -A

if git -C "$PUBLISH_DIR" diff --cached --quiet; then
  echo "No source changes to publish for $SOURCE_TAG." >&2
else
  git -C "$PUBLISH_DIR" commit -m "Publish source for ${SHORT_SHA}" >&2
  git -C "$PUBLISH_DIR" push origin HEAD:main >&2
fi

if git -C "$PUBLISH_DIR" show-ref --verify --quiet "refs/tags/$SOURCE_TAG"; then
  echo "Reusing existing source tag $SOURCE_TAG." >&2
else
  git -C "$PUBLISH_DIR" tag -a "$SOURCE_TAG" -m "Corresponding source for ${SHORT_SHA}"
  git -C "$PUBLISH_DIR" push origin "$SOURCE_TAG" >&2
fi

RELEASE_URL="$(gh release create "$SOURCE_TAG" \
  --repo "$PUBLIC_SOURCE_REPO" \
  --title "Source release ${SHORT_SHA}" \
  --notes "Complete corresponding source for application version ${SHORT_SHA}. No credentials, participant data, recordings, or runtime databases are included." \
  --verify-tag)"
printf '%s\n' "$RELEASE_URL"

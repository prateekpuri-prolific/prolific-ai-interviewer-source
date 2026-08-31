#!/usr/bin/env bash
set -euo pipefail

# Publish the source for the exact committed version before provisioning it.
# This wrapper intentionally refuses dirty working trees through publish.sh, so
# the participant-facing source link cannot drift from the deployed code.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PUBLIC_SOURCE_REPO="${PUBLIC_SOURCE_REPO:-prateekpuri-prolific/prolific-ai-interviewer-source}"

SOURCE_CODE_URL="$(PUBLIC_SOURCE_REPO="$PUBLIC_SOURCE_REPO" "$ROOT_DIR/deploy/source-release/publish.sh")"
export SOURCE_CODE_URL

echo "Deploying with corresponding source: $SOURCE_CODE_URL" >&2
exec "$ROOT_DIR/deploy/gcp/provision.sh"

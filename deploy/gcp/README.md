# GCP production pilot

This deployment is sized for a real Prolific pilot without prematurely turning
OASIS into a distributed platform.

## Topology

- One `e2-standard-2` Compute Engine VM runs Caddy, the React frontend, two
  FastAPI workers, Redis, and the Cloud SQL Auth Proxy.
- Cloud SQL for PostgreSQL 16 stores study definitions, Prolific IDs,
  transcripts, turns, consent timestamps, and completion state.
- Cloud SQL automated backups, point-in-time recovery, and storage auto-growth
  are enabled. PostgreSQL is never exposed directly to the internet.
- Redis stores coordination state only. Losing it may interrupt interviews that
  are in progress, but completed transcripts remain in Cloud SQL.
- Caddy terminates HTTPS and keeps interview WebSockets on the same origin as
  the participant UI. The default hostname uses the reserved static IP through
  `sslip.io`; set `DOMAIN=research.example.com` before provisioning if a real
  DNS record already points at the reserved IP.
- Audio recording remains off for the current agent. Speech is sent to the
  selected transcription provider and only the transcript is persisted.

Two API workers use at most 20 PostgreSQL connections in total under the
production defaults. Because interview sockets do not reserve a database
connection while a participant is thinking, this is appropriate for a
10-person soft launch and a 50-person initial fielding. Run the included load
smoke test before increasing simultaneous recruitment substantially.

## Before provisioning

1. Review the participant information and decide the public researcher name,
   contact route, retention period, and withdrawal window.
2. The private working repository has a companion public source repository.
   Use `release-and-provision.sh` so the exact committed version is exported,
   checked for secrets and runtime data, tagged, and linked from the participant
   landing page before deployment. A private repository link by itself is not
   sufficient.
3. Sign in with `gcloud auth login` and select the billing-enabled project.
4. Verify that `OPENAI_API_KEY` is present in the local `.env`. The script sends
   it directly to Google Secret Manager without printing it.
5. Review regional data requirements. The default region is `europe-west2`.
6. Review the billable resources: one VM, one static IPv4 address, one
   `db-g1-small` Cloud SQL instance with 20 GB SSD, backups, and egress/API use.

## Provision or update

The preferred command publishes a matching source release first and then passes
its immutable URL to the provisioner:

```bash
cd /path/to/oasis
GCP_PROJECT=your-project-id deploy/gcp/release-and-provision.sh
```

The public release repository defaults to
`prateekpuri-prolific/prolific-ai-interviewer-source`. Override it only when the
replacement repository already exists and is public:

```bash
PUBLIC_SOURCE_REPO=your-org/your-public-source-repository \
GCP_PROJECT=your-project-id \
deploy/gcp/release-and-provision.sh
```

When Compute Engine permissions are unavailable but Cloud Run is approved, the
same application can run as a single, multi-container Cloud Run service. Nginx
serves the participant UI and proxies HTTP/WebSocket traffic to the backend;
Cloud SQL remains durable storage and Redis is a one-instance coordination
sidecar. The service is intentionally capped at one always-warm instance for
the pilot:

```bash
GCP_PROJECT=your-project-id deploy/gcp/deploy-cloud-run.sh
```

If the deployer can write Artifact Registry images but cannot submit Cloud
Build jobs, build and push the same images from the local Docker engine:

```bash
BUILD_METHOD=local GCP_PROJECT=your-project-id deploy/gcp/deploy-cloud-run.sh
```

Optional overrides:

```bash
GCP_REGION=europe-west2 \
GCP_ZONE=europe-west2-a \
DOMAIN=research.example.com \
GCP_PROJECT=your-project-id \
deploy/gcp/release-and-provision.sh
```

If organization policy prevents your account from changing project IAM, an
administrator can provide an existing runtime service account that already has
Cloud SQL Client permission. The deployer writes application secrets into a
root-readable environment file on the VM, so the runtime account does not need
Secret Manager access:

```bash
RUNTIME_SERVICE_ACCOUNT_EMAIL=approved-runtime@your-project.iam.gserviceaccount.com \
GCP_PROJECT=your-project-id \
deploy/gcp/release-and-provision.sh
```

The source publisher and provisioner are idempotent. An already-published commit
reuses its release URL; infrastructure is reused, migrations are applied, the
containers are rebuilt, and the public dependency-aware health check must pass.

To publish a source snapshot without deploying infrastructure, run:

```bash
deploy/source-release/publish.sh
```

Only committed source can be released or deployed through the wrapper. Internal
evaluation transcripts and research outputs are not part of the production
bundle and are excluded. Credentials, participant data, recordings, and runtime
databases are never copied into the public source repository.

## Participant-device pilot checks

The landing page lets participants choose voice or typing before starting. Its
microphone test requests permission only after the participant clicks the test,
samples sound locally for 2.5 seconds, uploads nothing, and releases the device.
The session export includes `initial_response_mode` and
`microphone_check_status` (`ready`, `quiet`, `denied`, `unavailable`, or `idle`).
Review these after the soft launch before estimating a population-wide failure
rate. Keep typing available as the fallback; do not exclude someone solely
because their browser microphone fails.

## Operations

```bash
# Service status and logs
gcloud compute ssh oasis-pilot --zone europe-west2-a --command \
  'cd /opt/oasis && sudo docker compose -f docker-compose.gcp.yml ps'

gcloud compute ssh oasis-pilot --zone europe-west2-a --command \
  'cd /opt/oasis && sudo docker compose -f docker-compose.gcp.yml logs --tail=200 backend'

# Retrieve the generated researcher password without putting it in source code
gcloud secrets versions access latest --secret=oasis-admin-password
```

Cloud SQL is the system of record. Before a schema change or large fielding,
also create an on-demand Cloud SQL backup. Do not use `docker compose down -v`
on production: that would erase Redis/Caddy volumes and is unnecessary for an
update.

## When to scale further

Do not add a load balancer, managed Redis, or a container orchestrator for the
first 50-person study. Revisit the topology when measured peak concurrency is
regularly above 30, a single-VM interruption is unacceptable, or multiple
studies need guaranteed isolation. At that point, move Redis to Memorystore,
publish immutable images, and run the stateless web services behind a managed
load balancer while retaining Cloud SQL.

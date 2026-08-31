#!/bin/sh
set -eu

python - <<'PY'
import os
import socket
import time

host = os.environ.get("POSTGRES_HOST", "postgres")
port = int(os.environ.get("POSTGRES_PORT", "5432"))
deadline = time.monotonic() + 90

while True:
    try:
        with socket.create_connection((host, port), timeout=2):
            break
    except OSError:
        if time.monotonic() >= deadline:
            raise SystemExit(f"Postgres at {host}:{port} was not reachable within 90 seconds")
        time.sleep(2)
PY

alembic upgrade head

workers="${WEB_CONCURRENCY:-1}"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers "$workers"

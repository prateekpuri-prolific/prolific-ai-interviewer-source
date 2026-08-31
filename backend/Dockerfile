FROM python:3.12-slim

WORKDIR /app

# System deps for asyncpg
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Wait for Postgres, run migrations once, then start the configured number of
# API workers.  The wait is important when Cloud SQL is reached through its
# local Auth Proxy container.
RUN chmod +x /app/scripts/start.sh
CMD ["/app/scripts/start.sh"]

# Dockerfile for PostgreSQL 16 + pgvector
FROM postgres:16

# Install pgvector from Debian repo (no extra repo needed)
RUN apt-get update && \
    apt-get install -y --no-install-recommends postgresql-16-pgvector && \
    rm -rf /var/lib/apt/lists/*

COPY ./init-pgvector.sql /docker-entrypoint-initdb.d/
FROM python:3.12-slim
WORKDIR /srv/app

# kubectl for the tool link. Pinned: an agent whose tooling changes under it on
# a rebuild is an agent whose behaviour you cannot reason about.
ARG KUBECTL_VERSION=v1.33.4
ARG TARGETARCH=amd64
RUN set -eu; \
    apt-get update && apt-get install -y --no-install-recommends curl ca-certificates; \
    curl -fsSL -o /usr/local/bin/kubectl \
      "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${TARGETARCH}/kubectl"; \
    chmod 0755 /usr/local/bin/kubectl; \
    apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Deliberately narrower than agentmemory's: no asyncpg, no mcp. This container
# sits on the serving path, so its dependency surface is a liability.
RUN pip install --no-cache-dir \
    fastapi==0.115.* uvicorn==0.30.* httpx==0.27.* prometheus-client==0.20.*

COPY gateway/ /srv/app/gateway/
EXPOSE 8010
USER 1000:1000
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8010"]

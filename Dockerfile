FROM python:3.12-slim
WORKDIR /srv/app

# Deliberately narrow: no asyncpg, no MCP SDK (mcpclient.py speaks the wire
# protocol). This container sits on the serving path, so its dependency
# surface is a liability. No kubectl either: tools live behind MCP_URL.
RUN pip install --no-cache-dir \
    fastapi==0.115.* uvicorn==0.30.* httpx==0.27.* prometheus-client==0.20.*

COPY gateway/ /srv/app/gateway/
EXPOSE 8010
USER 1000:1000
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8010"]

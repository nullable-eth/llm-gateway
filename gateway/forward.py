"""Transparent reverse-proxy plumbing.

The client's bytes go up and the server's bytes come back untouched; the tee
only watches a copy. Authorization is forwarded verbatim and never inspected,
so llama.cpp keeps enforcing LLAMA_API_KEY exactly as it does today — the
proxy holds no key and can grant no access.

Accept-Encoding is dropped on the way up so the response arrives as identity.
Over a loopback hop into a sidecar that costs nothing, and it means the tee
reads plaintext instead of having to decompress a copy of every stream.
"""
import httpx

from . import config

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate",
              "proxy-authorization", "te", "trailer", "transfer-encoding",
              "upgrade"}
DROP_UP = HOP_BY_HOP | {"host", "content-length", "accept-encoding"}
DROP_DOWN = HOP_BY_HOP | {"content-length", "content-encoding"}


def client() -> httpx.AsyncClient:
    # No read timeout: a 262k-context reply on three V100s can take minutes,
    # and the proxy must never be the thing that gives up on it.
    return httpx.AsyncClient(
        timeout=httpx.Timeout(None, connect=config.CONNECT_TIMEOUT_S),
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
        follow_redirects=False)


def upstream_headers(headers) -> dict:
    return {k: v for k, v in headers.items() if k.lower() not in DROP_UP}


def downstream_headers(headers) -> dict:
    return {k: v for k, v in headers.items() if k.lower() not in DROP_DOWN}


def url_for(path: str, query: str) -> str:
    url = f"{config.UPSTREAM}/{path.lstrip('/')}"
    return f"{url}?{query}" if query else url

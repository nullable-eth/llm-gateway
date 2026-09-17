"""A minimal MCP client (Streamable HTTP) for the tool loop.

The gateway has no tools of its own. It is a client of ONE MCP endpoint —
normally an MCP gateway that multiplexes many servers — and offers the model
whatever that endpoint lists. What a tool may do is decided behind that
endpoint (each server's own credentials and RBAC, the MCP gateway's policies),
never here, so adding, removing or restricting a capability is a deployment
change, not a code change.

Deliberately small: JSON-RPC over POST, JSON or SSE responses, the session
header, one re-initialise when a session expires. No SDK dependency, because
this process sits on the serving path.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import time

import httpx

from . import config

log = logging.getLogger("gateway.mcp")

PROTOCOL_VERSION = "2025-06-18"


class McpError(Exception):
    pass


class _Expired(Exception):
    pass


class Client:
    def __init__(self, url: str, token: str = "", timeout: float = 120.0,
                 tools_ttl: float = 60.0):
        self.url, self.token, self.timeout, self.tools_ttl = url, token, timeout, tools_ttl
        self.session: str | None = None
        self.instructions: str = ""
        self._ids = itertools.count(1)
        self._init_lock = asyncio.Lock()
        self._tools: list[dict] = []
        self._tools_at = 0.0
        self._http: httpx.AsyncClient | None = None

    # ----------------------------------------------------------- transport
    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=5.0))
        return self._http

    def _headers(self) -> dict:
        h = {"Accept": "application/json, text/event-stream",
             "Content-Type": "application/json",
             "MCP-Protocol-Version": PROTOCOL_VERSION}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        if self.session:
            h["Mcp-Session-Id"] = self.session
        return h

    async def _post(self, payload: dict, timeout: float | None = None) -> dict | None:
        rid = payload.get("id")
        c = self._client()
        async with c.stream("POST", self.url, json=payload, headers=self._headers(),
                            timeout=timeout or self.timeout) as r:
            if r.status_code == 404 and self.session:
                raise _Expired()
            if r.status_code >= 400:
                body = (await r.aread()).decode("utf-8", "replace")
                raise McpError(f"HTTP {r.status_code}: {body[:300]}")
            sid = r.headers.get("mcp-session-id")
            if sid:
                self.session = sid
            if rid is None:
                return None
            ctype = r.headers.get("content-type", "")
            if ctype.startswith("text/event-stream"):
                data: list[str] = []
                async for line in r.aiter_lines():
                    if line.startswith("data:"):
                        data.append(line[5:].lstrip())
                        continue
                    if line == "" and data:
                        msg = _parse("\n".join(data))
                        data = []
                        if isinstance(msg, dict) and msg.get("id") == rid:
                            return msg
                if data:
                    msg = _parse("\n".join(data))
                    if isinstance(msg, dict) and msg.get("id") == rid:
                        return msg
                raise McpError("stream ended without a response")
            return _parse((await r.aread()).decode("utf-8", "replace"))

    async def _initialize(self) -> None:
        self.session = None
        res = await self._post({"jsonrpc": "2.0", "id": next(self._ids), "method": "initialize",
                                "params": {"protocolVersion": PROTOCOL_VERSION,
                                           "capabilities": {},
                                           "clientInfo": {"name": "llm-gateway",
                                                          "version": "1"}}})
        result = _result(res)
        self.instructions = str(result.get("instructions") or "")
        await self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self._ready = True

    async def _ensure(self) -> None:
        if getattr(self, "_ready", False):
            return
        async with self._init_lock:
            if not getattr(self, "_ready", False):
                await self._initialize()

    async def request(self, method: str, params: dict | None = None,
                      timeout: float | None = None) -> dict:
        await self._ensure()
        for attempt in (0, 1):
            payload = {"jsonrpc": "2.0", "id": next(self._ids), "method": method}
            if params is not None:
                payload["params"] = params
            try:
                return _result(await self._post(payload, timeout))
            except _Expired:
                if attempt:
                    raise McpError("session expired twice")
                log.info("mcp: session expired; re-initialising")
                self._ready = False
                async with self._init_lock:
                    await self._initialize()
        raise McpError("unreachable")

    # ---------------------------------------------------------------- API
    async def list_tools(self) -> list[dict]:
        """The endpoint's tools, cached for tools_ttl. A failure serves the
        last good list (or none) rather than failing the chat request."""
        if self._tools_at and time.monotonic() - self._tools_at < self.tools_ttl:
            return self._tools
        try:
            tools, cursor = [], None
            for _ in range(50):
                res = await self.request("tools/list", {"cursor": cursor} if cursor else {},
                                         timeout=30)
                tools += res.get("tools") or []
                cursor = res.get("nextCursor")
                if not cursor:
                    break
            self._tools, self._tools_at = tools, time.monotonic()
        except Exception as e:
            self._ready = False
            log.warning("mcp: tools/list failed (%s); serving %d cached tool(s)",
                        e, len(self._tools))
            self._tools_at = time.monotonic() - self.tools_ttl + 10   # retry soon
        return self._tools

    async def call_tool(self, name: str, arguments: dict) -> tuple[str, bool]:
        """(text, is_error)."""
        res = await self.request("tools/call", {"name": name, "arguments": arguments})
        return render(res), bool(res.get("isError"))

    def annotations(self, name: str) -> dict:
        for t in self._tools:
            if t.get("name") == name:
                return t.get("annotations") or {}
        return {}

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None


def _parse(text: str):
    try:
        return json.loads(text)
    except ValueError as e:
        raise McpError(f"not JSON: {text[:200]}") from e


def _result(msg) -> dict:
    if not isinstance(msg, dict):
        raise McpError("empty response")
    if msg.get("error"):
        err = msg["error"]
        raise McpError(f"{err.get('message') or err}" if isinstance(err, dict) else str(err))
    res = msg.get("result")
    return res if isinstance(res, dict) else {}


def render(result: dict) -> str:
    """A tool result as text for the model."""
    parts = []
    for c in result.get("content") or []:
        kind = c.get("type")
        if kind == "text":
            parts.append(str(c.get("text") or ""))
        elif kind == "resource":
            r = c.get("resource") or {}
            parts.append(str(r.get("text") or f"[resource {r.get('uri', '')}]"))
        elif kind == "resource_link":
            parts.append(f"[resource {c.get('uri', '')}]")
        else:
            parts.append(f"[{kind} content omitted]")
    if not any(p.strip() for p in parts) and result.get("structuredContent") is not None:
        parts = [json.dumps(result["structuredContent"], ensure_ascii=False)]
    return "\n".join(parts)


def to_openai(tool: dict) -> dict:
    schema = tool.get("inputSchema") or {}
    if not isinstance(schema, dict) or schema.get("type") != "object":
        schema = {"type": "object", "properties": {}}
    schema = {k: v for k, v in schema.items() if k != "$schema"}
    desc = tool.get("description") or tool.get("title") or ""
    return {"type": "function", "function": {"name": tool["name"], "description": desc,
                                             "parameters": schema}}


_client: Client | None = None


def get() -> Client | None:
    global _client
    if not config.MCP_URL:
        return None
    if _client is None or _client.url != config.MCP_URL:
        _client = Client(config.MCP_URL, config.mcp_token(), config.MCP_CALL_TIMEOUT_S,
                         config.MCP_TOOLS_TTL_S)
    else:
        _client.token = config.mcp_token()
    return _client

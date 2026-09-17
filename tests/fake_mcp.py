"""A fake MCP endpoint (Streamable HTTP) for the gateway tests.

Enough of the protocol to exercise the client: initialize + session header,
paginated tools/list, tools/call answered as JSON or SSE, tool errors,
JSON-RPC (policy) errors, an expiring session and a bearer token.
"""
import itertools
import json

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

TOKEN = "mcptest"
STATE = {"sessions": set(), "inits": 0, "expire": False, "bad_auth": 0, "calls": []}
_sid = itertools.count(1)

TOOLS = [
    {"name": "kubernetes_pods_list", "description": "List pods in a namespace.",
     "inputSchema": {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
                     "properties": {"namespace": {"type": "string"}}},
     "annotations": {"readOnlyHint": True}},
    {"name": "kubernetes_pods_delete", "description": "Delete a pod.",
     "inputSchema": {"type": "object", "properties": {"name": {"type": "string"},
                                                      "namespace": {"type": "string"}},
                     "required": ["name"]},
     "annotations": {"readOnlyHint": False, "destructiveHint": True}},
    {"name": "memory_get_context", "description": "Read around a search hit.",
     "inputSchema": {"type": "object", "properties": {"message_uuid": {"type": "string"},
                                                      "radius": {"type": "integer"}}},
     "annotations": {"readOnlyHint": True}},
    # page two
    {"name": "github_push_files", "description": "Push files to a branch.",
     "inputSchema": {"type": "object", "properties": {"branch": {"type": "string"}}}},
    {"name": "big_output", "description": "Returns a lot.", "inputSchema": {"type": "object"},
     "annotations": {"readOnlyHint": True}},
    {"name": "structured_only", "description": "Structured content, no text.",
     "annotations": {"readOnlyHint": True}},
]


def _call(name: str, args: dict):
    STATE["calls"].append((name, args))
    if name == "kubernetes_pods_list":
        ns = args.get("namespace", "default")
        return {"content": [{"type": "text", "text": f"FAKE pods in {ns}\npod/jellyfin-0  Running"}]}
    if name == "kubernetes_pods_delete":
        if args.get("namespace") == "ai":
            return {"isError": True, "content": [{"type": "text", "text":
                    'pods "x" is forbidden: ValidatingAdmissionPolicy denied the request'}]}
        return {"content": [{"type": "text", "text": f"Pod deleted successfully: {args.get('name')}"}]}
    if name == "memory_get_context":
        return {"content": [{"type": "text", "text":
                f"WINDOW around {args.get('message_uuid')} radius {args.get('radius', 3)}"}]}
    if name == "github_push_files":
        return None     # refused by policy: JSON-RPC error
    if name == "big_output":
        return {"content": [{"type": "text", "text": "x" * 20000}]}
    if name == "structured_only":
        return {"content": [], "structuredContent": {"ok": True, "n": 3}}
    return {"isError": True, "content": [{"type": "text", "text": f"no tool {name}"}]}


async def endpoint(request: Request):
    if request.headers.get("authorization") != f"Bearer {TOKEN}":
        STATE["bad_auth"] += 1
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    msg = await request.json()
    method, rid = msg.get("method"), msg.get("id")
    sid = request.headers.get("mcp-session-id")
    if method == "initialize":
        STATE["inits"] += 1
        new = f"s{next(_sid)}"
        STATE["sessions"].add(new)
        return JSONResponse({"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake", "version": "0"},
            "instructions": "FAKE MCP NOTES: prefer namespaces."}},
            headers={"Mcp-Session-Id": new})
    if not sid or sid not in STATE["sessions"]:
        return JSONResponse({"error": "no session"}, status_code=400 if not sid else 404)
    if STATE["expire"]:
        STATE["expire"] = False
        STATE["sessions"].clear()
        return JSONResponse({"error": "session expired"}, status_code=404)
    if rid is None:
        return Response(status_code=202)
    if method == "tools/list":
        cursor = (msg.get("params") or {}).get("cursor")
        page = TOOLS[3:] if cursor == "p2" else TOOLS[:3]
        result = {"tools": page}
        if cursor != "p2":
            result["nextCursor"] = "p2"
        return JSONResponse({"jsonrpc": "2.0", "id": rid, "result": result})
    if method == "tools/call":
        p = msg.get("params") or {}
        res = _call(p.get("name"), p.get("arguments") or {})
        if res is None:
            body = {"jsonrpc": "2.0", "id": rid, "error": {
                "code": -32602, "message": "Unknown: tool call refused by policy"}}
        else:
            body = {"jsonrpc": "2.0", "id": rid, "result": res}
        if p.get("name") == "kubernetes_pods_list":
            # answered as SSE, with a notification first, as real servers do
            async def gen():
                note = {"jsonrpc": "2.0", "method": "notifications/progress",
                        "params": {"progress": 1}}
                yield f"event: message\ndata: {json.dumps(note)}\n\n"
                yield f"event: message\ndata: {json.dumps(body)}\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        return JSONResponse(body)
    return JSONResponse({"jsonrpc": "2.0", "id": rid,
                         "error": {"code": -32601, "message": "method not found"}})

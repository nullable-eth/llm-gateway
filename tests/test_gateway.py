#!/usr/bin/env python3
"""End-to-end smoke test for the gateway.

Runs a fake llama.cpp and the real proxy in one event loop against a scratch
vault, then asserts on the markdown that comes out AND on what
vaultio.chunk_transcript makes of it — the rendering is only correct if the
chunker agrees, so both are checked.

    python tests/test_gateway.py
"""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAULT = Path(tempfile.mkdtemp(prefix="capture-vault-"))
(VAULT / "One-Offs").mkdir()

os.environ.update(
    VAULT_ROOT=str(VAULT),
    CAPTURE_DIR=".staging/Chats/Live Capture",
    GATEWAY_UPSTREAM="http://127.0.0.1:18000",
    CAPTURE_IDLE_S="0",
    CAPTURE_SWEEP_S="3600",          # sweeps are driven by hand here
    CAPTURE_MAX_OPEN_S="999999",
    COMPACT_RESERVE="100",
    COMPACT_KEEP_TAIL="4",
    PROTECTED="llm-expert,cluster-agent",
)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import httpx                                                    # noqa: E402
import uvicorn                                                  # noqa: E402
from fastapi import FastAPI, Request                            # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse                # noqa: E402

# Verbatim copy of agentmemory app/vaultio.py. The gateway writes markdown
# that agentmemory's scanner must chunk; this asserts both halves agree.
# Refresh it when that file changes -- see README.
import vaultio_ref as vaultio                                   # noqa: E402
from gateway.main import app as proxy_app, writer               # noqa: E402

PROXY = "http://127.0.0.1:18010"
NEXT: dict = {}
NEXT_QUEUE: list = []   # popped per upstream call, for multi-step loops
FAILURES: list = []

fake = FastAPI()


@fake.get("/health")
async def health():
    return {"status": "ok"}


def _frame(delta: dict, finish=None) -> bytes:
    obj = {"id": "x", "object": "chat.completion.chunk", "model": "qwen3.8-27b",
           "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {json.dumps(obj)}\n\n".encode()


def _pieces(text: str, n: int = 7):
    return [text[i:i + n] for i in range(0, len(text), n)] if text else []


# --- the three endpoints the compactor uses on the model server -------------
FAKE_N_CTX = 2000
LAST_UPSTREAM: dict = {}


@fake.get("/props")
async def props(request: Request):
    if not request.headers.get("authorization"):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return {"default_generation_settings": {"n_ctx": FAKE_N_CTX},
            "total_slots": 1}


@fake.post("/apply-template")
async def apply_template(request: Request):
    b = await request.json()
    return {"prompt": "\n".join(
        f"<|{m.get('role')}|>{m.get('content') or ''}" for m in b["messages"])}


@fake.post("/tokenize")
async def tokenize(request: Request):
    b = await request.json()
    return {"tokens": [0] * (len(b.get("content", "")) // 4)}   # 1 tok / 4 chars


@fake.post("/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    LAST_UPSTREAM.clear()
    LAST_UPSTREAM.update(body)
    last = (body.get("messages") or [{}])[-1].get("content") or ""
    if last.startswith("STOP."):                 # the compactor's summary call
        return {"id": "s", "model": "m", "choices": [
            {"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant",
                "content": "STATE: user is migrating the cluster; PVCs renamed."}}]}
    spec = NEXT_QUEUE.pop(0) if NEXT_QUEUE else dict(NEXT)
    if spec.get("status"):
        return JSONResponse({"error": {"message": "fake failure"}},
                            status_code=spec["status"])
    if not body.get("stream"):
        msg = {"role": "assistant", "content": spec.get("content", "")}
        if spec.get("reasoning"):
            msg["reasoning_content"] = spec["reasoning"]
        if spec.get("tool_calls"):
            msg["tool_calls"] = spec["tool_calls"]
        return {"id": "x", "model": "qwen3.8-27b",
                "choices": [{"index": 0, "message": msg,
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 22}}

    async def gen():
        yield _frame({"role": "assistant"})
        for piece in _pieces(spec.get("reasoning", "")):
            yield _frame({"reasoning_content": piece})
            await asyncio.sleep(spec.get("delay", 0))
        for piece in _pieces(spec.get("content", ""), spec.get("piece", 7)):
            yield _frame({"content": piece})
            await asyncio.sleep(spec.get("delay", 0))
        for i, call in enumerate(spec.get("tool_calls", [])):
            yield _frame({"tool_calls": [
                {"index": i, "id": call["id"], "type": "function",
                 "function": {"name": call["function"]["name"],
                              "arguments": ""}}]})
            for piece in _pieces(call["function"]["arguments"], 5):
                yield _frame({"tool_calls": [
                    {"index": i, "function": {"arguments": piece}}]})
        yield _frame({}, finish="stop")
        yield b"data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# --- llama.cpp's web UI, which it only serves gzipped ------------------------
import gzip as _gz
UI_HTML = b"<!doctype html><title>llama.cpp</title>"


@fake.get("/")
async def ui(request: Request):
    from fastapi.responses import Response, PlainTextResponse
    if "gzip" not in (request.headers.get("accept-encoding") or ""):
        return PlainTextResponse("Error: gzip is not supported by this browser")
    return Response(_gz.compress(UI_HTML), media_type="text/html",
                    headers={"Content-Encoding": "gzip"})


SEEN_AE: list = []


@fake.middleware("http")
async def _record_ae(request: Request, call_next):
    if request.url.path == "/v1/chat/completions":
        SEEN_AE.append(request.headers.get("accept-encoding"))
    return await call_next(request)


# --- a fake Prometheus and Alertmanager (read-only tools) --------------------
@fake.get("/prom/api/v1/query")
async def prom_query(query: str):
    if query == "bad(":
        return JSONResponse({"status": "error", "errorType": "bad_data", "error": "parse error"}, 400)
    return {"status": "success", "data": {"resultType": "vector", "result": [
        {"metric": {"__name__": "up", "job": "kube-state-metrics"}, "value": [1, "1"]},
        {"metric": {"__name__": "up", "job": "node"}, "value": [1, "0"]}]}}


@fake.get("/prom/api/v1/query_range")
async def prom_range(query: str, start: float, end: float, step: str):
    t = int(start)
    return {"status": "success", "data": {"resultType": "matrix", "result": [
        {"metric": {"pod": "shared-pg-3"},
         "values": [[t, "0"], [t + 60, "0"], [t + 120, "3"], [t + 180, "8"]]}]}}


@fake.get("/am/api/v2/alerts")
async def am_alerts():
    return [{"labels": {"alertname": "KubePodCrashLooping", "namespace": "databases"},
             "annotations": {"summary": "Pod is crash looping."},
             "startsAt": "2026-09-16T21:51:00Z",
             "status": {"state": "active", "silencedBy": [], "inhibitedBy": []}},
            {"labels": {"alertname": "NodeBondingDegraded"}, "annotations": {},
             "startsAt": "2026-09-16T19:00:00Z",
             "status": {"state": "suppressed", "silencedBy": ["s1"], "inhibitedBy": []}}]


@fake.get("/am/api/v2/silences")
async def am_silences():
    return [{"id": "s1", "status": {"state": "active"}, "endsAt": "2026-09-23T19:53:07Z",
             "createdBy": "cluster-agent", "comment": "operator said ignore",
             "matchers": [{"name": "alertname", "value": "NodeBondingDegraded"}]},
            {"id": "s0", "status": {"state": "expired"}, "matchers": []}]


# --- a fake GitHub, just the REST calls gittools makes ---------------------
import hashlib as _hl

GH = {"branches": {}, "commits": {}, "trees": {}, "pulls": [], "calls": []}


def _gh_sha(obj) -> str:
    return _hl.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def gh_reset():
    files = {"kubernetes/apps/media/sonarr/helmrelease.yaml":
             "spec:\n  values:\n    resources: { requests: { memory: 512Mi } }\n",
             "README.md": "hello\n"}
    t = _gh_sha(files)
    c = _gh_sha({"tree": t, "parents": []})
    GH.update(branches={"main": c}, commits={c: {"tree": t, "parents": []}},
              trees={t: files}, pulls=[], calls=[])


@fake.api_route("/gh/{rest:path}", methods=["GET", "POST", "PATCH"])
async def github(rest: str, request: Request):
    import base64 as _b64
    GH["calls"].append((request.method, rest))
    if request.headers.get("authorization") != "Bearer ghtest":
        return JSONResponse({"message": "Bad credentials"}, status_code=401)
    body = await request.json() if request.method != "GET" else {}
    q = request.query_params
    if rest == "search/code":
        return {"items": [{"path": p} for p in GH["trees"][GH["commits"][GH["branches"]["main"]]["tree"]]
                          if q["q"].split()[0] in GH["trees"][GH["commits"][GH["branches"]["main"]]["tree"]][p]]}
    parts = rest.split("/")
    if parts[0] != "repos" or "/".join(parts[1:3]) != "nullable-eth/Whitehorse":
        return JSONResponse({"message": "Not Found"}, status_code=404)
    tail = "/".join(parts[3:])
    def files_at(ref):
        sha = GH["branches"].get(ref, ref)
        return GH["trees"][GH["commits"][sha]["tree"]] if sha in GH["commits"] else None
    if tail == "":
        return {"default_branch": "main"}
    if tail.startswith("git/ref/heads/"):
        b = tail[len("git/ref/heads/"):]
        if b not in GH["branches"]:
            return JSONResponse({"message": "Not Found"}, status_code=404)
        return {"object": {"sha": GH["branches"][b]}}
    if tail.startswith("contents/"):
        f = files_at(q.get("ref", "main")) or {}
        path = tail[len("contents/"):]
        if path not in f:
            return JSONResponse({"message": "Not Found"}, status_code=404)
        return {"encoding": "base64", "content": _b64.b64encode(f[path].encode()).decode()}
    if tail.startswith("git/trees/") and request.method == "GET":
        f = files_at(tail[len("git/trees/"):])
        return {"tree": [{"path": p, "type": "blob"} for p in sorted(f)]}
    if tail.startswith("git/commits/") and request.method == "GET":
        return {"tree": {"sha": GH["commits"][tail[len("git/commits/"):]]["tree"]}}
    if tail == "git/trees":
        files = dict(GH["trees"][body["base_tree"]])
        for e in body["tree"]:
            if e.get("sha", "x") is None:
                files.pop(e["path"], None)
            else:
                files[e["path"]] = e["content"]
        t = _gh_sha(files); GH["trees"][t] = files
        return {"sha": t}
    if tail == "git/commits":
        c = _gh_sha(body); GH["commits"][c] = {"tree": body["tree"], "parents": body["parents"],
                                               "message": body["message"]}
        return {"sha": c}
    if tail == "git/refs":
        GH["branches"][body["ref"][len("refs/heads/"):]] = body["sha"]
        return {"ref": body["ref"]}
    if tail.startswith("git/refs/heads/") and request.method == "PATCH":
        GH["branches"][tail[len("git/refs/heads/"):]] = body["sha"]
        return {}
    if tail == "pulls" and request.method == "POST":
        pr = {"number": len(GH["pulls"]) + 1, "title": body["title"], "body": body["body"],
              "base": body["base"], "head": {"ref": body["head"]},
              "html_url": f"https://github.test/pr/{len(GH['pulls']) + 1}"}
        GH["pulls"].append(pr)
        return pr
    if tail == "pulls":
        head = q.get("head", "")
        return [p for p in GH["pulls"] if not head or head.endswith(":" + p["head"]["ref"])]
    return JSONResponse({"message": f"fake has no {tail}"}, status_code=404)


# ----------------------------------------------------------------- helpers
def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {label}")
    else:
        FAILURES.append(label)
        print(f"  FAIL {label}" + (f"\n       {detail}" if detail else ""))


def capture_files() -> list:
    d = VAULT / ".staging" / "Chats" / "Live Capture"
    return sorted(d.glob("*.md")) if d.is_dir() else []


def body_of(text: str) -> str:
    m = vaultio.FM_RE.match(text)
    return text[m.end():] if m else text


def rows(text: str) -> list:
    return list(vaultio.chunk_transcript(body_of(text), 6000))


async def chat(messages, stream=True, headers=None, abort_after=None, extra=None):
    payload = {"model": "qwen3.8-27b", "messages": messages, "stream": stream,
               **(extra or {})}
    async with httpx.AsyncClient(timeout=30) as c:
        if not stream:
            r = await c.post(f"{PROXY}/v1/chat/completions", json=payload,
                             headers=headers or {})
            return r.json()
        async with c.stream("POST", f"{PROXY}/v1/chat/completions",
                            json=payload, headers=headers or {}) as r:
            seen = 0
            async for _ in r.aiter_bytes():
                seen += 1
                if abort_after is not None and seen >= abort_after:
                    break
    if abort_after is not None:
        # The tee's finally block runs when the server notices the hangup,
        # which is a beat after the client side has let go.
        await asyncio.sleep(2.0)
    return None


async def stream_frames(messages, headers=None):
    """POST a streamed chat and return [(seconds_since_start, obj)]."""
    import time as _t
    payload = {"model": "qwen3.6-35b-a3b", "messages": messages, "stream": True}
    out, t0 = [], _t.monotonic()
    async with httpx.AsyncClient(timeout=60) as c:
        async with c.stream("POST", f"{PROXY}/v1/chat/completions",
                            json=payload, headers=headers or {}) as r:
            async for line in r.aiter_lines():
                if line.startswith("data:") and line[5:].strip() != "[DONE]":
                    out.append((_t.monotonic() - t0, json.loads(line[5:])))
    return out


def deltas(frames, key):
    return "".join((o["choices"][0]["delta"].get(key) or "")
                   for _, o in frames if o.get("choices"))


def sweep():
    writer.sweep_once(force=True)


async def settle():
    """Wait for the writer to finish applying what the proxy handed it."""
    try:
        await asyncio.wait_for(writer.queue.join(), timeout=10)
    except asyncio.TimeoutError:
        check("writer drained", False, "capture queue never emptied")


SYS = {"role": "system", "content": "You are the Whitehorse cluster agent."}


async def run_all():
    global NEXT

    # ---------------------------------------------------------------- [1]
    print("\n[1] streamed multi-turn — reasoning, content, continuation")
    NEXT = {"reasoning": "The pod is CrashLooping; check events first.",
            "content": "Run `kubectl -n ai describe pod qwen-coder`."}
    u1 = {"role": "user", "content": "qwen-coder won't start.\nWhat now?"}
    await chat([SYS, u1])
    await settle()
    a1 = {"role": "assistant", "content": NEXT["content"]}
    NEXT = {"content": "Then the image pull is what's failing."}
    u2 = {"role": "user", "content": "It says ErrImagePull."}
    a2 = {"role": "assistant", "content": NEXT["content"]}
    await chat([SYS, u1, a1, u2])
    await settle()
    sweep()

    files = capture_files()
    check("one transcript written", len(files) == 1, str([f.name for f in files]))
    if not files:
        return
    main_file = files[0]
    text = main_file.read_text(encoding="utf-8")
    fm = vaultio.parse_frontmatter(text)
    check("filename carries date and title", "qwen-coder" in main_file.name,
          main_file.name)
    check("status is unfiled", fm.get("status") == "unfiled", fm.get("status"))
    check("type is chat-transcript", fm.get("type") == "chat-transcript")
    check("capture marker set", fm.get("capture") == "proxy")
    check("message_count is 5", fm.get("message_count") == "5",
          fm.get("message_count"))
    check("exchanges is 2", fm.get("exchanges") == "2", fm.get("exchanges"))
    check("source_uuid present", bool(fm.get("source_uuid")))

    r = rows(text)
    check("chunker sees 5 messages", len(r) == 5, str(len(r)))
    senders = [x[2] for x in r]
    check("senders parse for every chunk",
          senders == ["User", "User", "Claude", "User", "Claude"], str(senders))
    check("timestamps parse for every chunk", all(x[3] for x in r),
          str([x[3] for x in r]))
    check("message uuids unique", len({x[0] for x in r}) == 5)
    check("reasoning captured", "Extended thinking" in text
          and "CrashLooping" in text)
    check("system prompt captured", "System prompt" in text
          and "Whitehorse cluster agent" in text)
    check("nothing truncated", "ErrImagePull" in text
          and "image pull is what's failing" in text)
    first_uuids = [x[0] for x in r]

    # ---------------------------------------------------------------- [2]
    print("\n[2] reopen after flush — history converges, uuids stable")
    NEXT = {"content": "Pin the sha- tag instead of latest."}
    u3 = {"role": "user", "content": "How do I fix it for good?"}
    await chat([SYS, u1, a1, u2, a2, u3])
    await settle()
    sweep()
    files = capture_files()
    check("reopened in place, no second file", len(files) == 1,
          str([f.name for f in files]))
    text = main_file.read_text(encoding="utf-8")
    r = rows(text)
    check("now 7 messages", len(r) == 7, str(len(r)))
    check("existing uuids unchanged (embeddings survive)",
          [x[0] for x in r][:5] == first_uuids)

    # --------------------------------------------------------------- [2b]
    print("\n[2b] flushed conversations hold no message bodies in memory")
    conv = next(c for c in writer.store.convs.values()
                if c.path and "qwen-coder" in c.path)
    check("compacted after flush", not conv.hydrated and conv.messages == [],
          f"hydrated={conv.hydrated} messages={len(conv.messages)}")
    check("still matchable — digests kept for every message",
          len(conv.digests) == 7, str(len(conv.digests)))
    check("index file still holds the bodies",
          len(json.loads(conv.index_path().read_text())["messages"]) == 7)
    check("a compacted save cannot blank the index", conv.save()
          and len(json.loads(conv.index_path().read_text())["messages"]) == 7)

    # ---------------------------------------------------------------- [3]
    print("\n[3] trimmed history — client drops the oldest turns")
    NEXT = {"content": "Because latest is mutable."}
    a3 = {"role": "assistant", "content": "Pin the sha- tag instead of latest."}
    u4 = {"role": "user", "content": "Why does that help?"}
    await chat([u2, a2, u3, a3, u4])          # system + first turn trimmed away
    await settle()
    sweep()
    check("trimmed request adopted, still one file", len(capture_files()) == 1,
          str([f.name for f in capture_files()]))
    r = rows(main_file.read_text(encoding="utf-8"))
    check("trimmed continuation appended, not duplicated", len(r) == 9,
          str(len(r)))

    # ---------------------------------------------------------------- [4]
    print("\n[4] tool call round trip")
    tsys = {"role": "system", "content": "Tool-using agent."}
    tu = {"role": "user", "content": "Check the events for me."}
    NEXT = {"reasoning": "I should look at events.",
            "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": "kubectl",
                                         "arguments": '{"args": "-n ai get events"}'}}]}
    await chat([tsys, tu])
    await settle()
    a_tc = {"role": "assistant", "content": None,
            "tool_calls": NEXT["tool_calls"]}
    tool_msg = {"role": "tool", "tool_call_id": "call_1",
                "content": "Failed to pull image: manifest unknown"}
    NEXT = {"content": "The registry rejected the pull."}
    await chat([tsys, tu, a_tc, tool_msg])
    await settle()
    sweep()
    tool_files = [f for f in capture_files() if f != main_file]
    check("tool conversation is its own transcript", len(tool_files) == 1,
          str([f.name for f in tool_files]))
    if tool_files:
        ttext = tool_files[0].read_text(encoding="utf-8")
        tr = rows(ttext)
        check("tool call rendered", "Tool call · `kubectl`" in ttext)
        check("tool arguments reassembled from fragments",
              '"-n ai get events"' in ttext, ttext[:0])
        check("tool result rendered with its name",
              "Tool result · <code>kubectl</code>" in ttext)
        check("tool result body kept",
              "manifest unknown" in ttext)
        check("tool message renders under a parseable sender",
              [x[2] for x in tr] == ["User", "User", "Claude", "User", "Claude"],
              str([x[2] for x in tr]))

    # ---------------------------------------------------------------- [5]
    print("\n[5] identical opening, re-fired after flush")
    alert = [{"role": "system", "content": "alert handler"},
             {"role": "user", "content": "ALERT: longhorn volume degraded"}]
    NEXT = {"content": "Acknowledged."}
    before = len(capture_files())
    await chat(alert)
    await settle()
    sweep()
    mid = capture_files()
    await chat(alert)
    await settle()
    sweep()
    after = capture_files()
    check("re-fired identical alert forks into its own transcript",
          len(after) == before + 2, f"{before} -> {len(mid)} -> {len(after)}")
    alert_files = [f for f in after if "longhorn" in f.name]
    uuids = {vaultio.parse_frontmatter(f.read_text(encoding="utf-8"))
             .get("source_uuid") for f in alert_files}
    check("the two runs have distinct source_uuids", len(uuids) == 2, str(uuids))
    check("collision suffix applied to the second filename",
          any("(" in f.name for f in alert_files),
          str([f.name for f in alert_files]))
    if alert_files:
        afm = vaultio.parse_frontmatter(
            alert_files[0].read_text(encoding="utf-8"))
        check("single exchange recorded as exchanges: 1",
              afm.get("exchanges") == "1", afm.get("exchanges"))

    # ---------------------------------------------------------------- [6]
    print("\n[6] non-streamed response")
    NEXT = {"content": "42.", "reasoning": "Counting."}
    obj = await chat([{"role": "user", "content": "Non streaming question?"}],
                     stream=False)
    check("non-streamed reply reaches the client intact",
          isinstance(obj, dict)
          and obj["choices"][0]["message"]["content"] == "42.", str(obj)[:200])
    await settle()
    sweep()
    ns = [f for f in capture_files() if "Non streaming" in f.name]
    check("non-streamed exchange captured", len(ns) == 1,
          str([f.name for f in capture_files()]))
    if ns:
        nstext = ns[0].read_text(encoding="utf-8")
        check("non-streamed reasoning captured", "Counting." in nstext)
        check("usage recorded", "prompt_tokens" in nstext)

    # ---------------------------------------------------------------- [7]
    print("\n[7] client hangs up mid-stream")
    # The client going away does not reliably stop the tee: the proxy keeps
    # draining upstream and archives the whole reply. That is the better
    # outcome — what the model produced is what gets kept — so the assertion
    # is that the exchange survives, not that it is marked short.
    NEXT = {"content": "one two three four five six seven eight nine ten",
            "delay": 0.08}
    await chat([{"role": "user", "content": "Abort me please."}],
               abort_after=3)
    await settle()
    sweep()
    ab = [f for f in capture_files() if "Abort me" in f.name]
    check("aborted stream still captured", len(ab) == 1,
          str([f.name for f in capture_files()]))
    if ab:
        abtext = ab[0].read_text(encoding="utf-8")
        check("what arrived is kept verbatim", "one two three" in abtext)

    # The truncation path itself, where the generator really is closed early.
    from gateway.capture import render, sse as ssemod
    acc = ssemod.ChatAccumulator()
    acc.feed(_frame({"content": "half a th"}))
    cut = acc.message(truncated=True)
    cut.ts = "2026-01-01T00:00:00Z"
    md = render.render_message(cut, "0" * 8, {})
    check("a genuinely cut-short reply is labelled",
          "client disconnected" in md and "stream_incomplete" in md, md)
    check("and keeps what it had", "half a th" in md)

    # ---------------------------------------------------------------- [8]
    print("\n[8] passthrough and accounting")
    async with httpx.AsyncClient(timeout=10) as c:
        h = await c.get(f"{PROXY}/health")
        check("non-chat endpoint proxied verbatim",
              h.status_code == 200 and h.json() == {"status": "ok"}, h.text)
        m = await c.get(f"{PROXY}/__capture/metrics")
        check("metrics served on the proxy's own path",
              "capture_requests_total" in m.text)
        s = await c.get(f"{PROXY}/__capture/status")
        check("status endpoint answers", s.status_code == 200, s.text)

    # ---------------------------------------------------------------- [9]
    print("\n[9] identified machine traffic writes nothing")
    NEXT = {"content": '{"node": "One-Offs", "tags": [], "confidence": 0.4}'}
    before_files = set(capture_files())
    before_index = set((VAULT / ".staging" / "Chats" / "Live Capture"
                        / ".index").glob("*.json"))
    obj = await chat([{"role": "system", "content": "huge CLAUDE.md here"},
                      {"role": "user", "content": "Where does this file go?"}],
                     stream=False,
                     headers={"X-Capture-Client": "agentmemory-filing"})
    await settle()
    sweep()
    check("suppressed call is still proxied normally",
          isinstance(obj, dict) and "One-Offs" in
          obj["choices"][0]["message"]["content"], str(obj)[:160])
    check("no transcript written", set(capture_files()) == before_files,
          str({f.name for f in set(capture_files()) - before_files}))
    after_index = set((VAULT / ".staging" / "Chats" / "Live Capture"
                       / ".index").glob("*.json"))
    check("nothing buffered either", after_index == before_index,
          str({f.name for f in after_index - before_index}))
    async with httpx.AsyncClient(timeout=10) as c:
        mt = (await c.get(f"{PROXY}/__capture/metrics")).text
    check("but it is counted, not silently dropped",
          'capture_suppressed_total{client="agentmemory-filing"} 1.0' in mt,
          [l for l in mt.splitlines() if "suppressed" in l])

    print("\n[9b] an unrecognised client name suppresses nothing")
    NEXT = {"content": "captured as normal"}
    await chat([{"role": "user", "content": "Rogue client question."}],
               stream=False, headers={"X-Capture-Client": "not-on-the-list"})
    await settle()
    sweep()
    check("unknown client is captured like any other",
          len([f for f in capture_files() if "Rogue client" in f.name]) == 1,
          str([f.name for f in capture_files()]))

    # --------------------------------------------------------------- [9c]
    print("\n[9c] context compaction")
    AUTH = {"Authorization": "Bearer testkey"}
    NEXT = {"content": "short answer"}
    await chat([{"role": "user", "content": "tiny"}], stream=False, headers=AUTH)
    check("a small conversation is forwarded untouched",
          len(LAST_UPSTREAM.get("messages", [])) == 1
          and not any(str(m.get("content","")).startswith("[COMPACTED")
                      for m in LAST_UPSTREAM["messages"]),
          str(len(LAST_UPSTREAM.get("messages", []))))

    # n_ctx 2000 * 0.75 - 100 reserve = 1400 tokens = ~5600 chars at 4 chars/tok
    big = [{"role": "system", "content": "You are the cluster agent."}]
    for i in range(30):
        big.append({"role": "user", "content": f"question {i} " + "x" * 300})
        big.append({"role": "assistant", "content": f"answer {i} " + "y" * 300})
    big.append({"role": "user", "content": "final question, please answer"})
    NEXT = {"content": "compacted answer"}
    await chat(big, stream=False, headers=AUTH)
    up = LAST_UPSTREAM.get("messages", [])
    check("an oversized conversation is compacted", len(up) < len(big),
          f"{len(big)} -> {len(up)}")
    check("the system prompt survives verbatim",
          up and up[0]["role"] == "system"
          and up[0]["content"] == "You are the cluster agent.", str(up[:1])[:120])
    check("a state summary replaces the middle",
          any("[COMPACTED CONVERSATION STATE]" in str(m.get("content", ""))
              for m in up))
    check("the recent tail survives verbatim",
          up and up[-1]["content"] == "final question, please answer",
          str(up[-1])[:120] if up else "none")
    await settle()
    sweep()
    cf = [f for f in capture_files() if "question 0" in f.name]
    check("the transcript archives the FULL history, not the compacted one",
          len(cf) == 1 and cf[0].read_text(encoding="utf-8").count("question ") >= 30,
          str([f.name for f in capture_files()]))
    if cf:
        ctext = cf[0].read_text(encoding="utf-8")
        check("and records that the model answered from a summary",
              "context_compacted" in ctext)

    print("\n[9d] compaction never breaks a request")
    NEXT = {"content": "unauth answer"}
    await chat(big, stream=False)          # no Authorization -> cannot compact
    check("without a key the request is forwarded whole, not refused",
          len(LAST_UPSTREAM.get("messages", [])) == len(big),
          f"{len(LAST_UPSTREAM.get('messages', []))} vs {len(big)}")

    from gateway.compact import Compactor
    paired = [{"role": "system", "content": "s"}]
    for i in range(6):
        paired.append({"role": "user", "content": f"u{i}"})
        paired.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "function": {"name": "kubectl", "arguments": "{}"}}]})
        paired.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"r{i}"})
    head, cut = Compactor._split(paired)
    tail = paired[cut:]
    ids = {tc["id"] for m in tail for tc in (m.get("tool_calls") or [])}
    check("a cut never orphans a tool result from its call",
          tail[0]["role"] != "tool"
          and all(m.get("tool_call_id") in ids
                  for m in tail if m["role"] == "tool"),
          f"cut={cut} roles={[m['role'] for m in tail]}")

    # --------------------------------------------------------------- [10]
    print("\n[10] sampling belongs to the server when stripping is on")
    import gateway.config as gcfg
    SAMPLING = {"temperature": 0.9, "top_p": 0.5, "top_k": 3,
                "presence_penalty": 1.0, "max_tokens": 64}
    NEXT = {"content": "sampled"}
    await chat([{"role": "user", "content": "Default passthrough sampling."}],
               stream=False, extra=SAMPLING)
    check("off by default: client sampling is forwarded untouched",
          LAST_UPSTREAM.get("temperature") == 0.9
          and LAST_UPSTREAM.get("top_k") == 3, str(LAST_UPSTREAM)[:200])
    was_strip, gcfg.STRIP_SAMPLING = gcfg.STRIP_SAMPLING, True
    try:
        await chat([{"role": "user", "content": "Stripped sampling please."}],
                   stream=False, extra=SAMPLING)
        check("on: every sampling field is dropped",
              not any(k in LAST_UPSTREAM for k in
                      ("temperature", "top_p", "top_k", "presence_penalty")),
              str(LAST_UPSTREAM)[:200])
        check("but non-sampling fields survive",
              LAST_UPSTREAM.get("max_tokens") == 64
              and LAST_UPSTREAM.get("messages"), str(LAST_UPSTREAM)[:200])
        await chat([{"role": "user", "content": "Streamed stripped sampling."}],
                   extra=SAMPLING)
        check("streamed requests are stripped too",
              "temperature" not in LAST_UPSTREAM
              and LAST_UPSTREAM.get("stream") is True, str(LAST_UPSTREAM)[:200])
        await chat([{"role": "user", "content": "Classify this file."}],
                   stream=False, extra={"temperature": 0.1},
                   headers={"X-Capture-Client": "agentmemory-filing"})
        check("an allowlisted machine client keeps its deliberate sampling",
              LAST_UPSTREAM.get("temperature") == 0.1, str(LAST_UPSTREAM)[:200])
        await chat([{"role": "user", "content": "Pretend client."}],
                   stream=False, extra={"temperature": 0.1},
                   headers={"X-Capture-Client": "not-on-the-list"})
        check("an unrecognised client name keeps nothing",
              "temperature" not in LAST_UPSTREAM, str(LAST_UPSTREAM)[:200])
        big_s = [dict(m) for m in big]
        big_s[-1] = {"role": "user", "content": "final sampled question"}
        await chat(big_s, stream=False, headers=AUTH, extra=SAMPLING)
        check("a compacted request is rebuilt without the client's sampling",
              any("[COMPACTED CONVERSATION STATE]" in str(m.get("content", ""))
                  for m in LAST_UPSTREAM.get("messages", []))
              and "temperature" not in LAST_UPSTREAM, str(LAST_UPSTREAM)[:200])
        async with httpx.AsyncClient(timeout=10) as c:
            mt = (await c.get(f"{PROXY}/__capture/metrics")).text
        check("stripping is counted",
              "gateway_sampling_stripped_total 4.0" in mt,
              [l for l in mt.splitlines() if "sampling_stripped" in l])
    finally:
        gcfg.STRIP_SAMPLING = was_strip

    # --------------------------------------------------------------- [11]
    print("\n[11] tool loop")
    import gateway.tools as gtools
    check("guard allows a plain read",
          gtools.kubectl_guard(["get", "pods", "-n", "ai"]) is None)
    check("guard refuses secrets outright",
          gtools.kubectl_guard(["get", "secrets"]) is not None)
    check("guard refuses exec",
          gtools.kubectl_guard(["exec", "-it", "pod"]) is not None)
    # The 2026-09-17 bypass: flags before the verb hid it from the guard.
    for argv in (["-n", "media", "exec", "deploy/x", "--", "sh"],
                 ["--namespace=media", "exec", "deploy/x"],
                 ["--context", "c", "-n", "ai", "attach", "p"],
                 ["-n", "x", "debug", "node/n"],
                 ["alpha", "debug", "p"],
                 ["--kubeconfig=/tmp/k", "get", "pods"],
                 ["--some-flag", "get", "pods"],
                 ["-n", "media"]):
        check(f"guard refuses {' '.join(argv)}", gtools.kubectl_guard(argv) is not None,
              str(gtools.kubectl_guard(argv)))
    check("guard allows a namespace flag before a read",
          gtools.kubectl_guard(["-n", "media", "get", "pods"]) is None,
          str(gtools.kubectl_guard(["-n", "media", "get", "pods"])))
    check("secrets are refused even behind a namespace flag",
          gtools.kubectl_guard(["-n", "media", "get", "secrets"]) is not None)
    check("a write behind a namespace flag counts as a mutation",
          gtools.is_mutation("run_kubectl", {"args": ["-n", "media", "delete", "pod", "p"]}))
    check("a read behind a namespace flag is not a mutation",
          not gtools.is_mutation("run_kubectl", {"args": ["-n", "media", "get", "pods"]}))
    check("guard refuses acting on a protected component",
          gtools.kubectl_guard(["rollout", "restart", "deploy/llm-expert"]) is not None)
    check("guard refuses mutations in propose mode",
          gtools.kubectl_guard(["delete", "pod", "something"]) is not None)

    # The action surface is now "anything kubectl can do" minus four structural
    # refusals. These check the refusals are the ones intended and that ordinary
    # repair verbs are not among them.
    check("guard refuses a shell into a workload",
          gtools.kubectl_guard(["port-forward", "svc/x", "8080"]) is not None)
    check("guard refuses impersonation",
          gtools.kubectl_guard(["get", "pods", "--as=system:admin"]) is not None)
    check("guard refuses touching service accounts",
          gtools.kubectl_guard(["delete", "serviceaccount", "x"]) is not None)
    check("guard refuses RBAC edits",
          gtools.kubectl_guard(["patch", "clusterrolebinding", "x"]) is not None)
    check("guard refuses a secret by slash spelling",
          gtools.kubectl_guard(["get", "secret/db-creds"]) is not None)
    real_mode2, gcfg.MODE = gcfg.MODE, "auto"
    try:
        for argv, label in ((["patch", "deployment", "x", "-p", "{}"], "patch"),
                            (["cordon", "whitehorse-media"], "cordon"),
                            (["drain", "whitehorse-media", "--ignore-daemonsets"], "drain"),
                            (["scale", "deploy/x", "--replicas=0"], "scale"),
                            (["delete", "volumeattachment", "csi-abc"], "delete volumeattachment")):
            check(f"auto mode allows {label}", gtools.kubectl_guard(argv) is None,
                  str(gtools.kubectl_guard(argv)))
        check("but still not into the brain it is thinking with",
              gtools.kubectl_guard(["scale", "deploy/llm-expert", "--replicas=0"]) is not None)
    finally:
        gcfg.MODE = real_mode2

    names = {t["function"]["name"] for t in gtools.TOOLS}
    check("memory search and context are both offered, not just search",
          {"search_memory", "get_context"} <= names, str(sorted(names)))

    # silence_alert: the mutation that lets a known-accepted alert stop costing
    # attention. Every guard here is load-bearing — a silence that is too broad,
    # unexplained, or longer than its reason is worse than the alert.
    check("silencing is offered", "silence_alert" in names, str(sorted(names)))

    # An empty finish() is not an answer. A phone client asked a question, the
    # model ran kubectl, then ended its turn with nothing in the report and the
    # caller got the literal string "(no summary given)".
    import gateway.agentloop as gloop
    check("a finish with a summary renders it",
          gloop._render_finish({"summary": "it was DNS"}) == "it was DNS")
    check("an empty finish is recognised as saying nothing",
          gloop._render_finish({}) == gloop.NOTHING_SAID, gloop._render_finish({}))
    check("and so is one with only blank fields",
          gloop._render_finish({"summary": "   ", "actions_taken": []}) == gloop.NOTHING_SAID)
    real_am, gcfg.ALERTMANAGER_URL = gcfg.ALERTMANAGER_URL, "http://am.test:9093"
    real_mode = gcfg.MODE
    try:
        check("a silence without an alertname is refused as too broad",
              (await gtools.silence_alert("", "operator said ignore it, cable moved"))
              .startswith("REFUSED"))
        check("a silence with no real reason is refused",
              (await gtools.silence_alert("NodeBondingDegraded", "known"))
              .startswith("REFUSED"))
        gcfg.MODE = "propose"
        out = await gtools.silence_alert("NodeBondingDegraded",
                                         "operator moved the cable; restoring when new "
                                         "cable is run", 168, {"instance": "192.168.1.12"})
        check("in propose mode it is recorded, not sent", out.startswith("PROPOSAL RECORDED"))
        check("and the proposal keeps the extra matcher, so the silence stays narrow",
              "192.168.1.12" in out, out)
        gcfg.SILENCE_MAX_HOURS = 720
        out = await gtools.silence_alert("X", "operator said to ignore this one", 99999)
        check("duration is capped rather than honoured", "720h" in out, out)
    finally:
        gcfg.ALERTMANAGER_URL, gcfg.MODE = real_am, real_mode
    check("search's description points at get_context rather than re-searching",
          "get_context" in [t["function"]["description"]
                            for t in gtools.TOOLS
                            if t["function"]["name"] == "search_memory"][0])
    check("an unknown tool is refused, not crashed on",
          (await gtools.dispatch("nope", {})).startswith("REFUSED"))

    # Nothing changes without the channel hearing about it. This is the
    # condition the agent is allowed to act under, so it is tested, not trusted.
    check("a read is not an action", not gtools.is_mutation("run_kubectl", {"args": ["get", "pods"]}))
    check("memory search is not an action", not gtools.is_mutation("search_memory", {"query": "x"}))
    check("deleting a pod is an action", gtools.is_mutation("run_kubectl", {"args": ["delete", "pod", "p"]}))
    check("a rollout restart is an action",
          gtools.is_mutation("run_kubectl", {"args": ["rollout", "restart", "deploy/x"]}))
    check("silencing is an action", gtools.is_mutation("silence_alert", {}))

    # Every tool call is reported on the stream as a structured event, before
    # and after, marked when it changes something. The caller decides where to
    # show it; the gateway knows nothing about Discord.
    check("the gateway has no Discord settings",
          not any(k.startswith("DISCORD") for k in vars(gcfg)))
    check("and no announcer", not hasattr(gtools, "announce") and not hasattr(gtools, "ANNOUNCE_TO"))
    said: list[dict] = []
    async def _emit(d): said.append(d["tool_event"])
    await gtools.dispatch("run_kubectl", {"args": ["get", "pods", "-n", "ai"]}, _emit)
    check("a read is reported as a non-mutating call and result",
          [(e["phase"], e["mutating"]) for e in said] == [("call", False), ("result", False)], str(said))
    said.clear()
    await gtools.dispatch("run_kubectl", {"args": ["delete", "pod", "nope-does-not-exist"]}, _emit)
    check("a mutation is reported before and after, marked mutating",
          [(e["phase"], e["mutating"]) for e in said] == [("call", True), ("result", True)], str(said))
    check("the attempt carries its arguments even when it is refused",
          said and said[0]["args"] == {"args": ["delete", "pod", "nope-does-not-exist"]}, str(said))
    check("and the outcome carries a summary", said and said[-1].get("summary"), str(said))
    check("without a listener, dispatch still runs",
          isinstance(await gtools.dispatch("run_kubectl", {"args": ["get", "ns"]}), str))
    async def _fake_ctx(u, r=3):
        return f"WINDOW around {u} radius {r}"
    real_ctx, gtools.get_context = gtools.get_context, _fake_ctx
    try:
        check("get_context dispatches with its uuid and radius",
              await gtools.dispatch("get_context",
                                    {"message_uuid": "abc", "radius": 5})
              == "WINDOW around abc radius 5")
    finally:
        gtools.get_context = real_ctx

    from gateway import policy
    merged = policy.apply([{"role": "system", "content": "You are Jeeves."},
                           {"role": "user", "content": "hi"}])
    check("policy is appended to the caller's system message, not replacing it",
          merged[0]["role"] == "system"
          and merged[0]["content"].startswith("You are Jeeves.")
          and "get_context" in merged[0]["content"], str(merged[0])[:120])
    check("policy does not disturb the rest of the conversation",
          merged[1:] == [{"role": "user", "content": "hi"}])
    added = policy.apply([{"role": "user", "content": "hi"}])
    check("a conversation with no system message gets one",
          added[0]["role"] == "system" and len(added) == 2)

    # run_kubectl is stubbed: a test that shells out to the real cluster is
    # not a test, it is an incident.
    real_kubectl, gtools.run_kubectl = gtools.run_kubectl, (
        lambda args: f"FAKE kubectl {' '.join(args)}\npod/jellyfin-0  Running")
    was_enabled, gcfg.TOOLS_ENABLED = gcfg.TOOLS_ENABLED, True
    try:
        NEXT_QUEUE[:] = [
            {"reasoning": "I should look at the pods.",
             "tool_calls": [{"id": "t1", "type": "function", "function": {
                 "name": "run_kubectl",
                 "arguments": '{"args": ["get", "pods", "-n", "ai"]}'}}]},
            {"content": "One pod is running in ai: jellyfin-0."},
        ]
        obj = await chat([{"role": "user", "content": "Which pods run in ai?"}],
                         stream=False, headers=AUTH)
        check("the client gets a finished answer, never a tool call",
              isinstance(obj, dict)
              and "jellyfin-0" in (obj["choices"][0]["message"].get("content") or "")
              and not obj["choices"][0]["message"].get("tool_calls"),
              str(obj)[:200])
        await settle()
        sweep()
        tf = [f for f in capture_files() if "Which pods run in ai" in f.name]
        check("the loop is archived even though the client never saw it",
              len(tf) == 1, str([f.name for f in capture_files()]))
        if tf:
            t = tf[0].read_text(encoding="utf-8")
            check("tool call recorded", "Tool call · `run_kubectl`" in t)
            check("tool result recorded", "FAKE kubectl get pods -n ai" in t)
            check("reasoning recorded", "I should look at the pods." in t)
            senders = [x[2] for x in rows(t)]
            check("every archived step still chunks with a sender",
                  senders == ["User", "Claude", "User", "Claude"], str(senders))
        # Wall clock, not just steps: a caller with its own timeout (cluster-agent
        # waits 300s) must get an answer, not an abandoned investigation.
        was_secs, gcfg.TOOL_MAX_SECONDS = gcfg.TOOL_MAX_SECONDS, 0
        try:
            NEXT_QUEUE[:] = [{"content": "Answering from what I have."}]
            obj = await chat([{"role": "user", "content": "Out of time please."}],
                             stream=False, headers=AUTH)
            check("a spent time budget withdraws the tools and forces an answer",
                  "tools" not in LAST_UPSTREAM
                  and "Answering from what I have."
                  in (obj["choices"][0]["message"].get("content") or ""),
                  f"tools_offered={'tools' in LAST_UPSTREAM}")
            check("and it says so in the prompt, so the model knows why",
                  "tool budget is spent" in json.dumps(LAST_UPSTREAM))
        finally:
            gcfg.TOOL_MAX_SECONDS = was_secs

        # ------------------------------------------------------------ [11b]
        print("\n[11b] a streaming client watches the loop live")
        NEXT_QUEUE[:] = [
            {"reasoning": "I need to look at the pods first.", "delay": 0.03,
             "tool_calls": [{"id": "s1", "type": "function", "function": {
                 "name": "run_kubectl",
                 "arguments": '{"args": ["get", "pods", "-n", "media"]}'}}]},
            {"reasoning": "jellyfin-0 is the only pod.", "delay": 0.03,
             "content": "One pod runs in media: jellyfin-0."},
        ]
        frames = await stream_frames(
            [{"role": "user", "content": "Stream me the media pods."}], AUTH)
        thinking = deltas(frames, "reasoning_content")
        answer = deltas(frames, "content")
        check("thinking from every step reaches the client",
              "look at the pods first" in thinking and "only pod" in thinking, thinking)
        check("each tool call is announced inside the thinking",
              "[tool] run_kubectl: kubectl get pods -n media" in thinking, thinking)
        events = [o["choices"][0]["delta"]["tool_event"] for _, o in frames
                  if o.get("choices") and o["choices"][0]["delta"].get("tool_event")]
        check("each tool call streams as a structured call + result event",
              [(e["phase"], e["name"]) for e in events]
              == [("call", "run_kubectl"), ("result", "run_kubectl")]
              and events[0]["args"] == {"args": ["get", "pods", "-n", "media"]}
              and not events[0]["mutating"], str(events))
        check("the answer is streamed and complete",
              answer == "One pod runs in media: jellyfin-0.", repr(answer))
        r_times = [t for t, o in frames if o.get("choices")
                   and o["choices"][0]["delta"].get("reasoning_content")]
        check("thinking arrives token by token over time, not in one burst",
              len(r_times) > 5 and r_times[-1] - r_times[0] > 0.15,
              f"{len(r_times)} chunks over {r_times[-1] - r_times[0] if r_times else 0:.2f}s")
        check("the client never sees a tool_call delta",
              not any(o["choices"][0]["delta"].get("tool_calls")
                      for _, o in frames if o.get("choices")))
        check("the stream ends with a finish_reason",
              frames and frames[-1][1]["choices"][0]["finish_reason"] == "stop",
              str(frames[-1][1]) if frames else "no frames")
        check("upstream steps were requested as streams",
              LAST_UPSTREAM.get("stream") is True, str(LAST_UPSTREAM.get("stream")))
        await settle()
        sweep()
        sf = [f for f in capture_files() if "Stream me the media pods" in f.name]
        check("a streamed loop is archived like any other", len(sf) == 1,
              str([f.name for f in capture_files()]))
        if sf:
            st = sf[0].read_text(encoding="utf-8")
            check("with its tool call and result",
                  "Tool call · `run_kubectl`" in st
                  and "FAKE kubectl get pods -n media" in st)

        NEXT_QUEUE[:] = [
            {"tool_calls": [{"id": "f2", "type": "function", "function": {
                "name": "finish",
                "arguments": json.dumps({"summary": "Streamed report."})}}]},
        ]
        frames = await stream_frames(
            [{"role": "user", "content": "Finish over a stream."}], AUTH)
        check("a finish() report is delivered as streamed content",
              deltas(frames, "content") == "Streamed report.",
              repr(deltas(frames, "content")))

        NEXT_QUEUE[:] = [{"status": 500}, {"status": 500}, {"status": 500}]
        frames = await stream_frames(
            [{"role": "user", "content": "Break over a stream."}], AUTH)
        check("a failure after the stream started is reported in-band",
              "[gateway: agent loop failed" in deltas(frames, "content"),
              repr(deltas(frames, "content")))
        NEXT_QUEUE.clear()

        NEXT_QUEUE[:] = [
            {"tool_calls": [{"id": "f1", "type": "function", "function": {
                "name": "finish",
                "arguments": json.dumps({"summary": "Image pull failed.",
                                         "actions_taken": ["read events"],
                                         "proposals": ["kubectl -n ai get ev"]})}}]},
        ]
        obj = await chat([{"role": "user", "content": "Diagnose the alert."}],
                         stream=False, headers=AUTH)
        content = obj["choices"][0]["message"].get("content") or ""
        check("finish() ends the run and becomes the answer",
              "Image pull failed." in content and "Proposals" in content
              and not obj["choices"][0]["message"].get("tool_calls"),
              content[:200])
    finally:
        gcfg.TOOLS_ENABLED = was_enabled
        gtools.run_kubectl = real_kubectl
        NEXT_QUEUE.clear()



async def run_git():
    print("\n[12] git tools: read, and propose changes only as PRs")
    import gateway.config as gcfg
    import gateway.gittools as git
    import gateway.tools as gtools
    gh_reset()
    saved = (gcfg.GIT_TOKEN, gcfg.GIT_API)
    gcfg.GIT_API = "http://127.0.0.1:18000/gh"
    try:
        gcfg.GIT_TOKEN = ""
        check("no token: git tools are not offered at all",
              not ({t["function"]["name"] for t in gtools.offered()} & git.NAMES))
        check("no token: a git call explains what is missing",
              "not configured" in await git.list_files("Whitehorse"))
        gcfg.GIT_TOKEN = "ghtest"
        check("with a token: git tools are offered",
              git.NAMES <= {t["function"]["name"] for t in gtools.offered()})
        import tempfile as _tf, os as _os
        tok = _tf.NamedTemporaryFile("w", delete=False); tok.write("fromfile\n"); tok.close()
        saved_env, saved_file = gcfg.GIT_TOKEN, gcfg.GIT_TOKEN_FILE
        gcfg.GIT_TOKEN, gcfg.GIT_TOKEN_FILE = "", tok.name
        check("a mounted token file is read without a restart", gcfg.git_token() == "fromfile")
        with open(tok.name, "w") as fh: fh.write("rotated")
        _os.utime(tok.name, (1, 1))
        check("and a rotated one is picked up", gcfg.git_token() == "rotated")
        gcfg.GIT_TOKEN_FILE = tok.name + ".missing"
        check("a missing file means no token", gcfg.git_token() == "")
        gcfg.GIT_TOKEN, gcfg.GIT_TOKEN_FILE = saved_env, saved_file

        out = await git.list_files("whitehorse", "kubernetes/apps")
        check("list is filtered by prefix, repo name case-insensitive",
              "sonarr/helmrelease.yaml" in out and "README" not in out, out)
        out = await git.read_file("Whitehorse", "kubernetes/apps/media/sonarr/helmrelease.yaml")
        check("read returns the file with a line header", "lines 1-4 of 4" in out
              and "memory: 512Mi" in out, out)
        check("a repo outside the allowlist is refused",
              (await git.read_file("upbound-official-build", "x")).startswith("REFUSED"))
        check("search finds text", "sonarr" in await git.search("Whitehorse", "512Mi"))

        path = "kubernetes/apps/media/sonarr/helmrelease.yaml"
        out = await git.open_pr("Whitehorse", "fix(media): raise sonarr memory",
                                "evidence here",
                                [{"path": path, "old": "memory: 512Mi", "new": "memory: 768Mi"}])
        check("an edit opens a PR", out.startswith("opened PR #1"), out)
        pr = GH["pulls"][0]
        branch = pr["head"]["ref"]
        check("on an agent/ branch, against main",
              branch.startswith("agent/fix-media-raise-sonarr-memory-") and pr["base"] == "main", branch)
        check("main is untouched", "512Mi" in GH["trees"][GH["commits"][GH["branches"]["main"]]["tree"]][path])
        check("the branch has the edit",
              "768Mi" in GH["trees"][GH["commits"][GH["branches"][branch]]["tree"]][path])
        check("the PR body says a human merges", "Nothing merges without a human" in pr["body"])

        out = await git.open_pr("Whitehorse", "fix(media): raise sonarr memory more", "again",
                                [{"path": path, "old": "768Mi", "new": "1Gi"}], branch=branch)
        check("revising its own branch updates the same PR, no second PR",
              "updating PR #1" in out and len(GH["pulls"]) == 1, out)
        check("revision reads the branch, not main",
              "1Gi" in GH["trees"][GH["commits"][GH["branches"][branch]]["tree"]][path])

        async def refused(label, **kw):
            args = dict(repo="Whitehorse", title="chore: something reasonable", body="b",
                        changes=[{"path": "README.md", "old": "hello", "new": "bye"}])
            args.update(kw)
            o = await git.open_pr(**args)
            check(label, o.startswith("REFUSED"), o)
        before = dict(GH["branches"])
        await refused("never commits to main", branch="main")
        await refused("never commits to a non-agent branch", branch="feature/x")
        await refused("an ambiguous or missing `old` is refused",
                      changes=[{"path": "README.md", "old": "nope", "new": "x"}])
        await refused("SOPS files are refused",
                      changes=[{"path": "kubernetes/apps/ai/llm-api-key.sops.yaml", "content": "x"}])
        await refused("CI workflows are refused",
                      changes=[{"path": ".github/workflows/build.yml", "content": "x"}])
        await refused("an edit that breaks YAML is refused",
                      changes=[{"path": path, "old": "spec:", "new": "spec: ["}])
        await refused("path traversal is refused",
                      changes=[{"path": "../etc/passwd", "content": "x"}])
        check("and none of the refusals wrote anything", GH["branches"] == before)

        out = await git.open_pr("Whitehorse", "docs: add a note file", "new file",
                                [{"path": "docs/note.md", "content": "note\n"},
                                 {"path": "README.md", "delete": True}])
        files = GH["trees"][GH["commits"][GH["branches"][GH["pulls"][-1]["head"]["ref"]]]["tree"]]
        check("create and delete in one PR", files.get("docs/note.md") == "note\n"
              and "README.md" not in files, str(sorted(files)))
        check("list_prs shows only agent PRs",
              "#1" in await git.list_prs("Whitehorse"))

        said = []
        async def _emit(d): said.append(d["tool_event"])
        await gtools.dispatch("git_read_file", {"repo": "Whitehorse", "path": "README.md"}, _emit)
        check("a git read is not a mutation", not any(e["mutating"] for e in said), str(said))
        said.clear()
        await gtools.dispatch("git_open_pr", {"repo": "Whitehorse", "title": "chore: announce test",
                              "body": "b", "changes": [{"path": "x.md", "content": "x"}]}, _emit)
        check("opening a PR is reported like any other action",
              len(said) == 2 and all(e["mutating"] for e in said)
              and said[0]["name"] == "git_open_pr" and "opened PR" in said[1]["summary"], str(said))
        check("the token never appears in any output",
              not any("ghtest" in str(v) for v in (out, said)))
    finally:
        gcfg.GIT_TOKEN, gcfg.GIT_API = saved


async def run_observe_and_proxy() -> None:
    AUTH = {"Authorization": "Bearer testkey"}
    print("\n[observe] read-only Prometheus / Alertmanager tools")
    import gateway.config as gcfg
    import gateway.tools as gtools
    saved = (gcfg.PROMETHEUS_URL, gcfg.ALERTMANAGER_URL)
    gcfg.PROMETHEUS_URL = "http://127.0.0.1:18000/prom"
    gcfg.ALERTMANAGER_URL = "http://127.0.0.1:18000/am"
    try:
        names = {t["function"]["name"] for t in gtools.offered()}
        check("the read-only tools are offered",
              {"prometheus_query", "prometheus_query_range", "list_alerts", "list_silences"} <= names, names)
        for n in ("prometheus_query", "prometheus_query_range", "list_alerts", "list_silences"):
            check(f"{n} is not an action", not gtools.is_mutation(n, {}))
        out = await gtools.dispatch("prometheus_query", {"query": "up"})
        check("an instant query renders one line per series",
              '2 series' in out and 'up{job="node"} 0' in out, out)
        out = await gtools.dispatch("prometheus_query", {"query": "bad("})
        check("a bad query reports Prometheus' error", "parse error" in out, out)
        out = await gtools.dispatch("prometheus_query_range", {"query": "restarts", "minutes": 5})
        check("a range query summarises and shows where it changed",
              'pod="shared-pg-3"' in out and "max=8" in out and "=3" in out, out)
        out = await gtools.dispatch("list_alerts", {})
        check("alerts list shows state, silenced ones marked",
              "KubePodCrashLooping [active]" in out and "(silenced)" in out, out)
        out = await gtools.dispatch("list_silences", {})
        check("silences list shows only live silences", "s1 [active]" in out and "s0" not in out, out)
        gcfg.PROMETHEUS_URL = gcfg.ALERTMANAGER_URL = ""
        check("unconfigured: the tools are not offered",
              "prometheus_query" not in {t["function"]["name"] for t in gtools.offered()})
    finally:
        gcfg.PROMETHEUS_URL, gcfg.ALERTMANAGER_URL = saved

    print("\n[proxy] encoding and client disconnects")
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{PROXY}/", headers={"Accept-Encoding": "gzip"})
    check("the web UI keeps its Content-Encoding and decodes to HTML",
          r.headers.get("content-encoding") == "gzip" and r.content == UI_HTML,
          (dict(r.headers), r.content[:60]))

    import gateway.main as gmain
    was_enabled = gcfg.TOOLS_ENABLED
    gcfg.TOOLS_ENABLED = True
    try:
        gcfg.TOOLS_ENABLED = False
        SEEN_AE.clear()
        NEXT_QUEUE[:] = [{"content": "ok"}]
        await stream_frames([{"role": "user", "content": "encoding check"}], AUTH)
        check("captured (teed) chat requests ask upstream for identity",
              SEEN_AE and all(a == "identity" for a in SEEN_AE), SEEN_AE)
        gcfg.TOOLS_ENABLED = True

        async def open_and_drop(headers):
            slow = {"reasoning": "thinking " * 60, "delay": 0.05, "content": "late"}
            NEXT_QUEUE[:] = [slow]
            payload = {"model": "m", "stream": True,
                       "messages": [{"role": "user", "content": "drop me"}]}
            async with httpx.AsyncClient(timeout=30) as c:
                async with c.stream("POST", f"{PROXY}/v1/chat/completions",
                                    json=payload, headers=headers) as r:
                    async for line in r.aiter_lines():
                        if "reasoning_content" in line:
                            break          # hang up mid-run
            await asyncio.sleep(2.5)
            return len(gmain._RUNS)

        left = await open_and_drop(AUTH)
        check("a chat client that hangs up cancels its run", left == 0, left)
        left = await open_and_drop({**AUTH, "X-Run-Detached": "1"})
        check("a detached run keeps going after the client hangs up", left == 1, left)
        for _ in range(100):
            if not gmain._RUNS:
                break
            await asyncio.sleep(0.1)
    finally:
        gcfg.TOOLS_ENABLED = was_enabled
        NEXT_QUEUE.clear()


async def main() -> int:
    up = uvicorn.Server(uvicorn.Config(fake, host="127.0.0.1", port=18000,
                                       log_level="error"))
    px = uvicorn.Server(uvicorn.Config(proxy_app, host="127.0.0.1", port=18010,
                                       log_level="error"))
    t1 = asyncio.create_task(up.serve())
    t2 = asyncio.create_task(px.serve())
    for _ in range(200):
        if up.started and px.started:
            break
        await asyncio.sleep(0.05)
    try:
        await run_all()
        await run_git()
        await run_observe_and_proxy()
    finally:
        up.should_exit = px.should_exit = True
        await asyncio.gather(t1, t2, return_exceptions=True)
    print(f"\nvault: {VAULT}")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

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
    MCP_URL="http://127.0.0.1:18000/mcp",
    MCP_TOKEN="mcptest",
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
BAD_HISTORY: list = []   # requests llama.cpp would have refused

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
    # Like llama.cpp: a history whose tool-call arguments are not JSON cannot
    # be rendered into the chat template, and every request carrying it fails.
    for m in body.get("messages") or []:
        for c in m.get("tool_calls") or []:
            try:
                json.loads((c.get("function") or {}).get("arguments") or "{}")
            except ValueError:
                BAD_HISTORY.append(c)
                return JSONResponse({"error": {"code": 500, "message":
                    "Failed to parse tool call arguments as JSON"}}, status_code=500)
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


# --- a fake MCP endpoint: the gateway's only source of tools -------------
import fake_mcp                                                 # noqa: E402

fake.add_api_route("/mcp", fake_mcp.endpoint, methods=["POST"])


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
    import gateway.mcpclient as gmcp
    from gateway import policy
    offered = await gtools.offered()
    names = [t["function"]["name"] for t in offered]
    check("every MCP tool is offered, across list pages, plus finish",
          names == [t["name"] for t in fake_mcp.TOOLS] + ["finish"], names)
    pods = next(t for t in offered if t["function"]["name"] == "kubernetes_pods_list")
    check("an MCP input schema becomes the function parameters",
          pods["function"]["parameters"].get("properties", {}).get("namespace")
          and "$schema" not in pods["function"]["parameters"], str(pods))
    so = next(t for t in offered if t["function"]["name"] == "structured_only")
    check("a tool without a schema still gets an object schema",
          so["function"]["parameters"] == {"type": "object", "properties": {}}, str(so))
    check("the endpoint was called with its token", fake_mcp.STATE["bad_auth"] == 0)
    check("the endpoint's own instructions travel with the policy",
          "FAKE MCP NOTES" in policy.text(), policy.text()[-200:])

    check("a tool annotated read-only is not an action",
          not gtools.is_mutation("kubernetes_pods_list", {}))
    check("a destructive tool is an action", gtools.is_mutation("kubernetes_pods_delete", {}))
    check("an unannotated tool is reported as an action, not exempt",
          gtools.is_mutation("github_push_files", {}))
    import re as _re
    was_re, gcfg.TOOL_READ_ONLY_RE = gcfg.TOOL_READ_ONLY_RE, _re.compile("^github_(get|push)_")
    try:
        check("an unannotated tool the operator names as read-only is not an action",
              not gtools.is_mutation("github_push_files", {}))
        gcfg.TOOL_READ_ONLY_RE = _re.compile("^kubernetes_")
        check("but a tool's own annotation wins over the pattern",
              gtools.is_mutation("kubernetes_pods_delete", {}))
    finally:
        gcfg.TOOL_READ_ONLY_RE = was_re

    # An empty finish() is not an answer. A phone client asked a question, the
    # model ran a tool, then ended its turn with nothing in the report and the
    # caller got the literal string "(no summary given)".
    import gateway.agentloop as gloop
    check("a finish with a summary renders it",
          gloop._render_finish({"summary": "it was DNS"}) == "it was DNS")
    check("a capability gap is reported under its own heading",
          "Capability gaps" in gloop._render_finish(
              {"summary": "x", "capability_gaps": ["no tool to clear an SMB attribute"]}),
          gloop._render_finish({"summary": "x", "capability_gaps": ["no tool"]}))
    check("an empty finish is recognised as saying nothing",
          gloop._render_finish({}) == gloop.NOTHING_SAID, gloop._render_finish({}))
    check("and so is one with only blank fields",
          gloop._render_finish({"summary": "   ", "actions_taken": []}) == gloop.NOTHING_SAID)

    # The gateway holds no credentials and no permission list: every tool
    # comes from the endpoint, every refusal from behind it.
    check("the gateway has no Discord settings",
          not any(k.startswith("DISCORD") for k in vars(gcfg)))
    check("nor any built-in tool, guard or permission setting",
          not any(hasattr(gtools, n) for n in ("run_kubectl", "kubectl_guard", "silence_alert", "TOOLS"))
          and not any(hasattr(gcfg, n) for n in ("MODE", "PROTECTED", "GIT_TOKEN", "HA_TOKEN")))

    said: list[dict] = []
    async def _emit(d): said.append(d["tool_event"])
    out = await gtools.dispatch("kubernetes_pods_list", {"namespace": "ai"}, _emit)
    check("a read is reported as a non-mutating call and result",
          [(e["phase"], e["mutating"]) for e in said] == [("call", False), ("result", False)], str(said))
    check("an SSE-framed result is read past its notifications",
          out.startswith("FAKE pods in ai"), out)
    said.clear()
    out = await gtools.dispatch("kubernetes_pods_delete", {"name": "x", "namespace": "ai"}, _emit)
    check("a mutation is reported before and after, marked mutating",
          [(e["phase"], e["mutating"]) for e in said] == [("call", True), ("result", True)], str(said))
    check("the attempt carries its arguments", said and said[0]["args"] == {"name": "x", "namespace": "ai"})
    check("a refusal from behind the endpoint is handed to the model as an error",
          out.startswith("ERROR:") and "forbidden" in out and said[-1]["error"], out)
    said.clear()
    out = await gtools.dispatch("github_push_files", {"branch": "main"}, _emit)
    check("a policy (JSON-RPC) refusal is an answer, not a crash",
          out.startswith("ERROR:") and "refused by policy" in out and said[-1]["error"], out)
    check("an unknown tool is refused without calling the endpoint",
          (await gtools.dispatch("nope", {})).startswith("ERROR: unknown tool"))
    out = await gtools.dispatch("big_output", {})
    check("oversized output is truncated with a hint", len(out) < 9000 and "truncated" in out, len(out))
    out = await gtools.dispatch("structured_only", {})
    check("structured content is rendered when there is no text", '"n": 3' in out, out)
    check("without a listener, dispatch still runs",
          (await gtools.dispatch("memory_get_context", {"message_uuid": "abc", "radius": 5}))
          == "WINDOW around abc radius 5")

    inits = fake_mcp.STATE["inits"]
    fake_mcp.STATE["expire"] = True
    out = await gtools.dispatch("memory_get_context", {"message_uuid": "u"})
    check("an expired session is re-initialised and the call retried",
          out.startswith("WINDOW around u") and fake_mcp.STATE["inits"] == inits + 1, out)

    real_url = gcfg.MCP_URL
    try:
        gcfg.MCP_URL = "http://127.0.0.1:1/mcp"
        down = [t["function"]["name"] for t in await gtools.offered()]
        check("an unreachable endpoint leaves only finish(), and the request still runs",
              down == ["finish"], down)
        gcfg.MCP_URL = ""
        check("no endpoint configured: only finish()",
              [t["function"]["name"] for t in await gtools.offered()] == ["finish"])
        check("and a call explains why", (await gtools.dispatch("x", {})).startswith("ERROR"))
    finally:
        gcfg.MCP_URL = real_url
    check("back on the real endpoint, the tools return", len(await gtools.offered()) > 1)

    merged = policy.apply([{"role": "system", "content": "You are Jeeves."},
                           {"role": "user", "content": "hi"}])
    check("policy is appended to the caller's system message, not replacing it",
          merged[0]["role"] == "system"
          and merged[0]["content"].startswith("You are Jeeves.")
          and "refused" in merged[0]["content"], str(merged[0])[:120])
    check("policy does not disturb the rest of the conversation",
          merged[1:] == [{"role": "user", "content": "hi"}])
    added = policy.apply([{"role": "user", "content": "hi"}])
    check("a conversation with no system message gets one",
          added[0]["role"] == "system" and len(added) == 2)

    was_enabled, gcfg.TOOLS_ENABLED = gcfg.TOOLS_ENABLED, True
    try:
        NEXT_QUEUE[:] = [
            {"reasoning": "I should look at the pods.",
             "tool_calls": [{"id": "t1", "type": "function", "function": {
                 "name": "kubernetes_pods_list",
                 "arguments": '{"namespace": "ai"}'}}]},
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
            check("tool call recorded", "Tool call · `kubernetes_pods_list`" in t)
            check("tool result recorded", "FAKE pods in ai" in t)
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
                 "name": "kubernetes_pods_list",
                 "arguments": '{"namespace": "media"}'}}]},
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
              '[tool] kubernetes_pods_list: {"namespace": "media"}' in thinking, thinking)
        events = [o["choices"][0]["delta"]["tool_event"] for _, o in frames
                  if o.get("choices") and o["choices"][0]["delta"].get("tool_event")]
        check("each tool call streams as a structured call + result event",
              [(e["phase"], e["name"]) for e in events]
              == [("call", "kubernetes_pods_list"), ("result", "kubernetes_pods_list")]
              and events[0]["args"] == {"namespace": "media"}
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
                  "Tool call · `kubernetes_pods_list`" in st
                  and "FAKE pods in media" in st)

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

        # max_tokens landing mid-tool-call: the arguments arrive cut off. The
        # call must not run (it would run with {}), and the conversation must
        # stay renderable, or llama.cpp 500s every later step and the run dies
        # (2026-09-20: HA incident, "Failed to parse tool call arguments").
        import fake_mcp as _fm
        before = list(_fm.STATE["calls"])
        BAD_HISTORY.clear()
        NEXT_QUEUE[:] = [
            {"tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "kubernetes_pods_get",
                "arguments": '{"namespace": "smart-h'}}]},
            {"tool_calls": [{"id": "c2", "type": "function", "function": {
                "name": "kubernetes_pods_list",
                "arguments": '{"namespace": "smart-home"}'}}]},
            {"content": "Home Assistant is crash-looping."},
        ]
        frames = await stream_frames(
            [{"role": "user", "content": "Cut-off tool call."}], AUTH)
        ran = _fm.STATE["calls"][len(before):]
        check("a cut-off tool call is not run",
              [n for n, _ in ran] == ["kubernetes_pods_list"], str(ran))
        check("and the conversation stays renderable for the model server",
              not BAD_HISTORY, str(BAD_HISTORY)[:200])
        check("the run carries on to an answer",
              "crash-looping" in deltas(frames, "content"),
              repr(deltas(frames, "content"))[:200])
        NEXT_QUEUE.clear()

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
        NEXT_QUEUE.clear()

    # --------------------------------------------------------------- [12]
    print("\n[12] capability packs (on-demand tool loading)")
    import gateway.packs as gpacks
    PACKS = {
        "cluster": {"match": ["kubernetes_*", "memory_*"], "when": "cluster ops",
                    "runbook": "Use kubectl carefully; the NAS is the hard line."},
        "code": {"match": ["github_*"], "when": "git and pull requests",
                 "runbook": "PRs only; never commit to main."},
        "ghost": {"match": ["nothing_here_*"], "when": "matches no live tool",
                  "runbook": "unused"},
    }
    was_packs = gcfg.MCP_PACKS
    gcfg.MCP_PACKS = json.dumps(PACKS)
    gpacks.reload()
    try:
        check("packs enabled once configured", gpacks.enabled())
        base = [t["function"]["name"] for t in await gtools.offered(set())]
        check("deferred: only finish + load_capability offered at the start",
              base == ["finish", "load_capability"], base)
        loaded = set()
        out = await gtools.apply_load("cluster", loaded)
        check("loading a pack reports its tools and returns its runbook",
              "kubernetes_pods_list" in out and "NAS is the hard line" in out
              and "cluster" in loaded, out[:200])
        after = [t["function"]["name"] for t in await gtools.offered(loaded)]
        check("its tools are offered, other packs' are not",
              "kubernetes_pods_list" in after and "memory_get_context" in after
              and "github_push_files" not in after, after)
        check("finish and load_capability stay offered",
              "finish" in after and "load_capability" in after, after)
        again = await gtools.apply_load("cluster", loaded)
        check("loading an already-loaded pack is a no-op message",
              "already loaded" in again, again)
        unknown = await gtools.apply_load("nope", set())
        check("loading an unknown capability lists the valid ones",
              unknown.startswith("ERROR") and "cluster" in unknown and "code" in unknown,
              unknown)
        empty = set()
        ghost = await gtools.apply_load("ghost", empty)
        check("a pack matching no live tool FAILS LOUD, not silently empty",
              ghost.startswith("ERROR") and "NO tools" in ghost and "ghost" not in empty,
              ghost)
        check("the capability manifest is injected into the policy",
              "load_capability" in policy.text() and "cluster: cluster ops" in policy.text(),
              policy.text()[-400:])

        was_enabled2, gcfg.TOOLS_ENABLED = gcfg.TOOLS_ENABLED, True
        try:
            NEXT_QUEUE[:] = [
                {"tool_calls": [{"id": "d0", "type": "function", "function": {
                    "name": "kubernetes_pods_list", "arguments": '{"namespace": "ai"}'}}]},
                {"tool_calls": [{"id": "d1", "type": "function", "function": {
                    "name": "load_capability", "arguments": '{"name": "cluster"}'}}]},
                {"tool_calls": [{"id": "d2", "type": "function", "function": {
                    "name": "kubernetes_pods_list", "arguments": '{"namespace": "ai"}'}}]},
                {"content": "One pod runs in ai: jellyfin-0."},
            ]
            before = len(fake_mcp.STATE["calls"])
            obj = await chat([{"role": "user", "content": "Which pods run in ai?"}],
                             stream=False, headers=AUTH)
            ran = [n for n, _ in fake_mcp.STATE["calls"][before:]]
            check("deferred loop: the unloaded call is blocked, the endpoint is hit "
                  "only after load", ran == ["kubernetes_pods_list"], str(ran))
            check("and the run reaches its answer",
                  "jellyfin-0" in (obj["choices"][0]["message"].get("content") or ""),
                  str(obj)[:200])
        finally:
            gcfg.TOOLS_ENABLED = was_enabled2
            NEXT_QUEUE.clear()
    finally:
        gcfg.MCP_PACKS = was_packs
        gpacks.reload()


async def run_proxy() -> None:
    AUTH = {"Authorization": "Bearer testkey"}
    import gateway.config as gcfg
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
        await run_proxy()
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

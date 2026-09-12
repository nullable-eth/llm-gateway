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


async def chat(messages, stream=True, headers=None, abort_after=None):
    payload = {"model": "qwen3.8-27b", "messages": messages, "stream": stream}
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

    # --------------------------------------------------------------- [11]
    print("\n[11] tool loop")
    import gateway.config as gcfg
    import gateway.tools as gtools
    check("guard allows a plain read",
          gtools.kubectl_guard(["get", "pods", "-n", "ai"]) is None)
    check("guard refuses secrets outright",
          gtools.kubectl_guard(["get", "secrets"]) is not None)
    check("guard refuses exec",
          gtools.kubectl_guard(["exec", "-it", "pod"]) is not None)
    check("guard refuses acting on a protected component",
          gtools.kubectl_guard(["rollout", "restart", "deploy/llm-expert"]) is not None)
    check("guard refuses mutations in propose mode",
          gtools.kubectl_guard(["delete", "pod", "something"]) is not None)

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
    finally:
        gcfg.TOOLS_ENABLED = was_enabled
        gtools.run_kubectl = real_kubectl
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

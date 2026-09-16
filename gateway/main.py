"""Capture proxy entrypoint — `uvicorn proxy.main:app`.

Every path is forwarded. /v1/chat/completions is additionally teed into the
vault; everything else passes through and is counted, so if something ever
starts generating on an endpoint with no capture adapter it shows up in
capture_uncaptured_total rather than being silently missed.

The vault never touches the request path. Capture happens after the last byte
has already reached the client, as one non-blocking put onto a bounded queue;
everything past that point can fail freely without the caller noticing.
"""
import asyncio
import json
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import generate_latest

from . import agentloop, compact, config, forward, metrics, policy, tools
from .capture import normalize, sse, store
from .capture.writer import Writer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("capture")
# httpx logs every request URL at INFO: noise on the serving path, and it once
# wrote a Discord webhook URL (a credential) into the pod log.
logging.getLogger("httpx").setLevel(logging.WARNING)

CHAT_PATH = "/v1/chat/completions"
# llama.cpp serves the un-versioned alias too, and plenty of clients use it —
# a phone app configured with the bare host was posting to /chat/completions and
# getting a raw passthrough: no tools, no compaction, and no capture. That last
# one matters most: "there is no path to the model that is not captured" was
# only true of the path we happened to name.
CHAT_PATHS = (CHAT_PATH, "/chat/completions")
# Generation endpoints with no capture adapter. Nothing here uses them; the
# counter exists so that stays true observably rather than by assumption.
UNADAPTED = {"/v1/completions", "/completions", "/completion", "/infill"}

writer = Writer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = forward.client()
    app.state.compactor = compact.Compactor(app.state.client)
    consume = asyncio.create_task(writer.consume())
    sweep = asyncio.create_task(writer.sweep())
    log.info("capture: proxying %s -> %s, vault %s/%s",
             config.PORT, config.UPSTREAM, config.VAULT_ROOT, config.CAPTURE_DIR)
    try:
        yield
    finally:
        sweep.cancel()
        await writer.drain()          # needs consume alive to drain the queue
        consume.cancel()
        await app.state.client.aclose()


app = FastAPI(title="capture-proxy", lifespan=lifespan)


# ------------------------------------------------------------- own surface
@app.get("/__capture/healthz")
async def healthz():
    return {"ok": True}


@app.get("/__capture/metrics")
async def prom():
    return PlainTextResponse(generate_latest(),
                             media_type="text/plain; version=0.0.4")


@app.get("/__capture/status")
async def status():
    convs = writer.store.convs
    return {"tracked": len(convs),
            "open": sum(1 for c in convs.values() if not c.flushed),
            "dirty": sum(1 for c in convs.values() if c.dirty),
            "queued": writer.queue.qsize()}


# ------------------------------------------------------------------ capture
def _record(parsed, acc, buf, started, headers, truncated, compaction=None) -> None:
    """Build one capture record and hand it to the writer. Never raises."""
    try:
        msgs = normalize.request_messages(parsed)
        if not msgs:
            return
        reply = None
        if acc is not None:
            if not acc.empty():
                reply = acc.message(truncated=truncated)
        elif buf is not None:
            try:
                obj = json.loads(bytes(buf).decode("utf-8", "replace"))
            except ValueError:
                obj = None
            if isinstance(obj, dict):
                reply = normalize.response_message(obj)
                if reply is not None:
                    if isinstance(obj.get("usage"), dict):
                        reply.extra["usage"] = obj["usage"]
                    reply.truncated = truncated
        if truncated:
            metrics.TRUNCATED.inc()
        # The archive keeps the full history, so the transcript reads as if
        # nothing was dropped — which would be misleading on its own. Record
        # that the model answered from a summary, and of what.
        if reply is not None and (compaction or {}).get("compacted"):
            reply.extra["context_compacted"] = compaction
        writer.submit({
            "messages": msgs,
            "reply": reply,
            "ts": started,
            "reply_ts": store.now_iso(),
            "client_id": (headers.get(config.HDR_CONV_ID) or "").strip(),
            "title_hint": (headers.get(config.HDR_TITLE) or "").strip(),
        })
    except Exception:
        log.exception("capture: record failed")
        metrics.DROPPED.labels(reason="record_error").inc()


def _record_loop(parsed, produced, started, headers, compaction) -> None:
    """Archive a tool run: the client's history plus every step the loop took.

    The caller saw one answer; the transcript holds the reasoning, each tool
    call, and each result. That asymmetry is the point of running the loop
    here rather than in the client.
    """
    try:
        msgs = normalize.request_messages(parsed)
        if not msgs:
            return
        steps = [normalize.from_wire(m) for m in produced if isinstance(m, dict)]
        reply = steps.pop() if steps else None
        if reply is not None and (compaction or {}).get("compacted"):
            reply.extra["context_compacted"] = compaction
        writer.submit({
            "messages": msgs + steps,
            "reply": reply,
            "ts": started,
            "reply_ts": store.now_iso(),
            "client_id": (headers.get(config.HDR_CONV_ID) or "").strip(),
            "title_hint": (headers.get(config.HDR_TITLE) or "").strip(),
        })
    except Exception:
        log.exception("capture: loop record failed")
        metrics.DROPPED.labels(reason="record_error").inc()


def _as_client_response(final: dict, wants_stream: bool):
    """Hand the loop's answer back in the shape the client asked for.

    A tool run is many upstream calls, so there is no single stream to relay.
    When the client asked for one, the finished answer is re-emitted as SSE —
    it arrives in chunks rather than token by token, which clients render
    fine and which beats failing the request because it wanted streaming.
    """
    if not wants_stream:
        return JSONResponse(final)

    msg = ((final.get("choices") or [{}])[0].get("message") or {})
    content = msg.get("content") or ""
    model = final.get("model") or "gateway"

    def frame(delta: dict, finish=None) -> bytes:
        return b"data: " + json.dumps({
            "id": final.get("id") or "gw", "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }).encode() + b"\n\n"

    async def gen():
        yield frame({"role": "assistant"})
        if msg.get("reasoning_content"):
            yield frame({"reasoning_content": msg["reasoning_content"]})
        for i in range(0, len(content), 512) or [0]:
            yield frame({"content": content[i:i + 512]})
        yield frame({}, finish=(final.get("choices") or [{}])[0].get(
            "finish_reason") or "stop")
        yield b"data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


KEEPALIVE_S = 15
# asyncio holds tasks weakly. A run whose client hung up is referenced by
# nothing else and could be collected mid-loop; this keeps it alive to finish.
_RUNS: set = set()


def _live_stream(model: str, run, on_done):
    """Stream a tool run as it happens.

    `run(emit)` is the loop; `emit` receives deltas which are relayed as SSE
    the moment they exist, so a client shows the model's thinking live and
    times it honestly. Tool execution produces no tokens, so SSE comments go
    out while it runs to keep proxies from calling the connection idle.

    The loop runs in its own task: a client that hangs up does not cancel it,
    so the archive still gets the whole run, exactly as before. Once the
    response has started, an HTTP status can no longer change, so a failure is
    reported in-band as a final content chunk.
    """
    queue: asyncio.Queue = asyncio.Queue()
    cid = "gw-" + store.now_iso()

    async def emit(delta: dict) -> None:
        await queue.put(("delta", delta))

    async def worker():
        try:
            final, produced = await run(emit)
            on_done(final, produced)
            await queue.put(("done", final))
        except Exception as e:                      # reported in-band below
            await queue.put(("error", e))

    task = asyncio.create_task(worker())
    _RUNS.add(task)
    task.add_done_callback(_RUNS.discard)

    def frame(delta: dict, finish=None, extra=None) -> bytes:
        obj = {"id": cid, "object": "chat.completion.chunk", "model": model,
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if extra:
            obj.update(extra)
        return b"data: " + json.dumps(obj).encode() + b"\n\n"

    async def gen():
        yield frame({"role": "assistant"})
        while True:
            try:
                kind, val = await asyncio.wait_for(queue.get(), KEEPALIVE_S)
            except asyncio.TimeoutError:
                yield b": working\n\n"
                continue
            if kind == "delta":
                yield frame(val)
                continue
            if kind == "done":
                ch = (val.get("choices") or [{}])[0]
                extra = {k: val[k] for k in ("usage", "timings") if val.get(k)}
                yield frame({}, finish=ch.get("finish_reason") or "stop", extra=extra)
            else:
                timed_out = isinstance(val, httpx.TimeoutException)
                metrics.UPSTREAM_ERRORS.labels(endpoint=CHAT_PATH).inc()
                if timed_out:
                    log.warning("agentloop timed out (streaming): %s", val)
                else:
                    log.error("agentloop failed (streaming): %r", val)
                what = "timed out" if timed_out else "failed"
                yield frame({"content": f"\n\n[gateway: agent loop {what}: {val}]"},
                            finish="stop")
            yield b"data: [DONE]\n\n"
            return

    return StreamingResponse(gen(), media_type="text/event-stream")


def _strip_sampling(body: bytes):
    """(new_body, obj) with the client's sampling fields removed, or None when
    there is nothing to remove or the body is not a JSON object."""
    try:
        obj = json.loads(body.decode("utf-8", "replace"))
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    dropped = [k for k in config.STRIP_SAMPLING_FIELDS if k in obj]
    if not dropped:
        return None
    for k in dropped:
        del obj[k]
    metrics.SAMPLING_STRIPPED.inc()
    log.debug("sampling: dropped client fields %s", dropped)
    return json.dumps(obj).encode("utf-8"), obj


# -------------------------------------------------------------- the proxy
@app.api_route("/{path:path}",
               methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD",
                        "OPTIONS"])
async def proxy(path: str, request: Request):
    body = await request.body()
    endpoint = "/" + path
    client: httpx.AsyncClient = request.app.state.client

    started = store.now_iso()
    parsed = None
    if endpoint in CHAT_PATHS and request.method == "POST":
        client_name = (request.headers.get(config.HDR_CLIENT) or "").strip()
        if client_name in config.NOLOG_CLIENTS:
            # Identified machine traffic: proxied exactly like anything else,
            # just not written down. Counted so it is still accounted for.
            metrics.SUPPRESSED.labels(client=client_name).inc()
        else:
            try:
                candidate = json.loads(body.decode("utf-8", "replace"))
            except ValueError:
                candidate = None
            if isinstance(candidate, dict) and candidate.get("messages"):
                parsed = candidate
    elif endpoint in UNADAPTED:
        metrics.UNCAPTURED.labels(endpoint=endpoint).inc()

    # Sampling belongs to the model server, not the client (see
    # config.STRIP_SAMPLING). Dropped from what is FORWARDED only: `parsed`
    # stays the client's own request, so the archive records what was sent,
    # while `fwd` is what compaction and the tool loop build on.
    fwd = parsed
    if (config.STRIP_SAMPLING and endpoint in CHAT_PATHS
            and request.method == "POST"
            and (request.headers.get(config.HDR_CLIENT) or "").strip()
            not in config.SAMPLING_KEEP_CLIENTS):
        stripped = _strip_sampling(body)
        if stripped is not None:
            body, obj = stripped
            if parsed is not None:
                fwd = obj

    # Compaction happens after capture has its copy of the original request and
    # before anything is forwarded, so the archive records what the client sent
    # while the model receives something that fits. Any failure forwards the
    # request untouched.
    compaction = {"compacted": False}
    if parsed is not None:
        try:
            new_body, compaction = await request.app.state.compactor.maybe_compact(
                fwd, request.headers.get("authorization") or "")
        except Exception:
            log.exception("compact: unexpected failure; forwarding unchanged")
            new_body = None
        if new_body is not None:
            body = json.dumps(new_body).encode("utf-8")
            metrics.COMPACTED.inc()

    # Tool link. The loop calls the model repeatedly and runs what it asks
    # for; the client gets one finished reply and needs no tool support of its
    # own. The archive gets the whole loop, which the client never saw.
    if parsed is not None and config.TOOLS_ENABLED:
        auth = request.headers.get("authorization") or ""
        loop_body = dict(new_body or fwd)
        loop_body["messages"] = policy.apply(loop_body.get("messages") or [])
        sent = list(loop_body["messages"])
        wants_stream = bool(parsed.get("stream"))
        if wants_stream:
            def on_done(final, produced):
                metrics.REQUESTS.labels(endpoint=endpoint, streamed="true").inc()
                metrics.TOOL_STEPS.inc(max(0, len(produced) - len(sent)))
                _record_loop(parsed, produced[len(sent):], started,
                             request.headers, compaction)
            compactor = request.app.state.compactor
            return _live_stream(
                str(parsed.get("model") or "gateway"),
                lambda emit: agentloop.run(client, config.UPSTREAM, loop_body,
                                           auth, compactor, emit=emit),
                on_done)
        try:
            final, produced = await agentloop.run(
                client, config.UPSTREAM, loop_body, auth,
                request.app.state.compactor)
        except httpx.TimeoutException as e:
            # 504, never 502. The distinction is the caller's whole retry
            # policy: 502 means "the brain is not there", which is a model
            # rollout and is worth waiting out, while this means "it is there
            # and still thinking", where a retry only starts a second identical
            # loop competing with the first for the same llama.cpp slots.
            # Reported as 502 once, this produced exactly that storm.
            metrics.UPSTREAM_ERRORS.labels(endpoint=endpoint).inc()
            log.warning("agentloop timed out: %s", e)
            return JSONResponse({"error": {"message": f"agent loop timed out: {e}",
                                           "type": "gateway_timeout"}}, status_code=504)
        except Exception as e:
            metrics.UPSTREAM_ERRORS.labels(endpoint=endpoint).inc()
            log.exception("agentloop failed")
            return JSONResponse({"error": {"message": f"agent loop failed: {e}",
                                           "type": "gateway_error"}}, status_code=502)
        metrics.REQUESTS.labels(endpoint=endpoint,
                                streamed=str(bool(parsed.get("stream"))).lower()).inc()
        metrics.TOOL_STEPS.inc(max(0, len(produced) - len(sent)))
        # Splice: the client's own history (never the compacted copy) plus
        # everything the loop produced, so the transcript is complete.
        _record_loop(parsed, produced[len(sent):], started, request.headers,
                     compaction)
        return _as_client_response(final, bool(parsed.get("stream")))

    upstream = client.build_request(
        request.method, forward.url_for(path, request.url.query),
        headers=forward.upstream_headers(request.headers), content=body)
    try:
        resp = await client.send(upstream, stream=True)
    except httpx.HTTPError as e:
        metrics.UPSTREAM_ERRORS.labels(endpoint=endpoint).inc()
        log.warning("capture: upstream error on %s: %s", endpoint, e)
        return JSONResponse(
            {"error": {"message": f"upstream unreachable: {e}",
                       "type": "proxy_error"}}, status_code=502)

    streamed = "text/event-stream" in (resp.headers.get("content-type") or "")
    metrics.REQUESTS.labels(endpoint=endpoint,
                            streamed="true" if streamed else "false").inc()

    capture = parsed is not None and resp.status_code < 400
    acc = sse.ChatAccumulator() if (capture and streamed) else None
    buf = bytearray() if (capture and not streamed) else None

    async def tee():
        truncated = True
        try:
            async for chunk in resp.aiter_raw():
                if acc is not None:
                    acc.feed(chunk)
                elif buf is not None and len(buf) < config.MAX_BODY:
                    buf.extend(chunk)
                yield chunk
            truncated = False
        finally:
            await resp.aclose()
            if capture:
                _record(parsed, acc, buf, started, request.headers, truncated,
                        compaction)

    return StreamingResponse(tee(), status_code=resp.status_code,
                             headers=forward.downstream_headers(resp.headers))

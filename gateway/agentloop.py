"""The agent loop: call the model, run what it asks for, repeat, answer.

The client sends one request and gets one reply. Everything between — the
tool calls, their output, the model's second and third thoughts — happens
here and is invisible to the caller, which is what lets a chat client with no
tool support ask questions that need tools.

It is invisible to the caller but NOT to the archive: the loop hands capture
the whole message list it built, so the transcript holds every tool call and
result even though the client only ever saw the final answer. That is the
single most useful thing this design gets for free.

Compaction runs *between steps*, not only on the way in. Tool output is what
actually overflows a window — a few `kubectl get events` dumps will do it —
and this is the same job cluster-agent's trim_old_results() did before the
gateway took it over.
"""
import asyncio
import json
import logging
import random
import time

import httpx

from . import config, packs, tools
from .capture.sse import ChatAccumulator

log = logging.getLogger("gateway")


class UpstreamError(Exception):
    """A step failed at the model server after its retries."""


def _describe(name: str, args: dict) -> str:
    """One line for the live thinking stream, so a watching client sees what
    the loop is doing instead of a silent pause."""
    detail = json.dumps(args, ensure_ascii=False)
    if len(detail) > 300:
        detail = detail[:300] + "…"
    return f"\n\n[tool] {name}: {detail}\n\n"


async def _post(client, url: str, body: dict, headers: dict, budget: float, emit):
    """One model call. Returns (status, response dict or None, error text).

    Without `emit` it is the plain non-streamed call. With it, the call is
    streamed and every reasoning/content delta is handed to `emit` as it
    arrives, then the step is reassembled into the same dict shape a
    non-streamed call returns, so the loop cannot tell the difference. Tool
    call fragments are NOT forwarded: the client never asked for tools.
    """
    if emit is None:
        r = await client.post(url, json=body, headers=headers, timeout=budget)
        if r.status_code >= 400:
            return r.status_code, None, r.text
        return r.status_code, r.json(), ""

    body = dict(body, stream=True)
    acc = ChatAccumulator()
    tail: dict = {}

    async def consume():
        async with client.stream("POST", url, json=body, headers=headers,
                                 timeout=budget) as r:
            if r.status_code >= 400:
                return r.status_code, (await r.aread()).decode("utf-8", "replace")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except ValueError:
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get("error"):
                    # llama.cpp reports a mid-stream failure (a malformed tool
                    # call, usually) as an error event; same class as a 500.
                    return 500, json.dumps(obj["error"])[:500]
                for k in ("id", "model", "timings"):
                    if obj.get(k) is not None:
                        tail[k] = obj[k]
                acc.event(obj)
                ch = (obj.get("choices") or [{}])[0]
                d = ch.get("delta") if isinstance(ch, dict) else None
                if not isinstance(d, dict):
                    continue
                out = {}
                r_ = d.get("reasoning_content", d.get("reasoning"))
                if isinstance(r_, str) and r_:
                    out["reasoning_content"] = r_
                if isinstance(d.get("content"), str) and d["content"]:
                    out["content"] = d["content"]
                if out:
                    await emit(out)
            return r.status_code, ""

    # A streamed read timeout is per chunk, not per call; bound the whole call
    # so the loop's deadlines mean what they meant when steps were not streamed.
    try:
        status, err = await asyncio.wait_for(consume(), timeout=budget)
    except asyncio.TimeoutError as e:
        raise httpx.ReadTimeout(f"model step exceeded {budget:.0f}s") from e
    if status >= 400:
        return status, None, err

    m = acc.message()
    message = {"role": "assistant", "content": m.content}
    if m.reasoning:
        message["reasoning_content"] = m.reasoning
    if m.tool_calls:
        message["tool_calls"] = [
            {"id": c["id"] or f"call_{i}", "type": "function",
             "function": {"name": c["name"], "arguments": c["arguments"]}}
            for i, c in enumerate(m.tool_calls)]
    resp = {"id": tail.get("id") or "gw", "object": "chat.completion",
            "model": tail.get("model") or acc.model or "",
            "choices": [{"index": 0, "message": message,
                         "finish_reason": acc.finish_reason or "stop"}]}
    if acc.usage:
        resp["usage"] = acc.usage
    if tail.get("timings"):
        resp["timings"] = tail["timings"]
    return status, resp, ""


def _args_of(call: dict) -> dict:
    fn = call.get("function") or {}
    raw = fn.get("arguments")
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except ValueError:
        return {}


NOTHING_SAID = "(no summary given)"


def _render_finish(args: dict) -> str:
    out = [str(args.get("summary") or "").strip()]
    for label, key in (("Actions taken", "actions_taken"), ("Proposals", "proposals"),
                       ("Capability gaps", "capability_gaps")):
        items = args.get(key) or []
        if isinstance(items, str):
            items = [items]
        if items:
            out.append(f"**{label}**\n" + "\n".join(f"- {i}" for i in items))
    return "\n\n".join(x for x in out if x) or NOTHING_SAID


async def run(client, upstream: str, body: dict, auth: str, compactor,
              emit=None) -> tuple[dict, list]:
    """Returns (final upstream response, full message list including tool traffic).

    `emit`, when given, is an async callable receiving chat-completion deltas
    ({"reasoning_content": ...} / {"content": ...}) as they happen: every
    step's thinking, a line per tool call, and the answer, token by token.
    That is what lets a streaming client show live thinking and a real
    "thought for" time instead of one burst when the whole loop is done.
    """
    messages = list(body.get("messages") or [])
    headers = {"Authorization": auth} if auth else {}
    last = None
    started = time.monotonic()
    deadline = started + config.TOOL_MAX_SECONDS
    hard_deadline = started + config.REQUEST_MAX_SECONDS
    asked_again = False          # one nudge, for a turn that came back empty
    loaded: set = set()          # capabilities loaded so far (deferred mode; packs.py)

    for step in range(config.TOOL_MAX_STEPS):
        # Tools are withdrawn on the last step, or once the wall-clock budget
        # is spent, so the model has to answer rather than start work nobody
        # will wait for.
        # Out of time means either budget: the tool budget, or so little left
        # against the hard deadline that another tool step could only be given a
        # stub of a timeout and die in the middle. Answering is always the
        # better use of the last seconds than a step that cannot finish.
        now = time.monotonic()
        remaining = hard_deadline - now
        out_of_time = now >= deadline or remaining < config.UPSTREAM_MIN_TIMEOUT_S
        offer_tools = step < config.TOOL_MAX_STEPS - 1 and not out_of_time
        if out_of_time and step:
            log.info("agentloop: time budget spent after %d step(s); "
                     "forcing an answer", step)
        call_body = dict(body)
        call_body["messages"] = messages
        call_body["stream"] = False
        offered_names: set = set()
        if offer_tools:
            offered_defs = await tools.offered(loaded)
            call_body["tools"] = offered_defs
            offered_names = {(d.get("function") or {}).get("name") for d in offered_defs}
        else:
            call_body.pop("tools", None)
            messages = messages + [{"role": "user", "content":
                "Your tool budget is spent. Answer now from what you already "
                "have, and say plainly what you could not determine."}]
            call_body["messages"] = messages

        # Never timeout=None: an unbounded call outlives the caller, which then
        # abandons the request and retries, and the retry competes with the loop
        # it just abandoned for the same llama.cpp slots.
        #
        # The answer gets its OWN budget rather than the remainder, because the
        # remainder is smallest exactly when the answer matters most — a run
        # that spent its time gathering evidence would otherwise be given ten
        # seconds to say what it found, time out, and report nothing at all.
        budget = (max(config.UPSTREAM_MIN_TIMEOUT_S, hard_deadline - time.monotonic())
                  if offer_tools else config.ANSWER_TIMEOUT_S)
        # llama.cpp answers 500 when a sampled tool call comes out malformed —
        # truncated mid-arguments, usually — and a fresh sample almost always
        # parses. cluster-agent used to absorb that when it talked to the model
        # directly; nothing did after the loop moved here, so one bad sample
        # became a 502 in a phone client's face and a 30-second backoff in the
        # agent's. Retried here, where the bad sample actually happens.
        attempt_body = call_body
        for attempt in range(3):
            status, last, err = await _post(client, f"{upstream}/v1/chat/completions",
                                            attempt_body, headers, budget, emit)
            if status < 500 or attempt == 2:
                break
            log.warning("upstream %s on step %d (attempt %d/3): %s",
                        status, step, attempt + 1, err[:160].replace("\n", " "))
            attempt_body = dict(call_body)
            if attempt == 0:
                # A retry of an identical request is not a resample: sampling is
                # deterministic enough that the same broken tool call comes back
                # verbatim, which is exactly what happened -- three attempts,
                # three identical `"1e4` truncations. Move the sampler.
                attempt_body["seed"] = random.randint(1, 2 ** 31 - 1)
                attempt_body["temperature"] = max(0.7, float(call_body.get("temperature") or 0))
            else:
                # Still broken: the tools are what it keeps malforming, so take
                # them away. A plain answer beats a 502 in someone's chat client.
                attempt_body.pop("tools", None)
                attempt_body["messages"] = messages + [{"role": "user", "content":
                    "Answer directly, in prose, without calling any tool."}]
            await asyncio.sleep(0.5 * (attempt + 1))
        if status >= 400 or last is None:
            raise UpstreamError(f"model server returned {status}: {err[:300]}")
        msg = ((last.get("choices") or [{}])[0].get("message") or {})
        calls = msg.get("tool_calls") or []
        # A tool call whose arguments are not valid JSON was cut off, almost
        # always by max_tokens landing mid-call after a long think. It must
        # neither run nor stay in the history as-is:
        #  - run, it executes with {} (every argument silently dropped), which
        #    is how `kubernetes_pods_get` came to be called with no name;
        #  - kept, llama.cpp cannot render the conversation back into the
        #    template and answers every later step with the same 500 ("Failed
        #    to parse tool call arguments as JSON"). Resampling cannot fix
        #    that: the broken JSON is in the REQUEST, so all three attempts
        #    fail in under a second and the whole run is lost.
        # So the history gets "{}" in its place, and the model is told what
        # happened and asked to make the call again.
        broken = {}
        for call in calls:
            fn = call.get("function") or {}
            raw = fn.get("arguments")
            if isinstance(raw, dict):
                continue
            try:
                ok = isinstance(json.loads(raw or "{}"), dict)
            except ValueError:
                ok = False
            if not ok:
                broken[id(call)] = str(raw or "")
                call["function"] = dict(fn, arguments="{}")
        if broken:
            finish = ((last.get("choices") or [{}])[0].get("finish_reason")) or "?"
            log.warning("agentloop: step %d produced %d cut-off tool call(s) (finish_reason=%s)",
                        step, len(broken), finish)

        assistant = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("reasoning_content"):
            assistant["reasoning_content"] = msg["reasoning_content"]
        if calls:
            assistant["tool_calls"] = calls
        messages.append(assistant)

        if not calls:
            # An empty message is not an answer either. This model puts its
            # thinking in reasoning_content and sometimes ends a tool run with
            # nothing in `content` at all — the phone client got a blank bubble
            # after the gateway had searched the archive for it. Ask once more
            # with the tools withdrawn; if it is still empty, hand back the
            # thinking rather than nothing.
            if not (msg.get("content") or "").strip():
                if not asked_again:
                    asked_again = True
                    log.info("agentloop: empty answer on step %d; asking again", step)
                    messages.append({"role": "user", "content":
                                     "That message was empty. Answer the question now, in prose, "
                                     "from what you already have."})
                    deadline = 0.0
                    continue
                text = (msg.get("reasoning_content") or "").strip() \
                    or "(the model returned an empty answer)"
                if emit is not None:
                    await emit({"content": text})
                last.setdefault("choices", [{}])[0]["message"] = {
                    "role": "assistant", "content": text}
                log.warning("agentloop: still empty; returned reasoning/placeholder")
            return last, messages

        # finish() ends the run. Its fields become the answer, so a caller
        # whose prompt asks for a structured report gets one.
        empty_finish = False
        for call in calls:
            if ((call.get("function") or {}).get("name")) == "finish":
                report = _render_finish(_args_of(call))
                if report == NOTHING_SAID:
                    # finish() with nothing in it is not an answer, and handing
                    # the caller "(no summary given)" is worse than useless — a
                    # chat client asked a question, the model ran kubectl to find
                    # out, and then ended its turn with an empty report. So treat
                    # it as a turn that has NOT finished: say so, withdraw the
                    # tools, and make it answer from what it already gathered.
                    log.info("agentloop: finish() carried no summary; forcing an answer")
                    messages.append({"role": "tool", "tool_call_id": call.get("id") or "",
                                     "content": "finish() needs a summary. Answer the question "
                                                "directly, in prose, from what you already have."})
                    empty_finish = True
                    break
                assistant["content"] = report
                assistant.pop("tool_calls", None)
                if emit is not None:
                    await emit({"content": report})
                last.setdefault("choices", [{}])[0]["message"] = dict(assistant)
                last["choices"][0]["finish_reason"] = "stop"
                return last, messages
        if empty_finish:
            deadline = 0.0          # spent: the next step is tool-free by definition
            continue

        for call in calls:
            name = (call.get("function") or {}).get("name") or ""
            if id(call) in broken:
                cut = broken[id(call)]
                out = ("ERROR: this call to %s was cut off before its arguments were complete "
                       "(received: %s), so it was NOT run. Make the call again with complete "
                       "arguments, and keep your thinking before a tool call short: the reply "
                       "has a length limit and the call comes last."
                       % (name, json.dumps(cut[:200])))
                log.info("tool %s NOT run: cut-off arguments", name)
                messages.append({"role": "tool", "tool_call_id": call.get("id") or "",
                                 "content": out})
                continue
            # load_capability is a gateway meta-tool (deferred loading), never
            # dispatched to the endpoint: it adds a pack's tools to `loaded`, so
            # the NEXT step's offered() carries them, and returns the runbook.
            if name == "load_capability":
                if emit is not None:
                    await emit({"reasoning_content": _describe(name, _args_of(call))})
                out = await tools.apply_load(_args_of(call).get("name"), loaded)
                log.info("load_capability(%s) -> %s", _args_of(call).get("name"),
                         "error" if out.startswith("ERROR") else "ok")
                messages.append({"role": "tool", "tool_call_id": call.get("id") or "",
                                 "content": out})
                continue
            # In deferred mode only what has been loaded is offered, so a call to
            # anything else means the model reached for an unloaded capability.
            # Tell it to load first rather than dispatching (legacy mode offers
            # everything, so offered_names holds it and this never fires).
            if packs.enabled() and name not in offered_names:
                out = ("ERROR: '%s' is not available — it belongs to a capability you have not "
                       "loaded. Call load_capability(...) for the capability that provides it "
                       "first; see the capability list in your instructions." % name)
                log.info("tool %s blocked: capability not loaded", name)
                messages.append({"role": "tool", "tool_call_id": call.get("id") or "",
                                 "content": out})
                continue
            if emit is not None:
                await emit({"reasoning_content": _describe(name, _args_of(call))})
            out = await tools.dispatch(name, _args_of(call), emit)
            log.info("tool %s -> %d chars", name, len(out))
            messages.append({"role": "tool", "tool_call_id": call.get("id") or "",
                             "content": out})

        # Tool results are the thing that overflows the window.
        try:
            shrunk, info = await compactor.maybe_compact(
                {"messages": messages}, auth)
            if shrunk is not None:
                messages = shrunk["messages"]
                log.info("agentloop: compacted mid-run (%s -> %s tokens)",
                         info.get("tokens_before"), info.get("tokens_after", "?"))
        except Exception:
            log.exception("agentloop: mid-run compaction failed; continuing")

    return last or {}, messages

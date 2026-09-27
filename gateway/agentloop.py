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
import hashlib
import itertools
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


async def _noop(_delta):
    return None


async def _post(client, url: str, body: dict, headers: dict, budget, emit):
    """One model call. Returns (status, response dict or None, error text).

    Always streamed from upstream, whether or not the caller streams: a stream
    is what makes a hang detectable. The read timeout is per chunk, so
    STEP_IDLE_TIMEOUT_S bounds silence, not work — a step may think for as
    long as it likes as long as tokens keep coming. `budget`, when not None,
    additionally bounds the whole call (only when a deadline is configured).
    Every reasoning/content delta goes to `emit` as it arrives, then the step
    is reassembled into the dict shape a non-streamed call returns. Tool call
    fragments are NOT forwarded: the client never asked for tools.
    """
    emit = emit or _noop
    body = dict(body, stream=True)
    timeout = httpx.Timeout(config.STEP_IDLE_TIMEOUT_S or None,
                            connect=config.CONNECT_TIMEOUT_S)
    acc = ChatAccumulator()
    tail: dict = {}

    async def consume():
        async with client.stream("POST", url, json=body, headers=headers,
                                 timeout=timeout) as r:
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

    # The read timeout above is per chunk. A configured deadline bounds the
    # whole call as well; with none, only silence ends a step.
    try:
        if budget is None:
            status, err = await consume()
        else:
            status, err = await asyncio.wait_for(consume(), timeout=budget)
    except asyncio.TimeoutError as e:
        raise httpx.ReadTimeout(f"model step exceeded {budget:.0f}s") from e
    except httpx.ReadTimeout as e:
        raise httpx.ReadTimeout(
            f"model server silent for {config.STEP_IDLE_TIMEOUT_S:.0f}s mid-step "
            "(hung slot or dead connection)") from e
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

# Why the tools were withdrawn, told to the model once.
BUDGET_SPENT = ("Your tool budget is spent. Answer now from what you already "
                "have, and say plainly what you could not determine.")
NO_PROGRESS = ("Your last several steps produced nothing new: the same calls, the same "
               "results. Stop and report now: what you found, what you changed, what is "
               "still open and what you would need to go further.")
FORCED = "forced"            # tools withdrawn, reason already given


def _fingerprint(name: str, args: dict, out: str) -> str:
    raw = json.dumps([name, args, out], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


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
    # Deadlines exist only when configured (0 = none, the default).
    deadline = started + config.TOOL_MAX_SECONDS if config.TOOL_MAX_SECONDS else None
    hard_deadline = started + config.REQUEST_MAX_SECONDS if config.REQUEST_MAX_SECONDS else None
    asked_again = False          # one nudge, for a turn that came back empty
    length_cuts = 0              # steps cut off by max_tokens before acting
    loaded: set = set()          # capabilities loaded so far (deferred mode; packs.py)
    # Why the tools are withdrawn, when they are; None while the model may act.
    force: str | None = None
    # Loop detection: every (call, result) pair seen this run, and how many
    # steps in a row produced nothing new.
    seen_results: set = set()
    stale_steps = 0
    # Learn the window up front, so the first tool result is capped against it
    # rather than against the fallback.
    try:
        await compactor.n_ctx(auth)
    except Exception:
        pass

    for step in itertools.count():
        if config.TOOL_MAX_STEPS and step >= config.TOOL_MAX_STEPS:
            break
        # Out of time means either configured budget: the tool budget, or so
        # little left against the hard deadline that another tool step could
        # only be given a stub of a timeout and die in the middle.
        now = time.monotonic()
        out_of_time = ((deadline is not None and now >= deadline)
                       or (hard_deadline is not None
                           and hard_deadline - now < config.UPSTREAM_MIN_TIMEOUT_S))
        last_step = bool(config.TOOL_MAX_STEPS) and step >= config.TOOL_MAX_STEPS - 1
        if force is None and (out_of_time or last_step):
            force = BUDGET_SPENT
            if step:
                log.info("agentloop: budget spent after %d step(s); forcing an answer", step)
        offer_tools = force is None
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
            if force in (BUDGET_SPENT, NO_PROGRESS):
                messages = messages + [{"role": "user", "content": force}]
                force = FORCED          # said once; later steps stay tool-free
            call_body["messages"] = messages

        # A step is bounded by silence (STEP_IDLE_TIMEOUT_S, in _post), never by
        # how long it works. Only a configured deadline adds a wall clock. The
        # forced answer gets its OWN budget rather than the remainder, because
        # the remainder is smallest exactly when the answer matters most.
        if not offer_tools and config.ANSWER_TIMEOUT_S:
            budget = config.ANSWER_TIMEOUT_S
        elif hard_deadline is not None:
            budget = max(config.UPSTREAM_MIN_TIMEOUT_S, hard_deadline - time.monotonic())
        else:
            budget = None
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

        # A step that ran into max_tokens before calling anything is not an
        # answer and not a reason to stop. 2026-09-26: the agent thought for a
        # whole 8k-token step about how to rebuild a truncated file, got cut off
        # with no content, fell into the empty-answer path below, lost its
        # tools for good, and reported "budget ran out" 5 calls into an 80-call
        # budget. Keep the tools and tell it to act.
        finish_reason = ((last.get("choices") or [{}])[0].get("finish_reason")) or ""
        if (not calls and finish_reason == "length" and offer_tools
                and length_cuts < config.LENGTH_CUT_RETRIES):
            length_cuts += 1
            log.info("agentloop: step %d cut off by max_tokens before acting (%d/%d); "
                     "continuing with tools", step, length_cuts, config.LENGTH_CUT_RETRIES)
            messages.append({"role": "user", "content":
                             "Your reply hit the per-step length limit and was cut off before "
                             "you acted, so nothing ran. Continue from where you were, without "
                             "re-planning: make the next tool call now and keep the thinking "
                             "before it short, or give your answer if you already have it."})
            continue

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
                    force = FORCED
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
            force = FORCED          # the next step is tool-free by definition
            continue

        progressed = False
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
                key = _fingerprint(name, _args_of(call), out)
                if key not in seen_results:
                    seen_results.add(key)
                    progressed = True
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
            args = _args_of(call)
            out = await tools.dispatch(name, args, emit)
            log.info("tool %s -> %d chars", name, len(out))
            key = _fingerprint(name, args, out)
            if key in seen_results:
                out += ("\n[This exact call already returned exactly this earlier in this "
                        "run. Use that result; do something different, or finish.]")
            else:
                seen_results.add(key)
                progressed = True
            messages.append({"role": "tool", "tool_call_id": call.get("id") or "",
                             "content": out})

        # Loop detection. Refusals, cut-off calls and capability loads count as
        # no progress: a model that keeps making them is stuck all the same.
        stale_steps = 0 if progressed else stale_steps + 1
        if config.NO_PROGRESS_STEPS and stale_steps >= config.NO_PROGRESS_STEPS and force is None:
            log.warning("agentloop: %d steps with no new tool results; forcing a report",
                        stale_steps)
            force = NO_PROGRESS

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

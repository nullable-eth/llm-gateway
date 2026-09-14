"""The agent loop: call the model, run what it asks for, repeat, answer.

The client sends one request and gets one reply. Everything between — the
tool calls, their output, the model's second and third thoughts — happens
here and is invisible to the caller, which is what lets a chat client with no
tool support ask questions that need kubectl.

It is invisible to the caller but NOT to the archive: the loop hands capture
the whole message list it built, so the transcript holds every tool call and
result even though the client only ever saw the final answer. That is the
single most useful thing this design gets for free.

Compaction runs *between steps*, not only on the way in. Tool output is what
actually overflows a window — a few `kubectl get events` dumps will do it —
and this is the same job cluster-agent's trim_old_results() did before the
gateway took it over.
"""
import json
import logging
import time

from . import config, tools

log = logging.getLogger("gateway")


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
    for label, key in (("Actions taken", "actions_taken"), ("Proposals", "proposals")):
        items = args.get(key) or []
        if isinstance(items, str):
            items = [items]
        if items:
            out.append(f"**{label}**\n" + "\n".join(f"- {i}" for i in items))
    return "\n\n".join(x for x in out if x) or NOTHING_SAID


async def run(client, upstream: str, body: dict, auth: str, compactor) -> tuple[dict, list]:
    """Returns (final upstream response, full message list including tool traffic)."""
    messages = list(body.get("messages") or [])
    headers = {"Authorization": auth} if auth else {}
    last = None
    started = time.monotonic()
    deadline = started + config.TOOL_MAX_SECONDS
    hard_deadline = started + config.REQUEST_MAX_SECONDS

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
        if offer_tools:
            call_body["tools"] = tools.TOOLS
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
        r = await client.post(f"{upstream}/v1/chat/completions", json=call_body,
                              headers=headers, timeout=budget)
        r.raise_for_status()
        last = r.json()
        msg = ((last.get("choices") or [{}])[0].get("message") or {})
        calls = msg.get("tool_calls") or []

        assistant = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("reasoning_content"):
            assistant["reasoning_content"] = msg["reasoning_content"]
        if calls:
            assistant["tool_calls"] = calls
        messages.append(assistant)

        if not calls:
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
                last.setdefault("choices", [{}])[0]["message"] = dict(assistant)
                last["choices"][0]["finish_reason"] = "stop"
                return last, messages
        if empty_finish:
            deadline = 0.0          # spent: the next step is tool-free by definition
            continue

        for call in calls:
            name = (call.get("function") or {}).get("name") or ""
            out = await tools.dispatch(name, _args_of(call))
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

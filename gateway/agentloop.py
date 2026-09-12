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


def _render_finish(args: dict) -> str:
    out = [str(args.get("summary") or "").strip()]
    for label, key in (("Actions taken", "actions_taken"), ("Proposals", "proposals")):
        items = args.get(key) or []
        if isinstance(items, str):
            items = [items]
        if items:
            out.append(f"**{label}**\n" + "\n".join(f"- {i}" for i in items))
    return "\n\n".join(x for x in out if x) or "(no summary given)"


async def run(client, upstream: str, body: dict, auth: str, compactor) -> tuple[dict, list]:
    """Returns (final upstream response, full message list including tool traffic)."""
    messages = list(body.get("messages") or [])
    headers = {"Authorization": auth} if auth else {}
    last = None
    deadline = time.monotonic() + config.TOOL_MAX_SECONDS

    for step in range(config.TOOL_MAX_STEPS):
        # Tools are withdrawn on the last step, or once the wall-clock budget
        # is spent, so the model has to answer rather than start work nobody
        # will wait for.
        out_of_time = time.monotonic() >= deadline
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

        r = await client.post(f"{upstream}/v1/chat/completions", json=call_body,
                              headers=headers, timeout=None)
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
        for call in calls:
            if ((call.get("function") or {}).get("name")) == "finish":
                report = _render_finish(_args_of(call))
                assistant["content"] = report
                assistant.pop("tool_calls", None)
                last.setdefault("choices", [{}])[0]["message"] = dict(assistant)
                last["choices"][0]["finish_reason"] = "stop"
                return last, messages

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

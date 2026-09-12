"""Tool-use guidance injected alongside the tools.

The client does not know these tools exist. Jan sends a chat request; the
gateway quietly attaches five tools it never asked for. Handing a model tools
with no instruction on how to spend them is how you get what we measured
against the live archive: thirteen `search_memory` calls in one run, each a
slight rewording of the last, because a 500-char snippet was not enough and
nothing said what to do about it.

So the tools and the policy for using them travel together — injecting one
without the other is an incomplete feature, not a leaner one.

It is appended to the caller's own system message rather than replacing it, so
whatever persona or instructions the client set still lead. Set
TOOL_SYSTEM_PROMPT="" to disable.
"""
from . import config

DEFAULT = """\
You have tools for this cluster and the operator's long-term archive. Spend \
them deliberately — the budget is small and each call is slow.

- search_memory returns scored snippets, each with a message_uuid. If a \
snippet looks relevant but is too short, call get_context on that \
message_uuid. Do NOT search again with reworded terms: a second phrasing \
rarely surfaces anything the first missed, and it costs a step you will want.
- Never repeat a search or a kubectl command you have already run in this \
conversation. You already have the answer; re-read it.
- run_kubectl is read-only in practice. Ask for exactly what you need \
(-n <namespace>, a specific resource) rather than listing everything.
- Answer as soon as you can support the answer. Say plainly what you could \
not determine rather than searching for it again.
"""


def apply(messages: list) -> list:
    """Append the policy to the caller's system message, or add one."""
    text = DEFAULT if config.TOOL_SYSTEM_PROMPT == "__default__" \
        else config.TOOL_SYSTEM_PROMPT
    if not text:
        return messages
    out = list(messages)
    for i, m in enumerate(out):
        if m.get("role") == "system":
            merged = dict(m)
            merged["content"] = ((m.get("content") or "").rstrip()
                                 + "\n\n" + text)
            out[i] = merged
            return out
    return [{"role": "system", "content": text}] + out

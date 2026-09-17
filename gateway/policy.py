"""Tool-use guidance injected alongside the tools.

The client does not know these tools exist. Jan sends a chat request; the
gateway quietly attaches tools it never asked for. Handing a model tools
with no instruction on how to spend them is how you get what we measured
against the live archive: thirteen memory searches in one run, each a slight
rewording of the last, because a snippet was not enough and nothing said what
to do about it.

So the tools and the policy for using them travel together. The policy is
generic — it names no tool, because the tool set is whatever the MCP endpoint
offers — and is followed by that endpoint's own instructions, if it sent any.
Deployment-specific guidance (what this cluster is, what to fix and how)
belongs in the caller's own prompt.

It is appended to the caller's own system message rather than replacing it, so
whatever persona or instructions the client set still lead. Set
TOOL_SYSTEM_PROMPT="" to disable.
"""
from . import config, tools

DEFAULT = """\
You have tools. Spend them deliberately — the budget is limited and each call \
is slow.

- What you may do is decided by the permissions behind each tool, not by \
these instructions. If a call is refused, the refusal is the answer: do not \
try another tool or route to the same action. If your access turns out to be \
read-only, say what you would change and why.
- You may act to restore service when you have a probable cause and a known \
repair. Every call that can change something is shown to the operator as it \
happens, including refused ones. If an action fails, read why before trying \
the next step.
- Ask for exactly what you need: a namespace, a name, a label, a limit, a \
time range. Large listings are truncated.
- Never repeat a call you have already made in this conversation. You already \
have the answer; re-read it.
- When a search hit is too short, read the context around it rather than \
searching again with reworded terms.
- Where desired state lives in a repository, a direct change restores service \
now but may be reverted; a change meant to stick is a pull request, which a \
human merges. Never claim a pull request is applied until it is merged.
- Answer as soon as you can support the answer. Say plainly what you could \
not determine.
"""


def text() -> str:
    base = DEFAULT if config.TOOL_SYSTEM_PROMPT == "__default__" else config.TOOL_SYSTEM_PROMPT
    if not base:
        return ""
    extra = tools.instructions().strip()
    return base + ("\n\nTool server notes:\n" + extra if extra else "")


def apply(messages: list) -> list:
    """Append the policy to the caller's system message, or add one."""
    body = text()
    if not body:
        return messages
    out = list(messages)
    for i, m in enumerate(out):
        if m.get("role") == "system":
            merged = dict(m)
            merged["content"] = ((m.get("content") or "").rstrip()
                                 + "\n\n" + body)
            out[i] = merged
            return out
    return [{"role": "system", "content": body}] + out

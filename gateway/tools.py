"""The tools the loop offers: whatever the MCP endpoint lists, plus finish().

The gateway holds no credentials and no permission list of its own. Every
capability arrives from the configured MCP endpoint (mcpclient.py), and what a
call may actually do is enforced where it runs: the MCP server's credentials,
Kubernetes RBAC and admission policy, the MCP gateway's authorization rules. A
refusal from any of those comes back as the tool's result and the model reads
it like any other output.

What stays here is reporting: every call is streamed to the caller before and
after as a `tool_event`, and every call that may change something is logged.
"""
import json
import logging

from . import config, mcpclient

log = logging.getLogger("gateway")

# Not dispatched — agentloop intercepts it and ends the run. It gives an agent
# an explicit way to say "I am done", and a caller whose prompt asks for a
# structured report gets the fields it asked for rather than prose.
FINISH = {"type": "function", "function": {"name": "finish",
    "description": "End the run with a report. Call this exactly once when diagnosis or action is complete.",
    "parameters": {"type": "object", "properties": {
        "summary": {"type": "string", "description": "what happened and root cause"},
        "actions_taken": {"type": "array", "items": {"type": "string"}},
        "proposals": {"type": "array", "items": {"type": "string"},
            "description": "changes a human should make or approve, exactly"},
        "capability_gaps": {"type": "array", "items": {"type": "string"},
            "description": "anything you could not do here that a tool, permission or "
                           "documented procedure would have let you do, one per item: what you "
                           "needed, what stopped you (the exact refusal or missing tool), and "
                           "the smallest thing that would fix it. Empty if nothing was missing"}},
        "required": ["summary"]}}}


async def offered() -> list:
    """The tools handed to the model on this call."""
    client = mcpclient.get()
    if client is None:
        return [FINISH]
    return [mcpclient.to_openai(t) for t in await client.list_tools()] + [FINISH]


def instructions() -> str:
    """The MCP endpoint's own usage instructions, if it sent any."""
    client = mcpclient.get()
    if client is None or not client.instructions:
        return ""
    return client.instructions[:config.MCP_INSTRUCTIONS_MAX]


def is_mutation(name: str, args: dict) -> bool:
    """May this call change something outside this process?

    Taken from the tool's own MCP annotations. A tool that does not declare
    itself read-only is reported as an action: an unannotated tool is shown to
    the operator rather than silently exempt.
    """
    client = mcpclient.get()
    ann = client.annotations(name) if client else {}
    if "readOnlyHint" in ann:
        return ann["readOnlyHint"] is not True
    # For servers that do not annotate: names the operator vouches for.
    return not (config.TOOL_READ_ONLY_RE and config.TOOL_READ_ONLY_RE.search(name))


async def dispatch(name: str, args: dict, emit=None) -> str:
    """Run a tool, reporting it before and after as a structured event.

    Before, because a run that dies mid-change must still leave a record of
    what it started; after, because "I ran this" and "this is what happened"
    are different facts. Refusals are reported too.

    The events go to `emit`, i.e. onto the caller's response stream as a
    `tool_event` delta. Where they are shown is the caller's business.
    """
    mutating = is_mutation(name, args)
    if mutating:
        log.info("ACTION %s %s", name, json.dumps(args)[:400])
    if emit is not None:
        await emit({"tool_event": {"phase": "call", "name": name, "args": args,
                                   "mutating": mutating}})
    out, failed = await _dispatch(name, args)
    head = out.strip().splitlines()[0] if out.strip() else "(no output)"
    if mutating or failed:
        log.info("RESULT %s%s -> %s", name, " (error)" if failed else "", head[:400])
    if emit is not None:
        await emit({"tool_event": {"phase": "result", "name": name, "mutating": mutating,
                                   "error": failed, "summary": head[:400],
                                   "output": out[:2000]}})
    return out


async def _dispatch(name: str, args: dict) -> tuple[str, bool]:
    client = mcpclient.get()
    if client is None:
        return f"ERROR: no tool '{name}' (no tool endpoint is configured)", True
    known = {t.get("name") for t in await client.list_tools()}
    if name not in known:
        return f"ERROR: unknown tool '{name}'", True
    try:
        text, is_error = await client.call_tool(name, args if isinstance(args, dict) else {})
    except Exception as e:
        # A policy refusal arrives here too (a JSON-RPC error from the MCP
        # gateway). It is an answer, not a crash: hand it to the model.
        return f"ERROR: {e}", True
    if len(text) > config.TOOL_OUTPUT_MAX:
        text = (text[:config.TOOL_OUTPUT_MAX]
                + f"\n[... truncated {len(text) - config.TOOL_OUTPUT_MAX} chars; "
                  "ask for less: a namespace, a name, a limit, a time range]")
    return (f"ERROR: {text}" if is_error else text) or "(no output)", is_error

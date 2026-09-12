"""Tool link — the gateway runs the loop, the client sees one answer.

Ported from cluster-agent, which owned this until the gateway existed. Moving
it here is what lets any OpenAI client — Jan, a script, cluster-agent — ask
questions about the cluster without implementing tool calling itself: the
gateway executes the tools and returns a finished reply. cluster-agent stops
being an agent and becomes what it always should have been, something that
receives an alert and asks a question.

THE GUARDS ARE THE POINT, and they came across unchanged. Read verbs pass;
everything in DENY is refused outright (secrets, exec, port-forward, and every
direct-write verb, because writes belong to GitOps); anything matching a
PROTECTED component is refused for non-read verbs, which is the rule that
stops the agent restarting its own brain; mutations need MODE=auto and are
otherwise recorded as proposals. The RBAC underneath is read-only and
enumerates resources so `secrets` is structurally absent rather than merely
denied.

Worth stating plainly: with tools always on, anything holding the inference
key can now reach kubectl through this gateway, where before that key only
bought inference. The guards bound what it can do, not who can try. A separate
tool credential is the intended next step — the blocker is that chat clients
(Jan among them) expose a fixed api-key field and no way to send an extra
header, so the credential would have to ride the model name or a second port.
"""
import asyncio
import json
import logging
import subprocess

import httpx

from . import config

log = logging.getLogger("gateway")

READ_VERBS = {"get", "describe", "logs", "top", "api-resources", "api-versions",
              "explain", "version", "cluster-info", "rollout"}
DENY = {"secret", "secrets", "exec", "attach", "cp", "port-forward", "proxy",
        "edit", "apply", "create", "patch", "replace", "label", "annotate",
        "cordon", "drain", "taint", "auth", "--token", "--kubeconfig"}
AUTO_OK = {("delete", "pod"), ("delete", "pods"), ("rollout", "restart")}


def kubectl_guard(args: list[str]) -> str | None:
    """Return a rejection reason, or None if the command may run."""
    if not args:
        return "empty command"
    low = [a.lower() for a in args]
    for tok in low:
        base = tok.split("=")[0]
        if base in DENY or tok in DENY:
            return f"'{tok}' is never permitted (read RBAC + GitOps: no direct writes)"
    for name in config.PROTECTED:
        if any(name in t for t in low[1:]):
            if low[0] not in READ_VERBS or low[0:2] == ["rollout", "restart"]:
                return f"target matches protected component '{name}' — self-preservation rule"
    verb = low[0]
    if verb in READ_VERBS and low[0:2] != ["rollout", "restart"]:
        return None
    pair = (verb, low[1] if len(low) > 1 else "")
    if pair in AUTO_OK or (verb, "restart") == ("rollout", "restart"):
        if config.MODE != "auto":
            return "propose mode: mutation recorded as a proposal, not executed"
        return None
    return f"verb '{verb}' is outside the action allowlist"


def run_kubectl(args: list[str]) -> str:
    why = kubectl_guard(args)
    if why:
        return f"REFUSED: {why}"
    try:
        r = subprocess.run(["kubectl", *args], capture_output=True, text=True,
                           timeout=config.KUBECTL_TIMEOUT_S)
        out = (r.stdout + ("\n" + r.stderr if r.stderr else "")).strip()
        return out[:config.TOOL_OUTPUT_MAX] or f"(exit {r.returncode}, no output)"
    except subprocess.TimeoutExpired:
        return f"REFUSED: command timed out after {config.KUBECTL_TIMEOUT_S}s"
    except FileNotFoundError:
        return "REFUSED: kubectl is not installed in this image"


async def ha_get_states(entity_id: str = "") -> str:
    if not config.HA_URL:
        return "Home Assistant is not configured (HA_URL unset)"
    url = f"{config.HA_URL}/api/states" + (f"/{entity_id}" if entity_id else "")
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(url, headers={"Authorization": f"Bearer {config.HA_TOKEN}"})
    return r.text[:config.TOOL_OUTPUT_MAX]


async def ha_call_service(domain: str, service: str, entity_id: str = "",
                          data: dict | None = None) -> str:
    if not config.HA_URL:
        return "Home Assistant is not configured (HA_URL unset)"
    if config.MODE != "auto":
        return f"PROPOSAL RECORDED (propose mode): ha {domain}.{service} on {entity_id or data}"
    body = dict(data or {})
    if entity_id:
        body["entity_id"] = entity_id
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post(f"{config.HA_URL}/api/services/{domain}/{service}",
                         headers={"Authorization": f"Bearer {config.HA_TOKEN}"},
                         json=body)
    return f"HTTP {r.status_code}: {r.text[:2000]}"


async def get_context(message_uuid: str, radius: int = 3) -> str:
    """Read the conversation around a search hit.

    Without this the model only ever sees 500-char snippets, and when a snippet
    is not enough it does the only thing it can — searches again, with slightly
    different words, until it runs out of steps. Observed doing exactly that
    six times in one run before this existed.
    """
    if not config.MEMORY_URL:
        return "memory service is not configured (MEMORY_URL unset)"
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get(f"{config.MEMORY_URL}/context/{message_uuid}",
                            params={"radius": radius},
                            headers={"Authorization": f"Bearer {config.MEMORY_TOKEN}"})
        if r.status_code == 404:
            return (f"no archived message with uuid {message_uuid} — use the "
                    f"message_uuid from a search_memory hit, verbatim")
        return r.text[:config.TOOL_OUTPUT_MAX]
    except Exception as exc:
        return f"memory unavailable: {exc}"


async def search_memory(query: str, k: int = 6) -> str:
    if not config.MEMORY_URL:
        return "memory service is not configured (MEMORY_URL unset)"
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get(f"{config.MEMORY_URL}/search",
                            params={"q": query, "k": k},
                            headers={"Authorization": f"Bearer {config.MEMORY_TOKEN}"})
        return r.text[:config.TOOL_OUTPUT_MAX]
    except Exception as exc:          # optional context, never fatal
        return f"memory unavailable: {exc}"


TOOLS = [
    {"type": "function", "function": {"name": "run_kubectl",
        "description": "Run a guarded kubectl command against this cluster. Read verbs always allowed; mutations only per policy.",
        "parameters": {"type": "object", "properties": {"args": {"type": "array", "items": {"type": "string"},
            "description": "argv after 'kubectl', e.g. ['get','pods','-n','media']"}}, "required": ["args"]}}},
    {"type": "function", "function": {"name": "ha_get_states",
        "description": "Home Assistant: read entity state(s). Empty entity_id lists all.",
        "parameters": {"type": "object", "properties": {"entity_id": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "ha_call_service",
        "description": "Home Assistant: call a service (e.g. switch.turn_off). Executes only in auto mode; otherwise recorded as a proposal.",
        "parameters": {"type": "object", "properties": {"domain": {"type": "string"}, "service": {"type": "string"},
            "entity_id": {"type": "string"}, "data": {"type": "object"}}, "required": ["domain", "service"]}}},
    # Not dispatched — agentloop intercepts it and ends the run. It exists so
    # an agent has an explicit way to say "I am done" instead of trailing off,
    # and so a caller whose prompt asks for a structured report still gets the
    # fields it asked for rather than prose.
    {"type": "function", "function": {"name": "finish",
        "description": "End the run with a report. Call this exactly once when diagnosis or action is complete.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description": "what happened and root cause"},
            "actions_taken": {"type": "array", "items": {"type": "string"}},
            "proposals": {"type": "array", "items": {"type": "string"},
                "description": "mutations a human should run or approve, exact commands"}},
            "required": ["summary"]}}},
    {"type": "function", "function": {"name": "search_memory",
        "description": "Search the operator's long-term conversation archive for prior incidents, decisions and known fixes. Returns scored snippets, each with a message_uuid. When a snippet is not enough, call get_context on its message_uuid rather than searching again.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
            "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_context",
        "description": "Read the archived conversation surrounding a search_memory hit, verbatim. Use the message_uuid from a hit; radius is how many messages either side.",
        "parameters": {"type": "object", "properties": {"message_uuid": {"type": "string"}, "radius": {"type": "integer"}},
            "required": ["message_uuid"]}}},
]


async def dispatch(name: str, args: dict) -> str:
    if name == "run_kubectl":
        return await asyncio.to_thread(run_kubectl, args.get("args", []))
    if name == "ha_get_states":
        return await ha_get_states(args.get("entity_id", ""))
    if name == "ha_call_service":
        return await ha_call_service(args.get("domain", ""), args.get("service", ""),
                                     args.get("entity_id", ""), args.get("data"))
    if name == "search_memory":
        return await search_memory(args.get("query", ""), int(args.get("k", 6) or 6))
    if name == "get_context":
        return await get_context(args.get("message_uuid", ""),
                                 int(args.get("radius", 3) or 3))
    return f"REFUSED: unknown tool '{name}'"

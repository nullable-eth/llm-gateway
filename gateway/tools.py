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
from datetime import datetime, timedelta, timezone
import subprocess

import httpx

from . import config, gittools, observe

log = logging.getLogger("gateway")

READ_VERBS = {"get", "describe", "logs", "top", "api-resources", "api-versions",
              "explain", "version", "cluster-info", "rollout", "events", "diff",
              "auth"}  # auth can-i only; see FORBIDDEN_SUBS

# There is no verb allowlist any more. The agent may take any action kubectl can
# take, because an SRE that can only delete pods and restart deployments cannot
# actually restore service -- it can only describe what someone else must do.
# What remains is four structural refusals, and they are not about danger:
#
#   1. Nothing that reads credentials. `secrets` is absent from the RBAC as well,
#      so this is belt and braces rather than the only guard.
#   2. Nothing that opens a shell or a tunnel into a workload -- exec, attach,
#      cp, port-forward, proxy, debug. These turn "the agent can act" into "the
#      agent can be made to do anything the workload can do", which is a much
#      larger claim than anything a prompt should be able to make.
#   3. No identity games: --as, --token, --kubeconfig. The agent acts as itself
#      or not at all, so the audit trail means something.
#   4. No RBAC edits. Permissions are a GitOps decision, not a runtime one.
#
# Everything else -- patch, apply, scale, cordon, drain, taint, delete, annotate
# -- is allowed in auto mode and recorded as a proposal otherwise. Every one of
# them is reported before and after as a tool event on the response stream
# (see dispatch), and the caller shows it: that visibility is the actual control.
FORBIDDEN_VERBS = {"exec", "attach", "cp", "port-forward", "proxy", "debug", "alpha", "plugin",
                   "certificate", "config"}
FORBIDDEN_SUBS = {"auth": {"can-i", "whoami"}}    # only these subcommands of auth
FORBIDDEN_TOKENS = {"--token", "--kubeconfig", "--as", "--as-group", "--as-uid"}
FORBIDDEN_RESOURCES = {"secret", "secrets", "serviceaccount", "serviceaccounts",
                       "clusterrole", "clusterroles", "role", "roles",
                       "rolebinding", "rolebindings", "clusterrolebinding",
                       "clusterrolebindings", "csr", "certificatesigningrequest",
                       "certificatesigningrequests"}


# Global flags that may come before the verb, and which of them take a value
# as the next argument. Anything else before the verb is refused: kubectl
# accepts flags anywhere, and a guard that only looked at argv[0] let
# `-n media exec ...` through (2026-09-17; RBAC refused it underneath).
PRE_VERB_VALUE_FLAGS = {"-n", "--namespace", "--context", "--cluster", "--user",
                        "--request-timeout", "-v", "--v"}
PRE_VERB_BOOL_FLAGS = {"--insecure-skip-tls-verify", "--match-server-version",
                       "--warnings-as-errors", "--disable-compression"}


def split_verb(low: list[str]) -> tuple[str | None, list[str], str | None]:
    """(verb, args from the verb on, error). Flags before the verb are skipped
    only if they are known global flags; anything else is an error."""
    i = 0
    while i < len(low):
        tok = low[i]
        if not tok.startswith("-"):
            return tok, low[i:], None
        name = tok.split("=", 1)[0]
        if name in PRE_VERB_VALUE_FLAGS:
            i += 1 if "=" in tok else 2
        elif name in PRE_VERB_BOOL_FLAGS:
            i += 1
        else:
            return None, [], (f"'{tok}' before the command is not permitted; "
                              f"put flags after the verb (e.g. get pods -n media)")
    return None, [], "no kubectl command given"


def kubectl_guard(args: list[str]) -> str | None:
    """Return a rejection reason, or None if the command may run."""
    if not args:
        return "empty command"
    everything = [a.lower() for a in args]
    verb, low, err = split_verb(everything)
    if err:
        return err
    if verb in FORBIDDEN_VERBS:
        return (f"'{verb}' is never permitted: it opens a shell or tunnel into a "
                f"workload, which is a bigger claim than any alert justifies")
    if verb in FORBIDDEN_SUBS and (len(low) < 2 or low[1] not in FORBIDDEN_SUBS[verb]):
        return f"'{verb}' is only permitted as: {' | '.join(sorted(FORBIDDEN_SUBS[verb]))}"
    for tok in everything:
        if tok.split("=")[0] in FORBIDDEN_TOKENS:
            return f"'{tok}' is never permitted: the agent acts as itself or not at all"
    for tok in low[1:]:
        # "secrets", "secret/foo", "secrets.v1." -- any spelling that names one.
        if tok.split("/")[0].split(".")[0] in FORBIDDEN_RESOURCES:
            return (f"'{tok}' is never permitted: credentials and permissions are "
                    f"not runtime state (and are absent from this RBAC anyway)")
    is_read = verb in READ_VERBS and low[0:2] != ["rollout", "restart"]
    for name in config.PROTECTED:
        if any(name in t for t in everything) and not is_read:
            return f"target matches protected component '{name}' — self-preservation rule"
    if is_read:
        return None
    if config.MODE != "auto":
        return "propose mode: mutation recorded as a proposal, not executed"
    return None


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


async def search_memory(query: str, k: int = 0, sender: str = "") -> str:
    """Search the archive, and return MANY shallow hits rather than few deep ones.

    Measured against the live archive on 14 Sep, with the operator's own "the
    bond failure is because I moved an ethernet cable ... ignore this for now"
    as the target. At k=8 -- the old default -- the agent's own phrasing put that
    line at rank 12 and rank 15: present, retrievable, and four places past where
    anyone looked. It reported "I searched and found none" and re-diagnosed a
    known-accepted state for the second day running.

    So: a much larger k, and each hit rendered compactly instead of raw JSON
    truncated at TOOL_OUTPUT_MAX. Raw JSON was the reason a big k could not help
    -- 25 hits of it overflow the budget and get cut off exactly where the late
    ranks live. One line per hit fits ~40 of them, and get_context is there for
    the two that matter.

    Only 4 of 50 hits on that query were the operator speaking; the rest were
    the assistant, including the agent's own past reports on this same alert --
    hence `sender`, which asks agentmemory for one speaker's messages only. That
    is how you ask for a DECISION rather than a discussion about one.
    """
    if not config.MEMORY_URL:
        return "memory service is not configured (MEMORY_URL unset)"
    k = k or config.MEMORY_SEARCH_K
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            params = {"q": query, "k": min(k, config.MEMORY_SEARCH_MAX_K)}
            if sender:
                params["sender"] = sender
            r = await c.get(f"{config.MEMORY_URL}/search", params=params,
                            headers={"Authorization": f"Bearer {config.MEMORY_TOKEN}"})
        hits = (r.json() or {}).get("hits")
        if hits is None:
            return r.text[:config.TOOL_OUTPUT_MAX]
        if not hits:
            return f"no hits for {query!r}"
        lines = [f"{len(hits)} hit(s), best first. Call get_context on a "
                 f"message_uuid to read around one."]
        for h in hits:
            who = h.get("sender") or "?"
            # 180, not 240: measured, 25 hits at 240 came to 7937 chars against
            # a TOOL_OUTPUT_MAX of 8000, and being truncated at the cap loses
            # the tail of the ranking — which is the exact failure this is
            # fixing. Headroom is the point.
            snip = " ".join((h.get("snippet") or "").split())[:180]
            lines.append(f"- [{who}] {h.get('date') or ''} {h.get('title') or ''} "
                         f"(uuid {h.get('message_uuid')}): {snip}")
        return "\n".join(lines)[:config.TOOL_OUTPUT_MAX]
    except Exception as exc:          # optional context, never fatal
        return f"memory unavailable: {exc}"


async def silence_alert(alertname: str, comment: str, hours: int = 168,
                        matchers: dict | None = None) -> str:
    """Stop an alert notifying, when the operator has already said to ignore it.

    Without this the only thing the agent could do with "ignore the bond, I
    moved the cable" was write it in a report and then investigate the same
    alert again six hours later, forever. A silence is the one mutation that
    makes a known-accepted state stop costing attention.

    Bounded on purpose: an alertname is required (a silence with only a node or
    namespace matcher swallows unrelated incidents), a comment is required and
    should say who decided and where that is recorded, and the duration is
    capped — a silence that outlives the reason for it is how an outage goes
    unnoticed.
    """
    if not config.ALERTMANAGER_URL:
        return "alertmanager is not configured (ALERTMANAGER_URL unset)"
    if not alertname.strip():
        return "REFUSED: alertname is required — a silence without one is too broad"
    if len(comment.strip()) < 15:
        return ("REFUSED: comment must say why this is being silenced and where "
                "the operator said so (cite the search_memory hit)")
    hours = max(1, min(int(hours or 168), config.SILENCE_MAX_HOURS))
    spec = [{"name": "alertname", "value": alertname, "isRegex": False, "isEqual": True}]
    for k, v in (matchers or {}).items():
        spec.append({"name": str(k), "value": str(v), "isRegex": False, "isEqual": True})
    if config.MODE != "auto":
        return (f"PROPOSAL RECORDED (propose mode): silence {spec} for {hours}h "
                f"— {comment}")
    now = datetime.now(timezone.utc)
    body = {"matchers": spec,
            "startsAt": now.isoformat(),
            "endsAt": (now + timedelta(hours=hours)).isoformat(),
            "createdBy": "cluster-agent",
            "comment": comment}
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            existing = await c.get(f"{config.ALERTMANAGER_URL}/api/v2/silences")
            for s in (existing.json() if existing.status_code == 200 else []):
                if s.get("status", {}).get("state") != "active":
                    continue
                if {(m["name"], m["value"]) for m in s.get("matchers", [])} == \
                   {(m["name"], m["value"]) for m in spec}:
                    return (f"already silenced until {s.get('endsAt')} "
                            f"(id {s.get('id')}); nothing to do")
            r = await c.post(f"{config.ALERTMANAGER_URL}/api/v2/silences", json=body)
        if r.status_code >= 300:
            return f"silence failed: HTTP {r.status_code}: {r.text[:500]}"
        return (f"silenced {spec} for {hours}h (id "
                f"{r.json().get('silenceID', '?')}): {comment}")
    except Exception as exc:
        return f"silence failed: {exc}"


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
        "description": "Search the operator's long-term conversation archive for prior incidents, decisions and known fixes. Returns one line per hit, each with a message_uuid; call get_context on a uuid to read around it rather than searching again. The [sender] tag says who spoke — an operator DECISION is usually what you want, and it will be in their words, not the assistant's. Phrase the query the way they would have said it and include their decision vocabulary: ignore, expected, known, on purpose, leave it, waiting on.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"},
            "sender": {"type": "string", "description": "restrict to one speaker: 'User' for what the operator themselves said — use this when you need a DECISION, such as whether they have already said to ignore something — or 'Claude' for assistant messages. Omit for everything. Older imported transcripts carry no speaker and drop out when this is set, so try it as well as, not instead of, an unfiltered search."}},
            "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_context",
        "description": "Read the archived conversation surrounding a search_memory hit, verbatim. Use the message_uuid from a hit; radius is how many messages either side.",
        "parameters": {"type": "object", "properties": {"message_uuid": {"type": "string"}, "radius": {"type": "integer"}},
            "required": ["message_uuid"]}}},
    {"type": "function", "function": {"name": "silence_alert",
        "description": "Stop an alert from notifying, for a bounded time. Use this ONLY when the operator has already said this state is known and should be ignored — quote that in the comment and cite the search_memory hit it came from. Never silence something you merely judged unimportant yourself; say so in the report and let a human decide.",
        "parameters": {"type": "object", "properties": {
            "alertname": {"type": "string", "description": "exact alertname to silence; required"},
            "comment": {"type": "string", "description": "why, in the operator's own words, and where they said it"},
            "hours": {"type": "integer", "description": "duration; default 168 (7 days)"},
            "matchers": {"type": "object", "description": "extra exact label matchers, e.g. {\"instance\":\"192.168.1.12\"}, to keep the silence narrow"}},
            "required": ["alertname", "comment"]}}},
]


def offered() -> list:
    """The tools handed to the model on this call. Git tools only when a
    token exists, so an unconfigured gateway does not advertise dead tools."""
    return (TOOLS + (observe.TOOLS if config.PROMETHEUS_URL or config.ALERTMANAGER_URL else [])
            + (gittools.TOOLS if config.git_token() else []))


def is_mutation(name: str, args: dict) -> bool:
    """Does this call change something outside this process?

    Asked of every dispatch rather than maintained as a list of "dangerous
    tools", so a tool added later is reported as an action by default instead of being
    silently exempt until someone remembers to add it here.
    """
    if name in ("ha_call_service", "silence_alert", "git_open_pr"):
        return True
    if name == "run_kubectl":
        verb, argv, err = split_verb([str(a).lower() for a in (args.get("args") or [])])
        if err or not verb:
            return True             # refused anyway; report the attempt
        return verb not in READ_VERBS or argv[:2] == ["rollout", "restart"]
    return False


async def dispatch(name: str, args: dict, emit=None) -> str:
    """Run a tool, reporting it before and after as a structured event.

    Before, because a run that dies mid-change must still leave a record of
    what it started; after, because "I ran this" and "this is what happened"
    are different facts and the second is the one worth reading. Refusals and
    proposals are reported too: "the agent tried to do X and was stopped" is
    exactly as interesting as "the agent did X".

    The events go to `emit`, i.e. onto the caller's response stream as a
    `tool_event` delta. Where they are shown (a chat window, the cluster-agent's
    incident post) is the caller's business; the gateway knows nothing about
    Discord. Every mutation is also logged here, whoever is listening.
    """
    mutating = is_mutation(name, args)
    if mutating:
        log.info("ACTION %s %s", name, json.dumps(args)[:400])
    if emit is not None:
        await emit({"tool_event": {"phase": "call", "name": name, "args": args,
                                   "mutating": mutating}})
    out = await _dispatch(name, args)
    head = out.strip().splitlines()[0] if out.strip() else "(no output)"
    if mutating:
        log.info("RESULT %s -> %s", name, head[:400])
    if emit is not None:
        await emit({"tool_event": {"phase": "result", "name": name, "mutating": mutating,
                                   "summary": head[:400], "output": out[:2000]}})
    return out


async def _dispatch(name: str, args: dict) -> str:
    if name == "run_kubectl":
        return await asyncio.to_thread(run_kubectl, args.get("args", []))
    if name in gittools.NAMES:
        return await gittools.dispatch(name, args)
    if name in observe.NAMES:
        return await observe.dispatch(name, args)
    if name == "ha_get_states":
        return await ha_get_states(args.get("entity_id", ""))
    if name == "ha_call_service":
        return await ha_call_service(args.get("domain", ""), args.get("service", ""),
                                     args.get("entity_id", ""), args.get("data"))
    if name == "search_memory":
        return await search_memory(args.get("query", ""), int(args.get("k", 0) or 0),
                                   str(args.get("sender") or ""))
    if name == "get_context":
        return await get_context(args.get("message_uuid", ""),
                                 int(args.get("radius", 3) or 3))
    if name == "silence_alert":
        return await silence_alert(args.get("alertname", ""), args.get("comment", ""),
                                   int(args.get("hours", 168) or 168),
                                   args.get("matchers"))
    return f"REFUSED: unknown tool '{name}'"

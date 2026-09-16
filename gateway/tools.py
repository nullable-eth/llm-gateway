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
import contextvars
import json
import logging
from datetime import datetime, timedelta, timezone
import subprocess

import httpx

from . import config, gittools

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
# them announces itself to Discord before and after, which is the actual control.
FORBIDDEN_VERBS = {"exec", "attach", "cp", "port-forward", "proxy", "debug",
                   "certificate", "config"}
FORBIDDEN_SUBS = {"auth": {"can-i", "whoami"}}    # only these subcommands of auth
FORBIDDEN_TOKENS = {"--token", "--kubeconfig", "--as", "--as-group", "--as-uid"}
FORBIDDEN_RESOURCES = {"secret", "secrets", "serviceaccount", "serviceaccounts",
                       "clusterrole", "clusterroles", "role", "roles",
                       "rolebinding", "rolebindings", "clusterrolebinding",
                       "clusterrolebindings", "csr", "certificatesigningrequest",
                       "certificatesigningrequests"}


def kubectl_guard(args: list[str]) -> str | None:
    """Return a rejection reason, or None if the command may run."""
    if not args:
        return "empty command"
    low = [a.lower() for a in args]
    verb = low[0]
    if verb in FORBIDDEN_VERBS:
        return (f"'{verb}' is never permitted: it opens a shell or tunnel into a "
                f"workload, which is a bigger claim than any alert justifies")
    if verb in FORBIDDEN_SUBS and (len(low) < 2 or low[1] not in FORBIDDEN_SUBS[verb]):
        return f"'{verb}' is only permitted as: {' | '.join(sorted(FORBIDDEN_SUBS[verb]))}"
    for tok in low:
        if tok.split("=")[0] in FORBIDDEN_TOKENS:
            return f"'{tok}' is never permitted: the agent acts as itself or not at all"
    for tok in low[1:]:
        # "secrets", "secret/foo", "secrets.v1." -- any spelling that names one.
        if tok.split("/")[0].split(".")[0] in FORBIDDEN_RESOURCES:
            return (f"'{tok}' is never permitted: credentials and permissions are "
                    f"not runtime state (and are absent from this RBAC anyway)")
    is_read = verb in READ_VERBS and low[0:2] != ["rollout", "restart"]
    for name in config.PROTECTED:
        if any(name in t for t in low[1:]) and not is_read:
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
    return TOOLS + (gittools.TOOLS if config.git_token() else [])


def is_mutation(name: str, args: dict) -> bool:
    """Does this call change something outside this process?

    Asked of every dispatch rather than maintained as a list of "dangerous
    tools", so a tool added later is announced by default instead of being
    silently exempt until someone remembers to add it here.
    """
    if name in ("ha_call_service", "silence_alert", "git_open_pr"):
        return True
    if name == "run_kubectl":
        argv = [str(a).lower() for a in (args.get("args") or [])]
        if not argv:
            return False
        return argv[0] not in READ_VERBS or argv[:2] == ["rollout", "restart"]
    return False


# Set per request from X-Discord-Thread. A ContextVar rather than a parameter
# because the thread belongs to the REQUEST, and threading it through the loop,
# the dispatcher and every tool signature would put Discord in the type of code
# that should not know Discord exists.
ANNOUNCE_TO: contextvars.ContextVar[str] = contextvars.ContextVar("announce_to", default="")


async def announce(text: str) -> None:
    """Post an action to Discord. Never raises, never blocks the tool.

    Bot only: the caller's incident thread, else DISCORD_CHANNEL_ID
    (#cluster-alerts). If the thread post fails, the channel is tried. Every
    post's status is checked: an HTTP error is a failure, not a success with a
    body nobody reads (how a deleted webhook once swallowed every
    chat-initiated action).
    """
    log.info("ACTION %s", text.replace("\n", " ")[:400])
    body = {"content": text[:1900], "allowed_mentions": {"parse": []}}
    targets = []
    if config.DISCORD_BOT_TOKEN:
        bot = {"Authorization": f"Bot {config.DISCORD_BOT_TOKEN}",
               "User-Agent": "llm-gateway (actions, 1.0)"}
        for channel, what in ((ANNOUNCE_TO.get(), "thread"),
                              (config.DISCORD_CHANNEL_ID, "channel")):
            if channel:
                targets.append((what, f"https://discord.com/api/v10/channels/{channel}/messages", bot))
    if not targets:
        log.error("action NOT announced: no bot token or Discord channel configured")
        return
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            for what, url, headers in targets:
                try:
                    r = await c.post(url, headers=headers, json=body)
                except httpx.HTTPError as exc:
                    log.warning("action post to %s failed: %s", what, type(exc).__name__)
                    continue
                if r.status_code < 300:
                    return
                log.warning("action post to %s failed: HTTP %s %s", what,
                            r.status_code, r.text[:120].replace("\n", " "))
        log.error("action NOT announced anywhere: %s", text[:200])
    except Exception as exc:
        log.warning("action post failed (continuing): %s", exc)


async def dispatch(name: str, args: dict) -> str:
    """Every mutation announces itself here, before and after.

    Before, because a run that dies mid-change must still leave a record of what
    it started; after, because "I ran this" and "this is what happened" are
    different facts and the second one is the one worth reading. Refusals and
    proposals are announced too: "the agent tried to do X and was stopped" is
    exactly as interesting as "the agent did X".
    """
    mutating = is_mutation(name, args)
    if mutating:
        await announce(f"**[cluster-agent] action** `{name}` {json.dumps(args)[:400]}")
    out = await _dispatch(name, args)
    if mutating:
        head = out.strip().splitlines()[0] if out.strip() else "(no output)"
        await announce(f"**[cluster-agent] result** `{name}` -> {head[:400]}")
    return out


async def _dispatch(name: str, args: dict) -> str:
    if name == "run_kubectl":
        return await asyncio.to_thread(run_kubectl, args.get("args", []))
    if name in gittools.NAMES:
        return await gittools.dispatch(name, args)
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

"""Gateway config — all knobs via env, nothing hardcoded.

Deliberately standalone: app/config.py requires PG_DSN at import time and the
sidecar has no business holding a database DSN.
"""
import os

UPSTREAM = os.environ.get("GATEWAY_UPSTREAM", "http://127.0.0.1:8000").rstrip("/")
VAULT_ROOT = os.environ.get("VAULT_ROOT", "/vault")
CAPTURE_DIR = os.environ.get("CAPTURE_DIR", ".staging/Chats/Live Capture")
PORT = int(os.environ.get("GATEWAY_PORT", "8010"))

# A conversation is buffered in the index (invisible to scanner and filing
# agent alike) and materialised as markdown only once it has been quiet this
# long. This is the whole point of the buffer: an agent mid-run must not be
# able to retrieve its own in-flight reasoning back out of RAG and mistake it
# for archived knowledge.
IDLE_S = int(os.environ.get("CAPTURE_IDLE_S", "1800"))            # 30 min
# Safety valve: a conversation that never goes quiet still gets written.
MAX_OPEN_S = int(os.environ.get("CAPTURE_MAX_OPEN_S", "43200"))   # 12 h
SWEEP_S = int(os.environ.get("CAPTURE_SWEEP_S", "60"))
# How long a flushed conversation stays reopenable by a late continuation.
REOPEN_S = int(os.environ.get("CAPTURE_REOPEN_S", "604800"))      # 7 d

# Vault writes are best-effort and must never touch the proxied request. A
# NAS outage drops the oldest records and increments a counter; that is its
# own class of problem and not one this process tries to solve.
QUEUE_MAX = int(os.environ.get("CAPTURE_QUEUE_MAX", "256"))
# Hard ceiling on conversations held in memory. Reached only when the vault
# has been unwritable long enough for unflushed work to pile up.
MAX_CONVERSATIONS = int(os.environ.get("CAPTURE_MAX_CONVERSATIONS", "500"))

# Ceiling on the copy of a non-streamed response body kept for parsing.
MAX_BODY = int(os.environ.get("CAPTURE_MAX_BODY", str(64 << 20)))

TITLE_MAX = int(os.environ.get("CAPTURE_TITLE_MAX", "60"))
# Client hints. Neither of these can affect whether a conversation is
# captured — they only name one the client already knows the identity of
# (cluster-agent's run id, say).
HDR_CONV_ID = "x-capture-conversation-id"
HDR_TITLE = "x-capture-title"

# Self-identification, and the ONLY thing that suppresses a transcript.
#
# Machine traffic whose transcript would carry no information: the filing
# agent's classification calls are a system prompt built from CLAUDE.md plus
# every Scope.md, a user prompt quoting a file already in the vault, and a
# verdict that is already persisted in filing_proposals. Archiving them would
# duplicate the vault into itself, repeatedly.
#
# An allowlist rather than a boolean opt-out, deliberately: a client cannot
# suppress itself by inventing a name. Only these exact values do, they are
# set here rather than by the caller, and an unrecognised value is captured
# and indexed like anything else. That also bounds the metric's label
# cardinality. It is not a security boundary — anything that can reach the
# endpoint can send a known name — but it is a far narrower hole than
# honouring any value, and the traffic still shows up in
# capture_suppressed_total even when no transcript is written.
HDR_CLIENT = "x-capture-client"
NOLOG_CLIENTS = {c.strip() for c in os.environ.get(
    "CAPTURE_NOLOG_CLIENTS", "agentmemory-filing").split(",") if c.strip()}

CONNECT_TIMEOUT_S = float(os.environ.get("CAPTURE_CONNECT_TIMEOUT_S", "5"))

# ----------------------------------------------------------------- compaction
# Keep a conversation inside the model's window regardless of which client sent
# it. Off means oversized requests go upstream untouched and fail there, which
# is what happened before this existed.
COMPACT_ENABLED = os.environ.get("COMPACT_ENABLED", "1") not in ("0", "false", "no")
# Fraction of n_ctx (read from the server's /props) a prompt may occupy before
# the middle of it is summarised.
COMPACT_AT = float(os.environ.get("COMPACT_AT", "0.75"))
# Messages at the end kept verbatim. Nudged earlier when the cut would split an
# assistant tool_call from its tool result.
COMPACT_KEEP_TAIL = int(os.environ.get("COMPACT_KEEP_TAIL", "8"))
# Generation headroom assumed when a request sets no max_tokens.
COMPACT_RESERVE = int(os.environ.get("COMPACT_RESERVE", "8192"))
COMPACT_SUMMARY_TOKENS = int(os.environ.get("COMPACT_SUMMARY_TOKENS", "2000"))
# Summaries are cached by the exact span they cover, so a client that resends
# its whole history every turn pays for one summary, not one per turn.
COMPACT_CACHE_MAX = int(os.environ.get("COMPACT_CACHE_MAX", "256"))

# Override when the probe is wrong. /props reports n_ctx, but with more than
# one slot the number that matters is the PER-SLOT context (n_ctx / n_parallel)
# — build a 200k prompt for a 131k slot and the server rejects it. Set this if
# the server's reporting and the slot size ever disagree.
COMPACT_N_CTX = int(os.environ.get("COMPACT_N_CTX", "0"))

# ------------------------------------------------------------------ sampling
# The model's sampling settings belong to the model, not to whichever chat
# client happened to send the request. llama-server's --temp/--top-p/... are
# only defaults, and Jan and Home Assistant both send their own values on every
# request, so a tier tuned to its author's recommendations answered each client
# differently. On, client sampling fields are dropped before forwarding and the
# server's flags decide. The archive still records what the client sent.
STRIP_SAMPLING = os.environ.get("GATEWAY_STRIP_SAMPLING", "0") not in ("0", "false", "no", "")
STRIP_SAMPLING_FIELDS = tuple(f.strip() for f in os.environ.get(
    "GATEWAY_STRIP_SAMPLING_FIELDS",
    "temperature,top_p,top_k,min_p,typical_p,presence_penalty,"
    "frequency_penalty,repeat_penalty,repetition_penalty,repeat_last_n,"
    "dynatemp_range,dynatemp_exponent,mirostat,mirostat_tau,mirostat_eta,"
    "xtc_probability,xtc_threshold").split(",") if f.strip())
# Machine clients whose sampling is deliberate and kept: agentmemory's filing
# classifier asks for temperature 0.1 and should get it. Same exact-match
# allowlist rule as NOLOG_CLIENTS — an unrecognised name keeps nothing.
SAMPLING_KEEP_CLIENTS = {c.strip() for c in os.environ.get(
    "GATEWAY_SAMPLING_KEEP_CLIENTS", "agentmemory-filing").split(",") if c.strip()}

# --------------------------------------------------------------------- tools
# Always on where enabled, because the chat clients in play expose a fixed
# api-key field and no way to send an extra header — per-request opt-in was
# never reachable from them. Off for the fast/voice tier: an 8B model answering
# "turn off the kitchen light" should not be paying for tool definitions, and
# should not be holding kubectl.
TOOLS_ENABLED = os.environ.get("GATEWAY_TOOLS", "0") not in ("0", "false", "no", "")
# Raised from 8 after a real incident: eight steps was not enough to even
# IDENTIFY a failing target, let alone fix it. The agent spent all eight on
# discovery, reported "budget ran out before I could check", and changed
# nothing, while the fault sat two kubectl calls away. Prefer dozens of cheap
# calls over a confident shrug.
TOOL_MAX_STEPS = int(os.environ.get("TOOL_MAX_STEPS", "40"))
# Wall-clock budget for starting NEW tool work, in seconds. A step budget does
# not bound time: steps get slower as the conversation grows, so 12 steps can
# be two minutes or twelve. Callers have their own timeouts — cluster-agent
# waits 1800s — and a loop that outruns them does the work, gets abandoned, and
# reports nothing. Past this the tools are withdrawn and the model must answer,
# leaving the remainder of the caller's patience for that final reply.
TOOL_MAX_SECONDS = int(os.environ.get("TOOL_MAX_SECONDS", "1200"))
# ...and a hard ceiling on the whole request, because withdrawing the tools does
# not bound anything on its own: the final generation was issued with no timeout
# at all, so a loop could and did run past the caller's patience and get
# abandoned mid-answer. Every upstream call now gets the time remaining against
# this deadline (never less than the floor), so the request either answers
# inside REQUEST_MAX_SECONDS or fails as a timeout the caller can report,
# instead of racing it.
REQUEST_MAX_SECONDS = int(os.environ.get("REQUEST_MAX_SECONDS", "1440"))
UPSTREAM_MIN_TIMEOUT_S = int(os.environ.get("UPSTREAM_MIN_TIMEOUT_S", "30"))
# The final answer's own budget, on top of the deadline above rather than
# inside it. Worst case per request is therefore ~REQUEST_MAX + ANSWER_TIMEOUT,
# which must stay under the caller's timeout (cluster-agent: 1800s). 1440+240
# leaves 2 minutes of headroom.
ANSWER_TIMEOUT_S = int(os.environ.get("ANSWER_TIMEOUT_S", "240"))
TOOL_OUTPUT_MAX = int(os.environ.get("TOOL_OUTPUT_MAX", "8000"))
KUBECTL_TIMEOUT_S = int(os.environ.get("KUBECTL_TIMEOUT_S", "60"))
# Guidance injected with the tools. The client never asked for the tools and
# cannot know how to budget them, so the policy travels with them. Empty
# disables. Default lives in policy.py.
TOOL_SYSTEM_PROMPT = os.environ.get("TOOL_SYSTEM_PROMPT", "__default__")

# propose | auto. Mutations are recorded rather than executed unless auto, and
# auto alone still does nothing without the phase-2 RBAC.
MODE = os.environ.get("MODE", "propose")
# Components the agent may not act on — the rule that stops it restarting the
# model it is thinking with.
PROTECTED = {p.strip() for p in os.environ.get("PROTECTED", "").split(",") if p.strip()}

HA_URL = os.environ.get("HA_URL", "").rstrip("/")
HA_TOKEN = os.environ.get("HA_TOKEN", "")
MEMORY_URL = os.environ.get("MEMORY_URL", "").rstrip("/")
MEMORY_TOKEN = os.environ.get("MEMORY_TOKEN", "")
# How many hits a memory search returns. Was effectively 6-8, which measured out
# as "the operator said it, at rank 12, and nobody looked past 8". Hits render
# one line each now, so a large k costs little and buys the tail of the ranking,
# which is where a decision from months ago actually sits.
MEMORY_SEARCH_K = int(os.environ.get("MEMORY_SEARCH_K", "25"))
MEMORY_SEARCH_MAX_K = int(os.environ.get("MEMORY_SEARCH_MAX_K", "50"))
# Alertmanager's own API, for silencing an alert the operator has already said
# to ignore. Empty disables the tool. The cap is what keeps a silence from
# outliving its reason: 30 days, and the default is a week.
ALERTMANAGER_URL = os.environ.get("ALERTMANAGER_URL", "").rstrip("/")
SILENCE_MAX_HOURS = int(os.environ.get("SILENCE_MAX_HOURS", "720"))
# Where every mutation announces itself, as it happens. The condition on the
# agent being allowed to change anything is that the change is visible: a report
# minutes later is not visibility, and a run that dies mid-way would take its
# only record with it. Empty means mutations are logged but not posted.
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "")
# Preferred over the webhook when set: with the bot token, an action can be
# announced INSIDE the incident thread the caller is working in (it passes the
# thread id in X-Discord-Thread), so the change and the alert that caused it sit
# in one place instead of two channels nobody reads together.
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")

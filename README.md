# llm-gateway

An OpenAI-API-transparent gateway that sits in front of a llama.cpp (or any
OpenAI-compatible) server and adds capabilities every caller gets, whatever
client they use:

- **logging** — every conversation archived as markdown into an agentmemory vault
- **compaction** — conversations that would overflow the context are summarised
- **tools** — the gateway runs the tool loop against one MCP endpoint, so
  clients with no tool support can still use every tool behind it

Point your Service at the gateway and let it forward to the model. Clients need
no changes at all.

## Why a gateway and not per-client

Each of these used to live somewhere else, once per caller. Compaction existed
only inside cluster-agent's alert loop, so a desktop chat client talking to the
same model had none of it and could run the model out of context. Tool calling
existed only there too, so nothing else could ask about the cluster. Logging
existed nowhere.

Putting them in the request path means one implementation, uniform behaviour on
every path, and callers that stay dumb. cluster-agent collapses to what it
should always have been: something that receives an alert and asks a question.

## The chain

```
client ─→ [capture] ─→ [compact] ─→ [tools] ─→ model
           observes     rewrites     may loop
```

**capture** never modifies the request. It takes a copy, buffers the
conversation in `<vault>/<capture dir>/.index/<uuid>.json`, and materialises
markdown only once the conversation has been quiet for `CAPTURE_IDLE_S`.
Nothing in `.index/` is indexed or searchable, which is deliberate: an agent
that searches memory mid-run must not be able to retrieve its own half-formed
reasoning and mistake it for an archived conclusion.

**compact** summarises the middle of an oversized conversation and forwards
`[system…, state summary, recent tail]`. The vault still holds the
conversation verbatim, so this costs nothing archivally — the transcript stays
complete and the reply records `context_compacted`. Token counts are exact,
from the server's own `/apply-template` and `/tokenize`.

**tools** runs the loop: call the model, execute what it asks for, repeat,
return one finished answer. The tools are whatever **one MCP endpoint**
(`MCP_URL`, Streamable HTTP) lists — normally an MCP gateway such as
[agentgateway](https://agentgateway.dev) multiplexing many servers. The gateway
has no tools of its own besides `finish()`, holds no tool credentials, and
keeps no permission list: what a call may do is decided behind the endpoint
(each server's credentials, Kubernetes RBAC and admission policy, the MCP
gateway's authorization rules), and a refusal comes back to the model as the
tool's result. Read-only access is a read-only role, not a mode here.

Every call is streamed to the caller as a `tool_event` delta (`call` with name,
args and `mutating`; `result` with summary, output and `error`), and every call
that may change something is logged. `mutating` comes from the tool's MCP
annotations: anything not marked `readOnlyHint` is reported as an action.

The tools arrive with a **use policy** appended to the caller's system message
(`gateway/policy.py`). The client never asked for these tools and cannot know
how to budget them, so injecting the tools without the guidance is an
incomplete feature: measured against the live archive, a model given a memory
search and no policy ran it thirteen times in one request, each a reword of
the last. The policy names no tool; the endpoint's own `instructions` are
appended to it. `TOOL_SYSTEM_PROMPT=""` disables it. The client sees a single reply; the archive gets
every tool call, result and intermediate thought. Compaction runs *between*
steps too, because tool output is what actually overflows a window.

## Credentials

The gateway holds no API key and should not be given one. Its own calls —
`/props`, `/tokenize`, the summariser, every step of the tool loop — reuse the
**caller's** `Authorization` header, so they are made on behalf of someone
already entitled to use the model. No header, no compaction and no loop.

**With tools enabled, the inference key reaches the tools.** Anything that may
call the model through this gateway may use everything `MCP_URL` offers, so
scope what the endpoint's credentials can do to what any holder of the
inference key may do, and restrict who can reach the model port.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GATEWAY_UPSTREAM` | `http://127.0.0.1:8000` | Model server to forward to |
| `GATEWAY_TOOLS` | `0` | Enable the tool loop (expert tiers only) |
| `VAULT_ROOT` | `/vault` | Mounted vault |
| `CAPTURE_DIR` | `.staging/Chats/Live Capture` | Vault-relative transcript folder |
| `CAPTURE_IDLE_S` | `1800` | Quiet period before a conversation is written |
| `CAPTURE_NOLOG_CLIENTS` | `agentmemory-filing` | `X-Capture-Client` names proxied but never written |
| `COMPACT_ENABLED` | `1` | Summarise oversized conversations |
| `COMPACT_AT` | `0.75` | Fraction of context a prompt may occupy |
| `COMPACT_KEEP_TAIL` | `8` | Recent messages kept verbatim |
| `COMPACT_N_CTX` | `0` | Override the probed window — **set this when the server runs more than one slot**, since what matters is `n_ctx / n_parallel` |
| `GATEWAY_STRIP_SAMPLING` | `0` | Drop client sampling fields (`temperature`, `top_p`, `top_k`, `min_p`, penalties, …) so the model server's `--temp`/`--top-p`/… flags always apply. The archive still records what the client sent |
| `GATEWAY_SAMPLING_KEEP_CLIENTS` | `agentmemory-filing` | `X-Capture-Client` names whose sampling is deliberate and kept |
| `MCP_URL` | *(empty)* | The tool endpoint (MCP Streamable HTTP). Empty: no tools but `finish()` |
| `MCP_TOKEN_FILE` / `MCP_TOKEN` | `/var/run/secrets/mcp/token` / *(empty)* | Bearer token for the endpoint, if it wants one. The file is re-read on change |
| `MCP_CALL_TIMEOUT_S` | `120` | One tool call's limit |
| `MCP_TOOLS_TTL_S` | `60` | How long a `tools/list` is reused |
| `MCP_INSTRUCTIONS_MAX` | `4000` | Cap on the endpoint instructions appended to the policy |
| `TOOL_MAX_STEPS` | `40` | Model+tool round trips before it must answer |
| `TOOL_MAX_SECONDS` | `1200` | Wall-clock budget for starting new tool work; past it the tools are withdrawn |
| `REQUEST_MAX_SECONDS` | `1440` | Ceiling on the whole request, loop included |
| `TOOL_OUTPUT_MAX` | `8000` | Characters of one tool result the model sees |
| `TOOL_SYSTEM_PROMPT` | *(built-in)* | Tool-use policy appended to the caller's system message; empty disables |

Full list in `gateway/config.py`, which is the only place env is read.

## Known edges

- **Streaming with tools is stitched live.** A tool run is many upstream
  calls, so a streaming client gets each step's `reasoning_content` and
  `content` relayed as it is generated, plus a `[tool] name: args` line in the
  thinking for every tool call. Tool-call deltas are never forwarded (the
  client did not ask for tools), SSE comments keep the connection alive while
  a tool runs, and because the status line is already sent, a failure arrives
  in-band as a final `[gateway: agent loop failed/timed out]` chunk rather
  than a 502/504. Non-streaming callers (cluster-agent) are unchanged.
- **A loop must finish inside its caller's patience.** Steps do not bound
  wall clock — they get slower as the conversation grows — so there is a
  separate `TOOL_MAX_SECONDS` budget, past which the tools are withdrawn and
  the model must answer. A loop that outruns its caller does the work, gets
  abandoned, and reports nothing: cluster-agent waits 300s and then posts
  `LLM error at step 1:` with an empty message, because `httpx.ReadTimeout`
  stringifies to nothing.
- **Concurrency is the server's.** With `--parallel 1` a tool loop holds the
  only slot for its whole run, so an alert investigation and an interactive
  question block each other. Raising `--parallel` partitions the same
  `--ctx-size` into more slots rather than costing memory — and compaction is
  what makes the smaller per-slot window safe.
- **`tests/vaultio_ref.py` is a verbatim copy** of agentmemory's
  `app/vaultio.py`. The gateway writes markdown that agentmemory's scanner must
  chunk, and this is what proves both halves still agree. It is a copy across a
  repo boundary and can drift: refresh it when that file changes.

## Test

```
pip install 'fastapi==0.115.*' 'uvicorn==0.30.*' 'httpx==0.27.*' 'prometheus-client==0.20.*'
python tests/test_gateway.py
```

Runs a fake model server and the real gateway against a scratch vault and
asserts on the rendered markdown, on what the chunker makes of it, on the
compaction decisions, and on the tool loop against a fake MCP endpoint
(`tests/fake_mcp.py`) — a test that calls a real cluster is not a test.

## Release

Push to `main` → `:latest` + `:sha-…`. Tag `vX.Y.Z` → `:X.Y.Z`. Pin by `sha-`.


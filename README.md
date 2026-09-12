# llm-gateway

An OpenAI-API-transparent gateway that sits in front of a llama.cpp (or any
OpenAI-compatible) server and adds capabilities every caller gets, whatever
client they use:

- **logging** — every conversation archived as markdown into an agentmemory vault
- **compaction** — conversations that would overflow the context are summarised
- **tools** — the gateway runs the tool loop, so clients with no tool support
  can still ask questions that need `kubectl` or the memory archive

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
return one finished answer. `search_memory` returns scored snippets each
carrying a `message_uuid`, and `get_context` reads the archived conversation
around one — without that second tool a model that finds a snippet too short
can only search again with different words, which it will do until it runs out
of steps.

The tools arrive with a **use policy** appended to the caller's system message
(`gateway/policy.py`). The client never asked for these tools and cannot know
how to budget them, so injecting the tools without the guidance is an
incomplete feature: measured against the live archive, a model given
search_memory and no policy ran it thirteen times in one request, each a
reword of the last. `TOOL_SYSTEM_PROMPT=""` disables it. The client sees a single reply; the archive gets
every tool call, result and intermediate thought. Compaction runs *between*
steps too, because tool output is what actually overflows a window.

## Credentials

The gateway holds no API key and should not be given one. Its own calls —
`/props`, `/tokenize`, the summariser, every step of the tool loop — reuse the
**caller's** `Authorization` header, so they are made on behalf of someone
already entitled to use the model. No header, no compaction and no loop.

**With tools enabled this is a privilege escalation and should be understood as
one.** Anything holding the inference key can reach `kubectl` through the
gateway, where that key previously bought only inference. The guards bound what
it can do, not who may try:

- read verbs pass; `DENY` refuses secrets, exec, attach, cp, port-forward and
  every direct-write verb, because writes belong to GitOps
- anything matching a `PROTECTED` component is refused for non-read verbs —
  the rule that stops the agent restarting the model it is thinking with
- mutations require `MODE=auto`, and are otherwise recorded as proposals
- the RBAC underneath is read-only and enumerates resources, so `secrets` is
  structurally absent rather than merely denied

The intended next step is a **separate tool credential** so a leaked inference
key cannot reach the cluster at all. The blocker is client-side: chat clients
in practice expose a fixed api-key field and no way to send an extra header, so
that credential has to ride a distinct port or model name rather than a header.
Until then, treat the inference key as cluster-read-capable.

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
| `MODE` | `propose` | `propose` or `auto`; auto alone still needs phase-2 RBAC |
| `PROTECTED` | *(empty)* | Components the agent may not act on |
| `TOOL_MAX_STEPS` | `12` | Model+tool round trips before it must answer |
| `TOOL_SYSTEM_PROMPT` | *(built-in)* | Tool-use policy appended to the caller's system message; empty disables |
| `HA_URL` / `HA_TOKEN` | *(empty)* | Home Assistant; empty disables those tools |
| `MEMORY_URL` / `MEMORY_TOKEN` | *(empty)* | agentmemory search; empty disables that tool |

Full list in `gateway/config.py`, which is the only place env is read.

## Known edges

- **Streaming with tools is re-emitted, not relayed.** A tool run is many
  upstream calls, so there is no single stream to pass through; the finished
  answer is chunked into SSE frames. Clients render it fine, but it arrives in
  blocks rather than token by token.
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
compaction decisions, and on the tool guards. `run_kubectl` is stubbed — a test
that shells out to a real cluster is not a test.

## Release

Push to `main` → `:latest` + `:sha-…`. Tag `vX.Y.Z` → `:X.Y.Z`. Pin by `sha-`.


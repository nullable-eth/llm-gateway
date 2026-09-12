"""Context compaction — keep a conversation inside the model's window.

WHY HERE. Every client of the model already flows through this proxy, so one
implementation covers desktop chat clients, cluster-agent, and anything added
later, with no client-side changes — the same reason capture lives here.
cluster-agent has its own compaction, but that only ever applied to its own
alert loop.

WHAT IS AND ISN'T LOST. The vault already holds the conversation verbatim, so
compacting what the *model* sees costs nothing archivally: the transcript
stays complete and only the forwarded prompt shrinks. Capture records what the
client actually sent; the compaction is noted alongside it, because a reply
conditioned on a summary is a real behaviour change and the archive should say
so.

CREDENTIALS. The proxy holds no API key and must not start holding one. Its
own calls here — /props, /apply-template, /tokenize, and the summarising
completion — reuse the caller's Authorization header, so they are made on
behalf of a caller already entitled to use the model. No header, no
compaction. The calls go straight to the upstream socket, not back through
this proxy, so they are neither captured nor recursive.

FAILURE IS ALWAYS "FORWARD UNCHANGED". Compaction is an optimisation. Any
error — tokenizer unavailable, summariser refusing, malformed messages — falls
back to sending the request exactly as it arrived.
"""
import asyncio
import hashlib
import json
import logging
import time

import httpx

from . import config

log = logging.getLogger("capture")

SUMMARY_MARKER = "[COMPACTED CONVERSATION STATE]"
SUMMARY_PROMPT = (
    "STOP. Do not answer the conversation above. Compact it into a dense state "
    "note that lets you continue it without the original text: who the user is "
    "and what they are working on, decisions made and their reasoning, exact "
    "names, paths, commands, identifiers and error strings that matter, what "
    "has been tried, what is still open, and the immediate next step. Preserve "
    "specifics over prose — a fact you drop is gone. Plain text, no preamble."
)

_ctx_cache: dict = {"n_ctx": 0, "at": 0.0}
_summaries: dict = {}          # digest -> (summary, last_used)


def _digest(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8", "replace")
    ).hexdigest()


def _chars(messages: list) -> int:
    n = 0
    for m in messages:
        c = m.get("content")
        n += len(c) if isinstance(c, str) else len(json.dumps(c, ensure_ascii=False))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            n += len(str(fn.get("arguments") or "")) + len(str(fn.get("name") or ""))
    return n


class Compactor:
    """One per process. Stateless per request apart from two caches: the
    server's context size, and summaries keyed by the exact span they cover —
    which matters because a client that resends its whole history every turn
    would otherwise pay for a fresh summary on every one of them."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------ upstream
    async def _post(self, path: str, body: dict, auth: str, timeout: float):
        r = await self.client.post(
            f"{config.UPSTREAM}{path}", json=body,
            headers={"Authorization": auth} if auth else {}, timeout=timeout)
        r.raise_for_status()
        return r.json()

    async def n_ctx(self, auth: str) -> int:
        if config.COMPACT_N_CTX:
            return config.COMPACT_N_CTX
        if _ctx_cache["n_ctx"] and time.time() - _ctx_cache["at"] < 3600:
            return _ctx_cache["n_ctx"]
        r = await self.client.get(
            f"{config.UPSTREAM}/props",
            headers={"Authorization": auth} if auth else {}, timeout=30)
        r.raise_for_status()
        gen = r.json().get("default_generation_settings") or {}
        n = int(gen.get("n_ctx") or 0)
        if n:
            _ctx_cache.update(n_ctx=n, at=time.time())
        return n

    async def count_tokens(self, messages: list, auth: str) -> int:
        """Exact count, via the server's own template and tokenizer."""
        t = await self._post("/apply-template", {"messages": messages}, auth, 60)
        prompt = t.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("apply-template returned no prompt")
        tok = await self._post("/tokenize", {"content": prompt}, auth, 120)
        return len(tok.get("tokens") or [])

    async def summarise(self, messages: list, auth: str) -> str:
        body = {"messages": messages + [{"role": "user", "content": SUMMARY_PROMPT}],
                "stream": False, "temperature": 0.2,
                "max_tokens": config.COMPACT_SUMMARY_TOKENS}
        out = await self._post("/v1/chat/completions", body, auth, 900)
        msg = ((out.get("choices") or [{}])[0].get("message") or {})
        text = (msg.get("content") or "").strip()
        if not text:
            # A reasoning model that spent its whole budget thinking leaves
            # content empty; the reasoning itself is still a usable state note.
            text = (msg.get("reasoning_content") or "").strip()
        if not text:
            raise ValueError("summariser returned nothing")
        return text

    # ------------------------------------------------------------ planning
    @staticmethod
    def _split(messages: list) -> tuple[list, int]:
        """(leading system messages, index where the verbatim tail starts).

        The tail boundary is nudged earlier until it is legal to cut there: a
        tail may not begin with a `tool` message, and every `tool` message in
        it must have the assistant `tool_calls` entry that produced it. Cutting
        mid-pair yields a request many servers reject outright, and one that no
        model can interpret.
        """
        head = [m for m in messages if m.get("role") == "system"]
        start = len(head)
        cut = max(start, len(messages) - config.COMPACT_KEEP_TAIL)
        while cut > start:
            tail = messages[cut:]
            if tail and tail[0].get("role") == "tool":
                cut -= 1
                continue
            ids = {tc.get("id") for m in tail for tc in (m.get("tool_calls") or [])
                   if tc.get("id")}
            orphan = any(m.get("role") == "tool" and m.get("tool_call_id")
                         and m["tool_call_id"] not in ids for m in tail)
            if orphan:
                cut -= 1
                continue
            break
        return head, cut

    async def _summary_for(self, span: list, auth: str) -> str:
        key = _digest([m.get("role") for m in span] + [_chars(span), len(span),
                                                       _digest(span)])
        hit = _summaries.get(key)
        if hit:
            _summaries[key] = (hit[0], time.time())
            return hit[0]
        text = await self.summarise(span, auth)
        _summaries[key] = (text, time.time())
        if len(_summaries) > config.COMPACT_CACHE_MAX:
            for k, _ in sorted(_summaries.items(), key=lambda kv: kv[1][1])[:64]:
                _summaries.pop(k, None)
        return text

    # --------------------------------------------------------------- entry
    async def maybe_compact(self, body: dict, auth: str) -> tuple[dict | None, dict]:
        """Returns (new_body_or_None, info). None means forward unchanged."""
        info: dict = {"compacted": False}
        messages = body.get("messages")
        if not config.COMPACT_ENABLED or not auth or not isinstance(messages, list):
            return None, info
        if len(messages) < config.COMPACT_KEEP_TAIL + 2:
            return None, info

        # Cheap gate first: ~3 chars per token is a deliberate under-estimate,
        # so the exact count only runs when a request is plausibly large. A
        # normal chat turn costs nothing.
        approx = _chars(messages) / 3
        try:
            n_ctx = await self.n_ctx(auth)
        except Exception as e:
            log.debug("compact: /props unavailable (%s); forwarding unchanged", e)
            return None, info
        if not n_ctx:
            return None, info
        reserve = int(body.get("max_tokens") or 0) or config.COMPACT_RESERVE
        budget = int(n_ctx * config.COMPACT_AT) - reserve
        if budget <= 0 or approx < budget * 0.6:
            return None, info

        try:
            exact = await self.count_tokens(messages, auth)
        except Exception as e:
            log.warning("compact: token count failed (%s); forwarding unchanged", e)
            return None, info
        info["tokens_before"] = exact
        if exact <= budget:
            return None, info

        head, cut = self._split(messages)
        span = messages[len(head):cut]
        if not span:
            # Nothing between the system prompt and the tail: the tail alone
            # is over budget, which compaction cannot fix. Let the server
            # decide rather than silently mangling the request.
            log.warning("compact: %d tokens over budget %d but nothing to "
                        "summarise; forwarding unchanged", exact, budget)
            return None, info
        try:
            async with self._lock:        # one slot upstream; serialise
                summary = await self._summary_for(span, auth)
        except Exception as e:
            log.warning("compact: summarise failed (%s); forwarding unchanged", e)
            return None, info

        new_messages = head + [{"role": "assistant",
                                "content": f"{SUMMARY_MARKER}\n{summary}"}] \
            + messages[cut:]
        new_body = dict(body)
        new_body["messages"] = new_messages
        try:
            info["tokens_after"] = await self.count_tokens(new_messages, auth)
        except Exception:
            pass
        info.update(compacted=True, summarised_messages=len(span),
                    kept_tail=len(messages) - cut, budget=budget, n_ctx=n_ctx)
        log.info("compact: %s tokens -> %s (summarised %d message(s), kept %d)",
                 info.get("tokens_before"), info.get("tokens_after", "?"),
                 len(span), len(messages) - cut)
        return new_body, info

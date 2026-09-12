"""Reassemble a streamed chat completion without touching the byte stream.

The client's bytes are forwarded untouched; this only watches a copy go past.
llama.cpp emits reasoning as `reasoning_content` deltas (--reasoning on);
other builds spell it `reasoning`, so both are accepted. Tool calls arrive as
fragments keyed by `index`, with arguments split across any number of chunks.
"""
import json

from .normalize import Msg, text_of

MAX_PENDING = 8 << 20          # guard against a stream with no newlines


def _call_order(k):
    """tool_call `index` is an int in practice; tolerate anything."""
    return (0, k, "") if isinstance(k, int) else (1, 0, str(k))


class ChatAccumulator:
    def __init__(self):
        self._buf = b""
        self._content: list[str] = []
        self._reasoning: list[str] = []
        self._calls: dict = {}
        self.finish_reason = ""
        self.usage = None
        self.model = ""
        self.saw_done = False
        self.extra_choices = False

    # ------------------------------------------------------------- feeding
    def feed(self, chunk: bytes) -> None:
        """Never raises: a capture-side parse failure must not break the tee."""
        try:
            self._buf += chunk
            if len(self._buf) > MAX_PENDING:
                self._buf = self._buf[-MAX_PENDING:]
            while b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                self._line(line.rstrip(b"\r"))
        except Exception:
            pass

    def _line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        payload = line[5:].strip()
        if payload == b"[DONE]":
            self.saw_done = True
            return
        try:
            obj = json.loads(payload.decode("utf-8", "replace"))
        except ValueError:
            return
        if isinstance(obj, dict):
            self.event(obj)

    def event(self, obj: dict) -> None:
        if isinstance(obj.get("usage"), dict):
            self.usage = obj["usage"]
        if isinstance(obj.get("model"), str):
            self.model = obj["model"]
        choices = obj.get("choices")
        if not isinstance(choices, list) or not choices:
            return
        if len(choices) > 1:
            self.extra_choices = True
        ch = choices[0]
        if not isinstance(ch, dict):
            return
        if ch.get("finish_reason"):
            self.finish_reason = str(ch["finish_reason"])
        delta = ch.get("delta")
        if not isinstance(delta, dict):
            delta = ch.get("message") if isinstance(ch.get("message"), dict) else {}
        self._delta(delta)

    def _delta(self, d: dict) -> None:
        c = d.get("content")
        if isinstance(c, str):
            self._content.append(c)
        elif c is not None:
            self._content.append(text_of(c))
        r = d.get("reasoning_content")
        if r is None:
            r = d.get("reasoning")
        if isinstance(r, str):
            self._reasoning.append(r)
        elif r is not None:
            self._reasoning.append(text_of(r))
        for tc in d.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            slot = self._calls.setdefault(
                tc.get("index", len(self._calls)),
                {"id": "", "name": "", "arguments": ""})
            if tc.get("id"):
                slot["id"] = str(tc["id"])
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            if fn.get("name"):
                slot["name"] += str(fn["name"])
            args = fn.get("arguments", tc.get("arguments"))
            if isinstance(args, str):
                slot["arguments"] += args
            elif args is not None:
                slot["arguments"] += text_of(args)

    # ------------------------------------------------------------- result
    def empty(self) -> bool:
        return not (self._content or self._reasoning or self._calls
                    or self.finish_reason)

    def message(self, truncated: bool = False) -> Msg:
        calls = [{"id": self._calls[i]["id"],
                  "name": self._calls[i]["name"] or "tool",
                  "arguments": self._calls[i]["arguments"]}
                 for i in sorted(self._calls, key=_call_order)]
        extra = {}
        # "stop" and "tool_calls" are how a reply normally ends: plumbing with
        # no reader value, and details() bodies survive into embed_text, so
        # surfacing them would put the same boilerplate in every assistant
        # chunk's embedding. Anything else (length, content_filter, ...) says
        # something happened and is kept.
        if self.finish_reason and self.finish_reason not in ("stop", "tool_calls"):
            extra["finish_reason"] = self.finish_reason
        if self.usage:
            extra["usage"] = self.usage
        if truncated or (not self.saw_done and not self.finish_reason):
            extra["stream_incomplete"] = True
        if self.extra_choices:
            extra["note"] = "server returned n>1; only choice 0 is reassembled"
        return Msg(role="assistant", content="".join(self._content),
                   reasoning="".join(self._reasoning), tool_calls=calls,
                   extra=extra, truncated=truncated)

"""OpenAI chat wire format -> canonical messages.

The identity payload deliberately excludes reasoning and tool-call ids. The
server runs with --reasoning-preserve, so a re-sent assistant turn may or may
not carry its reasoning_content depending on the client, and both spellings
have to hash to the same message. Tool-call ids are minted per response and
some clients regenerate them on replay.

Unrecognised fields are kept in `extra` and rendered (the importer's
safety-net idea) so a field a client starts sending shows up in the archive
rather than vanishing. They stay out of the identity payload: a field that
changes between turns must not fork a message's uuid.
"""
import json
from dataclasses import dataclass, field

KNOWN_MSG_FIELDS = {"role", "content", "reasoning_content", "reasoning",
                    "tool_calls", "tool_call_id", "name", "function_call"}


def jcanon(o) -> str:
    try:
        return json.dumps(o, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(o)


def text_of(content) -> str:
    """Flatten a content field that may be a string or a list of parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                out.append(part["text"])
            else:
                out.append(jcanon(part))
        return "\n".join(out)
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"]
    return jcanon(content)


def _tool_calls(raw) -> list:
    out = []
    for c in raw or []:
        if not isinstance(c, dict):
            out.append({"id": "", "name": "tool", "arguments": jcanon(c)})
            continue
        fn = c.get("function") if isinstance(c.get("function"), dict) else {}
        args = fn.get("arguments", c.get("arguments", ""))
        out.append({"id": c.get("id") or "",
                    "name": fn.get("name") or c.get("name") or "tool",
                    "arguments": args if isinstance(args, str) else jcanon(args)})
    return out


@dataclass
class Msg:
    role: str
    content: str = ""
    reasoning: str = ""
    tool_calls: list = field(default_factory=list)
    tool_call_id: str = ""
    name: str = ""
    extra: dict = field(default_factory=dict)
    ts: str = ""
    truncated: bool = False

    def payload(self) -> list:
        return [self.role, self.content,
                [[c["name"], c["arguments"]] for c in self.tool_calls],
                self.tool_call_id, self.name]

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content,
                "reasoning": self.reasoning, "tool_calls": self.tool_calls,
                "tool_call_id": self.tool_call_id, "name": self.name,
                "extra": self.extra, "ts": self.ts,
                "truncated": self.truncated}

    @staticmethod
    def from_dict(d: dict) -> "Msg":
        return Msg(role=d.get("role") or "user", content=d.get("content") or "",
                   reasoning=d.get("reasoning") or "",
                   tool_calls=d.get("tool_calls") or [],
                   tool_call_id=d.get("tool_call_id") or "",
                   name=d.get("name") or "", extra=d.get("extra") or {},
                   ts=d.get("ts") or "", truncated=bool(d.get("truncated")))


def from_wire(m: dict) -> Msg:
    """One message as a client sent it, or as the server returned it."""
    fc = m.get("function_call")
    calls = _tool_calls(m.get("tool_calls"))
    if not calls and isinstance(fc, dict):            # legacy single-call form
        calls = _tool_calls([{"function": fc}])
    return Msg(
        role=m.get("role") or "user",
        content=text_of(m.get("content")),
        reasoning=(m.get("reasoning_content") or m.get("reasoning") or ""),
        tool_calls=calls,
        tool_call_id=m.get("tool_call_id") or "",
        name=m.get("name") or "",
        extra={k: v for k, v in m.items()
               if k not in KNOWN_MSG_FIELDS and v not in (None, "", [], {})},
    )


def request_messages(body: dict) -> list:
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return []
    return [from_wire(m) for m in msgs if isinstance(m, dict)]


def response_message(body: dict) -> Msg | None:
    """Assistant turn from a non-streamed /v1/chat/completions response."""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0] if isinstance(choices[0], dict) else {}
    msg = first.get("message")
    if not isinstance(msg, dict):
        return None
    out = from_wire(msg)
    out.role = "assistant"
    if len(choices) > 1:
        out.extra["additional_choices"] = choices[1:]
    return out

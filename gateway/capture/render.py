"""Markdown rendering — the vault format contract.

Shapes are taken from importer/unpack_export.py on purpose: the same
`<!-- msg:uuid -->` marker, the same `## Sender · <ISO>` header, the same
details() wrapper for auxiliary content. vaultio.SENDER_RE is literally
`^## (User|Claude) · (\\S+)`, so those two words, that U+00B7, one space each
side and a whitespace-free timestamp are load-bearing: anything else silently
yields NULL sender/created_at on every chunk row.

OpenAI's `tool` role has no equivalent sender. It is rendered under `## User`
with the result in a details block, which is exactly what Anthropic's export
does — there, tool_result blocks live inside the human-sender message. Same
for `system`: it is content the caller sent, so it renders under User.

Nothing is ever truncated.
"""
import json
import re

WINDOWS_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
PRESERVE_ON_REWRITE = ("date", "status", "project", "tags")


def sanitize(name: str, maxlen: int = 120) -> str:
    name = WINDOWS_BAD.sub("", (name or "").replace("\n", " ")).strip().rstrip(".")
    return (name or "untitled")[:maxlen]


def yaml_q(s: str) -> str:
    return '"' + (s or "").replace("\\", "\\\\").replace('"', '\\"') + '"'


def fence(text: str) -> str:
    longest = max((len(m) for m in re.findall(r"`+", text or "")), default=0)
    return "`" * max(3, longest + 1)


def jdump(o) -> str:
    try:
        return json.dumps(o, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(o)


def details(summary: str, body: str, lang: str = "") -> str:
    f = fence(body)
    return (f"<details><summary>{summary}</summary>\n\n"
            f"{f}{lang}\n{body}\n{f}\n\n</details>")


def title_from(messages) -> str:
    """First non-empty line of the first user message."""
    for m in messages:
        if m.role != "user" or not m.content.strip():
            continue
        for line in m.content.splitlines():
            if line.strip():
                return line.strip()
    for m in messages:
        if m.content.strip():
            return m.content.strip().splitlines()[0].strip()
    return "untitled conversation"


def tool_name_map(messages) -> dict:
    out = {}
    for m in messages:
        for c in m.tool_calls:
            if c.get("id"):
                out[c["id"]] = c.get("name") or "tool"
    return out


def _pretty_args(args: str) -> str:
    try:
        return json.dumps(json.loads(args), ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return args


def _extra_block(extra: dict) -> str:
    if not extra:
        return ""
    return details(f"Other fields ({', '.join(sorted(extra))})",
                   jdump(extra), "json")


def render_message(m, uuid_: str, names: dict) -> str:
    sender = "Claude" if m.role == "assistant" else "User"
    out = [f"<!-- msg:{uuid_} -->", f"## {sender} · {m.ts}", ""]
    body = []

    if m.role == "system":
        if m.content.strip():
            body.append(details("⚙️ System prompt", m.content.rstrip()))
    elif m.role == "tool":
        name = m.name or names.get(m.tool_call_id) or "tool"
        body.append(details(
            f"Tool result · <code>{name}</code> ({len(m.content):,} chars)",
            m.content))
    else:
        if m.reasoning.strip():
            body.append(details("🧠 Extended thinking", m.reasoning.rstrip()))
        if m.content.strip():
            body.append(m.content.rstrip())

    for c in m.tool_calls:
        head = f"**🔧 Tool call · `{c.get('name') or 'tool'}`**"
        body.append(head)
        args = _pretty_args(c.get("arguments") or "")
        if args.strip() and args.strip() not in ("null", "{}", '""'):
            f = fence(args)
            body.append(f"{f}json\n{args}\n{f}")

    if m.truncated:
        body.append("*[the client disconnected before this reply finished; "
                    "what arrived is above, verbatim]*")

    extra = _extra_block(m.extra)
    if extra:
        body.append(extra)

    out.append("\n\n".join(x for x in body if x))
    return "\n".join(out)


def render_transcript(conv, messages, uuids) -> str:
    """conv: dict with uuid/title/date/created/updated. Full file, one pass."""
    names = tool_name_map(messages)
    exchanges = sum(1 for m in messages if m.role == "assistant")
    title = conv["title"]
    head = ["---",
            f"title: {yaml_q(title)}",
            f"date: {conv['date']}",
            "type: chat-transcript",
            "status: unfiled",
            f"created: {conv['created']}",
            f"updated: {conv['updated']}",
            f"message_count: {len(messages)}",
            f"exchanges: {exchanges}",
            f"source_uuid: {conv['uuid']}",
            "capture: proxy",
            "tags: []",
            "---", "",
            f"# {title}", "",
            "> Captured at the model proxy — every message, thought, tool call",
            "> and reply that crossed it. Complete.",
            "> Unfiled — see [[CLAUDE]] for how this gets placed into the graph.",
            ""]
    parts = list(head)
    for m, u in zip(messages, uuids):
        parts += [render_message(m, u, names), ""]
    return "\n".join(parts).rstrip() + "\n"


def raw_fm_lines(text: str) -> dict:
    m = FM_RE.match(text or "")
    if not m:
        return {}
    return {line.split(":", 1)[0].strip(): line
            for line in m.group(1).split("\n") if ":" in line}


def preserve_filed_frontmatter(new_md: str, old_text: str) -> str:
    """Re-apply filing fields from an existing copy, verbatim.

    A reopened conversation whose transcript has already been filed into a
    node must not be demoted back to `status: unfiled` or lose its project
    wikilink just because someone sent another turn. Same fields, same
    reasoning as the importer's version of this function.
    """
    old = raw_fm_lines(old_text)
    m = FM_RE.match(new_md)
    if not m:
        return new_md
    out, seen = [], set()
    for line in m.group(1).split("\n"):
        k = line.split(":", 1)[0].strip() if ":" in line else ""
        if k in PRESERVE_ON_REWRITE and k in old:
            out.append(old[k])
            seen.add(k)
        else:
            out.append(line)
    for k in PRESERVE_ON_REWRITE:
        if k in old and k not in seen:
            out.append(old[k])
    return "---\n" + "\n".join(out) + "\n---\n" + new_md[m.end():]

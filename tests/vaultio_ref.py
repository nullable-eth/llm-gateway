"""Vault parsing: frontmatter + msg-uuid chunking.

Chunk identity contract (spec): boundary = <!-- msg:uuid --> marker; identity =
(message_uuid, ordinal). Long messages split into continuation chunks under the
same uuid. Non-transcript notes chunk by heading with message_uuid=None and
ordinal = heading sequence. Stored text is verbatim; ONLY the text handed to
the embedder is lightly cleaned (details/fence plumbing stripped).
"""
import hashlib
import re

FM_RE = re.compile(r"^---\n(.*?\n)---\n", re.S)
MSG_RE = re.compile(r"<!-- msg:([0-9a-fA-F-]{8,}) -->")
HEAD_RE = re.compile(r"(?m)^## .+$")
SENDER_RE = re.compile(r"^## (User|Claude) \u00b7 (\S+)")
DETAILS_RE = re.compile(r"</?details>|<summary>.*?</summary>", re.S)
FENCE_INFO_RE = re.compile(r"(?m)^(`{3,})[A-Za-z0-9_-]*\s*$")


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def parse_frontmatter(text: str) -> dict:
    m = FM_RE.match(text)
    if not m:
        return {}
    fm = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            fm[k.strip()] = v.strip().strip('"').strip("'")
    tags = re.search(r"tags:\s*\[(.*?)\]", m.group(1))
    fm["_tags"] = [t.strip() for t in tags.group(1).split(",") if t.strip()] if tags else []
    return fm


def embed_text(raw: str) -> str:
    """Cleaned copy for the embedder only. Stored text stays verbatim."""
    t = DETAILS_RE.sub(" ", raw)
    t = FENCE_INFO_RE.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def _split_long(s: str, limit: int):
    """Split on paragraph boundaries, hard-split only as last resort."""
    if len(s) <= limit:
        return [s]
    out, cur = [], ""
    for para in s.split("\n\n"):
        if cur and len(cur) + len(para) + 2 > limit:
            out.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
        while len(cur) > limit:                       # single huge paragraph
            out.append(cur[:limit])
            cur = cur[limit:]
    if cur:
        out.append(cur)
    return out


def chunk_transcript(body: str, limit: int):
    """Yield (message_uuid, ordinal, sender, created_at, text)."""
    marks = list(MSG_RE.finditer(body))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        seg = body[m.end():end].strip()
        if not seg:
            continue
        sm = SENDER_RE.match(seg)
        sender = sm.group(1) if sm else None
        created = sm.group(2) if sm else None
        for j, piece in enumerate(_split_long(seg, limit)):
            yield m.group(1), j, sender, created, piece


def chunk_note(body: str, limit: int):
    """Non-transcript notes: (None, ordinal, None, None, text) per heading."""
    heads = list(HEAD_RE.finditer(body))
    if not heads:
        segs = [body.strip()] if body.strip() else []
    else:
        segs = []
        if body[:heads[0].start()].strip():
            segs.append(body[:heads[0].start()].strip())
        for i, h in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
            segs.append(body[h.start():end].strip())
    ordinal = 0
    for seg in segs:
        for piece in _split_long(seg, limit):
            yield None, ordinal, None, None, piece
            ordinal += 1

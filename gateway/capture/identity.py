"""Message identity and conversation alignment.

CONVERSATION UUIDS ARE RANDOM, deliberately. An earlier draft derived them
from a fingerprint of the opening messages, which meant two unrelated chats
that happened to start with the same text minted the same uuid, the same
source_uuid and the same filename, and interleaved into one file. Nothing
about a conversation uuid derives from content now, so that whole class is
gone. Continuity is carried by the on-vault index and is never re-derived.

MESSAGE UUIDS ARE DETERMINISTIC, equally deliberately: db.replace_chunks
carries an existing embedding forward only when (message_uuid, ordinal) match
AND the text is byte-identical. Re-sent history therefore converges onto the
same uuids, and re-flushing a reopened conversation re-embeds only the turns
that actually changed. The conversation uuid is the uuid5 namespace, so
identical text in two conversations still yields distinct message uuids —
which matters beyond dedup, since do_context resolves a hit with LIMIT 1 and
would otherwise hand back the wrong file's window.

ALIGNMENT, not set-matching. A client sends a contiguous suffix of the
conversation (trimmed from the front) plus whatever is new at the end. So
there is an offset o into the known history with known[o:] == req[:L]. The
largest such L is the match, and req[L:] is what is new. Occurrence counters
for duplicate messages come from position in the known history rather than
position in the request, so trimming does not shift them.
"""
import hashlib
import json
import uuid


def new_conversation_uuid() -> str:
    return str(uuid.uuid4())


def digest(payload) -> str:
    """Occurrence-free hash of one message payload.

    Alignment only ever tests payloads for equality, so it can run on these
    instead of on the text — which is what lets a flushed conversation stay
    matchable for its whole reopen window while holding no message bodies in
    memory. Must NOT include the occurrence counter: the counter is assigned
    per conversation, and a trimmed request computes a different one for the
    same message.
    """
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()


def key_for(payload, occurrence: int) -> str:
    """Stable content key for one message within one conversation."""
    raw = json.dumps([payload, occurrence], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()


def message_uuid(conversation_uuid: str, key: str) -> str:
    return str(uuid.uuid5(uuid.UUID(conversation_uuid), key))


def align(known: list, req: list, tail_gap: int = 0):
    """Largest (offset, length) with known[o:o+L] == req[:L].

    tail_gap is how far short of the end of `known` the overlap may stop. 0
    means the match must run to the last known message — the prefix-extension
    rule, which is what stops a re-fired identical alert from being adopted
    into the previous run. 1 additionally allows a regenerate, where the
    client re-sends everything except the trailing assistant turn.
    """
    n, m = len(known), len(req)
    if not req or not known:
        return None
    best = None
    for o in range(n):
        length = min(n - o, m)
        if length == 0:
            continue
        if o + length < n - tail_gap:
            continue
        if known[o:o + length] != req[:length]:
            continue
        if best is None or length > best[1]:
            best = (o, length)
    return best


def assign_occurrences(known: list, new: list) -> list:
    """Occurrence counter for each new message: how many identical messages
    already precede it, counting both the known history and earlier entries
    in this same batch. Operates on digests, so it needs no message bodies."""
    counts: dict[str, int] = {}
    for d in known:
        counts[d] = counts.get(d, 0) + 1
    out = []
    for d in new:
        out.append(counts.get(d, 0))
        counts[d] = counts.get(d, 0) + 1
    return out

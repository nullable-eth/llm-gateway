"""Conversation buffer, on-vault index, and transcript writer.

THE BUFFER IS THE POINT. A conversation lives in <capture dir>/.index/<uuid>.json
until it has been quiet for CAPTURE_IDLE_S, and only then becomes markdown.
Both scanner.scan_once and filingloop._candidates glob "*.md", so nothing in
.index/ is ever indexed, embedded, searched or filed — which is what stops an
agent mid-run from retrieving its own half-formed reasoning back out of RAG
and treating it as archived fact. The JSON lives on the same SMB share as the
transcripts, so a pod restart loses nothing that the share itself still has.
"""
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from .. import config, metrics
from . import identity, render
from .normalize import Msg

log = logging.getLogger("capture")

UUID_RE = re.compile(r"(?m)^source_uuid:\s*(\S+)\s*$")
UNLIMITED = 1 << 30


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def capture_dir() -> Path:
    return Path(config.VAULT_ROOT) / config.CAPTURE_DIR


def index_dir() -> Path:
    return capture_dir() / ".index"


def write_atomic(path: Path, text: str) -> None:
    """Temp file beside the target, then replace. Falls back to a direct
    write if the share refuses the rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    try:
        os.replace(tmp, path)
    except OSError:
        path.write_text(text, encoding="utf-8", newline="\n")
        try:
            tmp.unlink()
        except OSError:
            pass


class Conversation:
    def __init__(self, uuid_: str, created: str, client_id: str = ""):
        self.uuid = uuid_
        self.created = created
        self.updated = created
        self.date = created[:10]
        self.title = ""
        self.client_id = client_id
        self.messages: list = []
        self.digests: list = []   # one payload hash per message, always held
        self.keys: list = []
        self.uuids: list = []
        self.path: str | None = None
        self.flushed = False
        self.last_activity = time.time()
        self.opened = time.time()
        self.dirty = False        # index write owed to the vault
        # False once compacted: digests/keys/uuids are intact but the message
        # bodies live only in the index file. Rehydrate before rendering or
        # appending.
        self.hydrated = True

    # ------------------------------------------------------------ identity
    def extend(self, new_msgs: list, ts: str) -> None:
        fresh = [identity.digest(m.payload()) for m in new_msgs]
        occurrences = identity.assign_occurrences(self.digests, fresh)
        for m, d, occ in zip(new_msgs, fresh, occurrences):
            key = identity.key_for(m.payload(), occ)
            if not m.ts:
                m.ts = ts
            self.messages.append(m)
            self.digests.append(d)
            self.keys.append(key)
            self.uuids.append(identity.message_uuid(self.uuid, key))
        self.updated = ts
        self.last_activity = time.time()
        if not self.title and self.messages:
            self.title = render.sanitize(
                render.title_from(self.messages), config.TITLE_MAX)

    # --------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {"uuid": self.uuid, "created": self.created,
                "updated": self.updated, "date": self.date,
                "title": self.title, "client_id": self.client_id,
                "path": self.path, "flushed": self.flushed,
                "last_activity": self.last_activity, "opened": self.opened,
                "keys": self.keys, "uuids": self.uuids,
                "digests": self.digests,
                "messages": [m.as_dict() for m in self.messages]}

    @staticmethod
    def from_dict(d: dict) -> "Conversation":
        c = Conversation(d["uuid"], d.get("created") or now_iso(),
                         d.get("client_id") or "")
        c.updated = d.get("updated") or c.created
        c.date = d.get("date") or c.created[:10]
        c.title = d.get("title") or ""
        c.path = d.get("path")
        c.flushed = bool(d.get("flushed"))
        c.last_activity = float(d.get("last_activity") or time.time())
        c.opened = float(d.get("opened") or c.last_activity)
        c.keys = list(d.get("keys") or [])
        c.uuids = list(d.get("uuids") or [])
        c.messages = [Msg.from_dict(m) for m in d.get("messages") or []]
        c.digests = list(d.get("digests") or [])
        if len(c.digests) != len(c.messages):        # index written pre-digests
            c.digests = [identity.digest(m.payload()) for m in c.messages]
        return c

    def compact(self) -> None:
        """Drop message bodies, keep everything matching needs.

        A flushed conversation stays reopenable for CAPTURE_REOPEN_S — a week
        by default — and holding every message's text for that long is what
        would turn a busy day of long-context chats into hundreds of MB of
        resident memory. Alignment runs on digests, so the bodies are dead
        weight until someone actually continues the conversation.
        """
        self.messages = []
        self.hydrated = False

    def rehydrate(self) -> bool:
        """Reload message bodies from the index file. Needed before rendering
        or appending to a compacted conversation."""
        if self.hydrated:
            return True
        try:
            d = json.loads(self.index_path().read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            log.warning("capture: cannot rehydrate %s: %s", self.uuid, e)
            return False
        self.messages = [Msg.from_dict(m) for m in d.get("messages") or []]
        self.hydrated = True
        return True

    def index_path(self) -> Path:
        return index_dir() / f"{self.uuid}.json"

    def save(self) -> bool:
        """Best-effort. A failed index write is owed, not lost: the messages
        are already in memory and the sweeper retries while `dirty` is set."""
        if not self.hydrated:
            # to_dict() would serialise an empty message list over the index
            # file — the very buffer a reopen reads back. A compacted
            # conversation has no unsaved state by construction, so this is a
            # no-op rather than a failure.
            return True
        try:
            write_atomic(self.index_path(),
                         json.dumps(self.to_dict(), ensure_ascii=False))
        except OSError as e:
            self.dirty = True
            log.warning("capture: index write failed for %s: %s", self.uuid, e)
            return False
        self.dirty = False
        return True


class Store:
    """In-memory registry over the on-vault index. Single-task owned: only
    writer.py touches it, so no locking."""

    def __init__(self):
        self.convs: dict = {}

    # ------------------------------------------------------------- loading
    def load(self) -> int:
        d = index_dir()
        if not d.is_dir():
            return 0
        for f in sorted(d.glob("*.json")):
            try:
                c = Conversation.from_dict(
                    json.loads(f.read_text(encoding="utf-8")))
            except (OSError, ValueError, KeyError, TypeError):
                log.warning("capture: unreadable index %s; leaving it alone",
                            f.name)
                continue
            if c.flushed:
                c.compact()       # matchable, but no bodies resident
            self.convs[c.uuid] = c
        self.prune()
        return len(self.convs)

    def prune(self) -> None:
        """Drop flushed conversations past the reopen window. The transcript
        is the archive by then; the index entry is scaffolding."""
        cutoff = time.time() - config.REOPEN_S
        for u in [u for u, c in self.convs.items()
                  if c.flushed and c.last_activity < cutoff]:
            conv = self.convs.pop(u)
            try:
                conv.index_path().unlink()
            except OSError:
                pass

    # ------------------------------------------------------------ matching
    def match(self, req_digests: list, req_roles: list, client_id: str):
        """(conversation, overlap_length, how) or None. Runs on digests, so a
        compacted conversation matches without loading its bodies."""
        if not req_digests:
            return None
        if client_id:
            for c in self.convs.values():
                if c.client_id and c.client_id == client_id:
                    a = identity.align(c.digests, req_digests, UNLIMITED)
                    return (c, a[1] if a else 0, "client-id")
        best = None
        now = time.time()
        for c in self.convs.values():
            if c.flushed and (now - c.last_activity) > config.REOPEN_S:
                continue
            # Flushed: the overlap must reach the last known message (the
            # prefix-extension rule) so a re-fired identical alert starts its
            # own conversation instead of being adopted into the last run.
            # Open: one message of slack, which is a regenerate.
            a = identity.align(c.digests, req_digests,
                               0 if c.flushed else 1)
            if not a:
                continue
            _, length = a
            if not any(r != "system" for r in req_roles[:length]):
                continue
            if best is None or length > best[1]:
                best = (c, length, "reopen" if c.flushed else "open")
        return best

    # -------------------------------------------------------------- apply
    def apply(self, rec: dict) -> Conversation:
        req = rec.get("messages") or []
        hit = self.match([identity.digest(m.payload()) for m in req],
                         [m.role for m in req], rec.get("client_id") or "")
        conv = overlap = how = None
        if hit:
            candidate, overlap, how = hit
            if candidate.rehydrate():
                candidate.flushed = False
                conv = candidate
            else:
                # The index file is gone or unreadable, so the archived bodies
                # can't be recovered. Appending would write a transcript
                # missing its own history, so start fresh and leave the
                # existing file untouched.
                log.warning("capture: %s unrehydratable; starting a new "
                            "conversation rather than truncating it",
                            candidate.uuid)
        if conv is None:
            conv = Conversation(identity.new_conversation_uuid(), rec["ts"],
                                rec.get("client_id") or "")
            self.convs[conv.uuid] = conv
            overlap, how = 0, "new"
        metrics.ADOPTED.labels(how=how).inc()

        conv.extend(req[overlap:], rec["ts"])
        reply = rec.get("reply")
        if reply is not None:
            conv.extend([reply], rec.get("reply_ts") or rec["ts"])
        # An explicit title only applies while the filename is still unclaimed.
        if rec.get("title_hint") and conv.path is None:
            conv.title = render.sanitize(rec["title_hint"], config.TITLE_MAX)
        conv.save()
        return conv

    # -------------------------------------------------------------- flush
    def due(self, now: float) -> list:
        out = []
        for c in self.convs.values():
            if c.flushed or not c.messages:
                continue
            if (now - c.last_activity >= config.IDLE_S
                    or now - c.opened >= config.MAX_OPEN_S):
                out.append(c)
        return out

    def _source_uuid(self, p: Path):
        try:
            head = p.read_text(encoding="utf-8-sig", errors="replace")[:2000]
        except OSError:
            return None
        m = UUID_RE.search(head)
        return m.group(1) if m else None

    def _find_filed(self, filename: str, conv_uuid: str):
        """A transcript we wrote earlier that the filing agent has since moved
        into a node. Rewriting it there beats leaving a .staging duplicate."""
        root = Path(config.VAULT_ROOT)
        try:
            nodes = [d for d in root.iterdir()
                     if d.is_dir() and not d.name.startswith(".")]
        except OSError:
            return None
        for node in nodes:
            for cand in (node / "Chats" / "Full Transcripts" / filename,
                         node / filename):
                if cand.is_file() and self._source_uuid(cand) == conv_uuid:
                    return cand
        return None

    def target(self, conv: Conversation) -> Path:
        root = Path(config.VAULT_ROOT)
        if conv.path:
            p = root / conv.path
            if p.is_file():
                return p
            found = self._find_filed(Path(conv.path).name, conv.uuid)
            if found:
                return found
        base = f"{conv.date} - {render.sanitize(conv.title or 'untitled', 120)}"
        p = capture_dir() / f"{base}.md"
        if p.is_file() and self._source_uuid(p) != conv.uuid:
            p = capture_dir() / f"{base} ({conv.uuid[:6]}).md"
        return p

    def flush(self, conv: Conversation) -> str:
        """Render and write. Raises OSError so the caller can retry."""
        t0 = time.monotonic()
        if not conv.rehydrate():
            raise OSError(f"cannot rehydrate {conv.uuid} to render it")
        target = self.target(conv)
        md = render.render_transcript(
            {"uuid": conv.uuid, "title": conv.title or "untitled conversation",
             "date": conv.date, "created": conv.created,
             "updated": conv.updated or now_iso()},
            conv.messages, conv.uuids)
        if target.is_file():
            try:
                old = target.read_text(encoding="utf-8-sig", errors="replace")
            except OSError:
                old = ""
            if old:
                md = render.preserve_filed_frontmatter(md, old)
        write_atomic(target, md)
        conv.path = target.relative_to(Path(config.VAULT_ROOT)).as_posix()
        conv.flushed = True
        # Compact only once the index file is safely on disk — that file is
        # what a reopen rehydrates from.
        if conv.save():
            conv.compact()
        metrics.FLUSH_LAT.observe(time.monotonic() - t0)
        return conv.path

"""The only task that touches the vault.

By the time a record reaches this queue the proxied request is already
finished, so nothing here can slow one down — and nothing here may raise into
the request path. Every vault operation is best-effort: a NAS outage parks
work in memory, retries it on the next sweep, and drops the oldest once the
cap is hit. That is a deliberate ceiling, not an oversight; if the NAS is gone
that is its own class of problem.

Filesystem work runs in a thread so a stalled CIFS handle blocks the sweeper,
never the event loop serving the proxy.
"""
import asyncio
import logging
import time

from .. import config, metrics
from . import store

log = logging.getLogger("capture")


class Writer:
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=config.QUEUE_MAX)
        self.store = store.Store()

    # ------------------------------------------------- called by the proxy
    def submit(self, rec: dict) -> None:
        """Never blocks, never raises. Drops the oldest record under pressure
        — a stuck vault must not become backpressure on inference."""
        try:
            self.queue.put_nowait(rec)
        except asyncio.QueueFull:
            metrics.DROPPED.labels(reason="queue_full").inc()
            try:
                self.queue.get_nowait()
                self.queue.task_done()
                self.queue.put_nowait(rec)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass
        metrics.QUEUE_DEPTH.set(self.queue.qsize())

    # ------------------------------------------------------------- tasks
    async def consume(self) -> None:
        try:
            n = await asyncio.to_thread(self.store.load)
            log.info("capture: %d conversation(s) restored from the index", n)
        except Exception:
            log.exception("capture: index load failed; starting empty")
        while True:
            rec = await self.queue.get()
            try:
                await asyncio.to_thread(self.store.apply, rec)
            except Exception:
                log.exception("capture: apply failed; record dropped")
                metrics.DROPPED.labels(reason="apply_error").inc()
            finally:
                self.queue.task_done()
                metrics.QUEUE_DEPTH.set(self.queue.qsize())

    async def sweep(self) -> None:
        while True:
            await asyncio.sleep(config.SWEEP_S)
            try:
                await asyncio.to_thread(self.sweep_once)
            except Exception:
                log.exception("capture: sweep error")

    # ----------------------------------------------------------- internals
    def sweep_once(self, force: bool = False) -> None:
        now = time.time()
        healthy = True
        due = list(self.store.convs.values()) if force else self.store.due(now)
        for conv in due:
            if conv.flushed or not conv.messages:
                continue
            try:
                path = self.store.flush(conv)
            except OSError as e:
                healthy = False
                metrics.FLUSHES.labels(outcome="error").inc()
                log.warning("capture: flush failed for %s: %s", conv.uuid, e)
            else:
                metrics.FLUSHES.labels(outcome="written").inc()
                log.info("capture: wrote %s (%d message(s), %d exchange(s))",
                         path, len(conv.messages),
                         sum(1 for m in conv.messages if m.role == "assistant"))
        dirty = 0
        for conv in list(self.store.convs.values()):
            if conv.dirty:
                if conv.save():
                    continue
                healthy = False
                dirty += 1
        try:
            self.store.prune()
        except Exception:
            log.exception("capture: prune error")
        self._cap()
        metrics.PENDING_RETRY.set(dirty)
        metrics.VAULT_UP.set(1 if healthy else 0)
        metrics.OPEN_CONVS.set(
            sum(1 for c in self.store.convs.values() if not c.flushed))

    def _cap(self) -> None:
        convs = self.store.convs
        excess = len(convs) - config.MAX_CONVERSATIONS
        if excess <= 0:
            return
        oldest = sorted(convs.items(), key=lambda kv: kv[1].last_activity)
        for uuid_, conv in oldest[:excess]:
            convs.pop(uuid_, None)
            if not conv.flushed:
                metrics.DROPPED.labels(reason="conversation_cap").inc()
                log.warning("capture: dropped unflushed conversation %s at cap",
                            uuid_)

    async def drain(self) -> None:
        """Shutdown: apply what is queued, then write every open conversation
        rather than waiting out the idle timer."""
        try:
            await asyncio.wait_for(self.queue.join(), timeout=10)
        except (asyncio.TimeoutError, Exception):
            pass
        try:
            await asyncio.to_thread(self.sweep_once, True)
        except Exception:
            log.exception("capture: drain failed")

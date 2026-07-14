import asyncio

BATCH_SIZE = 64
FLUSH_INTERVAL_SECONDS = 0.5


class Batcher:
    """Accumulates items and flushes to a bounded queue on size or time.

    A batch is flushed when either BATCH_SIZE items have accumulated or
    FLUSH_INTERVAL_SECONDS has elapsed since the first item of the batch
    arrived, whichever comes first.
    """

    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self._batch = []
        self._timer_task: asyncio.Task | None = None

    async def add(self, item) -> None:
        if not self._batch:
            self._timer_task = asyncio.ensure_future(self._delayed_flush())
        self._batch.append(item)
        if len(self._batch) >= BATCH_SIZE:
            await self._flush(cancel_timer=True)

    async def _delayed_flush(self) -> None:
        await asyncio.sleep(FLUSH_INTERVAL_SECONDS)
        await self._flush(cancel_timer=False)

    async def flush(self) -> None:
        """Force-flush whatever is pending, e.g. on shutdown."""
        await self._flush(cancel_timer=True)

    async def _flush(self, cancel_timer: bool) -> None:
        if not self._batch:
            return
        batch, self._batch = self._batch, []
        # cancel_timer=False when called from the timer's own callback,
        # since a task cannot safely cancel itself mid-flush.
        if cancel_timer and self._timer_task is not None and not self._timer_task.done():
            self._timer_task.cancel()
        await self.queue.put(batch)

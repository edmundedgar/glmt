import asyncio
import json
import logging
from pathlib import Path

from ingester.batcher import Batcher
from ingester.jetstream_client import JetstreamClient
from ingester.preprocess import preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
CURSOR_PATH = DATA_DIR / "cursor.txt"
OUTPUT_PATH = DATA_DIR / "posts.jsonl"

# cid is required to build a com.atproto.repo.strongRef when labeling a
# specific post via Ozone's emitEvent (the lexicon marks it required, and
# Ozone validates request bodies strictly -- confirmed empirically, a
# strongRef without cid is rejected with 400 InvalidRequest). Jetstream
# commit events already carry it; previously it was silently discarded.


class CursorTracker:
    """Shared holder for the most recent event's time_us, so the consumer
    can persist a cursor after each batch flush without threading it
    through the (uri, text) batch payload."""

    def __init__(self):
        self.latest_time_us: int | None = None


async def produce(client: JetstreamClient, batcher: Batcher, cursor: CursorTracker) -> None:
    async for event in client.events():
        if event.get("kind") != "commit":
            continue
        commit = event.get("commit", {})
        if commit.get("operation") != "create":
            continue
        if commit.get("collection") != "app.bsky.feed.post":
            continue
        record = commit.get("record") or {}
        text = preprocess(record.get("text", ""))
        if not text:
            continue
        uri = f"at://{event['did']}/app.bsky.feed.post/{commit['rkey']}"
        cid = commit.get("cid")
        cursor.latest_time_us = event["time_us"]
        await batcher.add((uri, text, cid))


async def consume(queue: asyncio.Queue, client: JetstreamClient, cursor: CursorTracker) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "a") as f:
        while True:
            batch = await queue.get()
            for uri, text, cid in batch:
                f.write(json.dumps({"uri": uri, "text": text, "cid": cid}) + "\n")
            f.flush()
            if cursor.latest_time_us is not None:
                client.save_cursor(cursor.latest_time_us)
            log.info("flushed batch of %d posts (queue depth=%d)", len(batch), queue.qsize())


async def main() -> None:
    client = JetstreamClient(cursor_path=CURSOR_PATH, collections=["app.bsky.feed.post"])
    queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    batcher = Batcher(queue)
    cursor = CursorTracker()

    producer = asyncio.create_task(produce(client, batcher, cursor))
    consumer = asyncio.create_task(consume(queue, client, cursor))
    try:
        await asyncio.gather(producer, consumer)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await batcher.flush()


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import urlencode

import websockets
import zstandard as zstd
from websockets.exceptions import ConnectionClosed

log = logging.getLogger(__name__)

JETSTREAM_URL = "wss://jetstream2.us-east.bsky.network/subscribe"
DICTIONARY_PATH = Path(__file__).parent / "assets" / "zstd_dictionary"
REWIND_US = 5_000_000  # 5 second rewind on reconnect, for gapless playback
RECONNECT_BACKOFF_SECONDS = (1, 2, 5, 10, 30)


class JetstreamClient:
    """Consumes the Jetstream firehose and yields decoded commit events.

    Persists the last-seen `time_us` to disk so a restart resumes with a
    small rewind instead of replaying the whole history or dropping posts.
    """

    def __init__(
        self,
        cursor_path: Path,
        collections: list[str],
        url: str = JETSTREAM_URL,
        dictionary_path: Path = DICTIONARY_PATH,
    ):
        self._cursor_path = cursor_path
        self._collections = collections
        self._url = url
        with open(dictionary_path, "rb") as f:
            self._dictionary = zstd.ZstdCompressionDict(f.read())

    def _load_cursor(self) -> int | None:
        if not self._cursor_path.exists():
            return None
        try:
            return int(self._cursor_path.read_text().strip())
        except (ValueError, OSError):
            return None

    def save_cursor(self, time_us: int) -> None:
        tmp_path = self._cursor_path.with_suffix(".tmp")
        tmp_path.write_text(str(time_us))
        tmp_path.replace(self._cursor_path)

    def _build_url(self) -> str:
        params = [("wantedCollections", c) for c in self._collections]
        params.append(("compress", "true"))
        cursor = self._load_cursor()
        if cursor is not None:
            params.append(("cursor", str(cursor - REWIND_US)))
        return f"{self._url}?{urlencode(params)}"

    async def events(self) -> AsyncIterator[dict]:
        """Reconnects indefinitely, yielding decoded event dicts."""
        attempt = 0
        while True:
            url = self._build_url()
            decompressor = zstd.ZstdDecompressor(dict_data=self._dictionary)
            try:
                async with websockets.connect(url, max_size=None) as ws:
                    log.info("connected to jetstream: %s", url)
                    attempt = 0
                    async for raw in ws:
                        decompressed = decompressor.decompress(raw, max_output_size=1_000_000)
                        yield json.loads(decompressed)
            except (ConnectionClosed, OSError) as e:
                delay = RECONNECT_BACKOFF_SECONDS[min(attempt, len(RECONNECT_BACKOFF_SECONDS) - 1)]
                log.warning("jetstream connection lost (%s), reconnecting in %ss", e, delay)
                attempt += 1
                await asyncio.sleep(delay)

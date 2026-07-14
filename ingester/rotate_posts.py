"""Daily rotation for posts.jsonl -- prevents unbounded growth (already
past 1GB) by moving already-processed history out of the live file,
compressed, into data/archive/. Only rotates once
classifier/live_export_labels.py's cursor is caught up to (within
CAUGHT_UP_MARGIN_BYTES of) posts.jsonl's current size -- otherwise skips
this run entirely rather than risk permanently losing posts it hasn't
processed yet (e.g. during an ingester outage like the one this project
already hit once). local_llm_bulk_label.py doesn't track a cursor into
posts.jsonl and won't see archived-off history after rotation -- an
accepted tradeoff (it's a training-data accumulator with plenty of volume
already, not a completeness-critical path), not a bug.

Rotation mechanics: the ingester holds one append-mode file handle for its
whole process lifetime (`open(path, "a")` in ingester/main.py), so
renaming posts.jsonl out from under it does NOT redirect its writes -- it
must be stopped and restarted to open a fresh file at the canonical path.
live_export_labels.py needs no special handling for this: its byte-offset
cursor already detects a shrunk file (the fresh posts.jsonl starts at 0,
smaller than its old cursor) and resyncs automatically -- this script
relies on that behavior rather than duplicating it.

Usage (intended for a daily cron job):
    python -m ingester.rotate_posts
    python -m ingester.rotate_posts --dry-run
"""

import argparse
import gzip
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
POSTS_PATH = DATA_DIR / "posts.jsonl"
ROTATING_PATH = DATA_DIR / "posts.jsonl.rotating"
LIVE_EXPORT_CURSOR_PATH = DATA_DIR / "live_export_cursor.txt"
ARCHIVE_DIR = DATA_DIR / "archive"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
INGESTER_LOG_PATH = DATA_DIR / "ingester.log"

# How close live_export_labels.py's cursor must be to posts.jsonl's current
# size to count as "caught up". Generous on purpose -- a few thousand lines
# of normal steady-state lag is fine to rotate past; a real backlog (an
# outage, a multi-hour gap) will be far larger than this and correctly
# blocks rotation.
CAUGHT_UP_MARGIN_BYTES = 20 * 1024 * 1024  # 20MB, ~80-100k lines at typical post size

RETENTION_DAYS = 14

STOP_WAIT_SECONDS = 10
STOP_POLL_INTERVAL = 0.5


def is_caught_up() -> tuple[bool, str]:
    if not POSTS_PATH.exists():
        return False, "posts.jsonl doesn't exist"
    if not LIVE_EXPORT_CURSOR_PATH.exists():
        return False, "no live_export_cursor.txt yet"
    try:
        cursor = int(LIVE_EXPORT_CURSOR_PATH.read_text().strip())
    except ValueError:
        return False, "live_export_cursor.txt is malformed"

    size = POSTS_PATH.stat().st_size
    lag = size - cursor
    if lag > CAUGHT_UP_MARGIN_BYTES:
        return False, f"classifier is {lag / 1024 / 1024:.1f}MB behind (cursor={cursor}, size={size})"
    return True, f"caught up (cursor={cursor}, size={size}, lag={lag} bytes)"


def stop_ingester() -> None:
    subprocess.run(["pkill", "-f", "ingester.main"])
    deadline = time.monotonic() + STOP_WAIT_SECONDS
    while time.monotonic() < deadline:
        result = subprocess.run(["pgrep", "-f", "ingester.main"], capture_output=True)
        if result.returncode != 0:  # pgrep found nothing -- process is gone
            return
        time.sleep(STOP_POLL_INTERVAL)
    raise RuntimeError("ingester didn't stop within timeout -- aborting rotation, nothing was renamed")


def start_ingester() -> None:
    log_f = open(INGESTER_LOG_PATH, "a")
    subprocess.Popen(
        [str(VENV_PYTHON), "-m", "ingester.main"],
        cwd=REPO_ROOT,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def archive_rotated_file() -> Path:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = ARCHIVE_DIR / f"posts-{stamp}.jsonl.gz"
    with open(ROTATING_PATH, "rb") as f_in, gzip.open(archive_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    ROTATING_PATH.unlink()
    return archive_path


def prune_old_archives() -> list[Path]:
    if not ARCHIVE_DIR.exists():
        return []
    cutoff = time.time() - RETENTION_DAYS * 86400
    pruned = []
    for path in ARCHIVE_DIR.glob("posts-*.jsonl.gz"):
        if path.stat().st_mtime < cutoff:
            path.unlink()
            pruned.append(path)
    return pruned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    caught_up, reason = is_caught_up()
    print(reason)
    if not caught_up:
        print("skipping rotation this run")
        return

    old_size_mb = POSTS_PATH.stat().st_size / 1024 / 1024
    print(f"posts.jsonl is {old_size_mb:.1f}MB -- rotating")

    if args.dry_run:
        print("--dry-run: would stop ingester, rotate, restart ingester, archive, prune. Stopping here.")
        return

    print("stopping ingester...")
    stop_ingester()

    print("renaming posts.jsonl aside...")
    POSTS_PATH.rename(ROTATING_PATH)

    print("restarting ingester (fresh posts.jsonl, resumes firehose from its own saved cursor)...")
    start_ingester()

    print("compressing rotated file into data/archive/...")
    archive_path = archive_rotated_file()
    print(f"archived to {archive_path}")

    pruned = prune_old_archives()
    if pruned:
        print(f"pruned {len(pruned)} archive(s) older than {RETENTION_DAYS} days")

    print("done")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"rotation failed: {e}", file=sys.stderr)
        raise

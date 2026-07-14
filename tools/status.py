"""One-shot status report for the whole pipeline: what's running, what's
queued/backlogged, and where every cursor currently sits. Meant for a
quick glance ("is everything actually working right now") rather than
continuous monitoring -- run it whenever, no daemon involved.

Deliberately stdlib-only (subprocess/pathlib/datetime), no new
dependency, so it's always available even if the venv is in a weird state.

Usage:
    python -m tools.status
"""

import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
LABELER_DIR = REPO_ROOT / "labeler"

POSTS_PATH = DATA_DIR / "posts.jsonl"
INGESTER_CURSOR_PATH = DATA_DIR / "cursor.txt"
LIVE_EXPORT_CURSOR_PATH = DATA_DIR / "live_export_cursor.txt"
BULK_LABELED_PATH = DATA_DIR / "local_llm_bulk_labeled.jsonl"
ARCHIVE_DIR = DATA_DIR / "archive"
ROTATE_LOG_PATH = DATA_DIR / "rotate_posts.log"

PENDING_LABELS_PATH = LABELER_DIR / "pending-labels.jsonl"
INGEST_OFFSET_PATH = LABELER_DIR / ".ingest-offset"
LABELS_DB_PATH = LABELER_DIR / "labels.db"

CAUGHT_UP_MARGIN_BYTES = 20 * 1024 * 1024  # matches ingester/rotate_posts.py
STALL_WARNING_SECONDS = 5 * 60


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def human_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{seconds}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h{minutes}m"
    days, hours = divmod(hours, 24)
    return f"{days}d{hours}h"


def ago(timestamp: float) -> str:
    return f"{human_duration(datetime.now(timezone.utc).timestamp() - timestamp)} ago"


def find_process(pattern: str, exe_names: tuple[str, ...] = ("python", "node")) -> dict | None:
    """pgrep -f matches against the FULL command line, which can spuriously
    match wrapper shells that merely mention the pattern in their own
    invocation text. Filter down to processes whose actual binary (via
    /proc/pid/exe, not `ps -o comm=`) looks like the interpreter we
    expect. `comm` isn't reliable here -- observed a live node process
    reporting comm "MainThread" instead of "node" (something in its
    dependency stack calls prctl/pthread_setname_np and overwrites it),
    which silently broke an earlier version of this check."""
    result = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    for pid in (p for p in result.stdout.split() if p):
        try:
            exe = Path(f"/proc/{pid}/exe").resolve().name
        except OSError:
            continue  # process exited between pgrep and here
        if any(name in exe for name in exe_names):
            etimes = subprocess.run(["ps", "-o", "etimes=", "-p", pid], capture_output=True, text=True).stdout.strip()
            try:
                uptime_seconds = int(etimes)
            except ValueError:
                uptime_seconds = None
            return {"pid": pid, "uptime_seconds": uptime_seconds}
    return None


def systemd_status(unit: str) -> str | None:
    result = subprocess.run(["systemctl", "is-active", unit], capture_output=True, text=True)
    if result.returncode != 0 and not result.stdout.strip():
        return None  # systemctl unavailable or unit unknown entirely
    return result.stdout.strip()


def line(label: str, value: str, indent: int = 2) -> None:
    print(f"{' ' * indent}{label:<28} {value}")


def section(title: str) -> None:
    print(f"\n-- {title} --")


def report_processes() -> None:
    section("Processes")
    checks = [
        ("Ingester (firehose -> posts.jsonl)", "ingester.main", ("python",)),
        ("Live classifier export", "classifier.live_export_labels", ("python",)),
        ("Bulk labeler (Ollama)", "classifier.local_llm_bulk_label", ("python",)),
        ("Labeler server", "server.mjs", ("node",)),
    ]
    for label, pattern, comms in checks:
        proc = find_process(pattern, comms)
        if proc is None:
            line(label, "NOT RUNNING", indent=2)
        else:
            uptime = human_duration(proc["uptime_seconds"]) if proc["uptime_seconds"] is not None else "?"
            line(label, f"running, pid {proc['pid']}, up {uptime}", indent=2)

    tunnel = systemd_status("labeler-tunnel.service")
    if tunnel is not None:
        line("SSH tunnel (labeler-tunnel)", tunnel, indent=2)


def report_ingester() -> None:
    section("Ingester")
    if not INGESTER_CURSOR_PATH.exists():
        line("cursor", "no cursor.txt yet", indent=2)
        return
    try:
        time_us = int(INGESTER_CURSOR_PATH.read_text().strip())
    except ValueError:
        line("cursor", "malformed cursor.txt", indent=2)
        return
    cursor_dt = datetime.fromtimestamp(time_us / 1_000_000, tz=timezone.utc)
    lag = datetime.now(timezone.utc).timestamp() - time_us / 1_000_000
    line("firehose cursor", cursor_dt.strftime("%Y-%m-%d %H:%M:%S UTC"), indent=2)
    line("lag behind real time", human_duration(max(lag, 0)), indent=2)
    if POSTS_PATH.exists():
        stat = POSTS_PATH.stat()
        line("posts.jsonl", f"{human_bytes(stat.st_size)}, last write {ago(stat.st_mtime)}", indent=2)
    else:
        line("posts.jsonl", "does not exist", indent=2)


def report_backlog(title: str, cursor_path: Path, target_path: Path) -> None:
    section(title)
    if not cursor_path.exists():
        line("cursor", "no cursor file yet", indent=2)
        return
    try:
        offset = int(cursor_path.read_text().strip())
    except ValueError:
        line("cursor", "malformed cursor file", indent=2)
        return
    if not target_path.exists():
        line("target file", "does not exist", indent=2)
        return
    size = target_path.stat().st_size
    backlog = size - offset
    line("cursor (byte offset)", f"{offset:,}", indent=2)
    line("target file size", f"{size:,} ({human_bytes(size)})", indent=2)
    if backlog <= CAUGHT_UP_MARGIN_BYTES:
        status = "caught up" if backlog >= 0 else "cursor ahead of file? (check for a recent rotation)"
    else:
        status = f"BEHIND by {human_bytes(backlog)}"
    line("status", status, indent=2)


def report_bulk_labeler() -> None:
    section("Bulk labeler output (local_llm_bulk_labeled.jsonl)")
    if not BULK_LABELED_PATH.exists():
        line("rows", "file does not exist yet", indent=2)
        return
    stat = BULK_LABELED_PATH.stat()
    result = subprocess.run(["wc", "-l", str(BULK_LABELED_PATH)], capture_output=True, text=True)
    row_count = result.stdout.split()[0] if result.stdout else "?"
    line("rows", row_count, indent=2)
    staleness = datetime.now(timezone.utc).timestamp() - stat.st_mtime
    freshness = f"last write {ago(stat.st_mtime)}"
    if staleness > STALL_WARNING_SECONDS and find_process("classifier.local_llm_bulk_label", ("python",)):
        freshness += "  ⚠ process is running but hasn't written anything in a while -- possibly stalled"
    line("freshness", freshness, indent=2)


def report_rotation() -> None:
    section("posts.jsonl rotation")
    cron = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    installed = "rotate_posts" in cron.stdout
    line("cron job installed", "yes" if installed else "NO", indent=2)

    if ARCHIVE_DIR.exists():
        archives = sorted(ARCHIVE_DIR.glob("posts-*.jsonl.gz"))
        if archives:
            newest = max(archives, key=lambda p: p.stat().st_mtime)
            total_size = sum(p.stat().st_size for p in archives)
            line("archived segments", f"{len(archives)}, {human_bytes(total_size)} total", indent=2)
            line("last rotation", ago(newest.stat().st_mtime), indent=2)
        else:
            line("archived segments", "none yet", indent=2)
    else:
        line("archived segments", "none yet (no archive/ dir)", indent=2)

    if ROTATE_LOG_PATH.exists():
        tail = ROTATE_LOG_PATH.read_text().strip().splitlines()[-2:]
        for l in tail:
            line("last log line", l, indent=2)


def report_gpu() -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return
    section("GPU")
    used, total, util = (x.strip() for x in result.stdout.strip().split(","))
    line("memory", f"{used}MiB / {total}MiB", indent=2)
    line("utilization", f"{util}%", indent=2)


def main() -> None:
    print(f"=== labeler pipeline status ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}) ===")
    report_processes()
    report_ingester()
    report_backlog("Live classifier export (posts.jsonl -> pending-labels.jsonl)", LIVE_EXPORT_CURSOR_PATH, POSTS_PATH)
    report_backlog("Labeler server ingest (pending-labels.jsonl -> labels.db)", INGEST_OFFSET_PATH, PENDING_LABELS_PATH)
    report_bulk_labeler()
    report_rotation()
    report_gpu()
    print()


if __name__ == "__main__":
    main()

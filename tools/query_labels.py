"""Query labeler/labels.db -- the actual label store (@skyware/labeler's
own SQLite, see docs/NOTES.md's "Labeler server" section). Postgres is
installed on this box but unused; nothing lives there.

Usage:
    python -m tools.query_labels                        # label frequency summary, all time
    python -m tools.query_labels --since 24              # summary, last 24h only
    python -m tools.query_labels --uri "at://did:plc:.../app.bsky.feed.post/..."
    python -m tools.query_labels --label uspol            # most recent posts labeled uspol
    python -m tools.query_labels --label uspol --recent 50 --since 6
"""

import argparse
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "labeler" / "labels.db"


def connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise SystemExit(f"{DB_PATH} doesn't exist -- has the labeler server run yet?")
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def cutoff_str(hours: float) -> str:
    """cts is stored as '2026-07-14T09:00:12.190Z' -- SQLite's own
    datetime('now', ...) produces a differently-formatted string
    ('2026-07-14 09:00:15', space-separated, no millis/Z) that silently
    fails to compare correctly against it as text (confirmed empirically:
    a naive `cts >= datetime('now', '-1 hours')` returned the same count
    for a 1-hour and a 24-hour window). Building the cutoff in the same
    format as the stored column sidesteps the whole problem."""
    dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def to_bsky_link(uri: str) -> str | None:
    """at://{did}/app.bsky.feed.post/{rkey} -> https://bsky.app/profile/{did}/post/{rkey}.
    The DID works directly in the URL, no handle resolution needed.
    Returns None for anything that isn't a feed-post URI (nothing else
    lands in this table today, but don't guess a URL shape for a
    collection this wasn't written for)."""
    if not uri.startswith("at://"):
        return None
    parts = uri[len("at://") :].split("/")
    if len(parts) != 3 or parts[1] != "app.bsky.feed.post":
        return None
    did, _, rkey = parts
    return f"https://bsky.app/profile/{did}/post/{rkey}"


def show_summary(conn: sqlite3.Connection, since_hours: float | None) -> None:
    where, params = "", ()
    if since_hours is not None:
        where, params = "WHERE cts >= ?", (cutoff_str(since_hours),)
    rows = conn.execute(f"SELECT val, COUNT(*) FROM labels {where} GROUP BY val ORDER BY COUNT(*) DESC", params).fetchall()
    window = "all time" if since_hours is None else f"last {since_hours}h"
    print(f"label counts ({window}):")
    for val, count in rows:
        print(f"  {count:>8}  {val}")
    print(f"  {sum(c for _, c in rows):>8}  TOTAL")


def show_for_uri(conn: sqlite3.Connection, uri: str) -> None:
    rows = conn.execute("SELECT val, neg, cts FROM labels WHERE uri = ? ORDER BY cts", (uri,)).fetchall()
    if not rows:
        print(f"no labels found for {uri}")
        return
    link = to_bsky_link(uri)
    print(f"labels for {uri}:" + (f"\n  ({link})" if link else ""))
    for val, neg, cts in rows:
        print(f"  {'NEGATED ' if neg else ''}{val}  ({cts})")


def show_recent_for_label(conn: sqlite3.Connection, label: str, limit: int, since_hours: float | None) -> None:
    where, params = "WHERE val = ?", [label]
    if since_hours is not None:
        where += " AND cts >= ?"
        params.append(cutoff_str(since_hours))
    rows = conn.execute(f"SELECT uri, cts FROM labels {where} ORDER BY id DESC LIMIT ?", (*params, limit)).fetchall()
    if not rows:
        print(f"no labels found for {label!r}")
        return
    print(f"{len(rows)} most recent {label!r} label(s):")
    for uri, cts in rows:
        link = to_bsky_link(uri)
        print(f"  {cts}  {link or uri}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uri", help="show all labels for a specific post URI")
    parser.add_argument("--label", help="show recent posts with this label instead of the summary")
    parser.add_argument("--recent", type=int, default=20, help="rows to show with --label (default 20)")
    parser.add_argument("--since", type=float, default=None, help="only consider labels from the last N hours")
    args = parser.parse_args()

    conn = connect()
    if args.uri:
        show_for_uri(conn, args.uri)
    elif args.label:
        show_recent_for_label(conn, args.label, args.recent, args.since)
    else:
        show_summary(conn, args.since)


if __name__ == "__main__":
    main()

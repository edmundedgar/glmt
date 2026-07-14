"""Query Ozone's Postgres `label` table -- the actual label store since the
migration off @skyware/labeler (see docs/NOTES.md's "Migrating from
@skyware/labeler to Ozone" section). Reads the connection string straight
out of Ozone's own .env (outside this repo, at
/home/glmt/ozone-src/services/ozone/.env) rather than duplicating
credentials in a second place.

Usage:
    python -m tools.query_labels                        # label frequency summary, all time
    python -m tools.query_labels --since 24              # summary, last 24h only
    python -m tools.query_labels --uri "at://did:plc:.../app.bsky.feed.post/..."
    python -m tools.query_labels --label uspol            # most recent posts labeled uspol
    python -m tools.query_labels --label uspol --recent 50 --since 6
"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg

OZONE_ENV_PATH = Path("/home/glmt/ozone-src/services/ozone/.env")


def read_db_url() -> str:
    if not OZONE_ENV_PATH.exists():
        raise SystemExit(f"{OZONE_ENV_PATH} doesn't exist -- is Ozone set up? See docs/RUNBOOK.md section 9.")
    for line in OZONE_ENV_PATH.read_text().splitlines():
        if line.startswith("OZONE_DB_POSTGRES_URL="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"OZONE_DB_POSTGRES_URL not found in {OZONE_ENV_PATH}")


def connect() -> psycopg.Connection:
    return psycopg.connect(read_db_url())


def cutoff_str(hours: float) -> str:
    """cts is stored as '2026-07-14T09:00:12.190Z' (character varying, not
    a real timestamp type) -- match that exact format so a plain text >=
    comparison works. See the SQLite version of this script (git history)
    for why this matters: a naively-formatted cutoff can silently compare
    wrong and make every window return the same (wrong) count."""
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


def show_summary(conn: psycopg.Connection, since_hours: float | None) -> None:
    where, params = "", ()
    if since_hours is not None:
        where, params = "WHERE cts >= %s", (cutoff_str(since_hours),)
    rows = conn.execute(f"SELECT val, COUNT(*) FROM label {where} GROUP BY val ORDER BY COUNT(*) DESC", params).fetchall()
    window = "all time" if since_hours is None else f"last {since_hours}h"
    print(f"label counts ({window}):")
    for val, count in rows:
        print(f"  {count:>8}  {val}")
    print(f"  {sum(c for _, c in rows):>8}  TOTAL")


def show_for_uri(conn: psycopg.Connection, uri: str) -> None:
    rows = conn.execute("SELECT val, neg, cts FROM label WHERE uri = %s ORDER BY cts", (uri,)).fetchall()
    if not rows:
        print(f"no labels found for {uri}")
        return
    link = to_bsky_link(uri)
    print(f"labels for {uri}:" + (f"\n  ({link})" if link else ""))
    for val, neg, cts in rows:
        print(f"  {'NEGATED ' if neg else ''}{val}  ({cts})")


def show_recent_for_label(conn: psycopg.Connection, label: str, limit: int, since_hours: float | None) -> None:
    where, params = "WHERE val = %s", [label]
    if since_hours is not None:
        where += " AND cts >= %s"
        params.append(cutoff_str(since_hours))
    rows = conn.execute(f"SELECT uri, cts FROM label {where} ORDER BY id DESC LIMIT %s", (*params, limit)).fetchall()
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

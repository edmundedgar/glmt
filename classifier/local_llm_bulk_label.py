"""Resumable bulk labeler: run the local Ollama model over posts.jsonl for
as long as you want (any time the GPU is free), stop it whenever, and pick
up later from where it left off. Accumulates into one growing output file
that classifier/train.py can point at directly for periodic retraining.

Unlike local_llm_topic_list_label.py (fixed-sample A/B experiments,
overwrites its output each run), this script:
  - Skips URIs already present in the output file
  - Appends rather than overwrites
  - Has no required --limit -- runs until posts.jsonl is exhausted or killed
  - Reports cumulative totals across all past runs, not just this session

Usage:
    python -m classifier.local_llm_bulk_label                  # run until stopped
    python -m classifier.local_llm_bulk_label --limit 500      # cap new posts this run
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from classifier.local_llm_topic_list_label import (
    DATA_PATH,
    DEFAULT_MODEL,
    KNOWN_TOPICS,
    classify,
)

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "local_llm_bulk_labeled.jsonl"


def load_already_labeled_uris(path: Path) -> set[str]:
    """Tolerates a truncated/malformed final line -- if the process was
    killed mid-write last time, skip that line rather than crash on resume
    (the corresponding post just gets relabeled, which is harmless)."""
    if not path.exists():
        return set()
    uris = set()
    skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                uris.add(json.loads(line)["uri"])
            except json.JSONDecodeError:
                skipped += 1
    if skipped:
        print(f"warning: skipped {skipped} malformed line(s) in {path} (likely truncated by a kill mid-write)")
    return uris


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="max NEW posts to label this run; default unbounded")
    parser.add_argument("--no-think", type=lambda s: s.lower() != "false", default=True)
    args = parser.parse_args()

    done_uris = load_already_labeled_uris(OUTPUT_PATH)
    print(f"already labeled: {len(done_uris)} posts (resuming)")
    print(f"model={args.model} no_think={args.no_think} topics={len(KNOWN_TOPICS)}\n")

    label_counts = Counter()
    total_time = 0.0
    errors = 0
    processed_this_run = 0

    out_f = open(OUTPUT_PATH, "a")
    with open(DATA_PATH) as posts_f:
        for line in posts_f:
            if args.limit is not None and processed_this_run >= args.limit:
                break
            post = json.loads(line)
            if post["uri"] in done_uris:
                continue

            try:
                labels, elapsed, thinking = classify(args.model, post["text"], args.no_think)
            except Exception as e:
                print(f"error on {post['uri']}: {e}", flush=True)
                errors += 1
                continue

            total_time += elapsed
            processed_this_run += 1
            label_counts.update(labels)
            row = {"uri": post["uri"], "text": post["text"], "labels": labels, "thinking": thinking}
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            done_uris.add(post["uri"])

            if processed_this_run % 10 == 0:
                print(
                    f"  {processed_this_run} new this run "
                    f"({len(done_uris)} total accumulated), "
                    f"avg {total_time / processed_this_run:.2f}s/post",
                    flush=True,
                )
    out_f.close()

    print(f"\n{'=' * 60}")
    print(f"this run: {processed_this_run} new posts labeled, {errors} errors")
    if processed_this_run:
        print(f"avg {total_time / processed_this_run:.2f}s/post")
    print(f"cumulative total in {OUTPUT_PATH}: {len(done_uris)} posts")
    print("\nlabel frequency this run:")
    for label, count in label_counts.most_common():
        print(f"  {count:4d}  {label}")


if __name__ == "__main__":
    main()

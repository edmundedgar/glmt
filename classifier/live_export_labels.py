"""Continuously classify new posts as they're appended to posts.jsonl by
the ingester, appending (uri, label, confidence) rows to
labeler/pending-labels.jsonl as they're found. Resumable -- tracks how
many lines of posts.jsonl have already been processed in a cursor file,
same pattern as ingester/main.py's own cursor.

Usage:
    python -m classifier.live_export_labels
"""

import json
import time
from pathlib import Path

import torch

from classifier.model import MAX_TOKENS, TopicClassifier, StackedTopicHeads, load_pretrained

DATA_PATH = Path(__file__).parent.parent / "data" / "posts.jsonl"
WEIGHTS_DIR = Path(__file__).parent / "weights"
OUTPUT_PATH = Path(__file__).parent.parent / "labeler" / "pending-labels.jsonl"
CURSOR_PATH = Path(__file__).parent.parent / "data" / "live_export_cursor.txt"

TOPICS = ["uspol", "sports", "music", "donald-trump", "gaming", "mental-health", "technology"]
THRESHOLD = 0.5
BATCH_SIZE = 64
POLL_INTERVAL_SECONDS = 10
# Cap how many new lines get read (and held in memory) per cycle. Without
# this, a large backlog (e.g. the ingester running unattended for hours
# while this was stopped) gets read as ONE unbounded batch -- multi-million
# posts held in memory at once, no cursor save or output until the whole
# sweep finishes. This is the same class of bug that OOM-crashed
# labeler/server.mjs from unbounded growth over a long run.
MAX_LINES_PER_CYCLE = 5000


def load_cursor() -> int:
    if not CURSOR_PATH.exists():
        return 0
    try:
        return int(CURSOR_PATH.read_text().strip())
    except ValueError:
        return 0


def save_cursor(n: int) -> None:
    tmp = CURSOR_PATH.with_suffix(".tmp")
    tmp.write_text(str(n))
    tmp.replace(CURSOR_PATH)


def read_new_posts(path: Path, start_line: int, max_lines: int) -> tuple[list[dict], int]:
    posts = []
    line_count = start_line
    with open(path) as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            if len(posts) >= max_lines:
                break
            posts.append(json.loads(line))
            line_count = i + 1
    return posts, line_count


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    encoder, tokenizer, _ = load_pretrained()
    encoder.to(device)
    encoder.eval()

    classifier = TopicClassifier(encoder, TOPICS).to(device)
    for topic in TOPICS:
        state = torch.load(WEIGHTS_DIR / f"{topic}.pt", map_location=device)
        classifier.heads[topic].load_state_dict(state)
    stacked = StackedTopicHeads(dict(classifier.heads)).to(device)
    stacked.eval()

    cursor = load_cursor()
    print(f"starting from line {cursor}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    while True:
        posts, new_cursor = read_new_posts(DATA_PATH, cursor, MAX_LINES_PER_CYCLE)
        if not posts:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        found = 0
        with open(OUTPUT_PATH, "a") as out_f, torch.no_grad():
            for i in range(0, len(posts), BATCH_SIZE):
                batch = posts[i : i + BATCH_SIZE]
                texts = [p["text"] for p in batch]
                inputs = tokenizer(
                    texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS
                ).to(device)
                embeddings = classifier.embed(inputs["input_ids"], inputs["attention_mask"])
                probs = torch.sigmoid(stacked(embeddings))

                for j, post in enumerate(batch):
                    for k, topic in enumerate(TOPICS):
                        if probs[j, k].item() > THRESHOLD:
                            row = {"uri": post["uri"], "label": topic, "confidence": round(probs[j, k].item(), 3)}
                            out_f.write(json.dumps(row) + "\n")
                            out_f.flush()
                            found += 1

        cursor = new_cursor
        save_cursor(cursor)
        print(f"processed {len(posts)} new posts (line {cursor}), found {found} labels", flush=True)

        # A full chunk likely means there's more backlog waiting -- keep
        # going immediately rather than sleeping, so catch-up after a long
        # gap doesn't take forever. Only idle once a read comes back
        # short, meaning we've reached the live tip of posts.jsonl.
        if len(posts) < MAX_LINES_PER_CYCLE:
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()

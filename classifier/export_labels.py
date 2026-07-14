"""Classify a sample of real posts with our trained heads and export
(uri, label) pairs for the labeler server to ingest via createLabel(s).

Skips "death" -- registered in the labeler's label taxonomy but we never
actually trained a classifier for it.

Usage:
    python -m classifier.export_labels --limit 2000
"""

import argparse
import json
from pathlib import Path

import torch

from classifier.model import MAX_TOKENS, TopicClassifier, StackedTopicHeads, load_pretrained

DATA_PATH = Path(__file__).parent.parent / "data" / "posts.jsonl"
WEIGHTS_DIR = Path(__file__).parent / "weights"
OUTPUT_PATH = Path(__file__).parent.parent / "labeler" / "pending-labels.jsonl"

# Topics we have actual trained (F1 > 0.7) heads for -- see classifier/weights/.
# "death" is registered in the labeler's taxonomy but has no trained classifier.
TOPICS = ["uspol", "sports", "music", "donald-trump", "gaming", "mental-health", "technology"]
THRESHOLD = 0.5


def load_sample(path: Path, limit: int, stride: int) -> list[dict]:
    posts = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i % stride != 0:
                continue
            posts.append(json.loads(line))
            if len(posts) >= limit:
                break
    return posts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--stride", type=int, default=300, help="sample every Nth line for variety across the file")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    posts = load_sample(DATA_PATH, args.limit, args.stride)
    print(f"loaded {len(posts)} posts (every {args.stride}th line)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, tokenizer, _ = load_pretrained()
    encoder.to(device)
    encoder.eval()

    classifier = TopicClassifier(encoder, TOPICS).to(device)
    for topic in TOPICS:
        state = torch.load(WEIGHTS_DIR / f"{topic}.pt", map_location=device)
        classifier.heads[topic].load_state_dict(state)
    stacked = StackedTopicHeads(dict(classifier.heads)).to(device)
    stacked.eval()

    rows = []
    with torch.no_grad():
        for i in range(0, len(posts), args.batch_size):
            batch = posts[i : i + args.batch_size]
            texts = [p["text"] for p in batch]
            inputs = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS
            ).to(device)
            embeddings = classifier.embed(inputs["input_ids"], inputs["attention_mask"])
            probs = torch.sigmoid(stacked(embeddings))

            for j, post in enumerate(batch):
                for k, topic in enumerate(TOPICS):
                    if probs[j, k].item() > THRESHOLD:
                        rows.append({"uri": post["uri"], "label": topic, "confidence": round(probs[j, k].item(), 3)})

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    from collections import Counter

    counts = Counter(r["label"] for r in rows)
    print(f"\n{len(rows)} (post, label) pairs across {len(posts)} posts:")
    for label, count in counts.most_common():
        print(f"  {count:4d}  {label}")
    print(f"\nwritten to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

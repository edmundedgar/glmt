"""Group real posts by cardiffnlp's top predicted topic, full text, for
eyeballing what's actually in the firehose sample before picking custom
topics to train.
"""

import json
from collections import defaultdict
from pathlib import Path

import torch

from classifier.model import MAX_TOKENS, load_pretrained

DATA_PATH = Path(__file__).parent.parent / "data" / "posts.jsonl"
NUM_SAMPLES = 80


def load_posts(path: Path, limit: int) -> list[dict]:
    posts = []
    with open(path) as f:
        for line in f:
            posts.append(json.loads(line))
            if len(posts) >= limit:
                break
    return posts


def main() -> None:
    posts = load_posts(DATA_PATH, NUM_SAMPLES)
    texts = [p["text"] for p in posts]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, tokenizer, pretrained_model = load_pretrained()
    pretrained_model.to(device)

    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS
    ).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(pretrained_model(**inputs).logits)

    id2label = pretrained_model.config.id2label
    groups = defaultdict(list)
    for post, prob_row in zip(posts, probs):
        top_idx = int(torch.argmax(prob_row))
        groups[id2label[top_idx]].append((post["text"], float(prob_row[top_idx])))

    for label, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"\n## {label} ({len(items)})")
        for text, score in items:
            clean = text.replace("\n", " ")
            print(f"- ({score:.2f}) {clean}")


if __name__ == "__main__":
    main()

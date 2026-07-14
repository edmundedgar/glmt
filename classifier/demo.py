"""Run the classifier in its untrained state against real firehose posts.

Shows two things side by side:
1. cardiffnlp's own pretrained 19-topic head -- a baseline for how good the
   frozen encoder's representation already is, out of the box.
2. Our custom per-topic heads (per the spec's architecture), which start
   randomly initialized and have seen zero training examples yet.
"""

import json
import sys
from pathlib import Path

import torch

from classifier.model import MAX_TOKENS, TopicClassifier, load_pretrained

DATA_PATH = Path(__file__).parent.parent / "data" / "posts.jsonl"
DEMO_TOPICS = ["uspol", "pets"]
NUM_SAMPLES = 20


def load_posts(path: Path, limit: int) -> list[dict]:
    posts = []
    with open(path) as f:
        for line in f:
            posts.append(json.loads(line))
            if len(posts) >= limit:
                break
    return posts


def main() -> None:
    if not DATA_PATH.exists():
        print(f"No data at {DATA_PATH}. Run `python -m ingester.main` first.", file=sys.stderr)
        sys.exit(1)

    posts = load_posts(DATA_PATH, NUM_SAMPLES)
    texts = [p["text"] for p in posts]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    encoder, tokenizer, pretrained_model = load_pretrained()
    encoder.to(device)
    pretrained_model.to(device)

    classifier = TopicClassifier(encoder, DEMO_TOPICS).to(device)

    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS
    ).to(device)

    with torch.no_grad():
        pretrained_logits = pretrained_model(**inputs).logits
        pretrained_probs = torch.sigmoid(pretrained_logits)

    id2label = pretrained_model.config.id2label

    print(f"\n{'='*100}")
    print("1. PRETRAINED baseline (cardiffnlp's own 19-topic head, already trained)")
    print(f"{'='*100}")
    for i, post in enumerate(posts):
        probs = pretrained_probs[i]
        top_idx = torch.topk(probs, k=3).indices.tolist()
        top_labels = [f"{id2label[j]}={probs[j]:.2f}" for j in top_idx if probs[j] > 0.1]
        text_preview = post["text"].replace("\n", " ")[:80]
        print(f"  [{text_preview!r}]")
        print(f"    top labels: {top_labels or 'none > 0.1'}")

    print(f"\n{'='*100}")
    print(f"2. UNTRAINED custom heads {DEMO_TOPICS} (fresh nn.Linear, zero training examples seen)")
    print(f"{'='*100}")
    for topic in DEMO_TOPICS:
        with torch.no_grad():
            logits = classifier(inputs["input_ids"], inputs["attention_mask"], topic)
            probs = torch.sigmoid(logits)
        print(f"  topic={topic!r}")
        print(f"    probs: {[round(p, 3) for p in probs.tolist()]}")
        print(f"    mean={probs.mean():.3f} std={probs.std():.3f} (expect ~flat, no signal yet)")


if __name__ == "__main__":
    main()

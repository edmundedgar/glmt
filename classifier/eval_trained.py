"""Quick before/after sanity check: load the trained uspol head and compare
against the untrained baseline on a fresh sample of real posts (not the
LLM-labeled training data)."""

import json
from pathlib import Path

import torch

from classifier.model import MAX_TOKENS, TopicClassifier, load_pretrained
from classifier.train import TOPIC, WEIGHTS_DIR

DATA_PATH = Path(__file__).parent.parent / "data" / "posts.jsonl"
NUM_SAMPLES = 40


def main() -> None:
    posts = []
    with open(DATA_PATH) as f:
        for line in f:
            posts.append(json.loads(line))
            if len(posts) >= NUM_SAMPLES:
                break
    texts = [p["text"] for p in posts]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, tokenizer, _ = load_pretrained()
    encoder.to(device)

    untrained = TopicClassifier(encoder, [TOPIC]).to(device)
    trained = TopicClassifier(encoder, [TOPIC]).to(device)
    trained.heads[TOPIC].load_state_dict(torch.load(WEIGHTS_DIR / f"{TOPIC}.pt", map_location=device))

    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS
    ).to(device)

    with torch.no_grad():
        untrained_probs = torch.sigmoid(untrained(inputs["input_ids"], inputs["attention_mask"], TOPIC))
        trained_probs = torch.sigmoid(trained(inputs["input_ids"], inputs["attention_mask"], TOPIC))

    ranked = sorted(zip(texts, trained_probs.tolist(), untrained_probs.tolist()), key=lambda t: -t[1])
    print(f"{'trained':>8} {'untrained':>10}  text")
    for text, trained_p, untrained_p in ranked:
        clean = text.replace("\n", " ")[:90]
        print(f"{trained_p:8.3f} {untrained_p:10.3f}  {clean}")


if __name__ == "__main__":
    main()

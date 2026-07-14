"""Benchmark TopicClassifier inference throughput to see whether we can
keep up with the live firehose post-creation rate (~40-60 posts/sec,
measured empirically from the ingester) or need to subsample.

Usage:
    python -m classifier.benchmark
"""

import json
import time
from pathlib import Path

import torch

from classifier.model import MAX_TOKENS, TopicClassifier, load_pretrained

DATA_PATH = Path(__file__).parent.parent / "data" / "posts.jsonl"
BATCH_SIZES = [1, 16, 64, 128, 256]
NUM_TOPICS_SCENARIOS = [1, 20, 100]
WARMUP_ITERS = 3
TIMED_ITERS = 10


def load_sample_texts(path: Path, n: int) -> list[str]:
    texts = []
    with open(path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])
            if len(texts) >= n:
                break
    return texts


def benchmark_batch_size(classifier, tokenizer, texts: list[str], batch_size: int, topics: list[str], device: str) -> float:
    """Returns posts/sec, running the encoder + ALL given topic heads per batch."""
    batch = (texts * (batch_size // len(texts) + 1))[:batch_size]
    inputs = tokenizer(
        batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS
    ).to(device)

    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            embeddings = classifier.embed(inputs["input_ids"], inputs["attention_mask"])
            for topic in topics:
                classifier.heads[topic](embeddings)
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(TIMED_ITERS):
            embeddings = classifier.embed(inputs["input_ids"], inputs["attention_mask"])
            for topic in topics:
                classifier.heads[topic](embeddings)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    total_posts = batch_size * TIMED_ITERS
    return total_posts / elapsed


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    encoder, tokenizer, _ = load_pretrained()
    encoder.to(device)
    encoder.eval()

    max_topics = max(NUM_TOPICS_SCENARIOS)
    all_topic_names = [f"topic_{i}" for i in range(max_topics)]
    classifier = TopicClassifier(encoder, all_topic_names).to(device)
    classifier.eval()

    texts = load_sample_texts(DATA_PATH, 256)
    print(f"loaded {len(texts)} sample texts for benchmarking\n")

    print(f"{'batch_size':>10} {'n_topics':>8} {'posts/sec':>12}")
    for n_topics in NUM_TOPICS_SCENARIOS:
        topics = all_topic_names[:n_topics]
        for batch_size in BATCH_SIZES:
            rate = benchmark_batch_size(classifier, tokenizer, texts, batch_size, topics, device)
            print(f"{batch_size:>10} {n_topics:>8} {rate:>12.1f}")
        print()

    print("=" * 50)
    print("empirical firehose post-creation rate: ~40-60 posts/sec (measured from ingester)")


if __name__ == "__main__":
    main()

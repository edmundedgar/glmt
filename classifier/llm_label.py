"""Batch-label posts.jsonl for a topic using Claude Haiku 4.5, to bootstrap
training data for a custom TopicClassifier head.

Usage:
    python -m classifier.llm_label                 # labels data/posts.jsonl for "uspol"
    python -m classifier.llm_label --limit 500      # only label the first 500 posts

Requires ANTHROPIC_API_KEY (or an `ant auth login` profile) in the environment.
"""

import argparse
import json
import os
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

ENV_PATH = Path(__file__).parent.parent / ".env"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


MODEL = "claude-haiku-4-5"
DATA_PATH = Path(__file__).parent.parent / "data" / "posts.jsonl"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "labeled_uspol.jsonl"

TOPIC_NAME = "uspol"
TOPIC_DESCRIPTION = (
    "US politics: content about US political parties, politicians, elections, "
    "policy debates, or partisan commentary on US government and current affairs. "
    "Does not include politics of other countries unless directly compared to US politics."
)

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["yes", "no"]},
        "confidence": {"type": "number"},
    },
    "required": ["label", "confidence"],
    "additionalProperties": False,
}


def load_posts(path: Path, limit: int | None) -> list[dict]:
    posts = []
    with open(path) as f:
        for line in f:
            posts.append(json.loads(line))
            if limit is not None and len(posts) >= limit:
                break
    return posts


def build_prompt(text: str) -> str:
    return (
        f"Topic: {TOPIC_NAME}\nDefinition: {TOPIC_DESCRIPTION}\n\n"
        f'Post: "{text}"\n\n'
        "Does this post match the topic? Respond with your label and confidence (0.0-1.0)."
    )


def create_batch(client: anthropic.Anthropic, posts: list[dict]) -> str:
    requests = [
        Request(
            custom_id=str(i),
            params=MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": build_prompt(post["text"])}],
                output_config={"format": {"type": "json_schema", "schema": LABEL_SCHEMA}},
            ),
        )
        for i, post in enumerate(posts)
    ]
    batch = client.messages.batches.create(requests=requests)
    print(f"created batch {batch.id} ({len(requests)} requests)")
    return batch.id


def wait_for_batch(client: anthropic.Anthropic, batch_id: str) -> None:
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"status={batch.processing_status} "
            f"processing={counts.processing} succeeded={counts.succeeded} errored={counts.errored}"
        )
        if batch.processing_status == "ended":
            return
        time.sleep(30)


def collect_results(client: anthropic.Anthropic, batch_id: str, posts: list[dict]) -> list[dict]:
    labeled = []
    for result in client.messages.batches.results(batch_id):
        idx = int(result.custom_id)
        post = posts[idx]
        if result.result.type != "succeeded":
            print(f"skipping {post['uri']}: {result.result.type}")
            continue
        text = next(b.text for b in result.result.message.content if b.type == "text")
        parsed = json.loads(text)
        labeled.append(
            {
                "uri": post["uri"],
                "text": post["text"],
                "label": parsed["label"],
                "confidence": parsed["confidence"],
            }
        )
    return labeled


def main() -> None:
    load_dotenv(ENV_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    posts = load_posts(DATA_PATH, args.limit)
    print(f"loaded {len(posts)} posts from {DATA_PATH}")

    client = anthropic.Anthropic()
    batch_id = create_batch(client, posts)
    wait_for_batch(client, batch_id)
    labeled = collect_results(client, batch_id, posts)

    with open(OUTPUT_PATH, "w") as f:
        for row in labeled:
            f.write(json.dumps(row) + "\n")

    yes_count = sum(1 for r in labeled if r["label"] == "yes")
    print(f"wrote {len(labeled)} labels to {OUTPUT_PATH} ({yes_count} positive, {len(labeled) - yes_count} negative)")


if __name__ == "__main__":
    main()

"""Instead of a fixed binary topic, ask Claude to attach whatever topic
labels it thinks are appropriate to each post. Useful for discovering
candidate topics before committing to fixed per-topic heads.

Usage:
    python -m classifier.llm_freeform_label                    # batch over the whole dataset
    python -m classifier.llm_freeform_label --limit 500        # batch over the first 500 posts
    python -m classifier.llm_freeform_label --sync --limit 30  # quick synchronous sample, no batch/polling
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from classifier.llm_label import ENV_PATH, load_dotenv

MODEL = "claude-sonnet-5"
DATA_PATH = Path(__file__).parent.parent / "data" / "posts.jsonl"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "freeform_labels.jsonl"

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short, lowercase, hyphenated topic tags, e.g. 'us-politics', 'gaming', 'pets'.",
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}

PROMPT_TEMPLATE = (
    "Attach whatever topic labels you feel are appropriate to this social media post. "
    "Use concise, lowercase, hyphenated tags (e.g. 'us-politics', 'gaming', 'pets', 'sports'). "
    "Return as many or as few as genuinely apply -- an empty list is fine for pure banter, "
    "replies, or posts with no topical content.\n\n"
    'Post: "{text}"'
)


def load_posts(path: Path, offset: int, limit: int | None) -> list[dict]:
    posts = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            posts.append(json.loads(line))
            if limit is not None and len(posts) >= limit:
                break
    return posts


def run_sync(client: anthropic.Anthropic, posts: list[dict]) -> list[dict]:
    results = []
    for post in posts:
        response = client.messages.create(
            model=MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(text=post["text"])}],
            output_config={"format": {"type": "json_schema", "schema": LABEL_SCHEMA}},
        )
        text = next(b.text for b in response.content if b.type == "text")
        labels = json.loads(text)["labels"]
        results.append({"uri": post["uri"], "text": post["text"], "labels": labels})
    return results


def run_batch(client: anthropic.Anthropic, posts: list[dict]) -> list[dict]:
    requests = [
        Request(
            custom_id=str(i),
            params=MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(text=post["text"])}],
                output_config={"format": {"type": "json_schema", "schema": LABEL_SCHEMA}},
            ),
        )
        for i, post in enumerate(posts)
    ]
    batch = client.messages.batches.create(requests=requests)
    print(f"created batch {batch.id} ({len(requests)} requests)")

    while True:
        batch = client.messages.batches.retrieve(batch.id)
        counts = batch.request_counts
        print(
            f"status={batch.processing_status} "
            f"processing={counts.processing} succeeded={counts.succeeded} errored={counts.errored}"
        )
        if batch.processing_status == "ended":
            break
        time.sleep(30)

    results = []
    for result in client.messages.batches.results(batch.id):
        idx = int(result.custom_id)
        post = posts[idx]
        if result.result.type != "succeeded":
            print(f"skipping {post['uri']}: {result.result.type}")
            continue
        text_block = next((b for b in result.result.message.content if b.type == "text"), None)
        if text_block is None:
            print(f"skipping {post['uri']}: no text block (stop_reason={result.result.message.stop_reason})")
            continue
        labels = json.loads(text_block.text)["labels"]
        results.append({"uri": post["uri"], "text": post["text"], "labels": labels})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sync", action="store_true", help="synchronous per-request calls instead of Batch API")
    args = parser.parse_args()

    load_dotenv(ENV_PATH)
    client = anthropic.Anthropic()

    posts = load_posts(DATA_PATH, args.offset, args.limit)
    print(f"loaded {len(posts)} posts (offset={args.offset})")

    results = run_sync(client, posts) if args.sync else run_batch(client, posts)

    with open(OUTPUT_PATH, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(results)} labeled posts to {OUTPUT_PATH}")

    label_counts = Counter()
    for row in results:
        label_counts.update(row["labels"])

    print("=" * 60)
    print(f"top 40 labels across {len(results)} posts:")
    for label, count in label_counts.most_common(40):
        print(f"  {count:4d}  {label}")
    print(f"total distinct labels: {len(label_counts)}")


if __name__ == "__main__":
    main()

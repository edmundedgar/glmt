"""Give a local Ollama model our existing list of trained topics and ask it
to assign whichever apply (multi-label from a closed vocabulary), rather
than the open-ended freeform tagging we did with Claude. Compares against
the same posts' cardiffnlp-adjacent topic set we've already trained heads
for, to see how a local model handles a constrained version of the task.

Usage:
    python -m classifier.local_llm_topic_list_label
    python -m classifier.local_llm_topic_list_label --model qwen3:14b --no-think false --limit 20
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3:14b"
DATA_PATH = Path(__file__).parent.parent / "data" / "posts.jsonl"

# Our existing trained topic heads (classifier/weights/*.pt), plus
# additional well-represented topics pulled from the top-40 canonical
# labels discovered via freeform tagging on the full 9737-post sample
# (see data/freeform_labels_full_consolidated.jsonl). Skips vague
# catch-alls (banter, opinion) and pure-language tags (japanese-language,
# spanish-language, ...) -- language is metadata (commit.record.langs),
# not a topic in the same sense as the rest of this list.
KNOWN_TOPICS = [
    "us-politics",
    "sports",
    "music",
    "donald-trump",
    "gaming",
    "mental-health",
    "technology",
    "nsfw",
    "anime",
    "humor",
    "meme",
    "social-media",
    "politics",
    "weather",
    "food",
    "tv-shows",
    "nostalgia",
    "pets",
    "youtube",
    "art",
    "fandom",
    "daily-life",
    "health",
    "movies",
    "satire",
    "local-news",
    "travel",
    "social-commentary",
    "internet-culture",
    "football",
]

FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {"type": "string", "enum": KNOWN_TOPICS},
        }
    },
    "required": ["labels"],
}

PROMPT_TEMPLATE = (
    "Which of the following topics, if any, apply to this social media post?\n"
    f"Topics: {', '.join(KNOWN_TOPICS)}\n\n"
    'Post: "{text}"\n\n'
    "Return an array of matching topic names from the list above, or an empty "
    "array if none apply."
)


def load_posts(path: Path, limit: int) -> list[dict]:
    posts = []
    with open(path) as f:
        for line in f:
            posts.append(json.loads(line))
            if len(posts) >= limit:
                break
    return posts


def extract_labels(text: str) -> list[str]:
    """Without schema-constrained decoding, the model follows the prompt's
    literal instruction ("return an array") and emits a bare JSON array --
    not the {"labels": [...]} object our FORMAT_SCHEMA would have forced.
    Try the array form first (what we observe empirically), fall back to
    an object with a "labels" key in case a model wraps it differently."""
    try:
        start = text.rindex("[")
        end = text.rindex("]") + 1
        return json.loads(text[start:end])
    except ValueError:
        pass
    start = text.rindex("{")
    end = text.rindex("}") + 1
    return json.loads(text[start:end])["labels"]


def classify(model: str, text: str, no_think: bool) -> tuple[list[str], float, str]:
    """Returns (labels, elapsed_seconds, thinking_text). thinking_text is ""
    when no_think=True or the model didn't return one.

    Grammar-constrained decoding (the "format" JSON-schema param) forces the
    very first generated token to conform to the schema, which suppresses
    any thinking preamble regardless of /no_think -- confirmed empirically:
    with "format" set, eval_count stayed ~16 tokens whether /no_think was
    appended or not. To genuinely test thinking-enabled behavior we drop
    "format" entirely here and parse the model's natural JSON output instead.
    """
    prompt = PROMPT_TEMPLATE.format(text=text)
    body_dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_ctx": 4096},
    }
    if no_think:
        body_dict["messages"][0]["content"] += " /no_think"
        body_dict["format"] = FORMAT_SCHEMA
    body = json.dumps(body_dict).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    elapsed = time.perf_counter() - start
    labels = extract_labels(result["message"]["content"])
    thinking = result["message"].get("thinking", "") or ""
    return labels, elapsed, thinking


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--no-think", type=lambda s: s.lower() != "false", default=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_path = args.output or (
        Path(__file__).parent.parent
        / "data"
        / f"local_llm_known_topics_{'nothink' if args.no_think else 'thinking'}.jsonl"
    )

    posts = load_posts(DATA_PATH, args.limit)
    print(f"model={args.model} no_think={args.no_think} posts={len(posts)}")
    print(f"topics: {KNOWN_TOPICS}\n")

    from collections import Counter

    label_counts = Counter()
    total_time = 0.0
    errors = 0
    empty_count = 0

    out_f = open(output_path, "w")
    for i, post in enumerate(posts):
        try:
            labels, elapsed, thinking = classify(args.model, post["text"], args.no_think)
        except Exception as e:
            print(f"error on post {i}: {e}", flush=True)
            errors += 1
            continue
        total_time += elapsed
        label_counts.update(labels)
        if not labels:
            empty_count += 1
        row = {"uri": post["uri"], "text": post["text"], "labels": labels, "thinking": thinking}
        out_f.write(json.dumps(row) + "\n")
        out_f.flush()
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(posts)} done, avg {total_time / (i + 1 - errors):.2f}s/post", flush=True)
    out_f.close()

    n = len(posts) - errors
    print(f"\n{'=' * 60}")
    print(f"model={args.model} no_think={args.no_think}")
    print(f"posts={n} errors={errors} avg={total_time / n:.2f}s/post ({n / total_time:.2f} posts/sec)")
    print(f"saved to {output_path}")
    print(f"posts with no matching topic: {empty_count}/{n}\n")
    print("label frequency:")
    for label, count in label_counts.most_common():
        print(f"  {count:4d}  {label}")


if __name__ == "__main__":
    main()

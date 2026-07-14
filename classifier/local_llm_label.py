"""Try a local Ollama model for the same binary topic-labeling task we've
been doing with the Anthropic API, and compare agreement + speed against
the existing Claude-labeled data.

Usage:
    python -m classifier.local_llm_label                                    # 300-post sample
    python -m classifier.local_llm_label --model huihui_ai/qwen3.6-abliterated:35b --limit 50
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "richardyoung/qwen3-14b-abliterated:q4_K_M"
LABELED_PATH = Path(__file__).parent.parent / "data" / "labeled_uspol.jsonl"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "local_llm_labeled_uspol.jsonl"

TOPIC_NAME = "uspol"
TOPIC_DESCRIPTION = (
    "US politics: content about US political parties, politicians, elections, "
    "policy debates, or partisan commentary on US government and current affairs. "
    "Does not include politics of other countries unless directly compared to US politics."
)

FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["yes", "no"]},
        "confidence": {"type": "number"},
    },
    "required": ["label", "confidence"],
}

PROMPT_TEMPLATE = (
    f"Topic: {TOPIC_NAME}\nDefinition: {TOPIC_DESCRIPTION}\n\n"
    'Post: "{text}"\n\n'
    "Does this post match the topic? Respond with your label and confidence (0.0-1.0). "
    "/no_think"  # Qwen3 convention: skip the reasoning trace, answer directly
)


def load_reference(path: Path, limit: int) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def classify(model: str, text: str) -> tuple[str, float, float]:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}],
            "format": FORMAT_SCHEMA,
            "stream": False,
            "options": {"num_ctx": 4096},  # cap context so the model stays fully resident in GPU VRAM
        }
    ).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    elapsed = time.perf_counter() - start
    parsed = json.loads(result["message"]["content"])
    return parsed["label"], parsed["confidence"], elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()

    reference = load_reference(LABELED_PATH, args.limit)
    print(f"comparing local model {args.model!r} against {len(reference)} Claude-labeled posts\n")

    results = []
    tp = fp = tn = fn = 0
    total_time = 0.0
    errors = 0

    # Write incrementally -- a killed/timed-out run still leaves partial
    # results on disk instead of losing everything at the final write.
    out_f = open(OUTPUT_PATH, "w")
    for i, row in enumerate(reference):
        try:
            local_label, confidence, elapsed = classify(args.model, row["text"])
        except Exception as e:
            print(f"error on post {i}: {e}", flush=True)
            errors += 1
            continue
        total_time += elapsed
        claude_label = row["label"]

        if local_label == "yes" and claude_label == "yes":
            tp += 1
        elif local_label == "yes" and claude_label == "no":
            fp += 1
        elif local_label == "no" and claude_label == "no":
            tn += 1
        else:
            fn += 1

        result_row = {
            "uri": row["uri"],
            "text": row["text"],
            "claude_label": claude_label,
            "local_label": local_label,
            "local_confidence": confidence,
            "agree": local_label == claude_label,
        }
        results.append(result_row)
        out_f.write(json.dumps(result_row) + "\n")
        out_f.flush()
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(reference)} done, avg {total_time / (i + 1 - errors):.2f}s/post", flush=True)
    out_f.close()

    n = len(results)
    agree = tp + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n{'=' * 60}")
    print(f"model: {args.model}")
    print(f"posts: {n} ({errors} errors)")
    print(f"avg time/post: {total_time / n:.2f}s  ({n / total_time:.2f} posts/sec)")
    print(f"\nagreement with Claude labels: {agree}/{n} ({100 * agree / n:.1f}%)")
    print(f"treating Claude's label as reference:")
    print(f"  precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")
    print(f"  tp={tp} fp={fp} tn={tn} fn={fn}")
    print(f"\nresults saved to {OUTPUT_PATH}")

    disagreements = [r for r in results if not r["agree"]]
    if disagreements:
        print(f"\nsample disagreements ({min(10, len(disagreements))} of {len(disagreements)}):")
        for r in disagreements[:10]:
            clean = r["text"].replace("\n", " ")[:80]
            print(f"  claude={r['claude_label']} local={r['local_label']} ({r['local_confidence']:.2f})  {clean!r}")


if __name__ == "__main__":
    main()

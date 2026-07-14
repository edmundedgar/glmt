"""Consolidate the raw, fragmented label vocabulary from freeform LLM
tagging into a smaller canonical taxonomy (e.g. merge 'us-politics' /
'politics' / 'american-politics' into one canonical tag).

This is the "simpler process at the end" approach discussed earlier:
one cheap LLM call over the label vocabulary itself, not the posts.

Usage:
    python -m classifier.consolidate_labels
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import anthropic

from classifier.llm_label import ENV_PATH, load_dotenv

MODEL = "claude-sonnet-5"
INPUT_PATH = Path(__file__).parent.parent / "data" / "freeform_labels.jsonl"
MAPPING_PATH = Path(__file__).parent.parent / "data" / "label_consolidation_map.json"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "freeform_labels_consolidated.jsonl"

MAPPING_SCHEMA = {
    "type": "object",
    "properties": {
        "mapping": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "raw": {"type": "string"},
                    "canonical": {"type": "string"},
                },
                "required": ["raw", "canonical"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["mapping"],
    "additionalProperties": False,
}

PROMPT_TEMPLATE = """The following topic labels were independently applied to social media \
posts (no shared context between taggings), so near-duplicates and synonyms have \
proliferated -- e.g. "us-politics", "politics", and "american-politics" may all refer \
to the same concept.

Consolidate this into a smaller canonical taxonomy. For every raw label below, assign \
it a canonical label:
- Merge genuine synonyms, near-duplicates, and singular/plural variants (e.g. "meme" / "memes").
- When merging, prefer the clearer, more standard term as canonical -- not necessarily \
the most frequent one.
- Do NOT merge genuinely distinct topics just because they're related \
(e.g. keep "gaming" and "esports" separate if both appear meaningfully; \
keep "us-politics" separate from general "politics" if the raw list treats them differently).
- Every raw label must appear exactly once in the output, mapped to itself if it's already canonical.

Raw labels with frequency counts:
{label_list}
"""


def load_raw_labels(path: Path) -> Counter:
    counts = Counter()
    with open(path) as f:
        for line in f:
            counts.update(json.loads(line)["labels"])
    return counts


def build_prompt(counts: Counter) -> str:
    lines = [f"{label}: {count}" for label, count in counts.most_common()]
    return PROMPT_TEMPLATE.format(label_list="\n".join(lines))


MIN_COUNT_FOR_CONSOLIDATION = 2  # labels seen only once have nothing to merge into; skip them


def get_mapping(client: anthropic.Anthropic, counts: Counter) -> dict[str, str]:
    """Only sends labels with >= MIN_COUNT_FOR_CONSOLIDATION occurrences to the
    LLM -- the long tail of one-off labels is identity-mapped directly, both
    to keep the output small enough to fit a single response and because a
    label with no duplicates has nothing to consolidate against anyway."""
    frequent = Counter({label: c for label, c in counts.items() if c >= MIN_COUNT_FOR_CONSOLIDATION})
    rare = set(counts) - set(frequent)
    print(f"consolidating {len(frequent)} labels with >= {MIN_COUNT_FOR_CONSOLIDATION} occurrences "
          f"({len(rare)} singletons left as-is)")

    with client.messages.stream(
        model=MODEL,
        max_tokens=64000,
        messages=[{"role": "user", "content": build_prompt(frequent)}],
        output_config={
            "format": {"type": "json_schema", "schema": MAPPING_SCHEMA},
            "effort": "low",
        },
    ) as stream:
        response = stream.get_final_message()
    print(f"stop_reason={response.stop_reason} output_tokens={response.usage.output_tokens}")
    text = next(b.text for b in response.content if b.type == "text")
    parsed = json.loads(text)
    mapping = {entry["raw"]: entry["canonical"] for entry in parsed["mapping"]}

    missing = set(frequent) - set(mapping)
    for label in missing:
        mapping[label] = label
    for label in rare:
        mapping[label] = label
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--mapping-output", type=Path, default=MAPPING_PATH)
    args = parser.parse_args()

    load_dotenv(ENV_PATH)
    client = anthropic.Anthropic()

    raw_counts = load_raw_labels(args.input)
    print(f"{len(raw_counts)} distinct raw labels across the sample ({args.input})")

    mapping = get_mapping(client, raw_counts)
    args.mapping_output.write_text(json.dumps(mapping, indent=2, sort_keys=True))

    canonical_counts = Counter()
    rows = []
    with open(args.input) as f:
        for line in f:
            row = json.loads(line)
            canonical_labels = sorted({mapping[label] for label in row["labels"]})
            canonical_counts.update(canonical_labels)
            rows.append({"uri": row["uri"], "text": row["text"], "labels": canonical_labels})

    with open(args.output, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"consolidated to {len(canonical_counts)} canonical labels "
          f"(from {len(raw_counts)} raw, {len(raw_counts) - len(canonical_counts)} merged away)")
    print(f"mapping saved to {args.mapping_output}")
    print(f"consolidated labels saved to {args.output}")
    print("\ntop 40 canonical labels:")
    for label, count in canonical_counts.most_common(40):
        print(f"  {count:4d}  {label}")

    print("\nsample merges (raw -> canonical, where different):")
    merges = [(raw, canon) for raw, canon in sorted(mapping.items()) if raw != canon]
    for raw, canon in merges[:30]:
        print(f"  {raw!r:30s} -> {canon!r}")
    if len(merges) > 30:
        print(f"  ... and {len(merges) - 30} more")


if __name__ == "__main__":
    main()

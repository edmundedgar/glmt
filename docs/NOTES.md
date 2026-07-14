# Technical Notes

Everything learned building this out, organized by component. Written so you (or a fresh Claude session) can pick this back up without re-deriving it. Cross-reference `docs/spec.md` for the original design and `docs/RUNBOOK.md` for command syntax.

## Status vs. the spec

Only **Component 1 (Ingester)** and a prototype of **Components 2/5 (Classifier + Training)** exist, as offline CLI scripts — not the live pipeline the spec describes.

- **Built:** Ingester (full, matches spec), `TopicClassifier` model + training/eval/benchmark scripts, three different label-generation approaches (Anthropic binary, Anthropic freeform+consolidation, local Ollama models), 8 trained topic heads.
- **Not built:** Component 3 (Postgres Label DB), Component 4 (Labeler Server / `@skyware/labeler`), the live `asyncio.Queue` hookup between Ingester and a continuous Classifier Worker, the `/train` HTTP endpoint. Everything classifier-side currently reads from a flat `data/posts.jsonl` file in batch/offline mode, not from the ingester's queue in real time.
- Training labels currently come from LLM-generated data (Anthropic API or local Ollama models), not the spec's user-submitted URI list — this was a deliberate detour to answer "can we bootstrap training data automatically" before building the `/train` endpoint's user-facing flow.

---

## Ingester (`ingester/`)

Matches the spec closely. Two things worth knowing if you touch it:

**Jetstream protocol drift.** The GitHub repo's `main` branch has moved to a substantially different v2 (binary CBOR frames, segment-based backfill, different dictionary-fetch endpoint) — but as of this writing, the **live public endpoint** (`jetstream2.us-east.bsky.network`) still speaks the plain-JSON v1 protocol the spec assumes. Verified empirically by connecting directly and reading raw frames. **Re-verify this if picking the project back up after a gap** — it's clearly a live migration in progress upstream.

**zstd dictionary.** `compress=true` requires a dictionary to decompress; it's not embedded in the wire protocol. We vendored it from the jetstream repo at `internal/subscribe/zstd_dictionary` (112,640 bytes) — note this path, **not** the old `pkg/models/zstd_dictionary` (404s on the current `main` branch). Copied to `ingester/assets/zstd_dictionary`.

**Cursor resume behavior.** Reconnecting with a stale cursor makes Jetstream replay the backlog as fast as the network allows until it catches up to live tip. If you leave the ingester off for a while and restart it, expect a burst of much-higher-than-normal throughput for the first few minutes — **don't use that period to measure steady-state rate.** Measured steady-state: **~40-60 posts/sec** for `app.bsky.feed.post` creates specifically (post-preprocessing filter already applied).

---

## Classifier model (`classifier/model.py`)

### The `pooler_output` bug in the spec

The spec's code sample (`spec.md` line 70-72) does `self.encoder(...).pooler_output`. This is wrong for this specific checkpoint: `cardiffnlp/tweet-topic-latest-multi` is a `RobertaForSequenceClassification` and **ships no trained pooler**. Loading it via bare `AutoModel` silently fabricates a **randomly-initialized** `pooler.dense` to fill the gap — using `pooler_output` would mean feeding random noise into every downstream head.

Fix used throughout: load via `AutoModelForSequenceClassification`, pull out `.roberta` (the frozen base, shares weights with the pretrained 19-topic head), and use `last_hidden_state[:, 0, :]` — the `<s>` token's hidden state, which is RoBERTa's actual convention and what the pretrained classifier head is trained against. See `load_pretrained()` and `TopicClassifier.embed()`.

### `TopicClassifier`

Frozen encoder + `nn.ModuleDict` of one `nn.Linear(768, 1)` per topic, as specced. An untrained head is confirmed to produce flat, uninformative output (~0.45-0.5 mean, ~0.09 std, no correlation with content) — this is expected, not a bug, until training runs.

### `StackedTopicHeads` — throughput optimization for many topics

`TopicClassifier` loops over topics in Python (`for topic in topics: classifier.heads[topic](embeddings)`). This is fine up to ~100 topics (the loop overhead is negligible next to the encoder's own forward pass) but **degrades roughly linearly with topic count** beyond that — benchmarked:

| topics | looped (posts/sec, heads only) | stacked (posts/sec) |
|---|---|---|
| 100 | 17,397 | 2,198,754 |
| 1,000 | 2,465 | 1,931,481 |
| 5,000 | 480 | 470,321 |

`StackedTopicHeads` compiles many independently-trained `nn.Linear(768,1)` heads into one `nn.Linear(768, n_topics)` — mathematically identical output (diffs ~1e-7, pure floating point), just one matmul instead of N. **Training is unaffected** — keep training individual heads with `train.py` as usual; compile into the stacked form only for serving. Not yet wired into any live inference path; exists as a standalone module + benchmark.

### Throughput headroom vs. the firehose

At batch size 64 (matches the ingester's own batch-flush size — unplanned but convenient), the encoder alone does **~370-380 posts/sec** on an RTX 3060, independent of 1 vs. 100 topics (looped) since head compute is negligible at that scale. Firehose rate is ~40-60 posts/sec → **6-9x headroom** even before the stacking optimization. With `StackedTopicHeads`, that headroom holds even at 5,000+ topics, since head cost becomes negligible again.

### GPU memory

Measured (not just estimated): encoder + 9 heads + a batch of 64, training or inference, uses **~890-900 MiB** reserved. Confirms the spec's "~500MB + negligible per head" claim. See "GPU memory: local classifier + a loaded Ollama model coexist fine" further down for how this behaves alongside a local LLM.

---

## Anthropic API label generation (`classifier/llm_label.py`, `llm_freeform_label.py`, `consolidate_labels.py`)

### Structured output + adaptive thinking = silent truncation

**This bit us twice.** Sonnet 5 (and the Opus 4.x family) default to adaptive thinking at `effort: high`. If `max_tokens` is set too low for a task that induces a lot of deliberation, the model can burn the *entire* budget on thinking and return **zero output text** — `stop_reason: max_tokens`, content is a lone truncated `thinking` block, no `text` block at all. This happened in `consolidate_labels.py` with `max_tokens=16000` against 852 raw labels (it thought right up to the limit and never got to write the JSON).

**Fix:** for mechanical, non-reasoning-heavy tasks, explicitly set `output_config: {"effort": "low"}`, and don't lowball `max_tokens`. Any code that does `next(b.text for b in response.content if b.type == "text")` needs a `None`-safe fallback (`next((b for b in ... ), None)`), not a bare generator expression — a `StopIteration` from this exact failure mode crashed a batch-results parser mid-collection once real data was involved (see "batch result parsing" below).

### Batch API economics (real measured numbers, not estimates)

- **Binary yes/no** (Haiku 4.5, 9,743 posts): 3.0M input tokens, 166,547 output tokens → **$1.92** (with 50% batch discount, at $1/$5 per MTok pre-discount).
- **Freeform multi-label** (Sonnet 5, 500 posts): 205,504 input, 12,434 output → **$0.27** actual, scaling to **~$5-8** for the full 9,743-post equivalent (intro pricing vs. standard).
- Freeform tagging costs more per-post than binary because the output is a variable-length label array (~25 tokens/post avg) vs. a single `{label, confidence}` (~17 tokens/post).

### Batch result parsing gotchas

- **Refusals happen on innocuous content.** In one 9,243-post freeform batch, 5 posts got `stop_reason: "refusal"` with `category: "bio"` — on ROT13-encoded fandom discussion text (a spoiler-hiding convention, not actual bio-risk content) and duck-emoji spam posts linking to a shopping app. False positives from the safety classifier, not anything real. Always handle `stop_reason` before assuming a `text` block exists.
- **Truncation also loses the text block.** One post hit `stop_reason: "max_tokens"` with only a partial block and no parseable text (a non-English post in an unfamiliar script that apparently needed more tokens than budgeted). Same handling as refusals: check for the block's existence, skip and log rather than crash.
- **Write results incrementally, not just at the end.** An early version of the batch-polling scripts only wrote output after the full results loop completed. A crash or `kill`/timeout partway through the loop lost *everything*, including successfully-fetched results. Fixed by opening the output file once and calling `.write()` + `.flush()` after every single row — this pattern is now used in every script that iterates and writes (`llm_freeform_label.py`, `local_llm_label.py`, `local_llm_topic_list_label.py`, `local_llm_bulk_label.py`). Cheap insurance; do this by default in any future long-running collection script.

### Label consolidation (`consolidate_labels.py`)

Freeform tagging (no shared context between individual taggings) produces heavy synonym drift — 500 posts produced 852 distinct raw labels; the full 9,737-post run produced **7,110**. Two lessons from building the consolidation pass:

1. **Prompt examples over-anchor the model.** The first version's prompt included "e.g. keep `us-politics` separate from general `politics`" as an example of what *not* to over-merge. Result: the model left the *entire* politics-family (`us-politics`, `politics`, `political-opinion`, `political-commentary`) completely unconsolidated, even though it correctly merged plenty of other true synonyms (`democrats`→`democratic-party`, `trump`→`donald-trump`, `autism`→`neurodiversity`). One example in a prompt can generalize much further than intended — be wary of "don't merge X and Y"-style examples steering behavior on the whole neighborhood around X and Y.
2. **Frequency-threshold before sending to the LLM.** Trying to consolidate all 7,110 raw labels in one call blew the output budget — truncated mid-JSON, `JSONDecodeError`. Fix: only send labels with **≥2 occurrences** to the model (2,729 of 7,110 for the full dataset); singleton labels have nothing to merge into anyway, so identity-map them directly without spending a token. This dropped required output to ~60K tokens (fit comfortably at `max_tokens=64000`, `effort: "low"`) and preserved coverage (86.6% of all label *instances*, since the long tail is Zipfian — most of the volume is in the frequent labels).

**Important:** overlapping/hierarchical labels surviving consolidation (e.g. `politics` + `us-politics` both applying to the same post) is not necessarily a bug to fix — see "Overlapping labels are a feature, not a bug" below.

---

## Training (`classifier/train.py`)

Implements the spec's Component 5 training flow (minus the HTTP endpoint): freeze encoder, precompute embeddings once, train only the topic's `nn.Linear(768,1)` head with `BCEWithLogitsLoss`/`AdamW`/`lr=2e-4`/`batch_size=32`/up to 10 epochs/early-stop on validation F1 (patience=3)/deploy only if F1 > 0.7 (2:1 negative:positive sampling ratio, per spec default).

Supports two data sources via `--source binary` (yes/no + confidence, from `llm_label.py`) or `--source freeform` (multi-label list, from `consolidate_labels.py` or `local_llm_bulk_label.py` — same on-disk shape, just filtered to "does `--topic` appear in this row's `labels` list").

### Results so far — the key finding is about data volume, not method

| Topic | Source | Positives | Precision | Recall | F1 | Deployed? |
|---|---|---|---|---|---|---|
| uspol | binary (dedicated batch) | 638 | 0.852 | 0.906 | 0.878 | ✅ |
| us-politics | freeform, 500-post sample | 42 | 0.545 | 0.750 | 0.632 | ❌ |
| **us-politics** | **freeform, full 9,737-post sample** | **702** | **0.850** | **0.929** | **0.887** | ✅ |
| sports | freeform, full sample | 386 | 0.893 | 0.870 | 0.882 | ✅ |
| music | freeform, full sample | 324 | 0.927 | 0.797 | 0.857 | ✅ |
| donald-trump | freeform, full sample | 155 | 0.750 | 0.968 | 0.845 | ✅ |
| gaming | freeform, full sample | 464 | 0.915 | 0.707 | 0.798 | ✅ |
| mental-health | freeform, full sample | 143 | 0.840 | 0.750 | 0.792 | ✅ |
| technology | freeform, full sample | 120 | 0.889 | 0.667 | 0.762 | ✅ |
| nsfw | freeform, full sample | 221 | 0.821 | 0.523 | 0.639 | ❌ |
| anime | freeform, full sample | 162 | 0.560 | 0.4375 | 0.491 | ❌ |

**Rule of thumb: aim for 500+ positive examples**, not ~40-150. The `us-politics` 500→9,737-post comparison isolates this cleanly (same topic, same method, 42 vs. 702 positives: F1 0.632 → 0.887) — and validates that **freeform-tagged + consolidated data can match or beat purpose-built binary labeling** once volume is comparable (0.887 vs. 0.878, essentially tied, slightly favoring freeform on recall).

**Two topics failed to clear the deploy threshold despite reasonable data volume** — worth understanding before assuming more data alone fixes everything:
- `nsfw` (221 positives, F1 0.639): likely too heterogeneous a category for a linear probe on this embedding space — "NSFW" covers wildly different content (text, described imagery, varying explicitness) that probably doesn't cluster into one clean region.
- `anime` (162 positives, F1 0.491, barely above random): likely inconsistent *source* labeling — heavy overlap with neighboring freeform tags (`fandom`, `fan-art`, `japan`) means the freeform tagger itself probably applied `anime` inconsistently, muddying the training signal before the classifier ever sees it.

---

## Local model experiments (Ollama)

### Models tried

| Model | Size | Fits 12GB GPU? |
|---|---|---|
| `richardyoung/qwen3-14b-abliterated:q4_K_M` | 9.0-9.6 GB | Yes, 100% GPU |
| `qwen3:14b` (standard, non-abliterated) | 9.3-9.6 GB | Yes, 100% GPU |
| `huihui_ai/qwen3.6-abliterated:35b` | 23 GB | No — splits ~56%/44% CPU/GPU, slow |

### Abliteration hurts calibration — confirmed, not just suspected

The abliterated 14B model is fast (~0.95s/post) but badly miscalibrated for careful binary judgment: on a 300-post `uspol` comparison against Claude's labels, **recall=1.000 but precision=0.229** (F1 0.372) — it says "yes" on almost everything, including emoji spam and unrelated small talk, sometimes even while self-reporting *low* confidence (0.20-0.30) for that same "yes." Swapping to the standard (non-abliterated) `qwen3:14b` on the same task: **96.7% agreement with Claude, F1 0.750** — dramatically better, at the cost of being ~10x slower (9.24s/post vs 0.95s/post). Conclusion: the abliteration (refusal-removal fine-tuning) itself degrades judgment quality on this kind of task; it's not that "small local models can't classify."

### Ollama structured output silently disables thinking, regardless of `/no_think`

Important and non-obvious: passing Ollama's `format` parameter (JSON schema, used for reliable structured output) triggers **grammar-constrained decoding**, which forces the very first generated token to conform to the schema. This leaves no room for a `<think>...</think>` preamble — **thinking gets suppressed by `format` alone**, whether or not you also append the Qwen3 `/no_think` convention. Confirmed empirically: with `format` set, `eval_count` stayed ~16 tokens whether `/no_think` was appended or not; dropping `format` entirely let `eval_count` jump to ~180-500+ tokens and exposed a genuine separate `message.thinking` field in Ollama's response.

To actually test thinking-enabled behavior, you have to drop `format` and parse the model's natural output instead — and note the model then follows the prompt's literal phrasing rather than a fixed schema shape (asked for "an array", got a bare `[...]`, not the `{"labels": [...]}` object the schema would have forced). `classifier/local_llm_topic_list_label.py`'s `extract_labels()` tries the array form first, falls back to an object with a `labels` key.

**Practical upshot for this task specifically:** thinking-enabled vs. `/no_think` made no meaningful difference. Row-by-row on 15 identical posts: 86.7% exact agreement, and the 2 disagreements were the hardest/most ambiguous posts in the sample with defensible calls in *both* directions (not one mode being systematically better). Wall-clock was also nearly identical (11.92s vs 11.67s/post) despite ~20-30x more tokens generated when thinking — there's a large fixed per-request latency component (~9-12s) on this setup that isn't explained by `load_duration + prompt_eval_duration + eval_duration` in Ollama's own response timing fields; never fully diagnosed (ruled out: stuck/orphaned processes, checked via `ollama stop` + fresh reload, latency persisted). If picking this up again, worth `llama-server`-level profiling rather than trusting Ollama's reported duration breakdown.

**Decision made:** stick with `/no_think` + `format` schema — same quality, and it's the faster/simpler code path (structured output is easier to parse reliably than free text).

### Closed-list multi-label: topic list size is a real speed/coverage tradeoff

Expanded `KNOWN_TOPICS` from 9 to 30 (pulling from the top-40 canonical labels the freeform+consolidation pipeline discovered). Result on the same 49-60 posts:

| | 9 topics | 30 topics |
|---|---|---|
| Zero-labeled ("nothing matched") | 61% | 27% |
| Speed | 11.67s/post | ~19s/post (~65% slower) |

More candidate topics = better real coverage, but real cost — more enum options for the grammar to consider per token, evidently.

### Overlapping labels are a feature, not a bug

Expanding the topic list reintroduced the `politics`/`us-politics` co-occurrence pattern from the Claude freeform pipeline (both tags applied redundantly to the same clearly-US-political post). Initially treated this as a fragmentation problem to fix (dedupe, or drop one term) — **user correction: this is intentional.** Since each topic becomes an independent binary classifier head, having both a broad (`politics`) and narrow (`us-politics`) version lets an end user toggle either filter independently on different days ("hide all politics" vs. "just hide US politics, I still want to see UK/other coverage"). Don't "fix" this kind of overlap without checking whether it's actually load-bearing for the downstream use case.

### GPU memory: local classifier + a loaded Ollama model coexist fine

Measured, not assumed:

| | VRAM |
|---|---|
| `qwen3:14b` loaded | 9,337 MiB |
| + our classifier (9 heads, batch 64, inference or training) | +~890-900 MiB |
| **Combined** | **~10,230 MiB** of 12,288 MiB total |
| Free remaining | ~1.5-2.0 GB |

Both fit with headroom to spare. The only caveat is *compute* contention (GPU time-slicing) if both are actively crunching at the same literal moment — not a memory capacity problem.

### Resumable bulk labeling (`local_llm_bulk_label.py`) — kill-safety verified, not assumed

Built for "leave it running whenever the GPU is idle, interrupt freely." Verified with real `kill -9` tests, not just code review:

1. Started fresh, killed **before** any post finished → output file stayed empty. No corruption.
2. Started fresh, let 4 posts complete, `kill -9`'d mid-request on the 5th → file had **5 valid JSON lines** (one more had completed in the moment between the check and the kill landing) — no truncated/corrupted line.
3. Restarted with `--limit 3` → correctly reported `"already labeled: 5 posts"`, skipped all 5, added exactly 3 new ones → 8 total, **8 distinct URIs, zero duplicates**.

The mechanism: open the output file once, `write()` + `flush()` after every single completed row (never buffer multiple rows before writing), and build the "already done" skip-set from the output file's own contents by URI on startup. Also hardened `load_already_labeled_uris()` to skip (not crash on) a malformed/truncated final line, as belt-and-suspenders — never actually triggered in testing since the flush-per-row pattern already made corruption practically impossible, but cheap to have.

As of this writing, only 8 posts have accumulated in `data/local_llm_bulk_labeled.jsonl` — this needs a real long run before it's useful training data (compare to the 500-700+ positives needed per the training section above).

---

## Data file inventory (`data/`, gitignored)

Everything in `data/` is generated, not source-controlled. Key files, in rough pipeline order:

| File | Generated by | Notes |
|---|---|---|
| `posts.jsonl` | `ingester/main.py` | Grows unbounded; ~721K rows / ~170MB as of this writing. Not deduplicated across runs by design (append-only). |
| `cursor.txt` | `ingester/main.py` | Single `time_us` value, resume state. |
| `labeled_uspol.jsonl` | `llm_label.py` | Binary yes/no + confidence, one topic. |
| `freeform_labels*.jsonl` | `llm_freeform_label.py` | Several variants exist from incremental runs (`_first500`, `_rest9243`, `_full`) — see git history / conversation for provenance if the naming is unclear; `freeform_labels_full_consolidated.jsonl` is the canonical full-sample consolidated version. |
| `label_consolidation_map*.json` | `consolidate_labels.py` | raw label → canonical label mapping. |
| `local_llm_*.jsonl` | the three `local_llm_*.py` scripts | Comparison/experiment outputs; filenames encode the model/mode. `local_llm_bulk_labeled.jsonl` is the one growing accumulator meant for periodic retraining. |
| `eval_uspol_comparison.txt`, `freeform_labels_sample.txt` | ad hoc `eval_trained.py` / `browse.py` runs | Human-readable snapshots, not machine-consumed. |

**Scripts that overwrite their default output path on every run:** `llm_freeform_label.py`, `local_llm_label.py`, `local_llm_topic_list_label.py` (unless `--output`/`--data-path` is given). Back up anything you want to keep before rerunning — this bit us more than once during comparison experiments.

---

## Open questions / natural next steps

Carried over from the spec's own list, plus what we've since learned:

- **Language filtering** — the spec flags this as needed (base model is English-only); we saw concrete non-English mislabeling in early exploration. Not implemented in the ingester (`commit.record.langs` is available but unused).
- **Wire the live pipeline** — Ingester → `asyncio.Queue` → Classifier Worker → Label DB, per the spec, doesn't exist yet. Current classifier work is entirely offline/batch against `posts.jsonl`.
- **Components 3 & 4** (Postgres Label DB, `@skyware/labeler` TypeScript server) — not started.
- **Wire `StackedTopicHeads` into a real serving path** — built and benchmarked standalone, not yet used by any inference script.
- **Grow `local_llm_bulk_labeled.jsonl`** — only 8 rows so far; needs a real unattended run (hours, whenever the GPU is free) before it's useful for retraining anything.
- **Storage optimization** — still deferred per spec, unaddressed.
- **The Jetstream v1→v2 migration** — re-check the live endpoint's protocol if this project sits untouched for a while; the upstream rewrite was clearly mid-flight when this was built.

# Runbook

Quick reference for running each piece. For *why* things are built this way and the gotchas behind them, see `docs/NOTES.md`. For the original design, see `docs/spec.md`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

Anthropic API key goes in a gitignored `.env` file at the repo root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

(`classifier/llm_label.py` has a tiny hand-rolled `.env` loader — no `python-dotenv` dependency.)

For local-model experiments, [Ollama](https://ollama.com) must be running (`ollama serve`, usually already a systemd service) with at least one model pulled — we used `qwen3:14b` (the non-abliterated one; see NOTES.md for why).

All commands below assume you're at the repo root with the venv's `python` on the invocation (`.venv/bin/python -m ...`).

---

## 1. Ingester — collect posts from the firehose

```bash
python -m ingester.main
```

Runs until killed; appends preprocessed `(uri, text)` pairs to `data/posts.jsonl` and persists resume state to `data/cursor.txt`. Safe to stop/restart — it resumes ~5s before where it left off. If restarted after a long gap, expect a fast "catch-up" burst before it settles into live real-time rate (~40-60 posts/sec for `app.bsky.feed.post` creates).

## 2. Classifier baseline (no training required)

```bash
python -m classifier.demo      # untrained custom head vs. cardiffnlp's pretrained 19-topic head, side by side
python -m classifier.browse    # groups a sample of posts.jsonl by cardiffnlp's top predicted topic
```

## 3. Generate training labels via the Anthropic API

**Binary yes/no for one topic** (cheap, Haiku, uses Batch API):

```bash
python -m classifier.llm_label --limit 500   # omit --limit to label the whole posts.jsonl
```
→ `data/labeled_uspol.jsonl`

**Open-ended multi-label tagging** ("attach whatever topics apply", Sonnet 5, Batch API):

```bash
python -m classifier.llm_freeform_label --limit 500
python -m classifier.llm_freeform_label --sync --limit 30   # quick synchronous sample, no batch/polling wait
```
→ `data/freeform_labels.jsonl` (overwrites — copy it first if you want to keep a prior run)

**Consolidate the resulting label vocabulary** (freeform tagging fragments into synonyms — `us-politics`/`politics`/`political-opinion` etc.):

```bash
python -m classifier.consolidate_labels --input data/freeform_labels.jsonl --output data/freeform_labels_consolidated.jsonl
```

## 4. Train a topic head

```bash
python -m classifier.train                                              # binary source, topic=uspol
python -m classifier.train --source freeform --topic us-politics        # freeform source, any canonical label
python -m classifier.train --source freeform --topic gaming --data-path data/local_llm_bulk_labeled.jsonl
```

Trains only the frozen-encoder's per-topic `nn.Linear` head (few seconds on GPU). Saves to `classifier/weights/<topic>.pt` **only if validation F1 > 0.7** — otherwise it prints the metrics and declines to save (per spec).

## 5. Evaluate

```bash
python -m classifier.eval_trained   # trained vs. untrained head, ranked comparison on fresh posts
```

## 6. Benchmark throughput

```bash
python -m classifier.benchmark   # encoder+heads throughput across batch sizes / topic counts
```

## 7. Local model (Ollama) experiments

**Binary yes/no vs. Claude's labels** (accuracy/speed comparison):

```bash
python -m classifier.local_llm_label --model qwen3:14b --limit 60
```
→ `data/local_llm_labeled_uspol.jsonl` (overwrites each run — back up first if comparing models)

**Closed-list multi-label** (give it our known topics, ask which apply):

```bash
python -m classifier.local_llm_topic_list_label --model qwen3:14b --limit 60
python -m classifier.local_llm_topic_list_label --model qwen3:14b --no-think false --limit 15   # thinking enabled
```
→ `data/local_llm_known_topics_{nothink,thinking}.jsonl`

**Resumable bulk labeling** (the "leave it running whenever the GPU is free" mode):

```bash
python -m classifier.local_llm_bulk_label                # runs until posts.jsonl is exhausted or killed
python -m classifier.local_llm_bulk_label --limit 500     # cap new posts this invocation
```
→ appends to `data/local_llm_bulk_labeled.jsonl`, skips anything already labeled. Safe to `kill -9` and restart any time — verified empirically (see NOTES.md). Feed the growing file into `classifier/train.py --source freeform --data-path data/local_llm_bulk_labeled.jsonl` periodically to retrain.

---

## Typical end-to-end flow for a new topic

1. Pick a topic name and a one-line definition.
2. Either: `llm_label.py` with that topic (fast, cheap, purpose-built binary data), **or** rely on the general-purpose `llm_freeform_label.py` + `consolidate_labels.py` pipeline if the topic already showed up in a freeform tagging pass.
3. `classifier/train.py --topic <name> ...` — check the printed F1. Below 0.7 usually means too few positive examples (rule of thumb from experience: aim for 500+ positives, not ~40) rather than a fundamentally bad topic.
4. Weights land in `classifier/weights/<topic>.pt`, ready to be loaded into a `TopicClassifier` (or compiled into a `StackedTopicHeads` for fast multi-topic serving).

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

**`posts.jsonl` rotation.** Left alone, this file grows forever (already 1GB+). A cron job (installed via `crontab -e`, runs daily at 3am) handles this automatically:

```
0 3 * * * cd /home/glmt/glmt && .venv/bin/python -m ingester.rotate_posts >> data/rotate_posts.log 2>&1
```

(cron jobs start in `$HOME`, not the repo root, so the `cd` is required — a bare relative path here fails silently into the wrong directory.)

`ingester/rotate_posts.py` only rotates once `classifier/live_export_labels.py`'s cursor is caught up to within ~20MB of `posts.jsonl`'s current size — otherwise it skips that day entirely rather than risk archiving away posts nothing has processed yet. When it does rotate: stops the ingester, renames `posts.jsonl` aside, restarts the ingester (fresh empty file, firehose resumes from its own saved cursor), then compresses the rotated file into `data/archive/posts-<timestamp>.jsonl.gz` and prunes archives older than 14 days. Safe to run manually any time: `python -m ingester.rotate_posts` (add `--dry-run` to just see whether it would rotate, without doing anything).

`data/rotate_posts.log` (from the cron redirect) and `data/ingester.log` (the restarted ingester's own stdout/stderr) are both useful for checking a rotation actually happened correctly.

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

**Naming quirk:** the currently-deployed political-content head is named `uspol` (matches the label identifier declared in `labeler/labels.json`), but the local-model freeform data tags the same concept `us-politics` (matches the Anthropic freeform pipeline's canonical label). Training against `--topic us-politics` saves to `classifier/weights/us-politics.pt` — copy that over `uspol.pt` to actually deploy it:

```bash
python -m classifier.train --source freeform --topic us-politics --data-path data/local_llm_bulk_labeled.jsonl
cp classifier/weights/us-politics.pt classifier/weights/uspol.pt
```

Every other currently-deployed topic (`sports`, `music`, `donald-trump`, `gaming`, `mental-health`, `technology`) uses the same tag name in both the local-model data and the deployed identifier, so no rename is needed for those.

**Before retraining any deployed head, back up the current weights** (`mkdir -p classifier/weights_backup_<name> && cp classifier/weights/*.pt classifier/weights_backup_<name>/`, then `ls` the backup dir to confirm files actually landed) — `train.py` only overwrites a weight file if the new F1 clears 0.7, but a technically-passing head can still be a regression on data the old one handled well. If you deploy a new head, **restart `live_export_labels.py`** afterward — it loads weights once at startup and won't pick up a change on disk.

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

## 8. Live pipeline: classify continuously and serve real labels

Three long-running processes, meant to run concurrently (all resumable — safe to kill and restart independently):

```bash
python -m ingester.main                    # firehose -> data/posts.jsonl
python -m classifier.live_export_labels    # posts.jsonl -> labeler/pending-labels.jsonl, polls every 10s
node --env-file=.env labeler/server.mjs    # pending-labels.jsonl -> signed labels, serves queryLabels/subscribeLabels
```

(There's also `classifier/export_labels.py --limit 2000`, a one-shot batch version that samples `posts.jsonl` instead of tailing it continuously — useful for a quick backfill or a one-off check, not for ongoing serving.)

`live_export_labels.py` tracks its own cursor (`data/live_export_cursor.txt`, separate from the ingester's) as a **byte offset** into `posts.jsonl`, and processes new lines in bounded chunks (`MAX_LINES_PER_CYCLE = 5000`) rather than reading the whole file every cycle — see NOTES.md if touching this, it OOM'd once before the bounded-chunk cap was added. `server.mjs` tracks a byte offset into `pending-labels.jsonl` (`labeler/.ingest-offset`) the same way. Both detect the file shrinking (truncated/rotated) and resync from the start instead of breaking — this makes it safe to eventually rotate/prune `posts.jsonl` (currently unbounded-growth, gigabytes and counting) without having to coordinate a cursor reset by hand.

## 9. Labeler server setup (one-time, `labeler/`)

```bash
cd labeler
npm install
```

Needs these in the repo-root `.env` (gitignored):

```
LABELLER_DID=did:plc:...
LABELLER_SIGNING_KEY=<secp256k1 private key, hex>
APP_PASSWORD=<bsky app password, for the account-setup scripts only>
```

`LABELLER_DID`/`LABELLER_SIGNING_KEY` come from running `npx @skyware/labeler setup` once against the labeler's bsky account (see NOTES.md for why a bsky account is needed at all — short version: `app.bsky.labeler.service` is a PDS-backed repo record, so the labeler needs a PDS, and the easiest way to get one is a regular account). That command also declares the labeler's endpoint and signing key into the account's did:plc document.

**Label taxonomy** (`labeler/labels.json`) is the source of truth for what labels exist. Push it to the account:

```bash
node --env-file=../.env declare-labels.mjs             # from labeler/
node --env-file=../.env declare-labels.mjs --dry-run    # preview without pushing
node --env-file=../.env get-labels.mjs                  # check what's currently declared
```

Re-run `declare-labels.mjs` (no `--dry-run`) any time labels aren't showing up in a client and you want to force the AppView to re-fetch the declaration — it's cheap and has fixed exactly this before.

**Verify the signing key matches the DID document** (useful if labels aren't verifying/showing up anywhere and you suspect a key mismatch):

```bash
node --env-file=../.env verify-signing-key.mjs "at://did:plc:.../app.bsky.feed.post/..."
```

Derives the pubkey from `.env`, compares it against the did:plc document's `#atproto_label` verification method (**not** `#atproto` — that's the account's regular PDS/repo key, a separate keypair), and checks a real label's signature from `queryLabels` against both. All three should agree.

**Then run the server** (see section 8 above): `node --env-file=.env labeler/server.mjs` — listens on port 14831, ingests `pending-labels.jsonl` into its own SQLite (`labeler/labels.db`, gitignored, owned entirely by `@skyware/labeler` — no external DB hookup, see NOTES.md).

For serving this publicly from a second box during development, see `deploy/README.md`.

## 10. Check overall status

```bash
python -m tools.status
```

One-shot report: which of the 4 long-running processes (ingester, live classifier export, bulk labeler, labeler server) are actually up, the SSH tunnel's systemd state, the ingester's firehose lag, how far behind each downstream stage's cursor is from its input file (posts.jsonl → pending-labels.jsonl → labels.db), the bulk labeler's row count and last-write freshness (flags if the process is running but hasn't written anything in 5+ minutes — this exact symptom was a real Ollama hang once), rotation cron/archive state, and GPU memory/utilization. No daemon, stdlib only, safe to run any time.

---

## Typical end-to-end flow for a new topic

1. Pick a topic name and a one-line definition.
2. Either: `llm_label.py` with that topic (fast, cheap, purpose-built binary data), **or** rely on the general-purpose `llm_freeform_label.py` + `consolidate_labels.py` pipeline if the topic already showed up in a freeform tagging pass.
3. `classifier/train.py --topic <name> ...` — check the printed F1. Below 0.7 usually means too few positive examples (rule of thumb from experience: aim for 500+ positives, not ~40) rather than a fundamentally bad topic.
4. Weights land in `classifier/weights/<topic>.pt`, ready to be loaded into a `TopicClassifier` (or compiled into a `StackedTopicHeads` for fast multi-topic serving).

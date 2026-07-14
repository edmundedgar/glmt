# Technical Notes

Everything learned building this out, organized by component. Written so you (or a fresh Claude session) can pick this back up without re-deriving it. Cross-reference `docs/spec.md` for the original design and `docs/RUNBOOK.md` for command syntax.

## Status vs. the spec

The full pipeline is live end-to-end — firehose to a real, publicly-queryable Bluesky labeler — but several pieces work differently than the spec originally sketched.

- **Built:** Ingester (full, matches spec). `TopicClassifier` + `StackedTopicHeads` model, training/eval/benchmark scripts. Three label-generation approaches (Anthropic binary, Anthropic freeform+consolidation, local Ollama models) plus a resumable unattended bulk-labeling loop. 7 deployed topic heads (`uspol`, `sports`, `music`, `donald-trump`, `gaming`, `mental-health`, `technology`) — `death` is declared in the label taxonomy but has no trained head yet. A live classification pipeline (`classifier/live_export_labels.py`) that tails `posts.jsonl` continuously. A working labeler server (`labeler/server.mjs`, built on `@skyware/labeler`) that signs and serves real labels via `queryLabels`/`subscribeLabels`, publicly reachable at `label.goat.navy` through a dev-phase SSH-tunnel+nginx relay (see `deploy/`).
- **Built differently than spec'd:** Component 3 (Label DB) isn't Postgres — `@skyware/labeler` owns its own SQLite (`labeler/labels.db`) with no external-DB hook, so the earlier prefix-table Postgres compaction design was never implemented (Postgres is installed on the box but unused). Component 4 (Labeler Server) is Node/TypeScript (`@skyware/labeler`), not the Python spec sketch — needed because of both the signing-algorithm mismatch (see below) and the lack of a callback-based DB integration point in the actual library. The Ingester→Classifier→Label-DB pipeline is poll-based (`live_export_labels.py` tails a file every 10s, `server.mjs` tails another file every 10s) rather than the spec's push-based `asyncio.Queue`.
- **Not built:** the `/train` HTTP endpoint (training is CLI-only, `classifier/train.py`), language filtering.
- Training labels come from LLM-generated data (Anthropic API and local Ollama models), not the spec's user-submitted URI list — a deliberate detour to answer "can we bootstrap training data automatically" before building the `/train` endpoint's user-facing flow.

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

`StackedTopicHeads` compiles many independently-trained `nn.Linear(768,1)` heads into one `nn.Linear(768, n_topics)` — mathematically identical output (diffs ~1e-7, pure floating point), just one matmul instead of N. **Training is unaffected** — keep training individual heads with `train.py` as usual; compile into the stacked form only for serving. Wired into both `classifier/export_labels.py` and `classifier/live_export_labels.py`.

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

### Retraining from `local_llm_bulk_labeled.jsonl` (local-model freeform tags, no consolidation pass)

Once the local-model bulk-labeling run accumulated 6,159 rows, all 7 currently-deployed heads were retrained against it (same `--source freeform` path, just pointed at the local-model output instead of the Anthropic-consolidated one — the on-disk shape is identical by design):

| Topic | Positives | F1 | vs. previous deployed F1 | Deployed? |
|---|---|---|---|---|
| uspol (`us-politics` tag) | 733 | 0.816 | 0.887 → 0.816 | ✅ (replaced) |
| sports | 223 | 0.867 | 0.882 → 0.867 | ✅ (replaced) |
| donald-trump | 117 | 0.826 | 0.845 → 0.826 | ✅ (replaced) |
| music | 219 | 0.735 | 0.857 → 0.735 | ✅ (replaced) |
| gaming | 268 | 0.744 | 0.798 → 0.744 | ✅ (replaced) |
| mental-health | 137 | 0.564 | 0.792 → *unchanged* | ❌ below threshold, old head kept |
| technology | 208 | 0.686 | 0.762 → *unchanged* | ❌ below threshold, old head kept |

Note the F1s here are mostly slightly *lower* than the Anthropic-consolidated versions they replaced — this data hasn't been through a consolidation pass (raw freeform tags straight from the local model, e.g. `politics` and `us-politics` both appear as separate uncombined tags), and the local model itself is a smaller/less careful tagger than Claude. The point of retraining on it isn't "better than the Anthropic data" — it's that this dataset keeps growing for free from idle GPU time, so it's the practical way to keep improving heads (especially the two that never had enough Anthropic-sourced positives to clear 0.7 at all) without spending more API budget. `mental-health` and `technology` didn't clear the deploy bar this round, most likely simply on data volume (137/208 positives vs. 700+ for `uspol`) — worth retrying once the accumulator grows further.

Previous deployed weights should be backed up before any promotion (`cp classifier/weights/*.pt classifier/weights_backup_<name>/`) — check there if a retrain regresses something in practice despite passing the F1 gate. **Get the target directory name right**: a `mkdir classifier/weights_backup_X` followed by `cp ... classifier/weights_backup_Y/` (mismatched name, e.g. from a shell `$(date ...)` substitution not matching a hardcoded fallback string) fails silently if stderr is redirected to `/dev/null` — this happened once here, and the "backup" was actually empty when the real thing was needed. The original Anthropic-trained weights for the 5 replaced topics were recovered by simply retraining from `data/freeform_labels_full_consolidated.jsonl` again — `train.py` is fully deterministic (`SEED = 0`, `torch.manual_seed(SEED)`), and the reproduced F1s matched the original table exactly (0.887/0.882/0.857/0.845/0.798). They now live in `classifier/weights_backup_original_anthropic/` (gitignored, matches `classifier/weights_backup*/`). **Verify a backup actually landed files** (`ls` it) before trusting it, and don't rely on `2>/dev/null` to hide errors from a step whose success you actually care about.

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

### Use Ollama's native `think` API field, not the `/no_think` prompt convention — the latter is unreliable

**Superseded finding, kept for the record:** an early test suggested passing Ollama's `format` parameter (JSON schema) alone reliably suppressed thinking via grammar-constrained decoding, regardless of whether `/no_think` was also appended to the prompt (`eval_count` stayed ~16 tokens either way in that test). **This did not hold up under sustained real-world use.** Once `local_llm_bulk_label.py` ran unattended over hundreds of posts, the *exact same* request shape (format set, `/no_think` in the prompt) sometimes produced a full thinking trace anyway (3000+ chars, 40-50x slower) and sometimes didn't — same code, non-deterministic outcome. `/no_think` is a text convention the model can probabilistically ignore; it was never a hard constraint, the small early sample just didn't happen to surface a miss.

**Fix:** use Ollama's `"think": false` request field instead — a real API-level control, not a prompt hint. Verified reliable across many repeated calls (~0.85-1.5s/post, zero thinking every time, no variance). `classifier/local_llm_topic_list_label.py`'s `classify()` now sends `"think": not no_think` directly; there's no longer a need for a "parse thinking out of free text" fallback path, since `format` + `think` work cleanly together regardless of which way `think` is set (thinking-enabled responses expose it in a separate `message.thinking` field, not interleaved with the structured `content`).

**Practical upshot for this task specifically:** thinking-enabled vs. non-thinking made no meaningful difference to label quality on a small side-by-side (both modes got the two hardest/most ambiguous posts "wrong" in defensible-either-way fashion, nothing systematic). Given that, and that non-thinking is both faster and immune to the reliability problem above, non-thinking is the default (`--no-think` defaults to `True`).

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

As of this writing, `data/local_llm_bulk_labeled.jsonl` has grown to 6,000+ rows from unattended runs and is already being used to retrain deployed heads (see updated results table above) — the "needs a real long run" concern from early testing is resolved.

### Ollama `--context-shift` + no `num_predict` cap = a request that can hang forever

Found the hard way: a bulk-labeling run silently stalled for 45+ minutes — process still alive, GPU still at 100%, but the output file stopped growing entirely. The Ollama server here is launched with `--context-shift`, which lets generation keep going past `num_ctx` by dropping old tokens instead of stopping, and the client-side request had no `num_predict` cap and no timeout — so a single post that triggered a degenerate repetition loop in the model had nothing to stop it. `resp.read()`/`urlopen()` just blocked indefinitely waiting for a response that was never going to finish.

**Fix:** `classify()` now sets `"num_predict": 512` in the request options (real JSON label output for this schema never comes close — a full 30-topic array is under ~150 tokens) and passes `timeout=60` to `urlopen()`. Belt-and-suspenders: the cap should make runaway generations impossible, the timeout catches it anyway if something still goes long, and the caller (`local_llm_bulk_label.py`'s per-post `try/except`) already treats a failure as "log and skip this post," so hitting either safeguard degrades to a missed post rather than a hung pipeline. Confirmed benign in practice — after the fix, resuming a run produced 1 truncated-JSON error in the first 30 posts (an unusually verbose generation hitting the cap), everything else processed normally at the usual ~1.2-1.6s/post.

---

## Labeler server (`labeler/`)

### Why a bsky account is needed at all

The base AT Protocol doesn't require a labeler to have a bsky account — but `app.bsky.labeler.service` (the record declaring "this DID is a labeler, here's its endpoint and label taxonomy") is a normal PDS-backed repo record, and a DID needs a PDS to host any repo record at all. A regular bsky account is the easy way to get a DID with a working PDS, so the labeler ended up being a real (if second) bsky account. It migrates cleanly like any other did:plc account if the labeler ever needs to move PDS/host.

### Signing: secp256k1, not Ed25519 (spec was wrong)

The spec's pseudocode assumed Ed25519. Confirmed via `@skyware/labeler`'s actual source (`util/crypto.js`: `k256Sign` — SHA-256 hash of the dag-cbor-encoded label, signed with secp256k1, `lowS: true`) and independently cross-checked against Bluesky's Go reference implementation (`indigo`'s `atcrypto` package), which also only supports secp256k1/p256 for repo signing — no Ed25519 anywhere in the stack.

### Two separate signing keys live on the same DID document — don't confuse them

A labeler account's did:plc document has **two** `verificationMethod` entries: `#atproto` (the account's normal PDS/repo commit-signing key — every bsky account has this) and `#atproto_label` (a dedicated key just for signing labels, set up separately via `plcSetupLabeler`/`npx @skyware/labeler setup`). `LABELLER_SIGNING_KEY` in `.env` corresponds to `#atproto_label`, **not** `#atproto`. Comparing a derived pubkey against the wrong one of these looks exactly like a real key mismatch (different multibase strings) but isn't — cost some real back-and-forth before catching the mixup. `labeler/verify-signing-key.mjs` checks the right one (`#atproto_label`) and also verifies a real signature from a live `queryLabels` response end-to-end (derives pubkey from `.env` → dag-cbor-encodes the exact label shape `@skyware/labeler` signs, `{src, uri, val, cts, neg, ver}` — notably *not* including the `id` field, which is a storage-assigned autoincrement added after signing — → checks the ECDSA signature).

### `subscribeLabels` is a real WebSocket, not SSE (spec was wrong here too)

Confirmed both from `@skyware/labeler`'s type signature (`SubscriptionHandler = WebsocketHandler<...>`) and empirically (`curl` gets back `HTTP/1.1 101 Switching Protocols` with binary DAG-CBOR frames). This mattered for the nginx relay config (`deploy/nginx-label.goat.navy.conf`) — a config tuned for SSE (`proxy_set_header Connection '';`) actively strips the `Upgrade`/`Connection` headers a WebSocket handshake needs, breaking `subscribeLabels` outright. Fixed with the standard `map $http_upgrade $connection_upgrade { default upgrade; '' close; }` pattern, which lets one location block correctly handle both plain HTTP (`queryLabels`) and the WS upgrade (`subscribeLabels`).

### `@skyware/labeler` owns its own storage — no external DB hookup

The spec's pseudocode imagined a callback-based `queryLabels` handler backed by our own Postgres schema. The actual library doesn't support this: `LabelerServer` owns a SQLite database directly (via `@libsql/client`, `dbPath`/`dbUrl`+`dbToken`) and labels go in via its own `createLabel`/`createLabels` instance methods — there's no pluggable storage interface. This is why `labeler/server.mjs` is a thin polling adapter (tail `pending-labels.jsonl`, call `createLabels`) rather than a real DB writer, and why the earlier prefix-table Postgres compaction design (numeric user-ID + rkey-as-timestamp compaction, discussed for the "100-1000x more users than posts" scaling case) was never implemented — Postgres is installed on the box per that discussion but currently unused.

### Two unbounded-growth OOM bugs, same root cause, same fix pattern

Both `labeler/server.mjs` and `classifier/live_export_labels.py` are long-running loops that poll an ever-growing append-only file (`pending-labels.jsonl`, `posts.jsonl`) for new rows. Both were originally written to **re-read and re-parse the entire file every poll cycle** — fine at first, but cost grows with the file, and eventually:

- `server.mjs` OOM-crashed (`JavaScript heap out of memory`, SIGABRT) after the pending-labels file grew to ~92.5MB / 743K lines from a large catch-up burst.
- `live_export_labels.py` was caught before it crashed (proactively stopped once it hit ~3.7GB RSS and climbing) — a second unattended multi-hour ingester run had grown `posts.jsonl` to 4.5M lines, and the read-everything-per-cycle pattern was about to hold ~3.4M posts in memory at once.

**Fix, same shape in both:** track how much of the file has already been consumed (`server.mjs`: a byte offset via `fs.promises.open` + positional `read()`, persisted to `.ingest-offset`; `live_export_labels.py`: originally a line-count cursor, since migrated to match — see below) and only process what's newly appended, with a hard per-cycle cap (`live_export_labels.py`: `MAX_LINES_PER_CYCLE = 5000`) so even a huge backlog gets worked off incrementally instead of as one unbounded batch. **General lesson for this codebase:** any loop that repeatedly reads a file which is *also* being appended to by another process needs bounded/incremental reads from the start — this bit us twice with the identical shape, worth checking for a third time before it happens again.

### A second, deeper `server.mjs` memory leak — unresolved, mitigated with auto-restart

Distinct from the bug above (that one was a genuine logic bug — re-reading the whole file every cycle — and is fully fixed). This one showed up *after* that fix, once `labels.db` grew past ~2M rows: `server.mjs` still eventually crashes with `FATAL ERROR: ... JavaScript heap out of memory` during sustained catch-up, and along the way becomes largely unresponsive to HTTP requests (including a trivial `SELECT 1` health check) for extended stretches.

**What this ruled out, each confirmed by isolated benchmark against the real, large database** (not guessed):
- `server.createLabels()` itself: benchmarked directly (bypassing the file-polling code entirely), including after `server.start()` had been called — stayed at a constant ~1.1ms/call with flat memory across thousands of calls.
- The file-read/`JSON.parse`/`Map`-building step: also benchmarked in isolation — ~15ms total for a full capped-size cycle's worth of rows.
- SQLite WAL checkpoint cost: manually ran `PRAGMA wal_checkpoint(PASSIVE)` against the live 536MB database — instant (0.00s).
- Overlapping poll cycles: ruled out by construction (the `setTimeout`-based sequential loop, see above) and confirmed only one `server.mjs` process was ever running.
- External load via the public tunnel: killed the SSH tunnel process outright and retested locally — still failed identically, so it isn't real internet traffic overwhelming the server.
- General event-loop starvation from a tight microtask loop: added a `YIELD_EVERY_N_LABELS`-based explicit yield (`await setImmediate`) — reduced but did not eliminate the problem. A separate unrelated heartbeat timer (`setInterval` logging every 500ms, nothing to do with the ingest loop) mostly ticked on schedule but showed an 80+ second gap shortly before the eventual crash — so *something* genuinely blocks hard some of the time, it's just not explained by any of the above.

**Working theory, not confirmed:** something inside `@skyware/labeler`'s own internals — `LabelerServer` instance state, `emitLabel`'s per-call DAG-CBOR frame encoding (`frameToBytes`), or something in the WebSocket/`@fastify/websocket` layer — retains memory proportional to the *cumulative* number of `createLabels` calls made over the process's lifetime, not just the current cycle's. This would explain why isolated benchmarks (a few thousand calls total, short-lived process) never reproduced it, while the real server (tens of thousands of cumulative calls across many cycles in one long-lived process) does. Not verified — would need to bisect `@skyware/labeler` versions or instrument inside the library itself to confirm.

**Mitigation shipped, not a fix:** `MAX_BYTES_PER_CYCLE` was cut from 2MB down to 256KB (less work, and less of whatever leaks, per cycle), and the process now runs under supervision that restarts it automatically on crash:
- `deploy/labeler-server.service` — a proper systemd unit (`Restart=always`), matching the existing `labeler-tunnel.service` pattern. Requires `sudo` to install, which this session doesn't have — needs to be installed by hand (`sudo cp ... && sudo systemctl enable --now labeler-server.service`, see RUNBOOK.md).
- `labeler/run_supervised.sh` — a plain bash restart-loop, no sudo needed, used as an immediate stopgap until the systemd unit is installed.

Before this mitigation existed, the exact failure mode was "crashes and then sits silently down for hours until someone happens to check" — which is what actually happened earlier in this session (found via `tools/status.py` showing the process simply gone, no crash report anyone had seen). The auto-restart doesn't fix the leak, but it changes the failure mode from "silent extended outage" to "brief automatic recovery," which is a real improvement even without the root cause in hand.

**If picking this up again:** the library-internals theory above is the most promising unexplored lead. Also worth trying: pin/downgrade `@skyware/labeler` to an earlier version and see if the leak reproduces (isolates whether this is a regression in a specific version), or run with `--max-old-space-size` set low deliberately to force more frequent, cheaper GC and see if that changes the crash frequency/behavior in an informative way.

### `live_export_labels.py`'s cursor: line-count → byte-offset migration

The original fix above used a line-count cursor (`for i, line in enumerate(f): if i < start_line: continue`) rather than matching `server.mjs`'s byte-offset approach — it worked, but had a real gap: a line count is only meaningful for the *exact* file it was measured against. If `posts.jsonl` were ever truncated or rotated (the natural next step given it's unbounded-growth and already over a gigabyte), the cursor would have no way to distinguish "the file legitimately shrank" from "we're just early in a fresh file" — it would either silently stall for as long as it takes the file to regrow past the old line count, or throw `FileNotFoundError` outright if the file happened to be mid-delete during a poll.

**Fixed** by switching to the same byte-offset + shrink-detection pattern as `server.mjs`: `read_new_posts()` now `seek()`s to a byte offset, reads complete (`\n`-terminated) lines via `readline()` up to the per-cycle cap, and leaves a trailing partial line for next cycle rather than parsing it early (mirrors `server.mjs`'s `leftover` handling). A shrink (`stat().st_size < byte_offset`) triggers an explicit resync-from-0 with a log line, not a silent hang or crash.

**Migrating a live line-count cursor to a byte offset** isn't a no-op — you can't just reinterpret the same number under the new meaning (a line count of 2.77M is nowhere near byte 2.77M in a gigabyte-scale file). Converted correctly: stop the process, read the exact number of lines the old cursor represented straight from `posts.jsonl` with `readline()` in a loop, take `f.tell()` at that point as the new byte-offset cursor, write that, then restart. Confirmed correct by observing the resumed process pick up exactly where the old line-based one left off — no gap, no reprocessed duplicates.

### `posts.jsonl` rotation (`ingester/rotate_posts.py`)

The byte-offset cursor above wasn't just a robustness nicety — it's the actual enabling mechanism for rotation. `posts.jsonl` grows forever with no retention, so something has to periodically remove old history. Why this isn't just a `logrotate` config:

- **logrotate's `copytruncate` can only shrink a file from the end.** We need the opposite: drop the already-processed *prefix*, keep the unprocessed *tail* intact. There's no cheap in-place operation for that (removing bytes from the front of a file means rewriting everything after them), so this isn't solvable as a truncate at all.
- **logrotate's default rename+reopen model needs the writer to cooperate** (handle a reopen signal, or be restarted) — that part *is* a fit here, since `ingester/main.py` opens `posts.jsonl` once (`open(path, "a")`) for its whole process lifetime and already tolerates being killed and restarted cleanly.
- **The real blocker: logrotate has no way to ask "is it safe to rotate right now?"** before acting. A schedule alone risks rotating away posts `live_export_labels.py` hasn't processed yet — exactly the scenario that happened for real when the ingester silently stopped for ~3 hours; rotating blind during a gap like that would have permanently discarded that backlog instead of just delaying it.

So this is a small dedicated script instead, using the rename+restart model logrotate would use, but gated correctly:

1. Compare `live_export_labels.py`'s byte-offset cursor (`data/live_export_cursor.txt`) against `posts.jsonl`'s current size. If the gap exceeds `CAUGHT_UP_MARGIN_BYTES` (20MB — generous compared to steady-state lag, tiny compared to a real backlog), **skip rotation entirely** for this run. This is the whole safety property; everything else is bookkeeping.
2. Stop the ingester (`pkill -f ingester.main`, poll until it's actually gone — don't rename while it might still write).
3. Rename `posts.jsonl` → `posts.jsonl.rotating`, then restart the ingester immediately (it creates a fresh `posts.jsonl` via the same `open(path, "a")`, and resumes the firehose from its own saved `time_us` cursor — any posts missed during the few-second stop/restart window get caught by Jetstream's own replay-from-cursor behavior, same as any other restart).
4. **`live_export_labels.py` needs no explicit handling for any of this.** Its shrink-detection (`stat().st_size < byte_offset` → resync from 0) already covers "the file I'm reading just got replaced with a smaller one" — that's exactly what a fresh post-rotation `posts.jsonl` looks like from its point of view. This is the payoff of having fixed that cursor before building rotation on top of it.
5. Only after the ingester is back up: gzip-compress `posts.jsonl.rotating` into `data/archive/posts-<UTC timestamp>.jsonl.gz` (deferred until after restart so compression time doesn't extend the ingester's downtime), then prune archives older than 14 days.

**Accepted tradeoff:** `local_llm_bulk_label.py` re-scans `posts.jsonl` fresh on every run and doesn't track a cursor into it at all — after a rotation, whatever it hadn't yet gotten to in the archived-off portion is simply unreachable to it (it never reads `data/archive/`). Deliberately not solved: it's a training-data accumulator with millions of posts to draw from already, so losing some unprocessed backlog on rotation just means marginally fewer future training examples, not a correctness problem. Revisit only if that accumulator ever becomes coverage-starved.

Runs daily via cron at 3am (`crontab -e`) rather than a size threshold — simpler, and the catch-up gate already makes it safe regardless of how much the file has grown by then. Verified end-to-end: the caught-up gate correctly declined to rotate while a real ~475MB backlog existed, and `archive_rotated_file()`/`prune_old_archives()` were tested in isolation (round-tripped a fake rotated file through gzip, pruned a synthetically-aged archive without touching a real one) before trusting them against production data. Also worth remembering for next time: cron jobs start in `$HOME`, not wherever you happen to be when you install the crontab entry — a relative path without a leading `cd /path/to/repo &&` fails silently into the wrong directory instead of erroring obviously.

---

## Status tool (`tools/status.py`)

One-shot report covering all the state scattered across this session's notes: which long-running processes are actually alive, the ingester's firehose lag, each downstream cursor's backlog against its input file, the bulk labeler's row count and write-freshness, rotation/archive state, GPU usage. Stdlib-only, no daemon.

**`ps -o comm=` is not a reliable way to identify a process's interpreter.** The first version of this tool matched processes via `pgrep -f <pattern>` and then filtered by checking `ps -o comm=` for a `python`/`node` prefix, to reject wrapper shells that merely mention the pattern in their own invocation text (a real problem: e.g. `pgrep -f "server.mjs"` also matches a `bash -c '... server.mjs'` launcher wrapper, not just the actual `node server.mjs` process). This silently misidentified the real `server.mjs` process as NOT RUNNING on first use — its `comm` reported as `MainThread`, not `node` (something in its dependency stack, likely libuv or a native addon, calls `prctl`/`pthread_setname_np` and overwrites the kernel-visible process name). Fixed by resolving `/proc/<pid>/exe` instead, which points at the actual binary path regardless of what the process has renamed its `comm` to. **General lesson:** `comm` is a mutable, self-reported label (any process can rename itself); if you need to know what binary a PID actually is, resolve `/proc/<pid>/exe`.

## Query tool (`tools/query_labels.py`)

Read-only queries against `labeler/labels.db` (label frequency summary, all labels for a URI, recent posts for a given label, optionally windowed by `--since <hours>`) — for when you want to actually look at what's being labeled without hand-writing SQL each time. **Postgres is not involved** — it's installed on this box (from the earlier compaction-design discussion, see "Labeler server" above) but holds no data; the real store is `@skyware/labeler`'s own SQLite.

**`sqlite3`'s `datetime('now', ...)` does not compare correctly against this table's timestamp format, and fails silently rather than erroring.** `cts` is stored as `2026-07-14T09:00:12.190Z` (`T` separator, millisecond precision, `Z` suffix), but `datetime('now', '-1 hours')` produces `2026-07-14 08:00:15` (space separator, no millis, no `Z`). Both columns have TEXT affinity, so `cts >= datetime('now', '-1 hours')` compares them as plain strings — and because `'T'` (0x54) sorts after `' '` (0x20), *every* `cts` value counts as `>=` *any* such cutoff regardless of actual time, so the filter silently becomes a no-op. Caught by testing empirically before trusting it: a `--since 1` and a `--since 24` query returned the exact same count (the full table) until fixed. **Fix:** build the cutoff string in Python, in the exact same format as the stored column (`dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"`), and compare that against `cts` directly — sidesteps SQLite's datetime parsing entirely. **General lesson:** when comparing TEXT-affinity timestamp columns in SQLite, either confirm both sides are in the *exact* same string format, or verify with two different windows that should obviously produce different counts — don't trust a single query that merely "runs without error."

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

**Closed since the last pass** (kept here briefly for continuity, drop this line entirely on the next cleanup): live pipeline is wired (poll-based, not push-based — see "Labeler server" above); Components 3 & 4 exist in modified form; `StackedTopicHeads` is wired into both `export_labels.py` and `live_export_labels.py`; `local_llm_bulk_labeled.jsonl` has real volume now (6,000+ rows, growing) and has already been used for a retrain.

Still open:

- **Language filtering** — the spec flags this as needed (base model is English-only); we saw concrete non-English mislabeling in early exploration. Not implemented in the ingester (`commit.record.langs` is available but unused).
- **`mental-health` and `technology` heads need more data** — didn't clear the F1 > 0.7 deploy bar on the local-model retrain (137 and 208 positives respectively); worth retrying once `local_llm_bulk_labeled.jsonl` grows further, or once that data goes through a consolidation pass like the Anthropic freeform data did.
- **Storage optimization / the Postgres compaction design** — moot for now since `@skyware/labeler` owns its own SQLite with no external-DB hook, but would become relevant again if the labeler server is ever swapped out or made to read from an independent Label DB.
- **The Jetstream v1→v2 migration** — re-check the live endpoint's protocol if this project sits untouched for a while; the upstream rewrite was clearly mid-flight when this was built.
- **Push-based live pipeline** — current pipeline is polling at every stage (10s intervals in both `live_export_labels.py` and `server.mjs`), which works but adds up to ~20s worst-case latency from post ingestion to label availability. Not a problem yet; would matter if latency requirements tighten.
- **`num_predict`/timeout guards against Ollama hangs are a mitigation, not a guarantee** — they bound the damage from a runaway generation but don't explain *why* `--context-shift` + a particular post triggers one; if bulk-labeling stalls again, check for the same symptom (process alive, GPU busy, output file static) before assuming something new.

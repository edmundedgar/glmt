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

Four long-running processes, meant to run concurrently (all resumable — safe to kill and restart independently). Several have silently died at least once with nothing running to notice or restart them, so all four have systemd units (`deploy/*.service`) with `Restart=always` — this is the recommended way to run them:

```bash
sudo cp deploy/ingester.service deploy/live-export.service deploy/ozone.service deploy/ozone-bridge.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ingester live-export ozone ozone-bridge
sudo systemctl status ingester live-export ozone ozone-bridge
journalctl -u ingester -u live-export -u ozone -u ozone-bridge -f
```

Manual/foreground equivalents, if you don't want the systemd units installed:

```bash
python -m ingester.main                    # firehose -> data/posts.jsonl
python -m classifier.live_export_labels    # posts.jsonl -> labeler/pending-labels.jsonl, polls every 10s
node --env-file=.env services/ozone/api.ts # (from ozone-src/services/ozone) -- signs and serves queryLabels/subscribeLabels
node --env-file=../.env ozone_bridge.mjs   # (from labeler/) -- pending-labels.jsonl -> Ozone via emitEvent, polls every 10s
```

Note `local_llm_bulk_label.py` (section 7) deliberately does **not** have a systemd unit — you want manual control over when it runs so it can yield the GPU to other work, and `Restart=always` would fight that.

(There's also `classifier/export_labels.py --limit 2000`, a one-shot batch version that samples `posts.jsonl` instead of tailing it continuously — useful for a quick backfill or a one-off check, not for ongoing serving.)

`live_export_labels.py` tracks its own cursor (`data/live_export_cursor.txt`, separate from the ingester's) as a **byte offset** into `posts.jsonl`, and processes new lines in bounded chunks (`MAX_LINES_PER_CYCLE = 5000`) rather than reading the whole file every cycle — see NOTES.md if touching this, it OOM'd once before the bounded-chunk cap was added. `ozone_bridge.mjs` tracks a byte offset into `pending-labels.jsonl` (`labeler/.ozone-ingest-offset`) the same way. Both detect the file shrinking (truncated/rotated) and resync from the start instead of breaking — this makes it safe to eventually rotate/prune `posts.jsonl` (currently unbounded-growth, gigabytes and counting) without having to coordinate a cursor reset by hand.

## 9. Ozone setup (current labeler backend)

Ozone (`bluesky-social/atproto`'s production moderation/labeler service) replaced `@skyware/labeler` as the labeler backend — see NOTES.md for why (unmaintained, unresolved memory leak). Built from source; no Docker available on this box.

**One-time setup:**

```bash
git clone --depth 1 https://github.com/bluesky-social/atproto.git /home/glmt/ozone-src   # OUTSIDE the repo, not a submodule
cd /home/glmt/ozone-src
corepack enable && corepack prepare pnpm@11.11.0 --activate
PUPPETEER_SKIP_DOWNLOAD=true pnpm install --frozen-lockfile
pnpm run --recursive --stream --filter '@atproto/aws...' --filter '@atproto/ozone...' build
```

**Database** (Postgres already installed on this box, role `glmt` has superuser — no sudo needed):

```bash
psql -U glmt -d postgres -c "CREATE DATABASE ozone OWNER glmt;"
psql -U glmt -d postgres -c "ALTER ROLE glmt WITH PASSWORD '<generate one>';"   # TCP connections need password auth, unlike the local peer-auth socket
```

**Config** (`/home/glmt/ozone-src/services/ozone/.env`, gitignored by virtue of living outside the repo):

```
OZONE_PORT=14831
OZONE_PUBLIC_URL=https://label.goat.navy
OZONE_SERVER_DID=<same LABELLER_DID as the rest of the project -- no new identity needed>
OZONE_DB_POSTGRES_URL=postgresql://glmt:<password>@127.0.0.1:5432/ozone
OZONE_DB_MIGRATE=1
OZONE_APPVIEW_URL=https://api.bsky.app
OZONE_APPVIEW_DID=did:web:api.bsky.app
OZONE_DID_PLC_URL=https://plc.directory
OZONE_ADMIN_PASSWORD=<generate one -- this is Ozone's own admin console password, unrelated to the bsky account password>
OZONE_SIGNING_KEY_HEX=<same LABELLER_SIGNING_KEY as the rest of the project>
OZONE_ADMIN_DIDS=<same LABELLER_DID>
```

Reuses the existing labeler identity end to end — same DID, same signing key already registered on its `#atproto_label` verification method. `OZONE_DB_MIGRATE=1` runs all pending migrations on every startup (idempotent, cheap once caught up — no separate migration step needed). Port matches what the tunnel/nginx already forward (see `deploy/README.md`), so nothing on that side needed to change when switching from `server.mjs`.

**Run:** `node --env-file=.env --enable-source-maps api.ts` from `services/ozone/` (real entrypoint — package.json's own `"start"` script points at a `dist/bin.js` that doesn't actually exist in this repo, don't use it).

## 10. Ozone bridge (`labeler/ozone_bridge.mjs`)

Ozone has no equivalent of `@skyware/labeler`'s `createLabels()` — every label goes through authenticated `tools.ozone.moderation.emitEvent`. Two non-obvious requirements this surfaced, both confirmed empirically against the real server, not assumed:

1. **Service-auth, not a session token.** A plain `com.atproto.server.createSession` access token gets rejected (`BadJwtType`). Needs a token from `com.atproto.server.getServiceAuth` scoped to `aud=<Ozone's DID>` and `lxm=tools.ozone.moderation.emitEvent` — the bridge mints a fresh one (5 min expiry) at the start of every poll cycle rather than trying to cache/refresh across cycles.
2. **A real `cid`, not just a `uri`.** `com.atproto.repo.strongRef` (the subject shape for labeling a specific post) marks `cid` as required, and Ozone validates request bodies strictly — a strongRef without one is rejected with 400. This is why `ingester/main.py` now captures `cid` from Jetstream commit events (previously discarded) and `live_export_labels.py` propagates it through to `pending-labels.jsonl`. Rows written before this change (or by anything that hasn't picked it up yet) have `"cid": null`; the bridge skips those rather than fetching the CID on demand (an extra network round-trip per post, and the post may no longer exist by then) — logged as a skip count, not silently dropped.

```bash
node --env-file=../.env ozone_bridge.mjs    # from labeler/
```

Needs `LABELLER_DID` and `APP_PASSWORD` in `.env` (the same account credentials used for the earlier `declare-labels.mjs`/`verify-signing-key.mjs` setup). Resolves the account's PDS from its DID document at startup, logs in once, and re-logs-in automatically on a 401 from an expired session.

For serving this publicly from a second box during development, see `deploy/README.md`.

## 11. Labeler server setup — `@skyware/labeler` (historical, no longer the active backend)

Superseded by Ozone (section 9) — kept here in case of a future rollback. `server.mjs` still exists and works; nothing about it broke, it was replaced because the library itself is archived/unmaintained and has an unresolved memory leak (see NOTES.md).

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

**Run:** `node --env-file=.env labeler/server.mjs` — listens on port 14831, ingests `pending-labels.jsonl` into its own SQLite (`labeler/labels.db`, gitignored, owned entirely by `@skyware/labeler` — no external DB hookup, see NOTES.md). **Run it supervised, not bare** (`./labeler/run_supervised.sh` or `deploy/labeler-server.service`) — it has a known unresolved memory leak (see NOTES.md).

## 12. Check overall status

```bash
python -m tools.status
```

One-shot report: which of the 5 long-running processes (ingester, live classifier export, bulk labeler, Ozone, Ozone bridge) are actually up, the SSH tunnel's systemd state, the ingester's firehose lag, how far behind each downstream stage's cursor is from its input file (posts.jsonl → pending-labels.jsonl → Ozone), the bulk labeler's row count and last-write freshness (flags if the process is running but hasn't written anything in 5+ minutes — this exact symptom was a real Ollama hang once), rotation cron/archive state, and GPU memory/utilization. No daemon, stdlib only, safe to run any time.

## 13. Query what's actually been labeled

```bash
python -m tools.query_labels                              # label frequency summary, all time
python -m tools.query_labels --since 24                    # summary, last 24h only
python -m tools.query_labels --uri "at://did:plc:.../app.bsky.feed.post/..."
python -m tools.query_labels --label uspol                 # most recent posts labeled uspol
python -m tools.query_labels --label uspol --recent 50 --since 6
```

Queries Ozone's `label` table in Postgres directly (reads the connection string from `/home/glmt/ozone-src/services/ozone/.env`, the same DB Ozone itself uses). Schema: `label(id, src, uri, cid, val, neg, cts, exp, sig, signingKeyId)`. (Historical note: before the Ozone migration this queried `labeler/labels.db`, `@skyware/labeler`'s own SQLite — see NOTES.md if that file still exists and you're not sure which is current.)

---

## Typical end-to-end flow for a new topic

1. Pick a topic name and a one-line definition.
2. Either: `llm_label.py` with that topic (fast, cheap, purpose-built binary data), **or** rely on the general-purpose `llm_freeform_label.py` + `consolidate_labels.py` pipeline if the topic already showed up in a freeform tagging pass.
3. `classifier/train.py --topic <name> ...` — check the printed F1. Below 0.7 usually means too few positive examples (rule of thumb from experience: aim for 500+ positives, not ~40) rather than a fundamentally bad topic.
4. Weights land in `classifier/weights/<topic>.pt`, ready to be loaded into a `TopicClassifier` (or compiled into a `StackedTopicHeads` for fast multi-topic serving).

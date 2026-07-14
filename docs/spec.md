# Bluesky Topic Labeler — Technical Specification

## Overview

A system that consumes the Bluesky firehose via Jetstream, classifies posts by topic using fine-tuned ML models, and serves those classifications as an AT Protocol labeler. Users can define custom topics by uploading example posts; the system trains a binary classifier per topic and emits labels that feed generators can query to filter content.

---

## Architecture

```
Jetstream ──► Ingester ──► Batch Queue ──► Classifier Worker ──► Label DB
                                                                       │
                                                               Labeler Server
                                                                       │
                                                              AppView / Feed Generators
```

Five components:

1. **Ingester** — consumes Jetstream, preprocesses post text, feeds batch queue
2. **Classifier Worker** — pulls batches, runs GPU inference, writes labels
3. **Label DB** — Postgres table of `(uri, label, seq)`
4. **Labeler Server** — AT Protocol labeler endpoints (`queryLabels`, `subscribeLabels`)
5. **Training Service** — accepts user-uploaded post corpora, fine-tunes classifiers, hot-swaps them into the worker

---

## Component 1: Ingester

**Language:** Python (asyncio)

**Connection:** WebSocket to `wss://jetstream2.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post` with zstd compression enabled.

**Cursor management:** Persist the last `time_us` value to disk (simple file) on every batch flush. On restart, reconnect with `cursor=<last_time_us - 5_000_000>` (5 second rewind) for gapless playback.

**Preprocessing:** Before queuing, apply Cardiff NLP's recommended preprocessing:
- Replace `@mentions` with `@user`
- Replace URLs starting with `http` with `http`
- Strip leading/trailing whitespace
- Skip posts with empty text after preprocessing
- Skip posts where `commit.operation != "create"` (don't classify edits/deletes)
- Store the AT-URI (`at://did:.../app.bsky.feed.post/rkey`) alongside the text

**Batching:** Accumulate posts into batches of 64. Flush the batch if either:
- 64 posts have accumulated, or
- 500ms has elapsed since the batch started

This ensures low latency on quiet periods without sacrificing GPU throughput on busy ones.

**Output:** Push batches as `[(uri, preprocessed_text), ...]` lists to a shared in-process queue (Python `asyncio.Queue`). Use a bounded queue (max 1000 batches) to apply backpressure if the classifier falls behind.

---

## Component 2: Classifier Worker

**Language:** Python (runs in a separate thread with its own event loop to avoid blocking the asyncio ingester)

**Model architecture:** Single shared encoder (`cardiffnlp/tweet-topic-latest-multi`) with N independent binary classification heads, one per topic. The encoder is frozen; only classification heads are updated during fine-tuning.

```python
class TopicClassifier(nn.Module):
    def __init__(self, encoder, topic_names):
        self.encoder = encoder  # frozen RoBERTa
        self.heads = nn.ModuleDict({
            name: nn.Linear(768, 1) for name in topic_names
        })
    
    def forward(self, input_ids, attention_mask, topic):
        with torch.no_grad():
            embeddings = self.encoder(input_ids, attention_mask).pooler_output
        return self.heads[topic](embeddings)
```

**Inference:** For each batch:
1. Tokenize all texts together (max 128 tokens, padding to batch max)
2. Single encoder forward pass (no_grad, shared across all heads)
3. Pass pooler output through each active head in parallel
4. Apply sigmoid; threshold at 0.5 per topic
5. For each (post, topic) where sigmoid > threshold, emit a label

**Label emission:** Write to Label DB, then signal the labeler server to push to subscribers.

**Model hot-swap:** When a new classifier head is trained, the training service writes new weights to a temp file and signals the worker via a threading.Event. The worker swaps in the new head between batches (not mid-batch). No restart required.

**GPU memory:** RoBERTa-base encoder is ~500MB VRAM. Each classification head is negligible (768 × 1 weights). 12GB VRAM easily handles 100+ topics simultaneously.

---

## Component 3: Label DB

**Engine:** Postgres

**Schema:**

```sql
CREATE TABLE labels (
    seq        BIGSERIAL PRIMARY KEY,
    uri        TEXT NOT NULL,
    label      TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX labels_uri_idx ON labels (uri);
CREATE INDEX labels_label_idx ON labels (label);
-- seq is already indexed as PRIMARY KEY
```

Note: `uri` stored as full AT-URI string. No hashing in the initial implementation (per decision to defer storage optimisation). At ~200 bytes per row including overhead, 10M labeled posts/day is ~2GB/day — acceptable for now, revisit if storage becomes a concern.

**Label negation:** AT Protocol supports negating (removing) a label by emitting a label record with `neg: true`. Store negations as separate rows with a `neg BOOLEAN NOT NULL DEFAULT FALSE` column (add this to the schema). The labeler server handles deduplication when serving.

Add `neg` column:

```sql
ALTER TABLE labels ADD COLUMN neg BOOLEAN NOT NULL DEFAULT FALSE;
```

---

## Component 4: Labeler Server

**Language:** TypeScript, using `@skyware/labeler`

**Setup:** The labeler needs a dedicated Bluesky account (not your personal account). Run `npx @skyware/labeler setup` to initialise it, which registers the signing key in the DID document.

**Endpoints served by `@skyware/labeler`:**
- `GET /xrpc/com.atproto.label.queryLabels` — queried by feed generators and AppView. Accepts `uriPatterns[]`, returns matching label records. The library handles this; you provide a DB query callback.
- `GET /xrpc/com.atproto.label.subscribeLabels` — SSE stream for AppView to stay in sync. Library handles cursor/replay logic using the `seq` column.
- `GET /xrpc/_health` — health check

**`@skyware/labeler` integration:**

```typescript
import { LabelerServer } from "@skyware/labeler";
import { pool } from "./db";

const server = new LabelerServer({
  did: process.env.LABELER_DID,
  signingKey: process.env.LABELER_SIGNING_KEY,
});

// Called when AppView queries labels for a URI
server.on("queryLabels", async (uriPatterns, cursor, limit) => {
  // Query Postgres, return label records
  // @skyware/labeler handles signing and CBOR formatting
});

server.start(14831);
```

**Cursor reliability:** The labeler server must track the AppView's subscription cursor. If the labeler restarts, the AppView reconnects with its last known cursor, and the labeler replays from that `seq` value. The `seq BIGSERIAL` in Postgres gives us this for free — always query `WHERE seq > $cursor ORDER BY seq ASC`.

**Port:** 14831 (default for `@skyware/labeler`). Expose via reverse proxy (nginx/caddy) with TLS.

---

## Component 5: Training Service

**Language:** Python

**Trigger:** HTTP endpoint `POST /train` accepting a JSON body:

```json
{
  "topic": "uspol",
  "positive_uris": ["at://...", "at://..."],
  "negative_uris": ["at://..."]  // optional; if empty, sample randomly from DB
}
```

**Training flow:**

1. Fetch post text for each URI from Bluesky AppView (`app.bsky.feed.getPosts`, batched 25 at a time)
2. Preprocess text (same pipeline as ingester)
3. If no negative examples provided, sample 2× the positive count from recent firehose posts (stored in a rolling 48h buffer in Postgres)
4. Split 80/20 train/validation
5. Fine-tune classification head:
   - Freeze encoder entirely
   - Train only `nn.Linear(768, 1)` for this topic
   - `BCEWithLogitsLoss`, AdamW, lr=2e-4, batch size 32, up to 10 epochs with early stopping on validation F1
   - On GPU: completes in seconds to low minutes
6. Evaluate: report precision, recall, F1 on validation set
7. If F1 > 0.7 (configurable threshold), save weights and hot-swap into classifier worker
8. If F1 < threshold, return the metrics to the caller with a warning rather than deploying

**Rolling buffer for negatives:** The ingester writes all post texts (not just labeled ones) to a `posts_buffer` table with a 48h TTL. Used as the negative sampling pool for training.

```sql
CREATE TABLE posts_buffer (
    uri        TEXT PRIMARY KEY,
    text       TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Clean up old posts via pg_cron or a periodic job:
DELETE FROM posts_buffer WHERE created_at < NOW() - INTERVAL '48 hours';
```

**User-facing workflow:**

1. User assembles a list of example post URIs (e.g. by searching Bluesky, or using the Bluesky web UI to collect examples)
2. User POSTs them to the training endpoint
3. System responds with training metrics
4. If metrics are good, classifier is live within seconds
5. User subscribes their feed generator to query this labeler for the new topic

---

## Feed Generator Integration

The feed generator (separate service, not specified here) queries this labeler via `queryLabels` when assembling a feed page. Workflow:

1. Fetch N pages from upstream feed (spacecowboy's "For You" etc.)
2. For each post URI, batch-query `queryLabels` on this labeler
3. For each labeled post, apply the user's slider value as a probabilistic drop rate: `if label == "uspol" and random() > slider_value: skip`
4. Return the surviving posts as the feed skeleton

The slider value (0.0–1.0) is stored per-user in the feed generator's own DB and updated via a small companion web UI.

---

## Infrastructure

**Single box deployment** (your GPU machine):

| Service | Language | Process manager |
|---|---|---|
| Ingester | Python | systemd |
| Classifier Worker | Python | systemd (with GPU access) |
| Labeler Server | TypeScript/Node | systemd |
| Training Service | Python | systemd |
| Postgres | — | systemd |
| Nginx | — | systemd |

**Environment variables:**
- `LABELER_DID` — the DID of your labeler account
- `LABELER_SIGNING_KEY` — Ed25519 private key (hex), generated during setup
- `DATABASE_URL` — Postgres connection string

**TLS:** Certbot/Let's Encrypt via nginx. The labeler must be reachable over HTTPS for Bluesky's AppView to subscribe to it.

---

## Data Flow Summary

```
Bluesky network
      │
      ▼ (WebSocket, ~850MB/day)
Jetstream
      │
      ▼
Ingester (Python asyncio)
  - preprocess text
  - batch into groups of 64
  - write to posts_buffer (48h TTL)
      │
      ▼ (asyncio.Queue)
Classifier Worker (Python + PyTorch)
  - encoder forward pass (GPU)
  - N classification heads
  - threshold sigmoid outputs
      │
      ▼ (Postgres INSERT)
labels table (seq, uri, label, neg)
      │
      ├──► subscribeLabels SSE stream → AppView (ongoing)
      └──► queryLabels responses → Feed generators (on demand)
```

---

## Open Questions / Deferred Decisions

- **Storage optimisation:** Deferred. If label storage becomes a problem, implement the URI hashing + on-demand signing approach discussed during design.
- **Multi-user:** The training endpoint and slider are currently single-user. Multi-user would require per-user classifier heads and per-user slider state — feasible but adds complexity.
- **Backfill:** Labels only apply to posts seen after the labeler starts. No backfill of historical posts unless explicitly triggered.
- **Language filtering:** Cardiff NLP's topic model is English-only. Consider filtering by post language (`commit.record.langs`) in the ingester before classifying.
- **Rate limiting on training:** Unlimited retraining could cause GPU contention. Consider a simple queue and cooldown per topic.
- **Topic namespace:** Label values like `"uspol"` are global on your labeler. Consider namespacing if the system grows: `"topic:uspol"`, `"topic:pets"` etc.

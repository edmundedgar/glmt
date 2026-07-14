// Starts the actual labeler server, then continuously polls
// pending-labels.jsonl (appended to live by
// classifier/live_export_labels.py) for new rows and ingests them via
// createLabels -- so freshly classified posts flow through to
// subscribeLabels subscribers in near-real-time, not just once at
// startup. Serves queryLabels/subscribeLabels via @skyware/labeler's own
// SQLite-backed storage (labels.db in this directory).
//
// pending-labels.jsonl only ever grows (it's an append-only log fed by a
// separately-running Python process) and can reach hundreds of thousands
// of lines over a long run. An earlier version of this poll loop re-read
// and re-parsed the WHOLE file on every cycle -- that got more expensive
// every 10s as the file grew, and eventually crashed the process with
// "JavaScript heap out of memory" after ingesting ~590k lines over a long
// run. This version tracks a byte offset and only reads bytes appended
// since the last poll, so cost per cycle stays roughly constant
// regardless of how large the file has grown.
//
// Usage:
//   node --env-file=../.env server.mjs

import { open, readFile, writeFile, rename } from "node:fs/promises";
import { LabelerServer } from "@skyware/labeler";

const PORT = 14831;
const POLL_INTERVAL_MS = 10_000;
const PENDING_PATH = new URL("./pending-labels.jsonl", import.meta.url).pathname;
const OFFSET_PATH = new URL("./.ingest-offset", import.meta.url).pathname;

const server = new LabelerServer({
  did: process.env.LABELLER_DID,
  signingKey: process.env.LABELLER_SIGNING_KEY,
  dbPath: new URL("./labels.db", import.meta.url).pathname,
});

async function loadOffset() {
  try {
    return parseInt((await readFile(OFFSET_PATH, "utf8")).trim(), 10) || 0;
  } catch {
    return 0;
  }
}

async function saveOffset(n) {
  // write-then-rename so a crash mid-write can't corrupt the offset file
  const tmp = `${OFFSET_PATH}.tmp`;
  await writeFile(tmp, String(n));
  await rename(tmp, OFFSET_PATH);
}

let byteOffset = await loadOffset();
console.log(`resuming pending-labels.jsonl ingest from byte offset ${byteOffset}`);
let leftover = ""; // a trailing partial line held back from the previous read

async function ingestNewLines() {
  let handle;
  try {
    handle = await open(PENDING_PATH, "r");
  } catch {
    return; // file doesn't exist yet
  }

  try {
    const stat = await handle.stat();
    if (stat.size < byteOffset) {
      // file was truncated/replaced (e.g. wiped for a clean restart) --
      // resync from the beginning instead of silently going stale.
      console.log(`pending-labels.jsonl shrank (${stat.size} < ${byteOffset}) -- resyncing from start`);
      byteOffset = 0;
      leftover = "";
      await saveOffset(byteOffset);
    }
    if (stat.size <= byteOffset) return; // nothing new

    const length = stat.size - byteOffset;
    const buffer = Buffer.alloc(length);
    await handle.read(buffer, 0, length, byteOffset);
    byteOffset = stat.size;
    await saveOffset(byteOffset);

    const chunk = leftover + buffer.toString("utf8");
    const lines = chunk.split("\n");
    leftover = lines.pop() ?? ""; // last element may be a not-yet-flushed partial line

    const rows = lines.filter(Boolean).map((line) => JSON.parse(line));
    if (rows.length === 0) return;

    const byUri = new Map();
    for (const row of rows) {
      if (!byUri.has(row.uri)) byUri.set(row.uri, []);
      byUri.get(row.uri).push(row.label);
    }

    let created = 0;
    for (const [uri, labels] of byUri) {
      await server.createLabels({ uri }, { create: labels });
      created += labels.length;
    }
    console.log(`[${new Date().toISOString()}] ingested ${created} new labels across ${byUri.size} URIs (byte offset ${byteOffset})`);
  } finally {
    await handle.close();
  }
}

await ingestNewLines();

server.start(PORT, (err, address) => {
  if (err) throw err;
  console.log(`labeler server listening on ${address}`);
});

setInterval(() => {
  ingestNewLines().catch((err) => console.error("ingestNewLines error:", err));
}, POLL_INTERVAL_MS);

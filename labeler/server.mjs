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
// Several things bounded here, all found the hard way after the DB grew
// past ~2M rows and a single ingest cycle started taking longer than
// POLL_INTERVAL_MS:
//   1. Each cycle's read is capped (MAX_BYTES_PER_CYCLE) rather than
//      reading everything new in one shot -- a long outage (this server
//      down while classifier/live_export_labels.py keeps writing) can
//      leave hundreds of MB unread, and reading + buffering all of it at
//      once is the same class of unbounded-memory problem as above, just
//      not fully closed off by the byte-offset fix alone.
//   2. The poll loop reschedules itself only after the current cycle
//      finishes (setTimeout, not setInterval). setInterval fires on a
//      fixed clock regardless of whether the previous async callback is
//      still running -- once cycles started taking >10s, new ticks piled
//      up *concurrently*, racing on the shared byteOffset state.
//   3. The createLabels loop yields back to the event loop every
//      YIELD_EVERY_N_LABELS calls (a bare `await setImmediate` spin), in
//      case a tight run of sequential awaits is starving other event loop
//      phases. Helps, but did NOT fully fix item 4 below on its own.
//
// UNRESOLVED as of this writing: even with all three of the above in
// place, this process still eventually OOMs ("FATAL ERROR: ... JavaScript
// heap out of memory") during sustained heavy catch-up (multiple millions
// of rows in labels.db). Confirmed NOT explained by any of: createLabels()
// itself (benchmarked in isolation against the real, large DB -- stayed
// at a constant ~1.1ms/call, no memory growth, even after server.start()
// had been called), the file-read/JSON-parse/Map-building step (also
// benchmarked in isolation -- ~15ms total for a full capped-size cycle),
// or WAL checkpoint cost (manually checkpointed the real DB -- instant).
// A heartbeat timer added purely to check whether the event loop itself
// was frozen showed it mostly ticking on schedule, but with an
// intermittent 80+ second stall before the eventual OOM crash -- so
// something IS blocking hard some of the time, likely inside
// @skyware/labeler's own internals (LabelerServer instance state,
// possibly something in emitLabel's per-call frame encoding, or
// something WebSocket-related) given none of our own code reproduces it
// in isolation. Root cause not found. Mitigated, not fixed: run this
// under something that restarts it automatically (see
// deploy/labeler-server.service) so a crash means a brief gap in service
// rather than silent extended downtime, and MAX_BYTES_PER_CYCLE is kept
// small to reduce how much work (and therefore how much of whatever is
// leaking) happens per cycle. If picking this up again: bisect by
// swapping @skyware/labeler versions, or instrument emitLabel/formatLabel
// directly rather than our own polling code, since everything on our side
// of the library boundary has now been benchmarked clean.
//
// Usage:
//   node --env-file=../.env server.mjs

import { open, readFile, writeFile, rename } from "node:fs/promises";
import { LabelerServer } from "@skyware/labeler";

const PORT = 14831;
const POLL_INTERVAL_MS = 10_000;
const MAX_BYTES_PER_CYCLE = 256 * 1024; // deliberately small -- see the OOM note above; less work per cycle, more (and cheaper) restarts if it crashes
const YIELD_EVERY_N_LABELS = 100; // let the event loop breathe periodically during a big cycle
const PENDING_PATH = new URL("./pending-labels.jsonl", import.meta.url).pathname;
const OFFSET_PATH = new URL("./.ingest-offset", import.meta.url).pathname;

const yieldToEventLoop = () => new Promise((resolve) => setImmediate(resolve));

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
      await saveOffset(byteOffset);
    }
    if (stat.size <= byteOffset) return; // nothing new

    const length = Math.min(stat.size - byteOffset, MAX_BYTES_PER_CYCLE);
    const buffer = Buffer.alloc(length);
    await handle.read(buffer, 0, length, byteOffset);
    const text = buffer.toString("utf8");

    // Only ever advance byteOffset to a confirmed line boundary within
    // what was just read -- never past a partial trailing line. A capped
    // read essentially never lands exactly on a "\n", so there's always
    // some trailing fragment; rather than holding it in an in-memory
    // `leftover` variable (the previous approach), just don't claim those
    // bytes yet and let the next cycle re-read them from the file, which
    // is the source of truth. This is what makes an abrupt kill -9 safe:
    // whatever byteOffset is on disk always points exactly at a line
    // boundary, so there's no separate fragile state that can go stale
    // across a restart. (An in-memory `leftover` variable NOT persisted
    // across restarts, combined with a byteOffset that WAS persisted past
    // the fragment it held, corrupted a real line after a restart here --
    // "at://did:plc:..." got split into a lost "{"uri": "at://did:" prefix
    // and an orphaned "plc:..." remainder that failed to parse as JSON.)
    const lastNewline = text.lastIndexOf("\n");
    if (lastNewline === -1) return; // no complete line in this chunk yet -- try again next cycle

    const usableText = text.slice(0, lastNewline + 1);
    byteOffset += Buffer.byteLength(usableText, "utf8");
    await saveOffset(byteOffset);

    const rows = usableText.split("\n").filter(Boolean).map((line) => JSON.parse(line));
    if (rows.length === 0) return;

    const byUri = new Map();
    for (const row of rows) {
      if (!byUri.has(row.uri)) byUri.set(row.uri, []);
      byUri.get(row.uri).push(row.label);
    }

    let created = 0;
    let sinceYield = 0;
    for (const [uri, labels] of byUri) {
      await server.createLabels({ uri }, { create: labels });
      created += labels.length;
      sinceYield += labels.length;
      if (sinceYield >= YIELD_EVERY_N_LABELS) {
        sinceYield = 0;
        await yieldToEventLoop();
      }
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

async function pollLoop() {
  try {
    await ingestNewLines();
  } catch (err) {
    console.error("ingestNewLines error:", err);
  }
  setTimeout(pollLoop, POLL_INTERVAL_MS);
}

setTimeout(pollLoop, POLL_INTERVAL_MS);

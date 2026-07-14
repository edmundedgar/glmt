// Bridges classifier/live_export_labels.py's output (labeler/pending-labels.jsonl)
// into Ozone via authenticated tools.ozone.moderation.emitEvent calls --
// the replacement for server.mjs's direct createLabels() calls into
// @skyware/labeler's own SQLite, now that Ozone is the labeler backend.
//
// Ozone doesn't expose anything like createLabels(): every label has to go
// through emitEvent, authenticated as a real account. Two things this
// requires that server.mjs never had to deal with:
//
//   1. Service-auth, not a plain session token. A regular
//      com.atproto.server.createSession accessJwt gets rejected with
//      "BadJwtType: Invalid jwt type \"at+jwt\"" -- confirmed empirically.
//      Cross-service calls need a token from com.atproto.server.getServiceAuth,
//      scoped to a specific aud (Ozone's DID) and lxm (the exact method).
//      These default to a 60s expiry, so a fresh one is minted at the start
//      of every cycle here rather than cached/reused across cycles --
//      simplest thing that can't go stale mid-batch.
//   2. A real `cid`, not just a `uri`. Ozone validates request bodies against
//      the lexicon, and com.atproto.repo.strongRef marks cid as required --
//      confirmed empirically (a strongRef without cid is rejected with 400
//      InvalidRequest). pending-labels.jsonl rows written before the ingester
//      started capturing cid (see ingester/main.py) won't have one; those
//      rows are skipped here rather than guessed at or fetched on demand
//      (fetching per-post would mean an extra network round-trip per label,
//      and the post may no longer exist by the time we get to it).
//
// Usage:
//   node --env-file=../.env ozone_bridge.mjs

import { open, readFile, writeFile, rename } from "node:fs/promises";

const OZONE_URL = process.env.OZONE_URL ?? "http://127.0.0.1:14831";
const POLL_INTERVAL_MS = 10_000;
const MAX_BYTES_PER_CYCLE = 2 * 1024 * 1024; // same reasoning as server.mjs: bounds memory and per-cycle latency regardless of backlog size
const SERVICE_AUTH_EXP_SECONDS = 300; // comfortably longer than one bounded cycle should ever take

const PENDING_PATH = new URL("./pending-labels.jsonl", import.meta.url).pathname;
const OFFSET_PATH = new URL("./.ozone-ingest-offset", import.meta.url).pathname;

const did = process.env.LABELLER_DID;
const password = process.env.APP_PASSWORD;
if (!did || !password) {
  throw new Error("LABELLER_DID and APP_PASSWORD must be set (via --env-file=../.env)");
}

async function resolvePds(did) {
  const res = await fetch(`https://plc.directory/${encodeURIComponent(did)}`);
  if (!res.ok) throw new Error(`could not resolve ${did}: ${res.status}`);
  const doc = await res.json();
  const svc = doc.service?.find((s) => s.id === "#atproto_pds");
  if (!svc) throw new Error(`no PDS service entry in DID document for ${did}`);
  return svc.serviceEndpoint;
}

const pds = await resolvePds(did);
console.log(`resolved PDS: ${pds}`);

let session = null; // { accessJwt } -- refreshed by re-login on demand, not proactively

async function login() {
  const res = await fetch(`${pds}/xrpc/com.atproto.server.createSession`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ identifier: did, password }),
  });
  if (!res.ok) throw new Error(`login failed: ${res.status} ${await res.text()}`);
  const data = await res.json();
  session = { accessJwt: data.accessJwt };
  console.log("logged in, session refreshed");
}

async function getServiceAuthToken() {
  if (!session) await login();
  const exp = Math.floor(Date.now() / 1000) + SERVICE_AUTH_EXP_SECONDS;
  const params = new URLSearchParams({
    aud: did, // OZONE_SERVER_DID -- our own labeler DID, since we run Ozone ourselves
    lxm: "tools.ozone.moderation.emitEvent",
    exp: String(exp),
  });
  let res = await fetch(`${pds}/xrpc/com.atproto.server.getServiceAuth?${params}`, {
    headers: { Authorization: `Bearer ${session.accessJwt}` },
  });
  if (res.status === 401) {
    // session expired -- re-login once and retry
    await login();
    res = await fetch(`${pds}/xrpc/com.atproto.server.getServiceAuth?${params}`, {
      headers: { Authorization: `Bearer ${session.accessJwt}` },
    });
  }
  if (!res.ok) throw new Error(`getServiceAuth failed: ${res.status} ${await res.text()}`);
  return (await res.json()).token;
}

async function emitLabelEvent(serviceJwt, uri, cid, labels) {
  const body = {
    event: {
      $type: "tools.ozone.moderation.defs#modEventLabel",
      createLabelVals: labels,
      negateLabelVals: [],
    },
    subject: { $type: "com.atproto.repo.strongRef", uri, cid },
    createdBy: did,
  };
  const res = await fetch(`${OZONE_URL}/xrpc/tools.ozone.moderation.emitEvent`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${serviceJwt}` },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`emitEvent failed for ${uri}: ${res.status} ${await res.text()}`);
}

async function loadOffset() {
  try {
    return parseInt((await readFile(OFFSET_PATH, "utf8")).trim(), 10) || 0;
  } catch {
    return 0;
  }
}

async function saveOffset(n) {
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
      console.log(`pending-labels.jsonl shrank (${stat.size} < ${byteOffset}) -- resyncing from start`);
      byteOffset = 0;
      await saveOffset(byteOffset);
    }
    if (stat.size <= byteOffset) return; // nothing new

    const length = Math.min(stat.size - byteOffset, MAX_BYTES_PER_CYCLE);
    const buffer = Buffer.alloc(length);
    await handle.read(buffer, 0, length, byteOffset);
    const text = buffer.toString("utf8");

    // Only ever advance byteOffset to a confirmed line boundary within what
    // was just read -- see server.mjs for why (an in-memory "leftover"
    // variable for a partial trailing line corrupted a real row here once,
    // after a restart lost that unpersisted state while the persisted
    // offset had already moved past it).
    const lastNewline = text.lastIndexOf("\n");
    if (lastNewline === -1) return;

    const usableText = text.slice(0, lastNewline + 1);
    byteOffset += Buffer.byteLength(usableText, "utf8");
    await saveOffset(byteOffset);

    const rows = usableText.split("\n").filter(Boolean).map((line) => JSON.parse(line));
    if (rows.length === 0) return;

    const byUri = new Map(); // uri -> { cid, labels: Set }
    let skippedNoCid = 0;
    for (const row of rows) {
      if (!row.cid) {
        skippedNoCid++;
        continue;
      }
      if (!byUri.has(row.uri)) byUri.set(row.uri, { cid: row.cid, labels: new Set() });
      byUri.get(row.uri).labels.add(row.label);
    }
    if (byUri.size === 0) {
      if (skippedNoCid > 0) console.log(`skipped ${skippedNoCid} row(s) with no cid (pre-dates ingester cid capture)`);
      return;
    }

    const serviceJwt = await getServiceAuthToken();
    let emitted = 0;
    let failed = 0;
    for (const [uri, { cid, labels }] of byUri) {
      try {
        await emitLabelEvent(serviceJwt, uri, cid, [...labels]);
        emitted += labels.size;
      } catch (err) {
        failed++;
        console.error(`emitEvent error: ${err.message}`);
      }
    }
    console.log(
      `[${new Date().toISOString()}] emitted ${emitted} labels across ${byUri.size} URIs` +
        (skippedNoCid > 0 ? ` (skipped ${skippedNoCid} rows with no cid)` : "") +
        (failed > 0 ? ` (${failed} URIs failed)` : "") +
        ` (byte offset ${byteOffset})`,
    );
  } finally {
    await handle.close();
  }
}

await ingestNewLines();

async function pollLoop() {
  try {
    await ingestNewLines();
  } catch (err) {
    console.error("ingestNewLines error:", err);
  }
  setTimeout(pollLoop, POLL_INTERVAL_MS);
}

setTimeout(pollLoop, POLL_INTERVAL_MS);
console.log("ozone_bridge running");

// One-off check: does the LABELLER_SIGNING_KEY in .env actually match the
// signing key declared in our did:plc document, and does it actually
// produce the signatures we see on labels served by our own queryLabels
// endpoint? All three should agree; if any diverge that would explain
// labels not being trusted/displayed by clients.
//
// Usage:
//   node --env-file=../.env verify-signing-key.mjs

import { secp256k1 as k256 } from "@noble/curves/secp256k1";
import { sha256 } from "@noble/hashes/sha256";
import { encode as cborEncode } from "@atcute/cbor";
import * as ui8 from "uint8arrays";

const SECP256K1_DID_PREFIX = new Uint8Array([0xe7, 0x01]);
const DID = process.env.LABELLER_DID;
const PRIVATE_KEY_HEX = process.env.LABELLER_SIGNING_KEY;

if (!DID || !PRIVATE_KEY_HEX) {
  console.error("LABELLER_DID / LABELLER_SIGNING_KEY not set");
  process.exit(1);
}

// --- 1. derive the pubkey from the private key in .env ---
const privKeyBytes = ui8.fromString(PRIVATE_KEY_HEX, "hex");
const pubKeyCompressed = k256.getPublicKey(privKeyBytes, true); // compressed, 33 bytes
const prefixedBytes = ui8.concat([SECP256K1_DID_PREFIX, pubKeyCompressed]);
const derivedMultibase = "z" + ui8.toString(prefixedBytes, "base58btc");
console.log("derived pubkey (did:key multibase):", derivedMultibase);

// --- 2. fetch the did:plc document and pull out the #atproto key ---
const plcRes = await fetch(`https://plc.directory/${encodeURIComponent(DID)}`);
if (!plcRes.ok) throw new Error(`could not resolve ${DID}: ${plcRes.status}`);
const didDoc = await plcRes.json();
// NOTE: labels are signed with the dedicated #atproto_label key, NOT the
// account's regular #atproto (PDS/repo) signing key -- these are two
// separate verification methods in the DID document.
const vm = didDoc.verificationMethod?.find(
  (m) => m.id === `${DID}#atproto_label` || m.id === "#atproto_label",
);
if (!vm) throw new Error("no #atproto_label verificationMethod in DID document");
console.log("did:plc declares   (did:key multibase):", vm.publicKeyMultibase, "(type:", vm.type, ")");

const didDocMatchesDerived = vm.publicKeyMultibase === derivedMultibase;
console.log(didDocMatchesDerived ? "MATCH: .env key matches did:plc document" : "MISMATCH: .env key does NOT match did:plc document");

// --- 3. pull a real signed label from our own queryLabels endpoint and verify it ---
const sampleUri = process.argv[2];
if (!sampleUri) {
  console.log("\n(no sample URI passed as argv[2] -- skipping live signature check)");
  process.exit(didDocMatchesDerived ? 0 : 1);
}

const qlRes = await fetch(
  `https://label.goat.navy/xrpc/com.atproto.label.queryLabels?uriPatterns=${encodeURIComponent(sampleUri)}`,
);
if (!qlRes.ok) throw new Error(`queryLabels failed: ${qlRes.status}`);
const { labels } = await qlRes.json();
if (!labels?.length) throw new Error("no labels returned for that URI");
const label = labels[0];
console.log(`\nchecking signature on label id=${label.id} val=${label.val} uri=${label.uri}`);

const toSign = { src: label.src, uri: label.uri, val: label.val, cts: label.cts, neg: !!label.neg, ver: label.ver };
if (label.cid) toSign.cid = label.cid;
if (label.exp) toSign.exp = label.exp;

const msgBytes = cborEncode(toSign);
const msgHash = sha256(msgBytes);
const sigBytes = ui8.fromString(label.sig.$bytes, "base64");

const validAgainstDerived = k256.verify(sigBytes, msgHash, pubKeyCompressed, { lowS: false });
console.log(validAgainstDerived ? "MATCH: signature verifies against .env-derived pubkey" : "MISMATCH: signature does NOT verify against .env-derived pubkey");

const didDocPubkeyBytes = prefixedBytes.slice(2).length === 33
  ? (() => {
      const raw = ui8.fromString(vm.publicKeyMultibase.slice(1), "base58btc");
      return raw.slice(2); // strip the 0xe7 0x01 prefix
    })()
  : null;
if (didDocPubkeyBytes) {
  const validAgainstDidDoc = k256.verify(sigBytes, msgHash, didDocPubkeyBytes, { lowS: false });
  console.log(validAgainstDidDoc ? "MATCH: signature verifies against did:plc-declared pubkey" : "MISMATCH: signature does NOT verify against did:plc-declared pubkey");
}

process.exit(didDocMatchesDerived && validAgainstDerived ? 0 : 1);

// Syncs labels.json (the source of truth for this labeler's label taxonomy)
// to the account's app.bsky.labeler.service record. Re-run any time
// labels.json changes -- this always pushes the FULL set (the underlying
// operation is a full-record overwrite, there's no "add one" at the
// protocol level).
//
// Usage:
//   node --env-file=../.env declare-labels.mjs
//   node --env-file=../.env declare-labels.mjs --dry-run

import { readFile } from "node:fs/promises";
import { setLabelerLabelDefinitions, getLabelerLabelDefinitions } from "@skyware/labeler/scripts";

const HANDLE = "label.goat.navy";

// Defaults match what the setup wizard used for the first two labels
// (uspol, death) -- override per-entry in labels.json if a label needs
// different behavior (e.g. blurs/severity for something more sensitive
// than a topic tag).
const DEFAULTS = {
  severity: "inform",
  blurs: "none",
  defaultSetting: "warn",
  adultOnly: false,
};

function toLexiconDefinition(entry) {
  const { identifier, name, description, ...overrides } = entry;
  return {
    identifier,
    ...DEFAULTS,
    ...overrides,
    locales: [{ lang: "en", name, description: description ?? "" }],
  };
}

const dryRun = process.argv.includes("--dry-run");

const raw = JSON.parse(await readFile(new URL("./labels.json", import.meta.url), "utf8"));
const definitions = raw.map(toLexiconDefinition);

const credentials = {
  identifier: HANDLE,
  password: process.env.APP_PASSWORD,
};
if (!credentials.password) {
  throw new Error("APP_PASSWORD not set -- run with e.g. `node --env-file=../.env declare-labels.mjs`");
}

console.log(`labels.json declares ${definitions.length} label(s): ${definitions.map((d) => d.identifier).join(", ")}`);

if (dryRun) {
  console.log("--dry-run: not pushing. Would set:");
  console.log(JSON.stringify(definitions, null, 2));
} else {
  await setLabelerLabelDefinitions(credentials, definitions);
  console.log("pushed.");
}

const confirmed = await getLabelerLabelDefinitions(credentials);
console.log(`account now reports ${confirmed.length} label(s): ${confirmed.map((d) => d.identifier).join(", ")}`);

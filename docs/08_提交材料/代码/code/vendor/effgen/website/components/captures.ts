// The recorded `effgen` sessions the product pages show, read out of the JSON
// `scripts/gen_capture_data.py` builds from `data/captures/`.
//
// Every frame on `/code`, `/cli`, `/models` and `/production` comes through here. Nothing on either page
// is a transcription of a session: a frame either resolves to one of these
// recordings or it does not exist, which is what stops a plausible-looking
// example that was never run from reaching a page.

import codeRaw from "@/data/captures.code.json";
import cliRaw from "@/data/captures.cli.json";
import modelsRaw from "@/data/captures.models.json";
import productionRaw from "@/data/captures.production.json";
import type { Capture, CapturedDocument, CaptureSet } from "@/data/captures.types";

const codeSet = codeRaw as unknown as CaptureSet;
const cliSet = cliRaw as unknown as CaptureSet;
const modelsSet = modelsRaw as unknown as CaptureSet;
const productionSet = productionRaw as unknown as CaptureSet;

function pick(set: CaptureSet, slug: string): Capture {
  const capture = set.captures[slug];
  if (!capture) {
    throw new Error(
      `No capture named "${slug}". Add it to COMMANDS in ` +
        "scripts/gen_capture_data.py and re-run the generator.",
    );
  }
  return capture;
}

function pickDocument(set: CaptureSet, slug: string): CapturedDocument {
  const document = set.documents[slug];
  if (!document) {
    throw new Error(
      `No captured document named "${slug}". Add it to JSON_DOCUMENTS in ` +
        "scripts/gen_capture_data.py and re-run the generator.",
    );
  }
  return document;
}

/** A recorded `effgen code` session, by slug. */
export const codeCapture = (slug: string): Capture => pick(codeSet, slug);

/** A recorded command-line session, by slug. */
export const cliCapture = (slug: string): Capture => pick(cliSet, slug);

/** A recorded run of one of the catalog commands, by slug. */
export const modelsCapture = (slug: string): Capture => pick(modelsSet, slug);

/** A recorded request to, or run against, a live server, by slug. */
export const productionCapture = (slug: string): Capture => pick(productionSet, slug);

/** A JSON document one of the commands emitted, parsed. */
export const codeDocument = (slug: string): CapturedDocument => pickDocument(codeSet, slug);
export const cliDocument = (slug: string): CapturedDocument => pickDocument(cliSet, slug);

export type { Capture, CapturedDocument } from "@/data/captures.types";

// The shape of `data/captures.code.json` and `data/captures.cli.json`, which
// `scripts/gen_capture_data.py` writes from the recorded sessions in
// `data/captures/`.
//
// Regenerate them — never edit them — with:
//
//   export PATH=/path/to/your/effgen/env/bin:$PATH   # or activate it however you do
//   python scripts/gen_capture_data.py
//
// `python scripts/gen_capture_data.py --check` exits non-zero when either file
// no longer matches the captures it was built from.

/** One run of text and the attributes the terminal printed it with. */
export type AnsiSpan = [attributes: string, text: string];

export interface Capture {
  /** The recording under `data/captures/`, kept as the record. */
  file: string;
  /** The command that produced it. Rendered beside every frame. */
  command: string;
  /** Of the recording, so a frame can be traced back to its bytes. */
  sha256: string;
  bytes: number;
  /** The capture with escape sequences resolved. What the reader saw. */
  text: string;
  lines: number;
  /** Present only where the colour is the subject: the four named themes. */
  spans?: AnsiSpan[][];
}

export interface CapturedDocument {
  file: string;
  command: string;
  sha256: string;
  /** The parsed JSON the command emitted. */
  document: Record<string, unknown>;
}

export interface CaptureSet {
  generated_at: string;
  captures: Record<string, Capture>;
  documents: Record<string, CapturedDocument>;
}

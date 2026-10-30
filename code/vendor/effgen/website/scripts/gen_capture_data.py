#!/usr/bin/env python3
"""Turn the captured `effgen` sessions in ``data/captures/`` into the JSON the
product pages render, and write ``data/captures.code.json`` and
``data/captures.cli.json``.

    python scripts/gen_capture_data.py            # write both files
    python scripts/gen_capture_data.py --check    # exit 1 if either is stale

Every file under ``data/captures/`` is the unedited stdout and stderr of the
command named in ``COMMANDS`` below, recorded on a real terminal against effGen
1.0.0. **Nothing here is retyped and nothing is rewritten.** Two lossless
transformations happen, and only these two:

* escape sequences are removed and a carriage-return rewrite (a progress line
  overwriting itself) is resolved to the text that was left on the screen, which
  is what a reader saw;
* for the four theme captures, the SGR colour codes are turned into per-span
  attributes instead of being discarded — a theme *is* colour, so a page that
  showed those four captures in one colour would be showing nothing.

The checked-in ``.ansi`` and ``.txt`` files are the record. If a frame on the
site ever disagrees with what the command does, the capture is retaken; it is
never edited to agree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CAPTURES = REPO / "data" / "captures"
OUT_CODE = REPO / "data" / "captures.code.json"
OUT_CLI = REPO / "data" / "captures.cli.json"
OUT_MODELS = REPO / "data" / "captures.models.json"
OUT_PRODUCTION = REPO / "data" / "captures.production.json"

# `slug: (file, the command that produced it, whether the colour is the point)`.
#
# The command is not decoration: it is what a reader needs in order to reproduce
# the frame, and it is rendered beside every capture on the page.
COMMANDS: dict[str, tuple[str, str, bool]] = {
    # ---- effgen code -----------------------------------------------------
    "code-loop": (
        "code-loop.ansi",
        'effgen code "Add title_case(text) to textkit/case.py: title-case each '
        "word, raise TypeError on a non-string. Re-export it from "
        "textkit/__init__.py. Add tests/test_case.py covering both cases. Then "
        'run pytest and tell me the result." -y -m openai:gpt-5.4',
        False,
    ),
    "code-plan": (
        "code-plan.ansi",
        'effgen code "Add a strip_accents(text) helper to textkit/accents.py and '
        're-export it from the package." --plan -m openai:gpt-5.4',
        False,
    ),
    "code-session": ("code-session.ansi", "effgen code -m gemini:gemini-3.1-flash-lite", False),
    "code-review": (
        "code-review.ansi",
        'effgen code --review -p "is this change consistent with AGENTS.md?" '
        "-m openai:gpt-5-nano",
        False,
    ),
    "code-review-nothing": ("code-review-nothing.ansi", "effgen code --review", False),
    "code-undo": ("code-undo.ansi", "effgen code --undo", False),
    "code-failed-actions": (
        "code-failed-actions.ansi",
        'effgen code "Add title_case(text) to textkit/case.py … Run pytest and '
        'report the result." -y -m openai:gpt-5-mini',
        False,
    ),
    "code-two-modes": ("code-two-modes.txt", "effgen code -p \"anything\" --plan --auto-edit", False),
    "code-review-norepo": (
        "code-review-norepo.txt",
        'effgen code --review -p "review this"   # outside a repository',
        False,
    ),
    "code-git-refusals": (
        "code-git-refusals.txt",
        "python -c 'from effgen.cli.code.git_actions import unsafe_shell_git; …'",
        False,
    ),
    "code-help": ("cli-help-code.txt", "effgen code --help", False),
    # ---- the command line ------------------------------------------------
    "cli-help-root": ("cli-help-root.txt", "effgen --help", False),
    "cli-top": (
        "cli-top.ansi",
        "effgen top --once",
        False,
    ),
    "cli-battle": (
        "cli-battle.ansi",
        'effgen battle "Explain a B-tree in two sentences." '
        "-m openai:gpt-5-nano,gemini:gemini-3.1-flash-lite --no-animation",
        False,
    ),
    "cli-theme-default": ("cli-theme-default.ansi", "effgen --theme default models info gemini:gemini-3.1-flash-lite", True),
    "cli-theme-high-contrast": ("cli-theme-high-contrast.ansi", "effgen --theme high-contrast models info gemini:gemini-3.1-flash-lite", True),
    "cli-theme-monochrome": ("cli-theme-monochrome.ansi", "effgen --theme monochrome models info gemini:gemini-3.1-flash-lite", True),
    "cli-theme-light": ("cli-theme-light.ansi", "effgen --theme light models info gemini:gemini-3.1-flash-lite", True),
    "cli-completion-bash": ("cli-completion-bash.txt", "effgen --completion bash", False),
    "cli-runs-list": ("cli-runs-list.ansi", "effgen runs list --limit 6", False),
    "cli-runs-show": ("cli-runs-show.ansi", "effgen runs show 7bd4e604266d", False),
    "cli-sessions-list": ("cli-sessions-list.ansi", "effgen sessions list", False),
    "cli-cost-today": ("cli-cost-today.ansi", "effgen cost today", False),
    "cli-quickstart-init": ("cli-quickstart-init.ansi", "effgen quickstart --init .", False),
    "cli-doctor": ("cli-doctor.ansi", "effgen doctor", False),
    "cli-models-info": ("cli-models-info.ansi", "effgen models info openai:gpt-5-mini", False),
    "cli-chat-session": ("cli-chat-session.ansi", "effgen chat -m gemini:gemini-3.1-flash-lite", False),
    "cli-report-refusal": ("cli-report-refusal.txt", "effgen report notaresult.json", False),
    # ---- the model catalog ------------------------------------------------
    "models-browse-default": (
        "models-browse-default.ansi", "effgen models browse --limit 10", False),
    "models-browse-free-tools": (
        "models-browse-free-tools.ansi", "effgen models browse --free -t --limit 10", False),
    "models-browse-vision": (
        "models-browse-vision.ansi",
        "effgen models browse --vision --min-context 200000 --sort context --desc --limit 10",
        False),
    "models-browse-cheap": (
        "models-browse-cheap.ansi",
        "effgen models browse -t --max-price-in 0.10 --sort price-in --limit 10", False),
    "models-browse-provider": (
        "models-browse-provider.ansi", "effgen models browse --provider cerebras", False),
    "models-browse-search": (
        "models-browse-search.ansi", "effgen models browse --search qwen --limit 10", False),
    "models-list": ("models-list.ansi", "effgen models list", False),
    # ---- running it for real --------------------------------------------
    "serve-unauthenticated": (
        "serve-unauthenticated.txt",
        "curl -s -o /dev/null -w 'HTTP %{http_code}\\n' http://127.0.0.1:8000/v1/models"
        " ; curl -s http://127.0.0.1:8000/v1/models",
        False),
    "serve-completion": (
        "serve-completion.txt",
        "curl -s http://127.0.0.1:8000/v1/chat/completions"
        " -H \"Authorization: Bearer $EFFGEN_API_KEY\" -H 'Content-Type: application/json'"
        " -d '{\"model\":\"openai:gpt-5-nano\",\"messages\":"
        "[{\"role\":\"user\",\"content\":\"Reply with the single word ok.\"}]}'",
        False),
    "serve-rate-limit": (
        "serve-rate-limit.txt",
        "for i in $(seq 1 20); do curl -s -o /dev/null -w '%{http_code} '"
        " http://127.0.0.1:8000/v1/models -H \"Authorization: Bearer $EFFGEN_API_KEY\";"
        " done   # the server was started with EFFGEN_RATE_LIMIT=10",
        False),
    "serve-audit": (
        "serve-audit.txt",
        "tail -4 \"$EFFGEN_AUDIT_DIR/$(date -u +%F).jsonl\"",
        False),
    "serve-metrics": ("serve-metrics.txt", "curl -s http://127.0.0.1:8000/metrics", False),
    "serve-loadtest": (
        "serve-loadtest.txt",
        "effgen loadtest --url http://127.0.0.1:8000 --model openai:gpt-5-nano"
        " --concurrency 4 --duration 20",
        False),
    "eval-gate-pass": (
        "eval-gate-pass.txt",
        "effgen eval --suite math -m openai:gpt-5-nano --max-cases 5 --fail-under 0.8",
        False),
    "eval-gate-fail": (
        "eval-gate-fail.txt",
        "effgen eval --suite math -m openai:gpt-5-nano --max-cases 5 --fail-under 1.01",
        False),
    "eval-baseline": (
        "eval-baseline.txt",
        "effgen eval --suite math -m openai:gpt-5-nano --max-cases 5 --compare-baseline",
        False),
    "cost-budget": (
        "cost-budget.txt",
        "effgen cost set-budget 5.00 ; effgen cost today",
        False),
}

# Which page each capture belongs to. A capture used by both is emitted twice;
# the two files are imported by different routes, so nothing is shared.
CODE_PAGE = {
    "code-loop", "code-plan", "code-session", "code-review", "code-review-nothing",
    "code-undo", "code-failed-actions", "code-two-modes", "code-review-norepo",
    "code-git-refusals", "code-help",
}
CLI_PAGE = {
    "cli-help-root", "cli-top", "cli-battle", "cli-theme-default",
    "cli-theme-high-contrast", "cli-theme-monochrome", "cli-theme-light",
    "cli-completion-bash", "cli-runs-list", "cli-runs-show", "cli-sessions-list",
    "cli-cost-today", "cli-quickstart-init", "cli-doctor", "cli-models-info",
    "cli-chat-session", "cli-report-refusal",
}
MODELS_PAGE = {
    "models-browse-default", "models-browse-free-tools", "models-browse-vision",
    "models-browse-cheap", "models-browse-provider", "models-browse-search",
    "models-list", "cli-models-info",
}
PRODUCTION_PAGE = {
    "serve-unauthenticated", "serve-completion", "serve-rate-limit", "serve-audit",
    "serve-metrics", "serve-loadtest", "eval-gate-pass", "eval-gate-fail",
    "eval-baseline", "cost-budget",
}

# JSON documents a page quotes rather than renders as a terminal frame.
JSON_DOCUMENTS = {
    "code-json": ("code-json.json", "effgen code -p \"list the public helpers this package exports\" --plan --json"),
    "cli-top-json": ("cli-top.json", "effgen top --json"),
}

SGR = re.compile(r"\x1b\[([0-9;]*)m")
OSC = re.compile(r"\x1b[\]P].*?(?:\x07|\x1b\\)", re.S)
CSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

# The eight ANSI colours, by the name the renderer knows them by. A theme maps
# its semantic roles onto these, so these are what a capture carries.
COLOR_NAMES = {
    30: "black", 31: "red", 32: "green", 33: "yellow",
    34: "blue", 35: "magenta", 36: "cyan", 37: "white",
    90: "bright-black", 91: "bright-red", 92: "bright-green", 93: "bright-yellow",
    94: "bright-blue", 95: "bright-magenta", 96: "bright-cyan", 97: "bright-white",
}


def resolve_screen(raw: str) -> str:
    """Remove escape sequences and settle carriage-return rewrites.

    A pty transcript carries `\\r\\n` line endings and, where a progress line
    redrew itself, several versions of one line separated by a bare `\\r`. What
    the reader saw is the last of them.
    """
    text = OSC.sub("", raw)
    text = CSI.sub("", text)
    text = text.replace("\r\n", "\n")
    lines = [line.split("\r")[-1] for line in text.split("\n")]
    return "\n".join(lines).rstrip() + "\n"


def to_spans(raw: str) -> list[list[list]]:
    """Turn a capture into per-line spans carrying the attributes it printed.

    Each span is ``[attrs, text]``: ``attrs`` is a short string built from
    ``b`` (bold), ``d`` (dim), ``i`` (italic), ``u`` (underline) and ``r``
    (reverse), optionally followed by ``:`` and a colour name. Nothing is
    inferred — an attribute is present only because the command emitted the code
    that sets it.
    """
    text = OSC.sub("", raw).replace("\r\n", "\n")

    bold = dim = italic = underline = reverse = False
    color: str | None = None
    lines: list[list[list]] = [[]]

    def attrs() -> str:
        flags = "".join(
            flag for flag, on in
            (("b", bold), ("d", dim), ("i", italic), ("u", underline), ("r", reverse))
            if on
        )
        return f"{flags}:{color}" if color else flags

    def emit(chunk: str) -> None:
        for index, piece in enumerate(chunk.split("\n")):
            if index:
                lines.append([])
            # A carriage return means the line was redrawn; keep what was left.
            if "\r" in piece:
                lines[-1] = []
                piece = piece.split("\r")[-1]
            if piece:
                lines[-1].append([attrs(), piece])

    position = 0
    for match in SGR.finditer(text):
        emit(text[position:match.start()])
        position = match.end()
        codes = [int(part or 0) for part in match.group(1).split(";")] or [0]
        index = 0
        while index < len(codes):
            code = codes[index]
            if code == 0:
                bold = dim = italic = underline = reverse = False
                color = None
            elif code == 1:
                bold = True
            elif code == 2:
                dim = True
            elif code == 3:
                italic = True
            elif code == 4:
                underline = True
            elif code == 7:
                reverse = True
            elif code == 22:
                bold = dim = False
            elif code == 23:
                italic = False
            elif code == 24:
                underline = False
            elif code == 27:
                reverse = False
            elif code in COLOR_NAMES:
                color = COLOR_NAMES[code]
            elif code == 39:
                color = None
            elif code == 38 and index + 2 < len(codes) and codes[index + 1] == 5:
                color = f"x{codes[index + 2]}"
                index += 2
            index += 1
    emit(text[position:])

    while lines and not lines[-1]:
        lines.pop()
    return lines


def build(slug: str) -> dict:
    file_name, command, wants_spans = COMMANDS[slug]
    path = CAPTURES / file_name
    raw_bytes = path.read_bytes()
    raw = raw_bytes.decode("utf-8", "replace")
    text = resolve_screen(raw)
    entry = {
        "file": file_name,
        "command": command,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "bytes": len(raw_bytes),
        "text": text,
        "lines": text.count("\n"),
    }
    if wants_spans:
        entry["spans"] = to_spans(raw)
    return entry


def collect() -> tuple[dict, dict, dict, dict]:
    code = {"captures": {}, "documents": {}}
    cli = {"captures": {}, "documents": {}}
    models = {"captures": {}, "documents": {}}
    production = {"captures": {}, "documents": {}}
    for slug in COMMANDS:
        entry = build(slug)
        if slug in CODE_PAGE:
            code["captures"][slug] = entry
        if slug in CLI_PAGE:
            cli["captures"][slug] = entry
        if slug in MODELS_PAGE:
            models["captures"][slug] = entry
        if slug in PRODUCTION_PAGE:
            production["captures"][slug] = entry

    for slug, (file_name, command) in JSON_DOCUMENTS.items():
        path = CAPTURES / file_name
        raw_bytes = path.read_bytes()
        entry = {
            "file": file_name,
            "command": command,
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "document": json.loads(raw_bytes.decode("utf-8")),
        }
        (code if slug.startswith("code") else cli)["documents"][slug] = entry

    return code, cli, models, production


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit 1 when a checked-in file no longer matches the captures")
    args = parser.parse_args()

    missing = [name for name, _, _ in COMMANDS.values() if not (CAPTURES / name).exists()]
    missing += [name for name, _ in JSON_DOCUMENTS.values() if not (CAPTURES / name).exists()]
    if missing:
        print("missing capture(s): " + ", ".join(sorted(missing)), file=sys.stderr)
        return 1

    code, cli, models, production = collect()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    failures = []
    for out, fresh in ((OUT_CODE, code), (OUT_CLI, cli), (OUT_MODELS, models),
                       (OUT_PRODUCTION, production)):
        payload = {"generated_at": stamp, **fresh}
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        if args.check:
            if not out.exists():
                failures.append(f"{out.relative_to(REPO)} does not exist")
                continue
            current = json.loads(out.read_text(encoding="utf-8"))
            current.pop("generated_at", None)
            if current != fresh:
                failures.append(f"{out.relative_to(REPO)} is stale")
            continue
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out.relative_to(REPO)} ({len(text):,} bytes)")

    if args.check:
        if failures:
            for line in failures:
                print(line, file=sys.stderr)
            return 1
        print("data/captures.*.json match data/captures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

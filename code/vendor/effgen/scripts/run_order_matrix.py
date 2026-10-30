#!/usr/bin/env python3
"""Run the offline test suite over several collection orders and report what moved.

A test that only passes because of what ran before it fails on somebody else's machine
weeks later, in an order nobody can reproduce. This driver runs the whole offline
suite in a single process once per order — collection order, reversed, and a shuffle
at each requested seed — writes a per-lane timing report for each run, and separates a
failure with a known cause (``tests/flake_register.toml``) from one without.

    scripts/run_order_matrix.py --outdir /tmp/orders
    scripts/run_order_matrix.py --outdir /tmp/orders --hermetic --seeds 1 2 3
    scripts/run_order_matrix.py --outdir /tmp/orders --orders fixed reversed

The exit code is non-zero when any run had a failure that is not in the register, so
the driver can be a CI step as it stands.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests._harness import flake_register  # noqa: E402
from tests._harness.lanes import lane_of_nodeid  # noqa: E402

DEFAULT_SELECTOR = "not gpu and not api and not live and not docker and not expensive"
DEFAULT_SEEDS = (20260720, 91001, 91002, 91003)

# A short-summary line names a node id: a path ending in .py, optionally followed by
# `::`. Captured log records start with the same two words, so the shape of what
# follows is what tells a result apart from a logger called "ERROR effgen.models...".
_OUTCOME = re.compile(r"^(FAILED|ERROR)\s+([\w./\\-]+\.py(?:::\S+)?)(?:\s|$)")
_COUNTS = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed|deselected)")


def _order_arguments(order: str, seed: int | None) -> tuple[list[str], dict[str, str]]:
    """Return the pytest arguments and environment for one order."""
    if order == "fixed":
        return [], {}
    if order == "reversed":
        return [], {"EFFGEN_TEST_REVERSE_ORDER": "1"}
    if order == "shuffle":
        # `-o addopts=` drops the project's default `-p no:randomly`; the options it
        # also drops are put back explicitly so the run is otherwise identical.
        return (
            [
                "-o",
                "addopts=",
                "-p",
                "randomly",
                f"--randomly-seed={seed}",
                "-ra",
                "--strict-markers",
                "--strict-config",
            ],
            {},
        )
    raise ValueError(f"unknown order: {order}")


def _parse_counts(text: str) -> dict[str, int]:
    tail = text.strip().splitlines()[-1] if text.strip() else ""
    counts: dict[str, int] = {}
    for number, label in _COUNTS.findall(tail):
        counts[label.rstrip("s") if label != "passed" else label] = int(number)
    return counts


def _parse_failures(text: str) -> list[str]:
    found: list[str] = []
    for line in text.splitlines():
        match = _OUTCOME.match(line.strip())
        if match:
            nodeid = match.group(2)
            if nodeid not in found:
                found.append(nodeid)
    return found


def run_one(
    label: str,
    order: str,
    seed: int | None,
    outdir: Path,
    selector: str,
    hermetic: bool,
    target: str,
) -> dict[str, object]:
    arguments, extra_env = _order_arguments(order, seed)
    log = outdir / f"{label}.log"
    timing = outdir / f"{label}-lanes.json"

    env = dict(os.environ)
    env.update(extra_env)
    env["EFFGEN_LANE_TIMING"] = str(timing)
    env["EFFGEN_LANE_TIMING_TABLE"] = "1"
    if hermetic:
        env["EFFGEN_TEST_HERMETIC"] = "1"
    else:
        env.pop("EFFGEN_TEST_HERMETIC", None)

    command = [
        sys.executable,
        "-m",
        "pytest",
        target,
        "-m",
        selector,
        "--tb=line",
        "--no-header",
        "-q",
        "-rf",
        "-p",
        "no:cacheprovider",
        *arguments,
    ]

    started = time.time()
    with log.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(command) + "\n\n")
        handle.flush()
        completed = subprocess.run(
            command, cwd=str(REPO_ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT
        )
    elapsed = time.time() - started

    text = log.read_text(encoding="utf-8", errors="replace")
    failures = _parse_failures(text)
    entries = flake_register.load()
    registered = [f for f in failures if flake_register.entry_for(f, entries) is not None]
    unregistered = [f for f in failures if f not in registered]

    return {
        "label": label,
        "order": order,
        "seed": seed,
        "hermetic": hermetic,
        "returncode": completed.returncode,
        "wall_s": round(elapsed, 1),
        "counts": _parse_counts(text),
        "failures": failures,
        "known_flakes": registered,
        "unregistered_failures": unregistered,
        "log": str(log),
        "lane_timing": str(timing) if timing.exists() else None,
    }


def _summarise(runs: list[dict[str, object]]) -> str:
    lines = ["# Order matrix", "", "| run | order | seed | wall (s) | passed | failed | skipped | new failures |", "|---|---|---|---|---|---|---|---|"]
    for run in runs:
        counts = run["counts"]  # type: ignore[index]
        lines.append(
            "| {label} | {order} | {seed} | {wall} | {passed} | {failed} | {skipped} | {new} |".format(
                label=run["label"],
                order=run["order"],
                seed=run["seed"] if run["seed"] is not None else "-",
                wall=run["wall_s"],
                passed=counts.get("passed", 0),
                failed=counts.get("failed", 0) + counts.get("error", 0),
                skipped=counts.get("skipped", 0),
                new=len(run["unregistered_failures"]),  # type: ignore[arg-type]
            )
        )
    lines.append("")

    everywhere: dict[str, int] = {}
    for run in runs:
        for nodeid in run["unregistered_failures"]:  # type: ignore[union-attr]
            everywhere[nodeid] = everywhere.get(nodeid, 0) + 1
    if everywhere:
        lines += ["## Failures with no entry in the register", ""]
        lines += ["| test | lane | runs it failed in |", "|---|---|---|"]
        for nodeid, count in sorted(everywhere.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"| `{nodeid}` | {lane_of_nodeid(nodeid)} | {count}/{len(runs)} |")
        lines.append("")
    else:
        lines += ["No failure outside the register in any order.", ""]

    known: dict[str, int] = {}
    for run in runs:
        for nodeid in run["known_flakes"]:  # type: ignore[union-attr]
            known[nodeid] = known.get(nodeid, 0) + 1
    if known:
        lines += ["## Failures with a registered cause", ""]
        lines += ["| test | cause | runs it failed in |", "|---|---|---|"]
        entries = flake_register.load()
        for nodeid, count in sorted(known.items()):
            entry = flake_register.entry_for(nodeid, entries)
            lines.append(f"| `{nodeid}` | {entry.cause if entry else '?'} | {count}/{len(runs)} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", required=True, type=Path, help="where logs and reports go")
    parser.add_argument(
        "--orders",
        nargs="+",
        default=["fixed", "reversed", "shuffle"],
        choices=["fixed", "reversed", "shuffle"],
        help="which orders to run",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS), help="shuffle seeds"
    )
    parser.add_argument("--selector", default=DEFAULT_SELECTOR, help="-m expression")
    parser.add_argument("--target", default="tests", help="what to collect")
    parser.add_argument(
        "--hermetic",
        action="store_true",
        help="run with the ambient environment removed (EFFGEN_TEST_HERMETIC=1)",
    )
    args = parser.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    plan: list[tuple[str, str, int | None]] = []
    for order in args.orders:
        if order == "shuffle":
            for seed in args.seeds:
                plan.append((f"shuffle-{seed}", "shuffle", seed))
        else:
            plan.append((order, order, None))

    runs: list[dict[str, object]] = []
    for label, order, seed in plan:
        print(f"[order-matrix] {label} ...", flush=True)
        run = run_one(
            label, order, seed, args.outdir, args.selector, args.hermetic, args.target
        )
        counts = run["counts"]  # type: ignore[index]
        print(
            f"[order-matrix] {label}: rc={run['returncode']} "
            f"{counts.get('passed', 0)} passed, {len(run['failures'])} failed, "  # type: ignore[arg-type]
            f"{len(run['unregistered_failures'])} outside the register, "  # type: ignore[arg-type]
            f"{run['wall_s']}s",
            flush=True,
        )
        runs.append(run)

    (args.outdir / "order-matrix.json").write_text(
        json.dumps(runs, indent=2) + "\n", encoding="utf-8"
    )
    summary = _summarise(runs)
    (args.outdir / "order-matrix.md").write_text(summary + "\n", encoding="utf-8")
    print(summary)

    return 1 if any(run["unregistered_failures"] for run in runs) else 0


if __name__ == "__main__":
    raise SystemExit(main())

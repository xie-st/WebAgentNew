#!/usr/bin/env python
"""Fetch the small local models effGen's quick starts use.

``./install.sh --full`` and ``./install.sh --download-models`` run this so the
first local run does not stall on a multi-gigabyte download. Everything here is
optional: effGen fetches a model the first time it is asked for one, and a
cloud provider needs no download at all.

    python scripts/download_models.py            # the recommended set
    python scripts/download_models.py --list     # what that is, and how big
    python scripts/download_models.py --models Qwen/Qwen2.5-1.5B-Instruct

Weights land wherever ``HF_HUB_CACHE`` points, which is the same cache
``transformers`` and ``vLLM`` read. This script never overrides it: a download
directory chosen here would put a second copy on disk and hide the first.
"""

from __future__ import annotations

import argparse
import sys

#: What a fresh install gets, smallest first. Both are instruction-tuned and
#: run on a modest GPU or on CPU, which is what the quick starts assume.
RECOMMENDED = [
    ("Qwen/Qwen2.5-1.5B-Instruct", 3.1, "the quick-start model; runs on CPU"),
    ("Qwen/Qwen2.5-3B-Instruct", 6.2, "the configured default for local runs"),
]


def _print_set(models: list[tuple[str, float, str]]) -> None:
    total = sum(size for _, size, _ in models)
    print(f"{len(models)} model(s), about {total:.1f} GB in total:\n")
    for repo, size, why in models:
        print(f"  {repo:<38} {size:>5.1f} GB   {why}")
    print()


def _cache_location() -> str:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return HF_HUB_CACHE
    except Exception:
        return "the default HuggingFace cache"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch the small local models effGen's quick starts use.",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="REPO_ID",
        help="Fetch these repo ids instead of the recommended set.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Show what would be fetched, and how big, without fetching it.",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help=(
            "Ask before fetching. With no terminal to ask on, the recommended "
            "set is fetched without a prompt, so an unattended install does "
            "not stop here."
        ),
    )
    args = parser.parse_args()

    chosen = (
        [(repo, 0.0, "requested on the command line") for repo in args.models]
        if args.models else list(RECOMMENDED)
    )

    print(f"Cache: {_cache_location()}")
    _print_set(chosen)

    if args.list:
        return 0

    if args.interactive and sys.stdin.isatty():
        answer = input("Fetch these now? [Y/n]: ").strip().lower()
        if answer not in ("", "y", "yes"):
            print("Skipped. effGen will fetch a model the first time it needs one.")
            return 0
    elif args.interactive:
        print("No terminal to ask on — fetching the recommended set.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is not installed, so nothing was fetched.\n"
            "Install it with: pip install huggingface_hub\n"
            "This is not fatal: effGen fetches a model the first time it needs "
            "one, and a cloud provider needs no download at all.",
            file=sys.stderr,
        )
        return 0

    failed: list[str] = []
    for repo, _, _ in chosen:
        print(f"Fetching {repo} ...")
        try:
            # No cache_dir: HF_HUB_CACHE decides, so this shares the cache
            # transformers and vLLM already read from.
            snapshot_download(repo_id=repo)
            print(f"  done: {repo}")
        except Exception as exc:
            failed.append(f"{repo}: {exc}")
            print(f"  could not fetch {repo}: {exc}", file=sys.stderr)

    if failed:
        print(
            f"\n{len(failed)} of {len(chosen)} could not be fetched. effGen will "
            "fetch a model the first time it needs one, so this does not stop "
            "you using it:",
            file=sys.stderr,
        )
        for line in failed:
            print(f"  {line}", file=sys.stderr)
        # A download that did not happen is not a failed install — the models
        # are a convenience, and the installer must not report failure for one.
        return 0

    print(f"\nFetched {len(chosen)} model(s) into {_cache_location()}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

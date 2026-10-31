"""Inventory current working files; copy only when --destination is supplied."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3]
ROOTS = (
    "README.md", ".gitignore", "code/harness", "code/vendor/effgen",
    "code/official_client", "docs/01_路线与框架", "docs/03_策略与论文方法",
    "docs/05_赛题与要求", "docs/06_分阶段实施",
    "docs/07_汇报材料/演示程序", "docs/07_汇报材料/最终验收准备与现场执行.md",
    "docs/08_提交材料/迁移工具", "docs/08_提交材料/迁移准备.md",
)
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache",
             ".ruff_cache", "node_modules", ".next", ".pptx-build", "browser-profile"}


def excluded(path: Path) -> bool:
    parts = path.parts
    name = path.name.lower()
    return (
        any(p in SKIP_DIRS or p.endswith(".egg-info") for p in parts)
        or name.endswith((".pyc", ".pyo", ".local.yaml", ".local.yml", ".local.json"))
        or (name != "example.env" and (name.endswith(".env") or name.startswith(".env")))
        or name.startswith("agent-connection")
        or name == "platform-account.local.json"
    )


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def candidates(base: Path, source: Path):
    if base.is_file():
        yield base
        return
    for current, dirs, names in os.walk(base, followlinks=False):
        current = Path(current)
        for name in dirs:
            child = current / name
            if not excluded(child.relative_to(source)) and child.is_symlink():
                raise SystemExit(f"Review symbolic link before copying: {child.relative_to(source)}")
        dirs[:] = sorted(name for name in dirs if not excluded((current / name).relative_to(source)))
        for name in sorted(names):
            yield current / name


def inventory(source: Path) -> list[dict]:
    files = []
    for root in ROOTS:
        base = source / root
        if not base.exists():
            raise SystemExit(f"Missing required source: {root}")
        for path in candidates(base, source):
            relative = path.relative_to(source)
            if excluded(relative):
                continue
            if path.is_symlink():
                raise SystemExit(f"Review symbolic link before copying: {relative}")
            if path.is_file():
                files.append({"path": relative.as_posix(), "bytes": path.stat().st_size,
                              "sha256": digest(path)})
    return sorted(files, key=lambda item: item["path"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--destination", type=Path, help="New or empty package directory")
    parser.add_argument("--output", type=Path, help="Save preparation inventory JSON")
    args = parser.parse_args()
    source = args.source.resolve()
    files = inventory(source)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "source": str(source),
                "files": files, "file_count": len(files),
                "total_bytes": sum(item["bytes"] for item in files),
                "status": "inventory_only"}
    if args.destination:
        target = args.destination.resolve()
        # A new destination keeps later patches from leaving deleted source files behind.
        if target == source or source.is_relative_to(target):
            raise SystemExit("Destination must not contain the source tree")
        for root in ROOTS:
            origin = source / root
            if target == origin or target.is_relative_to(origin):
                raise SystemExit(f"Destination overlaps copied source: {root}")
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise SystemExit("Destination must be a new or empty directory; no files overwritten")
        target.mkdir(parents=True, exist_ok=True)
        for item in files:
            origin, copied = source / item["path"], target / item["path"]
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, copied)
            if digest(copied) != item["sha256"]:
                raise SystemExit(f"Source changed during copy; repeat with fresh destination: {item['path']}")
        # Catch patches and additions/deletions that arrived during the copy.
        if inventory(source) != files:
            raise SystemExit("Source changed during copy; repeat with fresh destination")
        manifest["status"] = "copied_and_hash_verified"
        manifest["destination"] = str(target)
        (target / "delivery-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

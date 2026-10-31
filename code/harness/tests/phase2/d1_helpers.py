from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path


def fetch(url: str, data: bytes | None = None, *, timeout: float = 10.0) -> str:
    request = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def fetch_status(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=10.0) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def screenshot(page, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(path), type="png")
    return path

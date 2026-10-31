"""
Gemini Files API helpers for effGen.

Wraps google.genai file upload so callers work with a simple FileRef
object rather than raw SDK types.

Usage::

    from effgen.models.gemini_files import upload_file
    from effgen.models.gemini_adapter import GeminiAdapter
    from effgen.models.base import GenerationConfig

    ref = upload_file("paper.pdf")
    adapter = GeminiAdapter("gemini-3.1-flash-lite")
    adapter.load()
    result = adapter.generate("Summarise this document.", files=[ref])

Limits:
- 2 GB per file (Google limit).
- Supported MIME types: text/plain, application/pdf, image/*, audio/*, video/*
  and others listed in Google's Files API docs.
- Uploaded files are stored on Google's servers for 48 hours.
"""

from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB — Gemini Files API hard limit.


@dataclass
class FileRef:
    """
    Lightweight reference to a file uploaded via the Gemini Files API.

    Attributes:
        uri: The ``gs://`` or ``https://`` URI returned by the Files API.
        mime_type: MIME type inferred or provided at upload time.
        display_name: Human-readable label (defaults to the file name).
        raw: The raw SDK file object, available for low-level access.
    """
    uri: str
    mime_type: str
    display_name: str = ""
    raw: Any = field(default=None, repr=False)


def upload_file(
    path: str | Path,
    *,
    api_key: str | None = None,
    mime_type: str | None = None,
    display_name: str | None = None,
    wait_for_active: bool | None = None,
    wait_timeout_s: float = 60.0,
    poll_interval_s: float = 2.0,
) -> FileRef:
    """Upload *path* to the Gemini Files API and return a :class:`FileRef`.

    Args:
        path: Local path to the file to upload. Must be < 2 GB.
        api_key: Google API key. Falls back to ``GOOGLE_API_KEY`` env var.
        mime_type: Override auto-detected MIME type.
        display_name: Human-readable label stored with the file.
        wait_for_active: When ``True`` poll the Files API until the uploaded
            file reaches the ``ACTIVE`` state.  When ``None`` (default), waits
            automatically for video / audio uploads since these are processed
            asynchronously.
        wait_timeout_s: Max seconds to wait for ACTIVE state (default 60).
        poll_interval_s: Seconds between status polls (default 2).

    Returns:
        :class:`FileRef` pointing to the uploaded file.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the Google API key is missing.
        RuntimeError: If the upload fails or the file never becomes ACTIVE.
    """
    try:
        import google.genai as genai  # type: ignore[import]
        from google.genai import types as genai_types  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "google-genai package is not installed. "
            "Install it with: pip install google-genai"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    size = path.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        gib = size / (1024 ** 3)
        raise ValueError(
            f"File '{path}' is {gib:.2f} GiB, which exceeds the Gemini Files API "
            f"limit of 2 GiB. Split the file or use a different ingestion path."
        )

    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "Google API key not provided. Set GOOGLE_API_KEY environment "
            "variable or pass api_key=... to upload_file()."
        )

    detected_mime, _ = mimetypes.guess_type(str(path))
    resolved_mime = mime_type or detected_mime or "application/octet-stream"
    resolved_name = display_name or path.name

    client = genai.Client(api_key=key)
    try:
        sdk_file = client.files.upload(
            file=path,
            config=genai_types.UploadFileConfig(
                mime_type=resolved_mime,
                display_name=resolved_name,
            ),
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini Files API upload failed: {exc}") from exc

    uri = getattr(sdk_file, "uri", None) or getattr(sdk_file, "name", str(sdk_file))

    # Auto-wait for video/audio uploads (processed asynchronously by Gemini).
    should_wait = wait_for_active
    if should_wait is None:
        should_wait = resolved_mime.startswith(("video/", "audio/"))

    if should_wait:
        import time as _time
        file_name = getattr(sdk_file, "name", None)
        if file_name:
            deadline = _time.time() + wait_timeout_s
            while _time.time() < deadline:
                try:
                    status = client.files.get(name=file_name)
                except Exception:
                    break
                state = str(getattr(status, "state", "")).upper()
                if "ACTIVE" in state:
                    sdk_file = status
                    break
                if "FAILED" in state:
                    raise RuntimeError(
                        f"Gemini Files API processing failed for '{file_name}': state={state}"
                    )
                _time.sleep(poll_interval_s)
            else:
                raise RuntimeError(
                    f"Gemini Files API upload timed out waiting for ACTIVE state "
                    f"after {wait_timeout_s}s (file={file_name})."
                )

    return FileRef(
        uri=uri,
        mime_type=resolved_mime,
        display_name=resolved_name,
        raw=sdk_file,
    )

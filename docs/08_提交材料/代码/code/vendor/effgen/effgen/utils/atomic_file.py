"""Replace a file's contents in one step, safely for concurrent writers.

Writing state to ``<path>.tmp`` and renaming it over ``<path>`` survives a
crash mid-write, but the fixed temporary name is shared: two writers of the
same file open it at the same moment, their writes interleave, and the rename
publishes the mix — or the second rename finds nothing there because the first
already moved it away. Each writer here gets its own temporary file in the
destination directory, so the rename publishes exactly what that writer wrote
and the last one to finish wins whole.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["atomic_write_text"]

#: Permission bits requested when creating the temporary. The kernel subtracts
#: the process umask, so a file created here ends up with exactly the mode an
#: ordinary ``open(path, "w")`` would have given it. Reading the umask to work
#: this out by hand would mean setting it process-wide for an instant, which a
#: concurrent writer in another thread would see.
_CREATE_MODE = 0o666


def _open_new(directory: Path, name: str) -> tuple[int, Path]:
    """Create a uniquely named file in *directory* and return its handle."""
    while True:
        temporary = directory / f".{name}.{os.urandom(6).hex()}.tmp"
        try:
            handle = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _CREATE_MODE
            )
        except FileExistsError:  # pragma: no cover - 48 random bits collided
            continue
        return handle, temporary


def atomic_write_text(path: str | os.PathLike, text: str, encoding: str = "utf-8") -> str:
    """Write *text* to *path*, replacing it in one step.

    The replacement keeps the permissions the destination already had, and a
    file being created for the first time gets the mode an ordinary write would
    have given it.

    Args:
        path: Destination file. Its parent directory must already exist.
        text: The complete new contents.
        encoding: Text encoding for the write.

    Returns:
        The destination path as a string.
    """
    destination = Path(path)
    directory = destination.parent
    handle, temporary = _open_new(directory, destination.name)
    try:
        try:
            stream = os.fdopen(handle, "w", encoding=encoding)
        except BaseException:
            os.close(handle)
            raise
        with stream:
            try:
                os.fchmod(handle, stat.S_IMODE(os.stat(destination).st_mode))
            except FileNotFoundError:
                # No destination yet, so the umask-derived mode above is the one
                # a plain write would have produced.
                pass
            stream.write(text)
        os.replace(temporary, destination)
    except BaseException:
        # A failed write leaves the previous contents in place rather than a
        # partial file, and never leaves its temporary behind.
        try:
            os.unlink(temporary)
        except OSError:
            # Best-effort cleanup only. The write failure below is what the
            # caller needs to see; a temporary that cannot be removed must not
            # replace it with a less useful error.
            logger.debug("Could not remove %s", temporary, exc_info=True)
        raise
    return str(destination)

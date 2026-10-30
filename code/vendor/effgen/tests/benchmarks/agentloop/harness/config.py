"""Where the copied modules look for their data files.

``tools/builtin.py`` and ``benchmarks/memory.py`` are verbatim copies that read
``DATA_DIR``. In the harness they came from it is a directory in that
repository; here it is wherever the caller says, and the modules that use it
report a stated reason when the path is absent rather than answering from an
empty index.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Root of the corpora the copied tools read: the local knowledge database the
#: retrieval sets search, and the two conversation files the memory sets load.
#: Nothing here is vendored — these are hundreds of megabytes — so the path is
#: given by ``AGENTLOOP_DATA_DIR`` and defaults to a directory that need not
#: exist. A caller that needs one of those corpora checks ``data_dir_status``
#: first and says what it could not measure.
DATA_DIR = Path(os.environ.get("AGENTLOOP_DATA_DIR", Path.cwd() / "agentloop_data"))


def data_dir_status() -> tuple[bool, str]:
    """Whether ``DATA_DIR`` holds a knowledge database, and what to say if not."""
    db = DATA_DIR / "knowledge_db"
    if db.is_dir():
        return True, f"knowledge database at {db}"
    return False, (
        f"no knowledge database under {DATA_DIR} "
        "(set AGENTLOOP_DATA_DIR to the directory holding knowledge_db/)"
    )


__all__ = ["DATA_DIR", "data_dir_status"]

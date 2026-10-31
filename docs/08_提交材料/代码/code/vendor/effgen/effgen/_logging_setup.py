"""Library logging hygiene, applied on package import.

A library must never emit log records or configure the root logger on import —
that is the application's job. Importing this module (which ``effgen/__init__``
does first, before any other submodule) attaches a :class:`logging.NullHandler`
to the package logger so ``import effgen`` is silent unless the host app opts in
(``logging.basicConfig`` / ``dictConfig``), and pins noisy third-party loggers
to ``WARNING``.

Kept as a separate, import-only module so the side-effecting setup runs via an
ordinary import statement at the top of ``effgen/__init__`` without pushing the
package's own imports past it.
"""

from __future__ import annotations

import logging

# NullHandler on the package logger: no output unless the app configures logging.
logging.getLogger("effgen").addHandler(logging.NullHandler())

# faiss' module loader logs "Loading faiss with AVX2 support…" at INFO the first
# time it is imported (pulled in transitively by the vector store). That is
# third-party INFO noise the user did not opt into. Pin the faiss loggers to
# WARNING *before* faiss is imported so a normal `import effgen` stays quiet
# while genuine faiss warnings/errors still surface.
for _faiss_logger in ("faiss", "faiss.loader"):
    logging.getLogger(_faiss_logger).setLevel(logging.WARNING)

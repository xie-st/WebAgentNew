"""effGen Jupyter integration.

Provides IPython magics for interactive use of the effGen framework inside
Jupyter notebooks and the IPython REPL.

Usage
-----
Load the extension in a notebook cell::

    %load_ext effgen.jupyter

Then use the magics::

    %effgen_chat Tell me about large language models
    %%effgen_agent coding
    Write a Python function that computes Fibonacci numbers iteratively.
    %effgen_metrics
"""

from __future__ import annotations


def load_ipython_extension(ipython: object) -> None:  # noqa: ANN001
    """Register effGen magics with the running IPython kernel.

    Called automatically by ``%load_ext effgen.jupyter``.
    """
    from effgen.jupyter.magics import EffgenMagics

    ipython.register_magics(EffgenMagics)  # type: ignore[attr-defined]


__all__ = ["load_ipython_extension"]

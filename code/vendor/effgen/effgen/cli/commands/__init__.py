"""Command implementations for the ``effgen`` CLI.

Each module holds the body of one command group; :mod:`effgen.cli._main`
owns argument parsing and dispatch and re-exports these names, so
``effgen.cli._main`` stays the stable import path. Handlers are imported
on demand rather than here, so importing this package pulls in nothing.
"""

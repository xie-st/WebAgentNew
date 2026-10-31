"""Per-command argument declarations for the ``effgen`` CLI.

Each module holds the ``add_argument`` calls for one family of commands;
``effgen.cli._main.create_parser`` builds the top-level parser, declares the
global flags, and calls these builders in the order the commands appear in
``effgen --help``. Builders are imported on demand rather than here, so
importing this package pulls in nothing.
"""

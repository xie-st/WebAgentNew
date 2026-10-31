"""The ``effgen create-plugin`` command: generate a plugin project scaffold.

:mod:`effgen.cli._main` parses arguments and dispatches; it imports this at
module scope and re-exports these names, so ``effgen.cli._main._create_plugin_scaffold``
keeps resolving. Writes a package, its tool and plugin modules, its
``pyproject.toml`` entry point and a README from the templates shipped as
package data under ``effgen/cli/_templates/plugin/``.
"""

from __future__ import annotations

from pathlib import Path


def _render_plugin_template(filename: str, replacements: dict[str, str]) -> str:
    """Load a scaffold template from package data and substitute placeholders.

    Templates live in ``effgen/cli/_templates/plugin/`` (shipped as package data)
    rather than being embedded here, so the generated plugin stays in sync with
    BaseTool and is covered by a create→install→import→run test.
    """
    from importlib import resources

    # Anchor on the real ``effgen.cli`` package and traverse into the data dir,
    # which works for both editable checkouts and installed wheels.
    text = (
        resources.files("effgen.cli")
        .joinpath("_templates", "plugin", filename)
        .read_text(encoding="utf-8")
    )
    for token, value in replacements.items():
        text = text.replace(token, value)
    return text


def _create_plugin_scaffold(plugin_name: str, output_dir: str = ".") -> int:
    """Generate a plugin project scaffold."""
    # Normalize to a valid Python package name (entry points + imports need it).
    pkg_name = plugin_name.replace("-", "_")
    if not pkg_name.isidentifier():
        print(
            f"Error: '{plugin_name}' is not a valid plugin name. Use letters, "
            "digits and underscores (must start with a letter)."
        )
        return 1

    plugin_class = pkg_name.title().replace("_", "")
    replacements = {
        "__PLUGIN_NAME__": pkg_name,
        "__PLUGIN_CLASS__": plugin_class,
    }

    base = Path(output_dir) / f"effgen-plugin-{pkg_name}"
    pkg = base / pkg_name
    try:
        pkg.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Error: Directory {base} already exists.")
        return 1

    try:
        (pkg / "__init__.py").write_text(_render_plugin_template("init.py.tmpl", replacements))
        (pkg / "tools.py").write_text(_render_plugin_template("tools.py.tmpl", replacements))
        (pkg / "plugin.py").write_text(_render_plugin_template("plugin.py.tmpl", replacements))
        (base / "pyproject.toml").write_text(_render_plugin_template("pyproject.toml.tmpl", replacements))
        (base / "README.md").write_text(_render_plugin_template("README.md.tmpl", replacements))
    except Exception as e:
        print(f"Error: failed to write scaffold files: {e}")
        return 1

    print(f"Created plugin scaffold at {base}/")
    print(f"  {pkg / 'tools.py'}       — add your custom tools here")
    print(f"  {pkg / 'plugin.py'}     — register tools in the plugin class")
    print(f"  {base / 'pyproject.toml'} — package metadata & entry point")
    print("\nNext: cd into it and `pip install -e .` to register the plugin.")
    return 0

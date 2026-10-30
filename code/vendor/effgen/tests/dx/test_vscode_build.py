"""Tests for the VSCode extension build.

Validates that:
1. package.json is well-formed and has required fields.
2. tsconfig.json is valid JSON.
3. ``npm ci && npm run compile`` produces the output JS file (integration).
4. The compiled extension entry-point exists on disk.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

VSCODE_DIR = Path(__file__).parents[2] / "tools" / "vscode-effgen"


# ---------------------------------------------------------------------------
# Static / schema checks (no npm required)
# ---------------------------------------------------------------------------


class TestPackageJson:
    def test_package_json_exists(self):
        assert (VSCODE_DIR / "package.json").exists()

    def test_package_json_valid_json(self):
        pkg = json.loads((VSCODE_DIR / "package.json").read_text())
        assert isinstance(pkg, dict)

    def test_required_fields(self):
        pkg = json.loads((VSCODE_DIR / "package.json").read_text())
        for field in ("name", "version", "displayName", "main", "engines", "contributes"):
            assert field in pkg, f"Missing field: {field}"

    def test_engines_vscode(self):
        pkg = json.loads((VSCODE_DIR / "package.json").read_text())
        assert "vscode" in pkg["engines"]

    def test_compile_script_present(self):
        pkg = json.loads((VSCODE_DIR / "package.json").read_text())
        assert "compile" in pkg.get("scripts", {})

    def test_main_points_to_out(self):
        pkg = json.loads((VSCODE_DIR / "package.json").read_text())
        assert pkg["main"].startswith("./out/")

    def test_commands_defined(self):
        pkg = json.loads((VSCODE_DIR / "package.json").read_text())
        commands = pkg.get("contributes", {}).get("commands", [])
        names = [c["command"] for c in commands]
        assert "effgen.runPrompt" in names
        assert "effgen.showRegistry" in names

    def test_configuration_properties(self):
        pkg = json.loads((VSCODE_DIR / "package.json").read_text())
        props = (
            pkg.get("contributes", {})
            .get("configuration", {})
            .get("properties", {})
        )
        assert "effgen.serverUrl" in props
        assert "effgen.defaultModel" in props

    def test_version_matches_effgen(self):
        import effgen

        pkg = json.loads((VSCODE_DIR / "package.json").read_text())
        # The extension version tracks the effGen package release version.
        assert pkg["version"] == effgen.__version__


class TestTsConfig:
    def test_tsconfig_exists(self):
        assert (VSCODE_DIR / "tsconfig.json").exists()

    def test_tsconfig_valid_json(self):
        cfg = json.loads((VSCODE_DIR / "tsconfig.json").read_text())
        assert isinstance(cfg, dict)

    def test_outdir_is_out(self):
        cfg = json.loads((VSCODE_DIR / "tsconfig.json").read_text())
        assert cfg["compilerOptions"]["outDir"] == "./out"

    def test_module_commonjs(self):
        cfg = json.loads((VSCODE_DIR / "tsconfig.json").read_text())
        assert cfg["compilerOptions"]["module"] == "commonjs"


class TestSourceFile:
    def test_extension_ts_exists(self):
        assert (VSCODE_DIR / "src" / "extension.ts").exists()

    def test_activate_function_present(self):
        src = (VSCODE_DIR / "src" / "extension.ts").read_text()
        assert "export async function activate" in src

    def test_deactivate_function_present(self):
        src = (VSCODE_DIR / "src" / "extension.ts").read_text()
        assert "export function deactivate" in src

    def test_completion_provider_present(self):
        src = (VSCODE_DIR / "src" / "extension.ts").read_text()
        assert "EffgenCompletionProvider" in src

    def test_hover_provider_present(self):
        src = (VSCODE_DIR / "src" / "extension.ts").read_text()
        assert "EffgenHoverProvider" in src

    def test_codelens_provider_present(self):
        src = (VSCODE_DIR / "src" / "extension.ts").read_text()
        assert "EffgenCodeLensProvider" in src

    def test_builtin_templates_present(self):
        src = (VSCODE_DIR / "src" / "extension.ts").read_text()
        # Verify some known template names
        for name in ("general", "coding", "reasoning", "analysis"):
            assert f'name: "{name}"' in src, f"Template '{name}' missing from extension.ts"


# ---------------------------------------------------------------------------
# Compiled output checks
# ---------------------------------------------------------------------------


_COMPILED_JS = VSCODE_DIR / "out" / "extension.js"


@pytest.mark.skipif(
    not _COMPILED_JS.exists(),
    reason=(
        "compiled out/extension.js absent (gitignored build artifact) — run "
        "`npm run compile` in tools/vscode-effgen; the compile itself is "
        "exercised by TestNpmCompile when npm is available"
    ),
)
class TestCompiledOutput:
    def test_compiled_js_exists(self):
        """The compiled out/extension.js must exist after npm run compile."""
        assert _COMPILED_JS.exists(), (
            "out/extension.js not found — run `npm run compile` in tools/vscode-effgen"
        )

    def test_compiled_js_exports_activate(self):
        js = _COMPILED_JS.read_text()
        # CommonJS exports
        assert "exports.activate" in js or "activate" in js


# ---------------------------------------------------------------------------
# Integration: npm compile (skipped if npm not available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("npm") is None, reason="npm not installed")
class TestNpmCompile:
    def test_npm_compile_succeeds(self):
        """Run npm ci + npm run compile inside the extension directory."""
        # npm ci installs pinned deps; then compile runs tsc
        result = subprocess.run(
            ["npm", "ci"],
            cwd=VSCODE_DIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"npm ci failed:\n{result.stderr}"

        result = subprocess.run(
            ["npm", "run", "compile"],
            cwd=VSCODE_DIR,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"npm run compile failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        # Compiled output must exist
        assert (VSCODE_DIR / "out" / "extension.js").exists()

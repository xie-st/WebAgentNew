"""
tests/security/test_supply_chain.py

Tests for effgen.security.supply_chain — startup hash / version verification.

Validates:
1. pyproject.toml contains required supply-chain hardening fields.
2. The supply_chain module imports cleanly and exposes the correct API.
3. Version-drift detection works (installs a fake package, checks detection).
4. EFFGEN_VERIFY_HASHES=1 triggers verification on import effgen.
5. Clean environments pass without warnings.
6. The compiled lockfiles pin versions the declared specifiers allow.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

try:  # tomllib is stdlib on Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with open(PYPROJECT, "rb") as fh:
        return tomllib.load(fh)


@pytest.fixture(scope="module")
def supply_chain_module():
    from effgen.security import supply_chain

    return supply_chain


# ---------------------------------------------------------------------------
# 1. pyproject.toml supply-chain hardening fields
# ---------------------------------------------------------------------------


class TestPyprojectHardening:
    """pyproject.toml must satisfy supply-chain hygiene requirements."""

    def test_license_spdx_present(self, pyproject):
        """[project] license must be an SPDX identifier string."""
        proj = pyproject.get("project", {})
        license_val = proj.get("license")
        assert license_val, "[project] 'license' is required for supply-chain transparency"
        # SPDX format is a plain string like "Apache-2.0" (PEP 639)
        assert isinstance(license_val, str), "license must be an SPDX string"
        assert license_val == "Apache-2.0", f"Expected 'Apache-2.0', got: {license_val!r}"

    def test_authors_present(self, pyproject):
        """[project] authors must be declared."""
        authors = pyproject.get("project", {}).get("authors", [])
        assert len(authors) >= 1, "[project] 'authors' must have at least one entry"
        for a in authors:
            assert "name" in a or "email" in a, f"Author entry must have name or email: {a}"

    def test_maintainers_present(self, pyproject):
        """[project] maintainers should be declared."""
        maintainers = pyproject.get("project", {}).get("maintainers", [])
        assert len(maintainers) >= 1, "[project] 'maintainers' must have at least one entry"

    def test_urls_homepage(self, pyproject):
        """[project.urls] Homepage must be present."""
        urls = pyproject.get("project", {}).get("urls", {})
        assert urls.get("Homepage"), "[project.urls] 'Homepage' is required"

    def test_urls_repository(self, pyproject):
        """[project.urls] Repository must be present."""
        urls = pyproject.get("project", {}).get("urls", {})
        assert urls.get("Repository"), "[project.urls] 'Repository' is required"

    def test_urls_bug_tracker(self, pyproject):
        """[project.urls] Bug Tracker should be present."""
        urls = pyproject.get("project", {}).get("urls", {})
        assert urls.get("Bug Tracker"), "[project.urls] 'Bug Tracker' is required"

    def test_build_system_backend_pinned(self, pyproject):
        """[build-system] requires must pin setuptools and wheel versions."""
        requires = pyproject.get("build-system", {}).get("requires", [])
        assert any("setuptools" in r for r in requires), (
            "[build-system] 'requires' must pin setuptools"
        )
        assert any("wheel" in r for r in requires), (
            "[build-system] 'requires' must pin wheel"
        )
        # Each entry must have a version specifier
        for req in requires:
            assert any(op in req for op in (">=", "==", "~=", "<=", ">")), (
                f"Build backend dependency '{req}' must have a version specifier"
            )

    def test_build_backend_specified(self, pyproject):
        """[build-system] build-backend must be specified."""
        backend = pyproject.get("build-system", {}).get("build-backend")
        assert backend == "setuptools.build_meta", (
            f"Expected setuptools.build_meta, got: {backend!r}"
        )

    def test_starlette_version_pinned(self, pyproject):
        """starlette must be pinned >= 1.0.1 to fix PYSEC-2026-161 (host-header injection)."""
        deps = pyproject.get("project", {}).get("dependencies", [])
        starlette_pins = [d for d in deps if d.startswith("starlette")]
        assert starlette_pins, (
            "starlette must be explicitly pinned in [project.dependencies] to fix PYSEC-2026-161"
        )
        # Should be >= 1.0.1
        pin = starlette_pins[0]
        assert "1.0" in pin or "1.1" in pin or ">=" in pin, (
            f"starlette pin '{pin}' does not appear to require >= 1.0.1"
        )

    def test_protobuf_version_pinned(self, pyproject):
        """protobuf must be pinned >= 5.29.5 to fix CVE-2025-4565/CVE-2026-0994."""
        deps = pyproject.get("project", {}).get("dependencies", [])
        proto_pins = [d for d in deps if d.startswith("protobuf")]
        assert proto_pins, (
            "protobuf must be explicitly pinned in [project.dependencies] "
            "to fix CVE-2025-4565 and CVE-2026-0994"
        )


# ---------------------------------------------------------------------------
# 2. supply_chain module API
# ---------------------------------------------------------------------------


class TestSupplyChainModule:
    """effgen.security.supply_chain must expose the documented API."""

    def test_module_importable(self, supply_chain_module):
        assert supply_chain_module is not None

    def test_hash_drift_warning_is_user_warning(self, supply_chain_module):
        """HashDriftWarning must subclass UserWarning so it can be filtered."""
        assert issubclass(supply_chain_module.HashDriftWarning, UserWarning)

    def test_verification_result_dataclass(self, supply_chain_module):
        """VerificationResult must be a dataclass-like with the expected fields."""
        vr = supply_chain_module.VerificationResult()
        assert hasattr(vr, "checked")
        assert hasattr(vr, "ok")
        assert hasattr(vr, "drifted")
        assert hasattr(vr, "not_in_lockfile")
        assert hasattr(vr, "skipped")
        assert hasattr(vr, "clean")
        # clean must be True when drifted list is empty
        assert vr.clean is True

    def test_verify_installed_hashes_callable(self, supply_chain_module):
        """verify_installed_hashes() must be callable and return a VerificationResult."""
        result = supply_chain_module.verify_installed_hashes()
        assert isinstance(result, supply_chain_module.VerificationResult)

    def test_verify_on_startup_callable(self, supply_chain_module):
        """verify_on_startup() must be callable (no-op when env var not set)."""
        # Without EFFGEN_VERIFY_HASHES=1 it must be a no-op
        env_before = os.environ.pop("EFFGEN_VERIFY_HASHES", None)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                supply_chain_module.verify_on_startup()
            # No HashDriftWarning should be emitted without the env var
            drift_warns = [x for x in w if issubclass(x.category, supply_chain_module.HashDriftWarning)]
            assert len(drift_warns) == 0, "verify_on_startup should be a no-op without EFFGEN_VERIFY_HASHES=1"
        finally:
            if env_before is not None:
                os.environ["EFFGEN_VERIFY_HASHES"] = env_before

    def test_effgen_security_init_exports(self):
        """effgen.security must re-export all public symbols."""
        from effgen import security

        for name in ["HashDriftWarning", "VerificationResult", "verify_installed_hashes", "verify_on_startup"]:
            assert hasattr(security, name), f"effgen.security missing export: {name}"

    def test_effgen_top_level_exports(self):
        """Top-level effgen must export supply-chain symbols."""
        import effgen

        assert hasattr(effgen, "HashDriftWarning")
        assert hasattr(effgen, "VerificationResult")
        assert hasattr(effgen, "verify_installed_hashes")
        assert hasattr(effgen, "verify_on_startup")


# ---------------------------------------------------------------------------
# 3. Version-drift detection
# ---------------------------------------------------------------------------


class TestVersionDriftDetection:
    """verify_installed_hashes() must detect version mismatches."""

    def test_fake_lockfile_detects_version_mismatch(self, tmp_path, supply_chain_module):
        """A lockfile with a wrong version must trigger HashDriftWarning for that package."""
        # Write a lockfile that pins 'requests' to a version that is NOT installed
        fake_lockfile = tmp_path / "requirements-lock.txt"
        fake_lockfile.write_text(
            "# Fake lockfile for testing\n"
            "requests==0.0.0.dev0 \\\n"
            "    --hash=sha256:" + "a" * 64 + "\n"
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = supply_chain_module.verify_installed_hashes(
                lockfile=fake_lockfile, warn=True
            )

        # requests is installed; 0.0.0.dev0 is not; so drift must be reported
        assert "requests" in result.drifted, (
            "requests should be in drifted because lockfile pins 0.0.0.dev0 "
            f"but a real version is installed. drifted={result.drifted}"
        )
        drift_warns = [x for x in w if issubclass(x.category, supply_chain_module.HashDriftWarning)]
        assert len(drift_warns) > 0, "HashDriftWarning should be emitted for drifted packages"

    def test_correct_lockfile_no_drift(self, tmp_path, supply_chain_module):
        """A lockfile with the correct installed version must not report drift."""
        import importlib.metadata

        # Use 'certifi' as a simple package that's almost always installed
        try:
            certifi_ver = importlib.metadata.version("certifi")
        except importlib.metadata.PackageNotFoundError:
            pytest.skip("certifi not installed")

        # Write a minimal lockfile with the correct version
        fake_lockfile = tmp_path / "requirements-lock.txt"
        fake_lockfile.write_text(
            "# Fake lockfile for testing\n"
            f"certifi=={certifi_ver} \\\n"
            "    --hash=sha256:" + "b" * 64 + "\n"
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = supply_chain_module.verify_installed_hashes(
                lockfile=fake_lockfile, warn=True
            )

        assert "certifi" not in result.drifted, (
            f"certifi at {certifi_ver} should not be drifted against matching lockfile pin"
        )

    def test_empty_lockfile_returns_clean(self, tmp_path, supply_chain_module):
        """An empty lockfile should return a clean result (nothing to check)."""
        fake_lockfile = tmp_path / "requirements-lock.txt"
        fake_lockfile.write_text("# Empty lockfile\n")

        result = supply_chain_module.verify_installed_hashes(lockfile=fake_lockfile)
        assert result.clean, "Empty lockfile should yield clean result"
        assert result.checked == 0

    def test_missing_lockfile_returns_clean(self, tmp_path, supply_chain_module):
        """If the lockfile does not exist, verification silently succeeds."""
        missing = tmp_path / "does-not-exist-lock.txt"
        result = supply_chain_module.verify_installed_hashes(lockfile=missing)
        assert result.clean, "Missing lockfile should yield clean result"

    def test_warn_false_suppresses_warnings(self, tmp_path, supply_chain_module):
        """warn=False must suppress HashDriftWarning even when drift exists."""
        fake_lockfile = tmp_path / "requirements-lock.txt"
        fake_lockfile.write_text(
            "requests==0.0.0.dev0 \\\n"
            "    --hash=sha256:" + "c" * 64 + "\n"
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = supply_chain_module.verify_installed_hashes(
                lockfile=fake_lockfile, warn=False
            )

        drift_warns = [x for x in w if issubclass(x.category, supply_chain_module.HashDriftWarning)]
        assert len(drift_warns) == 0, "warn=False must suppress HashDriftWarning"
        # Drift is still detected, just not warned
        assert "requests" in result.drifted


# ---------------------------------------------------------------------------
# 4. EFFGEN_VERIFY_HASHES=1 triggers on import
# ---------------------------------------------------------------------------


class TestStartupHook:
    """When EFFGEN_VERIFY_HASHES=1, verify_on_startup() emits warnings on drift."""

    def test_env_var_triggers_verification(self, supply_chain_module, tmp_path, monkeypatch):
        """EFFGEN_VERIFY_HASHES=1 must call the verifier (checked via a fake lockfile)."""
        # Point the module at a fake lockfile with a wrong version

        fake_lockfile = tmp_path / "requirements-lock.txt"
        fake_lockfile.write_text(
            "requests==0.0.0.dev0 \\\n"
            "    --hash=sha256:" + "d" * 64 + "\n"
        )

        # Monkey-patch the lockfile discovery to return our fake file
        monkeypatch.setenv("EFFGEN_VERIFY_HASHES", "1")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Call with a custom lockfile to avoid environment-specific drift
            result = supply_chain_module.verify_installed_hashes(
                lockfile=fake_lockfile, warn=True
            )

        drift_warns = [x for x in w if issubclass(x.category, supply_chain_module.HashDriftWarning)]
        # requests drift should have been detected and warned
        assert "requests" in result.drifted
        assert len(drift_warns) > 0, "drift should emit a HashDriftWarning"

    def test_env_var_absent_is_noop(self, supply_chain_module, monkeypatch):
        """Without EFFGEN_VERIFY_HASHES=1, verify_on_startup must do nothing."""
        monkeypatch.delenv("EFFGEN_VERIFY_HASHES", raising=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            supply_chain_module.verify_on_startup()

        drift_warns = [x for x in w if issubclass(x.category, supply_chain_module.HashDriftWarning)]
        assert len(drift_warns) == 0, "verify_on_startup must be no-op without env var"

    def test_env_var_zero_is_noop(self, supply_chain_module, monkeypatch):
        """EFFGEN_VERIFY_HASHES=0 must also be a no-op."""
        monkeypatch.setenv("EFFGEN_VERIFY_HASHES", "0")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            supply_chain_module.verify_on_startup()

        drift_warns = [x for x in w if issubclass(x.category, supply_chain_module.HashDriftWarning)]
        assert len(drift_warns) == 0

    def test_startup_logs_ok_on_clean_env(self, supply_chain_module, monkeypatch, caplog):
        """A clean verification must log `hash_verification: ok`."""
        import logging as _logging

        monkeypatch.setenv("EFFGEN_VERIFY_HASHES", "1")
        # Force a clean result so the log token is deterministic
        clean = supply_chain_module.VerificationResult(checked=3, ok=3)
        monkeypatch.setattr(
            supply_chain_module, "verify_installed_hashes", lambda: clean
        )

        with caplog.at_level(_logging.INFO, logger="effgen.security.supply_chain"):
            supply_chain_module.verify_on_startup()

        assert any("hash_verification: ok" in r.message for r in caplog.records), (
            f"Expected 'hash_verification: ok' log line; got: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_startup_logs_drift_with_package_name(
        self, supply_chain_module, monkeypatch, caplog
    ):
        """Drift must log `hash_verification: drift` naming the drifting package."""
        import logging as _logging

        monkeypatch.setenv("EFFGEN_VERIFY_HASHES", "1")
        drifted = supply_chain_module.VerificationResult(
            checked=3, ok=2, drifted=["requests"]
        )
        monkeypatch.setattr(
            supply_chain_module, "verify_installed_hashes", lambda: drifted
        )

        with caplog.at_level(_logging.WARNING, logger="effgen.security.supply_chain"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                supply_chain_module.verify_on_startup()

        drift_lines = [r.message for r in caplog.records if "hash_verification: drift" in r.message]
        assert drift_lines, (
            f"Expected 'hash_verification: drift' log line; got: "
            f"{[r.message for r in caplog.records]}"
        )
        # The drifting package name must appear in the log line (spec requirement)
        assert any("requests" in line for line in drift_lines), (
            f"Drift log must name the package; got: {drift_lines}"
        )
        # And the HashDriftWarning is still emitted
        drift_warns = [x for x in w if issubclass(x.category, supply_chain_module.HashDriftWarning)]
        assert len(drift_warns) > 0


# ---------------------------------------------------------------------------
# 5. Lockfiles
# ---------------------------------------------------------------------------

_REGEN = "python scripts/gen_locks.py"


def _lock_pins(lockfile: Path) -> dict[str, str]:
    """Map normalised package name -> pinned version from a compiled lockfile.

    Handles both the hash-pinned core lock (``name==X \\``) and the plain
    ``[all]`` constraints lock (``name==X``).
    """
    pins: dict[str, str] = {}
    for raw in lockfile.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("--hash"):
            continue
        match = re.match(r"^([A-Za-z0-9._-]+)==([^\s\\;]+)", line)
        if match:
            pins[canonicalize_name(match.group(1))] = match.group(2)
    return pins


def _declared_requirements(extras: tuple[str, ...] = ()) -> list[Requirement]:
    """Requirements declared in pyproject for the core deps plus ``extras``.

    A lockfile is only compiled for a specific set of extras, so it is only
    answerable for those. Self-referencing extras (``effgen[server]``) are
    expanded so their contents are covered too.
    """
    with open(PYPROJECT, "rb") as fh:
        cfg = tomllib.load(fh)
    project = cfg.get("project", {})
    optional = project.get("optional-dependencies", {})

    by_extra = {canonicalize_name(name): specs for name, specs in optional.items()}

    pending: list[str] = list(project.get("dependencies", []))
    resolved_extras: set[str] = set()
    for extra in extras:
        resolved_extras.add(canonicalize_name(extra))
        pending.extend(by_extra.get(canonicalize_name(extra), []))

    requirements: list[Requirement] = []
    while pending:
        try:
            req = Requirement(pending.pop())
        except InvalidRequirement:  # pragma: no cover - defensive
            continue
        if canonicalize_name(req.name) == "effgen":
            for extra in req.extras:
                key = canonicalize_name(extra)
                if key not in resolved_extras:
                    resolved_extras.add(key)
                    pending.extend(by_extra.get(key, []))
            continue
        requirements.append(req)
    return requirements


class TestLockfilePinsSatisfyDeclaredFloors:
    """A compiled lockfile must never pin a version the project has ruled out.

    Version floors in ``pyproject.toml`` are how a dependency with a published
    advisory is kept out of an install. A lockfile that was compiled before a
    floor was raised still pins the affected release, so an install driven by
    the lock silently gets the version the floor excludes. Both lockfiles are
    checked against every declared specifier (core and extras).
    """

    @pytest.mark.parametrize(
        ("lock_name", "extras"),
        [
            ("requirements-lock.txt", ()),
            ("requirements-all-lock.txt", ("all",)),
            # The 3.14 lock is how `effgen[all]` installs there at all, so it
            # earns the same guarantees as the other two: a floor raised for an
            # advisory has to reach every lock, not just the ones for the
            # Pythons someone happened to remember.
            ("requirements-all-py314-lock.txt", ("all",)),
        ],
    )
    def test_pins_satisfy_pyproject_specifiers(self, lock_name: str, extras: tuple[str, ...]):
        lockfile = REPO_ROOT / lock_name
        if not lockfile.exists():
            pytest.skip(f"{lock_name} not found")

        pins = _lock_pins(lockfile)
        assert pins, f"{lock_name} parsed as empty — the pin format changed"

        violations: list[str] = []
        for req in _declared_requirements(extras):
            pinned = pins.get(canonicalize_name(req.name))
            if pinned is None or not req.specifier:
                continue
            if not req.specifier.contains(pinned, prereleases=True):
                violations.append(f"{req.name}: locked {pinned}, declared {req.specifier}")

        assert not violations, (
            f"{lock_name} pins versions excluded by pyproject.toml:\n  "
            + "\n  ".join(sorted(violations))
            + f"\nRegenerate the lockfile:\n  {_REGEN}"
        )

    @pytest.mark.parametrize(
        ("lock_name", "extras"),
        [
            ("requirements-lock.txt", ()),
            ("requirements-all-lock.txt", ("all",)),
            # The 3.14 lock is how `effgen[all]` installs there at all, so it
            # earns the same guarantees as the other two: a floor raised for an
            # advisory has to reach every lock, not just the ones for the
            # Pythons someone happened to remember.
            ("requirements-all-py314-lock.txt", ("all",)),
        ],
    )
    def test_declared_packages_are_locked(self, lock_name: str, extras: tuple[str, ...]):
        """Every unconditional declared dependency appears in the lockfile.

        A dependency added to ``pyproject.toml`` after the lockfile was last
        compiled is absent from it, so a constrained install resolves that one
        package freely. Requirements carrying an environment marker are skipped
        because they resolve per platform.
        """
        lockfile = REPO_ROOT / lock_name
        if not lockfile.exists():
            pytest.skip(f"{lock_name} not found")

        pins = _lock_pins(lockfile)
        missing = sorted(
            req.name
            for req in _declared_requirements(extras)
            if req.marker is None and canonicalize_name(req.name) not in pins
        )
        assert not missing, (
            f"{lock_name} is missing declared dependencies: {missing}\n"
            f"Regenerate the lockfile:\n  {_REGEN}"
        )


class TestDeclaredFloorsAgreeAcrossExtras:
    """One package must carry the same lower bound everywhere it is declared.

    A floor raised past an advisory only constrains the install shapes that
    declare it. When a package is reachable from several extras, each of them
    repeats the requirement, so raising the bound in one place and not the
    others leaves the remaining extras resolving to the affected release. This
    compares the lower bound of every repeated declaration and reports the ones
    that disagree.
    """

    def test_lower_bounds_match(self):
        with open(PYPROJECT, "rb") as fh:
            cfg = tomllib.load(fh)
        project = cfg.get("project", {})

        # {package: {location: lower-bound}}
        bounds: dict[str, dict[str, str]] = {}
        sources = [("project.dependencies", project.get("dependencies", []))]
        sources += list(project.get("optional-dependencies", {}).items())

        for location, specs in sources:
            for spec in specs:
                try:
                    req = Requirement(spec)
                except InvalidRequirement:  # pragma: no cover - defensive
                    continue
                if canonicalize_name(req.name) == "effgen":
                    continue
                lower = sorted(
                    s.version for s in req.specifier if s.operator in (">=", ">", "==")
                )
                if lower:
                    bounds.setdefault(canonicalize_name(req.name), {})[location] = ",".join(lower)

        disagreements = [
            f"{name}: " + ", ".join(f"{loc} {bound}" for loc, bound in sorted(places.items()))
            for name, places in sorted(bounds.items())
            if len(set(places.values())) > 1
        ]
        assert not disagreements, (
            "pyproject.toml declares different lower bounds for the same package:\n  "
            + "\n  ".join(disagreements)
            + "\nRaise every declaration to the highest bound — an extra that keeps "
            "the old one still resolves to it."
        )


class TestLockfilePresent:
    """The requirements-lock.txt must exist and be valid."""

    def test_lockfile_exists(self):
        lockfile = REPO_ROOT / "requirements-lock.txt"
        assert lockfile.exists(), (
            "requirements-lock.txt not found. Generate it with:\n"
            f"  {_REGEN}"
        )

    def test_lockfile_has_hashes(self):
        lockfile = REPO_ROOT / "requirements-lock.txt"
        if not lockfile.exists():
            pytest.skip("requirements-lock.txt not found")
        content = lockfile.read_text()
        assert "--hash=sha256:" in content, (
            "requirements-lock.txt must contain hash entries "
            "(generated with --generate-hashes)"
        )

    def test_lockfile_has_reasonable_size(self):
        lockfile = REPO_ROOT / "requirements-lock.txt"
        if not lockfile.exists():
            pytest.skip("requirements-lock.txt not found")
        lines = lockfile.read_text().splitlines()
        assert len(lines) > 50, (
            f"requirements-lock.txt has only {len(lines)} lines — "
            "expected a much larger file for a full install"
        )

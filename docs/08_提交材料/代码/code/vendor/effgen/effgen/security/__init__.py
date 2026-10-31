"""
effgen.security — Security utilities for effGen.

Sub-modules:
  supply_chain  — startup integrity verification (EFFGEN_VERIFY_HASHES=1).
  sandbox       — code execution sandboxing (DockerSandbox, SubprocessSandbox).
"""

from effgen.security.sandbox import (
    DockerSandbox,
    FirecrackerSandbox,
    OffSandbox,
    SandboxBase,
    SandboxConfig,
    SandboxResult,
    SubprocessSandbox,
    get_sandbox,
    reset_sandbox_cache,
)
from effgen.security.supply_chain import (
    HashDriftWarning,
    VerificationResult,
    verify_installed_hashes,
    verify_on_startup,
)

__all__ = [
    # supply_chain
    "HashDriftWarning",
    "VerificationResult",
    "verify_installed_hashes",
    "verify_on_startup",
    # sandbox
    "SandboxConfig",
    "SandboxResult",
    "SandboxBase",
    "DockerSandbox",
    "SubprocessSandbox",
    "FirecrackerSandbox",
    "OffSandbox",
    "get_sandbox",
    "reset_sandbox_cache",
]

"""Every provider adapter against every way a provider call goes wrong.

One table: nine adapters by nine failure classes. Each cell drives the real
adapter through its real SDK over a real socket against a loopback peer that
produces the failure at the transport layer, and asserts the same five things —
a typed, classified error; the right retry verdict; no hang; no credential
handed back to the caller; and never a quiet success.

Covering a new provider is a row in
:data:`tests.unit.failure_injection.PROVIDERS`; covering a new failure class is
a row in :data:`~tests.unit.failure_injection.FAILURES` plus one handler on the
fault server. Neither needs a new module, and
:func:`test_the_matrix_covers_every_adapter_and_class` fails if either is
skipped.
"""

from __future__ import annotations

import pytest

from .failure_injection import (
    DIALECTS,
    FAILURES,
    PROVIDERS,
    SENTINEL_CREDENTIAL,
    FaultServer,
    MatrixReport,
    run_cell,
)

# The whole matrix binds loopback sockets and answers itself; nothing here
# reaches a provider, and no credential is read from the environment.
pytestmark = pytest.mark.timeout(120)


@pytest.mark.parametrize(
    "provider", PROVIDERS, ids=[p.name for p in PROVIDERS]
)
@pytest.mark.parametrize(
    "failure", FAILURES, ids=[f.name for f in FAILURES]
)
def test_failure_cell(provider, failure) -> None:
    """One provider, one failure class, every rule."""
    outcome = run_cell(provider, failure)
    if outcome.skipped:
        pytest.skip(f"{provider.name}/{failure.name}: {outcome.skipped}")
    report = MatrixReport([outcome])
    report.assert_all_green(f"{provider.name} × {failure.name}")


def test_the_matrix_covers_every_adapter_and_class() -> None:
    """The table has to stay a table.

    A provider added to the cloud adapter set without a row here would ship
    with none of its failure behavior pinned, and a failure class without a
    handler for every dialect would silently cover fewer providers than the
    table claims.
    """
    from effgen.models.registry import ProviderRegistry

    # Registered names that are not a hosted provider, and so have no row.
    # Every cell here is defined in terms of a service effGen calls on the
    # caller's behalf with a credential it reads from the environment.
    NOT_A_HOSTED_PROVIDER = {
        # A protocol, not a service: the endpoint is whatever the caller runs,
        # and the adapter reads no credential — api_key defaults to a
        # placeholder because a local server checks nothing. The wire behaviour
        # it inherits is the "openai" row, which drives the same dialect
        # through the same adapter class against the same fault server.
        # tests/models/test_openai_compatible.py covers what is specific to it,
        # including an endpoint nobody is serving.
        "openai_compatible",
    }
    covered = {p.name for p in PROVIDERS} | NOT_A_HOSTED_PROVIDER
    missing = set(ProviderRegistry.list_providers()) - covered
    assert not missing, (
        f"{sorted(missing)} are registered providers with no row in the failure "
        f"matrix. Add a ProviderSpec to tests/unit/failure_injection.py."
    )

    for failure in FAILURES:
        if not failure.mode:
            continue
        assert hasattr(FaultServer, f"_inject_{failure.mode}"), (
            f"failure class {failure.name!r} names mode {failure.mode!r}, which "
            f"the fault server has no handler for"
        )

    for dialect in {p.dialect for p in PROVIDERS}:
        assert dialect in DIALECTS, f"no wire dialect named {dialect!r}"

    for provider in PROVIDERS:
        assert provider.sdk_module, (
            f"{provider.name} names no SDK module, so on an install without "
            f"that provider's extra its cells fail on the import instead of "
            f"recording that they were not covered"
        )


def test_the_sentinel_credential_is_recognisable_to_the_redactor() -> None:
    """The whole matrix's credential check rests on this being redactable.

    A sentinel the process redactor does not match would make every
    "credential not echoed" cell pass for the wrong reason.
    """
    from effgen.observability.redact import get_redactor

    scrubbed = get_redactor().scrub(f"key={SENTINEL_CREDENTIAL}")
    assert SENTINEL_CREDENTIAL not in scrubbed

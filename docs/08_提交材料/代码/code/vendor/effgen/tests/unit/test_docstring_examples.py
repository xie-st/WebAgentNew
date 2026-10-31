"""``>>>`` examples that claim to be executable must actually execute.

effGen carries `>>>` examples in docstrings and ran no doctest lane, so they
were only ever read. Running them over the whole package finds two populations,
and conflating them is why the question "should the examples be executable?" had
no answer:

* **executable** — examples over pure helpers, which run and produce a stable
  transcript. These are documentation that can be wrong, so they are held to it
  here.
* **illustrative** — examples that load a model, build an agent, open a provider
  connection, or emit a log line whose visibility depends on handler
  configuration. Making these executable would mean either a live provider or
  rewriting them into something less useful as documentation, so they are
  declared below with the reason instead.

The lane is the executable list. A module may only leave it by being declared
illustrative, which is a visible edit with a reason attached — the point being
that an example cannot silently stop being checked.
"""
from __future__ import annotations

import contextlib
import doctest
import importlib
import io

import pytest

FLAGS = (
    doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE | doctest.IGNORE_EXCEPTION_DETAIL
)

#: Modules whose ``>>>`` examples run and are held to their transcripts.
EXECUTABLE_MODULES = (
    "effgen.api.embeddings",
    "effgen.config.loader",
    "effgen.gpu.monitor",
    "effgen.models.cerebras_models",
    "effgen.rag.chunking",
    "effgen.utils.validators",
)

#: Modules whose examples are illustrative, and why. Each needs something a
#: unit test must not have — a provider, a GPU, a model download — or produces
#: output that depends on how logging happens to be configured.
ILLUSTRATIVE_MODULES = {
    "effgen.client": "constructs a client and calls a provider",
    "effgen.domains.base": "builds an agent, which needs a model",
    "effgen.gpu.allocator": "allocates real GPUs",
    "effgen.models": "loads a model",
    "effgen.models.model_loader": "loads a model",
    "effgen.presets.registry": "builds an agent, which needs a model",
    "effgen.prompts.optimizer": "runs an optimization loop against a model",
    "effgen.utils": "package overview; mixes agent and logging examples",
    "effgen.utils.logging": "emits log lines whose visibility is handler-dependent",
    "effgen.utils.metrics": "records timings, so the transcript is not stable",
}


def _run(module_name: str) -> tuple[int, int, str]:
    """Return ``(blocks, failures, report)`` for one module's examples."""
    module = importlib.import_module(module_name)
    tests = [t for t in doctest.DocTestFinder().find(module) if t.examples]
    failures = 0
    buffer = io.StringIO()
    for test in tests:
        runner = doctest.DocTestRunner(optionflags=FLAGS, verbose=False)
        with contextlib.redirect_stdout(buffer):
            runner.run(test, clear_globs=False)
        failures += runner.failures
    return len(tests), failures, buffer.getvalue()


@pytest.mark.parametrize("module_name", EXECUTABLE_MODULES)
def test_every_executable_example_runs_and_matches_its_transcript(module_name):
    blocks, failures, report = _run(module_name)
    assert blocks, f"{module_name} is on the executable list but carries no examples"
    assert failures == 0, report


def test_an_illustrative_module_carries_a_reason():
    """A module leaves the lane only with a stated reason, never silently."""
    assert all(reason.strip() for reason in ILLUSTRATIVE_MODULES.values())
    assert not set(EXECUTABLE_MODULES) & set(ILLUSTRATIVE_MODULES)


def test_no_module_with_examples_is_unaccounted_for():
    """Every ``>>>`` block in the package is either checked or declared.

    Without this, adding examples to a new module would leave them unchecked and
    undeclared — which is the state the whole package was in.
    """
    import pkgutil

    import effgen

    accounted = set(EXECUTABLE_MODULES) | set(ILLUSTRATIVE_MODULES)
    unaccounted = []
    for module_info in pkgutil.walk_packages(effgen.__path__, "effgen."):
        name = module_info.name
        if name in accounted:
            continue
        try:
            module = importlib.import_module(name)
        except Exception:  # noqa: BLE001 - an optional extra that is not installed
            continue
        try:
            if any(t.examples for t in doctest.DocTestFinder().find(module)):
                unaccounted.append(name)
        except Exception:  # noqa: BLE001 - a module doctest cannot introspect
            continue
    assert not unaccounted, (
        "modules carrying >>> examples that are neither checked nor declared "
        f"illustrative: {unaccounted}"
    )

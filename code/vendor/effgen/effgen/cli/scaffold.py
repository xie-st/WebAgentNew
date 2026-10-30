"""Project scaffolding for ``effgen quickstart --init``.

Turns an empty directory into a project that runs: an agent configuration
``effgen run -c`` reads, an ``.env`` template naming the provider variables
without inventing values, one runnable example script, a ``.gitignore`` that
keeps the filled-in ``.env`` out of version control, and a printed list of the
next three commands.

Nothing here prompts, so the same call works on a terminal, in a pipe and in
CI. Every write lands inside the target directory; the one file written outside
it is the daily spend cap, which uses the same ``~/.effgen/budget.json`` that
``effgen cost set-budget`` writes and is only set when no cap exists yet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "PROJECT_CONFIG_NAME",
    "ENV_TEMPLATE_NAME",
    "EXAMPLE_NAME",
    "GITIGNORE_NAME",
    "SCAFFOLD_FILES",
    "DEFAULT_DAILY_BUDGET_USD",
    "DEFAULT_MAX_TOKENS",
    "ScaffoldResult",
    "provider_env_keys",
    "render_project_config",
    "render_env_template",
    "render_example",
    "render_gitignore",
    "resolve_model",
    "resolve_max_tokens",
    "next_commands",
    "scaffold_project",
]

#: The agent configuration ``effgen run -c`` reads.
PROJECT_CONFIG_NAME = "effgen.yaml"
#: The key template a user copies to ``.env`` and fills in.
ENV_TEMPLATE_NAME = ".env.example"
#: The runnable example script.
EXAMPLE_NAME = "example.py"
#: The ignore file that keeps the filled-in ``.env`` out of version control.
GITIGNORE_NAME = ".gitignore"

#: Every file the scaffold writes, in the order it reports them, with the
#: one-line description printed beside each.
SCAFFOLD_FILES: tuple[tuple[str, str], ...] = (
    (PROJECT_CONFIG_NAME, "model, prompt and per-run caps"),
    (ENV_TEMPLATE_NAME, "key names, no values"),
    (EXAMPLE_NAME, "a runnable agent script"),
    (GITIGNORE_NAME, "keeps .env out of git"),
)

#: The daily spend cap set when the user has none, in USD.
DEFAULT_DAILY_BUDGET_USD = 1.0

#: The per-answer output budget written for an ordinary model. A model that
#: spends part of its budget on hidden reasoning gets the larger one the
#: framework asks for instead — see :func:`resolve_max_tokens`.
DEFAULT_MAX_TOKENS = 512

#: The sample question the scaffolded example and the printed commands use.
SAMPLE_TASK = "What is 25 * 17?"


@dataclass
class ScaffoldResult:
    """What :func:`scaffold_project` wrote, skipped, or could not write."""

    directory: Path
    model: str
    #: Why this model was chosen, for the caller to print beside it.
    model_reason: str = ""
    created: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    #: The daily cap that is now in force, or ``None`` when none was set.
    daily_budget_usd: float | None = None
    #: True when this call set the cap rather than finding one already set.
    budget_was_set: bool = False


def provider_env_keys() -> list[tuple[str, list[str]]]:
    """Return ``(provider, [env var, ...])`` for every registered provider.

    Read from the provider registry rather than from a hardcoded list, so a
    template written today names the variables this build actually checks.
    """
    try:
        from effgen.models.auth import check_keys
    except Exception:  # noqa: BLE001 - a template must not depend on adapters
        return []
    # Import the adapters so they self-register, the same way `effgen doctor`
    # does before it reports the key table.
    for module in (
        "anthropic_adapter", "cerebras_adapter", "fireworks_adapter",
        "gemini_adapter", "groq_adapter", "hf_inference_adapter",
        "openai_adapter", "replicate_adapter", "together_adapter",
    ):
        try:
            __import__(f"effgen.models.{module}")
        except Exception:  # noqa: BLE001 - a provider that will not import is skipped
            continue
    try:
        results = check_keys()
    except Exception:  # noqa: BLE001
        return []
    out: list[tuple[str, list[str]]] = []
    for provider in sorted(results):
        keys = [str(k) for k in results[provider].get("env_keys_checked", []) if k]
        if keys:
            out.append((provider, keys))
    return out


def next_commands(config_name: str = PROJECT_CONFIG_NAME) -> list[str]:
    """The three commands printed after a scaffold, in the order to run them."""
    return [
        f"cp {ENV_TEMPLATE_NAME} .env  # then paste one key into it",
        "effgen doctor  # confirm effGen sees it",
        f'effgen run "{SAMPLE_TASK}" -c {config_name}',
    ]


def resolve_max_tokens(model: str) -> int:
    """The per-answer output budget *model* needs to answer at all.

    A reasoning model spends part of its output budget on hidden reasoning
    before it emits a visible token, so the ordinary budget can be consumed
    entirely and return an empty — but still billed — answer. The framework
    already knows which models those are and how much headroom they need; the
    scaffold asks it rather than writing one number for every model.
    """
    try:
        from types import SimpleNamespace

        from effgen.models._adapter_utils import (
            default_max_output_tokens,
            needs_reasoning_headroom,
        )

        # Both helpers read ``model_name`` off an adapter; a bare string has no
        # such attribute and would silently read as "no headroom needed".
        probe = SimpleNamespace(model_name=model)
        if needs_reasoning_headroom(probe):
            return max(DEFAULT_MAX_TOKENS, int(default_max_output_tokens(probe)))
    except Exception:  # noqa: BLE001 - a scaffold still writes a usable budget
        pass
    return DEFAULT_MAX_TOKENS


def render_project_config(model: str, *, config_name: str = PROJECT_CONFIG_NAME) -> str:
    """The agent configuration, as YAML text.

    Carries only the keys ``effgen run`` reads from a config file, so loading
    it reports nothing unapplied.
    """
    steps = "\n".join(f"#   {i}. {c}" for i, c in enumerate(next_commands(config_name), 1))
    max_tokens = resolve_max_tokens(model)
    headroom = ""
    if max_tokens > DEFAULT_MAX_TOKENS:
        headroom = (
            f"#\n# {model} spends part of its output budget on hidden reasoning,\n"
            f"# so max_tokens is {max_tokens} here rather than the usual "
            f"{DEFAULT_MAX_TOKENS}. A smaller\n"
            "# budget can run out before the first visible word, and that empty\n"
            "# answer is still billed.\n"
        )
    return f"""\
# effGen project configuration.
#
# Used by:  effgen run "your task" -c {config_name}
#
# Next three commands:
{steps}
#
# max_tokens caps what one answer may cost; max_iterations caps how many steps
# the agent may take on one task. Both apply to every run that loads this file.
# A daily spend cap across all runs is separate: effgen cost set-budget 1.00
{headroom}
model: {model}
system_prompt: You are a helpful assistant. Answer concisely and say when you are unsure.
temperature: 0.2
max_tokens: {max_tokens}
max_iterations: 5
"""


def render_env_template(providers: list[tuple[str, list[str]]] | None = None) -> str:
    """The ``.env`` template: every provider variable, with no value filled in.

    A credential is named and left empty. Nothing here is a key, and an empty
    value counts as absent, so copying this file to ``.env`` unchanged leaves
    ``effgen doctor`` reporting exactly the keys the environment already has.

    The endpoint variables are the exception and are written commented out. They
    do not name a credential, they name an address, and every OpenAI-protocol
    client on the machine reads ``OPENAI_BASE_URL`` — including ones that treat
    an empty value as a real address and send their requests to nowhere. A line
    the user has to uncomment cannot do that by being copied unchanged.
    """
    from effgen.models._base_url import BASE_URL_ENV_VARS

    if providers is None:
        providers = provider_env_keys()
    lines = [
        "# effGen provider keys.",
        "#",
        f"#   cp {ENV_TEMPLATE_NAME} .env",
        "#   $EDITOR .env          # paste your key after the '=' of one line",
        "#   effgen doctor         # confirms which keys effGen can see",
        "#",
        "# effGen loads the nearest .env automatically, walking up from the",
        "# directory you run in. A variable already set in the environment wins",
        "# over this file, and an empty value counts as no key at all. Set",
        "# EFFGEN_NO_DOTENV=1 to skip the file search entirely.",
        "#",
        "# .env is listed in .gitignore — keep it that way.",
        "",
    ]
    if not providers:
        lines.append("# No providers are registered in this installation.")
        return "\n".join(lines) + "\n"
    for provider, keys in providers:
        lines.append(f"# {provider}")
        for key in keys:
            if key in BASE_URL_ENV_VARS:
                lines.append(f"# {key}=          # an address, not a key: uncomment to use")
            else:
                lines.append(f"{key}=")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def render_example(model: str, *, config_name: str = PROJECT_CONFIG_NAME) -> str:
    """The runnable example script, as Python source."""
    max_tokens = resolve_max_tokens(model)
    return f'''\
"""One effGen agent, one question, one answer — with what it cost.

Run it from this directory:

    python {EXAMPLE_NAME}
    python {EXAMPLE_NAME} "your own question"

The model and the caps match {config_name}, so this script and
`effgen run -c {config_name}` do the same thing.
"""

from __future__ import annotations

import sys

from effgen import Agent, AgentConfig, load_env

# Keep in step with the `model:` line in {config_name}.
MODEL = "{model}"


def main() -> int:
    # Reads the .env next to this file (and any parent), so a key pasted there
    # is picked up without exporting anything.
    load_env()

    task = " ".join(sys.argv[1:]) or "{SAMPLE_TASK}"
    agent = Agent(AgentConfig(
        name="example",
        model=MODEL,
        system_prompt="You are a helpful assistant. Answer concisely.",
        temperature=0.2,
        max_tokens={max_tokens},
        max_iterations=5,
    ))
    try:
        response = agent.run(task)
    finally:
        agent.close()

    print(response.text)
    if not response.success:
        # A failure carries its own message in the text above; the exit code is
        # what a script downstream can act on.
        return 1

    metadata = response.metadata or {{}}
    cost = metadata.get("cost_usd")
    tokens = metadata.get("total_tokens")
    if cost is not None:
        print(f"\\ncost: ${{cost:.6f}}  tokens: {{tokens}}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def render_gitignore() -> str:
    """The ignore file, covering the filled-in ``.env`` and local state."""
    return """\
# Provider keys — never commit this file.
.env

# effGen local state (sessions, runs, checkpoints, baselines).
.effgen/
checkpoints/

# Python
__pycache__/
*.py[cod]
.venv/
"""


def resolve_model() -> tuple[str, str]:
    """Return ``(model_id, reason)`` for the scaffold to write into the files.

    The id is always written in the ``provider:model`` form. A scaffolded file
    carries no ``--provider`` flag beside it, and a bare cloud id would be
    routed by whatever the catalog matched first, so the prefix is what makes
    the written config mean one thing.
    """
    try:
        from effgen.cli.commands._shared import _quickstart_suggest_model

        model_id, provider, reason = _quickstart_suggest_model()
        if provider and ":" not in model_id:
            model_id = f"{provider}:{model_id}"
        return model_id, reason
    except Exception:  # noqa: BLE001 - a scaffold must still write something usable
        return "transformers:Qwen/Qwen2.5-1.5B-Instruct", "model catalog unavailable"


def _set_budget(result: ScaffoldResult, budget: float | None) -> None:
    """Put a daily spend cap in force, unless one is already configured.

    *budget* of ``None`` means "leave the cap alone"; ``0`` and below means the
    caller asked for no cap. An existing cap is never overwritten — the scaffold
    reports it instead.
    """
    from effgen.cli.commands.cost import configured_daily_budget, set_daily_budget

    existing = configured_daily_budget()
    if existing is not None:
        result.daily_budget_usd = existing
        return
    if budget is None or budget <= 0:
        return
    try:
        set_daily_budget(budget)
    except OSError as e:
        result.errors.append(f"could not set the daily spend cap: {e}")
        return
    result.daily_budget_usd = budget
    result.budget_was_set = True


def scaffold_project(
    directory: str | os.PathLike[str] | None = None,
    *,
    force: bool = False,
    budget: float | None = DEFAULT_DAILY_BUDGET_USD,
    model: str | None = None,
) -> ScaffoldResult:
    """Write the project files into *directory* and return what happened.

    Args:
        directory: Where to write. Created if missing; defaults to the current
            directory.
        force: Overwrite a file that is already there. Without it an existing
            file is left untouched and reported as skipped.
        budget: Daily spend cap in USD to put in force when none is configured.
            ``None`` or a non-positive value leaves the cap alone.
        model: Model id to write into the files. Defaults to the same choice
            ``effgen quickstart`` and ``effgen run`` make with no ``-m``.

    Returns:
        A :class:`ScaffoldResult` naming what was created, skipped and refused.
        Writes never leave *directory*, apart from the spend cap.
    """
    target = Path(directory).expanduser() if directory else Path.cwd()
    if model:
        model_id, model_reason = model, "given with -m/--model"
    else:
        model_id, model_reason = resolve_model()
    result = ScaffoldResult(directory=target, model=model_id, model_reason=model_reason)

    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        result.errors.append(f"could not create {target}: {e}")
        return result

    bodies = {
        PROJECT_CONFIG_NAME: render_project_config(model_id),
        ENV_TEMPLATE_NAME: render_env_template(),
        EXAMPLE_NAME: render_example(model_id),
        GITIGNORE_NAME: render_gitignore(),
    }
    for name, _description in SCAFFOLD_FILES:
        path = target / name
        if path.exists() and not force:
            result.skipped.append(name)
            continue
        try:
            path.write_text(bodies[name], encoding="utf-8")
        except OSError as e:
            result.errors.append(f"could not write {name}: {e}")
            continue
        result.created.append(name)

    _set_budget(result, budget)
    return result

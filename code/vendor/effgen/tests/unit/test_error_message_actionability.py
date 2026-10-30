"""Every failure a user reads names what happened and what to do next.

A typed effGen error is the framework's failure contract, so its message has to
carry three properties:

* **Actionable.** The first sentence says what happened; a later sentence says
  what the reader can do about it — an imperative ("set the key", "run
  ``effgen models list``") or an enumeration of what would have been accepted.
  A single sentence naming a symptom leaves the reader with nowhere to go.
* **Bounded.** A message that quotes upstream text — a provider error body, a
  parser complaint, an SDK string — quotes a bounded amount of it. Providers
  echo the rejected request back, so an unbounded quote buries the reason in
  kilobytes of payload wherever the message lands: a terminal panel, a log
  line, an API envelope.
* **Redacted.** Credentials that an upstream echoed back never survive into the
  message.

Two groups are checked, because effGen's errors compose their messages in two
places. A class that defines its own ``__init__`` builds the message itself, so
it is constructed here and its ``str()`` is read. A class that passes its
arguments through to ``Exception`` gets its message from the ``raise``
statement, so those statements are read from the source instead.

Exempt, with the reason each time:

* **Private classes** (a leading underscore). These are internal control-flow
  signals caught by the module that raises them; no user reads one.
* **:mod:`effgen.reliability.chaos`.** Fault injection synthesizes provider
  failures on purpose, and a synthetic fault that did not read like the real
  one would not exercise the handling under test.
* **:class:`effgen.errors.RunStoppedError`.** Its ``__init__`` attaches the run
  it carries and forwards the run's own outcome statement as the message, so
  there is nothing here for it to compose. That statement is built by the loop
  and checked where it is built.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import re
from pathlib import Path

import pytest

import effgen

PACKAGE_ROOT = Path(effgen.__file__).parent

# Engines for hardware this suite does not run on; importing them to enumerate
# their errors would fail for a reason that has nothing to do with wording.
_UNIMPORTABLE = {"effgen.models.mlx_engine", "effgen.models.mlx_vlm_engine"}

# Fault injection imitates a provider failure deliberately; see the module
# docstring.
_EXEMPT_MODULES = {"effgen.reliability.chaos"}

# Classes whose ``__init__`` attaches structured data and forwards a message
# built elsewhere; see the module docstring.
_FORWARDS_ITS_MESSAGE = {"effgen.errors.RunStoppedError"}

# Parameters that carry text produced somewhere else — a provider body, an SDK
# message, a parser complaint. These are the values that can arrive long or
# carrying a credential.
_UPSTREAM_PARAMS = frozenset({
    "message", "msg", "reason", "detail", "details", "body", "text", "error",
    "cause", "refusal_message", "extra", "hint", "install_instructions",
    "explanation", "description", "context",
})

# A credential shaped the way OpenAI issues them, planted in the upstream text
# so a message that quotes the body verbatim is caught.
_PLANTED_KEY = "sk-proj-AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHHIIIIJJJJKKKKLLLL"

# What a provider that echoes the rejected call back sends: long, and carrying
# the credential it was given.
_UPSTREAM_BODY = (
    f"Incorrect API key provided: {_PLANTED_KEY}. " + "PAYLOAD" * 2000
)

# Short, and deliberately free of anything that reads as guidance. The
# actionability check builds errors with this so the guidance a message carries
# has to be the framework's own — an upstream whose text happens to contain an
# imperative must not stand in for a message that says nothing.
_NEUTRAL_BODY = "the upstream said no"

# Longest message a reader should have to scan. Well above the room every
# composed message needs for its own prose plus one bounded quote, and far
# below the size of the payload planted above.
_MESSAGE_LIMIT = 2000

# Verbs that tell the reader to do something. This is the vocabulary the
# framework's messages are written against, not an attempt at every English
# imperative: a message reaching for a verb outside it should either use one of
# these or earn its place here.
_IMPERATIVES = (
    "add", "adjust", "avoid", "await", "call", "cancel", "check", "choose",
    "clear", "configure", "consider", "convert", "correct", "create",
    "declare", "delete", "disable", "disambiguate", "downgrade", "drop",
    "enable", "ensure", "escape", "export", "fetch", "fix", "free", "give",
    "grant",
    "import", "increase", "inspect", "install", "keep", "lower", "make",
    "mint", "move", "name", "narrow", "open", "pass", "pick", "point",
    "prefer", "prefix", "provide", "qualify", "quote", "raise", "re-export",
    "re-run", "rebuild", "reduce", "register", "reinstall", "remove",
    "rename", "rephrase", "replace", "request", "rerun", "restart", "restore",
    "retry", "route", "run", "save", "see", "send", "set", "shorten", "slow",
    "specify", "split", "start", "supply", "swap", "try", "unset", "update",
    "upgrade", "use", "verify", "wait", "widen", "wrap",
)

# Phrases that tell the reader what would have been accepted instead.
_ENUMERATIONS = (
    "available", "expected", "known", "supported", "valid", "either", "one of",
    "must be", "did you mean", "instead", "accepted", "allowed", "choices",
)

_ACTION_CUE = re.compile(
    r"(?<![\w-])(?:" + "|".join(_IMPERATIVES) + r")(?![\w-])"
    r"|(?:" + "|".join(p.replace(" ", r"\s+") for p in _ENUMERATIONS) + r")",
    re.IGNORECASE,
)


def _sentences(message: str) -> list[str]:
    """Split *message* into sentences, treating a newline as a break.

    A ``{}`` marks where a runtime value is interpolated when the message came
    from a source template rather than from a constructed error. It stands in
    for text, so it is replaced by a word here — otherwise a sentence ending
    right before one would not be seen to end.
    """
    flattened = message.replace("{}", " value ").replace("\n", ". ")
    flattened = re.sub(r"\s+", " ", flattened).strip()
    return [s for s in re.split(r"(?<=[.!?])\s+", flattened) if s.strip()]


def is_actionable(message: str) -> bool:
    """True when *message* says what happened and then what to do next."""
    parts = _sentences(message)
    return len(parts) >= 2 and any(_ACTION_CUE.search(s) for s in parts[1:])


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _discover_error_classes() -> dict[str, type[BaseException]]:
    """Every exception class effGen defines, keyed by its qualified name."""
    found: dict[str, type[BaseException]] = {}
    for module in pkgutil.walk_packages([str(PACKAGE_ROOT)], prefix="effgen."):
        if module.name in _UNIMPORTABLE:
            continue
        try:
            imported = importlib.import_module(module.name)
        except Exception:  # noqa: BLE001 - an optional dependency is absent
            continue
        for obj in vars(imported).values():
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseException)
                and getattr(obj, "__module__", "").startswith("effgen.")
                and obj.__module__ not in _EXEMPT_MODULES
                and not obj.__qualname__.startswith("_")
            ):
                found.setdefault(f"{obj.__module__}.{obj.__qualname__}", obj)
    return found


ERROR_CLASSES = _discover_error_classes()

# The forwarding classes are named, so a typo in the set is a failure rather
# than a silent exemption.
assert _FORWARDS_ITS_MESSAGE <= set(ERROR_CLASSES), sorted(
    _FORWARDS_ITS_MESSAGE - set(ERROR_CLASSES)
)


def _composes_its_message(cls: type[BaseException]) -> bool:
    """True when *cls* builds its message rather than forwarding what it is given.

    An ``__init__`` anywhere in the effGen part of the class's ancestry counts:
    a subclass that adds only a name still gets its parent's composed message.
    """
    return any(
        "__init__" in ancestor.__dict__
        and getattr(ancestor, "__module__", "").startswith("effgen.")
        for ancestor in cls.__mro__
    )


#: Classes that build their own message, and so are checked by construction.
COMPOSING = {
    n: c for n, c in ERROR_CLASSES.items()
    if _composes_its_message(c) and n not in _FORWARDS_ITS_MESSAGE
}

#: Classes whose message comes from the ``raise`` statement instead.
PASS_THROUGH = {
    n: c for n, c in ERROR_CLASSES.items()
    if not _composes_its_message(c) and n not in _FORWARDS_ITS_MESSAGE
}


def _synthesize(name: str, param: inspect.Parameter, upstream: str) -> object:
    """A stand-in argument for *param*, matching what an adapter would pass.

    *upstream* is the text used for every parameter that carries something
    produced elsewhere. Passing the neutral body also empties the optional
    collections, so a message whose guidance is conditional on a populated
    ``suggestions`` list is checked on the branch that has none.
    """
    minimal = upstream is _NEUTRAL_BODY
    annotation = str(param.annotation)
    if name in _UPSTREAM_PARAMS:
        return upstream
    if "list" in annotation or "Sequence" in annotation:
        if "tuple" in annotation and "Exception" in annotation:
            return [("openai", "gpt-5-nano", RuntimeError(upstream))] * 3
        if "str" in annotation and not minimal:
            return ["alpha", "beta"]
        if name == "providers":
            # Read positionally by the message it builds; never empty.
            return ["groq", "together"]
        return []
    if "dict" in annotation or "Mapping" in annotation:
        return {}
    if "tuple" in annotation:
        return ("openai", "gpt-5-nano")
    if "Exception" in annotation or "BaseException" in annotation:
        return RuntimeError(upstream)
    if "bool" in annotation:
        return True
    if "float" in annotation:
        return 1.5
    if "int" in annotation and "str" not in annotation:
        return 3
    if name in ("provider", "provider_name"):
        return "openai"
    if name in ("model", "model_name", "model_id"):
        return "gpt-5-nano"
    if name == "providers":
        return ["groq", "together"]
    return "value"


def _build(cls: type[BaseException], upstream: str = _UPSTREAM_BODY) -> BaseException:
    """Construct *cls* the way a caller carrying upstream text would."""
    signature = inspect.signature(cls.__init__)
    params = [
        (n, p)
        for n, p in signature.parameters.items()
        if n != "self"
        and p.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if not params:
        return cls(upstream)
    return cls(**{n: _synthesize(n, p, upstream) for n, p in params})


# ---------------------------------------------------------------------------
# Group A — classes that compose their own message
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("qualname", sorted(COMPOSING))
def test_composed_message_is_actionable(qualname: str) -> None:
    message = str(_build(COMPOSING[qualname], _NEUTRAL_BODY))
    assert is_actionable(message), (
        f"{qualname} reports a symptom with nothing to do about it:\n"
        f"  {message[:400]}"
    )


@pytest.mark.parametrize("qualname", sorted(COMPOSING))
def test_composed_message_bounds_upstream_text(qualname: str) -> None:
    message = str(_build(COMPOSING[qualname]))
    assert len(message) <= _MESSAGE_LIMIT, (
        f"{qualname} quotes {len(message)} characters of upstream text, over "
        f"the {_MESSAGE_LIMIT}-character limit. Abbreviate what it echoes."
    )


@pytest.mark.parametrize("qualname", sorted(COMPOSING))
def test_composed_message_redacts_credentials(qualname: str) -> None:
    message = str(_build(COMPOSING[qualname]))
    assert _PLANTED_KEY not in message, (
        f"{qualname} echoes a credential an upstream sent back."
    )


#: A doubled terminator, or one terminator butted against another. Produced by
#: appending guidance to text that already punctuated itself.
_DOUBLED_TERMINATOR = re.compile(r"\.\.(?!\.)|[!?]\.")

#: Upstream bodies that do and do not punctuate themselves. Providers vary, and
#: the join has to read correctly either way.
_TERMINATION_CASES = (
    "the upstream said no",
    "the upstream said no.",
    "the upstream said no!",
    "the upstream said no:",
)


@pytest.mark.parametrize("qualname", sorted(COMPOSING))
def test_composed_message_joins_its_guidance_cleanly(qualname: str) -> None:
    """Guidance is one sentence on from the symptom, however upstream ended it."""
    for upstream in _TERMINATION_CASES:
        message = str(_build(COMPOSING[qualname], upstream))
        found = _DOUBLED_TERMINATOR.search(message)
        assert not found, (
            f"{qualname} doubles a sentence terminator at {found.start()} when "
            f"the upstream text ends {upstream[-1]!r}:\n  {message[:400]}"
        )


# ---------------------------------------------------------------------------
# Group B — messages written at the ``raise`` statement
# ---------------------------------------------------------------------------


def _template(node: ast.AST | None) -> str | None:
    """Render a message expression as a template, or ``None`` if it cannot be."""
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else "{}"
    if isinstance(node, ast.JoinedStr):
        return "".join(
            v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "{}"
            for v in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _template(node.left), _template(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return _template(node.left)
    if isinstance(node, ast.IfExp):
        # Either branch may be the one the reader sees.
        body, orelse = _template(node.body), _template(node.orelse)
        return None if body is None or orelse is None else f"{body} {orelse}"
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "format":
            return _template(func.value)
        if isinstance(func, ast.Attribute) and func.attr == "join" and node.args:
            separator = _template(func.value)
            argument = node.args[0]
            if separator is not None and isinstance(argument, ast.List | ast.Tuple):
                parts = [_template(e) for e in argument.elts]
                if all(p is not None for p in parts):
                    return separator.join(parts)  # type: ignore[arg-type]
    return None


def _raise_sites() -> list[tuple[str, int, str, str | None]]:
    """Every ``raise`` of a pass-through effGen error, with its message."""
    names = {cls.__name__ for cls in PASS_THROUGH.values()}
    sites: list[tuple[str, int, str, str | None]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - the package parses
            continue
        relative = str(path.relative_to(PACKAGE_ROOT.parent))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)):
                continue
            func = node.exc.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name not in names:
                continue
            message = _template(node.exc.args[0]) if node.exc.args else None
            sites.append((relative, node.lineno, name, message))
    return sites


RAISE_SITES = _raise_sites()


def test_pass_through_raise_sites_were_found() -> None:
    """The source scan reaches the raise statements it is meant to read."""
    assert len(RAISE_SITES) >= 50, (
        f"only {len(RAISE_SITES)} raise sites found; the scan is not reaching "
        "the source"
    )


def test_pass_through_messages_are_actionable() -> None:
    unactionable = [
        (path, line, name, message)
        for path, line, name, message in RAISE_SITES
        if not (message and is_actionable(message))
    ]
    report = "\n".join(
        f"  {p}:{ln}  {n}  {(m or '<not a literal message>')[:120]}"
        for p, ln, n, m in unactionable
    )
    assert not unactionable, (
        f"{len(unactionable)} failure messages report a symptom with nothing "
        f"to do about it:\n{report}"
    )


# ---------------------------------------------------------------------------
# The tool result surface
# ---------------------------------------------------------------------------
#
# A tool reports its failure in ``ToolResult.error`` rather than by raising, so
# neither group above reaches it — and a rejected call is the failure a caller
# meets most often. The registry is walked and every tool is called with no
# arguments at all, which is the shape a caller reaches by omitting one.


def _tool_refusals() -> list[tuple[str, str]]:
    """Return ``(tool name, refusal)`` for every tool that rejects an empty call."""
    import asyncio

    from effgen.tools import get_registry

    registry = get_registry()
    refusals: list[tuple[str, str]] = []
    for name in sorted(registry.list_tools()):
        try:
            tool = registry.get_tool_sync(name)
            result = asyncio.run(asyncio.wait_for(tool.execute(), timeout=20))
        except Exception:  # noqa: BLE001 - a tool needing an absent SDK is not this gate's subject
            continue
        if not result.success and result.error:
            refusals.append((name, result.error))
    return refusals


def test_a_tool_rejecting_an_empty_call_says_what_to_pass() -> None:
    """Every tool refusal names the symptom and then what the call should pass."""
    refusals = _tool_refusals()
    assert len(refusals) >= 40, (
        f"only {len(refusals)} tools refused an empty call; the registry walk is "
        "not reaching the built-in tools"
    )
    unactionable = [(name, text) for name, text in refusals if not is_actionable(text)]
    report = "\n".join(f"  {name}  {text[:140]}" for name, text in unactionable)
    assert not unactionable, (
        f"{len(unactionable)} tool refusals report a symptom with nothing to do "
        f"about it:\n{report}"
    )


def test_a_tool_refusal_names_the_accepted_values_of_an_enum() -> None:
    """A rejected selector says which values the tool accepts."""
    import asyncio

    from effgen.tools import get_registry

    tool = get_registry().get_tool_sync("datetime")
    result = asyncio.run(tool.execute())
    assert not result.success
    assert "Accepted values for 'operation'" in result.error, result.error
    accepted = next(
        (p.enum for p in tool.metadata.parameters if p.name == "operation" and p.enum),
        None,
    )
    assert accepted, "the datetime tool no longer declares an operation enum"
    for value in accepted[:3]:
        assert str(value) in result.error, f"{value} missing from {result.error}"

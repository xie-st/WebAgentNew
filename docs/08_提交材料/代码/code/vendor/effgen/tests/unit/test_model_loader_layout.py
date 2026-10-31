"""Layout of the ``effgen.models.model_loader`` split.

The loader is one facade plus four concerns:

* ``model_loader_cloud`` — the deferred provider-adapter imports and the methods
  that construct the OpenAI, Anthropic and Gemini adapters;
* ``model_loader_local`` — the torch-availability helpers and the methods that
  construct the Transformers, vLLM, GGUF and MLX engines;
* ``model_loader_capacity`` — free-VRAM inspection and the automatic
  quantization and tensor-parallel choices;
* ``model_loader_routing`` — resolving a model id and its optional
  ``provider:``/``engine:`` prefix to the thing that serves it;
* ``model_loader`` — the class, its declared model tables, validation, the
  unload paths and the ``load_model()`` convenience function.

Five invariants keep that split safe:

* every name has exactly one owning module;
* the facade stays the single import path: every name it published is still
  importable from it and is the same object;
* imports go one way only, so the graph stays acyclic;
* the facade's *module scope* pulls no provider adapter, no local engine and no
  torch. Every one of those is imported inside the function that constructs it.
  Hoisting one to module scope reintroduces the cycle
  ``anthropic_adapter -> core.messages -> core.agent -> model_loader`` and makes
  ``from effgen import Agent`` pull torch for a pure cloud-API workflow;
* the six adapter memo globals stay in the module that rebinds them. They are
  ``None`` until their getter first runs, so re-exporting one by value would
  freeze the copy at ``None`` while the real memo moved on — with nothing
  raised. Only the getter *functions* are re-exported.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

import effgen.models.model_loader as facade
from effgen.models.model_loader import ModelLoader

PACKAGE = "effgen.models"
FACADE = "model_loader"

#: Every module the split is made of, the facade last.
SPLIT_MODULES = (
    "model_loader_cloud",
    "model_loader_local",
    "model_loader_capacity",
    "model_loader_routing",
    FACADE,
)

#: Which module owns each loader member.
OWNERS = {
    # routing
    "load_model": "model_loader_routing",
    "_detect_model_type": "model_loader_routing",
    # cloud adapters
    "_load_openai_model": "model_loader_cloud",
    "_load_anthropic_model": "model_loader_cloud",
    "_load_gemini_model": "model_loader_cloud",
    # local engines
    "_load_huggingface_model": "model_loader_local",
    "_vllm_usable": "model_loader_local",
    "_load_with_vllm": "model_loader_local",
    "_load_with_mlx": "model_loader_local",
    "_load_with_mlx_vlm": "model_loader_local",
    "_load_with_transformers": "model_loader_local",
    # capacity
    "_free_vram_gb": "model_loader_capacity",
    "_auto_select_quantization": "model_loader_capacity",
    "_auto_select_quantization_bits": "model_loader_capacity",
    "_auto_select_tensor_parallel": "model_loader_capacity",
    # the facade keeps the declared tables, validation and teardown
    "OPENAI_MODELS": FACADE,
    "ANTHROPIC_MODELS": FACADE,
    "GEMINI_MODELS": FACADE,
    "_LOCAL_ENGINE_PREFIXES": FACADE,
    "__init__": FACADE,
    "_validate_model": FACADE,
    "unload_model": FACADE,
    "unload_all": FACADE,
    "get_loaded_models": FACADE,
    "get_model_info": FACADE,
}

#: The tables are the loader's declared configuration and are read from outside
#: the class, so they stay attributes of the class itself rather than inherited.
DECLARED_TABLES = ("OPENAI_MODELS", "ANTHROPIC_MODELS", "GEMINI_MODELS", "_LOCAL_ENGINE_PREFIXES")

#: The deferred cloud-adapter getters. These are re-exported from the facade.
GETTERS = (
    "_get_hf_inference_adapter",
    "_get_replicate_adapter",
    "_get_cerebras_adapter",
    "_get_groq_adapter",
    "_get_together_adapter",
    "_get_fireworks_adapter",
)

#: The memo each getter rebinds. These are *not* re-exported: a copy taken at
#: import time would stay ``None`` forever.
MEMO_GLOBALS = (
    "_HFInferenceAdapter",
    "_ReplicateAdapter",
    "_CerebrasAdapter",
    "_GroqAdapter",
    "_TogetherAdapter",
    "_FireworksAdapter",
)

#: Module-level helpers the local module owns and the facade re-exports.
LOCAL_HELPERS = ("_require_torch", "_missing_module")

#: The facade defines the loader class, the convenience function and its logger.
FACADE_DEFINITIONS = {"ModelLoader", "load_model", "logger"}

#: What each module may import from within the split.
ALLOWED_IMPORTS: dict[str, set[str]] = {
    "model_loader_cloud": set(),
    "model_loader_local": set(),
    "model_loader_capacity": set(),
    "model_loader_routing": {"model_loader_cloud"},
    FACADE: set(SPLIT_MODULES) - {FACADE},
}

#: Nothing in this list may be imported at the facade's module scope.
FORBIDDEN_AT_MODULE_SCOPE = (
    "torch",
    "transformers",
    "vllm",
    "mlx",
    "openai",
    "anthropic",
    "google",
    "groq",
    "cerebras",
    "together",
    "fireworks",
    "replicate",
)

#: Adapter and engine modules of this package that must not be reached at the
#: facade's module scope either.
FORBIDDEN_SIBLING_SUFFIXES = ("_adapter", "_engine")

#: Everything a caller can reach on the loader class.
LOADER_MEMBERS = (
    "ANTHROPIC_MODELS",
    "GEMINI_MODELS",
    "OPENAI_MODELS",
    "_LOCAL_ENGINE_PREFIXES",
    "_auto_select_quantization",
    "_auto_select_quantization_bits",
    "_auto_select_tensor_parallel",
    "_detect_model_type",
    "_free_vram_gb",
    "_load_anthropic_model",
    "_load_gemini_model",
    "_load_huggingface_model",
    "_load_openai_model",
    "_load_openai_compatible_model",
    "_load_with_mlx",
    "_load_with_mlx_vlm",
    "_load_with_transformers",
    "_load_with_vllm",
    "_validate_model",
    "_vllm_usable",
    "get_loaded_models",
    "get_model_info",
    "load_model",
    "unload_all",
    "unload_model",
)

#: No module in the split may grow past this.
MAX_LINES = 500


def module(name: str):
    return importlib.import_module(f"{PACKAGE}.{name}")


def source_path(name: str) -> Path:
    return Path(inspect.getsourcefile(module(name)))


def tree_of(name: str) -> ast.Module:
    return ast.parse(source_path(name).read_text(encoding="utf-8"))


def top_level_names(name: str) -> set[str]:
    """Every name the module binds at module scope, imports excluded."""
    names: set[str] = set()
    for node in tree_of(name).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def defined_members(name: str) -> set[str]:
    """Every loader member the module defines in a class body."""
    found: set[str] = set()
    for node in ast.walk(tree_of(name)):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    found.add(item.name)
                elif isinstance(item, ast.Assign):
                    found.update(t.id for t in item.targets if isinstance(t, ast.Name))
    return found


def _import_module_argument(node: ast.Call) -> str | None:
    """The literal module name a dynamic import call names, if it is one."""
    func = node.func
    dynamic = (isinstance(func, ast.Attribute) and func.attr == "import_module") or (
        isinstance(func, ast.Name) and func.id in ("import_module", "__import__")
    )
    if not dynamic or not node.args:
        return None
    first = node.args[0]
    return first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) else None


def imported_split_modules(name: str) -> set[str]:
    """The modules of this split that *name* imports, however it spells them.

    All four spellings count: ``from .model_loader_x import y``,
    ``from effgen.models import model_loader_x`` (which names the module in the
    alias rather than in the module path), ``import effgen.models.model_loader_x``
    and a dynamic ``importlib.import_module("effgen.models.model_loader_x")``.
    """
    found: set[str] = set()

    def note(dotted: str) -> None:
        tail = dotted.split(".")[-1]
        if tail in SPLIT_MODULES:
            found.add(tail)

    for node in ast.walk(tree_of(name)):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                note(node.module)
            for alias in node.names:
                if alias.name in SPLIT_MODULES:
                    found.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                note(alias.name)
        elif isinstance(node, ast.Call):
            dotted = _import_module_argument(node)
            if dotted:
                note(dotted)
    return found


def module_scope_import_targets(name: str) -> set[str]:
    """Every module named by an import at module scope, dotted path included.

    A ``TYPE_CHECKING`` block is excluded: it never executes, which is the whole
    reason the deferred names are annotated from inside one.
    """
    targets: set[str] = set()
    for node in tree_of(name).body:
        if isinstance(node, ast.If):
            continue  # TYPE_CHECKING
        for sub in ast.walk(node) if isinstance(node, ast.Try) else [node]:
            if isinstance(sub, ast.Import):
                targets.update(a.name for a in sub.names)
            elif isinstance(sub, ast.ImportFrom) and sub.module:
                targets.add(sub.module)
                targets.update(f"{sub.module}.{a.name}" for a in sub.names)
    return targets


# ---------------------------------------------------------------------------
# One owner per name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("member", "owner"), sorted(OWNERS.items()))
def test_each_member_is_defined_by_exactly_one_module(member, owner):
    assert member in defined_members(owner), f"{owner} does not define {member}"
    others = [m for m in SPLIT_MODULES if m != owner and member in defined_members(m)]
    assert not others, f"{member} is also defined in {others}"


def test_facade_defines_only_the_class_the_function_and_its_logger():
    assert top_level_names(FACADE) == FACADE_DEFINITIONS


def test_declared_tables_stay_on_the_class_itself():
    for name in DECLARED_TABLES:
        assert name in vars(ModelLoader), f"{name} is inherited rather than declared"


def test_loader_member_surface_is_unchanged():
    seen = {n for n in dir(ModelLoader) if not n.startswith("__")}
    assert seen == set(LOADER_MEMBERS)


# ---------------------------------------------------------------------------
# The facade stays the single import path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", GETTERS)
def test_facade_re_exports_every_adapter_getter_as_the_same_object(name):
    assert hasattr(facade, name), f"{name} no longer imports from the facade"
    assert getattr(facade, name) is getattr(module("model_loader_cloud"), name)


@pytest.mark.parametrize("name", LOCAL_HELPERS)
def test_facade_re_exports_the_local_helpers_as_the_same_object(name):
    assert getattr(facade, name) is getattr(module("model_loader_local"), name)


@pytest.mark.parametrize("name", MEMO_GLOBALS)
def test_memo_globals_are_not_re_exported_by_value(name):
    """A copy of one of these taken at import time would stay ``None`` forever.

    Each is ``None`` until its getter first runs, and the getter rebinds the
    global in the module that declares it. A ``from ... import`` in the facade
    would bind the ``None``, and nothing would raise as the two drifted apart.
    """
    cloud = module("model_loader_cloud")
    assert name in vars(cloud), f"{name} is not declared in model_loader_cloud"
    assert not hasattr(facade, name), (
        f"{name} is re-exported by value; the copy cannot follow the memo"
    )


def test_getter_memoizes_into_its_own_module():
    cloud = module("model_loader_cloud")
    first = cloud._get_cerebras_adapter()
    assert cloud._CerebrasAdapter is first
    assert cloud._get_cerebras_adapter() is first


def test_loader_class_is_the_same_object_everywhere():
    import effgen
    import effgen.models

    assert effgen.ModelLoader is ModelLoader
    assert effgen.models.ModelLoader is ModelLoader
    assert effgen.load_model is facade.load_model


# ---------------------------------------------------------------------------
# Imports go one way only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SPLIT_MODULES)
def test_split_modules_only_import_downwards(name):
    allowed = ALLOWED_IMPORTS[name]
    assert imported_split_modules(name) <= allowed, (
        f"{name} imports {sorted(imported_split_modules(name) - allowed)}"
    )


def test_no_sibling_imports_the_facade():
    for name in SPLIT_MODULES:
        if name == FACADE:
            continue
        assert FACADE not in imported_split_modules(name), f"{name} imports the facade"


# ---------------------------------------------------------------------------
# Nothing heavy at the facade's module scope
# ---------------------------------------------------------------------------


def test_facade_module_scope_pulls_no_adapter_engine_or_torch():
    package_dir = source_path(FACADE).parent
    for target in module_scope_import_targets(FACADE):
        root = target.split(".")[0]
        assert root not in FORBIDDEN_AT_MODULE_SCOPE, f"{target} is imported at module scope"
        tail = target.split(".")[-1]
        # Only a tail that is itself a module of this package can be an adapter or
        # an engine; a plain imported name such as ``_get_groq_adapter`` is a
        # function, and importing that is exactly how the deferral works.
        if (package_dir / f"{tail}.py").exists():
            assert not tail.endswith(FORBIDDEN_SIBLING_SUFFIXES), (
                f"{target} is imported at module scope"
            )


def test_importing_the_loader_pulls_no_heavy_dependency():
    """Proven in a clean interpreter, since this one has already imported torch."""
    code = (
        "import sys; import effgen.models.model_loader;"
        "heavy=[m for m in ('torch','transformers','vllm','openai','anthropic') "
        "if m in sys.modules]; print(','.join(heavy) or 'none')"
    )
    env = dict(os.environ, EFFGEN_NO_DOTENV="1")
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300, env=env
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "none", out.stdout


@pytest.mark.parametrize("name", SPLIT_MODULES)
def test_no_module_in_the_split_grows_past_the_cap(name):
    lines = len(source_path(name).read_text(encoding="utf-8").splitlines())
    assert lines <= MAX_LINES, f"{name} is {lines} lines"

"""Orchestrates prompt chains: sequential, conditional, iterative and parallel steps.

Execution patterns are tuned for small language models.
"""

from __future__ import annotations

import ast
import asyncio
import logging
import operator as _op
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Safe condition evaluator
#
# Chain conditions ("quality_score >= 0.8") can reference variables that hold
# prior step results — i.e. *model output*. Evaluating them with eval() turns a
# prompt-injected step result into code execution, and an empty-builtins sandbox
# is escapable via attribute traversal on literals
# (``().__class__.__base__.__subclasses__()``). Instead we parse the condition to
# an AST and walk a strict whitelist: comparisons, boolean/arith ops, literals,
# variable *names* (bound to values, never interpolated into evaluable text), and
# a fixed set of safe builtins. Attribute access and arbitrary calls are refused,
# so a value like ``"__import__('os')"`` is only ever compared as a data string.
# --------------------------------------------------------------------------- #
class _UnsafeCondition(ValueError):
    """Raised when a chain condition uses a construct outside the whitelist."""


_BINOPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
}
_UNARYOPS = {ast.UAdd: _op.pos, ast.USub: _op.neg, ast.Not: _op.not_}
_CMPOPS = {
    ast.Eq: _op.eq,
    ast.NotEq: _op.ne,
    ast.Lt: _op.lt,
    ast.LtE: _op.le,
    ast.Gt: _op.gt,
    ast.GtE: _op.ge,
    ast.Is: _op.is_,
    ast.IsNot: _op.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}
# Calls are limited to these side-effect-free builtins (by bare name only).
_SAFE_CALLS = {
    "len": len, "str": str, "int": int, "float": float, "bool": bool,
    "abs": abs, "min": min, "max": max, "round": round, "sum": sum,
    "any": any, "all": all, "sorted": sorted,
}
# A {var} reference binds the variable as a name; it is NOT string-substituted.
_BRACE_VAR_RE = re.compile(r"\{([A-Za-z_]\w*)\}")


def _eval_condition_node(node: ast.AST, names: dict[str, Any]) -> Any:
    """Recursively evaluate a whitelisted AST node (no eval/compile, ever)."""
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result: Any = True
            for value in node.values:
                result = _eval_condition_node(value, names)
                if not result:
                    return result
            return result
        result = False
        for value in node.values:
            result = _eval_condition_node(value, names)
            if result:
                return result
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_condition_node(node.operand, names))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](
            _eval_condition_node(node.left, names),
            _eval_condition_node(node.right, names),
        )
    if isinstance(node, ast.Compare):
        left = _eval_condition_node(node.left, names)
        for op, comparator in zip(node.ops, node.comparators):
            op_fn = _CMPOPS.get(type(op))
            if op_fn is None:
                raise _UnsafeCondition(f"comparison '{type(op).__name__}' not allowed")
            right = _eval_condition_node(comparator, names)
            if not op_fn(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        raise _UnsafeCondition(f"unknown variable '{node.id}'")
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        items = [_eval_condition_node(e, names) for e in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(items)
        if isinstance(node, ast.Set):
            return set(items)
        return items
    if isinstance(node, ast.Dict):
        return {
            _eval_condition_node(k, names): _eval_condition_node(v, names)
            for k, v in zip(node.keys, node.values)
        }
    if isinstance(node, ast.Subscript):
        container = _eval_condition_node(node.value, names)
        key = _eval_condition_node(node.slice, names)
        return container[key]
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_CALLS:
            raise _UnsafeCondition("only safe builtin calls are allowed")
        if node.keywords:
            raise _UnsafeCondition("keyword arguments are not allowed")
        args = [_eval_condition_node(a, names) for a in node.args]
        return _SAFE_CALLS[node.func.id](*args)
    # Anything else — Attribute, Lambda, comprehensions, walrus, f-strings, … —
    # is refused. This is what blocks ``().__class__`` style escapes.
    raise _UnsafeCondition(f"unsupported expression '{type(node).__name__}'")


def safe_eval_condition(expression: str, names: dict[str, Any]) -> bool:
    """Evaluate a chain condition string safely.

    ``{var}`` references are rewritten to the bare name ``var`` and resolved from
    ``names`` (values are bound, never interpolated into the evaluated text), so
    model-supplied values can only act as data, never as code.
    """
    expr = _BRACE_VAR_RE.sub(r"\1", expression)
    tree = ast.parse(expr, mode="eval")
    return bool(_eval_condition_node(tree.body, names))


class ChainType(Enum):
    """Types of prompt chains."""
    SEQUENTIAL = "sequential"
    CONDITIONAL = "conditional"
    ITERATIVE = "iterative"
    PARALLEL = "parallel"
    HYBRID = "hybrid"


class StepStatus(Enum):
    """Status of a chain step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ChainStep:
    """Represents a single step in a prompt chain."""

    name: str
    type: str = "prompt"  # prompt, tool, function, parallel_group
    prompt: str | None = None
    tool: str | None = None
    function: Callable | None = None
    output_var: str | None = None
    condition: str | None = None
    max_retries: int = 0
    retry_delay: float = 0.0
    timeout: float | None = None
    dependencies: list[str] = field(default_factory=list)
    parallel_steps: list['ChainStep'] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Runtime state
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: str | None = None
    retries: int = 0

    def reset(self) -> None:
        """Reset step to initial state."""
        self.status = StepStatus.PENDING
        self.result = None
        self.error = None
        self.retries = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'type': self.type,
            'prompt': self.prompt,
            'tool': self.tool,
            'output_var': self.output_var,
            'condition': self.condition,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'timeout': self.timeout,
            'dependencies': self.dependencies,
            'metadata': self.metadata,
            'status': self.status.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ChainStep':
        """Create from dictionary."""
        # Remove runtime state if present
        data = {k: v for k, v in data.items() if k not in ['status', 'result', 'error', 'retries']}

        # Handle nested parallel steps
        if 'parallel_steps' in data and data['parallel_steps']:
            data['parallel_steps'] = [
                cls.from_dict(step) if isinstance(step, dict) else step
                for step in data['parallel_steps']
            ]

        return cls(**data)


@dataclass
class ChainState:
    """Manages state during chain execution."""

    variables: dict[str, Any] = field(default_factory=dict)
    step_results: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    iteration_count: int = 0

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the chain state."""
        self.variables[name] = value
        logger.debug(f"Set variable '{name}': {str(value)[:100]}")

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a variable from the chain state."""
        return self.variables.get(name, default)

    def set_step_result(self, step_name: str, result: Any) -> None:
        """Store result from a step."""
        self.step_results[step_name] = result
        logger.debug(f"Stored result for step '{step_name}'")

    def get_step_result(self, step_name: str, default: Any = None) -> Any:
        """Get result from a previous step."""
        return self.step_results.get(step_name, default)

    def interpolate_string(self, text: str) -> str:
        """Interpolate variables in a string using {variable} syntax."""
        if not text:
            return text

        # Find all variables in the text
        pattern = r'\{([^}]+)\}'
        matches = re.findall(pattern, text)

        result = text
        for match in matches:
            # Check if variable exists
            if match in self.variables:
                value = self.variables[match]
                # Convert to string if needed
                value_str = str(value) if value is not None else ""
                result = result.replace(f"{{{match}}}", value_str)
            else:
                logger.warning(f"Variable '{match}' not found in state")

        return result

    def evaluate_condition(self, condition: str) -> bool:
        """
        Evaluate a condition expression.

        Supports simple comparisons and variable checks
        Examples:
            - "quality_score >= 0.8"
            - "result_type == 'success'"
            - "iteration_count < 3"
        """
        if not condition:
            return True

        names = {**self.variables, 'iteration_count': self.iteration_count}
        try:
            return safe_eval_condition(condition, names)
        except _UnsafeCondition as e:
            logger.error(f"Rejected unsafe condition '{condition}': {e}")
            return False
        except SyntaxError as e:
            logger.error(f"Could not parse condition '{condition}': {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to evaluate condition '{condition}': {e}")
            return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'variables': self.variables,
            'step_results': self.step_results,
            'metadata': self.metadata,
            'iteration_count': self.iteration_count,
        }


@dataclass
class PromptChain:
    """Represents a complete prompt chain."""

    name: str
    description: str | None = None
    chain_type: ChainType = ChainType.SEQUENTIAL
    steps: list[ChainStep] = field(default_factory=list)
    max_iterations: int = 10
    early_stopping_condition: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: ChainStep) -> None:
        """Add a step to the chain."""
        self.steps.append(step)

    def get_step(self, name: str) -> ChainStep | None:
        """Get a step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def reset(self) -> None:
        """Reset all steps to initial state."""
        for step in self.steps:
            step.reset()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'chain_type': self.chain_type.value,
            'steps': [step.to_dict() for step in self.steps],
            'max_iterations': self.max_iterations,
            'early_stopping_condition': self.early_stopping_condition,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'PromptChain':
        """Create from dictionary."""
        chain_type = ChainType(data.get('chain_type', 'sequential'))
        steps = [ChainStep.from_dict(s) for s in data.get('steps', [])]

        return cls(
            name=data['name'],
            description=data.get('description'),
            chain_type=chain_type,
            steps=steps,
            max_iterations=data.get('max_iterations', 10),
            early_stopping_condition=data.get('early_stopping_condition'),
            metadata=data.get('metadata', {}),
        )


class ChainManager:
    """
    Manages prompt chain execution with support for:
    - Sequential chains (linear flow)
    - Conditional chains (branching logic)
    - Iterative chains (loops with refinement)
    - Parallel chains (concurrent execution)
    - Hybrid chains (combination of above)
    """

    def __init__(self, executor: Callable | None = None) -> None:
        """
        Initialize chain manager

        Args:
            executor: Function to execute prompts/tools (receives prompt, returns result)
        """
        self.chains: dict[str, PromptChain] = {}
        self.executor = executor or self._default_executor
        self.tool_registry: dict[str, Callable] = {}
        self.function_registry: dict[str, Callable] = {}

        logger.info("ChainManager initialized")

    def _default_executor(self, prompt: str, **kwargs) -> str:
        """Default executor (just returns the prompt)"""
        logger.warning("Using default executor - no actual execution")
        return f"[SIMULATED] {prompt}"

    def set_executor(self, executor: Callable) -> None:
        """Set the prompt executor function."""
        self.executor = executor
        logger.info("Executor function set")

    def register_tool(self, name: str, tool_func: Callable) -> None:
        """Register a tool function."""
        self.tool_registry[name] = tool_func
        logger.debug(f"Registered tool: {name}")

    def register_function(self, name: str, func: Callable) -> None:
        """Register a custom function."""
        self.function_registry[name] = func
        logger.debug(f"Registered function: {name}")

    def load_chain_from_yaml(self, filepath: str | Path) -> PromptChain:
        """
        Load chain definition from YAML file.

        Args:
            filepath: Path to YAML file

        Returns:
            Loaded PromptChain
        """
        filepath = Path(filepath)

        try:
            with open(filepath) as f:
                data = yaml.safe_load(f)

            chain = PromptChain.from_dict(data)
            self.add_chain(chain)

            logger.info(f"Loaded chain '{chain.name}' from {filepath}")
            return chain

        except Exception as e:
            logger.error(f"Failed to load chain from {filepath}: {e}")
            raise

    def add_chain(self, chain: PromptChain) -> None:
        """Add a chain to the manager."""
        self.chains[chain.name] = chain
        logger.debug(f"Added chain: {chain.name}")

    def get_chain(self, name: str) -> PromptChain | None:
        """Get a chain by name."""
        return self.chains.get(name)

    async def execute_chain(
        self,
        chain: str | PromptChain,
        initial_state: dict[str, Any] | None = None,
        callbacks: dict[str, Callable] | None = None
    ) -> ChainState:
        """
        Execute a prompt chain.

        Args:
            chain: Chain name or PromptChain object
            initial_state: Initial variables for the chain
            callbacks: Optional callbacks for events (on_step_start, on_step_complete, etc.)

        Returns:
            Final ChainState
        """
        # Get chain object
        if isinstance(chain, str):
            chain_obj = self.get_chain(chain)
            if chain_obj is None:
                raise ValueError(f"Chain not found: {chain}")
        else:
            chain_obj = chain

        # Initialize state
        state = ChainState(variables=initial_state or {})

        # Reset chain
        chain_obj.reset()

        # Execute based on chain type
        if chain_obj.chain_type == ChainType.SEQUENTIAL:
            await self._execute_sequential(chain_obj, state, callbacks)

        elif chain_obj.chain_type == ChainType.CONDITIONAL:
            await self._execute_conditional(chain_obj, state, callbacks)

        elif chain_obj.chain_type == ChainType.ITERATIVE:
            await self._execute_iterative(chain_obj, state, callbacks)

        elif chain_obj.chain_type == ChainType.PARALLEL:
            await self._execute_parallel(chain_obj, state, callbacks)

        elif chain_obj.chain_type == ChainType.HYBRID:
            await self._execute_hybrid(chain_obj, state, callbacks)

        else:
            raise ValueError(f"Unknown chain type: {chain_obj.chain_type}")

        logger.info(f"Chain '{chain_obj.name}' execution completed")
        return state

    async def _execute_sequential(
        self,
        chain: PromptChain,
        state: ChainState,
        callbacks: dict[str, Callable] | None = None
    ):
        """Execute steps sequentially"""
        for step in chain.steps:
            # Check early stopping condition
            if chain.early_stopping_condition:
                if state.evaluate_condition(chain.early_stopping_condition):
                    logger.info("Early stopping condition met")
                    break

            await self._execute_step(step, state, callbacks)

    async def _execute_conditional(
        self,
        chain: PromptChain,
        state: ChainState,
        callbacks: dict[str, Callable] | None = None
    ):
        """Execute steps with conditional branching"""
        for step in chain.steps:
            # Check step condition
            if step.condition:
                if not state.evaluate_condition(step.condition):
                    step.status = StepStatus.SKIPPED
                    logger.debug(f"Step '{step.name}' skipped (condition not met)")
                    continue

            await self._execute_step(step, state, callbacks)

    async def _execute_iterative(
        self,
        chain: PromptChain,
        state: ChainState,
        callbacks: dict[str, Callable] | None = None
    ):
        """Execute steps iteratively with refinement"""
        for iteration in range(chain.max_iterations):
            state.iteration_count = iteration + 1
            logger.info(f"Iteration {state.iteration_count}/{chain.max_iterations}")

            # Execute all steps
            for step in chain.steps:
                await self._execute_step(step, state, callbacks)

            # Check early stopping
            if chain.early_stopping_condition:
                if state.evaluate_condition(chain.early_stopping_condition):
                    logger.info(f"Early stopping at iteration {state.iteration_count}")
                    break

            # Reset steps for next iteration
            for step in chain.steps:
                step.reset()

    async def _execute_parallel(
        self,
        chain: PromptChain,
        state: ChainState,
        callbacks: dict[str, Callable] | None = None
    ):
        """Execute steps in parallel"""
        tasks = [
            self._execute_step(step, state, callbacks)
            for step in chain.steps
        ]

        await asyncio.gather(*tasks)

    async def _execute_hybrid(
        self,
        chain: PromptChain,
        state: ChainState,
        callbacks: dict[str, Callable] | None = None
    ):
        """Execute hybrid chain with mixed patterns"""
        for step in chain.steps:
            # Check if this is a parallel group
            if step.type == "parallel_group" and step.parallel_steps:
                # Execute parallel steps
                tasks = [
                    self._execute_step(substep, state, callbacks)
                    for substep in step.parallel_steps
                ]
                await asyncio.gather(*tasks)
            else:
                # Execute single step
                await self._execute_step(step, state, callbacks)

    async def _execute_step(
        self,
        step: ChainStep,
        state: ChainState,
        callbacks: dict[str, Callable] | None = None
    ):
        """Execute a single chain step"""
        callbacks = callbacks or {}

        # Callback: on_step_start
        if 'on_step_start' in callbacks:
            callbacks['on_step_start'](step, state)

        step.status = StepStatus.RUNNING
        logger.debug(f"Executing step: {step.name}")

        try:
            result = None

            # Execute based on step type
            if step.type == "prompt":
                result = await self._execute_prompt_step(step, state)

            elif step.type == "tool":
                result = await self._execute_tool_step(step, state)

            elif step.type == "function":
                result = await self._execute_function_step(step, state)

            else:
                raise ValueError(f"Unknown step type: {step.type}")

            # Store result
            step.result = result
            step.status = StepStatus.COMPLETED

            # Save to state
            if step.output_var:
                state.set_variable(step.output_var, result)
                state.set_step_result(step.name, result)

            # Callback: on_step_complete
            if 'on_step_complete' in callbacks:
                callbacks['on_step_complete'](step, state, result)

        except Exception as e:
            logger.error(f"Step '{step.name}' failed: {e}")
            step.error = str(e)
            step.status = StepStatus.FAILED

            # Retry logic
            if step.retries < step.max_retries:
                step.retries += 1
                logger.info(f"Retrying step '{step.name}' ({step.retries}/{step.max_retries})")

                if step.retry_delay > 0:
                    await asyncio.sleep(step.retry_delay)

                step.status = StepStatus.PENDING
                await self._execute_step(step, state, callbacks)
            else:
                # Callback: on_step_error
                if 'on_step_error' in callbacks:
                    callbacks['on_step_error'](step, state, e)
                raise

    async def _execute_prompt_step(self, step: ChainStep, state: ChainState) -> str:
        """Execute a prompt step"""
        if not step.prompt:
            raise ValueError(f"Step '{step.name}' has no prompt")

        # Interpolate variables in prompt
        prompt = state.interpolate_string(step.prompt)

        # Execute with timeout if specified
        if step.timeout:
            result = await asyncio.wait_for(
                self._run_executor(prompt),
                timeout=step.timeout
            )
        else:
            result = await self._run_executor(prompt)

        return result

    async def _execute_tool_step(self, step: ChainStep, state: ChainState) -> Any:
        """Execute a tool step"""
        if not step.tool:
            raise ValueError(f"Step '{step.name}' has no tool specified")

        tool_func = self.tool_registry.get(step.tool)
        if tool_func is None:
            raise ValueError(f"Tool not found: {step.tool}")

        # Prepare tool arguments from state
        tool_args = {}
        if step.prompt:
            tool_args['input'] = state.interpolate_string(step.prompt)

        # Execute tool
        result = tool_func(**tool_args)

        # Handle async tools
        if asyncio.iscoroutine(result):
            result = await result

        return result

    async def _execute_function_step(self, step: ChainStep, state: ChainState) -> Any:
        """Execute a custom function step"""
        if step.function:
            func = step.function
        elif step.metadata.get('function_name'):
            func_name = step.metadata['function_name']
            func = self.function_registry.get(func_name)
            if func is None:
                raise ValueError(f"Function not found: {func_name}")
        else:
            raise ValueError(f"Step '{step.name}' has no function")

        # Execute function with state
        result = func(state)

        # Handle async functions
        if asyncio.iscoroutine(result):
            result = await result

        return result

    async def _run_executor(self, prompt: str) -> str:
        """Run the executor, handling both sync and async"""
        result = self.executor(prompt)

        if asyncio.iscoroutine(result):
            result = await result

        return result

    def execute_chain_sync(
        self,
        chain: str | PromptChain,
        initial_state: dict[str, Any] | None = None,
        callbacks: dict[str, Callable] | None = None
    ) -> ChainState:
        """
        Synchronous wrapper for execute_chain.

        Args:
            chain: Chain name or PromptChain object
            initial_state: Initial variables
            callbacks: Optional callbacks

        Returns:
            Final ChainState
        """
        # Use the centralized bridge so this documented sync facade also works
        # from inside a running event loop (Jupyter/FastAPI) instead of raising
        # "asyncio.run() cannot be called from a running event loop" and leaking
        # a "coroutine was never awaited" warning.
        from ..utils.async_bridge import run_coroutine_sync

        return run_coroutine_sync(self.execute_chain(chain, initial_state, callbacks))

    def create_sequential_chain(
        self,
        name: str,
        prompts: list[str | tuple[str, str]],
        description: str | None = None
    ) -> PromptChain:
        """
        Create a simple sequential chain from a list of prompts.

        Args:
            name: Chain name
            prompts: List of prompts or (prompt, output_var) tuples
            description: Chain description

        Returns:
            Created PromptChain
        """
        steps = []

        for i, prompt_def in enumerate(prompts):
            if isinstance(prompt_def, tuple):
                prompt, output_var = prompt_def
            else:
                prompt = prompt_def
                output_var = f"step_{i}_result"

            step = ChainStep(
                name=f"step_{i}",
                type="prompt",
                prompt=prompt,
                output_var=output_var
            )
            steps.append(step)

        chain = PromptChain(
            name=name,
            description=description,
            chain_type=ChainType.SEQUENTIAL,
            steps=steps
        )

        self.add_chain(chain)
        return chain

    def save_chain(self, chain: PromptChain, filepath: str | Path) -> None:
        """
        Save chain to YAML file.

        Args:
            chain: Chain to save
            filepath: Output file path
        """
        filepath = Path(filepath)

        with open(filepath, 'w') as f:
            yaml.dump(chain.to_dict(), f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved chain '{chain.name}' to {filepath}")


def create_default_chain_manager(executor: Callable | None = None) -> ChainManager:
    """Create chain manager with default configuration."""
    manager = ChainManager(executor=executor)
    logger.info("Created default chain manager")
    return manager

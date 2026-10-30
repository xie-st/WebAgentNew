"""Prompt management, chaining and optimization tuned for small language models.

Designed for SLMs in the 1B-7B parameter range.
"""

from .agent_system_prompt import AgentSystemPromptBuilder
from .chain_manager import (
    ChainManager,
    ChainState,
    ChainStep,
    ChainType,
    PromptChain,
    StepStatus,
    create_default_chain_manager,
)
from .optimizer import (
    ModelSize,
    OptimizationConfig,
    OptimizationResult,
    PromptOptimizer,
    create_optimizer_for_model,
)
from .template_manager import (
    FewShotExample,
    PromptTemplate,
    TemplateManager,
    create_default_template_manager,
)
from .tool_prompt_generator import ToolPromptGenerator

__all__ = [
    # Template Manager
    'PromptTemplate',
    'FewShotExample',
    'TemplateManager',
    'create_default_template_manager',

    # Chain Manager
    'ChainType',
    'StepStatus',
    'ChainStep',
    'ChainState',
    'PromptChain',
    'ChainManager',
    'create_default_chain_manager',

    # Tool Prompt Generator
    'ToolPromptGenerator',

    # Agent System Prompt Builder
    'AgentSystemPromptBuilder',

    # Optimizer
    'ModelSize',
    'OptimizationConfig',
    'OptimizationResult',
    'PromptOptimizer',
    'create_optimizer_for_model',
]

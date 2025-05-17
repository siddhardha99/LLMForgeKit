"""LLMForgeKit: A toolkit for building LLM-powered applications."""

__version__ = "0.1.0"

# Core components
from llmforgekit.core.config import LLMForgeKitConfig, config
from llmforgekit.core.logging import get_logger, logger, setup_logging
from llmforgekit.core.errors import (
    AuthenticationError,
    ConfigError,
    LLMForgeKitError,
    LLMProviderError,
    ParserError,
    PluginError,
    PromptError,
    RateLimitError,
    ToolError,
    ValidationError,
    WorkflowError,
)
from llmforgekit.core.base import (
    Cache,
    LLMProvider,
    OutputParser,
    Plugin,
    PromptTemplate,
    Tool,
    Workflow,
    WorkflowStep,
)

# Services
from llmforgekit.services.llm import BaseLLMProvider, OpenAIProvider

__all__ = [
    # Core
    "LLMForgeKitConfig",
    "config",
    "logger",
    "get_logger",
    "setup_logging",
    
    # Errors
    "LLMForgeKitError",
    "ConfigError",
    "LLMProviderError",
    "PromptError",
    "ParserError",
    "WorkflowError",
    "ToolError",
    "PluginError",
    "ValidationError",
    "AuthenticationError",
    "RateLimitError",
    
    # Base classes
    "LLMProvider",
    "PromptTemplate",
    "OutputParser",
    "Tool",
    "WorkflowStep",
    "Workflow",
    "Plugin",
    "Cache",
    
    # LLM providers
    "BaseLLMProvider",
    "OpenAIProvider",
]
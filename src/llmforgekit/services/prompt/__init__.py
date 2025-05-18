"""Prompt management for LLMForgeKit."""

from llmforgekit.services.prompt.templates import StringPromptTemplate, JinjaPromptTemplate
from llmforgekit.services.prompt.library import PromptLibrary, PromptVersion
from llmforgekit.services.prompt.dynamic import (
    DynamicPromptGenerator,
    DynamicPromptTemplate,
    PromptComponent,
)

# Create a global prompt library instance
prompt_library = PromptLibrary()

__all__ = [
    "StringPromptTemplate",
    "JinjaPromptTemplate",
    "PromptLibrary",
    "PromptVersion",
    "DynamicPromptGenerator",
    "DynamicPromptTemplate",
    "PromptComponent",
    "prompt_library",
]
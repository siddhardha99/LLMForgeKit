"""Base classes and interfaces for LLMForgeKit.

This module defines the abstract base classes that form the foundation
of the LLMForgeKit architecture. These classes define the interfaces
that concrete implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class LLMProvider(ABC):
    """Base class for LLM provider implementations.
    
    This abstract class defines the interface that all LLM providers
    (like OpenAI, Anthropic, etc.) must implement.
    """
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send to the model
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            The generated text
            
        Raises:
            LLMProviderError: If the generation fails
            AuthenticationError: If authentication fails
            RateLimitError: If rate limiting occurs
        """
        pass
    
    @abstractmethod
    def generate_with_metadata(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a response with additional metadata.
        
        Args:
            prompt: The prompt to send to the model
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            A dictionary containing the generated text and metadata like:
            {
                "text": "The generated response",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                },
                "model": "gpt-3.5-turbo",
                ...
            }
            
        Raises:
            LLMProviderError: If the generation fails
            AuthenticationError: If authentication fails
            RateLimitError: If rate limiting occurs
        """
        pass


class PromptTemplate(ABC):
    """Base class for prompt templates.
    
    This abstract class defines the interface for prompt templates,
    which are used to generate prompts with variable substitution.
    """
    
    @abstractmethod
    def format(self, **kwargs: Any) -> str:
        """Format the template with provided variables.
        
        Args:
            **kwargs: Variables to insert into the template
            
        Returns:
            The formatted prompt
            
        Raises:
            PromptError: If formatting fails (e.g., missing variables)
        """
        pass
    
    @property
    @abstractmethod
    def variables(self) -> List[str]:
        """Get the list of variables in this template.
        
        Returns:
            A list of variable names used in this template
        """
        pass


class OutputParser(ABC):
    """Base class for output parsers.
    
    This abstract class defines the interface for output parsers,
    which convert raw LLM responses into structured data.
    """
    
    @abstractmethod
    def parse(self, output: str) -> Any:
        """Parse the LLM output.
        
        Args:
            output: The raw output from the LLM
            
        Returns:
            The parsed result (could be a dict, list, or custom object)
            
        Raises:
            ParserError: If parsing fails
        """
        pass


class Tool(ABC):
    """Base class for tools that can be used by the LLM.
    
    Tools are external functionalities that LLMs can use to perform
    actions like searching the web, accessing databases, etc.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the tool.
        
        Returns:
            The tool name
        """
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Get the description of the tool.
        
        Returns:
            A description of what the tool does and how to use it
        """
        pass
    
    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with the provided parameters.
        
        Args:
            **kwargs: Parameters for the tool
            
        Returns:
            The result of the tool execution
            
        Raises:
            ToolError: If the tool execution fails
        """
        pass


class WorkflowStep(ABC):
    """Base class for workflow steps.
    
    Workflow steps are individual operations in a multi-step process,
    each operating on a shared state.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the workflow step.
        
        Returns:
            The step name
        """
        pass
    
    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the workflow step.
        
        Args:
            state: The current workflow state
            
        Returns:
            The updated workflow state
            
        Raises:
            WorkflowError: If the step execution fails
        """
        pass


class Workflow(ABC):
    """Base class for workflows.
    
    Workflows are sequences of steps that can be executed together.
    """
    
    @abstractmethod
    def add_step(self, step: WorkflowStep) -> None:
        """Add a step to the workflow.
        
        Args:
            step: The workflow step to add
        """
        pass
    
    @abstractmethod
    def run(self, initial_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the entire workflow.
        
        Args:
            initial_state: Optional initial state. If not provided,
                           an empty state will be used.
            
        Returns:
            The final workflow state after all steps have been executed
            
        Raises:
            WorkflowError: If the workflow execution fails
        """
        pass


class Plugin(ABC):
    """Base class for plugins.
    
    Plugins are extensions that add functionality to the LLMForgeKit system.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the plugin.
        
        Returns:
            The plugin name
        """
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Get the version of the plugin.
        
        Returns:
            The plugin version
        """
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the plugin.
        
        Raises:
            PluginError: If initialization fails
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the plugin and free any resources."""
        pass


class Cache(ABC):
    """Base class for caching implementations.
    
    Caches can store and retrieve LLM responses to avoid duplicate requests.
    """
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value, or None if not found
        """
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: Optional time-to-live in seconds
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a value from the cache.
        
        Args:
            key: The cache key
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all values from the cache."""
        pass
"""Error handling for LLMForgeKit.

This module defines custom exception classes for different types of errors
that can occur within the LLMForgeKit library.
"""

from typing import Any, Dict, Optional


class LLMForgeKitError(Exception):
    """Base exception for all LLMForgeKit errors.
    
    All other exceptions in the library inherit from this class,
    making it easy to catch any error from the library.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception.
        
        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error details
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ConfigError(LLMForgeKitError):
    """Error related to configuration issues.
    
    Raised when there are problems with loading, parsing, or validating
    configuration values.
    """
    pass


class LLMProviderError(LLMForgeKitError):
    """Error related to LLM provider communication.
    
    Raised when there are issues communicating with LLM providers like
    OpenAI, Anthropic, etc. This could be due to API errors, rate limiting,
    or other provider-specific issues.
    """
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        response: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the exception.
        
        Args:
            message: Human-readable error message
            provider: Name of the LLM provider (e.g., "openai", "anthropic")
            status_code: HTTP status code if applicable
            response: Raw response from the provider if available
            details: Additional error details
        """
        self.provider = provider
        self.status_code = status_code
        self.response = response
        
        # Combine all details
        combined_details = details or {}
        if provider:
            combined_details["provider"] = provider
        if status_code:
            combined_details["status_code"] = status_code
        if response:
            combined_details["response"] = response
        
        super().__init__(message, combined_details)


class PromptError(LLMForgeKitError):
    """Error related to prompt processing.
    
    Raised when there are issues with prompt templates, such as
    missing variables or invalid template syntax.
    """
    pass


class ParserError(LLMForgeKitError):
    """Error related to output parsing.
    
    Raised when there are issues parsing or validating the output
    from an LLM provider.
    """
    
    def __init__(
        self,
        message: str,
        output: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the exception.
        
        Args:
            message: Human-readable error message
            output: The raw output that failed to parse
            details: Additional error details
        """
        self.output = output
        
        # Combine all details
        combined_details = details or {}

class WorkflowError(LLMForgeKitError):
    """Error related to workflow execution.
    
    Raised when there are issues with executing a workflow,
    such as step failures or invalid workflow definitions.
    """
    
    def __init__(
        self,
        message: str,
        step: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the exception.
        
        Args:
            message: Human-readable error message
            step: The name of the workflow step that failed
            details: Additional error details
        """
        self.step = step
        
        # Combine all details
        combined_details = details or {}
        if step:
            combined_details["step"] = step
        
        super().__init__(message, combined_details)


class ToolError(LLMForgeKitError):
    """Error related to tool execution.
    
    Raised when there are issues with executing external tools,
    such as API failures, permission issues, etc.
    """
    
    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the exception.
        
        Args:
            message: Human-readable error message
            tool_name: The name of the tool that failed
            details: Additional error details
        """
        self.tool_name = tool_name
        
        # Combine all details
        combined_details = details or {}
        if tool_name:
            combined_details["tool_name"] = tool_name
        
        super().__init__(message, combined_details)


class PluginError(LLMForgeKitError):
    """Error related to plugins.
    
    Raised when there are issues with loading, initializing,
    or using plugins.
    """
    
    def __init__(
        self,
        message: str,
        plugin_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the exception.
        
        Args:
            message: Human-readable error message
            plugin_name: The name of the plugin that caused the error
            details: Additional error details
        """
        self.plugin_name = plugin_name
        
        # Combine all details
        combined_details = details or {}
        if plugin_name:
            combined_details["plugin_name"] = plugin_name
        
        super().__init__(message, combined_details)


class ValidationError(LLMForgeKitError):
    """Error related to validation failures.
    
    Raised when inputs or outputs fail validation checks.
    """
    pass


class AuthenticationError(LLMForgeKitError):
    """Error related to authentication.
    
    Raised when there are issues with API keys or other
    authentication mechanisms.
    """
    pass


class RateLimitError(LLMProviderError):
    """Error related to rate limiting.
    
    Raised when an LLM provider rate limits requests.
    Inherits from LLMProviderError.
    """
    pass
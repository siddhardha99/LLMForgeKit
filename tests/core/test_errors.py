"""Tests for the error handling system."""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from llmforgekit.core.errors import (
    AuthenticationError,
    ConfigError,
    LLMForgeKitError,
    LLMProviderError,
    ParserError,
    PromptError,
    RateLimitError,
    ToolError,
    WorkflowError,
)


def test_base_error():
    """Test the base error class."""
    print("\nTesting base error:")
    
    try:
        raise LLMForgeKitError("Something went wrong")
    except LLMForgeKitError as e:
        print(f"Error message: {e}")
        print(f"Error details: {e.details}")


def test_error_with_details():
    """Test an error with additional details."""
    print("\nTesting error with details:")
    
    try:
        raise LLMForgeKitError(
            message="Something went wrong", 
            details={"reason": "Invalid input", "code": 42}
        )
    except LLMForgeKitError as e:
        print(f"Error message: {e}")
        print(f"Error details: {e.details}")


def test_provider_error():
    """Test a provider-specific error."""
    print("\nTesting provider error:")
    
    try:
        raise LLMProviderError(
            message="API request failed",
            provider="openai",
            status_code=429,
            response={"error": {"message": "Rate limit exceeded"}}
        )
    except LLMProviderError as e:
        print(f"Error message: {e}")
        print(f"Provider: {e.provider}")
        print(f"Status code: {e.status_code}")
        print(f"Response: {e.response}")


def test_parser_error():
    """Test a parser error."""
    print("\nTesting parser error:")
    
    try:
        raise ParserError(
            message="Failed to parse JSON response",
            output="{invalid: json}"
        )
    except ParserError as e:
        print(f"Error message: {e}")
        print(f"Failed output: {e.output}")


def test_error_hierarchy():
    """Test the error hierarchy."""
    print("\nTesting error hierarchy:")
    
    # Create some errors
    base_error = LLMForgeKitError("Base error")
    provider_error = LLMProviderError("Provider error")
    rate_limit_error = RateLimitError("Rate limit error")
    
    # Test inheritance
    print(f"base_error is LLMForgeKitError: {isinstance(base_error, LLMForgeKitError)}")
    print(f"provider_error is LLMForgeKitError: {isinstance(provider_error, LLMForgeKitError)}")
    print(f"rate_limit_error is LLMProviderError: {isinstance(rate_limit_error, LLMProviderError)}")
    print(f"rate_limit_error is LLMForgeKitError: {isinstance(rate_limit_error, LLMForgeKitError)}")


def test_error_handling():
    """Test practical error handling."""
    print("\nTesting error handling in practice:")
    
    def simulate_api_call(should_fail=False):
        if should_fail:
            raise LLMProviderError(
                message="API request failed",
                provider="openai",
                status_code=401,
                response={"error": {"message": "Invalid API key"}}
            )
        return "Success!"
    
    try:
        # Try a failing API call
        result = simulate_api_call(should_fail=True)
        print(f"Result: {result}")
    except LLMProviderError as e:
        print(f"Caught provider error: {e}")
        print(f"Provider: {e.provider}")
        print(f"Status code: {e.status_code}")
        
        # Check for specific error conditions
        if e.status_code == 401:
            print("Authentication failed. Please check your API key.")
        elif e.status_code == 429:
            print("Rate limit exceeded. Please try again later.")
    except LLMForgeKitError as e:
        print(f"Caught generic error: {e}")


if __name__ == "__main__":
    test_base_error()
    test_error_with_details()
    test_provider_error()
    test_parser_error()
    test_error_hierarchy()
    test_error_handling()
    print("\nAll error tests completed!")
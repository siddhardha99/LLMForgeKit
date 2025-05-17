"""OpenAI LLM provider implementation."""

import json
from typing import Any, Dict, List, Optional, Union

import requests

from llmforgekit.core.config import LLMForgeKitConfig
from llmforgekit.core.errors import (
    AuthenticationError,
    LLMProviderError,
    RateLimitError,
)
from llmforgekit.core.logging import get_logger
from llmforgekit.services.llm.base import BaseLLMProvider

logger = get_logger("services.llm.openai")


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider implementation.
    
    This provider allows you to use OpenAI's language models like GPT-3.5
    and GPT-4 through their API.
    """
    
    # API endpoints
    CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
    
    def __init__(
        self,
        config: Optional[LLMForgeKitConfig] = None,
        model: str = "gpt-3.5-turbo",
    ):
        """Initialize the OpenAI provider.
        
        Args:
            config: Configuration for the provider
            model: The model to use for generation
            
        Raises:
            AuthenticationError: If the API key is not found
        """
        super().__init__(config)
        self.model = model
        
        # Ensure we have an API key
        if not self.config.openai_api_key:
            raise AuthenticationError(
                "OpenAI API key not found. Please set it in your config or as an environment variable."
            )
        
        self.api_key = self.config.openai_api_key
        logger.debug(f"Initialized OpenAI provider with model: {model}")
    
    def _generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Generate text using the OpenAI API.
        
        Args:
            prompt: The prompt to send to the model
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            The generated text
            
        Raises:
            AuthenticationError: If authentication fails
            RateLimitError: If rate limiting occurs
            LLMProviderError: If the API request fails
        """
        response_data = self._make_api_request(prompt, max_tokens, temperature, **kwargs)
        
        # Extract the generated text from the response
        try:
            return response_data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            raise LLMProviderError(
                message=f"Failed to parse OpenAI response: {str(e)}",
                provider="openai",
                response=response_data,
            )
    
    def _generate_with_metadata(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate text with metadata using the OpenAI API.
        
        Args:
            prompt: The prompt to send to the model
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            A dictionary containing the generated text and metadata
            
        Raises:
            AuthenticationError: If authentication fails
            RateLimitError: If rate limiting occurs
            LLMProviderError: If the API request fails
        """
        response_data = self._make_api_request(prompt, max_tokens, temperature, **kwargs)
        
        # Extract the generated text and metadata from the response
        try:
            return {
                "text": response_data["choices"][0]["message"]["content"].strip(),
                "model": response_data["model"],
                "usage": response_data.get("usage", {}),
                "id": response_data.get("id"),
                "created": response_data.get("created"),
                "finish_reason": response_data["choices"][0].get("finish_reason"),
            }
        except (KeyError, IndexError) as e:
            raise LLMProviderError(
                message=f"Failed to parse OpenAI response: {str(e)}",
                provider="openai",
                response=response_data,
            )
    
    def _make_api_request(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Make a request to the OpenAI API.
        
        Args:
            prompt: The prompt to send to the model
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            The JSON response from the API
            
        Raises:
            AuthenticationError: If authentication fails
            RateLimitError: If rate limiting occurs
            LLMProviderError: If the API request fails
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        # Prepare the request payload
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        
        # Add max_tokens if specified
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in payload:
                payload[key] = value
        
        # Handle special case for 'messages' parameter
        if "messages" in kwargs:
            # If complete messages are provided, use those instead
            payload["messages"] = kwargs["messages"]
        
        logger.debug(f"Making OpenAI API request with model: {self.model}")
        
        try:
            # Make the API request
            response = requests.post(
                self.CHAT_COMPLETIONS_URL,
                headers=headers,
                json=payload,
                timeout=30,  # 30 second timeout
            )
            
            # Check for errors
            if response.status_code != 200:
                self._handle_error_response(response)
            
            # Parse the response
            return response.json()
            
        except requests.RequestException as e:
            raise LLMProviderError(
                message=f"Request to OpenAI API failed: {str(e)}",
                provider="openai",
            )
    
    def _handle_error_response(self, response: requests.Response) -> None:
        """Handle error responses from the OpenAI API.
        
        Args:
            response: The error response from the API
            
        Raises:
            AuthenticationError: If authentication fails (401)
            RateLimitError: If rate limiting occurs (429)
            LLMProviderError: For other API errors
        """
        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", "Unknown error")
        except Exception:
            error_message = f"API returned status code {response.status_code}"
        
        # Handle specific error codes
        if response.status_code == 401:
            raise AuthenticationError(
                message=f"OpenAI API authentication failed: {error_message}",
                details={"status_code": response.status_code},
            )
        elif response.status_code == 429:
            # Check for retry-after header
            retry_after = response.headers.get("retry-after")
            retry_seconds = int(retry_after) if retry_after and retry_after.isdigit() else None
            
            raise RateLimitError(
                message=f"OpenAI API rate limit exceeded: {error_message}",
                provider="openai",
                status_code=response.status_code,
                details={
                    "retry_after": retry_seconds,
                    "response": response.text,
                },
            )
        else:
            raise LLMProviderError(
                message=f"OpenAI API error: {error_message}",
                provider="openai",
                status_code=response.status_code,
                response=error_data if "error_data" in locals() else {"text": response.text},
            )
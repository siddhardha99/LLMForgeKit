"""Base implementation for LLM providers."""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from llmforgekit.core.base import LLMProvider
from llmforgekit.core.config import LLMForgeKitConfig, config
from llmforgekit.core.errors import LLMProviderError, RateLimitError
from llmforgekit.core.logging import get_logger

logger = get_logger("services.llm")


class BaseLLMProvider(LLMProvider):
    """Base implementation for all LLM providers.
    
    This class provides common functionality for all LLM providers,
    such as retry logic and error handling.
    """
    
    def __init__(self, config: Optional[LLMForgeKitConfig] = None):
        """Initialize the provider.
        
        Args:
            config: Configuration for the provider. If None, the global
                   config will be used.
        """
        self.config = config or config
        self.max_retries = self.config.max_retries
        self.retry_delay = self.config.retry_delay
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Generate a response from the LLM with retry logic.
        
        This method wraps the provider-specific implementation with
        retry logic to handle transient errors.
        
        Args:
            prompt: The prompt to send to the model
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            The generated text
            
        Raises:
            LLMProviderError: If the generation fails after all retries
        """
        retries = 0
        last_error = None
        
        while retries <= self.max_retries:
            try:
                return self._generate_text(prompt, max_tokens, temperature, **kwargs)
            except RateLimitError as e:
                # For rate limit errors, use the retry delay from the error if available
                last_error = e
                retries += 1
                
                if retries <= self.max_retries:
                    wait_time = getattr(e, "retry_after", self.retry_delay * (2 ** (retries - 1)))
                    logger.warning(
                        f"Rate limited by provider (attempt {retries}/{self.max_retries}). "
                        f"Retrying in {wait_time:.2f}s. Error: {str(e)}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Rate limited after {retries} attempts. Error: {str(e)}")
            except Exception as e:
                last_error = e
                retries += 1
                
                if retries <= self.max_retries:
                    wait_time = self.retry_delay * (2 ** (retries - 1))  # Exponential backoff
                    logger.warning(
                        f"LLM request failed (attempt {retries}/{self.max_retries}). "
                        f"Retrying in {wait_time:.2f}s. Error: {str(e)}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"LLM request failed after {retries} attempts. Error: {str(e)}")
        
        raise LLMProviderError(
            message=f"Failed to generate text after {retries} attempts: {str(last_error)}",
            details={"last_error": str(last_error)},
        )
    
    def generate_with_metadata(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a response with metadata, with retry logic.
        
        Args:
            prompt: The prompt to send to the model
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            A dictionary containing the generated text and metadata
            
        Raises:
            LLMProviderError: If the generation fails after all retries
        """
        retries = 0
        last_error = None
        
        while retries <= self.max_retries:
            try:
                return self._generate_with_metadata(prompt, max_tokens, temperature, **kwargs)
            except RateLimitError as e:
                # For rate limit errors, use the retry delay from the error if available
                last_error = e
                retries += 1
                
                if retries <= self.max_retries:
                    wait_time = getattr(e, "retry_after", self.retry_delay * (2 ** (retries - 1)))
                    logger.warning(
                        f"Rate limited by provider (attempt {retries}/{self.max_retries}). "
                        f"Retrying in {wait_time:.2f}s. Error: {str(e)}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Rate limited after {retries} attempts. Error: {str(e)}")
            except Exception as e:
                last_error = e
                retries += 1
                
                if retries <= self.max_retries:
                    wait_time = self.retry_delay * (2 ** (retries - 1))  # Exponential backoff
                    logger.warning(
                        f"LLM request failed (attempt {retries}/{self.max_retries}). "
                        f"Retrying in {wait_time:.2f}s. Error: {str(e)}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"LLM request failed after {retries} attempts. Error: {str(e)}")
        
        raise LLMProviderError(
            message=f"Failed to generate text after {retries} attempts: {str(last_error)}",
            details={"last_error": str(last_error)},
        )
    
    @abstractmethod
    def _generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Provider-specific implementation for text generation.
        
        This method must be implemented by subclasses.
        
        Args:
            prompt: The prompt to send to the model
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            The generated text
        """
        pass
    
    @abstractmethod
    def _generate_with_metadata(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Provider-specific implementation for generation with metadata.
        
        This method must be implemented by subclasses.
        
        Args:
            prompt: The prompt to send to the model
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            A dictionary containing the generated text and metadata
        """
        pass
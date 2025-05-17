"""Tests for the OpenAI provider."""

import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from llmforgekit.core.config import LLMForgeKitConfig
from llmforgekit.services.llm import OpenAIProvider
from llmforgekit.core.errors import AuthenticationError, LLMProviderError

# Create the tests directory structure if it doesn't exist
os.makedirs("tests/services", exist_ok=True)


def test_initialization():
    """Test initialization of the OpenAI provider."""
    print("\nTesting OpenAI provider initialization:")
    
    # Test initialization with valid API key
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("⚠️ No API key found in environment. Skipping API tests.")
        return
    
    try:
        config = LLMForgeKitConfig(openai_api_key=api_key)
        provider = OpenAIProvider(config=config, model="gpt-3.5-turbo")
        print(f"✅ Successfully initialized provider with model {provider.model}")
    except Exception as e:
        print(f"❌ Error initializing provider: {e}")
        return
    
    # Test initialization with invalid API key
    try:
        config = LLMForgeKitConfig(openai_api_key="invalid_key")
        provider = OpenAIProvider(config=config)
        print("✅ Provider initialized with invalid key (will fail on API calls)")
    except AuthenticationError as e:
        print(f"❌ Unexpected error with invalid key: {e}")
    
    # Test initialization with no API key
    try:
        config = LLMForgeKitConfig(openai_api_key=None)
        provider = OpenAIProvider(config=config)
        print("❌ Should have raised an error for missing API key")
    except AuthenticationError as e:
        print(f"✅ Correctly raised error: {e}")


def test_generation():
    """Test generation with the OpenAI provider."""
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("\n⚠️ No API key found in environment. Skipping generation tests.")
        return
    
    print("\nTesting OpenAI text generation:")
    
    # Initialize provider
    config = LLMForgeKitConfig(openai_api_key=api_key)
    provider = OpenAIProvider(config=config, model="gpt-3.5-turbo")
    
    # Test basic generation
    try:
        prompt = "Write a haiku about programming"
        print(f"Prompt: '{prompt}'")
        
        response = provider.generate(prompt, temperature=0.7)
        print(f"Response: '{response}'")
        print("✅ Successfully generated text")
    except Exception as e:
        print(f"❌ Error generating text: {e}")
    
    # Test generation with metadata
    try:
        prompt = "What is the capital of France?"
        print(f"\nPrompt with metadata: '{prompt}'")
        
        response = provider.generate_with_metadata(prompt, temperature=0.7)
        print(f"Generated text: '{response['text']}'")
        print(f"Model: {response['model']}")
        print(f"Usage: {response['usage']}")
        print("✅ Successfully generated text with metadata")
    except Exception as e:
        print(f"❌ Error generating text with metadata: {e}")
    
    # Test with invalid API key
    try:
        print("\nTesting with invalid API key:")
        bad_config = LLMForgeKitConfig(openai_api_key="invalid_key")
        bad_provider = OpenAIProvider(config=bad_config)
        
        bad_provider.generate("This should fail")
        print("❌ Should have raised an error for invalid API key")
    except (AuthenticationError, LLMProviderError) as e:
        print(f"✅ Correctly raised error: {e}")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("tests/services", exist_ok=True)
    
    test_initialization()
    test_generation()
    print("\nAll OpenAI provider tests completed!")
#!/usr/bin/env python3
"""Simple example using the OpenAI provider."""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llmforgekit.core.config import LLMForgeKitConfig
from llmforgekit.core.logging import setup_logging
from llmforgekit.services.llm import OpenAIProvider

# Set up logging
logger = setup_logging(log_level="INFO")


def main():
    """Run a simple example using the OpenAI provider."""
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY=your_api_key")
        return 1
    
    print("Initializing OpenAI provider...")
    
    # Create config
    config = LLMForgeKitConfig(openai_api_key=api_key)
    
    # Create provider
    provider = OpenAIProvider(config=config, model="gpt-3.5-turbo")
    
    # Generate text
    print("\nGenerating response...")
    
    try:
        # Basic generation
        prompt = "Write a short poem about programming in Python"
        response = provider.generate(prompt, temperature=0.7)
        
        print("\n=== Basic Generation ===")
        print(f"Prompt: {prompt}")
        print("\nResponse:")
        print(response)
        
        # Generation with metadata
        prompt = "Explain what an API is in one sentence"
        response_with_meta = provider.generate_with_metadata(prompt, temperature=0.7)
        
        print("\n=== Generation with Metadata ===")
        print(f"Prompt: {prompt}")
        print("\nResponse:")
        print(response_with_meta["text"])
        print("\nMetadata:")
        print(f"Model: {response_with_meta['model']}")
        print(f"Usage: {response_with_meta['usage']}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    print("\nExample completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
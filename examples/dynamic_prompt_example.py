#!/usr/bin/env python3
"""Example using the Dynamic Prompt Fabric with OpenAI."""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llmforgekit.core.config import LLMForgeKitConfig
from llmforgekit.core.logging import setup_logging
from llmforgekit.services.llm import OpenAIProvider
from llmforgekit.services.prompt import (
    DynamicPromptGenerator,
    PromptLibrary,
)


# Set up logging
logger = setup_logging(log_level="INFO")


def create_chat_prompt_generator():
    """Create a dynamic prompt generator for chat interactions."""
    generator = DynamicPromptGenerator(template_id="chat_agent")
    
    # Set system message
    generator.set_prefix(
        "You are an AI assistant named Claude that helps users with various tasks. "
        "Follow these guidelines when responding:"
    )
    
    # Add general instructions
    generator.add_component(
        "- Be helpful, friendly, and concise.\n"
        "- Stay on topic and answer the user's question directly.\n"
        "- If you don't know something, admit it rather than making up information.",
        name="general_guidelines",
        weight=1.0,  # Important, keep this unless absolutely necessary to remove
    )
    
    # Add persona-specific instructions
    generator.add_component(
        "- Respond in a casual, conversational tone.\n"
        "- Use humor when appropriate.\n"
        "- Feel free to use emojis occasionally.",
        name="casual_persona",
        conditions={"persona": "casual"},
        weight=0.7,
    )
    
    generator.add_component(
        "- Respond in a formal, professional tone.\n"
        "- Use precise, technical language.\n"
        "- Maintain a respectful, business-like manner.",
        name="professional_persona",
        conditions={"persona": "professional"},
        weight=0.7,
    )
    
    # Add topic-specific instructions
    generator.add_component(
        "- When answering programming questions, include code examples.\n"
        "- Explain code step by step.\n"
        "- Suggest best practices.",
        name="programming_topic",
        conditions={"topic": "programming"},
        weight=0.8,
    )
    
    generator.add_component(
        "- For science questions, cite current research when possible.\n"
        "- Explain complex concepts in simple terms.\n"
        "- Avoid technical jargon unless necessary.",
        name="science_topic",
        conditions={"topic": "science"},
        weight=0.8,
    )
    
    # Add user-specific customization
    generator.add_component(
        "The user's name is {user_name}. Address them by name occasionally.",
        name="user_personalization",
        conditions={"user_name": lambda name: name is not None},
        weight=0.5,
    )
    
    # Add the user's question
    generator.set_suffix(
        "User's question: {question}\n\n"
        "Respond to the question above following the guidelines."
    )
    
    return generator


def main():
    """Run the dynamic prompt fabric example."""
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY=your_api_key")
        return 1
    
    # Create config and LLM provider
    config = LLMForgeKitConfig(openai_api_key=api_key)
    llm = OpenAIProvider(config=config, model="gpt-3.5-turbo")
    
    # Create prompt library and add the dynamic prompt generator
    library = PromptLibrary()
    generator = create_chat_prompt_generator()
    library.add_template(
        name="chat_agent",
        template=generator.to_template(),
    )
    
    # Example 1: Casual programming question
    print("\n=== Example 1: Casual Programming Question ===")
    
    prompt1 = library.format_prompt(
        template_name="chat_agent",
        persona="casual",
        topic="programming",
        user_name="Alex",
        question="How do I create a for loop in Python?",
    )
    
    print("Dynamic Prompt:")
    print("--------------")
    print(prompt1)
    print("--------------")
    
    response1 = llm.generate(prompt1, temperature=0.7)
    
    print("\nLLM Response:")
    print(response1)
    
    # Example 2: Professional science question
    print("\n=== Example 2: Professional Science Question ===")
    
    prompt2 = library.format_prompt(
        template_name="chat_agent",
        persona="professional",
        topic="science",
        question="What is quantum entanglement?",
    )
    
    print("Dynamic Prompt:")
    print("--------------")
    print(prompt2)
    print("--------------")
    
    response2 = llm.generate(prompt2, temperature=0.7)
    
    print("\nLLM Response:")
    print(response2)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
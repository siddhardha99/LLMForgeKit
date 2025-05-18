"""Tests for the dynamic prompt generation system."""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from llmforgekit.services.prompt import (
    DynamicPromptGenerator,
    PromptComponent,
    PromptLibrary,
)


def test_dynamic_prompt_generator():
    """Test the DynamicPromptGenerator."""
    print("\nTesting DynamicPromptGenerator:")
    
    # Create a generator
    generator = DynamicPromptGenerator(template_id="chat_prompt")
    
    # Set prefix and suffix
    generator.set_prefix("You are an AI assistant that helps with various tasks.")
    generator.set_suffix("Please provide a helpful response.")
    
    # Add components
    generator.add_component(
        "When responding to questions about programming, include code examples.",
        name="programming_instruction",
        conditions={"topic": "programming"},
    )
    
    generator.add_component(
        "For questions about science, cite reputable sources when possible.",
        name="science_instruction",
        conditions={"topic": "science"},
    )
    
    generator.add_component(
        "The user's name is {user_name}. Be friendly and address them by name.",
        name="personalization",
        conditions={"user_name": lambda name: name is not None},
    )
    
    generator.add_component(
        "Keep responses concise and to the point.",
        name="conciseness",
        weight=0.5,  # Lower weight, will be removed first if truncation needed
    )
    
    # Generate prompts for different contexts
    programming_context = {
        "topic": "programming",
        "user_name": "Alice",
    }
    
    science_context = {
        "topic": "science",
    }
    
    general_context = {
        "user_name": "Bob",
    }
    
    # Generate and print prompts
    programming_prompt = generator.generate(programming_context)
    print("\nProgramming prompt:")
    print(programming_prompt)
    
    science_prompt = generator.generate(science_context)
    print("\nScience prompt:")
    print(science_prompt)
    
    general_prompt = generator.generate(general_context)
    print("\nGeneral prompt:")
    print(general_prompt)
    
    # Test truncation
    generator.set_max_length(100)
    truncated_prompt = generator.generate(programming_context)
    print("\nTruncated prompt (max 100 chars):")
    print(truncated_prompt)
    assert len(truncated_prompt) <= 100


def test_dynamic_prompt_with_library():
    """Test using DynamicPromptGenerator with PromptLibrary."""
    print("\nTesting DynamicPromptGenerator with PromptLibrary:")
    
    # Create a generator
    generator = DynamicPromptGenerator(template_id="product_description")
    
    # Set components
    generator.set_prefix("Product Description:")
    
    generator.add_component(
        "This is a premium product.",
        name="premium",
        conditions={"tier": "premium"},
    )
    
    generator.add_component(
        "This is a standard product.",
        name="standard",
        conditions={"tier": "standard"},
    )
    
    generator.add_component(
        "Features:\n- {feature1}\n- {feature2}",
        name="features",
    )
    
    generator.add_component(
        "Price: ${price}",
        name="price",
        conditions={"price": lambda p: p is not None},
    )
    
    # Convert to template and add to library
    template = generator.to_template()
    library = PromptLibrary()
    library.add_template(
        name="product_description",
        template=template,
    )
    
    # Use the template from the library
    premium_result = library.format_prompt(
        template_name="product_description",
        tier="premium",
        feature1="High quality",
        feature2="Durable",
        price=99.99,
    )
    
    print("\nPremium product description:")
    print(premium_result)
    
    standard_result = library.format_prompt(
        template_name="product_description",
        tier="standard",
        feature1="Good quality",
        feature2="Affordable",
        price=49.99,
    )
    
    print("\nStandard product description:")
    print(standard_result)


if __name__ == "__main__":
    test_dynamic_prompt_generator()
    test_dynamic_prompt_with_library()
    print("\nAll dynamic prompt tests completed!")
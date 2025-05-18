"""Tests for the prompt management system."""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from llmforgekit.core.errors import PromptError
from llmforgekit.services.prompt import (
    PromptLibrary,
    StringPromptTemplate,
    prompt_library,
)


def test_string_template():
    """Test the StringPromptTemplate."""
    print("\nTesting StringPromptTemplate:")
    
    # Create a template
    template = StringPromptTemplate("Hello, $name! Welcome to $service.")
    
    # Check variables
    print(f"Template variables: {template.variables}")
    assert "name" in template.variables
    assert "service" in template.variables
    
    # Format the template
    result = template.format(name="Alice", service="LLMForgeKit")
    print(f"Formatted result: {result}")
    assert result == "Hello, Alice! Welcome to LLMForgeKit."
    
    # Test missing variable
    try:
        template.format(name="Bob")
        print("❌ Should have raised an error for missing variable")
        assert False
    except PromptError as e:
        print(f"✅ Correctly raised error: {e}")


def test_prompt_library():
    """Test the PromptLibrary."""
    print("\nTesting PromptLibrary:")
    
    # Create a library
    library = PromptLibrary()
    
    # Add templates
    library.add_template(
        name="greeting",
        template="Hello, $name! Welcome to $service.",
        version="1.0.0",
    )
    
    library.add_template(
        name="greeting",
        template="Greetings, $name! Welcome to $service, powered by LLMForgeKit.",
        version="1.1.0",
    )
    
    library.add_template(
        name="help",
        template="Here's how to use $feature: $instructions",
    )
    
    # Check template listing
    templates = library.list_templates()
    print(f"Templates in library: {templates}")
    assert len(templates) == 2
    
    # Get a template by name
    template = library.get_template("greeting")
    print(f"Latest greeting template variables: {template.variables}")
    
    # Format a template - Using template_name instead of name
    result = library.format_prompt(
        template_name="greeting",
        name="Alice", 
        service="AI Assistant"
    )
    print(f"Formatted greeting: {result}")
    
    # Get a specific version - Using template_name instead of name
    result = library.format_prompt(
        template_name="greeting",
        version="1.0.0",
        name="Bob", 
        service="ChatBot"
    )
    print(f"Formatted greeting (v1.0.0): {result}")


if __name__ == "__main__":
    test_string_template()
    test_prompt_library()
    print("\nAll prompt tests completed!")
#!/usr/bin/env python3
"""Example using the Intelligent Semantic Aligner with OpenAI."""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("Warning: Pydantic is not installed, some examples will be skipped")

from llmforgekit.core.config import LLMForgeKitConfig
from llmforgekit.core.logging import setup_logging
from llmforgekit.services.llm import OpenAIProvider
from llmforgekit.services.prompt import StringPromptTemplate, PromptLibrary
from llmforgekit.services.parser import (
    JSONOutputParser,
    KeyValueParser,
    PydanticOutputParser,
    SemanticAligner,
)

# Set up logging
logger = setup_logging(log_level="INFO")


def create_template_library() -> PromptLibrary:
    """Create a prompt library with templates for different outputs."""
    library = PromptLibrary()
    
    # JSON output template
    json_template = """
    You are a helpful assistant that responds in JSON format.
    
    Please provide information about the following topic: {topic}
    
    Your response should be valid JSON with the following structure:
    {
        "topic": "The topic name",
        "summary": "A brief summary",
        "facts": ["Fact 1", "Fact 2", "Fact 3"],
        "source": "Your source of information"
    }
    
    Respond ONLY with the JSON object, nothing else.
    """
    library.add_template(
        name="json_response",
        template=json_template,
    )
    
    # Key-value output template
    kv_template = """
    You are a helpful assistant that responds in a structured format.
    
    Please provide information about the foll
    owing person: Ada Lovelace
    
    Format your response with EXACTLY these keys, each on a new line:
    Name: (the person's full name)
    Born: (birth date)
    Occupation: (primary occupation)
    Known For: (what they are famous for)
    
    Example format:
    Name: Albert Einstein
    Born: March 14, 1879
    Occupation: Theoretical Physicist
    Known For: Theory of Relativity
    
    Your response MUST follow this exact format with these exact keys, providing FACTUAL information about Ada Lovelace. Do NOT use placeholder text in parentheses.
    """
    library.add_template(
        name="person_info",
        template=kv_template,
    )
    
    # For Pydantic models
    if PYDANTIC_AVAILABLE:
        product_template = """
        You are a product information assistant.
        
        Please provide details about the following product: wireless noise-cancelling headphones
        
        Your response should include:
        - Product name (be specific and creative)
        - Description (detailed but concise)
        - Price (realistic market price)
        - Category (appropriate product category)
        - At least 3 features (highlight the most important ones)
        
        Format your response as valid JSON with the following structure:
        {
            "name": "Product name",
            "description": "Product description",
            "price": 99.99,
            "category": "Category name",
            "features": ["Feature 1", "Feature 2", "Feature 3"]
        }
        
        Respond ONLY with the JSON object, nothing else.
        """
        library.add_template(
            name="product_info",
            template=product_template,
        )
    
    # Return the library
    return library


def create_semantic_aligner() -> SemanticAligner:
    """Create a semantic aligner with various parsers."""
    aligner = SemanticAligner()
    
    # Register JSON parser
    aligner.register_parser(
        "json",
        JSONOutputParser(extract_json=True),
    )
    
    # Register key-value parser
    aligner.register_parser(
        "key_value",
        KeyValueParser(keys=["Name", "Born", "Occupation", "Known For"]),
    )
    
    # Register Pydantic parser if available
    if PYDANTIC_AVAILABLE:
        class Product(BaseModel):
            name: str
            description: str
            price: float
            category: str
            features: List[str]
        
        aligner.register_parser(
            "product",
            PydanticOutputParser(model=Product),
        )
    
    return aligner


def main():
    """Run the parser example."""
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY=your_api_key")
        return 1
    
    # Create config and LLM provider
    config = LLMForgeKitConfig(openai_api_key=api_key)
    llm = OpenAIProvider(config=config, model="gpt-3.5-turbo")
    
    # Create prompt library and aligner
    library = create_template_library()
    aligner = create_semantic_aligner()
    
    # Example 1: JSON output
    print("\n=== Example 1: JSON Output ===")
    
    # Create the prompt
    prompt = library.format_prompt(
        template_name="json_response",
        topic="quantum computing",
    )
    
    print("Sending prompt to generate JSON...")
    
    # Generate and parse
    try:
        result = aligner.generate_and_parse(
            llm=llm,
            prompt=prompt,
            parser_id="json",
            temperature=0.7,
        )
        
        print("\nSuccessfully parsed JSON:")
        print(f"Topic: {result['topic']}")
        print(f"Summary: {result['summary']}")
        print("Facts:")
        for i, fact in enumerate(result['facts'], 1):
            print(f"  {i}. {fact}")
        print(f"Source: {result['source']}")
    except Exception as e:
        print(f"Error parsing JSON: {e}")
    
    # Example 2: Key-value output
    print("\n=== Example 2: Key-Value Output ===")
    
    # Get the template directly (no variables to format)
    prompt = library.get_template("person_info")
    if prompt:
        prompt_text = prompt.format()
    else:
        # Fallback in case template is missing
        prompt_text = """
        Provide factual biographical information about Ada Lovelace in this exact format:
        
        Name: (full name)
        Born: (birth date)
        Occupation: (occupation)
        Known For: (achievements)
        
        Replace the parentheses with ACTUAL factual information about Ada Lovelace.
        """
    
    print("Sending prompt to generate key-value pairs about Ada Lovelace...")
    
    # Generate and parse with direct approach
    try:
        # Generate response directly
        raw_response = llm.generate(prompt_text, temperature=0.7)
        print("\nRaw response from LLM:")
        print(raw_response)
        
        # Parse the response
        result = aligner.parse(raw_response, "key_value")
        
        print("\nSuccessfully parsed key-value pairs:")
        for key, value in result.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error parsing key-value pairs: {e}")
    
    # Example 3: Pydantic model (if available)
    if PYDANTIC_AVAILABLE:
        print("\n=== Example 3: Pydantic Model ===")
        
        # Get the template directly (it has the product hardcoded now)
        prompt = library.get_template("product_info")
        if prompt:
            prompt_text = prompt.format()
        else:
            # Fallback in case template is missing
            prompt_text = """
            You are a product information assistant.
            
            Please provide details about wireless noise-cancelling headphones.
            
            Format your response as valid JSON with this structure:
            {
                "name": "Product name",
                "description": "Product description",
                "price": 99.99,
                "category": "Category name",
                "features": ["Feature 1", "Feature 2", "Feature 3"]
            }
            """
        
        print("Sending prompt to generate product info...")
        
        # Generate and parse
        try:
            # Generate response directly
            raw_response = llm.generate(prompt_text, temperature=0.7)
            print("\nRaw response from LLM:")
            print(raw_response)
            
            # Parse the response
            result = aligner.parse(raw_response, "product")
            
            print("\nSuccessfully parsed product info:")
            print(f"Name: {result.name}")
            print(f"Description: {result.description}")
            print(f"Price: ${result.price:.2f}")
            print(f"Category: {result.category}")
            print("Features:")
            for i, feature in enumerate(result.features, 1):
                print(f"  {i}. {feature}")
        except Exception as e:
            print(f"Error parsing product info: {e}")


if __name__ == "__main__":
    sys.exit(main())
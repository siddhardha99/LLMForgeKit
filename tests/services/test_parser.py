"""Tests for the parser system."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

from llmforgekit.core.errors import ParserError
from llmforgekit.services.parser import (
    EntityExtractor,
    JSONOutputParser,
    KeyValueParser,
    PydanticOutputParser,
    RegexParser,
    SemanticAligner,
)


def test_json_parser():
    """Test the JSONOutputParser."""
    print("\nTesting JSONOutputParser:")
    
    parser = JSONOutputParser(extract_json=True)
    
    # Test with clean JSON
    json_output = '{"name": "Alice", "age": 30, "skills": ["python", "javascript"]}'
    result = parser.parse(json_output)
    print(f"Clean JSON result: {result}")
    assert result["name"] == "Alice"
    assert result["age"] == 30
    
    # Test with JSON embedded in text
    text_output = """
    Here's the user information:
    
    ```json
    {"name": "Bob", "age": 25, "email": "bob@example.com"}
    ```
    
    Let me know if you need anything else!
    """
    result = parser.parse(text_output)
    print(f"Embedded JSON result: {result}")
    assert result["name"] == "Bob"
    assert result["email"] == "bob@example.com"
    
    # Test with malformed JSON
    malformed_output = """
    {
        name: "Charlie",
        age: 35,
        'email': 'charlie@example.com'
    }
    """
    try:
        result = parser.parse(malformed_output)
        print(f"Fixed malformed JSON: {result}")
        assert result["name"] == "Charlie"
    except ParserError as e:
        print(f"Could not fix malformed JSON: {e}")
    
    # Test confidence scoring
    result = parser.parse_with_confidence(json_output)
    print(f"Confidence for clean JSON: {result.confidence:.2f}")
    assert result.confidence > 0.7


def test_key_value_parser():
    """Test the KeyValueParser."""
    print("\nTesting KeyValueParser:")
    
    parser = KeyValueParser()
    
    # Test with key:value format
    kv_output = """
    Name: Alice Johnson
    Age: 30
    Email: alice@example.com
    Bio: Software developer with 5 years of experience.
    """
    result = parser.parse(kv_output)
    print(f"Key-value result: {result}")
    assert result["Name"] == "Alice Johnson"
    assert result["Email"] == "alice@example.com"
    
    # Test with specific keys
    parser_with_keys = KeyValueParser(keys=["Name", "Email"])
    result = parser_with_keys.parse(kv_output)
    print(f"Filtered key-value result: {result}")
    assert "Name" in result
    assert "Email" in result
    assert "Age" not in result
    
    # Test confidence scoring
    result = parser.parse_with_confidence(kv_output)
    print(f"Confidence for key-value: {result.confidence:.2f}")
    assert result.confidence > 0.5


def test_regex_parser():
    """Test the RegexParser."""
    print("\nTesting RegexParser:")
    
    # Parse phone numbers - use a non-named group pattern
    phone_parser = RegexParser(
        pattern=r"\((\d{3})\) \d{3}-\d{4}",  # Use normal capture group instead of named
        name="PhoneParser",
    )
    
    phone_output = """
    You can reach us at (555) 123-4567 or visit our website.
    For support, call (800) 555-1234.
    """
    
    result = phone_parser.parse(phone_output)
    print(f"Phone parser result: {result}")
    
    # Parse multiple groups - use named groups
    contact_parser = RegexParser(
        pattern=r"Name: (?P<name>[\w\s]+)\s+Email: (?P<email>[\w.@]+)",
        name="ContactParser",
    )
    
    contact_output = "Name: John Smith  Email: john@example.com"
    result = contact_parser.parse(contact_output)
    print(f"Contact parser result: {result}")
    assert result["name"] == "John Smith"
    assert result["email"] == "john@example.com"


def test_entity_extractor():
    """Test the EntityExtractor."""
    print("\nTesting EntityExtractor:")
    
    parser = EntityExtractor(entities=["email", "phone", "url", "date"])
    
    text = """
    Please contact john.doe@example.com or call (555) 123-4567.
    Visit our website at https://www.example.com for more information.
    The event will be held on 12/15/2025 at 3:00 PM.
    """
    
    result = parser.parse(text)
    print(f"Extracted entities: {result}")
    assert len(result["email"]) > 0
    assert len(result["phone"]) > 0
    assert len(result["url"]) > 0
    assert len(result["date"]) > 0


def test_pydantic_parser():
    """Test the PydanticOutputParser."""
    if not PYDANTIC_AVAILABLE:
        print("\nSkipping PydanticOutputParser test (pydantic not installed)")
        return
    
    print("\nTesting PydanticOutputParser:")
    
    # Define a Pydantic model
    class User(BaseModel):
        name: str
        age: int
        email: Optional[str] = None
        tags: List[str] = []
    
    parser = PydanticOutputParser(model=User)
    
    # Test with valid JSON
    json_output = '{"name": "Alice", "age": 30, "email": "alice@example.com", "tags": ["developer", "python"]}'
    result = parser.parse(json_output)
    print(f"Pydantic parser result: {result}")
    assert result.name == "Alice"
    assert result.age == 30
    
    # Test with missing fields
    try:
        invalid_output = '{"name": "Bob"}'  # Missing required 'age' field
        result = parser.parse(invalid_output)
        print("❌ Should have raised an error for missing required field")
    except ParserError as e:
        print(f"✅ Correctly raised error: {e}")


def test_semantic_aligner():
    """Test the SemanticAligner."""
    print("\nTesting SemanticAligner:")
    
    # Create aligner and register parsers
    aligner = SemanticAligner()
    
    aligner.register_parser("json", JSONOutputParser())
    aligner.register_parser("key_value", KeyValueParser())
    aligner.register_parser("phone", RegexParser(pattern=r"(?P<phone>\(\d{3}\) \d{3}-\d{4})"))
    
    # Test with JSON
    json_output = '{"name": "Alice", "age": 30}'
    result = aligner.parse(json_output, "json")
    print(f"Aligner JSON result: {result}")
    assert result["name"] == "Alice"
    
    # Test with key-value
    kv_output = "Name: Bob\nAge: 25"
    result = aligner.parse(kv_output, "key_value")
    print(f"Aligner key-value result: {result}")
    assert result["Name"] == "Bob"
    
    # Test multiple parsers
    outputs = [
        '{"name": "Charlie", "age": 35}',  # JSON
        "Name: Dave\nAge: 40",  # Key-value
        "Invalid output",  # Should fail
    ]
    
    for output in outputs:
        result = aligner.try_parsers(output, ["json", "key_value"])
        if result.success:
            print(f"Successfully parsed with {result.metadata.get('parser_id')}: {result.value}")
        else:
            print(f"Failed to parse: {result.error}")


if __name__ == "__main__":
    test_json_parser()
    test_key_value_parser()
    test_regex_parser()
    test_entity_extractor()
    test_pydantic_parser()
    test_semantic_aligner()
    print("\nAll parser tests completed!")
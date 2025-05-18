"""JSON parser implementations for LLMForgeKit."""

import json
import re
from typing import Any, Dict, List, Optional, Type, Union

try:
    import pydantic
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

from llmforgekit.core.errors import ParserError
from llmforgekit.core.logging import get_logger
from llmforgekit.services.parser.base import BaseOutputParser, OutputValidator, ParsingResult

logger = get_logger("services.parser.json")


class JSONOutputParser(BaseOutputParser):
    """Parser for JSON outputs.
    
    This parser extracts and parses JSON from LLM outputs.
    """
    
    def __init__(
        self,
        extract_json: bool = True,
        name: Optional[str] = None,
    ):
        """Initialize the parser.
        
        Args:
            extract_json: Whether to extract JSON from text
            name: Optional name for the parser
        """
        super().__init__(name=name or "JSONParser")
        self.extract_json = extract_json
    
    def _parse_output(self, output: str) -> Union[Dict[str, Any], List[Any]]:
        """Parse JSON from the LLM output.
        
        Args:
            output: The raw output from the LLM
            
        Returns:
            The parsed JSON data
            
        Raises:
            ValueError: If JSON parsing fails
        """
        if not output.strip():
            raise ValueError("Empty output")
        
        # Extract JSON if needed
        json_str = self._extract_json(output) if self.extract_json else output
        
        try:
            parsed = json.loads(json_str)
            return parsed
        except json.JSONDecodeError as e:
            # Try to fix common JSON errors
            fixed_json = self._fix_json(json_str)
            if fixed_json != json_str:
                try:
                    parsed = json.loads(fixed_json)
                    logger.info("Successfully fixed JSON and parsed it")
                    return parsed
                except json.JSONDecodeError:
                    pass  # If fixing didn't help, raise the original error
            
            raise ValueError(f"Invalid JSON: {str(e)}")
    
    def parse_with_confidence(self, output: str) -> ParsingResult:
        """Parse JSON with confidence score.
        
        Args:
            output: The raw output from the LLM
            
        Returns:
            A ParsingResult object
        """
        try:
            parsed = self.parse(output)
            
            # Calculate confidence based on JSON structure
            confidence = self._calculate_confidence(parsed, output)
            
            return ParsingResult(
                parsed_value=parsed,
                confidence=confidence,
            )
        except Exception as e:
            return ParsingResult(
                parsed_value=None,
                confidence=0.0,
                error=e,
            )
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text.
        
        Args:
            text: The text to extract JSON from
            
        Returns:
            The extracted JSON string
            
        Raises:
            ValueError: If no JSON is found
        """
        # Try to find JSON between triple backticks
        backtick_pattern = r"```(?:json)?\n([\s\S]*?)\n```"
        backtick_match = re.search(backtick_pattern, text)
        if backtick_match:
            return backtick_match.group(1).strip()
        
        # Try to find JSON between curly braces
        brace_pattern = r"(\{[\s\S]*\})"
        brace_match = re.search(brace_pattern, text)
        if brace_match:
            return brace_match.group(1).strip()
        
        # Try to find JSON between square brackets
        bracket_pattern = r"(\[[\s\S]*\])"
        bracket_match = re.search(bracket_pattern, text)
        if bracket_match:
            return bracket_match.group(1).strip()
        
        # If we can't find JSON, return the original text
        return text
    
    def _fix_json(self, json_str: str) -> str:
        """Try to fix common JSON errors.
        
        Args:
            json_str: The JSON string to fix
            
        Returns:
            The fixed JSON string
        """
        # Fix missing quotes around keys
        fixed = re.sub(r'(\s*)(\w+)(\s*):', r'\1"\2"\3:', json_str)
        
        # Fix single quotes
        fixed = fixed.replace("'", '"')
        
        # Fix trailing commas
        fixed = re.sub(r',\s*([\]}])', r'\1', fixed)
        
        # Fix missing quotes around values
        fixed = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_]*)\s*([,}])', r': "\1"\2', fixed)
        
        return fixed
    
    def _calculate_confidence(self, parsed: Any, original: str) -> float:
        """Calculate confidence score for the parsed JSON.
        
        Args:
            parsed: The parsed JSON data
            original: The original output string
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Start with high confidence for successful parsing
        confidence = 0.9
        
        # Check if extraction was needed
        if self.extract_json and original.strip() != json.dumps(parsed):
            confidence -= 0.1
        
        # Check structure
        if isinstance(parsed, dict):
            # More keys generally means more structured data
            num_keys = len(parsed)
            if num_keys < 2:
                confidence -= 0.1
            elif num_keys > 5:
                confidence += 0.05
        elif isinstance(parsed, list):
            # Empty lists are less confident
            if not parsed:
                confidence -= 0.2
        
        # Ensure confidence is within bounds
        return max(0.0, min(1.0, confidence))


class PydanticOutputParser(BaseOutputParser):
    """Parser for outputs that should conform to a Pydantic model.
    
    This parser extracts and parses JSON from LLM outputs, then validates
    it against a Pydantic model.
    """
    
    def __init__(
        self,
        model: Type,
        extract_json: bool = True,
        name: Optional[str] = None,
    ):
        """Initialize the parser.
        
        Args:
            model: The Pydantic model to validate against
            extract_json: Whether to extract JSON from text
            name: Optional name for the parser
            
        Raises:
            ImportError: If Pydantic is not installed
        """
        if not PYDANTIC_AVAILABLE:
            raise ImportError(
                "Pydantic is required for PydanticOutputParser. "
                "Install it with: pip install pydantic"
            )
        
        super().__init__(name=name or f"PydanticParser({model.__name__})")
        self.model = model
        self.json_parser = JSONOutputParser(extract_json=extract_json)
    
    def _parse_output(self, output: str) -> Any:
        """Parse JSON and validate against Pydantic model.
        
        Args:
            output: The raw output from the LLM
            
        Returns:
            The parsed and validated Pydantic model
            
        Raises:
            ValueError: If JSON parsing or validation fails
        """
        # Parse JSON
        json_data = self.json_parser.parse(output)
        
        # Validate against Pydantic model
        try:
            # Handle Pydantic v1 vs v2
            try:
                # Pydantic v2
                return self.model.model_validate(json_data)
            except AttributeError:
                # Pydantic v1
                return self.model.parse_obj(json_data)
        except Exception as e:
            raise ValueError(f"Validation failed: {str(e)}")
    
    def parse_with_confidence(self, output: str) -> ParsingResult:
        """Parse and validate with confidence score.
        
        Args:
            output: The raw output from the LLM
            
        Returns:
            A ParsingResult object
        """
        # First get JSON parsing result
        json_result = self.json_parser.parse_with_confidence(output)
        
        if not json_result.success:
            return ParsingResult(
                parsed_value=None,
                confidence=0.0,
                error=json_result.error,
            )
        
        # Validate against Pydantic model
        try:
            # Handle Pydantic v1 vs v2
            try:
                # Pydantic v2
                model_instance = self.model.model_validate(json_result.value)
            except AttributeError:
                # Pydantic v1
                model_instance = self.model.parse_obj(json_result.value)
            
            # Successful validation has high confidence
            return ParsingResult(
                parsed_value=model_instance,
                confidence=json_result.confidence + 0.1,  # Boost confidence for validation
                metadata={
                    "json_confidence": json_result.confidence,
                    "model": self.model.__name__,
                },
            )
        except Exception as e:
            return ParsingResult(
                parsed_value=None,
                confidence=json_result.confidence * 0.5,  # Reduce confidence for failed validation
                error=e,
                metadata={
                    "json_confidence": json_result.confidence,
                    "model": self.model.__name__,
                    "json_value": json_result.value,
                },
            )


class SchemaValidator(OutputValidator):
    """Validator for JSON schema validation.
    
    This validator checks if JSON data conforms to a schema.
    """
    
    def __init__(self, schema: Dict[str, Any]):
        """Initialize the validator.
        
        Args:
            schema: The JSON schema to validate against
            
        Raises:
            ImportError: If jsonschema is not installed
        """
        try:
            import jsonschema
            self.jsonschema = jsonschema
        except ImportError:
            raise ImportError(
                "jsonschema is required for SchemaValidator. "
                "Install it with: pip install jsonschema"
            )
        
        self.schema = schema
        self.validator = jsonschema.Draft7Validator(schema)
    
    def validate(self, parsed_output: Any) -> bool:
        """Validate the parsed output against the schema.
        
        Args:
            parsed_output: The output to validate
            
        Returns:
            True if the output is valid, False otherwise
        """
        try:
            self.validator.validate(parsed_output)
            return True
        except Exception:
            return False
    
    def get_validation_errors(self, parsed_output: Any) -> List[str]:
        """Get validation errors for the parsed output.
        
        Args:
            parsed_output: The output to validate
            
        Returns:
            A list of validation error messages
        """
        errors = []
        for error in self.validator.iter_errors(parsed_output):
            errors.append(f"{error.message} at {'.'.join(str(p) for p in error.path)}")
        return errors
    
    
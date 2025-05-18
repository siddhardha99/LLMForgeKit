"""Base parser implementations for LLMForgeKit.

This module provides base classes for parsing and validating LLM outputs.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

from llmforgekit.core.base import OutputParser
from llmforgekit.core.errors import ParserError
from llmforgekit.core.logging import get_logger

logger = get_logger("services.parser")


class BaseOutputParser(OutputParser):
    """Base class for output parsers.
    
    This class provides common functionality for all output parsers,
    including error handling and validation.
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the parser.
        
        Args:
            name: Optional name for the parser
        """
        self.name = name or self.__class__.__name__
    
    def parse(self, output: str) -> Any:
        """Parse the LLM output.
        
        Args:
            output: The raw output from the LLM
            
        Returns:
            The parsed result
            
        Raises:
            ParserError: If parsing fails
        """
        try:
            return self._parse_output(output)
        except Exception as e:
            # Wrap the exception in a ParserError
            raise ParserError(
                message=f"Failed to parse output: {str(e)}",
                output=output,
                details={"parser": self.name, "original_error": str(e)},
            ) from e
    
    def parse_with_fallback(self, output: str, fallback: Any) -> Any:
        """Parse the LLM output with a fallback value.
        
        Args:
            output: The raw output from the LLM
            fallback: The fallback value to return if parsing fails
            
        Returns:
            The parsed result, or the fallback value if parsing fails
        """
        try:
            return self.parse(output)
        except ParserError as e:
            logger.warning(f"Parsing failed, using fallback: {e}")
            return fallback
    
    @abstractmethod
    def _parse_output(self, output: str) -> Any:
        """Parse the LLM output (implementation).
        
        This method should be implemented by subclasses.
        
        Args:
            output: The raw output from the LLM
            
        Returns:
            The parsed result
            
        Raises:
            Exception: If parsing fails
        """
        pass


class OutputValidator(ABC):
    """Base class for output validators.
    
    Validators check if parsed output meets certain criteria.
    """
    
    @abstractmethod
    def validate(self, parsed_output: Any) -> bool:
        """Validate the parsed output.
        
        Args:
            parsed_output: The output to validate
            
        Returns:
            True if the output is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_validation_errors(self, parsed_output: Any) -> List[str]:
        """Get validation errors for the parsed output.
        
        Args:
            parsed_output: The output to validate
            
        Returns:
            A list of validation error messages
        """
        pass


class ValidatedOutputParser(BaseOutputParser):
    """Output parser with validation.
    
    This class adds validation to the parsing process.
    """
    
    def __init__(
        self,
        parser: BaseOutputParser,
        validators: Optional[List[OutputValidator]] = None,
        name: Optional[str] = None,
    ):
        """Initialize the parser.
        
        Args:
            parser: The base parser to use
            validators: Optional list of validators to apply
            name: Optional name for the parser
        """
        super().__init__(name=name or f"Validated({parser.name})")
        self.parser = parser
        self.validators = validators or []
    
    def _parse_output(self, output: str) -> Any:
        """Parse and validate the LLM output.
        
        Args:
            output: The raw output from the LLM
            
        Returns:
            The parsed and validated result
            
        Raises:
            ParserError: If parsing or validation fails
        """
        # Parse the output
        parsed = self.parser.parse(output)
        
        # Validate the output
        validation_errors = []
        for validator in self.validators:
            if not validator.validate(parsed):
                validation_errors.extend(validator.get_validation_errors(parsed))
        
        if validation_errors:
            raise ParserError(
                message="Validation failed",
                output=output,
                details={
                    "validation_errors": validation_errors,
                    "parsed_output": parsed,
                },
            )
        
        return parsed
    
    def add_validator(self, validator: OutputValidator) -> None:
        """Add a validator to the parser.
        
        Args:
            validator: The validator to add
        """
        self.validators.append(validator)


class ParsingResult:
    """Result of a parsing operation with confidence score.
    
    This class represents the result of a parsing operation, including
    the parsed result, confidence score, and any error information.
    """
    
    def __init__(
        self,
        parsed_value: Any,
        confidence: float = 1.0,
        error: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the parsing result.
        
        Args:
            parsed_value: The parsed value
            confidence: Confidence score (0.0 to 1.0)
            error: Optional error that occurred during parsing
            metadata: Optional metadata about the parsing process
        """
        self.value = parsed_value
        self.confidence = confidence
        self.error = error
        self.metadata = metadata or {}
        self.success = error is None
    
    def __bool__(self) -> bool:
        """Check if the parsing was successful.
        
        Returns:
            True if parsing was successful, False otherwise
        """
        return self.success
    
    def __str__(self) -> str:
        """Get a string representation of the result.
        
        Returns:
            A string representation
        """
        if self.success:
            return f"ParsingResult(value={self.value}, confidence={self.confidence:.2f})"
        else:
            return f"ParsingResult(error={self.error}, confidence={self.confidence:.2f})"
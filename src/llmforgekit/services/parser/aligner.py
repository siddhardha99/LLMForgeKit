"""Semantic aligner for LLMForgeKit.

This module provides the Intelligent Semantic Aligner service for
parsing, validating, and aligning LLM outputs with expected structures.
"""

from typing import Any, Dict, List, Optional, Type, Union

from llmforgekit.core.base import LLMProvider, OutputParser
from llmforgekit.core.errors import ParserError
from llmforgekit.core.logging import get_logger
from llmforgekit.services.parser.base import BaseOutputParser, ParsingResult

logger = get_logger("services.parser.aligner")


class SemanticAligner:
    """Intelligent Semantic Aligner for LLM outputs.
    
    This service orchestrates the parsing and validation of LLM outputs,
    aligning them with expected structures.
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the semantic aligner.
        
        Args:
            name: Optional name for the aligner
        """
        self.name = name or "SemanticAligner"
        self.parsers: Dict[str, BaseOutputParser] = {}
    
    def register_parser(self, parser_id: str, parser: BaseOutputParser) -> None:
        """Register a parser with the aligner.
        
        Args:
            parser_id: Identifier for the parser
            parser: The parser to register
        """
        self.parsers[parser_id] = parser
        logger.debug(f"Registered parser '{parser_id}': {parser.name}")
    
    def parse(self, output: str, parser_id: str) -> Any:
        """Parse an LLM output using a specific parser.
        
        Args:
            output: The raw output from the LLM
            parser_id: Identifier for the parser to use
            
        Returns:
            The parsed result
            
        Raises:
            ValueError: If the parser_id is not found
            ParserError: If parsing fails
        """
        parser = self.parsers.get(parser_id)
        if not parser:
            raise ValueError(f"Parser '{parser_id}' not found")
        
        return parser.parse(output)
    
    def parse_with_confidence(self, output: str, parser_id: str) -> ParsingResult:
        """Parse an LLM output with confidence score.
        
        Args:
            output: The raw output from the LLM
            parser_id: Identifier for the parser to use
            
        Returns:
            A ParsingResult object
            
        Raises:
            ValueError: If the parser_id is not found
        """
        parser = self.parsers.get(parser_id)
        if not parser:
            raise ValueError(f"Parser '{parser_id}' not found")
        
        # Check if the parser supports confidence scoring
        if hasattr(parser, "parse_with_confidence"):
            return parser.parse_with_confidence(output)
        
        # Fall back to standard parsing
        try:
            parsed = parser.parse(output)
            return ParsingResult(
                parsed_value=parsed,
                confidence=1.0,  # High confidence for successful parsing
            )
        except Exception as e:
            return ParsingResult(
                parsed_value=None,
                confidence=0.0,
                error=e,
            )
    
    def try_parsers(self, output: str, parser_ids: Optional[List[str]] = None) -> ParsingResult:
            """Try multiple parsers and return the best result.
            
            Args:
                output: The raw output from the LLM
                parser_ids: List of parser IDs to try (default: all registered parsers)
                
            Returns:
                The best parsing result
            """
            # Use all parsers if none specified
            if parser_ids is None:
                parser_ids = list(self.parsers.keys())
            
            # Try each parser
            results: List[Tuple[str, ParsingResult]] = []
            for parser_id in parser_ids:
                try:
                    parser = self.parsers.get(parser_id)
                    if not parser:
                        logger.warning(f"Parser '{parser_id}' not found, skipping")
                        continue
                    
                    result = self.parse_with_confidence(output, parser_id)
                    results.append((parser_id, result))
                    
                    logger.debug(
                        f"Parser '{parser_id}' result: "
                        f"success={result.success}, confidence={result.confidence:.2f}"
                    )
                except Exception as e:
                    logger.warning(f"Error trying parser '{parser_id}': {e}")
            
            # Find the best result
            if not results:
                return ParsingResult(
                    parsed_value=None,
                    confidence=0.0,
                    error=ValueError("No parsers were successful"),
                )
            
            # Sort by success and confidence
            results.sort(key=lambda x: (x[1].success, x[1].confidence), reverse=True)
            best_parser_id, best_result = results[0]
            
            # Add parser information to metadata
            if best_result.metadata is None:
                best_result.metadata = {}
            best_result.metadata["parser_id"] = best_parser_id
            best_result.metadata["parser_name"] = self.parsers[best_parser_id].name
            
            return best_result
    
    def parse_output(
        self,
        output: str,
        parser_ids: Optional[List[str]] = None,
        minimum_confidence: float = 0.7,
        fallback: Any = None,
    ) -> Union[Any, None]:
        """Parse an LLM output and return the result if confidence is high enough.
        
        Args:
            output: The raw output from the LLM
            parser_ids: List of parser IDs to try (default: all registered parsers)
            minimum_confidence: Minimum confidence score required
            fallback: Fallback value if parsing fails or confidence is too low
            
        Returns:
            The parsed result if confidence is high enough, otherwise fallback
        """
        result = self.try_parsers(output, parser_ids)
        
        if result.success and result.confidence >= minimum_confidence:
            return result.value
        
        logger.warning(
            f"Parsing did not meet confidence threshold "
            f"({result.confidence:.2f} < {minimum_confidence}), using fallback"
        )
        return fallback
    
    def generate_and_parse(
        self,
        llm: LLMProvider,
        prompt: str,
        parser_id: str,
        retries: int = 2,
        **llm_kwargs: Any,
    ) -> Any:
        """Generate an LLM response and parse it.
        
        This method combines generation and parsing into a single operation,
        with optional retries for parsing failures.
        
        Args:
            llm: The LLM provider to use
            prompt: The prompt to send to the LLM
            parser_id: Identifier for the parser to use
            retries: Number of retries if parsing fails
            **llm_kwargs: Additional arguments for the LLM provider
            
        Returns:
            The parsed result
            
        Raises:
            ParserError: If parsing fails after retries
        """
        parser = self.parsers.get(parser_id)
        if not parser:
            raise ValueError(f"Parser '{parser_id}' not found")
        
        for attempt in range(retries + 1):
            try:
                # Generate the response
                response = llm.generate(prompt, **llm_kwargs)
                
                # Parse the response
                return parser.parse(response)
            except ParserError as e:
                if attempt < retries:
                    logger.warning(
                        f"Parsing failed (attempt {attempt + 1}/{retries + 1}), "
                        f"retrying with the same prompt: {e}"
                    )
                else:
                    logger.error(f"Parsing failed after {retries + 1} attempts: {e}")
                    raise
    
    def generate_and_parse_with_feedback(
        self,
        llm: LLMProvider,
        prompt: str,
        parser_id: str,
        retries: int = 2,
        **llm_kwargs: Any,
    ) -> Any:
        """Generate an LLM response and parse it with feedback for retries.
        
        This method is similar to generate_and_parse, but it adds error information
        to the prompt when retrying.
        
        Args:
            llm: The LLM provider to use
            prompt: The prompt to send to the LLM
            parser_id: Identifier for the parser to use
            retries: Number of retries if parsing fails
            **llm_kwargs: Additional arguments for the LLM provider
            
        Returns:
            The parsed result
            
        Raises:
            ParserError: If parsing fails after retries
        """
        parser = self.parsers.get(parser_id)
        if not parser:
            raise ValueError(f"Parser '{parser_id}' not found")
        
        current_prompt = prompt
        
        for attempt in range(retries + 1):
            try:
                # Generate the response
                response = llm.generate(current_prompt, **llm_kwargs)
                
                # Parse the response
                return parser.parse(response)
            except ParserError as e:
                if attempt < retries:
                    # Add error information to the prompt
                    error_info = f"\n\nYour previous response could not be parsed correctly: {e}\n"
                    error_info += "Please try again and ensure your response is properly formatted.\n"
                    
                    current_prompt = prompt + error_info
                    
                    logger.warning(
                        f"Parsing failed (attempt {attempt + 1}/{retries + 1}), "
                        f"retrying with feedback: {e}"
                    )
                else:
                    logger.error(f"Parsing failed after {retries + 1} attempts: {e}")
                    raise
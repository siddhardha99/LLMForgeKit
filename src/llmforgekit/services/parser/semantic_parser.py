"""Semantic parser implementations for LLMForgeKit."""

import re
from typing import Any, Dict, List, Optional, Pattern, Tuple, Union

from llmforgekit.core.logging import get_logger
from llmforgekit.services.parser.base import BaseOutputParser, ParsingResult

logger = get_logger("services.parser.semantic")


class KeyValueParser(BaseOutputParser):
    """Parser for key-value pairs in text.
    
    This parser extracts key-value pairs from text using regex patterns.
    """
    
    def __init__(
        self,
        pattern: Optional[str] = None,
        keys: Optional[List[str]] = None,
        name: Optional[str] = None,
    ):
        """Initialize the parser.
        
        Args:
            pattern: Optional regex pattern for key-value pairs
            keys: Optional list of keys to extract
            name: Optional name for the parser
        """
        super().__init__(name=name or "KeyValueParser")
        
        # Default pattern: Key: Value or Key = Value
        self.pattern = pattern or r"(?P<key>\w+[\w\s]*)\s*[:=]\s*(?P<value>.+?)(?=\n\w+[\w\s]*\s*[:=]|\n\n|$)"
        self.compiled_pattern = re.compile(self.pattern, re.MULTILINE | re.DOTALL)
        
        self.keys = keys
    
    def _parse_output(self, output: str) -> Dict[str, str]:
        """Parse key-value pairs from the LLM output.
        
        Args:
            output: The raw output from the LLM
            
        Returns:
            Dictionary of key-value pairs
            
        Raises:
            ValueError: If parsing fails
        """
        if not output.strip():
            raise ValueError("Empty output")
        
        # Find all matches
        matches = self.compiled_pattern.finditer(output.strip())
        result = {}
        
        for match in matches:
            key = match.group("key").strip()
            value = match.group("value").strip()
            
            # Clean common formatting
            value = re.sub(r"^['\"]|['\"]$", "", value)  # Remove quotes
            
            # Convert keys to lowercase for comparison
            key_lower = key.lower()
            
            # Skip unwanted keys if keys are specified
            if self.keys:
                wanted = False
                for wanted_key in self.keys:
                    if wanted_key.lower() == key_lower:
                        key = wanted_key  # Use the exact case from wanted_key
                        wanted = True
                        break
                
                if not wanted:
                    continue
            
            result[key] = value
        
        # Check if we found any key-value pairs
        if not result:
            raise ValueError("No key-value pairs found in the output")
        
        # Check if all required keys were found
        if self.keys:
            missing_keys = [k for k in self.keys if k not in result]
            if missing_keys:
                logger.warning(f"Missing keys in output: {missing_keys}")
        
        return result
    
    def parse_with_confidence(self, output: str) -> ParsingResult:
        """Parse key-value pairs with confidence score.
        
        Args:
            output: The raw output from the LLM
            
        Returns:
            A ParsingResult object
        """
        try:
            parsed = self.parse(output)
            
            # Calculate confidence
            if self.keys:
                # Ratio of found keys to requested keys
                found_ratio = len(parsed) / len(self.keys)
                confidence = min(1.0, max(0.1, found_ratio))
            else:
                # Without specified keys, confidence is based on number of pairs
                # Increase the minimum confidence to 0.6
                confidence = min(1.0, max(0.6, min(len(parsed) * 0.1, 0.9)))
            
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
        except Exception as e:
            logger.warning(f"Parsing failed, using fallback. Error: {str(e)}\nOutput was: {output[:100]}...")
            return fallback
        

class RegexParser(BaseOutputParser):
    """Parser for regex patterns in text.
    
    This parser extracts values using regex patterns.
    """
    
    def __init__(
        self,
        pattern: str,
        group_names: Optional[List[str]] = None,
        name: Optional[str] = None,
    ):
        """Initialize the parser.
        
        Args:
            pattern: Regex pattern to match
            group_names: Optional list of group names to extract
            name: Optional name for the parser
        """
        super().__init__(name=name or "RegexParser")
        self.pattern = pattern
        self.compiled_pattern = re.compile(pattern, re.MULTILINE | re.DOTALL)
        self.group_names = group_names
    
    def _parse_output(self, output: str) -> Union[Dict[str, str], List[Dict[str, str]]]:
        """Parse values using regex pattern.
        
        Args:
            output: The raw output from the LLM
            
        Returns:
            Dictionary of group values or list of dictionaries for multiple matches
            
        Raises:
            ValueError: If no matches are found
        """
        if not output.strip():
            raise ValueError("Empty output")
        
        # Find all matches
        matches = list(self.compiled_pattern.finditer(output.strip()))
        
        if not matches:
            raise ValueError(f"No matches found for pattern: {self.pattern}")
        
        # Extract values
        results = []
        for match in matches:
            result = {}
            
            # Get all named groups
            named_groups = match.groupdict()
            if named_groups:
                for name, value in named_groups.items():
                    if self.group_names and name not in self.group_names:
                        continue
                    result[name] = value.strip() if value else ""
            # If no named groups, get numbered groups
            else:
                for i, value in enumerate(match.groups(), 1):
                    group_name = f"group{i}"
                    if self.group_names and group_name not in self.group_names:
                        continue
                    result[group_name] = value.strip() if value else ""
            
            results.append(result)
        
        # Return a single result if there's only one match
        if len(results) == 1:
            return results[0]
        
        return results


class EntityExtractor(BaseOutputParser):
    """Parser for extracting entities from text.
    
    This parser extracts entities like dates, numbers, emails, etc.
    """
    
    # Common entity patterns
    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(\+\d{1,2}\s?)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}\b",
        "url": r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+",
        "date": r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b",
        "time": r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?\b",
        "money": r"\$\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s(?:dollars|USD)",
        "percentage": r"\d+(?:\.\d+)?%",
        "number": r"(?<![a-zA-Z0-9])\d+(?:,\d{3})*(?:\.\d+)?(?![a-zA-Z0-9])",
    }
    
    def __init__(
        self,
        entities: Optional[List[str]] = None,
        custom_patterns: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
    ):
        """Initialize the parser.
        
        Args:
            entities: List of entity types to extract (from PATTERNS keys)
            custom_patterns: Dictionary of custom entity patterns
            name: Optional name for the parser
        """
        super().__init__(name=name or "EntityExtractor")
        
        # Use all patterns if no entities specified
        self.entities = entities or list(self.PATTERNS.keys())
        
        # Combine default patterns with custom patterns
        self.patterns = {**self.PATTERNS}
        if custom_patterns:
            self.patterns.update(custom_patterns)
        
        # Compile patterns
        self.compiled_patterns: Dict[str, Pattern] = {}
        for entity, pattern in self.patterns.items():
            if entity in self.entities:
                self.compiled_patterns[entity] = re.compile(pattern, re.IGNORECASE)
    
    def _parse_output(self, output: str) -> Dict[str, str]:
        """Parse key-value pairs from the LLM output.
        
        Args:
            output: The raw output from the LLM
            
        Returns:
            Dictionary of key-value pairs
            
        Raises:
            ValueError: If parsing fails
        """
        if not output.strip():
            raise ValueError("Empty output")
        
        # Find all matches
        matches = self.compiled_pattern.finditer(output.strip())
        result = {}
        
        for match in matches:
            key = match.group("key").strip()
            value = match.group("value").strip()
            
            # Clean common formatting
            value = re.sub(r"^['\"]|['\"]$", "", value)  # Remove quotes
            
            # Skip placeholder values
            if re.match(r'^\(.*\)$', value):
                logger.warning(f"Skipping placeholder value for key '{key}': {value}")
                continue
            
            # Convert keys to lowercase for comparison
            key_lower = key.lower()
            
            # Skip unwanted keys if keys are specified
            if self.keys:
                wanted = False
                for wanted_key in self.keys:
                    if wanted_key.lower() == key_lower:
                        key = wanted_key  # Use the exact case from wanted_key
                        wanted = True
                        break
                    
                if not wanted:
                    continue
            
            result[key] = value
        
        # Check if we found any key-value pairs
        if not result:
            # Print a snippet of the output to help diagnose the issue
            output_snippet = output[:200] + ('...' if len(output) > 200 else '')
            raise ValueError(f"No key-value pairs found in the output. Output snippet: {output_snippet}")
        
        # Check if all required keys were found
        if self.keys:
            missing_keys = [k for k in self.keys if k not in result]
            if missing_keys:
                logger.warning(f"Missing keys in output: {missing_keys}")
        
        return result
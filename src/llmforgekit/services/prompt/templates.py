"""Prompt template implementations for LLMForgeKit."""

import re
from string import Template
from typing import Any, Dict, List, Optional, Set

from llmforgekit.core.base import PromptTemplate
from llmforgekit.core.errors import PromptError
from llmforgekit.core.logging import get_logger

logger = get_logger("services.prompt")


class StringPromptTemplate(PromptTemplate):
    """Simple string-based prompt template using Python's string.Template.
    
    This template supports variable substitution with $variable or ${variable} syntax.
    """
    
    def __init__(self, template: str, template_id: Optional[str] = None):
        """Initialize the template.
        
        Args:
            template: The template string with $variable placeholders
            template_id: Optional identifier for the template
        """
        self.template_text = template
        self.template_id = template_id
        self._template = Template(template)
        self._variable_names = self._extract_variables(template)
        
        logger.debug(f"Created StringPromptTemplate with {len(self._variable_names)} variables")
    
    def format(self, **kwargs: Any) -> str:
        """Format the template with provided variables.
        
        Args:
            **kwargs: Variables to insert into the template
            
        Returns:
            The formatted prompt
            
        Raises:
            PromptError: If formatting fails (e.g., missing variables)
        """
        # Check for missing variables
        missing_vars = [var for var in self._variable_names if var not in kwargs]
        if missing_vars:
            raise PromptError(
                f"Missing variables in prompt template: {', '.join(missing_vars)}",
                details={"missing_variables": missing_vars}
            )
        
        try:
            return self._template.substitute(**kwargs)
        except KeyError as e:
            # This should not happen due to the check above, but just in case
            raise PromptError(f"Missing variable in prompt template: {e}")
        except ValueError as e:
            raise PromptError(f"Invalid prompt template: {e}")
    
    @property
    def variables(self) -> List[str]:
        """Get the list of variables in this template.
        
        Returns:
            A list of variable names used in this template
        """
        return list(self._variable_names)
    
    @staticmethod
    def _extract_variables(template: str) -> Set[str]:
        """Extract variable names from a template string.
        
        Args:
            template: The template string
            
        Returns:
            A set of variable names
        """
        # Match both $var and ${var} formats
        simple_vars = {match.group(1) for match in re.finditer(r'\$([a-zA-Z_][a-zA-Z0-9_]*)', template)}
        bracketed_vars = {match.group(1) for match in re.finditer(r'\${([a-zA-Z_][a-zA-Z0-9_]*)}', template)}
        return simple_vars.union(bracketed_vars)


class JinjaPromptTemplate(PromptTemplate):
    """Jinja2-based prompt template for more advanced formatting.
    
    This implementation uses Jinja2 for variable substitution and
    provides more advanced features like conditionals, loops, etc.
    """
    
    def __init__(self, template: str, template_id: Optional[str] = None):
        """Initialize the template.
        
        Args:
            template: The template string with Jinja2 syntax
            template_id: Optional identifier for the template
            
        Raises:
            ImportError: If Jinja2 is not installed
        """
        try:
            from jinja2 import Environment, meta, exceptions
        except ImportError:
            raise ImportError(
                "Jinja2 is required for JinjaPromptTemplate. "
                "Install it with: pip install jinja2"
            )
            
        self.template_text = template
        self.template_id = template_id
        self.jinja_env = Environment()
        
        try:
            # Parse the template
            self._template = self.jinja_env.from_string(template)
            
            # Extract variables
            ast = self.jinja_env.parse(template)
            self._variable_names = meta.find_undeclared_variables(ast)
            
            logger.debug(f"Created JinjaPromptTemplate with {len(self._variable_names)} variables")
        except exceptions.TemplateSyntaxError as e:
            raise PromptError(f"Invalid Jinja2 template syntax: {e}")
    
    def format(self, **kwargs: Any) -> str:
        """Format the template with provided variables.
        
        Args:
            **kwargs: Variables to insert into the template
            
        Returns:
            The formatted prompt
            
        Raises:
            PromptError: If formatting fails
        """
        try:
            return self._template.render(**kwargs)
        except Exception as e:
            raise PromptError(f"Failed to render Jinja2 template: {e}")
    
    @property
    def variables(self) -> List[str]:
        """Get the list of variables in this template.
        
        Returns:
            A list of variable names used in this template
        """
        return list(self._variable_names)
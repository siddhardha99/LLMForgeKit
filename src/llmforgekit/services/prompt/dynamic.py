"""Dynamic prompt generation for advanced use cases."""

from typing import Any, Dict, List, Optional, Union

from llmforgekit.core.base import PromptTemplate
from llmforgekit.core.errors import PromptError
from llmforgekit.core.logging import get_logger

logger = get_logger("services.prompt.dynamic")


class PromptComponent:
    """A component of a dynamic prompt.
    
    This class represents a section of a prompt that can be included
    or excluded based on conditions or context.
    """
    
    def __init__(
        self,
        content: str,
        name: Optional[str] = None,
        conditions: Optional[Dict[str, Any]] = None,
        weight: float = 1.0,
    ):
        """Initialize a prompt component.
        
        Args:
            content: The content of this component
            name: Optional name for this component
            conditions: Optional conditions for including this component
            weight: Importance weight (used for truncation decisions)
        """
        self.content = content
        self.name = name
        self.conditions = conditions or {}
        self.weight = weight
    
    def should_include(self, context: Dict[str, Any]) -> bool:
        """Check if this component should be included.
        
        Args:
            context: The current context
            
        Returns:
            True if the component should be included, False otherwise
        """
        if not self.conditions:
            return True
        
        for key, value in self.conditions.items():
            if key not in context:
                return False
            if context[key] != value:
                return False
        
        return True


class DynamicPromptGenerator:
    """Generator for dynamic prompts based on context and components.
    
    This class allows building prompts dynamically by selecting and
    assembling components based on the current context.
    """
    
    def __init__(self, template_id: Optional[str] = None):
        """Initialize a dynamic prompt generator.
        
        Args:
            template_id: Optional identifier for this generator
        """
        self.template_id = template_id
        self.components: List[PromptComponent] = []
        self.prefix: Optional[str] = None
        self.suffix: Optional[str] = None
        self.separator: str = "\n\n"
        self.max_length: Optional[int] = None
    
    def add_component(self, component: Union[str, PromptComponent], **kwargs) -> None:
        """Add a component to the generator.
        
        Args:
            component: The component to add
            **kwargs: If component is a string, these are passed to PromptComponent
        """
        if isinstance(component, str):
            component = PromptComponent(component, **kwargs)
        
        self.components.append(component)
        logger.debug(f"Added component to generator: {component.name or 'unnamed'}")
    
    def set_prefix(self, prefix: str) -> None:
        """Set the prefix for the prompt.
        
        Args:
            prefix: The prefix text
        """
        self.prefix = prefix
    
    def set_suffix(self, suffix: str) -> None:
        """Set the suffix for the prompt.
        
        Args:
            suffix: The suffix text
        """
        self.suffix = suffix
    
    def set_separator(self, separator: str) -> None:
        """Set the separator between components.
        
        Args:
            separator: The separator text
        """
        self.separator = separator
    
    def set_max_length(self, max_length: Optional[int]) -> None:
        """Set the maximum length for the prompt.
        
        Args:
            max_length: The maximum length in characters
        """
        self.max_length = max_length
    
    def generate(self, context: Dict[str, Any]) -> str:
        """Generate a prompt based on the current context.
        
        Args:
            context: The current context
            
        Returns:
            The generated prompt
        """
        # Select components based on conditions
        selected_components = [
            component for component in self.components
            if component.should_include(context)
        ]
        
        logger.debug(f"Selected {len(selected_components)} of {len(self.components)} components")
        
        # Process each component's content with string formatting
        processed_components = []
        for component in selected_components:
            try:
                # Try to format the content with the context variables
                processed_content = component.content.format(**context)
                processed_components.append(processed_content)
            except KeyError as e:
                # Log the error but include the component anyway
                logger.warning(f"Missing variable {e} in component '{component.name or 'unnamed'}'")
                processed_components.append(component.content)
        
        # Build the prompt
        parts = []
        if self.prefix:
            try:
                parts.append(self.prefix.format(**context))
            except KeyError:
                parts.append(self.prefix)
        
        parts.extend(processed_components)
        
        if self.suffix:
            try:
                parts.append(self.suffix.format(**context))
            except KeyError:
                parts.append(self.suffix)
        
        prompt = self.separator.join(parts)
        
        # Check length and truncate if needed
        if self.max_length and len(prompt) > self.max_length:
            logger.warning(f"Prompt exceeds max length ({len(prompt)} > {self.max_length}), truncating")
            prompt = self._truncate_prompt(prompt, selected_components)
        
        return prompt
    
    def to_template(self) -> PromptTemplate:
        """Convert this generator to a standard template.
        
        This allows using the generator with the PromptLibrary.
        
        Returns:
            A PromptTemplate that wraps this generator
        """
        return DynamicPromptTemplate(self)
    
    def _truncate_prompt(self, prompt: str, components: List[PromptComponent]) -> str:
        """Truncate a prompt to fit the maximum length.
        
        This method attempts to be smart about truncation by removing
        lower-weight components first.
        
        Args:
            prompt: The original prompt
            components: The selected components
            
        Returns:
            The truncated prompt
        """
        if not self.max_length:
            return prompt
        
        # If we just have prefix/suffix, do simple truncation
        if len(components) == 0:
            if self.prefix and self.suffix:
                # Keep half of max_length for prefix, half for suffix
                half_length = self.max_length // 2
                return self.prefix[:half_length] + self.suffix[-half_length:]
            elif self.prefix:
                return self.prefix[:self.max_length]
            elif self.suffix:
                return self.suffix[:self.max_length]
            else:
                return ""
        
        # Sort components by weight (ascending, so we remove lowest first)
        sorted_components = sorted(components, key=lambda c: c.weight)
        
        # Remove components one by one until we fit
        remaining_components = list(sorted_components)
        while remaining_components and len(prompt) > self.max_length:
            # Remove the lowest-weight component
            removed = remaining_components.pop(0)
            logger.debug(f"Removing component '{removed.name or 'unnamed'}' (weight: {removed.weight})")
            
            # Rebuild the prompt
            parts = []
            if self.prefix:
                parts.append(self.prefix)
            
            parts.extend(component.content for component in remaining_components)
            
            if self.suffix:
                parts.append(self.suffix)
            
            prompt = self.separator.join(parts)
        
        # If we still don't fit, do hard truncation
        if len(prompt) > self.max_length:
            return prompt[:self.max_length]
        
        return prompt


class DynamicPromptTemplate(PromptTemplate):
    """Template wrapper for DynamicPromptGenerator.
    
    This class implements the PromptTemplate interface for a
    DynamicPromptGenerator, allowing it to be used with the PromptLibrary.
    """
    
    def __init__(self, generator: DynamicPromptGenerator):
        """Initialize the template.
        
        Args:
            generator: The dynamic prompt generator
        """
        self.generator = generator
        self.template_id = generator.template_id
    
    def format(self, **kwargs: Any) -> str:
        """Format the template with provided variables.
        
        Args:
            **kwargs: Variables to insert into the template
            
        Returns:
            The formatted prompt
        """
        return self.generator.generate(kwargs)
    
    @property
    def variables(self) -> List[str]:
        """Get the list of variables in this template.
        
        Note: This returns an empty list as the variables are determined
        dynamically based on the components and conditions.
        
        Returns:
            An empty list
        """
        # Variables are determined dynamically, so we can't list them statically
        return []
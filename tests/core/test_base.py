"""Tests for the base classes."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from llmforgekit.core.base import (
    LLMProvider,
    OutputParser,
    PromptTemplate,
    Tool,
    WorkflowStep,
    Workflow,
)
from llmforgekit.core.errors import (
    LLMProviderError,
    ParserError,
    PromptError,
    WorkflowError,
)


# Test implementation of PromptTemplate
class SimplePromptTemplate(PromptTemplate):
    """A simple implementation of PromptTemplate for testing."""
    
    def __init__(self, template: str):
        """Initialize with a template string."""
        self._template = template
        self._vars = [
            var.split("}")[0]
            for var in template.split("{")[1:]
        ]
    
    def format(self, **kwargs: Any) -> str:
        """Format the template with provided variables."""
        try:
            return self._template.format(**kwargs)
        except KeyError as e:
            raise PromptError(f"Missing variable: {e}")
    
    @property
    def variables(self) -> List[str]:
        """Get the list of variables in this template."""
        return self._vars


# Test implementation of LLMProvider
class MockLLMProvider(LLMProvider):
    """A mock implementation of LLMProvider for testing."""
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Generate a mock response."""
        if "error" in prompt.lower():
            raise LLMProviderError("Mock error")
        return f"Mock response to: {prompt}"
    
    def generate_with_metadata(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a mock response with metadata."""
        if "error" in prompt.lower():
            raise LLMProviderError("Mock error")
        
        return {
            "text": f"Mock response to: {prompt}",
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(prompt.split()) * 2,
                "total_tokens": len(prompt.split()) * 3,
            },
            "model": "mock-model",
        }


# Test implementation of OutputParser
class SimpleJsonParser(OutputParser):
    """A simple JSON parser for testing."""
    
    def parse(self, output: str) -> Dict[str, Any]:
        """Parse JSON output."""
        import json
        
        try:
            return json.loads(output)
        except json.JSONDecodeError as e:
            raise ParserError(f"Invalid JSON: {e}", output=output)


# Test workflow step and workflow
class PrintStep(WorkflowStep):
    """A simple workflow step that prints a message."""
    
    def __init__(self, message: str):
        """Initialize with a message."""
        self._message = message
    
    @property
    def name(self) -> str:
        """Get the step name."""
        return "print_step"
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the step and update state."""
        print(f"Step message: {self._message}")
        state["last_message"] = self._message
        return state


class SimpleWorkflow(Workflow):
    """A simple workflow implementation for testing."""
    
    def __init__(self):
        """Initialize with an empty list of steps."""
        self._steps: List[WorkflowStep] = []
    
    def add_step(self, step: WorkflowStep) -> None:
        """Add a step to the workflow."""
        self._steps.append(step)
    
    def run(self, initial_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run all steps in the workflow."""
        state = initial_state or {}
        
        print("Starting workflow")
        
        for step in self._steps:
            try:
                print(f"Running step: {step.name}")
                state = step.run(state)
            except Exception as e:
                raise WorkflowError(f"Step {step.name} failed: {e}", step=step.name)
        
        print("Workflow complete")
        return state


def test_prompt_template():
    """Test the PromptTemplate implementation."""
    print("\nTesting PromptTemplate:")
    
    template = SimplePromptTemplate("Hello, {name}! Welcome to {service}.")
    
    # Test variables property
    print(f"Template variables: {template.variables}")
    
    # Test successful formatting
    formatted = template.format(name="Alice", service="LLMForgeKit")
    print(f"Formatted prompt: {formatted}")
    
    # Test error handling
    try:
        template.format(name="Bob")
        print("❌ Should have raised an error for missing variable")
    except PromptError as e:
        print(f"✅ Correctly raised error: {e}")


def test_llm_provider():
    """Test the LLMProvider implementation."""
    print("\nTesting LLMProvider:")
    
    provider = MockLLMProvider()
    
    # Test successful generation
    response = provider.generate("Tell me a joke")
    print(f"Generate response: {response}")
    
    # Test generation with metadata
    response_with_meta = provider.generate_with_metadata("What is 2+2?")
    print(f"Response with metadata: {response_with_meta}")
    
    # Test error handling
    try:
        provider.generate("This should cause an error")
        print("❌ Should have raised an error")
    except LLMProviderError as e:
        print(f"✅ Correctly raised error: {e}")


def test_output_parser():
    """Test the OutputParser implementation."""
    print("\nTesting OutputParser:")
    
    parser = SimpleJsonParser()
    
    # Test successful parsing
    valid_json = '{"name": "Alice", "age": 30}'
    parsed = parser.parse(valid_json)
    print(f"Parsed output: {parsed}")
    
    # Test error handling
    try:
        parser.parse("Invalid JSON {")
        print("❌ Should have raised an error for invalid JSON")
    except ParserError as e:
        print(f"✅ Correctly raised error: {e}")


def test_workflow():
    """Test the Workflow implementation."""
    print("\nTesting Workflow:")
    
    # Create a workflow
    workflow = SimpleWorkflow()
    
    # Add steps
    workflow.add_step(PrintStep("This is step 1"))
    workflow.add_step(PrintStep("This is step 2"))
    workflow.add_step(PrintStep("This is step 3"))
    
    # Run the workflow
    final_state = workflow.run({"initial": "value"})
    
    print(f"Final state: {final_state}")


if __name__ == "__main__":
    test_prompt_template()
    test_llm_provider()
    test_output_parser()
    test_workflow()
    print("\nAll base class tests completed!")
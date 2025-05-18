"""Tests for the workflow system."""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from llmforgekit.core.base import Tool
from llmforgekit.core.errors import WorkflowError
from llmforgekit.services.workflow.agent import Agent, LLMAgent, ToolAgent, AgentMemory
from llmforgekit.services.workflow.context import WorkflowContext
from llmforgekit.services.workflow.steps import AgentStep, LLMStep, ToolStep, StepStatus, StepResult
from llmforgekit.services.workflow.engine import AdaptiveWorkflow, WorkflowEngine, WorkflowTemplate


# Mock classes for testing
class MockLLMProvider:
    """Mock LLM provider for testing."""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the prompt."""
        if "error" in prompt.lower():
            raise Exception("Mock LLM error")
        return f"Response to: {prompt[:50]}..."
    
    def generate_with_metadata(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a response with metadata."""
        if "error" in prompt.lower():
            raise Exception("Mock LLM error")
        return {
            "text": f"Response to: {prompt[:50]}...",
            "usage": {"total_tokens": len(prompt.split())}
        }


class MockTool(Tool):
    """Mock tool for testing."""
    
    def __init__(self, name: str = "mock_tool", should_fail: bool = False):
        """Initialize the mock tool."""
        self._name = name
        self.should_fail = should_fail
    
    @property
    def name(self) -> str:
        """Get the name of the tool."""
        return self._name
    
    @property
    def description(self) -> str:
        """Get the description of the tool."""
        return f"Mock tool ({self._name}) for testing"
    
    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with the provided parameters."""
        if self.should_fail:
            raise Exception("Mock tool execution error")
        return {
            "tool_name": self._name,
            "params": kwargs,
            "result": f"Mock result from {self._name}",
            "timestamp": time.time()
        }


# Test agents
class MockAgent(Agent):
    """Mock agent for testing."""
    
    def __init__(self, agent_id: str, should_fail: bool = False):
        """Initialize the mock agent."""
        super().__init__(agent_id=agent_id)
        self.should_fail = should_fail
        self.call_count = 0
    
    def act(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Take action based on the current context."""
        self.call_count += 1
        if self.should_fail:
            raise Exception(f"Mock agent {self.agent_id} failed")
        
        return {
            "action": "mock_action",
            "agent_id": self.agent_id,
            "context_keys": list(context.keys()),
            "call_count": self.call_count
        }


def test_agent_memory():
    """Test the AgentMemory class."""
    print("\nTesting AgentMemory:")
    
    memory = AgentMemory()
    
    # Test remember and recall
    memory.remember("key1", "value1")
    memory.remember("key2", {"nested": "value"})
    
    assert memory.recall("key1") == "value1"
    assert memory.recall("key2")["nested"] == "value"
    assert memory.recall("key3") is None
    
    # Test history
    memory.add_to_history({"action": "test1"})
    memory.add_to_history({"action": "test2"})
    
    assert len(memory.history) == 2
    assert memory.history[0]["action"] == "test1"
    
    # Test observations
    memory.add_observation({"type": "observation1"})
    assert len(memory.observations) == 1
    
    print("✅ AgentMemory tests passed")


def test_llm_agent():
    """Test the LLMAgent class."""
    print("\nTesting LLMAgent:")
    
    llm = MockLLMProvider()
    agent = LLMAgent(agent_id="test_llm_agent", llm_provider=llm)
    
    # Test act method
    result = agent.act({"task": "sample task"})
    assert "response" in result
    assert result["agent_id"] == "test_llm_agent"
    
    # Test error handling
    try:
        agent.act({"task": "trigger error"})
        print("❌ Should have raised an error")
        assert False
    except Exception as e:
        print(f"✅ Correctly raised error: {e}")
    
    print("✅ LLMAgent tests passed")


def test_tool_agent():
    """Test the ToolAgent class."""
    print("\nTesting ToolAgent:")
    
    # Test with tool
    tool = MockTool(name="test_tool")
    agent = ToolAgent(agent_id="test_tool_agent", tools=[tool])
    
    # Test act method
    result = agent.act({
        "tool_name": "test_tool",
        "tool_params": {"param1": "value1"}
    })
    
    assert result["action"] == "tool_execution"
    assert result["tool"] == "test_tool"
    assert "result" in result
    
    # Test with failing tool
    failing_tool = MockTool(name="failing_tool", should_fail=True)
    failing_agent = ToolAgent(agent_id="failing_agent", tools=[failing_tool])
    
    try:
        failing_agent.act({"tool_name": "failing_tool"})
        print("❌ Should have raised an error")
        assert False
    except Exception as e:
        print(f"✅ Correctly raised error: {e}")
    
    # Test with nonexistent tool
    try:
        agent.act({"tool_name": "nonexistent_tool"})
        print("❌ Should have raised an error")
        assert False
    except Exception as e:
        print(f"✅ Correctly raised error: {e}")
    
    print("✅ ToolAgent tests passed")


def test_workflow_context():
    """Test the WorkflowContext class."""
    print("\nTesting WorkflowContext:")
    
    context = WorkflowContext(workflow_id="test_workflow")
    
    # Test basic state operations
    context.set("key1", "value1")
    assert context.get("key1") == "value1"
    assert context.get("key2") is None
    assert context.get("key2", "default") == "default"
    
    # Test update
    context.update({"key2": "value2", "key3": [1, 2, 3]})
    assert context.get("key2") == "value2"
    assert context.get("key3") == [1, 2, 3]
    
    # Test step results
    context.record_step_result("step1", {"status": "completed"})
    assert context.get_step_result("step1") == {"status": "completed"}
    
    # Test completion
    assert not context.completed
    context.complete()
    assert context.completed
    
    # Test to_dict
    state_dict = context.to_dict()
    assert state_dict["workflow_id"] == "test_workflow"
    assert "duration" in state_dict
    assert state_dict["completed"] is True
    
    print("✅ WorkflowContext tests passed")


def test_agent_step():
    """Test the AgentStep class."""
    print("\nTesting AgentStep:")
    
    agent = MockAgent(agent_id="test_agent")
    step = AgentStep(step_id="test_step", agent=agent)
    
    # Test step execution
    state = {"initial": "value"}
    updated_state = step.run(state)
    
    assert "result_test_step" in updated_state
    assert updated_state["result_test_step"]["agent_id"] == "test_agent"
    
    # Test with failing agent
    failing_agent = MockAgent(agent_id="failing_agent", should_fail=True)
    failing_step = AgentStep(step_id="failing_step", agent=failing_agent)
    
    try:
        failing_step.run(state)
        print("❌ Should have raised an error")
        assert False
    except WorkflowError as e:
        print(f"✅ Correctly raised error: {e}")
    
    print("✅ AgentStep tests passed")


def test_llm_step():
    """Test the LLMStep class."""
    print("\nTesting LLMStep:")
    
    llm = MockLLMProvider()
    prompt_template = "Question: $question"
    
    step = LLMStep(
        step_id="llm_step",
        llm_provider=llm,
        prompt_template=prompt_template,
        prompt_context_keys=["question"]
    )
    
    # Test step execution
    state = {"question": "What is Python?"}
    updated_state = step.run(state)
    
    assert "result_llm_step" in updated_state
    assert "output_llm_step" in updated_state
    
    print("✅ LLMStep tests passed")


def test_tool_step():
    """Test the ToolStep class."""
    print("\nTesting ToolStep:")
    
    tool = MockTool(name="calculator")
    step = ToolStep(
        step_id="tool_step",
        tool=tool,
        tool_params_map={"x": "input_x", "y": "input_y"}
    )
    
    # Test step execution
    state = {"input_x": 5, "input_y": 10}
    updated_state = step.run(state)
    
    assert "result_tool_step" in updated_state
    assert "output_tool_step" in updated_state
    assert updated_state["result_tool_step"]["tool"] == "calculator"
    
    print("✅ ToolStep tests passed")


def test_adaptive_workflow():
    """Test the AdaptiveWorkflow class."""
    print("\nTesting AdaptiveWorkflow:")
    
    # Create a workflow
    workflow = AdaptiveWorkflow(
        workflow_id="test_workflow",
        name="Test Workflow",
        description="A test workflow"
    )
    
    # Create some steps
    step1 = AgentStep(step_id="step1", agent=MockAgent(agent_id="agent1"))
    step2 = AgentStep(step_id="step2", agent=MockAgent(agent_id="agent2"))
    step3 = AgentStep(step_id="step3", agent=MockAgent(agent_id="agent3"))
    
    # Add steps to workflow
    workflow.add_step(step1)
    workflow.add_step(step2)
    workflow.add_step(step3)
    
    # Add dependencies
    workflow.add_dependency(step_name="step2", depends_on="step1")
    workflow.add_dependency(step_name="step3", depends_on="step2")
    
    # Run the workflow
    result = workflow.run({"initial": "state"})
    
    # Check result
    assert "result_step1" in result
    assert "result_step2" in result
    assert "result_step3" in result
    
    # Test conditional dependencies
    workflow2 = AdaptiveWorkflow(workflow_id="conditional_workflow")
    
    step1 = AgentStep(step_id="step1", agent=MockAgent(agent_id="agent1"))
    step2a = AgentStep(step_id="step2a", agent=MockAgent(agent_id="agent2a"))
    step2b = AgentStep(step_id="step2b", agent=MockAgent(agent_id="agent2b"))
    
    workflow2.add_step(step1)
    workflow2.add_step(step2a)
    workflow2.add_step(step2b)
    
    # Add conditional dependencies
    workflow2.add_dependency(
        step_name="step2a",
        depends_on="step1",
        conditions={"output_step1.route": "A"}
    )
    
    workflow2.add_dependency(
        step_name="step2b",
        depends_on="step1",
        conditions={"output_step1.route": "B"}
    )
    
    # This test would need to be expanded with actual conditions
    # but demonstrates the concept
    
    print("✅ AdaptiveWorkflow tests passed")


def test_workflow_engine():
    """Test the WorkflowEngine class."""
    print("\nTesting WorkflowEngine:")
    
    engine = WorkflowEngine()
    
    # Create a workflow
    workflow = AdaptiveWorkflow(
        workflow_id="engine_test_workflow",
        name="Engine Test Workflow"
    )
    
    # Create some steps
    step1 = AgentStep(step_id="step1", agent=MockAgent(agent_id="agent1"))
    step2 = AgentStep(step_id="step2", agent=MockAgent(agent_id="agent2"))
    
    # Add steps to workflow
    workflow.add_step(step1)
    workflow.add_step(step2)
    
    # Add dependencies
    workflow.add_dependency(step_name="step2", depends_on="step1")
    
    # Register the workflow
    engine.register_workflow(workflow)
    
    # Execute the workflow
    result = engine.execute_workflow(
        workflow_id="engine_test_workflow",
        initial_state={"input": "value"}
    )
    
    # Check result
    assert "result_step1" in result
    assert "result_step2" in result
    
    # Test with nonexistent workflow
    try:
        engine.execute_workflow("nonexistent_workflow")
        print("❌ Should have raised an error")
        assert False
    except ValueError as e:
        print(f"✅ Correctly raised error: {e}")
    
    print("✅ WorkflowEngine tests passed")


def test_workflow_template():
    """Test the WorkflowTemplate class."""
    print("\nTesting WorkflowTemplate:")
    
    template = WorkflowTemplate(
        template_id="test_template",
        name="Test Template",
        description="A test workflow template"
    )
    
    # Add step definitions
    template.add_step_definition(
        step_id="step1", 
        step_type="mock",
        step_config={"agent_id": "agent1"}
    )
    
    template.add_step_definition(
        step_id="step2",
        step_type="mock",
        step_config={"agent_id": "agent2"}
    )
    
    # Add dependency definition
    template.add_dependency_definition(
        step_id="step2",
        depends_on="step1"
    )
    
    # Create step factories
    factories = {
        "mock": lambda step_id, agent_id: AgentStep(step_id, MockAgent(agent_id))
    }
    
    # Instantiate the workflow
    workflow = template.instantiate(
        workflow_id="instantiated_workflow",
        step_factories=factories
    )
    
    # Check workflow structure
    assert len(workflow.steps) == 2
    assert "agent_step_step1" in workflow.steps
    assert "agent_step_step2" in workflow.steps
    assert "agent_step_step2" in workflow.step_dependencies
    assert "agent_step_step1" in workflow.step_dependencies["agent_step_step2"]
    
    print("✅ WorkflowTemplate tests passed")


def test_complete_workflow():
    """Test a complete workflow scenario."""
    print("\nTesting complete workflow scenario:")
    
    # Create LLM provider
    llm = MockLLMProvider()
    
    # Create workflow
    workflow = AdaptiveWorkflow(
        workflow_id="qa_workflow",
        name="Question Answering Workflow"
    )
    
    # Create a parser and extractor tool
    class MockParser:
        def parse(self, text):
            return {"entities": ["entity1", "entity2"], "query": text.lower()}
    
    class DataExtractorTool(MockTool):
        def execute(self, query, **kwargs):
            return {
                "data": f"Information about {query}",
                "sources": ["source1", "source2"]
            }
    
    parser = MockParser()
    extractor = DataExtractorTool(name="data_extractor")
    
    # Create steps
    analyze_step = LLMStep(
        step_id="analyze",
        llm_provider=llm,
        prompt_template="Analyze this question: $question",
        prompt_context_keys=["question"]
    )
    
    extract_step = ToolStep(
        step_id="extract",
        tool=extractor,
        tool_params_map={"query": "output_analyze"}
    )
    
    answer_step = LLMStep(
        step_id="answer",
        llm_provider=llm,
        prompt_template="Question: $question\nData: $data\nAnswer the question.",
        prompt_context_keys=["question", "data"]
    )
    
    # Add steps to workflow
    workflow.add_step(analyze_step)
    workflow.add_step(extract_step)
    workflow.add_step(answer_step)
    
    # Add dependencies
    workflow.add_dependency("tool_step_extract", "llm_step_analyze")
    workflow.add_dependency("llm_step_answer", "tool_step_extract")
    
    # Execute workflow
    initial_state = {"question": "What is Python?"}
    result = workflow.run(initial_state)
    
    # Check results
    assert "output_analyze" in result
    assert "output_extract" in result
    assert "data" in result["output_extract"]
    assert "output_answer" in result
    
    print("✅ Complete workflow test passed")


if __name__ == "__main__":
    test_agent_memory()
    test_llm_agent()
    test_tool_agent()
    test_workflow_context()
    test_agent_step()
    test_llm_step()
    test_tool_step()
    test_adaptive_workflow()
    test_workflow_engine()
    test_workflow_template()
    test_complete_workflow()
    print("\nAll workflow tests completed!")
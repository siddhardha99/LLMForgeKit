"""Agent-based workflow step implementations."""

from typing import Any, Dict, List, Optional, Type, Union

from llmforgekit.core.base import LLMProvider, OutputParser, Tool, WorkflowStep
from llmforgekit.core.errors import WorkflowError
from llmforgekit.core.logging import get_logger
from llmforgekit.services.prompt import PromptTemplate, PromptLibrary
from llmforgekit.services.parser import BaseOutputParser
from llmforgekit.services.workflow.agent import Agent, LLMAgent, ToolAgent
from llmforgekit.services.workflow.context import WorkflowContext

logger = get_logger("services.workflow.steps")


class StepStatus:
    """Status indicators for workflow steps."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepResult:
    """Result of a workflow step execution."""
    
    def __init__(
        self,
        step_id: str,
        status: str,
        output: Optional[Any] = None,
        error: Optional[Exception] = None,
    ):
        """Initialize the step result.
        
        Args:
            step_id: The ID of the step
            status: The step execution status
            output: Optional step output
            error: Optional error that occurred
        """
        self.step_id = step_id
        self.status = status
        self.output = output
        self.error = error
    
    @property
    def success(self) -> bool:
        """Check if the step execution was successful.
        
        Returns:
            True if successful, False otherwise
        """
        return self.status == StepStatus.COMPLETED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary.
        
        Returns:
            A dictionary representation of the result
        """
        return {
            "step_id": self.step_id,
            "status": self.status,
            "output": self.output,
            "error": str(self.error) if self.error else None,
            "success": self.success,
        }


class AgentStep(WorkflowStep):
    """Workflow step that uses an agent to perform a task.
    
    This step provides a high-level interface for integrating
    agents into workflows, allowing them to interact with the
    workflow context and take actions.
    """
    
    def __init__(
        self,
        step_id: str,
        agent: Agent,
        prompt_template: Optional[Union[str, PromptTemplate]] = None,
        output_parser: Optional[OutputParser] = None,
        prompt_context_keys: Optional[List[str]] = None,
    ):
        """Initialize the agent step.
        
        Args:
            step_id: Unique identifier for this step
            agent: The agent to use for this step
            prompt_template: Optional prompt template for LLM agents
            output_parser: Optional parser for agent outputs
            prompt_context_keys: Keys from the workflow context to include in prompts
        """
        self.step_id = step_id
        self.agent = agent
        self.prompt_template = prompt_template
        self.output_parser = output_parser
        self.prompt_context_keys = prompt_context_keys or []
    
    @property
    def name(self) -> str:
        """Get the name of the workflow step.
        
        Returns:
            The step name
        """
        return f"agent_step_{self.step_id}"
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the workflow step.
        
        Args:
            state: The current workflow state
            
        Returns:
            The updated workflow state
            
        Raises:
            WorkflowError: If the step execution fails
        """
        logger.info(f"Running agent step {self.step_id} with agent {self.agent.agent_id}")
        
        # Prepare context for the agent
        agent_context = {
            "current_step": self.step_id,
            "timestamp": state.get("timestamp", 0),
        }
        
        # Include specified context keys
        for key in self.prompt_context_keys:
            if key in state:
                agent_context[key] = state[key]
        
        try:
            # Execute the agent action
            result = self.agent.act(agent_context)
            
            # Process the result if needed
            if self.output_parser and "response" in result:
                try:
                    parsed_output = self.output_parser.parse(result["response"])
                    result["parsed_output"] = parsed_output
                except Exception as e:
                    logger.warning(f"Failed to parse agent output: {e}")
            
            # Update state with the result
            state[f"result_{self.step_id}"] = result
            
            # Also create a direct reference to the output
            if "parsed_output" in result:
                state[f"output_{self.step_id}"] = result["parsed_output"]
            elif "response" in result:
                state[f"output_{self.step_id}"] = result["response"]
            elif "result" in result:
                state[f"output_{self.step_id}"] = result["result"]
            
            logger.info(f"Agent step {self.step_id} completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Agent step {self.step_id} failed: {e}")
            raise WorkflowError(
                message=f"Agent step {self.step_id} failed: {str(e)}",
                step=self.step_id,
                details={"agent_id": self.agent.agent_id, "context": agent_context}
            )


class LLMStep(AgentStep):
    """Workflow step that uses an LLM agent.
    
    This specialized step provides additional functionality
    specific to LLM-based agents, including prompt management
    and output parsing.
    """
    
    def __init__(
        self,
        step_id: str,
        llm_provider: LLMProvider,
        prompt_template: Union[str, PromptTemplate],
        output_parser: Optional[OutputParser] = None,
        prompt_context_keys: Optional[List[str]] = None,
        agent_name: Optional[str] = None,
    ):
        """Initialize the LLM step.
        
        Args:
            step_id: Unique identifier for this step
            llm_provider: The LLM provider to use
            prompt_template: The prompt template for this step
            output_parser: Optional parser for LLM outputs
            prompt_context_keys: Keys from the workflow context to include in prompts
            agent_name: Optional name for the LLM agent
        """
        agent = LLMAgent(
            agent_id=f"llm_agent_{step_id}",
            llm_provider=llm_provider,
            name=agent_name or f"LLM Agent {step_id}",
        )
        
        super().__init__(
            step_id=step_id,
            agent=agent,
            prompt_template=prompt_template,
            output_parser=output_parser,
            prompt_context_keys=prompt_context_keys,
        )


class ToolStep(AgentStep):
    """Workflow step that uses a tool agent.
    
    This specialized step provides functionality for executing
    external tools within a workflow.
    """
    
    def __init__(
        self,
        step_id: str,
        tool: Tool,
        tool_params_map: Optional[Dict[str, str]] = None,
        agent_name: Optional[str] = None,
    ):
        """Initialize the tool step.
        
        Args:
            step_id: Unique identifier for this step
            tool: The tool to execute
            tool_params_map: Mapping from workflow state keys to tool parameters
            agent_name: Optional name for the tool agent
        """
        agent = ToolAgent(
            agent_id=f"tool_agent_{step_id}",
            tools=[tool],
            name=agent_name or f"Tool Agent {step_id}",
        )
        
        self.tool = tool
        self.tool_params_map = tool_params_map or {}
        
        super().__init__(
            step_id=step_id,
            agent=agent,
        )
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the workflow step.
        
        Args:
            state: The current workflow state
            
        Returns:
            The updated workflow state
            
        Raises:
            WorkflowError: If the step execution fails
        """
        logger.info(f"Running tool step {self.step_id} with tool {self.tool.name}")
        
        # Prepare context for the agent
        agent_context = {
            "current_step": self.step_id,
            "timestamp": state.get("timestamp", 0),
            "tool_name": self.tool.name,
            "tool_params": {},
        }
        
        # Map state values to tool parameters
        for param_name, state_key in self.tool_params_map.items():
            if state_key in state:
                agent_context["tool_params"][param_name] = state[state_key]
        
        try:
            # Execute the agent action (which will call the tool)
            result = self.agent.act(agent_context)
            
            # Update state with the result
            state[f"result_{self.step_id}"] = result
            
            # Also create a direct reference to the output
            if "result" in result:
                state[f"output_{self.step_id}"] = result["result"]
            
            logger.info(f"Tool step {self.step_id} completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Tool step {self.step_id} failed: {e}")
            raise WorkflowError(
                message=f"Tool step {self.step_id} failed: {str(e)}",
                step=self.step_id,
                details={"tool": self.tool.name, "params": agent_context.get("tool_params")}
            )
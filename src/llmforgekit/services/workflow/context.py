"""Context management for the workflow system."""

import time
from typing import Any, Dict, List, Optional, Set, Union

from llmforgekit.core.logging import get_logger

logger = get_logger("services.workflow.context")


class WorkflowContext:
    """Context for a workflow execution.
    
    This class manages the state and data for a workflow execution,
    including inputs, outputs, and intermediate results.
    """
    
    def __init__(
        self,
        workflow_id: Optional[str] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the workflow context.
        
        Args:
            workflow_id: Optional identifier for the workflow
            initial_state: Optional initial state for the workflow
        """
        self.workflow_id = workflow_id or f"workflow_{int(time.time())}"
        self.state: Dict[str, Any] = initial_state or {}
        self.step_results: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.end_time: Optional[float] = None
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update the workflow state.
        
        Args:
            updates: Dictionary of state updates
        """
        self.state.update(updates)
        self.history.append({
            "action": "state_update",
            "updates": updates,
            "timestamp": time.time(),
        })
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the workflow state.
        
        Args:
            key: The state key
            value: The value to set
        """
        self.state[key] = value
        self.history.append({
            "action": "state_set",
            "key": key,
            "value": value,
            "timestamp": time.time(),
        })
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the workflow state.
        
        Args:
            key: The state key
            default: Default value if key is not found
            
        Returns:
            The state value, or the default if not found
        """
        return self.state.get(key, default)
    
    def record_step_result(self, step_id: str, result: Any) -> None:
        """Record the result of a workflow step.
        
        Args:
            step_id: The ID of the workflow step
            result: The result of the step
        """
        self.step_results[step_id] = result
        self.history.append({
            "action": "step_completed",
            "step_id": step_id,
            "timestamp": time.time(),
        })
    
    def get_step_result(self, step_id: str) -> Optional[Any]:
        """Get the result of a workflow step.
        
        Args:
            step_id: The ID of the workflow step
            
        Returns:
            The step result, or None if not found
        """
        return self.step_results.get(step_id)
    
    def complete(self) -> None:
        """Mark the workflow as complete."""
        self.end_time = time.time()
        self.history.append({
            "action": "workflow_completed",
            "timestamp": self.end_time,
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the context to a dictionary.
        
        Returns:
            A dictionary representation of the context
        """
        return {
            "workflow_id": self.workflow_id,
            "state": self.state,
            "step_results": self.step_results,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": (self.end_time or time.time()) - self.start_time,
            "completed": self.end_time is not None,
        }
    
    @property
    def completed(self) -> bool:
        """Check if the workflow is completed.
        
        Returns:
            True if the workflow is completed, False otherwise
        """
        return self.end_time is not None
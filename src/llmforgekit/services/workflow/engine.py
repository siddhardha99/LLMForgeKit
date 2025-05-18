"""Adaptive Agentic Choreographer workflow engine."""

import time
from typing import Any, Dict, List, Optional, Set, Type, Union

from llmforgekit.core.base import Workflow, WorkflowStep
from llmforgekit.core.errors import WorkflowError
from llmforgekit.core.logging import get_logger
from llmforgekit.services.workflow.context import WorkflowContext
from llmforgekit.services.workflow.steps import StepResult, StepStatus

logger = get_logger("services.workflow.engine")


class AdaptiveWorkflow(Workflow):
    """Adaptive workflow implementation.
    
    This class provides a dynamic workflow execution engine that
    can adapt to changing conditions and make decisions based on
    step outcomes.
    """
    
    def __init__(
        self,
        workflow_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Initialize the workflow.
        
        Args:
            workflow_id: Unique identifier for this workflow
            name: Human-readable name for this workflow
            description: Description of this workflow
        """
        self.workflow_id = workflow_id
        self.name = name or workflow_id
        self.description = description
        self.steps: Dict[str, WorkflowStep] = {}
        self.step_dependencies: Dict[str, Set[str]] = {}
        self.step_conditions: Dict[str, Dict[str, Any]] = {}
        self.start_steps: Set[str] = set()
        self.end_steps: Set[str] = set()
    
    def add_step(self, step: WorkflowStep) -> None:
        """Add a step to the workflow.
        
        Args:
            step: The workflow step to add
        """
        self.steps[step.name] = step
        
        # By default, a new step has no dependencies and is a start step
        self.step_dependencies[step.name] = set()
        self.start_steps.add(step.name)
    
    def add_dependency(
        self,
        step_name: str,
        depends_on: str,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a dependency between steps.
        
        Args:
            step_name: The name of the dependent step
            depends_on: The name of the step this depends on
            conditions: Optional conditions for this dependency
            
        Raises:
            ValueError: If either step doesn't exist
        """
        if step_name not in self.steps:
            raise ValueError(f"Step '{step_name}' not found in workflow")
        if depends_on not in self.steps:
            raise ValueError(f"Dependency step '{depends_on}' not found in workflow")
        
        # Add the dependency
        self.step_dependencies[step_name].add(depends_on)
        
        # This is no longer a start step because it has dependencies
        if step_name in self.start_steps:
            self.start_steps.remove(step_name)
        
        # The dependency might be an end step if it has no other steps depending on it
        is_end_step = True
        for deps in self.step_dependencies.values():
            if depends_on in deps:
                is_end_step = False
                break
        if is_end_step:
            self.end_steps.add(depends_on)
        
        # If any step depends on this one, it's not an end step
        if step_name in self.end_steps:
            self.end_steps.remove(step_name)
        
        # Store conditions for this dependency if provided
        if conditions:
            if step_name not in self.step_conditions:
                self.step_conditions[step_name] = {}
            self.step_conditions[step_name][depends_on] = conditions
    
    def run(self, initial_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the workflow.
        
        Args:
            initial_state: Optional initial state for the workflow
            
        Returns:
            The final workflow state
            
        Raises:
            WorkflowError: If the workflow execution fails
        """
        # Create a workflow context
        context = WorkflowContext(
            workflow_id=self.workflow_id,
            initial_state=initial_state or {},
        )
        
        # Initialize execution state
        pending_steps = set(self.steps.keys())
        completed_steps: Set[str] = set()
        step_results: Dict[str, StepResult] = {}
        
        logger.info(f"Starting workflow {self.workflow_id}")
        
        # Execute steps until all are completed or no more can be executed
        while pending_steps:
            # Find steps that can be executed (all dependencies satisfied)
            ready_steps = []
            for step_name in pending_steps:
                if self._are_dependencies_satisfied(step_name, completed_steps, context.state):
                    ready_steps.append(step_name)
            
            if not ready_steps:
                # No steps ready, but some still pending - this is a deadlock
                raise WorkflowError(
                    message=f"Workflow {self.workflow_id} deadlocked: {len(pending_steps)} steps pending, none ready",
                    details={"pending_steps": list(pending_steps), "completed_steps": list(completed_steps)}
                )
            
            # Execute ready steps
            for step_name in ready_steps:
                step = self.steps[step_name]
                
                try:
                    logger.info(f"Executing step: {step_name}")
                    context.state["timestamp"] = time.time()
                    
                    # Execute the step
                    updated_state = step.run(context.state)
                    
                    # Update the context state
                    context.state = updated_state
                    
                    # Record the result
                    result = StepResult(
                        step_id=step_name,
                        status=StepStatus.COMPLETED,
                        output=context.state.get(f"output_{step_name}")
                    )
                    step_results[step_name] = result
                    context.record_step_result(step_name, result.to_dict())
                    
                    # Mark step as completed
                    pending_steps.remove(step_name)
                    completed_steps.add(step_name)
                    
                    logger.info(f"Step {step_name} completed successfully")
                    
                except Exception as e:
                    # Record the error
                    logger.error(f"Step {step_name} failed: {e}")
                    
                    result = StepResult(
                        step_id=step_name,
                        status=StepStatus.FAILED,
                        error=e
                    )
                    step_results[step_name] = result
                    context.record_step_result(step_name, result.to_dict())
                    
                    # Mark step as completed (failed, but still completed)
                    pending_steps.remove(step_name)
                    completed_steps.add(step_name)
                    
                    # Add error information to the context
                    context.state[f"error_{step_name}"] = str(e)
                    
                    # Decide whether to continue or fail the workflow
                    # For now, we fail the workflow
                    raise WorkflowError(
                        message=f"Workflow {self.workflow_id} failed at step {step_name}: {str(e)}",
                        step=step_name,
                        details={"context": context.to_dict()}
                    )
        
        # Mark the workflow as complete
        context.complete()
        logger.info(f"Workflow {self.workflow_id} completed successfully")
        
        return context.state
    
    def _are_dependencies_satisfied(
        self,
        step_name: str,
        completed_steps: Set[str],
        state: Dict[str, Any],
    ) -> bool:
        """Check if all dependencies for a step are satisfied.
        
        Args:
            step_name: The name of the step to check
            completed_steps: Set of completed step names
            state: The current workflow state
            
        Returns:
            True if all dependencies are satisfied, False otherwise
        """
        # Get dependencies for this step
        dependencies = self.step_dependencies.get(step_name, set())
        
        # No dependencies means ready to execute
        if not dependencies:
            return True
        
        # Check if all dependencies are completed
        for dependency in dependencies:
            if dependency not in completed_steps:
                return False
            
            # Check conditions for this dependency if any
            conditions = self.step_conditions.get(step_name, {}).get(dependency)
            if conditions:
                for key, expected_value in conditions.items():
                    # Special case for condition functions
                    if callable(expected_value):
                        if key not in state or not expected_value(state[key]):
                            return False
                    else:
                        if key not in state or state[key] != expected_value:
                            return False
        
        return True


class WorkflowEngine:
    """Engine for executing workflows.
    
    This class provides a central point for executing and
    managing workflows.
    """
    
    def __init__(self):
        """Initialize the workflow engine."""
        self.workflows: Dict[str, AdaptiveWorkflow] = {}
    
    def register_workflow(self, workflow: AdaptiveWorkflow) -> None:
        """Register a workflow with the engine.
        
        Args:
            workflow: The workflow to register
        """
        self.workflows[workflow.workflow_id] = workflow
        logger.info(f"Registered workflow: {workflow.workflow_id}")
    
    def execute_workflow(
        self,
        workflow_id: str,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a workflow.
        
        Args:
            workflow_id: The ID of the workflow to execute
            initial_state: Optional initial state for the workflow
            
        Returns:
            The final workflow state
            
        Raises:
            ValueError: If the workflow doesn't exist
            WorkflowError: If the workflow execution fails
        """
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow '{workflow_id}' not found")
        
        logger.info(f"Executing workflow: {workflow_id}")
        return workflow.run(initial_state)


class WorkflowTemplate:
    """Template for creating workflows.
    
    This class provides a way to define reusable workflow templates
    that can be instantiated with different parameters.
    """
    
    def __init__(
        self,
        template_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Initialize the workflow template.
        
        Args:
            template_id: Unique identifier for this template
            name: Human-readable name for this template
            description: Description of this template
        """
        self.template_id = template_id
        self.name = name or template_id
        self.description = description
        self.steps: List[Dict[str, Any]] = []
        self.dependencies: List[Dict[str, Any]] = []
    
    def add_step_definition(
        self,
        step_id: str,
        step_type: str,
        step_config: Dict[str, Any],
    ) -> None:
        """Add a step definition to the template.
        
        Args:
            step_id: Unique identifier for this step
            step_type: Type of step to create
            step_config: Configuration for the step
        """
        self.steps.append({
            "step_id": step_id,
            "step_type": step_type,
            "config": step_config,
        })
    
    def add_dependency_definition(
        self,
        step_id: str,
        depends_on: str,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a dependency definition to the template.
        
        Args:
            step_id: The ID of the dependent step
            depends_on: The ID of the step this depends on
            conditions: Optional conditions for this dependency
        """
        self.dependencies.append({
            "step_id": step_id,
            "depends_on": depends_on,
            "conditions": conditions,
        })
    
    def instantiate(
        self,
        workflow_id: str,
        step_factories: Dict[str, Type[WorkflowStep]],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> AdaptiveWorkflow:
        """Create a workflow instance from this template.
        
        Args:
            workflow_id: ID for the new workflow
            step_factories: Factories for creating step instances
            name: Optional name for the workflow
            description: Optional description for the workflow
            
        Returns:
            The instantiated workflow
            
        Raises:
            ValueError: If a step type is not found in the factories
        """
        # Create the workflow
        workflow = AdaptiveWorkflow(
            workflow_id=workflow_id,
            name=name or self.name,
            description=description or self.description,
        )
        
        # Create and add the steps
        step_instances: Dict[str, WorkflowStep] = {}
        for step_def in self.steps:
            step_type = step_def["step_type"]
            if step_type not in step_factories:
                raise ValueError(f"Step type '{step_type}' not found in factories")
            
            factory = step_factories[step_type]
            step = factory(step_def["step_id"], **step_def["config"])
            workflow.add_step(step)
            step_instances[step_def["step_id"]] = step
        
        # Add the dependencies
        for dep_def in self.dependencies:
            workflow.add_dependency(
                step_name=dep_def["step_id"],
                depends_on=dep_def["depends_on"],
                conditions=dep_def.get("conditions"),
            )
        
        return workflow
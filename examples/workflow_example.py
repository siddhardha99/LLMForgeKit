#!/usr/bin/env python3
"""Example using the Adaptive Agentic Choreographer with OpenAI."""

import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llmforgekit.core.config import LLMForgeKitConfig
from llmforgekit.core.errors import WorkflowError
from llmforgekit.core.logging import setup_logging
from llmforgekit.services.llm import OpenAIProvider
from llmforgekit.services.prompt import StringPromptTemplate
from llmforgekit.services.parser import JSONOutputParser, KeyValueParser
from llmforgekit.services.workflow.agent import LLMAgent
from llmforgekit.services.workflow.steps import LLMStep, ToolStep
from llmforgekit.services.workflow.engine import AdaptiveWorkflow, WorkflowEngine


# Define a simple tool for demonstration
class WebSearchTool:
    """Mock web search tool."""
    
    @property
    def name(self) -> str:
        """Get the name of the tool."""
        return "web_search"
    
    @property
    def description(self) -> str:
        """Get the description of the tool."""
        return "Searches the web for information"
    
    def execute(self, query: str) -> Dict[str, Any]:
        """Execute the tool with the provided parameters."""
        print(f"Performing web search for: {query}")
        # In a real implementation, this would perform an actual search
        # For this example, we return mock results
        return {
            "results": [
                {
                    "title": "Search Result 1",
                    "snippet": f"This is information about {query}...",
                    "url": f"https://example.com/1?q={query}"
                },
                {
                    "title": "Search Result 2",
                    "snippet": f"More details about {query}...",
                    "url": f"https://example.com/2?q={query}"
                }
            ]
        }


def main():
    """Run the workflow example."""
    # Set up logging
    logger = setup_logging(log_level="INFO")
    
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY=your_api_key")
        return 1
    
    # Create config and LLM provider
    config = LLMForgeKitConfig(openai_api_key=api_key)
    llm = OpenAIProvider(config=config, model="gpt-3.5-turbo")
    
    # Create parsers
    json_parser = JSONOutputParser(extract_json=True)
    kv_parser = KeyValueParser()
    
    # Create web search tool
    search_tool = WebSearchTool()
    
    print("\n=== Building Question Answering Workflow ===")
    
    # Create workflow steps
    
    # Step 1: Analyze the question
    analyze_template = StringPromptTemplate(
        """
        Analyze the following question and identify key search terms:
        
        Question: $question
        
        Respond in JSON format with the following structure:
        {
            "key_terms": ["term1", "term2", ...],
            "search_query": "optimized search query"
        }
        """
    )
    
    analyze_step = LLMStep(
        step_id="analyze_question",
        llm_provider=llm,
        prompt_template=analyze_template,
        output_parser=json_parser,
        prompt_context_keys=["question"],
    )
    
    # Step 2: Perform web search
    search_step = ToolStep(
        step_id="web_search",
        tool=search_tool,
        tool_params_map={"query": "output_analyze_question.search_query"},
    )
    
    # Step 3: Generate answer
    answer_template = StringPromptTemplate(
        """
        Based on the following search results, please answer the original question.
        
        Question: $question
        
        Search Results:
        $search_results
        
        Provide a comprehensive answer using the information from the search results.
        If the search results don't contain enough information, state what is missing.
        """
    )
    
    answer_step = LLMStep(
        step_id="generate_answer",
        llm_provider=llm,
        prompt_template=answer_template,
        prompt_context_keys=["question", "search_results"],
    )
    
    # Build the workflow
    workflow = AdaptiveWorkflow(
        workflow_id="question_answering_flow",
        name="Question Answering Workflow",
        description="A workflow that answers questions by searching the web",
    )
    
    # Add steps
    workflow.add_step(analyze_step)
    workflow.add_step(search_step)
    workflow.add_step(answer_step)
    
    # Add dependencies
    workflow.add_dependency(step_name="web_search", depends_on="analyze_question")
    workflow.add_dependency(step_name="generate_answer", depends_on="web_search")
    
    # Create workflow engine
    engine = WorkflowEngine()
    engine.register_workflow(workflow)
    
    # Run the workflow
    try:
        print("\n=== Running Question Answering Workflow ===")
        
        question = "What are the major features of Python 3.10?"
        
        initial_state = {
            "question": question,
        }
        
        print(f"Question: {question}")
        print("\nExecuting workflow...")
        
        result = engine.execute_workflow(
            workflow_id="question_answering_flow",
            initial_state=initial_state,
        )
        
        print("\n=== Workflow Execution Complete ===")
        print(f"\nFinal Answer:\n{result['output_generate_answer']}")
        
    except WorkflowError as e:
        print(f"Workflow execution failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
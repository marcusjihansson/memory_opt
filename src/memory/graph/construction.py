"""
Graph construction and workflow setup for LangGraph integration.
"""

from langgraph.graph import END, StateGraph

from ..dspy_wrapper import MemoryDSPyWrapper
from ..memory_manager import MemoryManager
from ..state import AgentState
from . import nodes
from .nodes import (
    agent_executor_node,
    memory_update_node,
    orchestrator_node,
    synthesizer_node,
)


def build_graph(redis_url: str, postgres_url: str, checkpointer=None):
    """Build graph with full memory system"""
    # Initialize memory manager
    nodes.memory_manager = MemoryManager(redis_url, postgres_url)
    # Initialize DSPy wrapper with default settings
    nodes.dspy_wrapper = MemoryDSPyWrapper(nodes.memory_manager)

    # Build graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("agent_executor", agent_executor_node)
    workflow.add_node("memory_update", memory_update_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Define flow
    workflow.set_entry_point("orchestrator")
    workflow.add_edge("orchestrator", "agent_executor")
    workflow.add_edge("agent_executor", "memory_update")
    workflow.add_edge("memory_update", "synthesizer")
    workflow.add_edge("synthesizer", END)

    # Use checkpointer if provided
    if checkpointer:
        app = workflow.compile(checkpointer=checkpointer)
    else:
        app = workflow.compile()

    return app

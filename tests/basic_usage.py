"""
Basic usage example of the three-tier memory system.
"""

import hashlib
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from src.memory import MemoryManager
from src.memory.state import AgentState

# Load environment variables from .env file
load_dotenv()


def build_graph_with_postgres_checkpointer(
    postgres_url: str, memory_manager: MemoryManager
):
    """Build graph with PostgreSQL checkpointer for state persistence"""

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node(
        "orchestrator", lambda state: orchestrator_node(state, memory_manager)
    )
    workflow.add_node(
        "agent_executor", lambda state: agent_executor_node(state, memory_manager)
    )
    workflow.add_node(
        "memory_update", lambda state: memory_update_node(state, memory_manager)
    )

    # Set entry point
    workflow.set_entry_point("orchestrator")

    # Add edges
    workflow.add_edge("orchestrator", "agent_executor")
    workflow.add_edge("agent_executor", "memory_update")
    workflow.add_edge("memory_update", END)

    return workflow


def orchestrator_node(state: AgentState, memory_manager: MemoryManager) -> AgentState:
    """Orchestrator with multi-layer memory access"""
    current_message = state["messages"][-1]["content"]

    # Build context using all memory layers
    context = memory_manager.get_context_for_agent(
        state,
        include_historical=True,  # Include long-term memory for routing
    )

    # Route based on comprehensive context
    next_agent = route_with_memory(current_message, context, state)

    return {
        **state,
        "next_agent": next_agent,
        "turn_count": state.get("turn_count", 0) + 1,
    }


def agent_executor_node(state: AgentState, memory_manager: MemoryManager) -> AgentState:
    """Execute agent with cached results check"""
    # Check cache first
    query_hash = hashlib.md5(state["messages"][-1]["content"].encode()).hexdigest()
    cached_result = memory_manager.cached.get_cached_agent_result(
        state["session_id"], state["next_agent"], query_hash
    )

    if cached_result:
        print(f"[CACHE HIT] Using cached result for {state['next_agent']}")
        result = cached_result
    else:
        # Build fresh context
        context = memory_manager.get_context_for_agent(state)

        # Execute agent (mock)
        result = execute_agent(
            state["next_agent"], state["messages"][-1]["content"], context
        )

        # Cache the result
        memory_manager.cached.cache_agent_result(
            state["session_id"], state["next_agent"], query_hash, result
        )

        # Save to memory
        memory_manager.save_interaction(state, state["messages"][-1], result)

    agent_outputs = state.get("agent_outputs", {})
    agent_outputs[state["next_agent"]] = result

    return {**state, "agent_outputs": agent_outputs}


def memory_update_node(state: AgentState, memory_manager: MemoryManager) -> AgentState:
    """Update short-term memory and working memory"""
    # Update short-term memory and working memory
    # Update conversation history
    history, importance_scores = memory_manager.short_term.update_conversation_history(
        state, state["messages"][-1]
    )

    # Update working memory with agent results
    working_mem = memory_manager.short_term.update_working_memory(
        state,
        f"last_{state['next_agent']}_result",
        state["agent_outputs"].get(state["next_agent"]),
    )

    return {
        **state,
        "conversation_history": history,
        "working_memory": working_mem,
        "memory_importance_scores": importance_scores,
    }


def route_with_memory(query: str, context: str, state: AgentState) -> str:
    """Route using comprehensive memory context"""
    # Mock routing logic
    query_lower = query.lower()
    if "database" in query_lower:
        return "database"
    elif "api" in query_lower:
        return "api"
    else:
        return "document"


def execute_agent(agent_name: str, query: str, context: str) -> dict:
    """Mock agent execution"""
    return {
        "source": agent_name,
        "data": f"Results from {agent_name} for: {query}",
        "context_used": len(context) > 0,
    }


def main():
    """
    Demonstrates three-tier memory in action:
    - First query: Misses cache, queries fresh
    - Second query: Hits cache for fast response
    - Uses PostgreSQL for persistence across sessions
    """

    print("=" * 80)
    print("THREE-TIER MEMORY SYSTEM")
    print("=" * 80)
    print("Layer 1: Redis (Cache) - Fast, ephemeral")
    print("Layer 2: Graph State (Short-term) - Active conversation")
    print("Layer 3: PostgreSQL (Long-term) - Persistent, searchable")
    print("=" * 80)

    # Load from environment variables
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    POSTGRES_URL = os.getenv(
        "POSTGRES_URL", "postgresql://memory_user:memory_pass@localhost:5432/memorydb"
    )
    print(f"Using POSTGRES_URL: {POSTGRES_URL}")

    # Initialize memory manager
    memory_manager = MemoryManager(redis_url=REDIS_URL, postgres_url=POSTGRES_URL)

    workflow = build_graph_with_postgres_checkpointer(POSTGRES_URL, memory_manager)

    # Use PostgreSQL for checkpoint persistence
    with PostgresSaver.from_conn_string(POSTGRES_URL) as checkpointer:
        checkpointer.setup()
        app = workflow.compile(checkpointer=checkpointer)

        session_id = "session-abc-123"
        user_id = "user-456"

        config = {"configurable": {"thread_id": session_id}}

        initial_state = {
            "messages": [
                {"role": "user", "content": "Find database records for order 789"}
            ],
            "conversation_history": [],
            "working_memory": {},
            "session_id": session_id,
            "user_id": user_id,
            "turn_count": 0,
            "agent_outputs": {},
            "next_agent": "",
            "final_response": "",
            "last_consolidation_turn": 0,
            "memory_importance_scores": {},
        }

        result = app.invoke(initial_state, config)

        print(f"\nResult: {result.get('agent_outputs', {})}")
        print("\nMemory layers used:")
        print("  - Cache: New entry created")
        print("  - Short-term: Updated in graph state")
        print("  - Long-term: Stored in PostgreSQL")


if __name__ == "__main__":
    main()

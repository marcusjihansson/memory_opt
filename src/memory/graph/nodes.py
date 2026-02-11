"""
Graph nodes for LangGraph integration.
"""

import hashlib
from datetime import datetime

import dspy

from ..state import AgentState

# Initialize memory manager (in production, inject via dependency injection)
memory_manager = None  # Will be initialized in build_graph
dspy_wrapper = None  # DSPy wrapper with memory integration


# DSPy program for demo (can be customized per agent)
class SentimentSignature(dspy.Signature):
    """Analyze the sentiment of a text."""

    text = dspy.InputField(desc="The text to analyze")
    sentiment = dspy.OutputField(desc="The sentiment: positive, negative, or neutral")


class SentimentAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(SentimentSignature)

    def forward(self, text, memory_context=None, memory_context_text=None):
        result = self.analyze(text=text)
        return result.sentiment


def orchestrator_node(state: AgentState) -> AgentState:
    """Orchestrator with multi-layer memory and semantic search"""
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


def agent_executor_node(state: AgentState) -> AgentState:
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
        # Execute DSPy agent with memory
        sentiment = dspy_wrapper.execute_with_memory(
            SentimentAnalyzer, state, {"text": state["messages"][-1]["content"]}
        )
        result = {
            "source": state["next_agent"],
            "data": sentiment,
            "context_used": True,
            "timestamp": datetime.now().isoformat(),
        }

        # Cache the result
        memory_manager.cached.cache_agent_result(
            state["session_id"], state["next_agent"], query_hash, result
        )

        # Save to memory
        memory_manager.save_interaction(state, state["messages"][-1], result)

    agent_outputs = state.get("agent_outputs", {})
    agent_outputs[state["next_agent"]] = result

    return {**state, "agent_outputs": agent_outputs}


def memory_update_node(state: AgentState) -> AgentState:
    """Update short-term memory and check for consolidation"""
    current_message = state["messages"][-1]

    # Update conversation history with importance scoring
    history, importance_scores = memory_manager.short_term.update_conversation_history(
        state, current_message
    )

    # Add assistant response
    agent_output = state["agent_outputs"].get(state["next_agent"], {})
    assistant_msg = {
        "role": "assistant",
        "content": str(agent_output.get("data", "")),
        "id": f"msg_{state['turn_count']}_assistant",
    }
    history, importance_scores = memory_manager.short_term.update_conversation_history(
        {
            **state,
            "conversation_history": history,
            "memory_importance_scores": importance_scores,
        },
        assistant_msg,
    )

    # Save to persistent storage
    memory_manager.save_interaction(state, current_message, agent_output)

    # Update working memory
    working_mem = memory_manager.short_term.update_working_memory(
        state, f"last_{state['next_agent']}_result", agent_output
    )

    # Check if consolidation is needed
    updated_state = {
        **state,
        "conversation_history": history,
        "working_memory": working_mem,
        "memory_importance_scores": importance_scores,
    }

    should_consolidate = memory_manager.short_term.should_consolidate(updated_state)

    if should_consolidate:
        print(f"\n[CONSOLIDATION TRIGGERED] Turn {state['turn_count']}")
        memory_manager.consolidate_memories(updated_state)
        updated_state["last_consolidation_turn"] = state["turn_count"]

    return updated_state


def synthesizer_node(state: AgentState) -> AgentState:
    """Synthesize final response"""
    agent_output = state["agent_outputs"].get(state["next_agent"], {})

    response = f"""Results from {agent_output.get("source", "agent")}:

{agent_output.get("data", "No data")}

Context used: {agent_output.get("context_used", False)}
"""

    return {**state, "final_response": response}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def route_with_memory(query: str, context: str, state: AgentState) -> str:
    """Route using comprehensive memory context"""
    query_lower = query.lower()

    # Simple routing logic
    if "database" in query_lower or "sql" in query_lower:
        return "database"
    elif "api" in query_lower or "external" in query_lower:
        return "api"
    else:
        return "document"

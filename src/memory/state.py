"""
Type definitions and state management for the memory system.
"""

from operator import add
from typing import Annotated, Any, TypedDict

from .types import Memory


class AgentState(TypedDict):
    """Central state with multi-layer memory"""

    messages: Annotated[list[dict[str, str]], add]

    # SHORT-TERM MEMORY
    conversation_history: list[dict[str, str]]
    working_memory: dict[str, Any]

    # Memory metadata
    session_id: str
    user_id: str
    turn_count: int

    # Agent execution
    agent_outputs: dict[str, Any]
    next_agent: str
    final_response: str

    # Memory consolidation tracking
    last_consolidation_turn: int
    memory_importance_scores: dict[str, float]

    # META-MEMORY LAYER
    narrative_state: dict[str, Any]  # NarrativeState
    coherence_violations: list[str]
    meta_reasoning: str
    long_term_memory: list[Memory]

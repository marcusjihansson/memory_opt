"""
Type definitions and state management for the memory system.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add


class AgentState(TypedDict):
    """Central state with multi-layer memory"""

    messages: Annotated[List[Dict[str, str]], add]

    # SHORT-TERM MEMORY
    conversation_history: List[Dict[str, str]]
    working_memory: Dict[str, Any]

    # Memory metadata
    session_id: str
    user_id: str
    turn_count: int

    # Agent execution
    agent_outputs: Dict[str, Any]
    next_agent: str
    final_response: str

    # Memory consolidation tracking
    last_consolidation_turn: int
    memory_importance_scores: Dict[str, float]

    # META-MEMORY LAYER
    narrative_state: Dict[str, Any]  # NarrativeState
    coherence_violations: List[str]
    meta_reasoning: str
    long_term_memory: List[Dict[str, Any]]  # List of Memory dicts

"""
Multi-layer memory system for LangGraph applications.

This package provides a five-tier memory architecture:
0. Meta-Memory (System 3): Coherence validation and narrative identity
1. Cached Memory: Redis with distributed locking
2. Short-term Memory: Active conversation context
3. Long-term Memory: PostgreSQL with semantic search
4. RLM Layer: Recursive Language Model for intelligent memory exploration
"""

from .embedding import EmbeddingService
from .memory_manager import MemoryManager
from .meta_memory import MetaMemory
from .rlm_memory import (
    ProductionRLMMemory,
    RLMMemoryExplorer,
    RLMMemoryReasoner,
    RLMMemorySynthesizer,
    format_memories_for_rlm,
    parse_rlm_memories,
)
from .state import AgentState
from .types import GoalLevel, MemoryStability, MemoryType, NarrativeState

__all__ = [
    "AgentState",
    "EmbeddingService",
    "MemoryManager",
    "MetaMemory",
    "MemoryType",
    "MemoryStability",
    "GoalLevel",
    "NarrativeState",
    # RLM modules
    "RLMMemoryExplorer",
    "RLMMemoryReasoner",
    "RLMMemorySynthesizer",
    "ProductionRLMMemory",
    "format_memories_for_rlm",
    "parse_rlm_memories",
]

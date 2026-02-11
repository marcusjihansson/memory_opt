"""
Shared type definitions and data structures for memory system.
This module eliminates circular dependencies by providing common types
that can be imported by all memory components without coupling.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypedDict

import numpy as np

# ============================================================================
# ENUMS
# ============================================================================


class MemoryType(Enum):
    """Types of memories stored in the system"""

    FACT = "fact"
    GOAL = "goal"
    CONSTRAINT = "constraint"
    COMMITMENT = "commitment"
    IDENTITY = "identity"


class MemoryStability(Enum):
    """Stability levels for memory retention"""

    VOLATILE = "volatile"  # May change frequently
    STABLE = "stable"  # Reliable but not core
    CORE = "core"  # Fundamental identity/constraints


class GoalLevel(Enum):
    """Hierarchical goal levels from Sophia III V2"""

    LIFE = "life"  # Top-level persistent goals
    SESSION = "session"  # Multi-turn conversation goals
    TACTICAL = "tactical"  # Single-turn immediate actions


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class Memory:
    """Enhanced memory with meta-cognitive attributes and embeddings"""

    content: str
    type: MemoryType
    confidence: float
    importance: float
    stability: MemoryStability
    source: str
    created_at: str
    last_verified: str
    session_id: str
    tags: list[str]
    reasoning: str
    embedding: np.ndarray | None = None
    related_memory_ids: list[str] = field(default_factory=list)
    contradicts_memory_ids: list[str] = field(default_factory=list)


@dataclass
class HierarchicalGoal:
    """
    Hierarchical goal structure
    Each goal knows its parent and must support parent goals
    """

    id: str
    content: str
    level: GoalLevel
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    progress: float = 0.0
    completed: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    importance: float = 0.5
    embedding: np.ndarray | None = None
    supports_goals: list[str] = field(default_factory=list)
    blocks_goals: list[str] = field(default_factory=list)


# ============================================================================
# TYPED DICTIONARIES
# ============================================================================


class NarrativeState(TypedDict):
    """The persistent narrative identity - meta-memory's core"""

    identity: dict[str, Any]
    goal_hierarchy: dict[str, HierarchicalGoal]
    constraints: list[str]
    commitments: list[str]
    session_arc: dict[str, Any]
    coherence_score: float

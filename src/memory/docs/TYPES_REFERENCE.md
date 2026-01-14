# Types Reference

## Overview

The `types.py` module provides shared type definitions used across the memory system, eliminating circular dependencies by providing common types that can be imported by all memory components without coupling.

## Core Principles

- **Single Source of Truth**: All memory-related types defined in one place
- **Zero Circular Dependencies**: Types can be imported anywhere without issues
- **Professional Type Safety**: Full type hints for IDE support and static analysis
- **Backward Compatibility**: Clean API surface for library users

## Enums

### MemoryType
Defines the type of information stored in memory.

```python
class MemoryType(Enum):
    FACT = "fact"           # Factual information
    GOAL = "goal"           # User or system objectives
    CONSTRAINT = "constraint"  # Behavioral boundaries
    COMMITMENT = "commitment"  # Promises made
    IDENTITY = "identity"   # Core user characteristics
```

**Usage**:
```python
from memory.types import MemoryType

memory = Memory(
    content="User prefers Python",
    type=MemoryType.FACT,
    confidence=0.9
)
```

### MemoryStability
Indicates how likely memory content is to change over time.

```python
class MemoryStability(Enum):
    VOLATILE = "volatile"   # May change frequently (e.g., temporary preferences)
    STABLE = "stable"       # Reliable but not fundamental (e.g., tool preferences)
    CORE = "core"           # Fundamental, rarely changes (e.g., dietary restrictions)
```

**Usage**:
```python
# Core identity memories
identity_memory = Memory(
    content="User is vegetarian",
    stability=MemoryStability.CORE
)

# Temporary preferences
temp_memory = Memory(
    content="User wants quick results today",
    stability=MemoryStability.VOLATILE
)
```

### GoalLevel
Hierarchical levels for goal management (Sophia III V2).

```python
class GoalLevel(Enum):
    LIFE = "life"           # Long-term persistent goals
    SESSION = "session"     # Multi-turn conversation goals
    TACTICAL = "tactical"   # Single-turn immediate actions
```

**Hierarchy**:
```
LIFE GOALS (fundamental)
    │
    ├─ "Help user succeed in AI development"
    └─ "Maintain helpful personality"
         ↓ must support

SESSION GOALS (scoped)
    │
    ├─ "Explain memory system architecture"
    └─ "Provide practical examples"
         ↓ must support

TACTICAL GOALS (immediate)
    │
    ├─ "Generate code example"
    └─ "Answer specific question"
```

## Data Classes

### Memory
The fundamental unit of stored information with meta-cognitive attributes.

```python
@dataclass
class Memory:
    content: str                    # The actual memory content
    type: MemoryType               # What kind of memory this is
    confidence: float              # How certain we are (0.0 to 1.0)
    importance: float              # How critical this memory is (0.0 to 1.0)
    stability: MemoryStability     # How likely to change
    source: str                    # Where this memory came from
    created_at: str               # ISO format timestamp
    last_verified: str            # When last confirmed
    session_id: str               # Which conversation session
    tags: List[str]               # Categorization tags
    reasoning: str                # Why this memory is important
    embedding: Optional[np.ndarray] = None  # Vector representation
    related_memory_ids: List[str] = field(default_factory=list)
    contradicts_memory_ids: List[str] = field(default_factory=list)
```

**Key Fields**:

- **Meta-Attributes**: `confidence`, `importance`, `stability` enable intelligent memory management
- **Provenance**: `source`, `created_at`, `session_id` track memory origin
- **Relationships**: `related_memory_ids`, `contradicts_memory_ids` enable semantic linking
- **Embeddings**: Vector representation for semantic search and similarity

**Usage**:
```python
from memory import MetaMemory
from memory.types import MemoryType, MemoryStability

meta_memory = MetaMemory(embedding_service)

memory = meta_memory.add_memory(
    content="User prefers functional programming",
    memory_type=MemoryType.FACT,
    confidence=0.85,
    importance=0.7,
    stability=MemoryStability.STABLE,
    tags=["programming", "preference"],
    reasoning="User mentioned preferring immutable data structures"
)
```

### HierarchicalGoal
Represents goals in the three-level hierarchy with progress tracking.

```python
@dataclass
class HierarchicalGoal:
    id: str                        # Unique identifier
    content: str                   # Goal description
    level: GoalLevel              # LIFE, SESSION, or TACTICAL
    parent_id: Optional[str] = None  # Parent goal ID (for SESSION/Tactical)
    children_ids: List[str] = field(default_factory=list)
    progress: float = 0.0          # Completion percentage (0.0 to 1.0)
    completed: bool = False        # Whether goal is finished
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    importance: float = 0.5        # Priority weighting
    embedding: Optional[np.ndarray] = None
    supports_goals: List[str] = field(default_factory=list)
    blocks_goals: List[str] = field(default_factory=list)
```

**Key Fields**:

- **Hierarchy**: `parent_id`, `children_ids` maintain tree structure
- **Progress**: `progress`, `completed` track advancement
- **Relationships**: `supports_goals`, `blocks_goals` handle goal interactions
- **Metadata**: `importance`, `embedding` for intelligent management

**Usage**:
```python
from memory.types import GoalLevel, HierarchicalGoal

# Life goal
life_goal = HierarchicalGoal(
    id="life_help_user",
    content="Help user become proficient in AI development",
    level=GoalLevel.LIFE,
    importance=1.0
)

# Session goal supporting life goal
session_goal = HierarchicalGoal(
    id="session_explain_memory",
    content="Explain MetaMemory system architecture",
    level=GoalLevel.SESSION,
    parent_id=life_goal.id,
    importance=0.8
)
```

## Typed Dictionaries

### NarrativeState
The persistent identity and state of the meta-cognitive agent.

```python
class NarrativeState(TypedDict):
    identity: Dict[str, Any]        # Agent's personality and values
    goal_hierarchy: Dict[str, HierarchicalGoal]  # All active goals
    constraints: List[str]         # Behavioral boundaries
    commitments: List[str]         # Promises made
    session_arc: Dict[str, Any]    # Current conversation phase
    coherence_score: float         # Self-assessed consistency (0.0 to 1.0)
```

**Structure**:
```python
narrative_state: NarrativeState = {
    "identity": {
        "role": "AI assistant with persistent memory",
        "values": ["helpful", "accurate", "consistent"],
        "capabilities": ["memory_management", "goal_tracking"],
        "limitations": ["cannot_access_external_apis"]
    },
    "goal_hierarchy": {
        "goal_123": HierarchicalGoal(...),
        "goal_456": HierarchicalGoal(...)
    },
    "constraints": [
        "maintain_coherence",
        "respect_user_preferences",
        "avoid_harmful_actions"
    ],
    "commitments": [
        "provide_accurate_information",
        "maintain_user_privacy"
    ],
    "session_arc": {
        "phase": "explanation",
        "turn_count": 15,
        "progress": 0.6
    },
    "coherence_score": 0.95
}
```

**Usage**:
```python
from memory.types import NarrativeState
from memory.meta_memory import create_default_narrative_state

# Create default narrative state
narrative_state: NarrativeState = create_default_narrative_state()

# Access components
identity = narrative_state["identity"]
goals = narrative_state["goal_hierarchy"]
coherence = narrative_state["coherence_score"]
```

## Import Patterns

### Recommended Imports
```python
# Import all types at once
from memory.types import (
    MemoryType,
    MemoryStability,
    GoalLevel,
    Memory,
    HierarchicalGoal,
    NarrativeState
)

# Or import the entire module
from memory import types
memory_type = types.MemoryType.FACT
```

### Avoid Deprecated Patterns
```python
# DON'T DO THIS - causes circular dependencies
from memory.meta_memory import MemoryType  # Old way, don't use
```

## Type Safety Benefits

### IDE Support
```python
# Full autocompletion and type checking
memory: Memory = create_memory(...)
goal: HierarchicalGoal = create_goal(...)

# Type errors caught at development time
memory.confidence = "high"  # TypeError: expected float, got str
```

### Static Analysis
```bash
# MyPy catches type issues
mypy src/memory/ --strict

# Example error:
# error: Incompatible types in assignment (expression has type "str", variable has type "float")
```

### Runtime Safety
```python
# Type hints help with debugging
def validate_memory(memory: Memory) -> bool:
    assert 0.0 <= memory.confidence <= 1.0, "Confidence must be between 0 and 1"
    assert memory.type in MemoryType, "Invalid memory type"
    return True
```

## Migration Guide

### From v1 (without types module)
```python
# Before
from memory.meta_memory import MemoryType, Memory, NarrativeState

# After
from memory.types import MemoryType, Memory, NarrativeState
```

### Type Checking Setup
```bash
# Install type checking tools
pip install mypy

# Run type checks
mypy src/memory/

# Configure strict checking
echo '[mypy-memory.*]
strict = True' > mypy.ini
```

## Testing Types

```python
import pytest
from memory.types import MemoryType, Memory, GoalLevel

def test_memory_creation():
    """Test Memory dataclass creation"""
    memory = Memory(
        content="Test memory",
        type=MemoryType.FACT,
        confidence=0.9,
        importance=0.7,
        stability=MemoryStability.STABLE,
        source="test",
        created_at="2024-01-01T00:00:00",
        last_verified="2024-01-01T00:00:00",
        session_id="session_123",
        tags=["test"],
        reasoning="Test memory creation"
    )

    assert memory.content == "Test memory"
    assert memory.type == MemoryType.FACT
    assert 0.0 <= memory.confidence <= 1.0

def test_goal_hierarchy():
    """Test HierarchicalGoal relationships"""
    parent = HierarchicalGoal(
        id="parent",
        content="Parent goal",
        level=GoalLevel.LIFE
    )

    child = HierarchicalGoal(
        id="child",
        content="Child goal",
        level=GoalLevel.SESSION,
        parent_id=parent.id
    )

    assert child.parent_id == parent.id
    assert parent.level == GoalLevel.LIFE
    assert child.level == GoalLevel.SESSION
```

## Related Documentation

- [MetaMemory Guide](META_MEMORY.md) - How types are used in MetaMemory
- [Architecture](ARCHITECTURE.md) - Why types module eliminates circular dependencies
- [Type Safety](TYPE_SAFETY.md) - Professional type checking practices
- [Testing Guide](TESTING.md) - Testing with type safety
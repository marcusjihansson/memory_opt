# MetaMemory - System 3 Meta-Cognitive Layer

## Overview

MetaMemory implements Sophia III's **System 3 architecture**, adding a meta-cognitive layer above traditional memory systems. While traditional memory systems answer "what do I remember?", MetaMemory answers "what do these memories mean for who I am and what I'm trying to achieve?"

## Architecture

### System 3 Components

```
┌─────────────────────────────────────────────┐
│          SYSTEM 3: META-MEMORY              │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │   Narrative State Tracker           │   │
│  │   - Identity (values, role)         │   │
│  │   - Goal Hierarchy (life/session)   │   │
│  │   - Constraints & Commitments       │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │   Coherence Validator               │   │
│  │   - Goal alignment check            │   │
│  │   - Constraint violation detection  │   │
│  │   - Identity consistency            │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │   Semantic Memory Engine (V2)       │   │
│  │   - Vector embeddings               │   │
│  │   - Cross-domain relationships      │   │
│  │   - Contradiction detection         │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

## Key Features

### 1. Narrative Identity
Maintains persistent agent personality across sessions:

```python
narrative_state = {
    "identity": {
        "role": "AI assistant with persistent memory",
        "values": ["helpful", "accurate", "consistent"]
    },
    "goal_hierarchy": {...},
    "constraints": ["maintain coherence", "respect user preferences"],
    "commitments": [...],
    "session_arc": {"phase": "active", "turn_count": 0},
    "coherence_score": 1.0
}
```

### 2. Coherence Validation
Pre-action consistency checking:

```python
result = meta_memory.check_coherence(
    proposed_action="Recommend a steak restaurant",
    narrative_state=narrative_state
)

if result['should_proceed']:
    print("Action is coherent")
else:
    print(f"Blocked: {result['violations']}")
```

### 3. Hierarchical Goal Management
Three-level goal structure ensuring alignment:

```
LIFE GOALS (persistent, fundamental)
    │
    ├─ "Help user build robust AI systems"
    └─ "Ensure user long-term success"
         ↓ must support

SESSION GOALS (multi-turn, scoped)
    │
    ├─ "Explain MetaMemory architecture"
    └─ "Provide implementation examples"
         ↓ must support

TACTICAL GOALS (immediate actions)
    │
    ├─ "Write code example"
    └─ "Document key concepts"
```

### 4. Semantic Conflict Detection
Automatically identifies contradictory information:

```python
# Detect conflicts between memories
conflict = meta_memory.detect_memory_conflicts(
    new_memory, existing_memories
)

if conflict['has_conflict']:
    print(f"Conflict: {conflict['reasoning']}")
    # Handle resolution
```

## Usage Examples

### Basic Initialization

```python
from memory import MetaMemory, EmbeddingService

# Initialize services
embedding_service = EmbeddingService()
meta_memory = MetaMemory(embedding_service)

# Create default narrative state
from memory.meta_memory import create_default_narrative_state
narrative_state = create_default_narrative_state()
```

### Adding Memories with Meta-Attributes

```python
from memory.types import MemoryType, MemoryStability

memory = meta_memory.add_memory(
    content="User prefers Python for data science",
    memory_type=MemoryType.FACT,
    confidence=0.9,
    stability=MemoryStability.STABLE,
    tags=["preference", "programming", "data-science"],
    reasoning="Explicitly stated by user"
)
```

### Coherence Checking in LangGraph

```python
def meta_memory_check_node(state):
    """LangGraph node for coherence validation"""
    meta_memory = MetaMemory(embedding_service)
    proposed_action = state.get("next_agent", "")

    coherence_result = meta_memory.check_coherence(
        proposed_action, state["narrative_state"]
    )

    if not coherence_result["should_proceed"]:
        state["next_agent"] = "replan"
        state["final_response"] = f"Blocked: {coherence_result['violations']}"

    return state
```

### Goal Hierarchy Management

```python
from memory.types import GoalLevel

# Create hierarchical goals
life_goal = meta_memory.goal_manager.create_goal(
    "Help user become proficient in AI development",
    GoalLevel.LIFE,
    importance=1.0
)

session_goal = meta_memory.goal_manager.create_goal(
    "Teach MetaMemory concepts",
    GoalLevel.SESSION,
    parent_id=life_goal.id,
    importance=0.8
)

# Add to narrative state
narrative_state["goal_hierarchy"][life_goal.id] = life_goal
narrative_state["goal_hierarchy"][session_goal.id] = session_goal
```

## Integration Patterns

### With MemoryManager

```python
from memory import MemoryManager

# MetaMemory is automatically integrated
memory_manager = MemoryManager(redis_url, postgres_url)

# Access MetaMemory instance
meta_memory = memory_manager.meta_memory

# Narrative state is managed automatically
narrative_state = memory_manager.load_narrative_state(agent_state)
```

### Standalone Usage

```python
# For testing or custom integrations
from memory import MetaMemory, EmbeddingService
from memory.types import create_default_narrative_state

embedding_service = EmbeddingService()
meta_memory = MetaMemory(embedding_service)
narrative_state = create_default_narrative_state()

# Use independently of MemoryManager
```

## Advanced Features

### Memory Attribution
Each memory includes reasoning and context:

```python
memory = Memory(
    content="User prefers open-source tools",
    reasoning="User mentioned avoiding proprietary solutions",
    confidence=0.8,
    source="conversation_analysis",
    tags=["preference", "open-source"]
)
```

### Cross-Domain Relationships
Memories are connected semantically, not just by type:

```python
# These memories are related even though different types:
memory1 = Memory(content="User is vegetarian", type=MemoryType.IDENTITY)
memory2 = Memory(content="User likes salads", type=MemoryType.FACT)

# MetaMemory detects the semantic relationship
related = meta_memory.find_related_memories(memory1, [memory2])
```

### Progress Tracking
Goals track completion and propagate progress upward:

```python
# Mark tactical goal complete
meta_memory.goal_manager.propagate_progress(
    narrative_state["goal_hierarchy"],
    tactical_goal_id,
    progress_delta=1.0
)

# Session goal progress updates automatically
# Life goal receives incremental progress
```

## Performance Considerations

### Memory Consolidation
- Automatic pruning of low-importance memories
- Semantic deduplication of similar content
- Importance-based retention policies

### Caching Strategies
- Narrative state cached in Redis
- Embedding computations cached
- Coherence validation results cached per session

### Scalability
- Vector similarity search optimized with pgvector
- Hierarchical goal queries bounded by depth
- Memory attribution metadata indexed for fast retrieval

## Troubleshooting

### Common Issues

**Import Errors**: Ensure using shared `types` module:
```python
# Correct
from memory.types import MemoryType, GoalLevel

# Incorrect (old way)
from memory.meta_memory import MemoryType, GoalLevel
```

**Coherence Always Failing**: Check narrative state initialization:
```python
# Ensure narrative state is properly loaded
narrative_state = memory_manager.load_narrative_state(agent_state)
agent_state["narrative_state"] = narrative_state
```

**Memory Not Persisting**: Check LongTermMemory integration:
```python
# MetaMemory requires LongTermMemory for persistence
meta_memory = MetaMemory(embedding_service, long_term_memory_instance)
```

### Debugging Tools

```python
# Check narrative state coherence
print(f"Coherence score: {narrative_state['coherence_score']}")

# Inspect goal hierarchy
for goal_id, goal in narrative_state["goal_hierarchy"].items():
    print(f"{goal.level.value}: {goal.content} ({goal.progress:.1f})")

# Validate memory structure
for memory in memories:
    assert hasattr(memory, 'embedding'), "Memory missing embedding"
    assert memory.confidence >= 0.0 and memory.confidence <= 1.0
```

## Migration Guide

See [`MIGRATION.md`](../MIGRATION.md) for upgrading from versions without MetaMemory.

## Related Documentation

- [Architecture Overview](../docs/ARCHITECTURE.md) - Design principles
- [Types Reference](../docs/TYPES_REFERENCE.md) - Complete type definitions
- [Testing Guide](../docs/TESTING.md) - Testing MetaMemory
- [Type Safety](../docs/TYPE_SAFETY.md) - Type checking practices
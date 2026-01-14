Query arrives
    ↓
┌─────────────────────────────────────────┐
│ SYSTEM 3: META-MEMORY          │ ← Narrative coherence check
│                                 │
│  ┌─────────────────────────────┐  │
│  │ Coherence Validator       │  │
│  │ - Goal alignment check   │  │
│  │ - Constraint detection    │  │
│  │ - Identity consistency    │  │
│  └─────────────────────────────┘  │
└─────────────────────────────────────────┘
           ↓ Coherent?
           ↓ Yes
┌─────────────────────┐
│ Distributed Lock │ ← Prevents concurrent access
└─────────────────────┘
↓
1. Check Redis cache (fastest) ← Hit? Return immediately
   ↓ Miss
2. Use Graph State (short-term)
   ↓
3. Query PostgreSQL (if historical context needed)
   ↓
   Execute agent with combined context
   ↓
   Save to all layers:

- MetaMemory: narrative state updates
- Redis: cache for next call
- Graph State: updated automatically
- PostgreSQL: permanent record + embeddings

## Meta-Memory Benefits

- **Identity Consistency**: Maintains agent personality across sessions
- **Coherence Prevention**: Blocks inconsistent actions before execution
- **Goal Alignment**: Ensures tactical actions support higher-level objectives
- **Conflict Detection**: Automatically identifies semantic contradictions
- **Narrative Continuity**: Preserves conversation red thread

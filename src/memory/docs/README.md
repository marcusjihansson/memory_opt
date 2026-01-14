with memory_manager.cached.distributed_lock(session_id): # Check cache # Execute agent # Update cache # All atomic - no race conditions

```

## 🏗️ Architecture Flow
```

Request arrives
↓
┌─────────────────────┐
│ Distributed Lock │ ← Prevents concurrent access
└─────────────────────┘
↓
┌─────────────────────┐
│ Redis Cache Check │ ← Layer 1: Instant (< 1ms)
└─────────────────────┘
↓ Cache miss
┌─────────────────────┐
│ Short-term Memory │ ← Layer 2: Current context
└─────────────────────┘
↓
┌─────────────────────┐
│ Semantic Search │ ← Layer 3: Historical relevance
│ (PostgreSQL) │ Vector similarity search
└─────────────────────┘
↓
Execute Agent with full context
↓
Save to all layers:

- Redis: cache for speed
- State: working memory
- PostgreSQL: persistent + embeddings
  ↓
  Check consolidation triggers:
- Turn threshold (15)
- Importance threshold (5 high)
- Size threshold (18 messages)
  ↓
  If triggered → Consolidate:
- Move important messages to long-term
- Prune short-term memory
- Generate session summary

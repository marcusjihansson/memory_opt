# Architecture Design

## Overview

This document explains the architectural design decisions that make the memory system professional, maintainable, and scalable. The architecture follows SOLID principles and modern Python practices.

## Core Design Principles

### 1. Single Responsibility Principle
Each module has one clear responsibility:

- **`types.py`**: Type definitions and data structures
- **`meta_memory.py`**: Meta-cognitive reasoning logic
- **`memory_manager.py`**: Coordination and orchestration
- **`long_term_memory.py`**: Persistence and retrieval
- **`embedding.py`**: Vector operations and similarity

### 2. Dependency Inversion Principle
High-level modules don't depend on low-level modules:

```python
# ✅ Good: MetaMemory depends on abstraction (EmbeddingService interface)
class MetaMemory:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_engine = EmbeddingEngine(embedding_service)

# ❌ Bad: Direct dependency on implementation
class MetaMemory:
    def __init__(self):
        self.embedding_service = OpenRouterEmbeddingService()  # Hardcoded
```

### 3. Open/Closed Principle
Modules are open for extension, closed for modification:

```python
# New embedding services can be added without changing MetaMemory
class MetaMemory:
    def __init__(self, embedding_service: EmbeddingService):
        # Works with any implementation
        self.embedding_engine = EmbeddingEngine(embedding_service)
```

## Circular Dependency Resolution

### The Problem
Traditional approaches create circular imports:

```
MemoryManager imports MetaMemory
MetaMemory imports LongTermMemory
LongTermMemory imports MemoryManager (for coordination)
    ↑ Circular dependency ↑
```

### The Solution
**Types Module + Dependency Injection**

#### Step 1: Extract Shared Types
```python
# src/memory/types.py - Zero dependencies, imported everywhere
from dataclasses import dataclass
from enum import Enum

class MemoryType(Enum):
    FACT = "fact"
    # ...
```

#### Step 2: Dependency Injection
```python
# MetaMemory doesn't import LongTermMemory directly
class MetaMemory:
    def __init__(self, embedding_service, long_term_memory=None):
        self.long_term_memory = long_term_memory  # Injected dependency

# MemoryManager provides the dependency
class MemoryManager:
    def __init__(self, redis_url, postgres_url):
        self.long_term = LongTermMemory(...)
        self.meta_memory = MetaMemory(embedding_service, self.long_term)
```

#### Step 3: Duck Typing for Optional Dependencies
```python
# No type restrictions, runtime checking
if self.long_term_memory and hasattr(self.long_term_memory, 'store_memory'):
    self.long_term_memory.store_memory(...)
```

### Benefits of This Approach

1. **Zero Circular Dependencies**: Clean import graph
2. **Testability**: Easy to mock dependencies
3. **Flexibility**: MetaMemory can work standalone or with persistence
4. **Maintainability**: Clear separation of concerns

## Module Architecture

### Package Structure
```
src/memory/
├── types.py           # Shared type definitions (foundation)
├── embedding.py       # Vector operations (infrastructure)
├── meta_memory.py     # Meta-cognitive logic (core business)
├── long_term_memory.py # Persistence (infrastructure)
├── memory_manager.py  # Coordination (application)
└── __init__.py       # Public API (interface)
```

### Import Flow
```
types.py ←───── All modules can import freely
    ↓
embedding.py ←── Infrastructure layer
    ↓
meta_memory.py ← Core business logic (uses infrastructure)
    ↓
long_term_memory.py ← Infrastructure layer
    ↓
memory_manager.py ← Application coordination
    ↓
__init__.py ← Public API surface
```

## Data Flow Architecture

### Memory Creation Flow
```
User Input
    ↓
MemoryManager.save_interaction()
    ↓
MetaMemory.add_memory() ← Creates Memory object with embedding
    ↓
LongTermMemory.store_memory() ← Persists with vector
    ↓
PostgreSQL (with pgvector) ← Semantic search enabled
```

### Query Flow
```
User Query
    ↓
MemoryManager.get_context_for_agent()
    ↓
1. Redis Cache Check (fast)
2. Short-term State (current)
3. LongTermMemory.semantic_search() (historical)
    ↓
MetaMemory Coherence Check ← Validates against narrative state
    ↓
Combined Context Returned
```

## Performance Architecture

### Caching Strategy
```
Level 1: Redis (instant, <1ms)
    - Agent results
    - Conversation summaries
    - Narrative state snapshots

Level 2: Memory State (fast, in-process)
    - Current conversation context
    - Working memory

Level 3: PostgreSQL (semantic, <100ms)
    - Historical memories
    - Vector similarity search
    - User profiles
```

### Optimization Techniques

#### 1. Batch Embeddings
```python
# Process multiple texts at once
embeddings = embedding_service.embed_batch(texts)

# Reduces API calls and latency
```

#### 2. Selective Persistence
```python
# Only persist important memories
if memory.importance > 0.6:
    long_term_memory.store_memory(memory)
```

#### 3. Lazy Loading
```python
# Load narrative state on demand
@property
def narrative_state(self):
    if not self._narrative_state:
        self._narrative_state = self.load_narrative_state()
    return self._narrative_state
```

## Error Handling Architecture

### Graceful Degradation
```python
# Embedding service failure → fallback to mock
try:
    embedding = self.embedding_service.embed_text(text)
except Exception:
    embedding = self.mock_embedding(text)
```

### Circuit Breaker Pattern
```python
# Database failure → continue without persistence
if self.long_term_memory:
    try:
        self.long_term_memory.store_memory(memory)
    except DatabaseError:
        logger.warning("Persistence failed, continuing without")
```

### Validation Layers
```python
# Input validation
def add_memory(self, content: str, ...) -> Memory:
    if not content or len(content) > MAX_LENGTH:
        raise ValueError("Invalid memory content")

    # Type validation
    if not isinstance(memory_type, MemoryType):
        raise TypeError("Invalid memory type")

    # Business logic validation
    if not (0.0 <= confidence <= 1.0):
        raise ValueError("Confidence must be between 0 and 1")
```

## Testing Architecture

### Unit Test Isolation
```python
# Test MetaMemory in isolation
def test_meta_memory_coherence():
    meta_memory = MetaMemory(embedding_service=mock_service)
    # No database dependencies
```

### Integration Test Scope
```python
# Test full system with real databases
@pytest.mark.integration
def test_full_memory_workflow():
    memory_manager = MemoryManager(redis_url, postgres_url)
    # End-to-end testing
```

### Test Fixtures
```python
@pytest.fixture
def embedding_service():
    return EmbeddingService()

@pytest.fixture
def narrative_state():
    return create_default_narrative_state()

@pytest.fixture
def agent_state(narrative_state):
    return AgentState(narrative_state=narrative_state, ...)
```

## Security Architecture

### Input Sanitization
```python
# Validate all user inputs
def sanitize_memory_content(content: str) -> str:
    # Remove potentially harmful content
    # Limit length
    # Validate encoding
    return cleaned_content
```

### Access Control
```python
# User isolation
def get_user_memories(user_id: str) -> List[Memory]:
    # Only return memories for specific user
    return self.db.query("SELECT * FROM memories WHERE user_id = %s", user_id)
```

### Privacy Protection
```python
# Data minimization
def store_memory(memory: Memory):
    # Only store necessary fields
    # Hash sensitive identifiers
    # Implement retention policies
```

## Scalability Architecture

### Horizontal Scaling
```
Load Balancer
    ↓
Multiple MemoryManager instances
    ↓
Shared Redis Cluster (cache)
    ↓
PostgreSQL Cluster (persistence)
```

### Database Optimization
```sql
-- Optimized indexes for memory queries
CREATE INDEX idx_memories_user_session ON memories(user_id, session_id);
CREATE INDEX idx_memories_embedding ON memories USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_memories_importance ON memories(importance DESC);
```

### Connection Pooling
```python
# Efficient database connection management
self.connection_pool = psycopg_pool.ConnectionPool(
    conninfo=self.conn_string,
    min_size=1,
    max_size=10
)
```

## Deployment Architecture

### Containerization
```dockerfile
# Multi-stage build for optimization
FROM python:3.11-slim as builder
# Install dependencies
COPY pyproject.toml .
RUN pip install --user -e .

FROM python:3.11-slim as runtime
# Minimal runtime image
COPY --from=builder /root/.local /root/.local
```

### Configuration Management
```python
# Environment-based configuration
@dataclass
class Config:
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    postgres_url: str = os.getenv("POSTGRES_URL")
    embedding_api_key: str = os.getenv("OPENROUTER_API_KEY")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
```

## Monitoring Architecture

### Metrics Collection
```python
# Performance monitoring
@timer
def get_context_for_agent(self, state):
    # Measure latency
    start_time = time.time()
    result = self._get_context_impl(state)
    self.metrics.record("context_retrieval_time", time.time() - start_time)
    return result
```

### Health Checks
```python
def health_check(self) -> Dict[str, bool]:
    return {
        "redis": self.redis.ping(),
        "postgres": self.postgres.health_check(),
        "embedding_service": self.embedding_service.health_check()
    }
```

## Migration Strategy

### Backward Compatibility
```python
# Support old import patterns during transition
# src/memory/__init__.py
from .types import MemoryType as _MemoryType
from .meta_memory import MemoryType as _OldMemoryType

# Provide compatibility shim
MemoryType = _MemoryType  # Use new implementation
```

### Gradual Rollout
1. **Phase 1**: Deploy new types alongside old ones
2. **Phase 2**: Update internal code to use new types
3. **Phase 3**: Remove backward compatibility shims
4. **Phase 4**: Clean up deprecated code

## Future Extensibility

### Plugin Architecture
```python
# Extensible embedding services
class EmbeddingPlugin(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass

# Register new plugins
embedding_plugins = {
    "openrouter": OpenRouterEmbedding,
    "local": LocalEmbedding,
    "mock": MockEmbedding
}
```

### Custom Memory Types
```python
# Allow user-defined memory types
class CustomMemoryType(Enum):
    USER_DEFINED = "custom"

# Extend validation logic
def validate_memory_type(memory_type: MemoryType) -> bool:
    return memory_type in MemoryType or isinstance(memory_type, CustomMemoryType)
```

This architecture provides a solid foundation for professional, maintainable, and scalable AI memory systems. The design decisions prioritize clean code, testability, and performance while maintaining flexibility for future enhancements.
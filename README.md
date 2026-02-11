# Memory System

[![CI](https://github.com/marcusjihansson/memory_opt/actions/workflows/ci.yml/badge.svg)](https://github.com/marcusjihansson/memory_opt/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Features

- **Five-Tier Architecture**:
  - **Layer 0**: Meta-Memory (System 3) - Coherence validation and narrative identity
  - **Layer 1**: Redis cache for instant responses (< 1ms)
  - **Layer 2**: Short-term memory with importance-based consolidation
  - **Layer 3**: PostgreSQL with pgvector for semantic search
  - **Layer 4**: RLM reasoning - DSPy-powered recursive exploration and synthesis

- **Meta-Cognitive Capabilities**:
  - **Coherence Validation**: Pre-action consistency checking with narrative state
  - **Hierarchical Goal Management**: Life → Session → Tactical goal alignment
  - **Semantic Conflict Detection**: Automatic identification of contradictory information
  - **Narrative Identity Persistence**: Maintains consistent agent personality across sessions
  - **Memory Attribution**: Understands WHY information is remembered and HOW it relates
  - **RLM Multi-Hop Reasoning**: Chain-of-thought reasoning across disparate memories using recursive language models
  - **Memory Synthesis**: Temporal aggregation and insight generation from large memory stores with safe fallbacks

- **Advanced Memory Management**:
  - Distributed locking for concurrent access
  - Automatic memory consolidation based on importance scores
  - Semantic search using vector embeddings
  - User profile tracking and personalized responses
  - Type-safe architecture with professional Python practices

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- uv package manager

### Setup

1. **Clone and setup the environment:**
   ```bash
   git clone <repository-url>
   cd memory
   chmod +x scripts/setup_dev.sh
   ./scripts/setup_dev.sh
   ```

   This will:
   - Start Redis and PostgreSQL containers
   - Install Python dependencies
   - Initialize the database schema
   - Run basic health checks

 2. **Run the examples:**
    ```bash
    # Basic usage with simple LangGraph integration
    uv run python examples/basic_usage.py

    # Advanced usage with full memory system and consolidation
    uv run python examples/advanced_usage.py

    # MetaMemory integration - coherence checking and narrative identity
    uv run python examples/integration_test.py

    # Full integration demo (requires Docker containers)
    uv run python examples/full_integration_demo.py
    ```

### Manual Setup (Alternative)

If you prefer to set up services manually:

1. **Start the services:**
   ```bash
   docker-compose up -d
   ```

 2. **Initialize the database:**
    ```bash
    uv run python scripts/init_db.py
    ```

 3. **Install dependencies:**
    ```bash
    uv sync
    ```

 4. **Run tests to verify setup:**
    ```bash
    # Unit tests (fast, no external dependencies)
    uv run pytest tests/unit/ -v

    # Integration tests (requires Docker containers)
    uv run pytest tests/integration/ -v -m integration
    ```

## Configuration

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

## Architecture Overview

### Memory Flow
```
Query arrives
    ↓
┌─────────────────────────────────────────┐
│ SYSTEM 3: META-MEMORY          │ ← Narrative coherence check
│                                 │
│  ┌─────────────────────────────┐  │
│  │ Coherence Validator       │  │
│  │ - Goal alignment          │  │
│  │ - Constraint checking     │  │
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
```

Layer 4 (RLM) can optionally run after retrieval to recursively explore related memories, perform multi-hop reasoning, and synthesize an answer. Falls back to standard retrieval when unavailable.

### Key Components

- **`MetaMemory`**: System 3 meta-cognitive layer with coherence validation
- **`MemoryManager`**: Coordinates all five memory layers (including optional RLM)
- **`CachedMemory`**: Redis-based fast caching with distributed locks
- **`ShortTermMemory`**: Importance-scored conversation context
- **`LongTermMemory`**: PostgreSQL with vector search capabilities
- **`EmbeddingService`**: Text embeddings for semantic search
- **`RLMMemoryExplorer`**: Recursive exploration of memory context via DSPy RLM
- **`RLMMemoryReasoner`**: Multi-hop chain reasoning with fallback to ChainOfThought
- **`RLMMemorySynthesizer`**: Temporal synthesis across memory stores
- **`ProductionRLMMemory`**: Production-ready module combining explorer, reasoner, and synthesizer
- **`types.py`**: Shared type definitions eliminating circular dependencies

## Development

### Testing
```bash
# Run all tests
uv run pytest

# Unit tests only (fast, no external dependencies)
uv run pytest tests/unit/ -v

# Integration tests (requires Docker containers)
uv run pytest tests/integration/ -v -m integration

# With coverage report
uv run pytest --cov=src/memory --cov-report=html
open htmlcov/index.html
```

### Type Checking
```bash
# Run type checker
uv run mypy src/memory/

# Fix issues automatically (install missing stubs)
uv run mypy src/memory/ --install-types
```

### Code Quality
```bash
# Lint and format
uv run ruff check .
uv run ruff format .

# Fix auto-fixable issues
uv run ruff check --fix .
```

### Database Management

**View logs:**
```bash
docker-compose logs -f postgres
docker-compose logs -f redis
```

**Reset database:**
```bash
docker-compose down -v  # Remove volumes
docker-compose up -d
uv run python scripts/init_db.py
```

**Stop services:**
```bash
docker-compose down
```

## API Reference

### MemoryManager

```python
from memory import MemoryManager

# Initialize with Redis and PostgreSQL URLs
memory_manager = MemoryManager(
    redis_url="redis://localhost:6379",
    postgres_url="postgresql://user:pass@localhost/memorydb"
)

# Get context for agent (combines all memory layers)
context = memory_manager.get_context_for_agent(state, include_historical=True)

# Save interaction across all layers
memory_manager.save_interaction(state, message, agent_result)
```

### MetaMemory API

```python
from memory import MetaMemory, MemoryType, MemoryStability, GoalLevel, NarrativeState

# Initialize with embedding service
meta_memory = MetaMemory(embedding_service)

# Add memory with meta-attributes
memory = meta_memory.add_memory(
    content="User prefers Python for data science",
    memory_type=MemoryType.FACT,
    confidence=0.9,
    stability=MemoryStability.STABLE,
    tags=["preference", "programming"]
)

# Check coherence before action
coherence_result = meta_memory.check_coherence(
    proposed_action="Recommend Python resources",
    narrative_state=narrative_state
)

# Access coherence validation
if coherence_result['should_proceed']:
    print("Action is coherent with goals")
else:
    print(f"Action blocked: {coherence_result['violations']}")
```

### AgentState

```python
from memory import AgentState

state: AgentState = {
    "messages": [{"role": "user", "content": "Hello"}],
    "session_id": "session-123",
    "user_id": "user-456",
    "turn_count": 0,
    "conversation_history": [],
    "working_memory": {},
    "agent_outputs": {},
    "memory_importance_scores": {},
    "last_consolidation_turn": 0,
    # New MetaMemory fields
    "narrative_state": create_default_narrative_state(),
    "coherence_violations": [],
    "meta_reasoning": "",
    "long_term_memory": []
}
```

### Shared Types

All memory-related types are centralized in the `types` module:

```python
from memory.types import (
    MemoryType,      # FACT, GOAL, CONSTRAINT, COMMITMENT, IDENTITY
    MemoryStability, # VOLATILE, STABLE, CORE
    GoalLevel,       # LIFE, SESSION, TACTICAL
    Memory,          # Memory dataclass with embeddings
    HierarchicalGoal,# Goal hierarchy structure
    NarrativeState   # Meta-memory narrative identity
)
```

## Migration Guide

### Upgrading to MetaMemory v2

If upgrading from versions without MetaMemory:

1. **Update imports** to use shared `types` module:
   ```python
   # Before
   from memory.meta_memory import MemoryType, MemoryStability

   # After
   from memory.types import MemoryType, MemoryStability, GoalLevel
   ```

2. **MetaMemory initialization** now uses dependency injection:
   ```python
   # Before
   meta = MetaMemory(embedding_service, long_term_memory)

   # After
   meta = MetaMemory(embedding_service, long_term_memory)  # Optional
   ```

3. **Run type checks** to ensure compatibility:
   ```bash
   uv run mypy src/memory/
   ```

4. **Run tests** to verify functionality:
   ```bash
   uv run pytest tests/unit/test_meta_memory.py
   ```

See [`MIGRATION.md`](MIGRATION.md) for detailed migration steps.

## Troubleshooting

### Connection Issues

**PostgreSQL connection refused:**
```bash
# Check if container is running
docker-compose ps

# Check logs
docker-compose logs postgres

# Restart services
docker-compose restart
```

**Redis connection failed:**
```bash
# Check Redis status
docker-compose exec redis redis-cli ping

# Restart Redis
docker-compose restart redis
```

### Import Errors

Make sure the package is installed in editable mode:
```bash
uv pip install -e .
```

### Performance Issues

- **Slow vector searches**: Ensure pgvector indexes are created
- **High memory usage**: Check consolidation triggers in ShortTermMemory
- **Cache misses**: Verify Redis connectivity and TTL settings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
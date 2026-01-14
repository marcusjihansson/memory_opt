# Migration Guide: MetaMemory v1 → v2

## Overview

This guide covers the migration from the original memory system to the enhanced MetaMemory v2 system. The v2 system introduces meta-cognitive capabilities, hierarchical goal management, and improved type safety while maintaining backward compatibility.

## Breaking Changes

### Import Path Changes
**Before:**
```python
from memory.meta_memory import MemoryType, Memory, NarrativeState
```

**After:**
```python
from memory.types import MemoryType, Memory, NarrativeState
from memory import MetaMemory
```

### MetaMemory Constructor Changes
**Before:**
```python
meta_memory = MetaMemory(embedding_service, long_term_memory)
```

**After:**
```python
meta_memory = MetaMemory(embedding_service, long_term_memory)  # Optional parameter
```

## Migration Steps

### Step 1: Update Dependencies
```bash
# Install new dependencies if not already installed
pip install psycopg pgvector  # For database support
pip install mypy  # For type checking

# Update project dependencies
uv sync
```

### Step 2: Update Import Statements

#### In Application Code
```python
# BEFORE
from memory import MemoryManager
from memory.meta_memory import MetaMemory, MemoryType

# AFTER
from memory import MemoryManager, MetaMemory
from memory.types import MemoryType, MemoryStability
```

#### In Test Files
```python
# BEFORE
from memory.meta_memory import MemoryType, create_default_narrative_state

# AFTER
from memory.types import MemoryType, create_default_narrative_state
```

### Step 3: Update MetaMemory Usage

#### Constructor Changes
```python
# BEFORE (required long_term_memory)
embedding_service = EmbeddingService()
long_term_memory = LongTermMemory(...)
meta_memory = MetaMemory(embedding_service, long_term_memory)

# AFTER (optional long_term_memory)
meta_memory = MetaMemory(embedding_service, long_term_memory)  # Still works
# OR
meta_memory = MetaMemory(embedding_service)  # Works without persistence
```

#### Memory Creation Changes
```python
# BEFORE
memory = meta_memory.add_memory("content", MemoryType.FACT, session_id="test")

# AFTER (same API, but now with optional persistence)
memory = meta_memory.add_memory(
    "content",
    MemoryType.FACT,
    session_id="test",
    persist=True,  # New optional parameter
    user_id="user123"  # New optional parameter
)
```

### Step 4: Update AgentState

#### Add New Fields
```python
# BEFORE
from memory import AgentState

state = AgentState(
    messages=[...],
    session_id="session_123",
    user_id="user_456",
    turn_count=0,
    # ... other fields
)

# AFTER
state = AgentState(
    messages=[...],
    session_id="session_123",
    user_id="user_456",
    turn_count=0,
    # ... existing fields
    # NEW: MetaMemory fields
    narrative_state={},  # Will be loaded automatically
    coherence_violations=[],
    meta_reasoning="",
    long_term_memory=[]
)
```

### Step 5: Update MemoryManager Usage

#### New Methods Available
```python
# BEFORE
memory_manager = MemoryManager(redis_url, postgres_url)

# AFTER (same initialization)
memory_manager = MemoryManager(redis_url, postgres_url)

# NEW: Load narrative state
narrative_state = memory_manager.load_narrative_state(agent_state)

# NEW: Access MetaMemory directly
coherence_result = memory_manager.meta_memory.check_coherence(
    "proposed action", narrative_state
)
```

### Step 6: Update Type Hints

#### Add Type Annotations
```python
# BEFORE
def process_memory(memory):
    return memory.content

# AFTER
from memory.types import Memory

def process_memory(memory: Memory) -> str:
    return memory.content
```

#### Update Function Signatures
```python
# BEFORE
def create_memory(content, memory_type):
    return Memory(...)

# AFTER
from memory.types import MemoryType
from typing import Optional

def create_memory(content: str, memory_type: MemoryType, confidence: Optional[float] = None):
    return Memory(
        content=content,
        type=memory_type,
        confidence=confidence or 0.8,
        # ... other required fields
    )
```

### Step 7: Update Tests

#### Test Fixtures
```python
# BEFORE
@pytest.fixture
def embedding_service():
    return EmbeddingService()

# AFTER (add new fixtures)
@pytest.fixture
def embedding_service():
    os.environ['OPENROUTER_API_KEY'] = 'test-key'
    return EmbeddingService()

@pytest.fixture
def mock_long_term_memory():
    from unittest.mock import Mock
    mock = Mock()
    mock.store_memory = Mock(return_value=True)
    return mock

@pytest.fixture
def narrative_state():
    from memory.types import create_default_narrative_state
    return create_default_narrative_state()
```

#### Test Imports
```python
# BEFORE
from memory.meta_memory import MetaMemory

# AFTER
from memory import MetaMemory
from memory.types import MemoryType
```

### Step 8: Enable Type Checking

#### Add Mypy Configuration
```toml
# pyproject.toml
[tool.mypy]
python_version = "3.11"
warn_return_any = True

[[tool.mypy.overrides]]
module = "memory.*"
disallow_untyped_defs = False  # Start permissive, enable later
```

#### Run Type Checks
```bash
# Initial type check (expect some errors)
mypy src/memory/

# Fix obvious type errors
# Gradually enable stricter checking
```

### Step 9: Update Examples

#### Basic Usage Example
```python
# BEFORE
from memory.meta_memory import create_default_narrative_state

# AFTER
from memory.types import create_default_narrative_state
```

#### Advanced Usage Example
```python
# BEFORE
from memory import MemoryManager
from memory.meta_memory import MetaMemory

# AFTER
from memory import MemoryManager, MetaMemory
from memory.types import GoalLevel
```

### Step 10: Run Tests and Validation

#### Unit Tests
```bash
# Run unit tests
pytest tests/unit/ -v

# Check that core functionality works
pytest tests/unit/test_meta_memory.py::TestMetaMemoryBasics::test_meta_memory_initialization -v
```

#### Integration Tests
```bash
# Run integration tests (requires Docker)
pytest tests/integration/ -v -m integration
```

#### Type Checking Validation
```bash
# Ensure no type errors
mypy src/memory/
```

## Compatibility Matrix

### Supported Versions
- **Python**: 3.11+ (tested)
- **PostgreSQL**: 16+ (with pgvector)
- **Redis**: 7+ (with JSON support)

### Backward Compatibility
- ✅ **MemoryManager API**: Unchanged
- ✅ **AgentState structure**: Extended, not broken
- ✅ **Core memory operations**: Unchanged
- ⚠️ **Import paths**: Updated for better organization
- ❌ **Old MetaMemory imports**: Deprecated (use new paths)

## Troubleshooting Migration

### Common Issues

#### Import Errors
**Problem:**
```python
ImportError: No module named 'memory.types'
```

**Solution:**
```bash
# Ensure you're using the updated package
pip install -e .
# Or
uv sync
```

#### Type Checking Errors
**Problem:**
```
error: Library stubs not installed for "psycopg"
```

**Solution:**
```bash
pip install types-psycopg2
# Or ignore in mypy.ini
[mypy-psycopg.*]
ignore_missing_imports = True
```

#### Test Failures
**Problem:** Tests fail with database connection errors

**Solution:**
```bash
# Start Docker services
docker-compose up -d

# Wait for services to be healthy
docker-compose ps

# Run tests again
pytest tests/integration/ -m integration
```

#### Circular Import Errors
**Problem:** Still getting circular import issues

**Solution:** The migration should have resolved this. If not:
```python
# Check that all imports use the new paths
# Ensure no old imports remain
grep -r "from memory.meta_memory import" src/
# Should return no results (except in old backup files)
```

### Rollback Plan
If migration fails, you can rollback:

1. **Revert import changes** to use old paths temporarily
2. **Keep using v1 API** until issues are resolved
3. **Gradual migration** - update one module at a time

## Benefits of Migration

### Developer Experience
- ✅ **Better IDE support** with type hints
- ✅ **Catch errors early** with static analysis
- ✅ **Improved documentation** with type annotations
- ✅ **Easier testing** with better isolation

### Code Quality
- ✅ **Zero circular dependencies** 
- ✅ **Professional architecture** following SOLID principles
- ✅ **Comprehensive test coverage** framework
- ✅ **Type safety** throughout the codebase

### Feature Enhancements
- ✅ **Meta-cognitive reasoning** with coherence validation
- ✅ **Hierarchical goal management** (Life → Session → Tactical)
- ✅ **Semantic conflict detection** between memories
- ✅ **Narrative state persistence** across sessions

## Next Steps

### Immediate Actions
1. **Run the migration** following the steps above
2. **Enable type checking** with mypy
3. **Run full test suite** to ensure functionality
4. **Update CI/CD** to include type checking

### Future Improvements
1. **Enable stricter type checking** gradually
2. **Add performance benchmarks** for MetaMemory operations
3. **Implement advanced conflict resolution** strategies
4. **Add monitoring and observability** for narrative states

### Getting Help
If you encounter issues during migration:

1. **Check the examples** - `examples/integration_test.py` shows working usage
2. **Review test failures** - Tests provide clear error messages
3. **Enable debug logging** - Set `LOG_LEVEL=DEBUG` for detailed output
4. **Check Docker services** - Ensure PostgreSQL and Redis are running

The migration provides significant improvements in code quality and functionality while maintaining a clear upgrade path.
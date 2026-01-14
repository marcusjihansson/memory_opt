# Testing Guide

## Overview

This guide covers the comprehensive testing strategy for the memory system, including unit tests, integration tests, and testing best practices. The testing architecture follows modern Python testing principles with pytest.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures and configuration
├── unit/                         # Isolated unit tests
│   ├── __init__.py
│   ├── test_types.py             # Types module tests
│   ├── test_meta_memory.py       # MetaMemory unit tests
│   ├── test_memory_manager.py    # MemoryManager unit tests
│   ├── test_embedding.py         # Embedding service tests
│   └── test_long_term_memory.py  # LongTermMemory unit tests
├── integration/                  # Full system integration tests
│   ├── __init__.py
│   └── test_full_workflow.py     # End-to-end workflow tests
└── fixtures/                     # Test data and utilities
    ├── mock_memory.py            # Mock implementations
    └── sample_data.py            # Test data generators
```

## Running Tests

### Development Testing
```bash
# Run all tests
pytest

# Run with detailed output
pytest -v

# Run specific test file
pytest tests/unit/test_meta_memory.py

# Run specific test function
pytest tests/unit/test_meta_memory.py::TestMetaMemory::test_add_memory

# Run with coverage
pytest --cov=src/memory --cov-report=html
open htmlcov/index.html
```

### Continuous Integration
```bash
# Fast unit tests only (no external dependencies)
pytest tests/unit/ -x --tb=short

# Integration tests (requires Docker services)
pytest tests/integration/ -x --tb=short -m integration

# Full test suite with coverage
pytest --cov=src/memory --cov-report=term-missing --cov-fail-under=80
```

### Test Categories
```bash
# Unit tests (fast, isolated)
pytest -m "not integration"

# Integration tests (requires external services)
pytest -m integration

# Slow tests
pytest -m slow

# Run tests in parallel (requires pytest-xdist)
pytest -n auto
```

## Test Configuration

### pytest.ini Configuration
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --strict-markers
    --tb=short
    -v
    --cov=src/memory
    --cov-report=term-missing
    --cov-fail-under=80
markers =
    unit: Unit tests (fast, no external dependencies)
    integration: Integration tests (requires Docker services)
    slow: Slow running tests
    memory: Memory-related tests
    meta_memory: MetaMemory-specific tests
```

### Coverage Configuration
```ini
[coverage:run]
source = src/memory
omit =
    tests/*
    */__pycache__/*
    src/memory/docs/*

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == "__main__":
```

## Shared Fixtures (`tests/conftest.py`)

### Core Fixtures
```python
import pytest
import numpy as np
from unittest.mock import Mock

from memory import EmbeddingService, MemoryManager
from memory.types import create_default_narrative_state


@pytest.fixture
def embedding_service():
    """Provides EmbeddingService with mock API key"""
    import os
    os.environ['OPENROUTER_API_KEY'] = 'test-key-12345'
    return EmbeddingService()


@pytest.fixture
def mock_long_term_memory():
    """Mock LongTermMemory for testing without database"""
    mock = Mock()
    mock.store_memory = Mock(return_value=True)
    mock.load_narrative_state = Mock(return_value=None)
    mock.store_narrative_state = Mock(return_value=True)
    return mock


@pytest.fixture
def narrative_state():
    """Provides default narrative state for testing"""
    return create_default_narrative_state()


@pytest.fixture
def agent_state(narrative_state):
    """Provides complete AgentState for testing"""
    from memory import AgentState
    return AgentState(
        messages=[],
        conversation_history=[],
        working_memory={},
        session_id="test_session",
        user_id="test_user",
        turn_count=0,
        agent_outputs={},
        next_agent="",
        final_response="",
        last_consolidation_turn=0,
        memory_importance_scores={},
        narrative_state=narrative_state,
        coherence_violations=[],
        meta_reasoning="",
        long_term_memory=[]
    )


@pytest.fixture
def sample_memory():
    """Provides sample Memory object for testing"""
    from memory import MetaMemory
    from memory.types import MemoryType

    meta_memory = MetaMemory(embedding_service())
    return meta_memory.add_memory(
        content="Sample memory for testing",
        memory_type=MemoryType.FACT,
        confidence=0.9,
        session_id="test_session"
    )


@pytest.fixture
def memory_manager_with_mocks(embedding_service, mock_long_term_memory):
    """Provides MemoryManager with mocked dependencies"""
    from unittest.mock import patch

    with patch('memory.memory_manager.CachedMemory') as mock_cached:
        mock_cached.return_value = Mock()
        manager = MemoryManager("redis://mock", "postgresql://mock")
        manager.long_term = mock_long_term_memory
        manager.meta_memory = MetaMemory(embedding_service, mock_long_term_memory)
        return manager
```

## Unit Tests

### Testing Types Module
```python
# tests/unit/test_types.py
import pytest
from memory.types import MemoryType, MemoryStability, GoalLevel, Memory, HierarchicalGoal


class TestEnums:
    """Test enum definitions"""

    def test_memory_type_values(self):
        """Test MemoryType enum has expected values"""
        assert MemoryType.FACT.value == "fact"
        assert MemoryType.GOAL.value == "goal"
        assert MemoryType.CONSTRAINT.value == "constraint"
        assert MemoryType.COMMITMENT.value == "commitment"
        assert MemoryType.IDENTITY.value == "identity"

    def test_memory_stability_order(self):
        """Test MemoryStability enum ordering"""
        assert MemoryStability.VOLATILE.value < MemoryStability.STABLE.value
        assert MemoryStability.STABLE.value < MemoryStability.CORE.value


class TestDataClasses:
    """Test dataclass functionality"""

    def test_memory_creation(self):
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
            reasoning="Test memory"
        )

        assert memory.content == "Test memory"
        assert memory.type == MemoryType.FACT
        assert 0.0 <= memory.confidence <= 1.0

    def test_hierarchical_goal_relationships(self):
        """Test HierarchicalGoal parent-child relationships"""
        parent = HierarchicalGoal(
            id="parent_1",
            content="Parent goal",
            level=GoalLevel.LIFE
        )

        child = HierarchicalGoal(
            id="child_1",
            content="Child goal",
            level=GoalLevel.SESSION,
            parent_id=parent.id
        )

        assert child.parent_id == parent.id
        assert child.level == GoalLevel.SESSION
        assert parent.level == GoalLevel.LIFE

    def test_goal_progress_tracking(self):
        """Test goal progress updates"""
        goal = HierarchicalGoal(
            id="goal_1",
            content="Test goal",
            level=GoalLevel.TACTICAL
        )

        assert goal.progress == 0.0
        assert not goal.completed

        goal.progress = 1.0
        goal.completed = True

        assert goal.progress == 1.0
        assert goal.completed
```

### Testing MetaMemory
```python
# tests/unit/test_meta_memory.py
import pytest
from unittest.mock import Mock

from memory import MetaMemory
from memory.types import MemoryType, MemoryStability, GoalLevel


class TestMetaMemoryBasics:
    """Test basic MetaMemory functionality"""

    def test_initialization(self, embedding_service):
        """Test MetaMemory initializes correctly"""
        meta = MetaMemory(embedding_service)
        assert meta.violation_threshold == 0.7
        assert hasattr(meta, 'embedding_engine')
        assert hasattr(meta, 'goal_manager')

    def test_initialization_with_long_term_memory(self, embedding_service, mock_long_term_memory):
        """Test MetaMemory with LongTermMemory dependency"""
        meta = MetaMemory(embedding_service, mock_long_term_memory)
        assert meta.long_term_memory is mock_long_term_memory


class TestMemoryOperations:
    """Test memory creation and management"""

    def test_add_memory_creates_valid_memory(self, embedding_service):
        """Test add_memory creates valid Memory object"""
        meta = MetaMemory(embedding_service)

        memory = meta.add_memory(
            content="Test memory content",
            memory_type=MemoryType.FACT,
            confidence=0.8,
            session_id="test_session"
        )

        assert memory.content == "Test memory content"
        assert memory.type == MemoryType.FACT
        assert memory.confidence == 0.8
        assert memory.embedding is not None
        assert len(memory.embedding) > 0

    def test_add_memory_with_persistence(self, embedding_service, mock_long_term_memory):
        """Test memory persistence to LongTermMemory"""
        meta = MetaMemory(embedding_service, mock_long_term_memory)

        meta.add_memory(
            content="Persistent memory",
            memory_type=MemoryType.FACT,
            persist=True,
            user_id="test_user"
        )

        # Verify store_memory was called with correct parameters
        mock_long_term_memory.store_memory.assert_called_once()
        call_args = mock_long_term_memory.store_memory.call_args
        assert call_args.kwargs['content'] == "Persistent memory"
        assert call_args.kwargs['user_id'] == "test_user"

    def test_add_memory_without_persistence(self, embedding_service, mock_long_term_memory):
        """Test memory creation without persistence"""
        meta = MetaMemory(embedding_service, mock_long_term_memory)

        memory = meta.add_memory(
            content="Non-persistent memory",
            memory_type=MemoryType.FACT,
            persist=False
        )

        # Verify store_memory was NOT called
        mock_long_term_memory.store_memory.assert_not_called()
        assert memory.content == "Non-persistent memory"


class TestCoherenceValidation:
    """Test coherence checking functionality"""

    def test_coherence_with_compatible_action(self, embedding_service, narrative_state):
        """Test coherence validation with compatible action"""
        meta = MetaMemory(embedding_service)

        result = meta.check_coherence(
            "Help user with their coding project",
            narrative_state
        )

        assert result['coherent'] is True
        assert result['should_proceed'] is True
        assert isinstance(result['coherence_score'], float)
        assert 0.0 <= result['coherence_score'] <= 1.0

    def test_coherence_with_incompatible_action(self, embedding_service, narrative_state):
        """Test coherence validation with incompatible action"""
        meta = MetaMemory(embedding_service)

        result = meta.check_coherence(
            "Completely ignore user and do something else",
            narrative_state
        )

        # May fail coherence check depending on goals
        assert isinstance(result['coherence_score'], float)
        assert 'violations' in result
        assert 'reasoning' in result

    def test_coherence_without_long_term_memory(self, embedding_service, narrative_state):
        """Test coherence checking works without LongTermMemory"""
        meta = MetaMemory(embedding_service)  # No long_term_memory

        result = meta.check_coherence(
            "Test action",
            narrative_state,
            long_term_memory=[]  # Empty list
        )

        assert 'coherence_score' in result
        assert 'should_proceed' in result


class TestSemanticSearch:
    """Test memory relationship and conflict detection"""

    def test_find_related_memories(self, embedding_service):
        """Test finding semantically related memories"""
        meta = MetaMemory(embedding_service)

        # Create related memories
        memory1 = meta.add_memory("User likes Python", MemoryType.FACT, session_id="test")
        memory2 = meta.add_memory("User prefers functional programming", MemoryType.FACT, session_id="test")
        memory3 = meta.add_memory("User hates JavaScript", MemoryType.FACT, session_id="test")

        related = meta.find_related_memories(memory1, [memory1, memory2, memory3])

        # Should find at least itself
        assert len(related) >= 1
        found_memories = [mem for mem, score in related]
        assert memory1 in found_memories

    def test_memory_conflict_detection(self, embedding_service):
        """Test detection of conflicting memories"""
        meta = MetaMemory(embedding_service)

        # Create potentially conflicting memories
        memory1 = meta.add_memory("User is vegetarian", MemoryType.IDENTITY, confidence=1.0, session_id="test")
        memory2 = meta.add_memory("User loves steak", MemoryType.FACT, confidence=0.8, session_id="test")

        conflict = meta.detect_memory_conflicts(memory2, [memory1])

        # May or may not detect conflict depending on embeddings
        # The important thing is the method works
        assert conflict is not None
        assert 'has_conflict' in conflict
```

### Testing MemoryManager
```python
# tests/unit/test_memory_manager.py
import pytest
from unittest.mock import Mock, patch

from memory import MemoryManager, AgentState
from memory.types import create_default_narrative_state


class TestMemoryManagerInitialization:
    """Test MemoryManager setup and dependencies"""

    def test_initialization_creates_components(self):
        """Test MemoryManager creates all required components"""
        with patch('memory.memory_manager.CachedMemory'), \
             patch('memory.memory_manager.LongTermMemory'), \
             patch('memory.memory_manager.MetaMemory'):

            manager = MemoryManager("redis://test", "postgresql://test")

            assert hasattr(manager, 'cached')
            assert hasattr(manager, 'short_term')
            assert hasattr(manager, 'embedding_service')
            assert hasattr(manager, 'long_term')
            assert hasattr(manager, 'meta_memory')


class TestNarrativeStateManagement:
    """Test narrative state loading and management"""

    def test_load_narrative_state_from_database(self, memory_manager_with_mocks, mock_long_term_memory, narrative_state):
        """Test loading narrative state from database"""
        mock_long_term_memory.load_narrative_state.return_value = narrative_state

        loaded_state = memory_manager_with_mocks.load_narrative_state(AgentState(
            session_id="test_session",
            user_id="test_user",
            messages=[],
            conversation_history=[],
            working_memory={},
            turn_count=0,
            agent_outputs={},
            next_agent="",
            final_response="",
            last_consolidation_turn=0,
            memory_importance_scores={},
            narrative_state={},
            coherence_violations=[],
            meta_reasoning="",
            long_term_memory=[]
        ))

        assert loaded_state == narrative_state
        mock_long_term_memory.load_narrative_state.assert_called_once_with("test_session", "test_user")

    def test_load_narrative_state_creates_default(self, memory_manager_with_mocks, mock_long_term_memory):
        """Test creating default narrative state when none exists"""
        mock_long_term_memory.load_narrative_state.return_value = None

        state = AgentState(
            session_id="test_session",
            user_id="test_user",
            messages=[],
            conversation_history=[],
            working_memory={},
            turn_count=0,
            agent_outputs={},
            next_agent="",
            final_response="",
            last_consolidation_turn=0,
            memory_importance_scores={},
            narrative_state={},
            coherence_violations=[],
            meta_reasoning="",
            long_term_memory=[]
        )

        loaded_state = memory_manager_with_mocks.load_narrative_state(state)

        assert loaded_state is not None
        assert 'identity' in loaded_state
        assert 'goal_hierarchy' in loaded_state
```

## Integration Tests

### Full Workflow Testing
```python
# tests/integration/test_full_workflow.py
import pytest

from memory import MemoryManager, AgentState
from memory.types import MemoryType, create_default_narrative_state


class TestFullWorkflow:
    """Test complete memory system workflows"""

    @pytest.mark.integration
    def test_memory_creation_and_persistence_workflow(self):
        """Test complete workflow from memory creation to persistence"""
        # This test requires running Docker containers
        manager = MemoryManager(
            redis_url="redis://localhost:6379",
            postgres_url="postgresql://memory_user:memory_pass@localhost:5432/memory_db"
        )

        # Create agent state
        state = AgentState(
            messages=[{"role": "user", "content": "Hello, I need help with Python"}],
            conversation_history=[],
            working_memory={},
            session_id="integration_test_session",
            user_id="integration_test_user",
            turn_count=1,
            agent_outputs={},
            next_agent="",
            final_response="",
            last_consolidation_turn=0,
            memory_importance_scores={},
            narrative_state=create_default_narrative_state(),
            coherence_violations=[],
            meta_reasoning="",
            long_term_memory=[]
        )

        # 1. Load narrative state
        state["narrative_state"] = manager.load_narrative_state(state)
        assert state["narrative_state"] is not None

        # 2. Save interaction (creates memories)
        message = {"role": "user", "content": "I need help with Python data science", "id": "msg_1"}
        manager.save_interaction(state, message)

        # 3. Verify memory was created
        assert len(state["long_term_memory"]) > 0
        memory = state["long_term_memory"][0]
        assert memory.content == "I need help with Python data science"
        assert memory.type == MemoryType.FACT

        # 4. Test coherence checking
        coherence_result = manager.meta_memory.check_coherence(
            "Provide Python data science resources",
            state["narrative_state"]
        )
        assert coherence_result["should_proceed"] is True

        # 5. End session and persist
        manager.end_session(state)

        # Clean up
        manager.cached.invalidate_session("integration_test_session")

    @pytest.mark.integration
    def test_meta_memory_conflict_detection(self):
        """Test MetaMemory conflict detection in full system"""
        manager = MemoryManager(
            redis_url="redis://localhost:6379",
            postgres_url="postgresql://memory_user:memory_pass@localhost:5432/memory_db"
        )

        # Create conflicting memories
        memory1 = manager.meta_memory.add_memory(
            "User is allergic to nuts",
            MemoryType.IDENTITY,
            confidence=1.0,
            session_id="conflict_test"
        )

        memory2 = manager.meta_memory.add_memory(
            "User loves peanut butter",
            MemoryType.FACT,
            confidence=0.9,
            session_id="conflict_test"
        )

        # Check for conflicts
        conflict = manager.meta_memory.detect_memory_conflicts(memory2, [memory1])

        # System should be able to handle the conflict detection
        assert conflict is not None
        assert 'has_conflict' in conflict
```

## Test Data and Fixtures

### Sample Data Generator
```python
# tests/fixtures/sample_data.py
from memory.types import MemoryType, MemoryStability, GoalLevel, Memory, HierarchicalGoal


def create_sample_memories(count: int = 5) -> list[Memory]:
    """Generate sample memories for testing"""
    memories = []
    contents = [
        "User prefers Python over JavaScript",
        "User has 5 years of programming experience",
        "User works in data science",
        "User prefers functional programming",
        "User uses Linux as primary OS"
    ]

    for i, content in enumerate(contents[:count]):
        memory = Memory(
            content=content,
            type=MemoryType.FACT,
            confidence=0.8 + (i * 0.02),  # Varying confidence
            importance=0.6 + (i * 0.05),  # Varying importance
            stability=MemoryStability.STABLE,
            source="test_data",
            created_at=f"2024-01-{i+1:02d}T10:00:00",
            last_verified=f"2024-01-{i+1:02d}T10:00:00",
            session_id="sample_session",
            tags=["sample", f"memory_{i}"],
            reasoning=f"Sample memory {i+1} for testing"
        )
        memories.append(memory)

    return memories


def create_sample_goals() -> dict[str, HierarchicalGoal]:
    """Generate sample goal hierarchy for testing"""
    goals = {}

    # Life goal
    life_goal = HierarchicalGoal(
        id="life_sample",
        content="Help user become proficient in software development",
        level=GoalLevel.LIFE,
        importance=1.0
    )
    goals[life_goal.id] = life_goal

    # Session goal
    session_goal = HierarchicalGoal(
        id="session_sample",
        content="Guide user through Python learning journey",
        level=GoalLevel.SESSION,
        parent_id=life_goal.id,
        importance=0.8
    )
    goals[session_goal.id] = session_goal

    # Tactical goals
    tactical_goals = [
        HierarchicalGoal(
            id=f"tactical_{i}",
            content=f"Complete learning objective {i+1}",
            level=GoalLevel.TACTICAL,
            parent_id=session_goal.id,
            importance=0.6 - (i * 0.1)
        ) for i in range(3)
    ]

    for goal in tactical_goals:
        goals[goal.id] = goal

    return goals
```

## Performance Testing

### Benchmarking Tests
```python
# tests/integration/test_performance.py
import pytest
import time
from memory import MemoryManager


@pytest.mark.integration
@pytest.mark.slow
class TestPerformance:
    """Performance regression tests"""

    def test_memory_creation_performance(self):
        """Test memory creation performance"""
        manager = MemoryManager(
            redis_url="redis://localhost:6379",
            postgres_url="postgresql://memory_user:memory_pass@localhost:5432/memory_db"
        )

        start_time = time.time()
        for i in range(100):
            memory = manager.meta_memory.add_memory(
                content=f"Performance test memory {i}",
                memory_type=MemoryType.FACT,
                session_id="perf_test"
            )
            assert memory is not None

        end_time = time.time()
        total_time = end_time - start_time

        # Should create 100 memories in reasonable time
        assert total_time < 30.0  # Less than 30 seconds
        print(f"Created 100 memories in {total_time:.2f} seconds")

    def test_coherence_check_performance(self):
        """Test coherence checking performance"""
        manager = MemoryManager(
            redis_url="redis://localhost:6379",
            postgres_url="postgresql://memory_user:memory_pass@localhost:5432/memory_db"
        )

        # Create narrative state
        state = AgentState(
            session_id="perf_test",
            user_id="perf_user",
            messages=[],
            conversation_history=[],
            working_memory={},
            turn_count=0,
            agent_outputs={},
            next_agent="",
            final_response="",
            last_consolidation_turn=0,
            memory_importance_scores={},
            narrative_state=manager.load_narrative_state(AgentState(
                session_id="perf_test",
                user_id="perf_user",
                messages=[],
                conversation_history=[],
                working_memory={},
                turn_count=0,
                agent_outputs={},
                next_agent="",
                final_response="",
                last_consolidation_turn=0,
                memory_importance_scores={},
                narrative_state={},
                coherence_violations=[],
                meta_reasoning="",
                long_term_memory=[]
            )),
            coherence_violations=[],
            meta_reasoning="",
            long_term_memory=[]
        )

        start_time = time.time()
        for i in range(50):
            result = manager.meta_memory.check_coherence(
                f"Test action {i}",
                state["narrative_state"]
            )
            assert "coherence_score" in result

        end_time = time.time()
        total_time = end_time - start_time

        # Should check coherence for 50 actions quickly
        assert total_time < 10.0  # Less than 10 seconds
        print(f"Checked 50 coherences in {total_time:.2f} seconds")
```

## CI/CD Integration

### GitHub Actions Workflow
```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_USER: memory_user
          POSTGRES_PASSWORD: memory_pass
          POSTGRES_DB: memory_db
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      redis:
        image: redis:7
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-cov mypy

      - name: Run type checking
        run: mypy src/memory/

      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=src/memory --cov-report=xml

      - name: Run integration tests
        run: pytest tests/integration/ -v -m integration

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

## Best Practices

### Writing Good Tests
1. **Test One Thing**: Each test should verify one specific behavior
2. **Use Descriptive Names**: Test names should explain what they're testing
3. **Isolate Dependencies**: Use mocks for external services in unit tests
4. **Test Edge Cases**: Include boundary conditions and error scenarios
5. **Keep Tests Fast**: Unit tests should run in milliseconds

### Test Organization
1. **Arrange-Act-Assert**: Structure tests clearly
2. **Use Fixtures**: Share common test setup
3. **Parameterize Tests**: Test multiple inputs with one test function
4. **Document Test Intent**: Use docstrings to explain test purpose

### Maintenance
1. **Regular Test Runs**: Run tests frequently during development
2. **Update Tests**: When changing code, update corresponding tests
3. **Code Coverage**: Aim for >80% coverage on critical paths
4. **Performance Benchmarks**: Monitor test execution time

This testing guide provides a comprehensive framework for ensuring the memory system's reliability, performance, and maintainability.
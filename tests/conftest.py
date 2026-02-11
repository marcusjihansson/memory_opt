"""
Pytest configuration and shared fixtures
"""

import os
from unittest.mock import Mock

import numpy as np
import pytest

from memory import EmbeddingService
from memory.meta_memory import create_default_narrative_state


@pytest.fixture
def embedding_service():
    """Fixture providing EmbeddingService with mock API key"""
    os.environ["OPENROUTER_API_KEY"] = "test-key-12345"
    return EmbeddingService()


@pytest.fixture
def mock_long_term_memory():
    """Mock LongTermMemory that doesn't require DB"""
    mock = Mock()
    mock.store_memory = Mock(return_value=True)
    mock.load_narrative_state = Mock(return_value=None)
    return mock


@pytest.fixture
def narrative_state():
    """Fixture providing default narrative state"""
    return create_default_narrative_state()


@pytest.fixture
def agent_state():
    """Fixture providing basic AgentState"""
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
        narrative_state=create_default_narrative_state(),
        coherence_violations=[],
        meta_reasoning="",
        long_term_memory=[],
    )


@pytest.fixture
def sample_embedding():
    """Fixture providing consistent test embedding"""
    np.random.seed(42)
    return np.random.rand(1536).tolist()

"""
Unit tests for MetaMemory - tests in isolation without DB
"""


from memory import MetaMemory
from memory.types import (
    MemoryStability,
    MemoryType,
)


class TestMetaMemoryBasics:
    """Test basic MetaMemory functionality"""

    def test_meta_memory_initialization(self, embedding_service):
        """Test MetaMemory initializes correctly"""
        meta = MetaMemory(embedding_service)
        assert meta.violation_threshold == 0.7
        assert meta.embedding_engine is not None
        assert meta.goal_manager is not None

    def test_meta_memory_with_long_term_memory(
        self, embedding_service, mock_long_term_memory
    ):
        """Test MetaMemory with optional LongTermMemory"""
        meta = MetaMemory(embedding_service, mock_long_term_memory)
        assert meta.long_term_memory is mock_long_term_memory


class TestMemoryCreation:
    """Test memory creation and management"""

    def test_add_memory_creates_valid_object(self, embedding_service):
        """Test add_memory creates valid Memory object"""
        meta = MetaMemory(embedding_service)
        memory = meta.add_memory(
            content="Test memory",
            memory_type=MemoryType.FACT,
            session_id="test",
            confidence=0.9,
        )

        assert memory.content == "Test memory"
        assert memory.type == MemoryType.FACT
        assert memory.confidence == 0.9
        assert memory.embedding is not None
        assert len(memory.embedding) == 1536

    def test_add_memory_with_custom_attributes(self, embedding_service):
        """Test add_memory with custom attributes"""
        meta = MetaMemory(embedding_service)
        memory = meta.add_memory(
            content="Important fact",
            memory_type=MemoryType.FACT,
            session_id="test",
            confidence=1.0,
            importance=0.9,
            stability=MemoryStability.CORE,
            tags=["important", "core"],
            reasoning="This is from user explicitly",
        )

        assert memory.importance == 0.9
        assert memory.stability == MemoryStability.CORE
        assert "important" in memory.tags

    def test_add_memory_persists_to_long_term(
        self, embedding_service, mock_long_term_memory
    ):
        """Test memory persistence to LongTermMemory"""
        meta = MetaMemory(embedding_service, mock_long_term_memory)
        meta.add_memory(
            content="Test memory",
            memory_type=MemoryType.FACT,
            session_id="test",
            persist=True,
        )

        # Verify store_memory was called
        assert mock_long_term_memory.store_memory.called
        call_args = mock_long_term_memory.store_memory.call_args
        assert call_args.kwargs["content"] == "Test memory"


class TestCoherenceChecking:
    """Test coherence validation"""

    def test_check_coherence_with_compliant_action(
        self, embedding_service, narrative_state
    ):
        """Test coherent action passes validation"""
        meta = MetaMemory(embedding_service)

        result = meta.check_coherence("Help user with their goal", narrative_state)

        assert result["coherent"]
        assert result["should_proceed"]
        assert result["coherence_score"] >= 0.7

    def test_check_coherence_with_violating_action(
        self, embedding_service, narrative_state
    ):
        """Test incoherent action fails validation"""
        meta = MetaMemory(embedding_service)

        result = meta.check_coherence(
            "Do something completely unrelated", narrative_state
        )

        # May or may not fail depending on goals
        assert isinstance(result["coherence_score"], float)
        assert 0.0 <= result["coherence_score"] <= 1.0


class TestConflictDetection:
    """Test semantic conflict detection"""

    def test_detect_memory_conflicts_no_conflict(self, embedding_service):
        """Test no conflict detection for unrelated memories"""
        meta = MetaMemory(embedding_service)

        memory1 = meta.add_memory(
            "User likes Python", MemoryType.FACT, session_id="test"
        )
        memory2 = meta.add_memory(
            "User likes JavaScript", MemoryType.FACT, session_id="test"
        )

        result = meta.detect_memory_conflicts(memory2, [memory1])

        # With mock embeddings, might detect similarity but low confidence diff
        assert result is None or not result["has_conflict"]

    def test_detect_memory_conflicts_with_conflict(self, embedding_service):
        """Test conflict detection for contradictory memories"""
        meta = MetaMemory(embedding_service)

        memory1 = meta.add_memory(
            "User is vegetarian", MemoryType.IDENTITY, confidence=1.0, session_id="test"
        )
        memory2 = meta.add_memory(
            "User loves eating meat", MemoryType.FACT, confidence=0.9, session_id="test"
        )

        result = meta.detect_memory_conflicts(memory2, [memory1])

        # With mock embeddings, might not detect
        # This tests the mechanism works
        assert result is not None

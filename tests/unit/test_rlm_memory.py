"""
Unit tests for RLM Memory - tests in isolation without DB or LLM calls
"""

from unittest.mock import MagicMock, patch

from memory.rlm_memory import (
    AgentMemoryRetrieval,
    ExploreMemory,
    MemoryChainReasoning,
    ProductionRLMMemory,
    RLMMemoryExplorer,
    RLMMemoryReasoner,
    RLMMemorySynthesizer,
    SynthesizeMemories,
    format_memories_for_rlm,
    parse_rlm_memories,
)


class TestRLMSignatures:
    """Test DSPy signature definitions"""

    def test_explore_memory_signature_fields(self):
        """Test ExploreMemory signature has correct fields"""
        # DSPy signatures use model_fields for field definitions
        fields = ExploreMemory.model_fields
        assert "query" in fields
        assert "memory_context" in fields
        assert "search_depth" in fields
        assert "exploration_strategy" in fields
        assert "relevant_memories" in fields
        assert "confidence" in fields

    def test_memory_chain_reasoning_signature_fields(self):
        """Test MemoryChainReasoning signature has correct fields"""
        fields = MemoryChainReasoning.model_fields
        assert "question" in fields
        assert "memory_bank" in fields
        assert "reasoning_chain" in fields
        assert "connected_memories" in fields
        assert "answer" in fields

    def test_synthesize_memories_signature_fields(self):
        """Test SynthesizeMemories signature has correct fields"""
        fields = SynthesizeMemories.model_fields
        assert "task" in fields
        assert "memory_database" in fields
        assert "temporal_range" in fields
        assert "search_plan" in fields
        assert "synthesis" in fields
        assert "supporting_evidence" in fields

    def test_agent_memory_retrieval_signature_fields(self):
        """Test AgentMemoryRetrieval signature has correct fields"""
        fields = AgentMemoryRetrieval.model_fields
        assert "current_situation" in fields
        assert "agent_memories" in fields
        assert "goal" in fields
        assert "memory_exploration" in fields
        assert "relevant_experiences" in fields
        assert "recommended_action" in fields


class TestRLMMemoryExplorer:
    """Test RLMMemoryExplorer module"""

    def test_initialization_default(self):
        """Test default initialization"""
        with patch("memory.rlm_memory.RLM"):
            explorer = RLMMemoryExplorer()
            assert explorer.max_iterations == 10

    def test_initialization_custom_iterations(self):
        """Test custom max_iterations"""
        with patch("memory.rlm_memory.RLM"):
            explorer = RLMMemoryExplorer(max_iterations=20)
            assert explorer.max_iterations == 20

    def test_forward_returns_prediction(self):
        """Test forward returns dspy.Prediction"""
        with patch("memory.rlm_memory.RLM") as mock_rlm_class:
            mock_rlm = MagicMock()
            mock_rlm.return_value = MagicMock(
                relevant_memories=["memory1", "memory2"],
                confidence=0.85,
            )
            mock_rlm_class.return_value = mock_rlm

            explorer = RLMMemoryExplorer()
            result = explorer(
                memory_context="Test context",
                query="Find relevant info",
                search_depth=3,
            )

            assert hasattr(result, "relevant_memories")
            assert hasattr(result, "confidence")
            assert hasattr(result, "search_depth")


class TestRLMMemoryReasoner:
    """Test RLMMemoryReasoner module"""

    def test_initialization_default(self):
        """Test default initialization"""
        with patch("memory.rlm_memory.RLM"):
            reasoner = RLMMemoryReasoner()
            assert reasoner.max_iterations == 15

    def test_forward_with_successful_rlm(self):
        """Test forward with successful RLM result"""
        with patch("memory.rlm_memory.RLM") as mock_rlm_class:
            mock_rlm = MagicMock()
            mock_rlm.return_value = MagicMock(
                reasoning_chain="Step 1 -> Step 2 -> Answer",
                answer="The answer is 42",
            )
            mock_rlm_class.return_value = mock_rlm

            reasoner = RLMMemoryReasoner()
            result = reasoner(
                memory_bank="Memory content here",
                question="What is the meaning?",
            )

            assert result.answer == "The answer is 42"
            assert result.method == "rlm"

    def test_forward_fallback_on_empty_result(self):
        """Test fallback is used when RLM returns empty"""
        with patch("memory.rlm_memory.RLM") as mock_rlm_class:
            mock_rlm = MagicMock()
            mock_rlm.return_value = MagicMock(
                reasoning_chain="",
                answer="",
            )
            mock_rlm_class.return_value = mock_rlm

            reasoner = RLMMemoryReasoner()
            # Mock the fallback
            reasoner.fallback = MagicMock()
            reasoner.fallback.return_value = MagicMock(
                reasoning_chain="Fallback reasoning",
                connected_memories=["mem1"],
                answer="Fallback answer",
            )

            result = reasoner(
                memory_bank="Memory content",
                question="What is this?",
                use_fallback=True,
            )

            assert result.method == "fallback"
            assert result.answer == "Fallback answer"

    def test_forward_fallback_on_error(self):
        """Test fallback is used when RLM raises exception"""
        with patch("memory.rlm_memory.RLM") as mock_rlm_class:
            mock_rlm = MagicMock()
            mock_rlm.side_effect = Exception("RLM failed")
            mock_rlm_class.return_value = mock_rlm

            reasoner = RLMMemoryReasoner()
            # Mock the fallback
            reasoner.fallback = MagicMock()
            reasoner.fallback.return_value = MagicMock(
                reasoning_chain="Error fallback",
                connected_memories=[],
                answer="Error fallback answer",
            )

            result = reasoner(
                memory_bank="Memory content",
                question="What is this?",
                use_fallback=True,
            )

            assert result.method == "fallback_error"
            assert "Error" not in result.answer or result.answer == "Error fallback answer"


class TestRLMMemorySynthesizer:
    """Test RLMMemorySynthesizer module"""

    def test_initialization_default(self):
        """Test default initialization"""
        with patch("memory.rlm_memory.RLM"):
            synthesizer = RLMMemorySynthesizer()
            assert synthesizer.max_iterations == 15

    def test_forward_with_successful_synthesis(self):
        """Test forward with successful synthesis"""
        with patch("memory.rlm_memory.RLM") as mock_rlm_class:
            mock_rlm = MagicMock()
            mock_rlm.return_value = MagicMock(
                synthesis="Synthesized insights",
                supporting_evidence=["evidence1", "evidence2"],
            )
            mock_rlm_class.return_value = mock_rlm

            synthesizer = RLMMemorySynthesizer()
            result = synthesizer(
                memory_database="All memories",
                task="Summarize patterns",
                temporal_range="last week",
            )

            assert result.synthesis == "Synthesized insights"
            assert result.method == "rlm"


class TestProductionRLMMemory:
    """Test ProductionRLMMemory module"""

    def test_initialization(self):
        """Test initialization creates all sub-modules"""
        with patch("memory.rlm_memory.RLM"):
            production = ProductionRLMMemory()
            assert production.explorer is not None
            assert production.reasoner is not None
            assert production.synthesizer is not None
            assert production.fallback is not None

    def test_explore_delegates_to_explorer(self):
        """Test explore method uses explorer"""
        with patch("memory.rlm_memory.RLM") as mock_rlm_class:
            mock_rlm = MagicMock()
            mock_rlm.return_value = MagicMock(
                relevant_memories=["mem1"],
                confidence=0.9,
            )
            mock_rlm_class.return_value = mock_rlm

            production = ProductionRLMMemory()
            result = production.explore(
                memory_context="Context",
                query="Query",
            )

            assert hasattr(result, "relevant_memories")

    def test_reason_delegates_to_reasoner(self):
        """Test reason method uses reasoner"""
        with patch("memory.rlm_memory.RLM") as mock_rlm_class:
            mock_rlm = MagicMock()
            mock_rlm.return_value = MagicMock(
                reasoning_chain="Chain",
                answer="Answer",
            )
            mock_rlm_class.return_value = mock_rlm

            production = ProductionRLMMemory()
            result = production.reason(
                memory_bank="Bank",
                question="Question",
            )

            assert hasattr(result, "answer")

    def test_synthesize_delegates_to_synthesizer(self):
        """Test synthesize method uses synthesizer"""
        with patch("memory.rlm_memory.RLM") as mock_rlm_class:
            mock_rlm = MagicMock()
            mock_rlm.return_value = MagicMock(
                synthesis="Synthesis",
                supporting_evidence=[],
            )
            mock_rlm_class.return_value = mock_rlm

            production = ProductionRLMMemory()
            result = production.synthesize(
                memory_database="Database",
                task="Task",
            )

            assert hasattr(result, "synthesis")


class TestHelperFunctions:
    """Test helper functions"""

    def test_format_memories_for_rlm_empty(self):
        """Test formatting empty memory list"""
        result = format_memories_for_rlm([])
        assert result == "No memories available."

    def test_format_memories_for_rlm_with_memories(self):
        """Test formatting memory list"""
        memories = [
            {
                "content": "User likes Python",
                "timestamp": "2024-01-01",
                "importance": 0.8,
                "type": "fact",
            },
            {
                "content": "User prefers dark mode",
                "timestamp": "2024-01-02",
                "importance_score": 0.6,
                "memory_type": "preference",
            },
        ]
        result = format_memories_for_rlm(memories)

        assert "[Memory 1]" in result
        assert "[Memory 2]" in result
        assert "User likes Python" in result
        assert "User prefers dark mode" in result

    def test_parse_rlm_memories_from_list(self):
        """Test parsing list of memories"""
        result = parse_rlm_memories(["memory1", "memory2"])
        assert result == ["memory1", "memory2"]

    def test_parse_rlm_memories_from_string(self):
        """Test parsing string of memories"""
        result = parse_rlm_memories("memory1\nmemory2\n")
        assert "memory1" in result
        assert "memory2" in result

    def test_parse_rlm_memories_from_prediction(self):
        """Test parsing from Prediction-like object"""
        mock_prediction = MagicMock()
        mock_prediction.relevant_memories = ["mem1", "mem2"]
        result = parse_rlm_memories(mock_prediction)
        assert result == ["mem1", "mem2"]


class TestRLMMemoryIntegration:
    """Integration-style tests for RLM memory modules"""

    def test_explorer_to_reasoner_flow(self):
        """Test flow from explorer to reasoner"""
        with patch("memory.rlm_memory.RLM") as mock_rlm_class:
            mock_rlm = MagicMock()
            mock_rlm.return_value = MagicMock(
                relevant_memories=["relevant memory 1"],
                confidence=0.9,
                reasoning_chain="Step by step",
                answer="Final answer",
            )
            mock_rlm_class.return_value = mock_rlm

            explorer = RLMMemoryExplorer()
            reasoner = RLMMemoryReasoner()

            # First explore
            explore_result = explorer(
                memory_context="All memories here",
                query="Find info about X",
            )

            # Then reason using explored memories
            memories_str = str(explore_result.relevant_memories)
            reason_result = reasoner(
                memory_bank=memories_str,
                question="What can we conclude about X?",
            )

            assert explore_result.confidence == 0.9
            assert reason_result.answer == "Final answer"

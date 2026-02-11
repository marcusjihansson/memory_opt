"""
RLM (Recursive Language Model) powered memory exploration and reasoning.
Provides DSPy signatures and modules for intelligent memory retrieval.
"""

from typing import Any

import dspy
from dspy.predict.rlm import RLM

# ============================================================================
# DSPy SIGNATURES FOR MEMORY OPERATIONS
# ============================================================================


class ExploreMemory(dspy.Signature):
    """Recursively explore memory database to find relevant information."""

    query: str = dspy.InputField(desc="What information to find")
    memory_context: str = dspy.InputField(desc="All memories as searchable text")
    search_depth: int = dspy.InputField(desc="How many recursive explorations to allow")

    exploration_strategy: str = dspy.OutputField(desc="How to decompose the search")
    relevant_memories: list[str] = dspy.OutputField(desc="Found relevant memories")
    confidence: float = dspy.OutputField(desc="Confidence in retrieval quality")


class MemoryChainReasoning(dspy.Signature):
    """Follow chains of reasoning across multiple memories."""

    question: str = dspy.InputField(desc="Question requiring multi-hop reasoning")
    memory_bank: str = dspy.InputField(desc="All available memories")

    reasoning_chain: str = dspy.OutputField(desc="Step-by-step memory traversal")
    connected_memories: list[str] = dspy.OutputField(desc="Memories used in chain")
    answer: str = dspy.OutputField(desc="Final synthesized answer")


class SynthesizeMemories(dspy.Signature):
    """Explore and synthesize information from vast memory stores."""

    task: str = dspy.InputField(desc="Task requiring memory synthesis")
    memory_database: str = dspy.InputField(desc="Complete memory context")
    temporal_range: str = dspy.InputField(
        desc="Time range to focus on (e.g., 'last week', 'all time')"
    )

    search_plan: str = dspy.OutputField(desc="Recursive exploration strategy")
    synthesis: str = dspy.OutputField(desc="Aggregated insights from memories")
    supporting_evidence: list[str] = dspy.OutputField(desc="Key memories referenced")


class AgentMemoryRetrieval(dspy.Signature):
    """Agent explores its memory to inform decision-making."""

    current_situation: str = dspy.InputField(desc="Current state/context")
    agent_memories: str = dspy.InputField(desc="All past experiences, knowledge, observations")
    goal: str = dspy.InputField(desc="Agent's current goal")

    memory_exploration: str = dspy.OutputField(desc="Recursive search through memories")
    relevant_experiences: list[str] = dspy.OutputField(desc="Past experiences that apply")
    recommended_action: str = dspy.OutputField(desc="Action based on memory insights")


class QuestionAnswer(dspy.Signature):
    """Answer questions based on memory context."""

    context: str = dspy.InputField(desc="The memory context to analyze")
    question: str = dspy.InputField(desc="The question to answer")
    answer: str = dspy.OutputField(desc="A comprehensive answer based on the context")


# ============================================================================
# RLM MEMORY MODULES
# ============================================================================


class RLMMemoryExplorer(dspy.Module):
    """
    Basic RLM-powered memory exploration module.
    Uses recursive context exploration to find relevant memories.
    """

    def __init__(self, max_iterations: int = 10):
        super().__init__()
        self.max_iterations = max_iterations
        self.rlm = RLM(
            signature="memory_context, query -> relevant_memories, confidence",
            max_iterations=max_iterations,
        )

    def forward(self, memory_context: str, query: str, search_depth: int = 3) -> dspy.Prediction:
        """
        Explore memory context to find relevant information.

        Args:
            memory_context: All memories as searchable text
            query: What information to find
            search_depth: How deep to search (affects iterations)

        Returns:
            Prediction with relevant_memories and confidence
        """
        result = self.rlm(memory_context=memory_context, query=query)

        relevant_memories = (
            result.relevant_memories if hasattr(result, "relevant_memories") else str(result)
        )
        confidence = result.confidence if hasattr(result, "confidence") else 0.5

        return dspy.Prediction(
            relevant_memories=relevant_memories,
            confidence=confidence,
            search_depth=search_depth,
        )


class RLMMemoryReasoner(dspy.Module):
    """
    Multi-hop reasoning module using RLM for chain-of-thought across memories.
    Follows reasoning chains to connect disparate memories.
    """

    def __init__(self, max_iterations: int = 15):
        super().__init__()
        self.max_iterations = max_iterations
        self.rlm = RLM(
            signature="memory_bank, question -> reasoning_chain, answer",
            max_iterations=max_iterations,
        )
        self.fallback = dspy.ChainOfThought(MemoryChainReasoning)

    def forward(
        self, memory_bank: str, question: str, use_fallback: bool = True
    ) -> dspy.Prediction:
        """
        Perform multi-hop reasoning across memories.

        Args:
            memory_bank: All available memories
            question: Question requiring multi-hop reasoning
            use_fallback: Whether to use ChainOfThought fallback on failure

        Returns:
            Prediction with reasoning_chain, connected_memories, and answer
        """
        try:
            result = self.rlm(memory_bank=memory_bank, question=question)

            reasoning_chain = result.reasoning_chain if hasattr(result, "reasoning_chain") else ""
            answer = result.answer if hasattr(result, "answer") else str(result)

            if answer:
                return dspy.Prediction(
                    reasoning_chain=reasoning_chain,
                    connected_memories=[],
                    answer=answer,
                    method="rlm",
                )

            if use_fallback:
                fallback_result = self.fallback(memory_bank=memory_bank[:8000], question=question)
                return dspy.Prediction(
                    reasoning_chain=fallback_result.reasoning_chain,
                    connected_memories=fallback_result.connected_memories,
                    answer=fallback_result.answer,
                    method="fallback",
                )

            return dspy.Prediction(
                reasoning_chain="",
                connected_memories=[],
                answer="Unable to reason over memories",
                method="none",
            )

        except Exception as e:
            if use_fallback:
                fallback_result = self.fallback(memory_bank=memory_bank[:8000], question=question)
                return dspy.Prediction(
                    reasoning_chain=fallback_result.reasoning_chain,
                    connected_memories=fallback_result.connected_memories,
                    answer=fallback_result.answer,
                    method="fallback_error",
                    error=str(e),
                )
            return dspy.Prediction(
                reasoning_chain="",
                connected_memories=[],
                answer=f"Error: {str(e)}",
                method="error",
                error=str(e),
            )


class RLMMemorySynthesizer(dspy.Module):
    """
    Synthesize insights from memory stores using RLM exploration.
    Aggregates information across temporal ranges.
    """

    def __init__(self, max_iterations: int = 15):
        super().__init__()
        self.max_iterations = max_iterations
        self.rlm = RLM(
            signature="memory_database, task, temporal_range -> synthesis, supporting_evidence",
            max_iterations=max_iterations,
        )
        self.fallback = dspy.ChainOfThought(SynthesizeMemories)

    def forward(
        self,
        memory_database: str,
        task: str,
        temporal_range: str = "all time",
        use_fallback: bool = True,
    ) -> dspy.Prediction:
        """
        Synthesize insights from memory database.

        Args:
            memory_database: Complete memory context
            task: Task requiring memory synthesis
            temporal_range: Time range to focus on
            use_fallback: Whether to use fallback on failure

        Returns:
            Prediction with synthesis, supporting_evidence, and search_plan
        """
        try:
            result = self.rlm(
                memory_database=memory_database,
                task=task,
                temporal_range=temporal_range,
            )

            synthesis = result.synthesis if hasattr(result, "synthesis") else str(result)
            evidence = result.supporting_evidence if hasattr(result, "supporting_evidence") else []

            if synthesis:
                return dspy.Prediction(
                    synthesis=synthesis,
                    supporting_evidence=evidence,
                    search_plan="RLM recursive exploration",
                    method="rlm",
                )

            if use_fallback:
                fallback_result = self.fallback(
                    memory_database=memory_database[:8000],
                    task=task,
                    temporal_range=temporal_range,
                )
                return dspy.Prediction(
                    synthesis=fallback_result.synthesis,
                    supporting_evidence=fallback_result.supporting_evidence,
                    search_plan=fallback_result.search_plan,
                    method="fallback",
                )

            return dspy.Prediction(
                synthesis="Unable to synthesize memories",
                supporting_evidence=[],
                search_plan="",
                method="none",
            )

        except Exception as e:
            if use_fallback:
                fallback_result = self.fallback(
                    memory_database=memory_database[:8000],
                    task=task,
                    temporal_range=temporal_range,
                )
                return dspy.Prediction(
                    synthesis=fallback_result.synthesis,
                    supporting_evidence=fallback_result.supporting_evidence,
                    search_plan=fallback_result.search_plan,
                    method="fallback_error",
                    error=str(e),
                )
            return dspy.Prediction(
                synthesis=f"Error: {str(e)}",
                supporting_evidence=[],
                search_plan="",
                method="error",
                error=str(e),
            )


class ProductionRLMMemory(dspy.Module):
    """
    Production-ready RLM memory system combining exploration, reasoning, and synthesis.
    Includes error handling, fallbacks, and method tracking.
    """

    def __init__(self, max_iterations: int = 15):
        super().__init__()
        self.explorer = RLMMemoryExplorer(max_iterations=max_iterations)
        self.reasoner = RLMMemoryReasoner(max_iterations=max_iterations)
        self.synthesizer = RLMMemorySynthesizer(max_iterations=max_iterations)
        self.fallback = dspy.ChainOfThought(QuestionAnswer)

    def explore(self, memory_context: str, query: str, search_depth: int = 3) -> dspy.Prediction:
        """Explore memories to find relevant information."""
        return self.explorer(memory_context=memory_context, query=query, search_depth=search_depth)

    def reason(self, memory_bank: str, question: str, use_fallback: bool = True) -> dspy.Prediction:
        """Perform multi-hop reasoning across memories."""
        return self.reasoner(memory_bank=memory_bank, question=question, use_fallback=use_fallback)

    def synthesize(
        self,
        memory_database: str,
        task: str,
        temporal_range: str = "all time",
        use_fallback: bool = True,
    ) -> dspy.Prediction:
        """Synthesize insights from memory stores."""
        return self.synthesizer(
            memory_database=memory_database,
            task=task,
            temporal_range=temporal_range,
            use_fallback=use_fallback,
        )

    def forward(self, context: str, question: str, use_fallback: bool = True) -> dspy.Prediction:
        """
        General-purpose memory Q&A using RLM.

        Args:
            context: Memory context to search
            question: Question to answer
            use_fallback: Whether to use fallback on failure

        Returns:
            Prediction with answer and method used
        """
        try:
            reasoning_result = self.reasoner(
                memory_bank=context, question=question, use_fallback=False
            )

            if reasoning_result.answer and reasoning_result.method == "rlm":
                return dspy.Prediction(
                    answer=reasoning_result.answer,
                    reasoning=reasoning_result.reasoning_chain,
                    method="rlm",
                    success=True,
                )

            if use_fallback:
                fallback_result = self.fallback(context=context[:8000], question=question)
                return dspy.Prediction(
                    answer=fallback_result.answer,
                    reasoning="",
                    method="fallback",
                    success=True,
                )

            return dspy.Prediction(
                answer="Unable to process query",
                reasoning="",
                method="none",
                success=False,
            )

        except Exception as e:
            if use_fallback:
                fallback_result = self.fallback(context=context[:8000], question=question)
                return dspy.Prediction(
                    answer=fallback_result.answer,
                    reasoning="",
                    method="fallback_error",
                    success=True,
                    error=str(e),
                )
            return dspy.Prediction(
                answer=f"Error: {str(e)}",
                reasoning="",
                method="error",
                success=False,
                error=str(e),
            )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def format_memories_for_rlm(memories: list[dict[str, Any]]) -> str:
    """
    Format a list of memory dictionaries into a text format suitable for RLM.

    Args:
        memories: List of memory dicts with 'content', 'timestamp', etc.

    Returns:
        Formatted string of all memories
    """
    if not memories:
        return "No memories available."

    formatted = []
    for i, mem in enumerate(memories, 1):
        content = mem.get("content", "")
        timestamp = mem.get("timestamp", mem.get("created_at", "unknown"))
        importance = mem.get("importance", mem.get("importance_score", 0.5))
        mem_type = mem.get("type", mem.get("memory_type", "fact"))

        formatted.append(
            f"[Memory {i}] ({timestamp}) [{mem_type}] (importance: {importance:.2f})\n" f"{content}"
        )

    return "\n\n".join(formatted)


def parse_rlm_memories(rlm_output: Any) -> list[str]:
    """
    Parse RLM output into a list of memory strings.

    Args:
        rlm_output: Output from RLM (could be string, list, or Prediction)

    Returns:
        List of memory strings
    """
    if isinstance(rlm_output, list):
        return [str(m) for m in rlm_output]
    elif isinstance(rlm_output, str):
        return [line.strip() for line in rlm_output.split("\n") if line.strip()]
    elif hasattr(rlm_output, "relevant_memories"):
        memories = rlm_output.relevant_memories
        if isinstance(memories, list):
            return [str(m) for m in memories]
        return [str(memories)]
    else:
        return [str(rlm_output)]

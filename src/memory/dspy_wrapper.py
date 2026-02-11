"""
DSPy wrapper for integrating with the memory system.
Provides memory-enabled execution of DSPy programs with configurable layers.
Supports RLM (Recursive Language Model) for intelligent memory exploration.
"""

from typing import Any

import dspy

from .memory_manager import MemoryManager
from .rlm_memory import ProductionRLMMemory, RLMMemoryExplorer
from .state import AgentState


class MemoryDSPyWrapper:
    """
    Wrapper that integrates DSPy programs with the memory system.
    Supports both structured and text memory formats.
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        cache: bool = True,
        short_memory: bool = False,
        long_memory: bool = False,
        enable_consolidation: bool = False,
        rlm_enabled: bool = False,
        rlm_max_iterations: int = 10,
    ):
        """
        Initialize the wrapper.

        Args:
            memory_manager: Instance of MemoryManager
            cache: Enable cached memory layer (default True)
            short_memory: Enable short-term memory layer (default False)
            long_memory: Enable long-term memory layer (default False)
            enable_consolidation: Automatically consolidate memories after execution (default False)
            rlm_enabled: Enable RLM for intelligent memory exploration (default False)
            rlm_max_iterations: Maximum iterations for RLM exploration (default 10)
        """
        self.memory_manager = memory_manager
        self.cache = cache
        self.short_memory = short_memory
        self.long_memory = long_memory
        self.enable_consolidation = enable_consolidation
        self.rlm_enabled = rlm_enabled
        self.rlm_max_iterations = rlm_max_iterations

        # Lazy-loaded RLM modules
        self._rlm_explorer: RLMMemoryExplorer | None = None
        self._rlm_production: ProductionRLMMemory | None = None

    @property
    def rlm_explorer(self) -> RLMMemoryExplorer:
        """Lazy-load RLM explorer."""
        if self._rlm_explorer is None:
            self._rlm_explorer = RLMMemoryExplorer(
                max_iterations=self.rlm_max_iterations
            )
        return self._rlm_explorer

    @property
    def rlm_production(self) -> ProductionRLMMemory:
        """Lazy-load production RLM module."""
        if self._rlm_production is None:
            self._rlm_production = ProductionRLMMemory(
                max_iterations=self.rlm_max_iterations
            )
        return self._rlm_production

    def get_memory_context(self, state: AgentState, format: str = "structured") -> Any:
        """
        Retrieve memory context in specified format.

        Args:
            state: Current agent state
            format: "structured" (dict) or "text" (string)

        Returns:
            Memory context in requested format
        """
        context_parts: list[str] = []
        structured_context: dict[str, Any] = {}

        # Cached memory
        if self.cache:
            cached = self.memory_manager.cached.get_cached_context(state["session_id"])
            if cached:
                context_parts.append(f"[CACHED] {cached.get('summary', '')}")
                structured_context["cached"] = cached

        # Short-term memory
        if self.short_memory:
            short_term = self.memory_manager.short_term.build_context_window(state)
            context_parts.append(short_term)
            structured_context["short_term"] = short_term

        # Long-term memory (semantic search)
        if self.long_memory:
            query = (
                state.get("messages", [])[-1]["content"]
                if state.get("messages")
                else ""
            )
            relevant_memories = self.memory_manager.long_term.semantic_search(
                query=query,
                user_id=state["user_id"],
                limit=3,
                min_similarity=0.75,
                time_window_days=30,
            )
            if relevant_memories:
                context_parts.append("\n[RELEVANT PAST CONTEXT]")
                for mem in relevant_memories:
                    context_parts.append(
                        f"  [{mem['similarity']:.2f}] {mem['content'][:150]}..."
                    )
                structured_context["long_term"] = relevant_memories

        if format == "text":
            return "\n".join(context_parts)
        else:
            return structured_context

    def get_memory_context_with_rlm(
        self, state: AgentState, query: str, search_depth: int = 3
    ) -> dict[str, Any]:
        """
        Retrieve memory context using RLM for intelligent exploration.

        Args:
            state: Current agent state
            query: Query to explore memories with
            search_depth: Depth of RLM exploration

        Returns:
            Dictionary with explored memories and RLM results
        """
        # First get standard memory context
        base_context = self.get_memory_context(state, format="text")

        # Use RLM to explore and find relevant memories
        exploration_result = self.rlm_explorer(
            memory_context=base_context,
            query=query,
            search_depth=search_depth,
        )

        return {
            "base_context": base_context,
            "rlm_exploration": {
                "relevant_memories": exploration_result.relevant_memories,
                "confidence": exploration_result.confidence,
                "search_depth": exploration_result.search_depth,
            },
            "method": "rlm",
        }

    def execute_with_rlm(
        self,
        state: AgentState,
        query: str,
        use_fallback: bool = True,
    ) -> Any:
        """
        Execute RLM-based memory exploration and reasoning.

        Args:
            state: Current agent state
            query: Query/question to answer using memories
            use_fallback: Whether to use fallback on RLM failure

        Returns:
            RLM execution result with answer and reasoning
        """
        # Get full memory context
        memory_context = self.get_memory_context(state, format="text")

        # Use production RLM for Q&A
        result = self.rlm_production(
            context=memory_context,
            question=query,
            use_fallback=use_fallback,
        )

        # Optional memory consolidation
        if self.enable_consolidation:
            self.memory_manager.consolidate_memories(state)

        return result

    def execute_with_memory(
        self,
        dspy_program_class: type[dspy.Module],
        state: AgentState,
        inputs: dict[str, Any],
        format: str = "structured",
    ) -> Any:
        """
        Execute DSPy program with memory context.

        Args:
            dspy_program_class: DSPy program class to instantiate and run
            state: Current agent state
            inputs: Input arguments for the DSPy program
            format: Memory format ("structured" or "text")

        Returns:
            DSPy program execution result
        """
        # Get memory context
        memory_context = self.get_memory_context(state, format)

        # Merge memory into inputs
        if format == "structured":
            inputs["memory_context"] = memory_context
        else:
            inputs["memory_context_text"] = memory_context

        # Execute DSPy program
        program = dspy_program_class()
        result = program(**inputs)

        # Optional memory consolidation
        if self.enable_consolidation:
            self.memory_manager.consolidate_memories(state)

        return result

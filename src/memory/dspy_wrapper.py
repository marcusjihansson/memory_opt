"""
DSPy wrapper for integrating with the memory system.
Provides memory-enabled execution of DSPy programs with configurable layers.
"""

from typing import Dict, Any, Type
import dspy
from .memory_manager import MemoryManager
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
    ):
        """
        Initialize the wrapper.

        Args:
            memory_manager: Instance of MemoryManager
            cache: Enable cached memory layer (default True)
            short_memory: Enable short-term memory layer (default False)
            long_memory: Enable long-term memory layer (default False)
            enable_consolidation: Automatically consolidate memories after execution (default False)
        """
        self.memory_manager = memory_manager
        self.cache = cache
        self.short_memory = short_memory
        self.long_memory = long_memory
        self.enable_consolidation = enable_consolidation

    def get_memory_context(self, state: AgentState, format: str = "structured") -> Any:
        """
        Retrieve memory context in specified format.

        Args:
            state: Current agent state
            format: "structured" (dict) or "text" (string)

        Returns:
            Memory context in requested format
        """
        context_parts = []
        structured_context = {}

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

    def execute_with_memory(
        self,
        dspy_program_class: Type[dspy.Module],
        state: AgentState,
        inputs: Dict[str, Any],
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

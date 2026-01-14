"""
Integrated memory manager coordinating all three memory layers.
"""

import hashlib
from typing import Any, Dict, Optional
from .cached_memory import CachedMemory
from .short_term_memory import ShortTermMemory
from .long_term_memory import LongTermMemory
from .embedding import EmbeddingService
from .state import AgentState


class MemoryManager:
    """
    Coordinates all three memory layers with intelligent consolidation.
    """

    def __init__(self, redis_url: str, postgres_url: str):
        self.cached = CachedMemory(redis_url)
        self.short_term = ShortTermMemory()
        self.embedding_service = EmbeddingService()
        self.long_term = LongTermMemory(postgres_url, self.embedding_service)
        # Import here to avoid circular dependency
        from .meta_memory import MetaMemory

        self.meta_memory = MetaMemory(self.embedding_service, self.long_term)

    def load_narrative_state(self, state: AgentState) -> Dict[str, Any]:
        """Load or initialize narrative state for the session"""
        narrative_state = self.long_term.load_narrative_state(
            state["session_id"], state["user_id"]
        )

        if not narrative_state:
            # Initialize with default
            from .meta_memory import create_default_narrative_state

            narrative_state = create_default_narrative_state()
            print("[META-MEMORY] Initialized new narrative state")

        return narrative_state

    def get_context_for_agent(
        self, state: AgentState, include_historical: bool = False
    ) -> str:
        """
        Build comprehensive context using all memory layers.

        Flow:
        1. Check cache (Redis) - instant
        2. Use short-term memory (state) - current context
        3. Semantic search long-term (PostgreSQL) - historical relevance
        """
        context_parts = []

        # Layer 1: Cached context
        cached_context = self.cached.get_cached_context(state["session_id"])
        if cached_context:
            context_parts.append(f"[CACHED] {cached_context.get('summary', '')}")

        # Layer 2: Short-term memory
        short_term_context = self.short_term.build_context_window(state)
        context_parts.append(short_term_context)

        # Layer 3: Semantic search if requested
        if include_historical:
            # For historical context, we need a query - use the last message
            query = (
                state.get("messages", [])[-1]["content"]
                if state.get("messages")
                else ""
            )
            relevant_memories = self.long_term.semantic_search(
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

        # User profile
        profile = self.long_term.get_user_profile(state["user_id"])
        if profile and profile.get("preferences"):
            context_parts.append(f"\n[USER PREFERENCES] {profile['preferences']}")

        return "\n".join(context_parts)

    def save_interaction(
        self,
        state: AgentState,
        message: Dict[str, str],
        agent_result: Optional[Dict] = None,
    ):
        """Save interaction across all memory layers"""
        session_id = state["session_id"]
        user_id = state["user_id"]

        # Layer 1: Update cache
        self.cached.cache_conversation_context(
            session_id,
            {
                "last_message": message,
                "turn": state["turn_count"],
                "summary": self.short_term.build_context_window(state),
            },
        )

        if agent_result:
            query_hash = hashlib.md5(message["content"].encode()).hexdigest()
            self.cached.cache_agent_result(
                session_id, state["next_agent"], query_hash, agent_result
            )

        # Layer 3: Store in PostgreSQL with importance score
        importance = self.short_term.calculate_importance(
            message,
            {"turn_count": state["turn_count"], "message_turn": state["turn_count"]},
        )

        self.long_term.store_message(
            session_id=session_id,
            user_id=user_id,
            message_id=message.get("id", f"msg_{state['turn_count']}"),
            role=message["role"],
            content=message["content"],
            importance_score=importance,
            metadata={"agent": state.get("next_agent"), "turn": state["turn_count"]},
            generate_embedding=True,
        )

        # Meta-memory: Add to long-term memory if available
        if (
            hasattr(state, "long_term_memory")
            and state.get("long_term_memory") is not None
        ):
            from .meta_memory import MemoryType

            meta_mem = self.meta_memory.add_memory(
                content=message["content"],
                memory_type=MemoryType.FACT,
                session_id=session_id,
                confidence=0.8,
                importance=importance,
                tags=["conversation", f"turn_{state['turn_count']}"],
            )
            state["long_term_memory"].append(meta_mem)

    def consolidate_memories(self, state: AgentState):
        """
        Consolidate short-term memories to long-term storage.

        This moves important context from active memory to persistent storage,
        freeing up space in short-term memory while preserving key information.
        """
        # Get high-importance messages from short-term memory
        history = state.get("conversation_history", [])
        importance_scores = state.get("memory_importance_scores", {})

        important_messages = [
            {**msg, "importance": importance_scores.get(msg.get("id", ""), 0.5)}
            for msg in history
            if importance_scores.get(msg.get("id", ""), 0.5) > 0.6
        ]

        if not important_messages:
            return

        # Generate summary (in production, use LLM)
        summary = (
            f"Consolidated {len(important_messages)} important messages from session"
        )

        # Store in long-term memory
        self.long_term.consolidate_session_memories(
            session_id=state["session_id"],
            user_id=state["user_id"],
            messages=important_messages,
            summary=summary,
        )

        print(
            f"[CONSOLIDATION] Moved {len(important_messages)} messages to long-term storage"
        )

    def end_session(self, state: AgentState):
        """End session and clean up"""
        # Final consolidation
        self.consolidate_memories(state)

        # Meta-memory: Update narrative state
        if hasattr(state, "narrative_state") and state.get("narrative_state"):
            state["narrative_state"]["session_arc"]["turn_count"] = state.get(
                "turn_count", 0
            )
            print(
                f"[META-MEMORY] Session arc updated: {state['narrative_state']['session_arc']}"
            )

        # Clear cache
        self.cached.invalidate_session(state["session_id"])

        print(f"[SESSION END] Session {state['session_id']} finalized")

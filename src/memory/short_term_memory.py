"""
Short-term memory management with importance scoring for intelligent consolidation.
"""

from typing import Any

from .state import AgentState


class ShortTermMemory:
    """
    Manages active conversation context with importance scoring
    for intelligent consolidation.
    """

    @staticmethod
    def calculate_importance(message: dict[str, str], context: dict[str, Any]) -> float:
        """
        Calculate importance score for a message (0.0 to 1.0).

        Factors:
        - Recency (newer = more important)
        - Length (longer = potentially more important)
        - Keywords (certain words indicate importance)
        - User vs assistant (user messages slightly more important)
        """
        score = 0.0

        # Recency bonus (decays over time)
        turn_age = context.get("turn_count", 0) - context.get("message_turn", 0)
        recency_score = 1.0 / (1.0 + turn_age * 0.1)
        score += recency_score * 0.4

        # Length factor (normalized)
        content_length = len(message.get("content", ""))
        length_score = min(content_length / 500.0, 1.0)
        score += length_score * 0.2

        # Keyword importance
        important_keywords = [
            "important",
            "remember",
            "always",
            "never",
            "prefer",
            "don't",
            "issue",
            "problem",
            "error",
        ]
        content_lower = message.get("content", "").lower()
        keyword_matches = sum(1 for kw in important_keywords if kw in content_lower)
        score += min(keyword_matches * 0.15, 0.3)

        # Role bonus
        if message.get("role") == "user":
            score += 0.1

        return min(score, 1.0)

    @staticmethod
    def update_conversation_history(
        state: AgentState, new_message: dict[str, str], max_history: int = 20
    ) -> tuple[list[dict[str, str]], dict[str, float]]:
        """
        Update conversation history with sliding window and importance tracking.
        """
        history = state.get("conversation_history", [])
        importance_scores = state.get("memory_importance_scores", {})

        # Add message with importance score
        message_id = f"msg_{state['turn_count']}"
        history.append({**new_message, "id": message_id})

        context = {
            "turn_count": state["turn_count"],
            "message_turn": state["turn_count"],
        }
        importance = ShortTermMemory.calculate_importance(new_message, context)
        importance_scores[message_id] = importance

        # Keep only recent messages, but prioritize important ones
        if len(history) > max_history:
            # Sort by importance, keep top N
            scored_history = [
                (msg, importance_scores.get(msg.get("id", ""), 0.5)) for msg in history
            ]
            scored_history.sort(key=lambda x: x[1], reverse=True)
            history = [msg for msg, _ in scored_history[:max_history]]

        return history, importance_scores

    @staticmethod
    def should_consolidate(state: AgentState) -> bool:
        """
        Determine if memory should be consolidated to long-term storage.

        Triggers:
        - Every 15 turns
        - High number of important messages
        - Session nearing context limit
        """
        turn_count = state.get("turn_count", 0)
        last_consolidation = state.get("last_consolidation_turn", 0)

        # Every 15 turns
        if turn_count - last_consolidation >= 15:
            return True

        # High importance message count
        importance_scores = state.get("memory_importance_scores", {})
        high_importance_count = sum(
            1 for score in importance_scores.values() if score > 0.7
        )
        if high_importance_count >= 5:
            return True

        # History size approaching limit
        history_size = len(state.get("conversation_history", []))
        if history_size >= 18:  # Close to max of 20
            return True

        return False

    @staticmethod
    def update_working_memory(
        state: AgentState, key: str, value: Any
    ) -> dict[str, Any]:
        """Update working memory scratchpad"""
        working_mem = state.get("working_memory", {})
        working_mem[key] = value
        return working_mem

    @staticmethod
    def build_context_window(state: AgentState) -> str:
        """Build context for agent calls from short-term memory"""
        context_parts = []

        # Add working memory items
        if state.get("working_memory"):
            context_parts.append("Active Context:")
            for k, v in state["working_memory"].items():
                context_parts.append(f"  {k}: {v}")

        # Add recent conversation (prioritize by importance)
        history = state.get("conversation_history", [])
        importance_scores = state.get("memory_importance_scores", {})

        # Get top 5 most important recent messages
        scored_msgs = [
            (msg, importance_scores.get(msg.get("id", ""), 0.5))
            for msg in history[-10:]  # From last 10
        ]
        scored_msgs.sort(key=lambda x: x[1], reverse=True)
        top_messages = [msg for msg, _ in scored_msgs[:5]]

        if top_messages:
            context_parts.append("\nKey Recent Messages:")
            for msg in top_messages:
                importance = importance_scores.get(msg.get("id", ""), 0.5)
                context_parts.append(
                    f"  [{importance:.2f}] {msg['role']}: {msg['content'][:100]}"
                )

        return "\n".join(context_parts)

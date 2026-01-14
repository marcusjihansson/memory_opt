"""
Meta-Memory Implementation for LangGraph - Enhanced Version
Implements meta-cognitive layer with vector embeddings and hierarchical goals.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import numpy as np

from .types import (
    MemoryType,
    MemoryStability,
    GoalLevel,
    Memory,
    HierarchicalGoal,
    NarrativeState,
)
from .embedding import EmbeddingService

# Forward reference for type hints only
if TYPE_CHECKING:
    from .long_term_memory import LongTermMemory


# ============================================================================
# ENHANCED: VECTOR EMBEDDING SYSTEM
# ============================================================================


class EmbeddingEngine:
    """
    Manages semantic embeddings for memory and goals
    Uses existing EmbeddingService
    """

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        return np.array(self.embedding_service.embed_text(text))

    def semantic_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between embeddings"""
        return self.embedding_service.cosine_similarity(
            embedding1.tolist(), embedding2.tolist()
        )

    def find_related_memories(
        self,
        query_embedding: np.ndarray,
        memory_pool: List[Memory],
        threshold: float = 0.7,
        top_k: int = 5,
    ) -> List[Tuple[Memory, float]]:
        """
        Find semantically related memories using embeddings
        Returns list of (memory, similarity_score) tuples
        """
        if not memory_pool:
            return []

        similarities = []
        for memory in memory_pool:
            if memory.embedding is not None:
                sim = self.semantic_similarity(query_embedding, memory.embedding)
                if sim >= threshold:
                    similarities.append((memory, sim))

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def detect_contradiction(
        self, mem1: Memory, mem2: Memory, contradiction_threshold: float = 0.8
    ) -> bool:
        """
        Detect semantic contradiction
        High similarity + opposing sentiment/meaning = contradiction
        """
        if mem1.embedding is None or mem2.embedding is None:
            return False

        similarity = self.semantic_similarity(mem1.embedding, mem2.embedding)

        # Simple heuristic: if very similar but different confidence/importance
        if similarity > contradiction_threshold:
            confidence_diff = abs(mem1.confidence - mem2.confidence)
            if confidence_diff > 0.5:
                return True

        return False


# ============================================================================
# ENHANCED: HIERARCHICAL GOAL MANAGEMENT
# ============================================================================


class GoalHierarchyManager:
    """
    Manages hierarchical goal structure
    Ensures tactical/session goals always support life-level goals
    """

    def __init__(self, embedding_engine: EmbeddingEngine):
        self.embedding_engine = embedding_engine

    def create_goal(
        self,
        content: str,
        level: GoalLevel,
        parent_id: Optional[str] = None,
        importance: float = 0.5,
    ) -> HierarchicalGoal:
        """Create new goal in hierarchy"""
        goal = HierarchicalGoal(
            id=f"goal_{hash(content + str(datetime.now()))}",
            content=content,
            level=level,
            parent_id=parent_id,
            importance=importance,
            embedding=self.embedding_engine.embed_text(content),
        )
        return goal

    def add_goal_to_hierarchy(
        self, hierarchy: Dict[str, HierarchicalGoal], new_goal: HierarchicalGoal
    ) -> Dict[str, HierarchicalGoal]:
        """Add goal and update parent-child relationships"""
        hierarchy[new_goal.id] = new_goal

        # Update parent's children list
        if new_goal.parent_id and new_goal.parent_id in hierarchy:
            parent = hierarchy[new_goal.parent_id]
            if new_goal.id not in parent.children_ids:
                parent.children_ids.append(new_goal.id)

        return hierarchy

    def validate_goal_alignment(
        self,
        hierarchy: Dict[str, HierarchicalGoal],
        proposed_goal: HierarchicalGoal,
        alignment_threshold: float = 0.6,
    ) -> Dict[str, Any]:
        """
        Validate that proposed goal supports parent goals
        """
        if not proposed_goal.parent_id:
            return {
                "aligned": True,
                "alignment_score": 1.0,
                "reasoning": "Top-level life goal",
                "supporting_goals": [],
            }

        parent = hierarchy.get(proposed_goal.parent_id)
        if not parent:
            return {
                "aligned": False,
                "alignment_score": 0.0,
                "reasoning": "Parent goal not found",
                "supporting_goals": [],
            }

        # Check semantic alignment with parent
        if proposed_goal.embedding is not None and parent.embedding is not None:
            alignment_score = self.embedding_engine.semantic_similarity(
                proposed_goal.embedding, parent.embedding
            )
        else:
            alignment_score = 0.5  # Unknown

        supports_chain = self._check_support_chain(hierarchy, proposed_goal)

        aligned = alignment_score >= alignment_threshold

        return {
            "aligned": aligned,
            "alignment_score": alignment_score,
            "reasoning": f"Goal {'supports' if aligned else 'conflicts with'} parent: {parent.content}",
            "supporting_goals": supports_chain,
            "parent_goal": parent.content,
        }

    def _check_support_chain(
        self, hierarchy: Dict[str, HierarchicalGoal], goal: HierarchicalGoal
    ) -> List[str]:
        """Trace support chain up to life-level goals"""
        chain = []
        current = goal

        while current.parent_id:
            parent = hierarchy.get(current.parent_id)
            if not parent:
                break
            chain.append(parent.content)
            current = parent

        return chain

    def get_active_goals_by_level(
        self, hierarchy: Dict[str, HierarchicalGoal], level: Optional[GoalLevel] = None
    ) -> List[HierarchicalGoal]:
        """Get all active (incomplete) goals at specified level"""
        goals = [g for g in hierarchy.values() if not g.completed]
        if level:
            goals = [g for g in goals if g.level == level]
        return goals

    def propagate_progress(
        self,
        hierarchy: Dict[str, HierarchicalGoal],
        goal_id: str,
        progress_delta: float,
    ) -> Dict[str, HierarchicalGoal]:
        """
        When a child goal progresses, propagate to parent
        """
        if goal_id not in hierarchy:
            return hierarchy

        goal = hierarchy[goal_id]
        goal.progress = min(1.0, goal.progress + progress_delta)

        if goal.progress >= 1.0:
            goal.completed = True

        # Propagate to parent
        if goal.parent_id and goal.parent_id in hierarchy:
            parent = hierarchy[goal.parent_id]
            children = [
                hierarchy[cid] for cid in parent.children_ids if cid in hierarchy
            ]

            # Parent progress = average of children progress
            if children:
                avg_progress = sum(c.progress for c in children) / len(children)
                parent.progress = avg_progress

                if avg_progress >= 1.0:
                    parent.completed = True

        return hierarchy


# ============================================================================
# META-MEMORY MANAGER
# ============================================================================


class MetaMemory:
    """
    Enhanced meta-cognitive layer with embeddings and hierarchical goals
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        long_term_memory=None,  # No type restriction, allow any object with store_memory
    ):
        self.violation_threshold = 0.7
        self.embedding_engine = EmbeddingEngine(embedding_service)
        self.goal_manager = GoalHierarchyManager(self.embedding_engine)
        self.long_term_memory = long_term_memory

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        session_id: str = "default",
        **kwargs,
    ) -> Memory:
        """Add memory with automatic embedding generation and optional persistence"""
        memory = Memory(
            content=content,
            type=memory_type,
            confidence=kwargs.get("confidence", 0.8),
            importance=kwargs.get("importance", 0.5),
            stability=kwargs.get("stability", MemoryStability.STABLE),
            source=kwargs.get("source", "system"),
            created_at=datetime.now().isoformat(),
            last_verified=datetime.now().isoformat(),
            session_id=session_id,
            tags=kwargs.get("tags", []),
            reasoning=kwargs.get("reasoning", ""),
            embedding=self.embedding_engine.embed_text(content),
        )

        # Store in long-term memory if available
        if self.long_term_memory and kwargs.get("persist", True):
            try:
                # Duck typing - check if method exists before calling
                if hasattr(self.long_term_memory, "store_memory"):
                    user_id = kwargs.get("user_id", "default")
                    confidence = kwargs.get("confidence", 0.8)
                    importance = kwargs.get("importance", 0.5)
                    stability = kwargs.get("stability", MemoryStability.STABLE)
                    source = kwargs.get("source", "system")
                    reasoning = kwargs.get("reasoning", "")
                    tags = kwargs.get("tags", [])

                    self.long_term_memory.store_memory(
                        session_id=session_id,
                        user_id=user_id,
                        content=content,
                        memory_type=str(memory_type.value),
                        confidence=confidence,
                        importance=importance,
                        tags=tags,
                        metadata={
                            "stability": stability.value,
                            "source": source,
                            "reasoning": reasoning,
                            "meta_memory": True,
                        },
                        generate_embedding=False,  # We already have the embedding
                        embedding=memory.embedding.tolist()
                        if memory.embedding is not None
                        else None,
                    )
                else:
                    print(
                        "[META-MEMORY] Warning: long_term_memory does not support store_memory"
                    )
            except Exception as e:
                print(f"[META-MEMORY] Failed to persist memory: {e}")

        return memory

    def find_related_memories(
        self, query_memory: Memory, memory_pool: List[Memory], threshold: float = 0.7
    ) -> List[Tuple[Memory, float]]:
        """Use embeddings to find semantically related memories"""
        if query_memory.embedding is None:
            return []

        return self.embedding_engine.find_related_memories(
            query_memory.embedding, memory_pool, threshold=threshold
        )

    def detect_memory_conflicts(
        self, new_memory: Memory, memory_pool: List[Memory]
    ) -> Optional[Dict[str, Any]]:
        """Detect conflicts using semantic similarity"""
        related = self.find_related_memories(new_memory, memory_pool, threshold=0.6)

        conflicts = []
        for existing_memory, similarity in related:
            if self.embedding_engine.detect_contradiction(existing_memory, new_memory):
                conflicts.append(
                    {
                        "existing": existing_memory,
                        "new": new_memory,
                        "similarity": similarity,
                        "confidence_delta": abs(
                            existing_memory.confidence - new_memory.confidence
                        ),
                        "resolution_strategy": self._recommend_resolution(
                            existing_memory, new_memory
                        ),
                    }
                )

        if conflicts:
            return {
                "has_conflict": True,
                "conflicts": conflicts,
                "recommended_action": "explicit_resolution",
                "reasoning": f"Semantic conflicts detected with {len(conflicts)} memories",
            }

        return None

    def check_coherence(
        self,
        proposed_action: str,
        narrative_state: NarrativeState,
        long_term_memory: Optional[List[Memory]] = None,
    ) -> Dict[str, Any]:
        """Check coherence with hierarchical goal validation"""
        violations = []
        reasoning = []

        # Create temporary goal for proposed action
        action_goal = self.goal_manager.create_goal(
            content=proposed_action, level=GoalLevel.TACTICAL, importance=0.7
        )

        hierarchy = narrative_state["goal_hierarchy"]

        # Find most relevant parent goal (session-level)
        session_goals = self.goal_manager.get_active_goals_by_level(
            hierarchy, GoalLevel.SESSION
        )

        if session_goals:
            best_alignment = 0.0
            aligned_parent = None

            for session_goal in session_goals:
                action_goal.parent_id = session_goal.id
                alignment = self.goal_manager.validate_goal_alignment(
                    hierarchy, action_goal, alignment_threshold=0.6
                )

                if alignment["alignment_score"] > best_alignment:
                    best_alignment = alignment["alignment_score"]
                    aligned_parent = session_goal

            if best_alignment < self.violation_threshold:
                violations.append(
                    f"Action doesn't support session goal: {aligned_parent.content if aligned_parent else 'unknown'}"
                )

            reasoning.append(
                f"Goal hierarchy alignment: {best_alignment:.2f} - "
                f"{'Supports' if best_alignment >= self.violation_threshold else 'Conflicts with'} "
                f"session goals"
            )

        # Check life-level goal support
        life_goals = self.goal_manager.get_active_goals_by_level(
            hierarchy, GoalLevel.LIFE
        )

        if life_goals:
            life_alignment_scores = []
            for life_goal in life_goals:
                if (
                    action_goal.embedding is not None
                    and life_goal.embedding is not None
                ):
                    score = self.embedding_engine.semantic_similarity(
                        action_goal.embedding, life_goal.embedding
                    )
                    life_alignment_scores.append((life_goal.content, score))

            if life_alignment_scores:
                avg_life_alignment = sum(s for _, s in life_alignment_scores) / len(
                    life_alignment_scores
                )
                reasoning.append(
                    f"Life goal alignment: {avg_life_alignment:.2f} - "
                    f"Action {'advances' if avg_life_alignment > 0.5 else 'may hinder'} life-level objectives"
                )

        # Check constraints
        constraint_check = self._check_constraints(
            proposed_action, narrative_state["constraints"]
        )
        if not constraint_check["passes"]:
            violations.append(f"Constraint violation: {constraint_check['violation']}")

        coherence_score = 1.0 - (len(violations) * 0.2)
        coherence_score = max(0.0, min(1.0, coherence_score))

        return {
            "coherent": len(violations) == 0,
            "coherence_score": coherence_score,
            "violations": violations,
            "reasoning": reasoning,
            "should_proceed": coherence_score >= self.violation_threshold,
        }

    def _recommend_resolution(self, existing: Memory, new: Memory) -> str:
        """Recommend strategy for resolving memory conflict"""
        if new.confidence > existing.confidence + 0.2:
            return "replace_with_new"
        elif existing.stability == MemoryStability.CORE:
            return "keep_existing"
        else:
            return "user_confirmation_needed"

    def _check_constraints(self, action: str, constraints: List[str]) -> Dict[str, Any]:
        """Check constraint violations (placeholder)"""
        return {"passes": True, "status": "No violations", "violation": None}


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================


def create_default_narrative_state() -> Dict[str, Any]:
    """Create a default narrative state for initialization"""
    embedding_service = EmbeddingService()
    meta_memory = MetaMemory(embedding_service)

    # Create default life goal
    life_goal = meta_memory.goal_manager.create_goal(
        "Help user achieve their goals through intelligent assistance",
        GoalLevel.LIFE,
        importance=1.0,
    )

    goal_hierarchy = {}
    goal_hierarchy = meta_memory.goal_manager.add_goal_to_hierarchy(
        goal_hierarchy, life_goal
    )

    return {
        "identity": {
            "role": "AI assistant with persistent memory",
            "values": ["helpful", "accurate", "consistent"],
        },
        "goal_hierarchy": goal_hierarchy,
        "constraints": ["maintain coherence", "respect user preferences"],
        "commitments": [],
        "session_arc": {"phase": "active", "turn_count": 0},
        "coherence_score": 1.0,
    }

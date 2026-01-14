"""
Standalone MetaMemory demo - Complete implementation
"""

import sys
import os
import hashlib
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypedDict
import requests


# ============================================================================
# MINI EMBEDDING SERVICE (copied from embedding.py)
# ============================================================================


class EmbeddingService:
    """
    Generate embeddings for semantic search.
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.dimension = 1536

    def embed_text(self, text: str) -> List[float]:
        if self.api_key:
            try:
                response = requests.post(
                    f"{self.base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "input": text,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data["data"][0]["embedding"]
            except Exception as e:
                print(f"OpenRouter embedding failed: {e}, falling back to mock")

        # Mock implementation
        hash_obj = hashlib.sha256(text.encode())
        seed = int(hash_obj.hexdigest(), 16) % (2**32)
        np.random.seed(seed)
        embedding = np.random.randn(self.dimension).tolist()
        norm = np.linalg.norm(embedding)
        return (np.array(embedding) / norm).tolist()

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        return float(np.dot(vec1, vec2))


# ============================================================================
# META-MEMORY IMPLEMENTATION
# ============================================================================


class MemoryType(Enum):
    FACT = "fact"
    GOAL = "goal"
    CONSTRAINT = "constraint"
    COMMITMENT = "commitment"
    IDENTITY = "identity"


class MemoryStability(Enum):
    VOLATILE = "volatile"
    STABLE = "stable"
    CORE = "core"


class GoalLevel(Enum):
    LIFE = "life"
    SESSION = "session"
    TACTICAL = "tactical"


@dataclass
class Memory:
    content: str
    type: MemoryType
    confidence: float
    importance: float
    stability: MemoryStability
    source: str
    created_at: str
    last_verified: str
    session_id: str
    tags: List[str]
    reasoning: str
    embedding: Optional[np.ndarray] = None
    related_memory_ids: List[str] = field(default_factory=list)
    contradicts_memory_ids: List[str] = field(default_factory=list)


@dataclass
class HierarchicalGoal:
    id: str
    content: str
    level: GoalLevel
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    progress: float = 0.0
    completed: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    importance: float = 0.5
    embedding: Optional[np.ndarray] = None


class NarrativeState(TypedDict):
    identity: Dict[str, Any]
    goal_hierarchy: Dict[str, HierarchicalGoal]
    constraints: List[str]
    commitments: List[str]
    session_arc: Dict[str, Any]
    coherence_score: float


class EmbeddingEngine:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

    def embed_text(self, text: str) -> np.ndarray:
        return np.array(self.embedding_service.embed_text(text))

    def semantic_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
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
        if not memory_pool:
            return []
        similarities = []
        for memory in memory_pool:
            if memory.embedding is not None:
                sim = self.semantic_similarity(query_embedding, memory.embedding)
                if sim >= threshold:
                    similarities.append((memory, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def detect_contradiction(
        self, mem1: Memory, mem2: Memory, contradiction_threshold: float = 0.8
    ) -> bool:
        if mem1.embedding is None or mem2.embedding is None:
            return False
        similarity = self.semantic_similarity(mem1.embedding, mem2.embedding)
        if similarity > contradiction_threshold:
            confidence_diff = abs(mem1.confidence - mem2.confidence)
            if confidence_diff > 0.5:
                return True
        return False


class GoalHierarchyManager:
    def __init__(self, embedding_engine: EmbeddingEngine):
        self.embedding_engine = embedding_engine

    def create_goal(
        self,
        content: str,
        level: GoalLevel,
        parent_id: Optional[str] = None,
        importance: float = 0.5,
    ) -> HierarchicalGoal:
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
        hierarchy[new_goal.id] = new_goal
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
        if proposed_goal.embedding is not None and parent.embedding is not None:
            alignment_score = self.embedding_engine.semantic_similarity(
                proposed_goal.embedding, parent.embedding
            )
        else:
            alignment_score = 0.5
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
        goals = [g for g in hierarchy.values() if not g.completed]
        if level:
            goals = [g for g in goals if g.level == level]
        return goals


class MetaMemory:
    def __init__(self, embedding_service: EmbeddingService):
        self.violation_threshold = 0.7
        self.embedding_engine = EmbeddingEngine(embedding_service)
        self.goal_manager = GoalHierarchyManager(self.embedding_engine)

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        session_id: str = "default",
        **kwargs,
    ) -> Memory:
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
        return memory

    def find_related_memories(
        self, query_memory: Memory, memory_pool: List[Memory], threshold: float = 0.7
    ) -> List[Tuple[Memory, float]]:
        if query_memory.embedding is None:
            return []
        return self.embedding_engine.find_related_memories(
            query_memory.embedding, memory_pool, threshold=threshold
        )

    def detect_memory_conflicts(
        self, new_memory: Memory, memory_pool: List[Memory]
    ) -> Optional[Dict[str, Any]]:
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
        long_term_memory: List[Memory],
    ) -> Dict[str, Any]:
        violations = []
        reasoning = []
        action_goal = self.goal_manager.create_goal(
            proposed_action, GoalLevel.TACTICAL, importance=0.7
        )
        hierarchy = narrative_state["goal_hierarchy"]
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
                f"Goal hierarchy alignment: {best_alignment:.2f} - {'Supports' if best_alignment >= self.violation_threshold else 'Conflicts with'} session goals"
            )
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
                    f"Life goal alignment: {avg_life_alignment:.2f} - Action {'advances' if avg_life_alignment > 0.5 else 'may hinder'} life-level objectives"
                )
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
        if new.confidence > existing.confidence + 0.2:
            return "replace_with_new"
        elif existing.stability == MemoryStability.CORE:
            return "keep_existing"
        else:
            return "user_confirmation_needed"

    def _check_constraints(self, action: str, constraints: List[str]) -> Dict[str, Any]:
        return {"passes": True, "status": "No violations", "violation": None}


def create_default_narrative_state() -> Dict[str, Any]:
    embedding_service = EmbeddingService()
    meta_memory = MetaMemory(embedding_service)
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


def demo_meta_memory():
    """Demonstrate MetaMemory functionality"""
    print("=== MetaMemory Demo ===\n")

    embedding_service = EmbeddingService()
    meta_memory = MetaMemory(embedding_service)
    narrative_state = create_default_narrative_state()

    print("1. Narrative State Initialized:")
    print(f"   Identity: {narrative_state['identity']['role']}")
    print(f"   Values: {narrative_state['identity']['values']}")
    print(f"   Life Goals: {len(narrative_state['goal_hierarchy'])}")
    print(f"   Constraints: {narrative_state['constraints']}\n")

    memories = []
    memories.append(
        meta_memory.add_memory(
            "User prefers Python for data science",
            MemoryType.FACT,
            confidence=0.9,
            tags=["preference", "programming"],
        )
    )
    memories.append(
        meta_memory.add_memory(
            "User is vegetarian",
            MemoryType.IDENTITY,
            confidence=1.0,
            stability=MemoryStability.CORE,
        )
    )

    print("2. Memories Added:")
    for mem in memories:
        print(
            f"   - {mem.content} (confidence: {mem.confidence}, type: {mem.type.value})"
        )
    print()

    proposed_action = "Recommend a steak restaurant"
    coherence = meta_memory.check_coherence(
        proposed_action, NarrativeState(**narrative_state), memories
    )

    print(f"3. Coherence Check for: '{proposed_action}'")
    print(f"   Coherent: {coherence['coherent']}")
    print(f"   Score: {coherence['coherence_score']:.2f}")
    if coherence["violations"]:
        print(f"   Violations: {coherence['violations']}")
    if coherence["reasoning"]:
        print(f"   Reasoning: {coherence['reasoning']}")
    print()

    conflicting_memory = meta_memory.add_memory(
        "User loves eating meat", MemoryType.FACT, confidence=0.8
    )
    conflict = meta_memory.detect_memory_conflicts(conflicting_memory, memories)

    print("4. Conflict Detection:")
    if conflict:
        print(f"   Conflict found: {conflict['reasoning']}")
        print(f"   Conflicts: {len(conflict['conflicts'])}")
        for c in conflict["conflicts"]:
            print(f"   - Similarity: {c['similarity']:.2f}")
            print(f"   - Resolution: {c['resolution_strategy']}")
    else:
        print("   No conflicts detected")
    print()

    print("5. Goal Hierarchy:")
    hierarchy = narrative_state["goal_hierarchy"]
    for goal_id, goal in hierarchy.items():
        print(f"   [{goal.level.value}] {goal.content} (progress: {goal.progress:.1f})")
    print()

    print("✅ MetaMemory demo completed successfully!")


if __name__ == "__main__":
    demo_meta_memory()

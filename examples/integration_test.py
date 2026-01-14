"""
MetaMemory Integration Test - Simplified version
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memory.meta_memory import (
    MetaMemory,
    create_default_narrative_state,
)
from memory.types import MemoryType, MemoryStability, GoalLevel
from memory.embedding import EmbeddingService


def demo_meta_memory_integration():
    """Test MetaMemory with real embeddings"""

    print("=== MetaMemory Integration Test ===\n")

    # Initialize with real embeddings
    embedding_service = EmbeddingService()
    meta_memory = MetaMemory(embedding_service)

    print("1. Initialized MetaMemory with embedding service")
    print(
        f"   API Key available: {'Yes' if os.getenv('OPENROUTER_API_KEY') else 'No (using mock)'}"
    )

    # Create narrative state
    narrative_state = create_default_narrative_state()
    print("\n2. Created narrative state with hierarchical goals")

    # Add some memories
    memories = []
    memories.append(
        meta_memory.add_memory(
            "User prefers Python for data science projects",
            MemoryType.FACT,
            confidence=0.9,
            tags=["preference", "programming", "data-science"],
        )
    )
    memories.append(
        meta_memory.add_memory(
            "User is vegetarian and avoids meat products",
            MemoryType.IDENTITY,
            confidence=1.0,
            stability=MemoryStability.CORE,
        )
    )

    print("\n3. Added memories with embeddings:")
    for mem in memories:
        print(f"   - {mem.content[:50]}...")
        print(f"     Type: {mem.type.value}, Confidence: {mem.confidence}")
        print(
            f"     Embedding dim: {len(mem.embedding) if mem.embedding is not None else 'None'}"
        )

    # Test coherence checking
    test_actions = [
        "Recommend a steakhouse for dinner",
        "Suggest Python data science courses",
        "Tell a joke about programming",
    ]

    print("\n4. Coherence Analysis:")
    for action in test_actions:
        coherence = meta_memory.check_coherence(action, narrative_state, memories)
        status = "✅ PASS" if coherence["should_proceed"] else "❌ FAIL"
        print(
            f"   '{action[:30]}...': {status} (score: {coherence['coherence_score']:.2f})"
        )
        if coherence["violations"]:
            print(f"      Issue: {coherence['violations'][0][:50]}...")

    # Test conflict detection
    conflicting_memory = meta_memory.add_memory(
        "User loves burgers and steaks", MemoryType.FACT, confidence=0.8
    )

    conflict = meta_memory.detect_memory_conflicts(conflicting_memory, memories)
    print("\n5. Conflict Detection:")
    if conflict and conflict["has_conflict"]:
        print(
            f"   ⚠️  Detected {conflict['conflicts'][0]['similarity']:.2f} similarity conflict"
        )
        print(f"   Resolution: {conflict['conflicts'][0]['resolution_strategy']}")
    else:
        print("   ✅ No conflicts detected")

    print("\n6. Goal Hierarchy Status:")
    hierarchy = narrative_state["goal_hierarchy"]
    for goal_id, goal in hierarchy.items():
        print(
            f"   {goal.level.value}: {goal.content[:40]}... (progress: {goal.progress:.1f})"
        )

    print("\n🎉 MetaMemory integration test completed!")
    print("Ready for full database integration when containers are available.")


if __name__ == "__main__":
    demo_meta_memory_integration()

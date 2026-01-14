"""
Full integration demo with real databases and MetaMemory
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memory import MemoryManager, AgentState


def demo_full_integration():
    """Demonstrate full MetaMemory integration with databases"""

    print("=== Full MetaMemory Integration Demo ===\n")

    # Initialize with real databases
    mm = MemoryManager(
        redis_url="redis://localhost:6379",
        postgres_url="postgresql://memory_user:memory_pass@localhost:5432/memory_db",
    )

    # Create agent state
    state = AgentState(
        messages=[
            {"role": "user", "content": "Hello, I'm looking for Python resources"}
        ],
        conversation_history=[],
        working_memory={},
        session_id="demo_session_001",
        user_id="demo_user_001",
        turn_count=1,
        agent_outputs={},
        next_agent="",
        final_response="",
        last_consolidation_turn=0,
        memory_importance_scores={},
        narrative_state={},  # Will be loaded
        coherence_violations=[],
        meta_reasoning="",
        long_term_memory=[],
    )

    # Load or initialize narrative state
    state["narrative_state"] = mm.load_narrative_state(state)

    # Add a session goal
    meta_memory = mm.meta_memory
    from memory.meta_memory import GoalLevel

    session_goal = meta_memory.goal_manager.create_goal(
        "Help user find Python learning resources",
        GoalLevel.SESSION,
        parent_id=list(state["narrative_state"]["goal_hierarchy"].keys())[
            0
        ],  # First life goal
        importance=0.8,
    )

    hierarchy = state["narrative_state"]["goal_hierarchy"]
    hierarchy = meta_memory.goal_manager.add_goal_to_hierarchy(hierarchy, session_goal)
    state["narrative_state"]["goal_hierarchy"] = hierarchy

    print("1. Session initialized with narrative state")
    print(
        f"   Life goals: {len([g for g in hierarchy.values() if g.level == GoalLevel.LIFE])}"
    )
    print(
        f"   Session goals: {len([g for g in hierarchy.values() if g.level == GoalLevel.SESSION])}"
    )

    # Simulate conversation and memory storage
    messages = [
        {"role": "user", "content": "I'm new to Python and want to learn data science"},
        {
            "role": "assistant",
            "content": "I'll help you find the best Python resources for data science",
        },
        {"role": "user", "content": "I prefer online courses over books"},
    ]

    for i, msg in enumerate(messages):
        state["turn_count"] = i + 1
        state["messages"].append(msg)

        # Save interaction (this will add to MetaMemory)
        mm.save_interaction(state, msg)

    print("\n2. Stored conversation in MetaMemory system")
    print(f"   Messages processed: {len(messages)}")
    print(f"   Memories in state: {len(state['long_term_memory'])}")

    # Test coherence checking
    proposed_actions = [
        "Recommend Python data science books",
        "Suggest online Python courses",
        "Tell a joke about programming",
    ]

    print("\n3. Coherence Checking:")
    for action in proposed_actions:
        coherence = meta_memory.check_coherence(action, state["narrative_state"])
        print(
            f"   '{action}': {'✅ PASS' if coherence['should_proceed'] else '❌ FAIL'} (score: {coherence['coherence_score']:.2f})"
        )
        if coherence["violations"]:
            print(f"      Violations: {coherence['violations'][0]}")

    # End session and persist
    mm.end_session(state)

    print("\n4. Session ended and data persisted")
    print("   ✅ Narrative state saved to PostgreSQL")
    print("   ✅ Memories consolidated")
    print("   ✅ Session arc updated")

    print("\n🎉 Full MetaMemory integration demo completed!")


if __name__ == "__main__":
    demo_full_integration()

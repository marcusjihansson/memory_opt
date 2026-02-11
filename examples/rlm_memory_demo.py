"""
RLM Memory Demo - Demonstrates RLM-powered memory exploration and reasoning.

This example shows how to use RLM (Recursive Language Models) for intelligent
memory exploration, multi-hop reasoning, and synthesis.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dspy
from dotenv import load_dotenv

from memory.rlm_memory import (
    ProductionRLMMemory,
    RLMMemoryExplorer,
    RLMMemoryReasoner,
    RLMMemorySynthesizer,
    format_memories_for_rlm,
)

load_dotenv()


def configure_dspy():
    """Configure DSPy with available LLM provider."""
    if os.getenv("OPENAI_API_KEY"):
        lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    elif os.getenv("ANTHROPIC_API_KEY"):
        lm = dspy.LM(
            "anthropic/claude-3-5-sonnet-20241022",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    elif os.getenv("OPENROUTER_API_KEY"):
        lm = dspy.LM("openrouter/openai/gpt-4o-mini", api_key=os.getenv("OPENROUTER_API_KEY"))
    else:
        print(
            "Warning: No API key found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY"
        )
        print("Running in mock mode for demonstration purposes.\n")
        return False

    dspy.configure(lm=lm)
    return True


def create_sample_memories():
    """Create sample memories for demonstration."""
    return [
        {
            "content": "User Marcus prefers Python for data science projects and has been using it for 5 years",
            "timestamp": "2024-01-15",
            "importance": 0.9,
            "type": "fact",
        },
        {
            "content": "User expressed interest in learning Rust for systems programming",
            "timestamp": "2024-02-01",
            "importance": 0.7,
            "type": "goal",
        },
        {
            "content": "Previous project involved building a recommendation system using collaborative filtering",
            "timestamp": "2024-01-20",
            "importance": 0.8,
            "type": "experience",
        },
        {
            "content": "User mentioned they work best in the morning and prefer detailed explanations",
            "timestamp": "2024-02-10",
            "importance": 0.6,
            "type": "preference",
        },
        {
            "content": "Completed a machine learning course on neural networks last month",
            "timestamp": "2024-01-28",
            "importance": 0.75,
            "type": "achievement",
        },
        {
            "content": "User is building an AI agent that needs persistent memory across sessions",
            "timestamp": "2024-02-15",
            "importance": 0.95,
            "type": "goal",
        },
        {
            "content": "Expressed frustration with RAG chunking losing context in long documents",
            "timestamp": "2024-02-12",
            "importance": 0.8,
            "type": "feedback",
        },
        {
            "content": "User's company is in the healthcare domain requiring HIPAA compliance",
            "timestamp": "2024-01-10",
            "importance": 0.85,
            "type": "constraint",
        },
    ]


def demo_basic_exploration():
    """Demonstrate basic RLM memory exploration."""
    print("=" * 60)
    print("1. BASIC RLM MEMORY EXPLORATION")
    print("=" * 60 + "\n")

    memories = create_sample_memories()
    memory_context = format_memories_for_rlm(memories)

    print("Memory context loaded with", len(memories), "memories\n")

    explorer = RLMMemoryExplorer(max_iterations=10)

    queries = [
        "What programming languages does the user know?",
        "What are the user's current goals?",
        "What constraints should I be aware of?",
    ]

    for query in queries:
        print(f"Query: {query}")
        try:
            result = explorer(memory_context=memory_context, query=query, search_depth=3)
            print(f"  Relevant memories: {result.relevant_memories}")
            print(f"  Confidence: {result.confidence}")
        except Exception as e:
            print(f"  [Mock mode] Would explore memories for: {query}")
            print(f"  Error: {e}")
        print()


def demo_multi_hop_reasoning():
    """Demonstrate multi-hop reasoning across memories."""
    print("=" * 60)
    print("2. MULTI-HOP REASONING")
    print("=" * 60 + "\n")

    memories = create_sample_memories()
    memory_bank = format_memories_for_rlm(memories)

    reasoner = RLMMemoryReasoner(max_iterations=15)

    question = "Given the user's experience with Python and ML, and their current goal of building an AI agent with memory, what technical approach would you recommend?"

    print(f"Question: {question}\n")

    try:
        result = reasoner(memory_bank=memory_bank, question=question)
        print(f"Reasoning chain: {result.reasoning_chain}")
        print(f"Answer: {result.answer}")
        print(f"Method used: {result.method}")
    except Exception as e:
        print("[Mock mode] Would perform multi-hop reasoning")
        print(f"Error: {e}")
    print()


def demo_memory_synthesis():
    """Demonstrate memory synthesis across temporal ranges."""
    print("=" * 60)
    print("3. MEMORY SYNTHESIS")
    print("=" * 60 + "\n")

    memories = create_sample_memories()
    memory_database = format_memories_for_rlm(memories)

    synthesizer = RLMMemorySynthesizer(max_iterations=15)

    task = "Synthesize a user profile based on all available memories"

    print(f"Task: {task}\n")

    try:
        result = synthesizer(
            memory_database=memory_database,
            task=task,
            temporal_range="all time",
        )
        print(f"Search plan: {result.search_plan}")
        print(f"Synthesis: {result.synthesis}")
        print(f"Supporting evidence: {result.supporting_evidence}")
        print(f"Method used: {result.method}")
    except Exception as e:
        print("[Mock mode] Would synthesize memories")
        print(f"Error: {e}")
    print()


def demo_production_rlm():
    """Demonstrate production RLM with all capabilities."""
    print("=" * 60)
    print("4. PRODUCTION RLM MEMORY SYSTEM")
    print("=" * 60 + "\n")

    memories = create_sample_memories()
    memory_context = format_memories_for_rlm(memories)

    production = ProductionRLMMemory(max_iterations=15)

    print("Using ProductionRLMMemory for comprehensive memory operations:\n")

    # Test exploration
    print("a) Exploring memories...")
    try:
        explore_result = production.explore(
            memory_context=memory_context,
            query="user preferences",
        )
        print(f"   Found: {explore_result.relevant_memories}")
    except Exception as e:
        print(f"   [Mock mode] Exploration error: {e}")

    # Test reasoning
    print("\nb) Reasoning over memories...")
    try:
        reason_result = production.reason(
            memory_bank=memory_context,
            question="What should I prioritize when helping this user?",
        )
        print(f"   Answer: {reason_result.answer}")
        print(f"   Method: {reason_result.method}")
    except Exception as e:
        print(f"   [Mock mode] Reasoning error: {e}")

    # Test synthesis
    print("\nc) Synthesizing insights...")
    try:
        synth_result = production.synthesize(
            memory_database=memory_context,
            task="Create a summary of user's technical background",
            temporal_range="last month",
        )
        print(f"   Synthesis: {synth_result.synthesis}")
    except Exception as e:
        print(f"   [Mock mode] Synthesis error: {e}")

    print()


def demo_comparison():
    """Compare traditional retrieval vs RLM retrieval."""
    print("=" * 60)
    print("5. TRADITIONAL VS RLM RETRIEVAL")
    print("=" * 60 + "\n")

    # Note: We create memories to show they're available for both approaches
    _ = create_sample_memories()  # Demonstrates memory format compatibility

    print("Traditional approach:")
    print("  - Pre-chunk memories")
    print("  - Embed chunks")
    print("  - Vector similarity search")
    print("  - Return top-k chunks")
    print("  - Context may be fragmented\n")

    print("RLM approach:")
    print("  - Load entire memory context")
    print("  - RLM recursively explores")
    print("  - Model adjusts search strategy")
    print("  - Multi-hop reasoning supported")
    print("  - No chunking loss\n")

    print("Key advantages of RLM for memory systems:")
    print("  1. Iterative refinement - can search, find partial matches, explore related")
    print("  2. No chunking loss - explores entire context without pre-chunking")
    print("  3. Context-aware retrieval - adjusts strategy based on findings")
    print("  4. Multi-hop reasoning - natural support for reasoning chains")
    print()


def main():
    """Run all RLM memory demonstrations."""
    print("\n" + "=" * 60)
    print("RLM MEMORY SYSTEM DEMONSTRATION")
    print("=" * 60 + "\n")

    has_api = configure_dspy()

    if not has_api:
        print("Note: Running demos without API - showing structure only.\n")

    demo_basic_exploration()
    demo_multi_hop_reasoning()
    demo_memory_synthesis()
    demo_production_rlm()
    demo_comparison()

    print("=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nTo use RLM memory in your project:")
    print("  from memory import ProductionRLMMemory, format_memories_for_rlm")
    print("  rlm = ProductionRLMMemory()")
    print("  result = rlm(context=memories, question='your question')")
    print()


if __name__ == "__main__":
    main()

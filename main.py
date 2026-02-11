"""
Main integration demo - Four-tier memory system with all layers enabled.

Demonstrates:
1. CachedMemory (Redis) - Fast context caching
2. ShortTermMemory - Importance scoring & conversation history
3. LongTermMemory (PostgreSQL+pgvector) - Persistent semantic search
4. MetaMemory - Hierarchical goals & coherence validation
5. RLMMemory (DSPy) - Recursive exploration & multi-hop reasoning
"""

import argparse
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import dspy
from dotenv import load_dotenv

from memory import AgentState, EmbeddingService, GoalLevel, MemoryManager, ProductionRLMMemory
from memory.meta_memory import MetaMemory, create_default_narrative_state
from memory.short_term_memory import ShortTermMemory

load_dotenv()


def configure_dspy() -> bool:
    """Configure DSPy with available LLM provider."""
    if os.getenv("OPENAI_API_KEY"):
        lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    elif os.getenv("ANTHROPIC_API_KEY"):
        lm = dspy.LM(
            "anthropic/claude-3-5-sonnet-20241022",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    elif os.getenv("OPENROUTER_API_KEY"):
        lm = dspy.LM(
            "openrouter/openai/gpt-4o-mini",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    else:
        print("[CONFIG] No API key found. RLM features will use fallback mode.")
        return False

    dspy.configure(lm=lm)
    print("[CONFIG] DSPy configured successfully")
    return True


class MockMemoryManager:
    """Mock memory manager for demo without databases."""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.meta_memory = MetaMemory(self.embedding_service)
        self.short_term = ShortTermMemory()
        self._memories: list[dict[str, Any]] = []
        self._cache: dict[str, Any] = {}

    def load_narrative_state(self, state: AgentState) -> dict[str, Any]:
        return create_default_narrative_state()

    def get_context_for_agent(
        self, state: AgentState, include_historical: bool = False
    ) -> str:
        context_parts = []
        if self._cache.get(state["session_id"]):
            context_parts.append(f"[CACHED] {self._cache[state['session_id']]}")
        context_parts.append(self.short_term.build_context_window(state))
        if include_historical and self._memories:
            context_parts.append("\n[HISTORICAL MEMORIES]")
            for mem in self._memories[-5:]:
                context_parts.append(f"  - {mem['content'][:100]}")
        return "\n".join(context_parts)

    def save_interaction(
        self,
        state: AgentState,
        message: dict[str, str],
        agent_result: dict | None = None,
    ):
        importance = self.short_term.calculate_importance(
            message,
            {"turn_count": state["turn_count"], "message_turn": state["turn_count"]},
        )
        self._memories.append(
            {
                "content": message["content"],
                "role": message["role"],
                "importance": importance,
                "turn": state["turn_count"],
            }
        )
        self._cache[state["session_id"]] = {
            "last_message": message,
            "turn": state["turn_count"],
        }
        print(f"    [CACHED] Context cached for session {state['session_id']}")
        print(f"    [SHORT-TERM] Importance score: {importance:.2f}")
        print("    [LONG-TERM] Stored message (mock)")

    def consolidate_memories(self, state: AgentState):
        importance_scores = state.get("memory_importance_scores", {})
        important_count = sum(1 for s in importance_scores.values() if s > 0.6)
        print(
            f"    [CONSOLIDATION] {important_count} high-importance memories identified"
        )

    def end_session(self, state: AgentState, *, clear_cache: bool = True):
        self.consolidate_memories(state)
        print(f"    [SESSION] Ended session {state['session_id']}")


def create_agent_state(session_id: str, user_id: str) -> AgentState:
    """Create initial agent state."""
    return AgentState(
        messages=[],
        conversation_history=[],
        working_memory={},
        session_id=session_id,
        user_id=user_id,
        turn_count=0,
        agent_outputs={},
        next_agent="",
        final_response="",
        last_consolidation_turn=0,
        memory_importance_scores={},
        narrative_state={},
        coherence_violations=[],
        meta_reasoning="",
        long_term_memory=[],
    )


def setup_session_goals(meta_memory: MetaMemory, state: AgentState) -> None:
    """Set up session-level goals and constraints."""
    hierarchy = state["narrative_state"]["goal_hierarchy"]

    life_goal_id = list(hierarchy.keys())[0]
    session_goal = meta_memory.goal_manager.create_goal(
        "Help user learn Python for data science with hands-on tutorials",
        GoalLevel.SESSION,
        parent_id=life_goal_id,
        importance=0.9,
    )
    hierarchy = meta_memory.goal_manager.add_goal_to_hierarchy(hierarchy, session_goal)

    state["narrative_state"]["goal_hierarchy"] = hierarchy
    state["narrative_state"]["constraints"].extend(
        [
            "User prefers hands-on tutorials with code examples",
            "Project requires HIPAA compliance for healthcare data",
            "Focus on Python and data science topics",
        ]
    )


def process_user_message(
    mm: MemoryManager | MockMemoryManager,
    state: AgentState,
    content: str,
) -> None:
    """Process a user message through all memory layers."""
    state["turn_count"] += 1
    message = {"role": "user", "content": content}
    state["messages"].append(message)

    history, importance_scores = ShortTermMemory.update_conversation_history(
        state, message, max_history=20
    )
    state["conversation_history"] = history
    state["memory_importance_scores"] = importance_scores

    print(f"\n  [USER] Turn {state['turn_count']}: {content[:60]}...")
    mm.save_interaction(state, message)

    if ShortTermMemory.should_consolidate(state):
        print("    [TRIGGER] Consolidation threshold reached")
        mm.consolidate_memories(state)
        state["last_consolidation_turn"] = state["turn_count"]


def check_action_coherence(
    meta_memory: MetaMemory,
    state: AgentState,
    proposed_action: str,
) -> dict[str, Any]:
    """Check if a proposed action is coherent with goals and constraints."""
    print(f'\n  [ACTION CHECK] "{proposed_action}"')

    coherence = meta_memory.check_coherence(
        proposed_action,
        state["narrative_state"],
    )

    # Enhance with explicit constraint checking (works without real embeddings)
    constraint_violations = check_explicit_constraints(
        proposed_action, state["narrative_state"]["constraints"]
    )

    if constraint_violations:
        coherence["violations"].extend(constraint_violations)
        coherence["coherence_score"] = max(
            0.0, coherence["coherence_score"] - 0.3 * len(constraint_violations)
        )
        coherence["should_proceed"] = coherence["coherence_score"] >= 0.7

    status = "PASS" if coherence["should_proceed"] else "FAIL"
    print(
        f"    [META-MEMORY] Coherence: {status} (score: {coherence['coherence_score']:.2f})"
    )

    if coherence["violations"]:
        for v in coherence["violations"]:
            print(f"    [VIOLATION] {v}")

    if coherence["reasoning"]:
        for r in coherence["reasoning"][:2]:
            print(f"    [REASONING] {r}")

    return coherence


def check_explicit_constraints(action: str, constraints: list[str]) -> list[str]:
    """
    Explicit constraint checking using keyword matching.
    This works reliably even without real embeddings.
    """
    violations = []
    action_lower = action.lower()

    # Define constraint rules
    rules = [
        {
            "keywords": [
                "javascript",
                "java ",
                "ruby",
                "go ",
                "golang",
                "rust",
                "switching to",
            ],
            "anti_keywords": ["python"],
            "constraint": "Focus on Python and data science topics",
            "violation": "Suggests non-Python technology when user is focused on Python",
        },
        {
            "keywords": [
                "without code",
                "no code",
                "without example",
                "no example",
                "without any code",
            ],
            "anti_keywords": ["with code", "with example"],
            "constraint": "User prefers hands-on tutorials with code examples",
            "violation": "Violates user preference for code examples",
        },
        {
            "keywords": [
                "unencrypted",
                "plain text file",
                "unsecured",
                "public storage",
                "not encrypted",
            ],
            "anti_keywords": [
                " encrypted",
                "securely",
                "hipaa compliant",
            ],  # space before encrypted to avoid 'unencrypted' match
            "constraint": "Project requires HIPAA compliance",
            "violation": "Violates HIPAA compliance requirement for healthcare data",
        },
    ]

    for rule in rules:
        has_violation_keyword = any(kw in action_lower for kw in rule["keywords"])
        has_anti_keyword = any(kw in action_lower for kw in rule["anti_keywords"])

        if has_violation_keyword and not has_anti_keyword:
            violations.append(f"{rule['violation']} (constraint: {rule['constraint']})")

    return violations


def test_memory_retrieval(
    mm: MemoryManager | MockMemoryManager,
    state: AgentState,
    query: str,
) -> str:
    """Test memory retrieval using context building."""
    print(f'\n  [RETRIEVAL] Query: "{query}"')

    start = time.time()
    context = mm.get_context_for_agent(state, include_historical=True)
    elapsed = (time.time() - start) * 1000

    print(f"    [CONTEXT] Retrieved in {elapsed:.1f}ms")
    print(f"    [CONTEXT] Length: {len(context)} chars")

    return context


def test_rlm_reasoning(
    context: str,
    question: str,
    has_api: bool,
) -> None:
    """Test RLM-powered reasoning over memories."""
    print(f'\n  [RLM] Question: "{question}"')

    if not has_api:
        print("    [RLM] Skipped - no API key configured")
        return

    try:
        rlm = ProductionRLMMemory(max_iterations=10)
        start = time.time()
        result = rlm(context=context, question=question, use_fallback=True)
        elapsed = (time.time() - start) * 1000

        print(f"    [RLM] Method: {result.method}")
        print(f"    [RLM] Time: {elapsed:.1f}ms")
        print(f"    [RLM] Answer: {result.answer[:200]}...")
    except Exception as e:
        print(f"    [RLM] Error: {e}")


def _print_persistence_verification(mm: MemoryManager, state: AgentState) -> None:
    print("\n" + "-" * 70)
    print("PERSISTENCE VERIFICATION")
    print("-" * 70)

    # Redis
    try:
        redis_client = mm.cached.redis_client
        total_keys = int(redis_client.dbsize())
        print(f"[REDIS] Total keys: {total_keys}")

        session_id = state["session_id"]
        matched = []
        for key in redis_client.scan_iter(match=f"*{session_id}*", count=100):
            matched.append(key)
            if len(matched) >= 10:
                break
        if matched:
            print(f"[REDIS] Sample keys matching '*{session_id}*': {matched}")
        else:
            print(f"[REDIS] No keys found matching '*{session_id}*'")
    except Exception as e:
        print(f"[REDIS] Verification failed: {e}")

    # Postgres
    try:
        import psycopg

        conn_string = mm.long_term.conn_string
        with psycopg.connect(conn_string) as conn:
            vector_installed = conn.execute(
                "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            print(f"[POSTGRES] pgvector installed: {'yes' if vector_installed else 'no'}")

            tables = [
                "conversations",
                "user_profiles",
                "session_summaries",
                "consolidation_log",
                "narrative_states",
            ]
            for t in tables:
                exists = conn.execute(
                    "SELECT to_regclass(%s)",
                    (f"public.{t}",),
                ).fetchone()[0]
                print(f"[POSTGRES] {t}: {'present' if exists else 'missing'}")

            # Counts (only if conversations exists)
            conv_exists = conn.execute(
                "SELECT to_regclass('public.conversations')"
            ).fetchone()[0]
            if conv_exists:
                total = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[
                    0
                ]
                session = conn.execute(
                    "SELECT COUNT(*) FROM conversations WHERE session_id = %s",
                    (state["session_id"],),
                ).fetchone()[0]
                user = conn.execute(
                    "SELECT COUNT(*) FROM conversations WHERE user_id = %s",
                    (state["user_id"],),
                ).fetchone()[0]
                print(f"[POSTGRES] conversations rows: total={total}, session={session}, user={user}")
    except Exception as e:
        print(f"[POSTGRES] Verification failed: {e}")


def run_demo(mock_mode: bool = False, *, keep_cache: bool = False) -> None:
    """Run the full memory system demo."""
    print("\n" + "=" * 70)
    print("FOUR-TIER MEMORY SYSTEM DEMO")
    print("=" * 70)

    has_api = configure_dspy()

    print("\n[INIT] Initializing memory system...")
    if mock_mode:
        print("[INIT] Running in MOCK mode (no databases)")
        mm = MockMemoryManager()
    else:
        try:
            mm = MemoryManager(
                redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
                postgres_url=os.getenv("DATABASE_URL")
                or os.getenv(
                    "POSTGRES_URL",
                    "postgresql://memory_user:memory_pass@localhost:5432/memorydb",
                ),
                rlm_enabled=True,
            )
            print("[INIT] Connected to Redis and PostgreSQL")
        except Exception as e:
            print(f"[INIT] Database connection failed: {e}")
            print("[INIT] Falling back to MOCK mode")
            mm = MockMemoryManager()

    state = create_agent_state("demo_session_001", "demo_user_001")
    state["narrative_state"] = mm.load_narrative_state(state)
    setup_session_goals(mm.meta_memory, state)

    print("\n[SETUP] Session initialized")
    print(f"  Session ID: {state['session_id']}")
    print(f"  User ID: {state['user_id']}")
    print(f"  Constraints: {len(state['narrative_state']['constraints'])}")

    # Phase 1: Build memory context
    print("\n" + "-" * 70)
    print("PHASE 1: Building Memory Context")
    print("-" * 70)

    user_messages = [
        "I'm learning Python for data science",
        "I prefer hands-on tutorials over reading documentation",
        "I'm working on a healthcare project that requires HIPAA compliance",
        "Remember: I always want code examples with detailed explanations",
    ]

    for msg in user_messages:
        process_user_message(mm, state, msg)

    # Phase 2: Test Short-term Memory importance scoring
    print("\n" + "-" * 70)
    print("PHASE 2: Short-term Memory Analysis")
    print("-" * 70)

    print("\n  [SHORT-TERM] Importance scores:")
    for msg_id, score in sorted(
        state["memory_importance_scores"].items(),
        key=lambda x: x[1],
        reverse=True,
    )[:5]:
        print(f"    {msg_id}: {score:.2f}")

    # Phase 3: Test Long-term Memory retrieval
    print("\n" + "-" * 70)
    print("PHASE 3: Long-term Memory Retrieval")
    print("-" * 70)

    context = test_memory_retrieval(
        mm, state, "What are the user's learning preferences?"
    )

    # Phase 4: Test MetaMemory coherence validation
    print("\n" + "-" * 70)
    print("PHASE 4: MetaMemory Coherence Validation")
    print("-" * 70)

    test_actions = [
        ("Suggest switching to JavaScript for web development", False),
        ("Recommend a tutorial without any code examples", False),
        ("Store patient data in an unencrypted plain text file", False),
        ("Suggest a Python pandas tutorial with hands-on examples", True),
    ]

    results = []
    for action, expected_pass in test_actions:
        coherence = check_action_coherence(mm.meta_memory, state, action)
        actual_pass = coherence["should_proceed"]
        match = "OK" if actual_pass == expected_pass else "UNEXPECTED"
        results.append((action[:40], expected_pass, actual_pass, match))

    print("\n  [SUMMARY] Coherence test results:")
    print("  " + "-" * 60)
    for action, expected, actual, match in results:
        exp_str = "PASS" if expected else "FAIL"
        act_str = "PASS" if actual else "FAIL"
        print(f"    {action}... | Expected: {exp_str} | Got: {act_str} | {match}")

    # Phase 5: Test RLM reasoning
    print("\n" + "-" * 70)
    print("PHASE 5: RLM Multi-hop Reasoning")
    print("-" * 70)

    process_user_message(
        mm, state, "Based on everything you know about me, recommend a project"
    )

    test_rlm_reasoning(
        context,
        "Given the user's interest in Python, data science, healthcare domain with HIPAA requirements, "
        "and preference for hands-on learning, what project would you recommend and why?",
        has_api,
    )

    # End session
    print("\n" + "-" * 70)
    print("SESSION CLEANUP")
    print("-" * 70)

    if isinstance(mm, MemoryManager):
        # Keep cache temporarily so verification can inspect it; optionally clear after.
        mm.end_session(state, clear_cache=False)
        _print_persistence_verification(mm, state)
        if not keep_cache:
            mm.cached.invalidate_session(state["session_id"])
    else:
        mm.end_session(state)

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print(f"\nTotal turns: {state['turn_count']}")
    print(f"Memories stored: {len(state.get('long_term_memory', []))}")
    print(
        f"Coherence tests: {len(test_actions)} ({sum(1 for r in results if r[3] == 'OK')}/{len(results)} as expected)"
    )


def main():
    parser = argparse.ArgumentParser(description="Four-tier memory system demo")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode without databases",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep Redis cache after run for inspection",
    )
    args = parser.parse_args()

    run_demo(mock_mode=args.mock, keep_cache=args.keep_cache)


if __name__ == "__main__":
    main()

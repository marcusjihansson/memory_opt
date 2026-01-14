"""
Memory-enabled DSPy example using the integrated memory system.
This demonstrates how to wrap DSPy programs with memory capabilities.
"""

import os

import dspy
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver

from src.memory.graph.construction import build_graph

# Load environment variables
load_dotenv()

# Configure DSPy with language model
lm = dspy.LM("openrouter/openai/gpt-3.5-turbo")
dspy.configure(lm=lm)


# Define DSPy signature and module (same as sample_agent.py but memory-aware)
class SentimentSignature(dspy.Signature):
    """Analyze the sentiment of a text."""

    text = dspy.InputField(desc="The text to analyze")
    memory_context = dspy.InputField(
        desc="Structured memory context from previous interactions", required=False
    )
    sentiment = dspy.OutputField(desc="The sentiment: positive, negative, or neutral")


class MemoryEnabledSentimentAnalyzer(dspy.Module):
    """
    DSPy module that can leverage memory context for better sentiment analysis.
    The memory_context can include past sentiments, user preferences, etc.
    """

    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(SentimentSignature)

    def forward(self, text, memory_context=None):
        # Use memory context if available to inform analysis
        if memory_context:
            # Example: Check for user sentiment patterns
            if "cached" in memory_context and memory_context["cached"]:
                # Extract past sentiments from cache (implementation depends on cache structure)
                pass

            # You can modify the prompt based on memory here
            enriched_text = (
                f"Previous context: {memory_context}\n\nText to analyze: {text}"
            )
        else:
            enriched_text = text

        result = self.analyze(text=enriched_text)
        return result.sentiment


def main():
    """Demonstrate memory-enabled DSPy with LangGraph orchestration"""

    # Database URLs from environment
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    postgres_url = os.getenv(
        "POSTGRES_URL", "postgresql://user:pass@localhost:5432/memory"
    )

    # Initialize Postgres checkpointer for persistence
    with PostgresSaver.from_conn_string(postgres_url) as checkpointer:
        checkpointer.setup()  # Create tables if needed

        # Build the memory-enabled graph
        app = build_graph(redis_url, postgres_url, checkpointer)

        # Example conversation with memory
        session_id = "demo_session_123"
        user_id = "user_456"

        # Initial state
        state = {
            "session_id": session_id,
            "user_id": user_id,
            "messages": [],
            "turn_count": 0,
            "agent_outputs": {},
        }

        # Add user message
        user_message = {
            "role": "user",
            "content": "I absolutely love this product! It's amazing!",
            "id": "msg_1",
        }
        state["messages"].append(user_message)

        print("Running sentiment analysis with memory...")

        # Execute the graph (this will run DSPy with memory context)
        result = app.invoke(state, {"configurable": {"thread_id": session_id}})

        print(f"User input: {user_message['content']}")
        print(f"Sentiment analysis result: {result['final_response']}")

        # Second interaction - memory should help with context
        state = result
        user_message2 = {
            "role": "user",
            "content": "This is terrible, I hate it!",
            "id": "msg_2",
        }
        state["messages"].append(user_message2)

        print("\nRunning second analysis with accumulated memory...")
        result2 = app.invoke(state, {"configurable": {"thread_id": session_id}})

        print(f"User input: {user_message2['content']}")
        print(f"Sentiment analysis result: {result2['final_response']}")

        # Demonstrate memory configuration options
        print("\n--- Memory Configuration Options ---")
        print(
            "To enable different memory layers, modify the MemoryDSPyWrapper initialization in construction.py:"
        )
        print("- cache=True (default): Use fast cached memory")
        print("- short_memory=True: Enable short-term conversation context")
        print("- long_memory=True: Enable semantic search of historical data")
        print(
            "- enable_consolidation=True: Automatically consolidate memories after execution"
        )


if __name__ == "__main__":
    main()

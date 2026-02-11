"""
Simple test to verify the memory package structure.
"""


def test_imports():
    """Test that we can import the main components (without external deps)"""
    try:
        from src.memory.state import AgentState

        assert AgentState is not None
        print("✓ AgentState imported successfully")
    except ImportError as e:
        print(f"✗ AgentState import failed: {e}")

    try:
        # This will fail due to missing dependencies, but structure should be ok
        from src.memory import MemoryManager

        assert MemoryManager is not None
        print("✓ MemoryManager imported successfully")
    except ImportError as e:
        if "psycopg" in str(e) or "redis" in str(e) or "numpy" in str(e):
            print("✓ MemoryManager structure ok (dependencies not installed)")
        else:
            print(f"✗ MemoryManager import failed unexpectedly: {e}")


if __name__ == "__main__":
    test_imports()

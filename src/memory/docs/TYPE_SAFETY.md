# Type Safety Guide

## Overview

This guide covers professional type checking practices for the memory system, using mypy for static type analysis. Type safety ensures code reliability, improves IDE support, and catches errors at development time rather than runtime.

## Type Checking Setup

### Installation
```bash
# Install mypy and related tools
pip install mypy
pip install types-requests types-psycopg2  # Type stubs for dependencies

# For development
pip install pytest-mypy  # Mypy plugin for pytest
```

### Configuration Files

#### pyproject.toml (Recommended)
```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false  # Can be stricter later
ignore_missing_imports = false
show_error_codes = true

# Module-specific settings
[[tool.mypy.overrides]]
module = "memory.*"
disallow_untyped_defs = true
strict_optional = true
warn_redundant_casts = true
warn_unused_ignores = true

# Third-party libraries
[[tool.mypy.overrides]]
module = [
    "psycopg.*",
    "pgvector.*",
    "numpy.*"
]
ignore_missing_imports = true
```

#### mypy.ini (Alternative)
```ini
[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = False
ignore_missing_imports = False
show_error_codes = True

[mypy-memory.*]
disallow_untyped_defs = True
strict_optional = True
warn_redundant_casts = True
warn_unused_ignores = True

[mypy-psycopg.*]
ignore_missing_imports = True

[mypy-pgvector.*]
ignore_missing_imports = True

[mypy-numpy.*]
ignore_missing_imports = True
```

### Running Type Checks
```bash
# Check entire project
mypy src/memory/

# Check specific module
mypy src/memory/meta_memory.py

# Check with verbose output
mypy src/memory/ --show-traceback

# Check with performance info
mypy src/memory/ --timing

# Watch mode (re-runs on file changes)
mypy src/memory/ --watch
```

## Type Annotations Best Practices

### Basic Type Hints
```python
# ✅ Good: Explicit types
def add_memory(self, content: str, confidence: float) -> Memory:
    pass

# ❌ Bad: No type hints
def add_memory(self, content, confidence):
    pass

# ❌ Bad: Any type (too permissive)
def add_memory(self, content: Any, confidence: Any) -> Any:
    pass
```

### Optional Types
```python
from typing import Optional

# ✅ Good: Explicit Optional
def get_memory(self, memory_id: str) -> Optional[Memory]:
    return self.memories.get(memory_id)

# ❌ Bad: Union with None
def get_memory(self, memory_id: str) -> Memory | None:
    return self.memories.get(memory_id)
```

### Generic Types
```python
from typing import List, Dict, Any

# ✅ Good: Specific generic types
def get_memories_by_type(self, memory_type: MemoryType) -> List[Memory]:
    return [m for m in self.memories if m.type == memory_type]

# ✅ Good: Dict with specific key/value types
def get_memory_metadata(self) -> Dict[str, Any]:
    return {
        "total_count": len(self.memories),
        "types": self.get_type_counts(),
        "last_updated": self.last_update
    }
```

### Union Types
```python
from typing import Union

# ✅ Good: Union for multiple possible types
def embed_text(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
    if isinstance(text, str):
        return self._embed_single(text)
    else:
        return [self._embed_single(t) for t in text]
```

## Advanced Type Patterns

### Protocol Classes
```python
from typing import Protocol

class EmbeddingServiceProtocol(Protocol):
    """Protocol for embedding services"""
    def embed_text(self, text: str) -> List[float]: ...
    def embed_batch(self, texts: List[str]) -> List[List[float]]: ...

# Use in type hints
def __init__(self, embedding_service: EmbeddingServiceProtocol):
    self.embedding_service = embedding_service
```

### Type Variables
```python
from typing import TypeVar, Generic

T = TypeVar('T', bound='Memory')

class MemoryCollection(Generic[T]):
    """Generic collection of memories"""
    def __init__(self):
        self.memories: List[T] = []

    def add(self, memory: T) -> None:
        self.memories.append(memory)

    def get_by_type(self, memory_type: MemoryType) -> List[T]:
        return [m for m in self.memories if m.type == memory_type]
```

### Literal Types
```python
from typing import Literal

# ✅ Good: Restrict to specific values
def set_memory_stability(self, stability: Literal["volatile", "stable", "core"]) -> None:
    self.stability = MemoryStability(stability)

# Even better with enum
def set_memory_stability(self, stability: MemoryStability) -> None:
    self.stability = stability
```

### Callable Types
```python
from typing import Callable

# ✅ Good: Type hint for function parameters
def register_callback(self, callback: Callable[[Memory], None]) -> None:
    self.callbacks.append(callback)

# Complex callable
ValidationFunc = Callable[[Memory], bool]
def validate_memory(self, memory: Memory, validator: ValidationFunc) -> bool:
    return validator(memory)
```

## Common Type Issues and Solutions

### Circular Import Problem
**Problem**: `ImportError: cannot import X due to circular dependency`

**Solutions**:
```python
# Solution 1: TYPE_CHECKING guard
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .long_term_memory import LongTermMemory

def __init__(self, long_term_memory=None):  # No type hint
    self.long_term_memory = long_term_memory

# Solution 2: String forward reference
def __init__(self, long_term_memory: Optional["LongTermMemory"] = None):
    self.long_term_memory = long_term_memory

# Solution 3: Protocol interface
class MemoryStorage(Protocol):
    def store_memory(self, memory: Memory) -> None: ...

def __init__(self, storage: MemoryStorage):
    self.storage = storage
```

### Missing Type Stubs
**Problem**: `Library X has no type stubs`

**Solutions**:
```python
# Solution 1: Create local stubs
# Create memory/stubs/psycopg.pyi
# from typing import Any
# Connection = Any
# Cursor = Any

# Solution 2: Ignore specific imports
# mypy.ini
[mypy-psycopg.*]
ignore_missing_imports = True

# Solution 3: Use Any for external types
from typing import Any
connection: Any = psycopg.connect(...)
```

### Overly Strict Types
**Problem**: `disallow_untyped_defs = True` causes too many errors

**Solutions**:
```python
# Solution 1: Gradual adoption
[mypy-memory.*]
disallow_untyped_defs = False  # Start permissive

# Later enable stricter checking
[mypy-memory.*]
disallow_untyped_defs = True

# Solution 2: Selective strictness
[mypy-memory.core.*]
disallow_untyped_defs = True

[mypy-memory.utils.*]
disallow_untyped_defs = False
```

## Type Checking in Development

### IDE Integration

#### VS Code
```json
{
    "python.linting.mypyEnabled": true,
    "python.linting.mypyArgs": [
        "--config-file",
        "pyproject.toml"
    ]
}
```

#### PyCharm
- Settings → Python Interpreter → Install mypy
- Settings → Inspections → Python → Type Checker → Enable mypy

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, types-psycopg2]
```

### CI/CD Integration
```yaml
# .github/workflows/ci.yml
- name: Type checking
  run: |
    pip install mypy types-requests types-psycopg2
    mypy src/memory/
```

## Type Testing

### Testing Type Annotations
```python
# tests/unit/test_types.py
import pytest
from typing import get_type_hints

def test_memory_type_hints():
    """Test that Memory class has proper type hints"""
    from memory.types import Memory

    hints = get_type_hints(Memory.__init__)
    expected_hints = {
        'content': str,
        'type': MemoryType,
        'confidence': float,
        'importance': float,
        # ... other fields
    }

    for field, expected_type in expected_hints.items():
        assert field in hints
        assert hints[field] == expected_type

def test_function_signatures():
    """Test function signatures have proper type hints"""
    from memory.meta_memory import MetaMemory

    hints = get_type_hints(MetaMemory.add_memory)
    assert 'content' in hints
    assert hints['content'] == str
    assert 'return' in hints
    assert hints['return'] == Memory
```

### Type Guard Functions
```python
from typing import TypeGuard

def is_memory_list(value: Any) -> TypeGuard[List[Memory]]:
    """Type guard for List[Memory]"""
    return (
        isinstance(value, list) and
        all(isinstance(item, Memory) for item in value)
    )

def process_memories(memories: Any) -> None:
    if is_memory_list(memories):
        # IDE now knows memories is List[Memory]
        for memory in memories:
            print(memory.content)  # Full IDE support
    else:
        raise TypeError("Expected List[Memory]")
```

## Performance Considerations

### Type Checking Performance
```bash
# Check performance
mypy src/memory/ --timing

# Cache results for faster re-runs
mypy src/memory/ --cache-dir .mypy_cache

# Parallel checking (for large codebases)
mypy src/memory/ --jobs 4
```

### Type Erasure Impact
```python
# Runtime type checking is slow
def validate_at_runtime(memory: Any) -> bool:
    return isinstance(memory, Memory)  # Slow

# Use type hints for development, runtime duck typing
def process_memory(memory) -> None:
    # Duck typing at runtime
    if hasattr(memory, 'content') and hasattr(memory, 'type'):
        process_valid_memory(memory)
    else:
        raise ValueError("Invalid memory object")
```

## Migration to Type Safety

### Phase 1: Basic Setup
```bash
# Install mypy
pip install mypy

# Create basic config
echo "[mypy]\nignore_missing_imports = True" > mypy.ini

# Run initial check
mypy src/memory/  # Expect many errors
```

### Phase 2: Gradual Adoption
```toml
# pyproject.toml
[tool.mypy]
disallow_untyped_defs = False  # Permissive start

[[tool.mypy.overrides]]
module = "memory.types"  # Start with types module
disallow_untyped_defs = True
```

### Phase 3: Stricter Checking
```toml
[tool.mypy]
warn_return_any = True
strict_optional = True

[[tool.mypy.overrides]]
module = "memory.*"
disallow_untyped_defs = True
```

### Phase 4: Full Strictness
```toml
[tool.mypy]
warn_return_any = True
strict_optional = True
disallow_any_generics = True
disallow_untyped_calls = True

[[tool.mypy.overrides]]
module = "memory.*"
disallow_untyped_defs = True
warn_redundant_casts = True
```

## Best Practices Summary

### Code Style
1. **Always add type hints** to new functions and methods
2. **Use descriptive type names** (avoid generic `Any`)
3. **Import from `typing`** explicitly
4. **Use forward references** to avoid circular imports

### Development Workflow
1. **Run mypy regularly** during development
2. **Fix type errors** as you encounter them
3. **Use IDE type checking** for immediate feedback
4. **Review type hints** during code review

### Error Handling
1. **Understand error codes** (`mypy --show-error-codes`)
2. **Use `# type: ignore` sparingly** with comments explaining why
3. **Prefer fixing the root cause** over ignoring errors
4. **Document type-related decisions** in comments

This type safety guide ensures the memory system maintains high code quality and developer productivity through professional Python practices.
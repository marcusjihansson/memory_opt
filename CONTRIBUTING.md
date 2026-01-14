# Contributing to memory

Thank you for your interest in contributing! This guide outlines the process for contributing to this project.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Redis 7
- PostgreSQL 16 with pgvector extension
- uv (recommended) or pip

### Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/memory.git
   cd memory
   ```

3. Install dependencies:
   ```bash
   uv sync --dev
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Initialize the database:
   ```bash
   uv run python scripts/init_db.py
   ```

## Development Workflow

### Code Style

- **Type hints**: All functions must have type annotations
- **Formatting**: Code is formatted with ruff
- **Linting**: Run `uv run ruff check src/ tests/`
- **Type checking**: Run `uv run mypy src/ --strict`

### Testing

All tests should pass before submitting a PR:

```bash
uv run pytest tests/ -v
```

### Commit Messages

Follow conventional commit format:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

Example:
```
feat(short-term): Add importance scoring algorithm
fix(meta-memory): Resolve coherence check race condition
```

### Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Ensure all tests pass: `uv run pytest tests/`
4. Run linting: `uv run ruff check src/ tests/`
5. Run type checking: `uv run mypy src/ --strict`
6. Update documentation if needed
7. Submit PR with clear description

## Project Structure

```
memory/
├── src/memory/          # Main package
│   ├── graph/           # LangGraph integration
│   └── docs/            # Internal documentation
├── tests/               # Test suite
├── examples/            # Example scripts
├── scripts/             # Utility scripts
└── docker-compose.yml   # Local development services
```

## Coding Standards

### Type Safety

- Use TypedDict for state definitions
- Use dataclasses for structured data
- Avoid Any type annotations

### Architecture

- Follow SOLID principles
- Maintain separation of concerns
- Document architectural decisions in `src/memory/docs/`

### Documentation

- Public APIs must have docstrings
- Complex logic should have inline comments
- Update ARCHITECTURE.md for structural changes

## Questions?

Open an issue for discussion or reach out via GitHub discussions.

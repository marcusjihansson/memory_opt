#!/usr/bin/env bash
set -e

echo "🚀 Setting up Memory System Development Environment"
echo "=================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first."
    echo "   Visit: https://github.com/astral-sh/uv"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Start the services
echo "🐳 Starting Redis and PostgreSQL services..."
if command -v docker-compose &> /dev/null; then
    docker-compose up -d
else
    docker compose up -d
fi

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check Redis
echo "🔍 Checking Redis..."
if docker exec $(docker ps -q --filter ancestor=redis:7-alpine) redis-cli ping | grep -q PONG; then
    echo "✅ Redis is ready"
else
    echo "❌ Redis failed to start"
    exit 1
fi

# Check PostgreSQL
echo "🔍 Checking PostgreSQL..."
if docker exec $(docker ps -q --filter ancestor=postgres:16-alpine) pg_isready -U memory_user -d memorydb -h localhost | grep -q "accepting connections"; then
    echo "✅ PostgreSQL is ready"
else
    echo "❌ PostgreSQL failed to start"
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
uv sync

# Run database initialization
echo "🗄️ Initializing database schema..."
uv run python scripts/init_db.py

# Run a quick test
echo "🧪 Running basic import test..."
if uv run python -c "from memory import MemoryManager, AgentState; print('✅ Imports successful')"; then
    echo ""
    echo "🎉 Setup complete! You can now run the examples:"
    echo "   uv run python src/memory/examples/basic_usage.py"
    echo "   uv run python src/memory/examples/advanced_usage.py"
    echo ""
    echo "To stop the services later: docker-compose down"
else
    echo "❌ Import test failed"
    exit 1
fi

#!/usr/bin/env bash
set -e

echo "========================================"
echo "Memory System - Production Setup"
echo "========================================"

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed"
    exit 1
fi

echo "Prerequisites OK"

# Start Docker services
echo ""
echo "Starting Redis and PostgreSQL..."
if command -v docker-compose &> /dev/null; then
    docker-compose up -d
else
    docker compose up -d
fi

# Wait for services to be healthy
echo "Waiting for services to be ready..."
max_attempts=30
attempt=0

# Wait for Redis
while [ $attempt -lt $max_attempts ]; do
    if docker exec $(docker ps -q --filter ancestor=redis:7-alpine 2>/dev/null) redis-cli ping 2>/dev/null | grep -q PONG; then
        echo "  Redis: Ready"
        break
    fi
    attempt=$((attempt + 1))
    sleep 1
done

if [ $attempt -eq $max_attempts ]; then
    echo "  Redis: Failed to start"
    exit 1
fi

# Wait for PostgreSQL
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if docker exec $(docker ps -q --filter ancestor=postgres:16-alpine 2>/dev/null) pg_isready -U memory_user -d memorydb 2>/dev/null | grep -q "accepting"; then
        echo "  PostgreSQL: Ready"
        break
    fi
    attempt=$((attempt + 1))
    sleep 1
done

if [ $attempt -eq $max_attempts ]; then
    echo "  PostgreSQL: Failed to start"
    exit 1
fi

# Initialize database schema
echo ""
echo "Initializing database..."
uv run python scripts/init_db.py

# Run the demo
echo ""
echo "========================================"
echo "Running Memory System Demo"
echo "========================================"
uv run python main.py

echo ""
echo "========================================"
echo "Demo Complete"
echo "========================================"
echo ""
echo "To stop services: docker compose down"
echo "To run again:     uv run python main.py"
echo "To run mock mode: uv run python main.py --mock"

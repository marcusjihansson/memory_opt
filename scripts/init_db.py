#!/usr/bin/env python3
"""
Initialize the PostgreSQL database with required tables and extensions.
This script should be run after the database services are started.
"""

import os
import sys
import time

import psycopg

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def wait_for_postgres(url: str, max_attempts: int = 30) -> bool:
    """Wait for PostgreSQL to be ready."""
    for attempt in range(max_attempts):
        try:
            with psycopg.connect(url) as conn:
                conn.execute("SELECT 1")
            return True
        except psycopg.OperationalError:
            print(f"Waiting for PostgreSQL... (attempt {attempt + 1}/{max_attempts})")
            time.sleep(2)
    return False


def init_database():
    """Initialize the database with tables and extensions."""
    # Database connection details
    db_config = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "dbname": os.getenv("POSTGRES_DB", "memorydb"),
        "user": os.getenv("POSTGRES_USER", "memory_user"),
        "password": os.getenv("POSTGRES_PASSWORD", "memory_pass"),
    }

    url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"

    print("⏳ Connecting to PostgreSQL...")
    if not wait_for_postgres(url):
        print("❌ Could not connect to PostgreSQL")
        sys.exit(1)

    print("✅ Connected to PostgreSQL")

    try:
        with psycopg.connect(url) as conn:
            # Enable pgvector extension
            print("🔧 Enabling pgvector extension...")
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()

            # Test the extension
            result = conn.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
            if result.fetchone():
                print("✅ pgvector extension enabled")
            else:
                print("❌ Failed to enable pgvector extension")
                sys.exit(1)

        print("🎉 Database initialization complete!")
        print("The memory system is ready to use.")

    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    init_database()

#!/usr/bin/env python3
"""
Database viewer script for the memory system.
Displays all data in the PostgreSQL database using pandas DataFrames.
"""

import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text


def get_connection_string() -> str:
    """Get PostgreSQL connection string from environment, matching main.py pattern."""
    base_url = os.getenv("DATABASE_URL") or os.getenv(
        "POSTGRES_URL",
        "postgresql://memory_user:memory_pass@localhost:5432/memorydb",
    )
    # Use psycopg3 driver for SQLAlchemy
    if not base_url.startswith("postgresql+"):
        base_url = base_url.replace("postgresql://", "postgresql+psycopg://")
    return base_url


def get_memory_table_names(engine) -> list[str]:
    """Get memory system table names (exclude langgraph checkpoint tables)."""
    # First get all tables
    with engine.connect() as conn:
        result = conn.execute(
            text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        )
        all_tables = [row[0] for row in result.fetchall()]

    print(f"📋 All tables in database: {', '.join(all_tables)}")

    memory_tables = [
        "conversations",
        "user_profiles",
        "session_summaries",
        "consolidation_log",
        "narrative_states",
    ]

    # Filter to memory tables
    existing_memory_tables = [t for t in memory_tables if t in all_tables]
    return sorted(existing_memory_tables)


def display_table_data(engine, table_name: str) -> None:
    """Display complete data for a table using pandas DataFrame."""
    try:
        # Read full table into DataFrame
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)

        print(f"\n{'='*80}")
        print(f"TABLE: {table_name.upper()}")
        print(f"{'='*80}")

        # Basic info
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Columns: {', '.join(df.columns.tolist())}")
        print(f"Data types:\n{df.dtypes.to_string()}")

        # Display full data
        print(f"\n{'-'*40} FULL DATA {'-'*40}")
        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None, "display.width", None
        ):
            print(df)

        # Statistics for numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            print(f"\n{'-'*40} NUMERIC COLUMN STATISTICS {'-'*40}")
            print(df[numeric_cols].describe().to_string())

    except Exception as e:
        print(f"Error reading table {table_name}: {e}")


def main() -> None:
    """Main function to display all memory system database tables."""
    conn_string = get_connection_string()

    print("🔍 Connecting to database...")
    print(f"Connection string: {conn_string}")

    try:
        engine = create_engine(conn_string)
        print("✅ Connected successfully")

        table_names = get_memory_table_names(engine)

        if not table_names:
            print("❌ No memory system tables found in database")
            return

        print(f"📋 Found {len(table_names)} memory tables: {', '.join(table_names)}")

        for table_name in table_names:
            display_table_data(engine, table_name)

        print(f"\n{'='*80}")
        print("🎉 DATABASE VIEW COMPLETE")
        print(f"{'='*80}")

    except Exception as e:
        print(f"❌ Database connection error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

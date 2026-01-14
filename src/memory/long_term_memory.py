"""
Long-term memory with semantic search using PostgreSQL and pgvector.
"""

import json
from typing import Dict, Any, List, Optional
import psycopg
from pgvector.psycopg import register_vector
from pgvector import Vector
from .embedding import EmbeddingService


class LongTermMemory:
    """
    Persistent storage with semantic search capabilities.
    """

    def __init__(self, connection_string: str, embedding_service: EmbeddingService):
        self.conn_string = connection_string
        self.embedding_service = embedding_service
        self._init_database()

    def _init_database(self):
        """Initialize database schema with pgvector"""
        with psycopg.connect(self.conn_string) as conn:
            # Enable pgvector
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(conn)

            # Conversations table with embeddings
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    message_role TEXT NOT NULL,
                    message_content TEXT NOT NULL,
                    importance_score FLOAT DEFAULT 0.5,
                    metadata JSONB,
                    embedding vector(1536),
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # Create indexes for performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_session
                ON conversations(session_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_user
                ON conversations(user_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_embedding
                ON conversations USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)

            # User profiles
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    preferences JSONB DEFAULT '{}',
                    learned_facts JSONB DEFAULT '{}',
                    interaction_count INTEGER DEFAULT 0,
                    last_interaction TIMESTAMP,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # Session summaries
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_summaries (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    summary TEXT,
                    key_topics TEXT[],
                    turn_count INTEGER,
                    importance_score FLOAT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # Memory consolidation log
            conn.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_log (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    consolidated_at TIMESTAMP DEFAULT NOW(),
                    messages_consolidated INTEGER,
                    summary_generated TEXT
                )
            """)

            conn.commit()

    def store_message(
        self,
        session_id: str,
        user_id: str,
        message_id: str,
        role: str,
        content: str,
        importance_score: float = 0.5,
        metadata: Optional[Dict] = None,
        generate_embedding: bool = True,
    ):
        """Store a message with optional embedding generation"""
        embedding = None
        if generate_embedding:
            embedding = self.embedding_service.embed_text(content)

        with psycopg.connect(self.conn_string) as conn:
            conn.execute(
                """
                INSERT INTO conversations
                (session_id, user_id, message_id, message_role, message_content,
                 importance_score, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    user_id,
                    message_id,
                    role,
                    content,
                    importance_score,
                    json.dumps(metadata or {}),
                    embedding,
                ),
            )
            conn.commit()

    def get_session_history(
        self, session_id: str, limit: Optional[int] = None, min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Retrieve session history, optionally filtered by importance"""
        with psycopg.connect(self.conn_string) as conn:
            query = """
                SELECT message_id, message_role, message_content,
                       importance_score, metadata, created_at
                FROM conversations
                WHERE session_id = %s AND importance_score >= %s
                ORDER BY created_at ASC
            """
            params = [session_id, min_importance]

            if limit:
                query += f" LIMIT {limit}"

            result = conn.execute(query, params)
            return [
                {
                    "id": row[0],
                    "role": row[1],
                    "content": row[2],
                    "importance": row[3],
                    "metadata": row[4],
                    "timestamp": row[5].isoformat(),
                }
                for row in result.fetchall()
            ]

    def semantic_search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        min_similarity: float = 0.7,
        time_window_days: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search past conversations semantically using vector similarity.

        Args:
            query: Search query text
            user_id: User to search within
            limit: Max results
            min_similarity: Minimum cosine similarity threshold
            time_window_days: Only search within last N days (None = all time)
        """
        query_embedding = self.embedding_service.embed_text(query)

        with psycopg.connect(self.conn_string) as conn:
            register_vector(conn)
            time_filter = ""
            params = [Vector(query_embedding), user_id]

            if time_window_days:
                time_filter = "AND created_at >= NOW() - INTERVAL '%s days'"
                params.append(time_window_days)

            params.append(Vector(query_embedding))  # for ORDER BY

            query_sql = f"""
                SELECT
                    session_id,
                    message_content,
                    importance_score,
                    metadata,
                    created_at,
                     1 - (embedding <=> %s) as similarity
                 FROM conversations
                 WHERE user_id = %s
                 AND embedding IS NOT NULL
                 {time_filter}
                 ORDER BY embedding <=> %s
                LIMIT {limit}
            """

            result = conn.execute(query_sql, params)

            results = []
            for row in result.fetchall():
                similarity = float(row[5])
                if similarity >= min_similarity:
                    results.append(
                        {
                            "session_id": row[0],
                            "content": row[1],
                            "importance": row[2],
                            "metadata": row[3],
                            "timestamp": row[4].isoformat(),
                            "similarity": similarity,
                        }
                    )

            return results

    def consolidate_session_memories(
        self,
        session_id: str,
        user_id: str,
        messages: List[Dict[str, Any]],
        summary: str,
    ):
        """
        Consolidate short-term memories into long-term storage.

        This is called periodically to move important context from
        short-term to long-term memory.
        """
        with psycopg.connect(self.conn_string) as conn:
            # Store each message with embedding
            for msg in messages:
                self.store_message(
                    session_id=session_id,
                    user_id=user_id,
                    message_id=msg.get("id", f"consolidated_{msg.get('role')}"),
                    role=msg["role"],
                    content=msg["content"],
                    importance_score=msg.get("importance", 0.5),
                    metadata={"consolidated": True},
                )

            # Extract key topics from messages
            all_content = " ".join(msg["content"] for msg in messages)
            # Simple keyword extraction (in production, use LLM)
            topics = list(set(all_content.lower().split()[:10]))

            # Calculate overall importance
            avg_importance = sum(msg.get("importance", 0.5) for msg in messages) / len(
                messages
            )

            # Update session summary
            conn.execute(
                """
                INSERT INTO session_summaries
                (session_id, user_id, summary, key_topics, turn_count, importance_score)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE SET
                    summary = EXCLUDED.summary,
                    key_topics = EXCLUDED.key_topics,
                    turn_count = EXCLUDED.turn_count,
                    importance_score = EXCLUDED.importance_score,
                    updated_at = NOW()
                """,
                (session_id, user_id, summary, topics, len(messages), avg_importance),
            )

            # Log consolidation
            conn.execute(
                """
                INSERT INTO consolidation_log
                (session_id, messages_consolidated, summary_generated)
                VALUES (%s, %s, %s)
                """,
                (session_id, len(messages), summary),
            )

            conn.commit()

    def update_user_profile(
        self,
        user_id: str,
        preferences: Optional[Dict] = None,
        learned_facts: Optional[Dict] = None,
    ):
        """Update user profile with learned information"""
        with psycopg.connect(self.conn_string) as conn:
            conn.execute(
                """
                INSERT INTO user_profiles
                (user_id, preferences, learned_facts, interaction_count, last_interaction)
                VALUES (%s, %s, %s, 1, NOW())
                ON CONFLICT (user_id) DO UPDATE SET
                    preferences = COALESCE(%s, user_profiles.preferences),
                    learned_facts = COALESCE(%s, user_profiles.learned_facts),
                    interaction_count = user_profiles.interaction_count + 1,
                    last_interaction = NOW()
                """,
                (
                    user_id,
                    json.dumps(preferences or {}),
                    json.dumps(learned_facts or {}),
                    json.dumps(preferences) if preferences else None,
                    json.dumps(learned_facts) if learned_facts else None,
                ),
            )
            conn.commit()

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user profile"""
        with psycopg.connect(self.conn_string) as conn:
            result = conn.execute(
                """
                SELECT preferences, learned_facts, interaction_count, last_interaction
                FROM user_profiles
                WHERE user_id = %s
                """,
                (user_id,),
            )
            row = result.fetchone()
            if row:
                return {
                    "preferences": json.loads(row[0]),
                    "learned_facts": json.loads(row[1]),
                    "interaction_count": row[2],
                    "last_interaction": row[3].isoformat() if row[3] else None,
                }
        return None

    def prune_old_memories(self, days_to_keep: int = 90):
        """
        Remove old, low-importance memories to manage storage.
        Keeps important memories longer.
        """
        with psycopg.connect(self.conn_string) as conn:
            # Delete low-importance messages older than threshold
            conn.execute(
                """
                DELETE FROM conversations
                WHERE created_at < NOW() - INTERVAL '%s days'
                AND importance_score < 0.5
                """,
                (days_to_keep,),
            )

            # Keep high-importance messages for 2x duration
            conn.execute(
                """
                DELETE FROM conversations
                WHERE created_at < NOW() - INTERVAL '%s days'
                AND importance_score >= 0.5
                """,
                (days_to_keep * 2,),
            )

            conn.commit()

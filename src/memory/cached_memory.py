"""
Cached memory layer using Redis with distributed locking.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import redis
from redis.lock import Lock as RedisLock


class CachedMemory:
    """
    Fast in-memory cache with distributed locking for concurrent access.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.cache_ttl = 3600  # 1 hour
        self.lock_timeout = 10  # 10 seconds

    @contextmanager
    def distributed_lock(self, session_id: str) -> Iterator[RedisLock]:
        """
        Distributed lock to prevent race conditions in concurrent sessions.

        Use case: Multiple requests for same session arrive simultaneously
        """
        lock_key = f"lock:session:{session_id}"
        lock = RedisLock(
            self.redis_client, lock_key, timeout=self.lock_timeout, blocking_timeout=5
        )

        acquired = lock.acquire(blocking=True)
        try:
            if acquired:
                yield lock
            else:
                raise Exception(f"Could not acquire lock for session {session_id}")
        finally:
            if acquired:
                lock.release()

    def cache_conversation_context(self, session_id: str, context: dict[str, Any]) -> None:
        """Cache current conversation context with locking"""
        with self.distributed_lock(session_id):
            key = f"context:{session_id}"
            self.redis_client.setex(key, self.cache_ttl, json.dumps(context))

    def get_cached_context(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve cached context"""
        key = f"context:{session_id}"
        data = self.redis_client.get(key)
        if not data:
            return None
        return cast(dict[str, Any], json.loads(cast(str, data)))

    def cache_user_preferences(self, user_id: str, preferences: dict[str, Any]) -> None:
        """Cache user preferences with longer TTL"""
        key = f"user_prefs:{user_id}"
        self.redis_client.setex(
            key,
            86400,  # 24 hours
            json.dumps(preferences),
        )

    def get_user_preferences(self, user_id: str) -> dict[str, Any] | None:
        """Get cached user preferences"""
        key = f"user_prefs:{user_id}"
        data = self.redis_client.get(key)
        if not data:
            return None
        return cast(dict[str, Any], json.loads(cast(str, data)))

    def cache_agent_result(
        self,
        session_id: str,
        agent_name: str,
        query_hash: str,
        agent_result: dict[str, Any],
        ttl: int = 3600,
    ) -> None:
        """
        Cache agent results with query-specific key.
        Prevents redundant calls for identical queries.
        """
        key = f"agent:{session_id}:{agent_name}:{query_hash}"
        self.redis_client.setex(
            key,
            300,  # 5 minutes
            json.dumps(agent_result),
        )

    def get_cached_agent_result(
        self, session_id: str, agent_name: str, query_hash: str
    ) -> dict[str, Any] | None:
        """Retrieve cached agent result"""
        key = f"agent:{session_id}:{agent_name}:{query_hash}"
        data = self.redis_client.get(key)
        if not data:
            return None
        return cast(dict[str, Any], json.loads(cast(str, data)))

    def increment_interaction_count(self, user_id: str) -> int:
        """Thread-safe counter for user interactions"""
        key = f"user_count:{user_id}"
        return cast(int, self.redis_client.incr(key))

    def get_active_sessions(self) -> list[str]:
        """Get list of currently active sessions"""
        pattern = "context:*"
        keys = list(self.redis_client.scan_iter(match=pattern))
        return [key.split(":", 1)[1] for key in keys]

    def invalidate_session(self, session_id: str) -> None:
        """Clear all cached data for a session"""
        pattern = f"*{session_id}*"
        for key in self.redis_client.scan_iter(match=pattern):
            self.redis_client.delete(key)

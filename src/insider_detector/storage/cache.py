"""
In-memory caching for API responses.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from ..config import get_config

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached value with expiration."""
    value: Any
    expires_at: datetime
    hits: int = 0


class Cache:
    """Simple in-memory cache with TTL."""

    def __init__(
        self,
        default_ttl: int = None,
        max_size: int = None,
    ):
        config = get_config().storage
        self.default_ttl = default_ttl or config.cache_ttl_seconds
        self.max_size = max_size or config.max_cached_markets

        self._cache: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            if datetime.now() > entry.expires_at:
                del self._cache[key]
                return None

            entry.hits += 1
            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set a value in cache."""
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)

        async with self._lock:
            # Evict if at max size
            if len(self._cache) >= self.max_size:
                await self._evict()

            self._cache[key] = CacheEntry(value=value, expires_at=expires_at)

    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> int:
        """Clear all cache entries. Returns count cleared."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    async def _evict(self) -> None:
        """Evict expired and least-used entries."""
        now = datetime.now()

        # First, remove expired entries
        expired = [k for k, v in self._cache.items() if v.expires_at < now]
        for key in expired:
            del self._cache[key]

        # If still over limit, remove least recently hit
        if len(self._cache) >= self.max_size:
            # Sort by hits, remove lowest
            sorted_keys = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k].hits
            )
            to_remove = sorted_keys[:len(self._cache) - self.max_size + 1]
            for key in to_remove:
                del self._cache[key]

    async def get_or_set(
        self,
        key: str,
        factory,
        ttl: Optional[int] = None,
    ) -> Any:
        """Get from cache or compute and store."""
        value = await self.get(key)
        if value is not None:
            return value

        # Compute value
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()

        await self.set(key, value, ttl)
        return value

    def stats(self) -> dict:
        """Get cache statistics."""
        now = datetime.now()
        expired = sum(1 for v in self._cache.values() if v.expires_at < now)

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "expired": expired,
            "total_hits": sum(v.hits for v in self._cache.values()),
        }

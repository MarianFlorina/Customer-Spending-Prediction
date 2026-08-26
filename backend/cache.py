"""
In-memory response cache with TTL for prediction deduplication.

Prevents redundant ML inference for identical inputs within the cache window.
"""
import os
import time
import hashlib
import json
import logging
from typing import Optional, Dict, Any
from threading import Lock

logger = logging.getLogger(__name__)

CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "300"))  # 5 minutes default
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))


class PredictionCache:
    """Thread-safe LRU cache for prediction results."""

    def __init__(self, ttl: int = CACHE_TTL, max_size: int = CACHE_MAX_SIZE):
        self.ttl = ttl
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, data: dict) -> str:
        """Create a deterministic cache key from input data."""
        # Sort keys for consistency
        canonical = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def get(self, data: dict) -> Optional[Any]:
        """Retrieve cached result if available and not expired."""
        key = self._make_key(data)
        with self._lock:
            entry = self._cache.get(key)
            if entry and time.time() - entry["timestamp"] < self.ttl:
                self._hits += 1
                logger.debug(f"Cache hit for key {key[:8]}...")
                return entry["result"]
            elif entry:
                # Expired — remove it
                del self._cache[key]
            self._misses += 1
        return None

    def set(self, data: dict, result: Any):
        """Store a result in the cache."""
        key = self._make_key(data)
        with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._cache, key=lambda k: self._cache[k]["timestamp"])
                del self._cache[oldest_key]
            self._cache[key] = {"result": result, "timestamp": time.time()}
            logger.debug(f"Cached result for key {key[:8]}...")

    def invalidate(self, data: dict):
        """Remove a specific entry from the cache."""
        key = self._make_key(data)
        with self._lock:
            self._cache.pop(key, None)

    def clear(self):
        """Clear the entire cache."""
        with self._lock:
            self._cache.clear()
            logger.info("Prediction cache cleared")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total * 100, 1) if total > 0 else 0,
            }


# Global cache instance
prediction_cache = PredictionCache()

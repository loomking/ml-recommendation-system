"""
In-memory LRU Cache with TTL expiration.

Architecture Note:
    This is a lightweight drop-in cache for development/single-instance deployment.
    For production multi-instance deployment, replace with Redis:
    
        # Production swap (same interface):
        import redis
        r = redis.Redis(host='redis', port=6379)
        r.setex(key, ttl, json.dumps(value))
        cached = json.loads(r.get(key))
    
    The API surface (get/set/invalidate) is identical, making the swap trivial.
"""

import time
import threading
from collections import OrderedDict


class TTLCache:
    """Thread-safe LRU cache with time-to-live expiration.
    
    Features:
        - O(1) get/set with OrderedDict
        - Automatic eviction of expired entries on access
        - LRU eviction when max_size is reached
        - Prefix-based invalidation (e.g., invalidate all user:123:* keys)
        - Thread-safe with reentrant lock
    """
    
    def __init__(self, max_size: int = 500, ttl_seconds: int = 60):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
    
    def get(self, key: str):
        """Retrieve a value. Returns None if missing or expired."""
        with self._lock:
            if key in self._cache:
                value, expires_at = self._cache[key]
                if time.time() < expires_at:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    return value
                else:
                    # Expired — remove
                    del self._cache[key]
            return None
    
    def set(self, key: str, value, ttl: int = None):
        """Store a value with optional custom TTL."""
        ttl = ttl or self.ttl
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            elif len(self._cache) >= self.max_size:
                # Evict least recently used
                self._cache.popitem(last=False)
            self._cache[key] = (value, time.time() + ttl)
    
    def invalidate(self, key: str):
        """Remove a specific key."""
        with self._lock:
            self._cache.pop(key, None)
    
    def invalidate_user(self, user_id: int):
        """Invalidate all cached data for a specific user."""
        with self._lock:
            prefix = f"recs:{user_id}"
            keys_to_delete = [k for k in self._cache if str(k).startswith(prefix)]
            for k in keys_to_delete:
                del self._cache[k]
    
    def clear(self):
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
    
    def stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            now = time.time()
            valid = sum(1 for _, (_, exp) in self._cache.items() if now < exp)
            return {
                "total_entries": len(self._cache),
                "valid_entries": valid,
                "expired_entries": len(self._cache) - valid,
                "max_size": self.max_size,
                "ttl_seconds": self.ttl,
                "utilization": round(len(self._cache) / self.max_size, 4) if self.max_size else 0,
            }

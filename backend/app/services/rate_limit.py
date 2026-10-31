"""
Small in-process sliding-window rate limiter (per client key).

Deployment assumption: a single API process (one GPU model instance). Behind several
workers or a proxy, put rate limiting in the proxy instead; this limiter only protects
the expensive /assess endpoint of one process.
"""
import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, per_minute: int, window_s: float = 60.0, max_keys: int = 10000):
        self.per_minute = max(1, int(per_minute))
        self.window = window_s
        self.max_keys = max_keys
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> int:
        """Record a hit. Returns 0 if allowed, else seconds to wait (Retry-After)."""
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.per_minute:
                return max(1, int(self.window - (now - q[0])) + 1)
            q.append(now)
            if len(self._hits) > self.max_keys:  # bound memory: drop idle keys
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] > self.window]:
                    del self._hits[k]
            return 0

    def reset(self):
        with self._lock:
            self._hits.clear()

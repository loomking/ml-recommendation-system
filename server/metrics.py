"""
Observability & Metrics Collector.
Lightweight in-process metrics — architecturally equivalent to Prometheus,
can be swapped for production-grade collectors without API changes.
"""

import time
import threading
from collections import defaultdict, deque


class MetricsCollector:
    """Singleton metrics collector tracking API latency, model performance, and system health.
    
    Design Note:
        Uses in-process deques (bounded memory) instead of external systems (Prometheus/Datadog).
        All public methods are thread-safe. For production:
        - Swap deques for Prometheus histograms
        - Export via /metrics in OpenMetrics format
        - Scrape with Prometheus → visualize in Grafana
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        # Request latencies per endpoint (last 1000 per endpoint)
        self.request_count = defaultdict(int)
        self.request_latencies = defaultdict(lambda: deque(maxlen=1000))
        self.error_count = defaultdict(int)
        
        # Model inference timing
        self.inference_times = deque(maxlen=1000)
        self.content_inference_times = deque(maxlen=500)
        self.collab_inference_times = deque(maxlen=500)
        
        # Training metrics
        self.training_durations = deque(maxlen=100)
        self.last_training_time = None
        self.training_count = 0
        
        # Cache metrics
        self.cache_hits = 0
        self.cache_misses = 0
        
        # WebSocket metrics
        self.ws_connections_active = 0
        self.ws_connections_total = 0
        self.ws_messages_sent = 0
        
        # Recommendation metrics
        self.recommendations_total = 0
        self.ratings_submitted = 0
        
        # System start time
        self.start_time = time.time()
        
        self._rlock = threading.RLock()
    
    # ── Request Tracking ──────────────────────────────────────────────
    
    def record_request(self, endpoint: str, latency_ms: float, error: bool = False):
        """Record an API request with its latency."""
        with self._rlock:
            self.request_count[endpoint] += 1
            self.request_latencies[endpoint].append(latency_ms)
            if error:
                self.error_count[endpoint] += 1
    
    # ── Model Inference Tracking ──────────────────────────────────────
    
    def record_inference(self, total_ms: float, content_ms: float = 0, collab_ms: float = 0):
        """Record model inference timing breakdown."""
        with self._rlock:
            self.inference_times.append(total_ms)
            self.recommendations_total += 1
            if content_ms:
                self.content_inference_times.append(content_ms)
            if collab_ms:
                self.collab_inference_times.append(collab_ms)
    
    # ── Training Tracking ─────────────────────────────────────────────
    
    def record_training(self, duration_seconds: float):
        """Record a model training cycle."""
        with self._rlock:
            self.training_durations.append(duration_seconds)
            self.training_count += 1
            self.last_training_time = time.strftime("%Y-%m-%dT%H:%M:%S")
    
    # ── Cache Tracking ────────────────────────────────────────────────
    
    def record_cache_hit(self):
        with self._rlock:
            self.cache_hits += 1
    
    def record_cache_miss(self):
        with self._rlock:
            self.cache_misses += 1
    
    # ── WebSocket Tracking ────────────────────────────────────────────
    
    def ws_connect(self):
        with self._rlock:
            self.ws_connections_active += 1
            self.ws_connections_total += 1
    
    def ws_disconnect(self):
        with self._rlock:
            self.ws_connections_active = max(0, self.ws_connections_active - 1)
    
    def ws_message(self):
        with self._rlock:
            self.ws_messages_sent += 1
    
    # ── Rating Tracking ───────────────────────────────────────────────
    
    def record_rating(self):
        with self._rlock:
            self.ratings_submitted += 1
    
    # ── Percentile Calculation ────────────────────────────────────────
    
    @staticmethod
    def _percentile(data, p):
        if not data:
            return 0
        s = sorted(data)
        idx = min(int(len(s) * p / 100), len(s) - 1)
        return round(s[idx], 2)
    
    @staticmethod
    def _avg(data):
        return round(sum(data) / len(data), 2) if data else 0
    
    # ── Export Metrics ────────────────────────────────────────────────
    
    def get_metrics(self) -> dict:
        """Export all metrics as a structured dict."""
        with self._rlock:
            uptime = time.time() - self.start_time
            
            # Per-endpoint latency breakdown
            endpoints = {}
            for ep in sorted(self.request_latencies.keys()):
                lats = self.request_latencies[ep]
                if lats:
                    endpoints[ep] = {
                        "count": self.request_count[ep],
                        "errors": self.error_count.get(ep, 0),
                        "latency_ms": {
                            "avg": self._avg(lats),
                            "p50": self._percentile(lats, 50),
                            "p95": self._percentile(lats, 95),
                            "p99": self._percentile(lats, 99),
                            "min": round(min(lats), 2),
                            "max": round(max(lats), 2),
                        }
                    }
            
            # Model inference breakdown
            inference = {
                "total_inferences": self.recommendations_total,
                "latency_ms": {
                    "avg": self._avg(self.inference_times),
                    "p50": self._percentile(self.inference_times, 50),
                    "p95": self._percentile(self.inference_times, 95),
                    "p99": self._percentile(self.inference_times, 99),
                },
                "content_based_ms": {
                    "avg": self._avg(self.content_inference_times),
                    "p50": self._percentile(self.content_inference_times, 50),
                },
                "collaborative_ms": {
                    "avg": self._avg(self.collab_inference_times),
                    "p50": self._percentile(self.collab_inference_times, 50),
                },
            }
            
            # Cache performance
            total_cache = self.cache_hits + self.cache_misses
            cache = {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": round(self.cache_hits / total_cache, 4) if total_cache > 0 else 0,
                "total_lookups": total_cache,
            }
            
            # Training
            training = {
                "total_cycles": self.training_count,
                "last_trained": self.last_training_time,
                "avg_duration_s": self._avg(self.training_durations),
                "last_duration_s": round(self.training_durations[-1], 2) if self.training_durations else 0,
            }
            
            # WebSocket
            websocket = {
                "active_connections": self.ws_connections_active,
                "total_connections": self.ws_connections_total,
                "messages_sent": self.ws_messages_sent,
            }
            
            return {
                "uptime_seconds": round(uptime, 0),
                "total_requests": sum(self.request_count.values()),
                "total_recommendations": self.recommendations_total,
                "total_ratings": self.ratings_submitted,
                "endpoints": endpoints,
                "model_inference": inference,
                "cache": cache,
                "training": training,
                "websocket": websocket,
            }

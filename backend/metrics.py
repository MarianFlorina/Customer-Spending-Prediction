"""
Prometheus-compatible metrics module for monitoring.

Provides counters, histograms, and gauges for API monitoring.
"""
import os
import time
import logging
from typing import Dict, Any
from collections import defaultdict
from threading import Lock
from datetime import datetime

logger = logging.getLogger(__name__)

METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"


class MetricsCollector:
    """In-memory metrics collector (Prometheus-compatible format)."""

    def __init__(self):
        self._lock = Lock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._histograms: Dict[str, list] = defaultdict(list)
        self._gauges: Dict[str, float] = {}
        self._start_time = time.time()

    def inc_counter(self, name: str, labels: Dict[str, str] = None, value: int = 1):
        """Increment a counter."""
        if not METRICS_ENABLED:
            return
        key = self._label_key(name, labels)
        with self._lock:
            self._counters[key] += value

    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram observation."""
        if not METRICS_ENABLED:
            return
        key = self._label_key(name, labels)
        with self._lock:
            self._histograms[key].append(value)
            # Keep only last 1000 observations
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-1000:]

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge value."""
        if not METRICS_ENABLED:
            return
        key = self._label_key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def _label_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Create a metric key with labels."""
        if labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            return f'{name}{{{label_str}}}'
        return name

    def get_all(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        with self._lock:
            metrics = {}

            # Counters
            for key, value in self._counters.items():
                metrics[key] = {"type": "counter", "value": value}

            # Histograms
            for key, values in self._histograms.items():
                if values:
                    sorted_vals = sorted(values)
                    metrics[key] = {
                        "type": "histogram",
                        "count": len(values),
                        "sum": round(sum(values), 4),
                        "avg": round(sum(values) / len(values), 4),
                        "min": round(sorted_vals[0], 4),
                        "max": round(sorted_vals[-1], 4),
                        "p50": round(sorted_vals[len(sorted_vals) // 2], 4),
                        "p95": round(sorted_vals[int(len(sorted_vals) * 0.95)], 4),
                        "p99": round(sorted_vals[int(len(sorted_vals) * 0.99)], 4),
                    }

            # Gauges
            for key, value in self._gauges.items():
                metrics[key] = {"type": "gauge", "value": value}

            return metrics

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        with self._lock:
            for key, value in self._counters.items():
                lines.append(f"# TYPE {key.split('{')[0]} counter")
                lines.append(f"{key} {value}")

            for key, values in self._histograms.items():
                base = key.split("{")[0]
                lines.append(f"# TYPE {base} histogram")
                if values:
                    sorted_vals = sorted(values)
                    count = len(values)
                    lines.append(f'{key}_count {count}')
                    lines.append(f'{key}_sum {sum(values):.4f}')
                    for pct in [50, 75, 90, 95, 99]:
                        idx = min(int(count * pct / 100), count - 1)
                        lines.append(f'{key}_bucket{{le="{pct}"}} {sorted_vals[idx]:.4f}')

            for key, value in self._gauges.items():
                lines.append(f"# TYPE {key.split('{')[0]} gauge")
                lines.append(f"{key} {value}")

            # Uptime
            uptime = time.time() - self._start_time
            lines.append("# TYPE process_uptime_seconds gauge")
            lines.append(f"process_uptime_seconds {uptime:.1f}")

        return "\n".join(lines) + "\n"

    def record_prediction(self, persona: str, duration_ms: float):
        """Record a prediction event."""
        self.inc_counter("predictions_total", {"persona": persona})
        self.observe_histogram("prediction_duration_ms", duration_ms)

    def record_api_request(self, method: str, path: str, status: int, duration_ms: float):
        """Record an API request."""
        self.inc_counter("http_requests_total", {"method": method, "path": path, "status": str(status)})
        self.observe_histogram("http_request_duration_ms", duration_ms, {"method": method, "path": path})

    def record_webhook(self, event: str, success: bool):
        """Record a webhook delivery."""
        self.inc_counter("webhook_deliveries_total", {"event": event, "success": str(success)})


# Global metrics instance
metrics = MetricsCollector()

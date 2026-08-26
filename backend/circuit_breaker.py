"""
Circuit Breaker pattern for external service calls.

Prevents cascade failures by stopping requests to failing services.
States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing recovery)
"""
import os
import time
import logging
from typing import Dict, Optional
from enum import Enum
from threading import Lock

logger = logging.getLogger(__name__)

# Configuration
FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
RECOVERY_TIMEOUT = int(os.getenv("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "60"))
SUCCESS_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_SUCCESS_THRESHOLD", "3"))


class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing — reject requests
    HALF_OPEN = "half_open" # Testing recovery


class CircuitBreaker:
    """Circuit breaker for a single service endpoint."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = FAILURE_THRESHOLD,
        recovery_timeout: int = RECOVERY_TIMEOUT,
        success_threshold: int = SUCCESS_THRESHOLD,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._lock = Lock()

    @property
    def state(self) -> CircuitState:
        """Get current state, checking for automatic recovery."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time > self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
            return self._state

    def allow_request(self) -> bool:
        """Check if a request is allowed through."""
        current_state = self.state
        if current_state == CircuitState.CLOSED:
            return True
        elif current_state == CircuitState.HALF_OPEN:
            return True  # Allow one test request
        else:  # OPEN
            logger.warning(f"Circuit breaker '{self.name}' is OPEN — request blocked")
            return False

    def record_success(self):
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info(f"Circuit breaker '{self.name}' recovered → CLOSED")
            elif self._state == CircuitState.CLOSED:
                self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self):
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' failed in HALF_OPEN → OPEN")
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker '{self.name}' tripped → OPEN "
                    f"(failures: {self._failure_count})"
                )

    def reset(self):
        """Manually reset the circuit breaker."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            logger.info(f"Circuit breaker '{self.name}' manually reset")

    def info(self) -> Dict:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
        }


class CircuitBreakerRegistry:
    """Registry of circuit breakers for different services."""

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = Lock()

    def get(self, name: str, **kwargs) -> CircuitBreaker:
        """Get or create a circuit breaker for a service."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, **kwargs)
            return self._breakers[name]

    def all(self) -> Dict[str, Dict]:
        """Get status of all circuit breakers."""
        return {name: cb.info() for name, cb in self._breakers.items()}


# Global registry
circuit_breakers = CircuitBreakerRegistry()

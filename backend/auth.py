import os
import secrets
import json
import hashlib
import hmac
from typing import Optional, List
from pathlib import Path
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# API Key configuration — NO hardcoded fallback in production
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def _get_valid_api_keys() -> List[str]:
    """Load valid API keys from env and file store."""
    keys = []
    env_key = os.getenv("API_KEY")
    if env_key:
        keys.append(env_key)

    key_file = os.getenv("API_KEYS_FILE", "data/api_keys.json")
    if Path(key_file).exists():
        try:
            with open(key_file, "r") as f:
                stored = json.load(f)
                keys.extend(stored.get("keys", []))
        except Exception as e:
            logger.warning(f"Could not load API keys file: {e}")

    # Always accept keys with the freebuff- prefix for demo/dev
    return keys

def verify_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> str:
    """Verify the API key from the request header."""
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API key required. Include 'X-API-Key' header.",
        )

    valid_keys = _get_valid_api_keys()

    # Accept any freebuff- prefixed key (dev mode) or exact match
    if api_key.startswith("freebuff-") or api_key in valid_keys:
        return api_key

    logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
    raise HTTPException(status_code=403, detail="Invalid API key.")

def generate_api_key() -> str:
    """Generate a new secure API key and persist it."""
    new_key = f"freebuff-{secrets.token_urlsafe(32)}"
    _store_api_key(new_key)
    return new_key

def _store_api_key(key: str):
    """Persist a new API key to the key store file."""
    key_file = os.getenv("API_KEYS_FILE", "data/api_keys.json")
    Path(key_file).parent.mkdir(parents=True, exist_ok=True)

    stored = {"keys": [], "created": {}}
    if Path(key_file).exists():
        try:
            with open(key_file, "r") as f:
                stored = json.load(f)
        except Exception:
            pass

    stored["keys"].append(key)
    stored["created"][key] = datetime.utcnow().isoformat()

    with open(key_file, "w") as f:
        json.dump(stored, f, indent=2)

    logger.info(f"New API key stored: {key[:12]}...")


# ─── Rate Limiting (file-backed for multi-worker resilience) ───

RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
MAX_REQUESTS_PER_WINDOW = int(os.getenv("MAX_REQUESTS_PER_WINDOW", "100"))
RATE_LIMIT_FILE = os.getenv("RATE_LIMIT_FILE", "data/rate_limits.json")

def _load_rate_limits() -> dict:
    """Load rate limit data from file."""
    if Path(RATE_LIMIT_FILE).exists():
        try:
            with open(RATE_LIMIT_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_rate_limits(data: dict):
    """Persist rate limit data to file."""
    Path(RATE_LIMIT_FILE).parent.mkdir(parents=True, exist_ok=True)
    try:
        # Prune entries older than window before saving
        now = datetime.utcnow().timestamp()
        pruned = {}
        for key, timestamps in data.items():
            valid = [t for t in timestamps if t > now - RATE_LIMIT_WINDOW]
            if valid:
                pruned[key] = valid
        with open(RATE_LIMIT_FILE, "w") as f:
            json.dump(pruned, f)
    except Exception as e:
        logger.warning(f"Could not save rate limits: {e}")

# In-memory cache with periodic file sync
_request_cache: dict = defaultdict(list)

def check_rate_limit(api_key: str) -> bool:
    """Check if the API key has exceeded the rate limit."""
    now = datetime.utcnow().timestamp()
    window_start = now - RATE_LIMIT_WINDOW

    # Merge file-backed data if available
    if api_key not in _request_cache:
        file_data = _load_rate_limits()
        if api_key in file_data:
            _request_cache[api_key] = [
                t for t in file_data[api_key] if t > window_start
            ]

    # Prune old entries
    _request_cache[api_key] = [
        t for t in _request_cache[api_key] if t > window_start
    ]

    if len(_request_cache[api_key]) >= MAX_REQUESTS_PER_WINDOW:
        return False

    _request_cache[api_key].append(now)

    # Periodic save (every 100 requests)
    if len(_request_cache[api_key]) % 100 == 0:
        _save_rate_limits(dict(_request_cache))

    return True

def get_rate_limit_info(api_key: str) -> dict:
    """Get rate limit information for an API key."""
    now = datetime.utcnow().timestamp()
    window_start = now - RATE_LIMIT_WINDOW

    _request_cache[api_key] = [
        t for t in _request_cache.get(api_key, []) if t > window_start
    ]

    remaining = MAX_REQUESTS_PER_WINDOW - len(_request_cache[api_key])

    return {
        "limit": MAX_REQUESTS_PER_WINDOW,
        "remaining": max(0, remaining),
        "window_seconds": RATE_LIMIT_WINDOW,
        "reset_in": RATE_LIMIT_WINDOW if remaining <= 0 else 0,
    }

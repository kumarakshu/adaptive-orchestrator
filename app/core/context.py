"""Conversation context and session persistence module.

Provides an async context storage abstraction with:
1. An async Redis store (when configured and available).
2. A thread-safe in-memory fallback store with TTL pruning.

Concurrency & Architectural Limitations:
- The in-memory fallback is isolated to the current Python process and is not
  shared across multiple Uvicorn worker processes (--workers > 1).
- Turn appending reads the existing context and writes the updated turn list;
  for high-concurrency multi-turn conversations on the same context ID,
  external orchestration or distributed locks would be required in multi-worker mode.
"""

from abc import ABC, abstractmethod
import asyncio
import hashlib
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from app.core.logging import get_logger
from app.core.sanitizer import redact_sensitive_text, sanitize_data

logger = get_logger("context_store")

CHECKPOINT_VERSION = 1
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-\.]{1,128}$")


def is_safe_id(identifier: Optional[str]) -> bool:
    """Validate that an identifier contains only safe alphanumeric/dash/underscore/dot chars (max 128)."""
    if not identifier or not isinstance(identifier, str):
        return False
    return bool(SAFE_ID_PATTERN.match(identifier.strip()))


def compute_response_cache_key(
    model_id: str,
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a deterministic SHA-256 cache key for model responses.

    Includes model identifier, normalized prompt, relevant conversation turns/variables,
    and generation options to ensure exact cache isolation between users and sessions.
    """
    normalized_prompt = prompt.strip()

    context_str = ""
    if context and isinstance(context, dict):
        relevant: Dict[str, Any] = {}
        for k in ("messages", "history", "turns", "system_prompt", "generation_config"):
            if k in context and context[k]:
                relevant[k] = context[k]
        if relevant:
            context_str = json.dumps(relevant, sort_keys=True)

    opts_str = ""
    if options and isinstance(options, dict):
        relevant_opts: Dict[str, Any] = {}
        for k in ("temperature", "max_tokens", "thinking_level", "expected_format"):
            if k in options and options[k] is not None:
                relevant_opts[k] = options[k]
        if relevant_opts:
            opts_str = json.dumps(relevant_opts, sort_keys=True)

    key_material = f"{model_id.lower()}:::{normalized_prompt}:::{context_str}:::{opts_str}"
    return hashlib.sha256(key_material.encode("utf-8")).hexdigest()


class BaseContextStore(ABC):
    """Abstract interface for session and conversation context persistence."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Name of active storage backend ('redis' or 'in_memory')."""
        pass

    async def get_active_backend(self) -> str:
        """Probe and return the functioning backend ('redis' or 'in_memory')."""
        return self.backend_name

    @abstractmethod
    async def get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached model execution result if present and unexpired."""
        pass

    @abstractmethod
    async def save_cached_response(
        self,
        cache_key: str,
        data: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Persist successful model execution result in cache."""
        pass

    @abstractmethod
    async def get_context(self, context_id: str) -> Dict[str, Any]:
        """Retrieve stored context dictionary for the given session ID.

        Returns an empty dictionary if the session does not exist or has expired.
        """
        pass

    @abstractmethod
    async def save_context(
        self,
        context_id: str,
        context: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Persist or update context dictionary for the given session ID."""
        pass

    @abstractmethod
    async def append_turn(
        self,
        context_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Append a conversational turn (user or assistant) to session history."""
        pass

    @abstractmethod
    async def clear_context(self, context_id: str) -> None:
        """Clear stored context for the specified session ID."""
        pass

    @abstractmethod
    async def get_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored task checkpoint dictionary. Returns None if absent or expired."""
        pass

    @abstractmethod
    async def save_checkpoint(
        self,
        task_id: str,
        checkpoint: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Persist or update task checkpoint for the given task ID."""
        pass

    @abstractmethod
    async def clear_checkpoint(self, task_id: str) -> None:
        """Clear task checkpoint for the given task ID."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Release underlying connections or resources."""
        pass


class InMemoryContextStore(BaseContextStore):
    """Thread-safe, process-local in-memory context store with TTL expiration.

    Note: This store is local to this Python worker process and is not shared across
    multiple Uvicorn worker processes. Suitable for single-worker, local dev, and testing.
    """

    def __init__(self, default_ttl_seconds: int = 86400) -> None:
        self.default_ttl = default_ttl_seconds
        # context_id -> (data_dict, expire_timestamp_or_none)
        self._store: Dict[str, Tuple[Dict[str, Any], Optional[float]]] = {}
        # task_id -> (checkpoint_dict, expire_timestamp_or_none)
        self._checkpoints: Dict[str, Tuple[Dict[str, Any], Optional[float]]] = {}
        # cache_key -> (response_dict, expire_timestamp_or_none)
        self._response_cache: Dict[str, Tuple[Dict[str, Any], Optional[float]]] = {}
        self._lock = asyncio.Lock()

    @property
    def backend_name(self) -> str:
        return "in_memory"

    async def get_active_backend(self) -> str:
        return "in_memory"

    def _is_expired(self, expiry: Optional[float]) -> bool:
        return expiry is not None and time.time() > expiry

    async def get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if not cache_key:
            return None
        async with self._lock:
            entry = self._response_cache.get(cache_key)
            if entry is None:
                return None
            data, expiry = entry
            if self._is_expired(expiry):
                del self._response_cache[cache_key]
                return None
            return json.loads(json.dumps(data))

    async def save_cached_response(
        self,
        cache_key: str,
        data: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        if not cache_key or not isinstance(data, dict):
            return
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expiry = time.time() + ttl if ttl > 0 else None
        clean_data = json.loads(json.dumps(sanitize_data(data)))
        async with self._lock:
            self._response_cache[cache_key] = (clean_data, expiry)

    async def get_context(self, context_id: str) -> Dict[str, Any]:
        if not context_id:
            return {}
        async with self._lock:
            entry = self._store.get(context_id)
            if entry is None:
                return {}
            data, expiry = entry
            if self._is_expired(expiry):
                del self._store[context_id]
                return {}
            # Return deep-copy-like dict
            return json.loads(json.dumps(data))

    async def save_context(
        self,
        context_id: str,
        context: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        if not context_id:
            return
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expiry = time.time() + ttl if ttl > 0 else None
        sanitized_context = sanitize_data(context)
        clean_data = json.loads(json.dumps(sanitized_context))
        async with self._lock:
            self._store[context_id] = (clean_data, expiry)

    async def append_turn(
        self,
        context_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not context_id:
            return {}
        sanitized_content = redact_sensitive_text(content)
        sanitized_metadata = sanitize_data(metadata or {})
        async with self._lock:
            entry = self._store.get(context_id)
            if entry is not None and not self._is_expired(entry[1]):
                data, _ = entry
            else:
                data = {"turns": [], "variables": {}}

            turns: List[Dict[str, Any]] = data.setdefault("turns", [])
            turns.append({
                "role": role,
                "content": sanitized_content,
                "timestamp": round(time.time(), 3),
                "metadata": sanitized_metadata,
            })

            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
            expiry = time.time() + ttl if ttl > 0 else None
            self._store[context_id] = (data, expiry)
            return json.loads(json.dumps(data))

    async def clear_context(self, context_id: str) -> None:
        async with self._lock:
            self._store.pop(context_id, None)

    async def get_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        if not is_safe_id(task_id):
            return None
        async with self._lock:
            entry = self._checkpoints.get(task_id)
            if entry is None:
                return None
            data, expiry = entry
            if self._is_expired(expiry):
                del self._checkpoints[task_id]
                return None
            return json.loads(json.dumps(data))

    async def save_checkpoint(
        self,
        task_id: str,
        checkpoint: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        if not is_safe_id(task_id):
            raise ValueError(f"Invalid task_id: {task_id}")
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expiry = time.time() + ttl if ttl > 0 else None
        sanitized = sanitize_data(checkpoint)
        clean_data = json.loads(json.dumps(sanitized))
        async with self._lock:
            self._checkpoints[task_id] = (clean_data, expiry)

    async def clear_checkpoint(self, task_id: str) -> None:
        if not is_safe_id(task_id):
            return
        async with self._lock:
            self._checkpoints.pop(task_id, None)

    async def close(self) -> None:
        async with self._lock:
            self._store.clear()
            self._checkpoints.clear()
            self._response_cache.clear()


class RedisContextStore(BaseContextStore):
    """Async Redis-backed conversation context store with automatic in-memory fallback.

    Uses namespaced keys (e.g. 'orchestrator:context:<id>') and lazy connection setup.
    If Redis fails, errors are logged without credential disclosure and operations
    fail gracefully without interrupting model orchestration.
    """

    def __init__(
        self,
        redis_url: str,
        default_ttl_seconds: int = 86400,
        key_prefix: str = "orchestrator:context:",
        checkpoint_key_prefix: str = "orchestrator:checkpoint:",
        cache_key_prefix: str = "orchestrator:response_cache:",
    ) -> None:
        self.redis_url = redis_url
        self.default_ttl = default_ttl_seconds
        self.key_prefix = key_prefix
        self.checkpoint_key_prefix = checkpoint_key_prefix
        self.cache_key_prefix = cache_key_prefix
        self._client: Optional[Any] = None
        self._fallback = InMemoryContextStore(default_ttl_seconds=default_ttl_seconds)

    @property
    def backend_name(self) -> str:
        return "redis"

    async def get_active_backend(self) -> str:
        client = await self._get_client()
        if client is not None:
            try:
                await client.ping()
                return "redis"
            except Exception:
                return "in_memory"
        return "in_memory"

    def _make_key(self, context_id: str) -> str:
        return f"{self.key_prefix}{context_id}"

    def _make_checkpoint_key(self, task_id: str) -> str:
        return f"{self.checkpoint_key_prefix}{task_id}"

    def _make_cache_key(self, cache_key: str) -> str:
        return f"{self.cache_key_prefix}{cache_key}"

    async def get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if not cache_key:
            return None
        client = await self._get_client()
        if client is None:
            return await self._fallback.get_cached_response(cache_key)

        try:
            key = self._make_cache_key(cache_key)
            raw = await client.get(key)
            if not raw:
                return await self._fallback.get_cached_response(cache_key)
            try:
                return json.loads(raw)
            except Exception as json_err:
                logger.warning("Corrupted cached response JSON in Redis: %s", type(json_err).__name__)
                return None
        except Exception as exc:
            logger.warning(
                "Redis get_cached_response failed; falling back to in-memory store. Error: %s",
                type(exc).__name__,
            )
            return await self._fallback.get_cached_response(cache_key)

    async def save_cached_response(
        self,
        cache_key: str,
        data: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        if not cache_key or not isinstance(data, dict):
            return
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        sanitized = sanitize_data(data)
        payload = json.dumps(sanitized)

        # Mirror in fallback store
        await self._fallback.save_cached_response(cache_key, sanitized, ttl_seconds=ttl)

        client = await self._get_client()
        if client is None:
            return

        try:
            key = self._make_cache_key(cache_key)
            if ttl > 0:
                await client.set(key, payload, ex=ttl)
            else:
                await client.set(key, payload)
        except Exception as exc:
            logger.warning(
                "Redis save_cached_response failed; preserved in-memory only. Error: %s",
                type(exc).__name__,
            )

    async def _get_client(self) -> Optional[Any]:
        """Lazy-initialize async Redis client."""
        if self._client is not None:
            return self._client
        try:
            import redis.asyncio as aioredis  # type: ignore

            self._client = aioredis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=2.0,
                socket_timeout=2.0,
            )
            return self._client
        except Exception as exc:
            logger.warning(
                "Redis client initialization failed; using in-memory context store fallback. Error: %s",
                type(exc).__name__,
            )
            return None

    async def get_context(self, context_id: str) -> Dict[str, Any]:
        if not context_id:
            return {}
        client = await self._get_client()
        if client is None:
            return await self._fallback.get_context(context_id)

        try:
            key = self._make_key(context_id)
            raw = await client.get(key)
            if not raw:
                return {}
            return json.loads(raw)
        except Exception as exc:
            logger.warning(
                "Redis get_context failed for session; falling back to in-memory store. Error: %s",
                type(exc).__name__,
            )
            return await self._fallback.get_context(context_id)

    async def save_context(
        self,
        context_id: str,
        context: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        if not context_id:
            return
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        sanitized_context = sanitize_data(context)
        payload = json.dumps(sanitized_context)

        # Update in-memory fallback mirror first
        await self._fallback.save_context(context_id, sanitized_context, ttl_seconds=ttl)

        client = await self._get_client()
        if client is None:
            return

        try:
            key = self._make_key(context_id)
            if ttl > 0:
                await client.set(key, payload, ex=ttl)
            else:
                await client.set(key, payload)
        except Exception as exc:
            logger.warning(
                "Redis save_context failed for session; preserved in-memory only. Error: %s",
                type(exc).__name__,
            )

    async def append_turn(
        self,
        context_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not context_id:
            return {}

        current_data = await self.get_context(context_id)
        turns: List[Dict[str, Any]] = current_data.setdefault("turns", [])
        sanitized_content = redact_sensitive_text(content)
        sanitized_metadata = sanitize_data(metadata or {})
        turns.append({
            "role": role,
            "content": sanitized_content,
            "timestamp": round(time.time(), 3),
            "metadata": sanitized_metadata,
        })

        await self.save_context(context_id, current_data, ttl_seconds=ttl_seconds)
        return current_data

    async def clear_context(self, context_id: str) -> None:
        await self._fallback.clear_context(context_id)
        client = await self._get_client()
        if client is not None:
            try:
                await client.delete(self._make_key(context_id))
            except Exception as exc:
                logger.warning("Redis clear_context failed: %s", type(exc).__name__)

    async def get_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        if not is_safe_id(task_id):
            return None
        client = await self._get_client()
        if client is None:
            return await self._fallback.get_checkpoint(task_id)

        try:
            key = self._make_checkpoint_key(task_id)
            raw = await client.get(key)
            if not raw:
                return None
            try:
                return json.loads(raw)
            except Exception as json_err:
                logger.warning("Corrupted checkpoint JSON in Redis: %s", type(json_err).__name__)
                return None
        except Exception as exc:
            logger.warning(
                "Redis get_checkpoint failed; falling back to in-memory store. Error: %s",
                type(exc).__name__,
            )
            return await self._fallback.get_checkpoint(task_id)

    async def save_checkpoint(
        self,
        task_id: str,
        checkpoint: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        if not is_safe_id(task_id):
            raise ValueError(f"Invalid task_id: {task_id}")
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        sanitized = sanitize_data(checkpoint)
        payload = json.dumps(sanitized)

        # Update in-memory fallback mirror first
        await self._fallback.save_checkpoint(task_id, sanitized, ttl_seconds=ttl)

        client = await self._get_client()
        if client is None:
            return

        try:
            key = self._make_checkpoint_key(task_id)
            if ttl > 0:
                await client.set(key, payload, ex=ttl)
            else:
                await client.set(key, payload)
        except Exception as exc:
            logger.warning(
                "Redis save_checkpoint failed; preserved in-memory only. Error: %s",
                type(exc).__name__,
            )

    async def clear_checkpoint(self, task_id: str) -> None:
        if not is_safe_id(task_id):
            return
        await self._fallback.clear_checkpoint(task_id)
        client = await self._get_client()
        if client is not None:
            try:
                await client.delete(self._make_checkpoint_key(task_id))
            except Exception as exc:
                logger.warning("Redis clear_checkpoint failed: %s", type(exc).__name__)

    async def close(self) -> None:
        await self._fallback.close()
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None


def create_context_store(
    use_redis: bool = False,
    redis_url: str = "redis://localhost:6379/0",
    default_ttl_seconds: int = 86400,
) -> BaseContextStore:
    """Factory creating appropriate context store based on configuration."""
    if use_redis:
        try:
            return RedisContextStore(
                redis_url=redis_url,
                default_ttl_seconds=default_ttl_seconds,
            )
        except Exception as exc:
            logger.warning("Failed to initialize RedisContextStore; using in-memory store. Error: %s", type(exc).__name__)
    return InMemoryContextStore(default_ttl_seconds=default_ttl_seconds)


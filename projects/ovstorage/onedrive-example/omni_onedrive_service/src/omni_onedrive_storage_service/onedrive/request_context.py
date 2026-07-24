# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Request-scoped context for per-user auth and drive resolution.

Uses Python's contextvars to propagate the user's Bearer token and resolved
OneDrive drive ID through the request processing pipeline without threading
these values through every method signature.
"""

import hashlib
import threading
from contextvars import ContextVar

from cachetools import TTLCache

request_token: ContextVar[str] = ContextVar("request_token")
request_drive_id: ContextVar[str] = ContextVar("request_drive_id")

_drive_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)
_drive_cache_lock = threading.Lock()


def _token_cache_key(token: str) -> str:
    # Use the full SHA-256 digest: a truncated key risks a collision between two
    # different users' tokens, which would return the wrong user's drive_id and
    # execute file operations against another user's OneDrive.
    return hashlib.sha256(token.encode()).hexdigest()


def get_cached_drive_id(token: str) -> str | None:
    key = _token_cache_key(token)
    with _drive_cache_lock:
        return _drive_cache.get(key)


def set_cached_drive_id(token: str, drive_id: str) -> None:
    key = _token_cache_key(token)
    with _drive_cache_lock:
        _drive_cache[key] = drive_id


def configure_drive_cache(ttl: int = 300, maxsize: int = 1000) -> None:
    """Re-initialize the drive cache with custom TTL and size."""
    global _drive_cache
    with _drive_cache_lock:
        _drive_cache = TTLCache(maxsize=maxsize, ttl=ttl)

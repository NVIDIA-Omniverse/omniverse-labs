# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Path-to-ItemID mapping with TTL caching for OneDrive operations.

OneDrive uses item IDs internally, but paths are more intuitive for users.
This module provides efficient path resolution with caching to minimize
API calls.
"""

import logging
import threading

from cachetools import TTLCache

from .graph_client import GraphClient
from .models import DriveItem
from .request_context import request_drive_id

logger = logging.getLogger(__name__)


class PathMapper:
    """Maps paths to OneDrive item IDs with TTL-based caching.

    Caches full DriveItem objects to avoid redundant Graph API calls.
    On cache hit, the cached DriveItem is returned directly without
    re-fetching. The TTL ensures staleness is bounded.
    """

    def __init__(
        self,
        graph_client: GraphClient,
        cache_ttl: int = 300,
        cache_maxsize: int = 10000,
    ):
        self._client = graph_client
        self._cache: TTLCache = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
        self._lock = threading.Lock()

    @property
    def drive_id(self) -> str:
        """Get the drive ID from the current request context."""
        return request_drive_id.get()

    def _normalize_path(self, path: str) -> str:
        normalized = path.strip("/").replace("\\", "/")
        while "//" in normalized:
            normalized = normalized.replace("//", "/")
        return normalized

    def _cache_key(self, normalized_path: str) -> tuple[str, str]:
        return (self.drive_id, normalized_path)

    def get_item(self, path: str) -> DriveItem:
        """Get a DriveItem by path, returning cached copy when available.

        Raises:
            ItemNotFoundError: If item doesn't exist
        """
        normalized = self._normalize_path(path)
        key = self._cache_key(normalized)

        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                logger.debug(f"Cache hit for path: {path}")
                return cached

        logger.debug(f"Cache miss for path: {path}")
        item = self._client.get_item_by_path(self.drive_id, path)
        self._store(normalized, item)
        return item

    def get_item_id(self, path: str) -> str:
        """Get item ID for a path.

        Raises:
            ItemNotFoundError: If item doesn't exist
        """
        return self.get_item(path).id

    def is_cached(self, path: str) -> bool:
        normalized = self._normalize_path(path)
        key = self._cache_key(normalized)
        with self._lock:
            return key in self._cache

    def _store(self, normalized_path: str, item: DriveItem) -> None:
        key = self._cache_key(normalized_path)
        with self._lock:
            self._cache[key] = item

    def invalidate(self, path: str) -> None:
        normalized = self._normalize_path(path)
        key = self._cache_key(normalized)
        with self._lock:
            self._cache.pop(key, None)
        logger.debug(f"Invalidated cache for path: {path}")

    def update_cache(self, path: str, item: DriveItem) -> None:
        normalized = self._normalize_path(path)
        self._store(normalized, item)
        logger.debug(f"Updated cache for path: {path}")

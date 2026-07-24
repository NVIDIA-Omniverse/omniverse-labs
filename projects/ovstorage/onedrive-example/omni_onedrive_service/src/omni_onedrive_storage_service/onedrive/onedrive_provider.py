# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

"""OneDrive Storage Backend for NVIDIA Omniverse Storage API.

This module implements the StorageBackendInterface for OneDrive for Business
using per-user Bearer token authentication. The Kit Client Library handles the
OAuth/OIDC flow; this module forwards the resulting token to the Microsoft
Graph API via request-scoped context variables.
"""

import base64
import builtins
import json
import logging
import os
import re
import time
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote, unquote, urlparse

_list = builtins.list

from omni_onedrive_storage_service.backends.storage_backend_interface import (
    EtagMismatchError,
    FolderMode,
    ListEntry,
    Metadata,
    OptimisticLockingSupport,
    RedirectUploadResult,
    StorageBackendInterface,
    VersionInfo,
    VersionsOrder,
)

from .exceptions import ItemNotFoundError
from .graph_client import LARGE_FILE_THRESHOLD, GraphClient
from .models import DriveItem
from .path_mapper import PathMapper
from .request_context import request_drive_id

logger = logging.getLogger(__name__)


# Upload chunk size for large files (10 MB - must be multiple of 320KB per Graph API docs)
UPLOAD_CHUNK_SIZE = 10 * 1024 * 1024


@dataclass
class UploadSessionInfo:
    """Information about an active Graph API upload session."""

    upload_url: str
    resource_address: str
    expiration: datetime | None
    total_size: int


class OneDriveStorageProvider(StorageBackendInterface):
    """Storage backend for OneDrive for Business via per-user Bearer tokens.

    The Kit Client Library handles OAuth/OIDC and passes a Bearer token on
    each request. This provider forwards that token to the Microsoft Graph API
    and resolves the user's OneDrive drive ID automatically via /me/drive.

    Resource Address Format:
        onedrive://me/<path/to/file>
        Versioned: onedrive://me/Documents/model.usd;3

    Resource Identity Format:
        onedrive-id://me/<base64-encoded-payload>
        Payload: {"drive_id": "...", "item_id": "...", "version_id": "...", "path": "..."}
    """

    IDENTITY_SCHEMA = "onedrive-id"
    ADDRESS_SCHEMA = "onedrive"

    def __init__(
        self,
        base_uri: str = "onedrive://me",
        cache_ttl: int = 300,
    ):
        """Initialize the OneDrive storage provider.

        Args:
            base_uri: Base URI for resource addresses (default "onedrive://me")
            cache_ttl: Path cache TTL in seconds
        """
        self._client = GraphClient()
        self._path_mapper = PathMapper(self._client, cache_ttl)

        self._base_uri = base_uri.rstrip("/") + "/"

        self._upload_sessions: dict[str, UploadSessionInfo] = {}

        logger.info(f"Initialized OneDrive storage provider (base_uri={self._base_uri})")

    @property
    def graph_client(self) -> GraphClient:
        """Expose the GraphClient for use by middleware."""
        return self._client

    @property
    def _drive_id(self) -> str:
        """The current user's drive ID, resolved from request context."""
        return request_drive_id.get()

    # =========================================================================
    # Configuration Properties
    # =========================================================================

    @property
    def base_uri(self) -> str:
        """Base URI for this storage backend instance."""
        return self._base_uri

    def folder_mode(self) -> FolderMode:
        """OneDrive has real folders (NATIVE mode)."""
        return FolderMode.NATIVE

    def get_optimistic_locking_support(self) -> OptimisticLockingSupport:
        """OneDrive supports ETags for conditional operations."""
        return OptimisticLockingSupport(
            write=True,
            delete=True,
            copy=False,  # Copy is async
            move=True,
        )

    # =========================================================================
    # Resource Address Validation and Conversion
    # =========================================================================

    def is_address_valid(self, resource_address: str) -> bool:
        """Check if a resource address is valid for this backend."""
        try:
            parsed = urlparse(resource_address)
            base_parsed = urlparse(self._base_uri)

            if parsed.scheme != base_parsed.scheme:
                return False
            return parsed.netloc == base_parsed.netloc
        except Exception:
            return False

    def is_version_address(self, resource_address: str) -> bool:
        """Check if address refers to a specific version."""
        parsed = urlparse(resource_address)
        return bool(re.search(r";[0-9]+$", parsed.path))

    def _extract_path(self, resource_address: str) -> str:
        """Extract relative, URL-decoded path from a resource address.

        Kit and other well-behaved clients percent-encode reserved characters
        (e.g. space -> ``%20``) on the wire. ``urlparse().path`` preserves that
        encoding verbatim, so we must ``unquote`` here to obtain the raw
        folder/file name (``New project``). GraphClient re-encodes exactly
        once when building the Graph API URL; skipping this decode causes
        double-encoding (``New%2520project``) and spurious 404s.

        The version suffix (``;<index>``) is a structural delimiter that lives
        in the *encoded* path, so it is stripped before decoding. This mirrors
        ``is_version_address``/``_extract_version_index`` (which also inspect the
        encoded path) and preserves literal semicolons inside a file name
        (encoded as ``%3B``), which would otherwise be truncated.
        """
        parsed = urlparse(resource_address)
        path = re.sub(r";[0-9]+$", "", parsed.path)
        path = unquote(path)
        if path.startswith("/"):
            path = path[1:]
        if path.endswith("/"):
            path = path[:-1]
        return path

    def _extract_version_index(self, resource_address: str) -> int | None:
        """Extract version index from versioned address."""
        parsed = urlparse(resource_address)
        match = re.search(r";([0-9]+)$", parsed.path)
        if match:
            return int(match.group(1))
        return None

    def _build_address(self, path: str, version_index: int | None = None) -> str:
        """Construct an outbound resource address from a decoded path.

        ``_extract_path`` returns paths in their raw, decoded form, so any
        address we hand back to the client must be re-encoded to be a valid
        URI (e.g. spaces must become ``%20``).
        """
        encoded = quote(path.lstrip("/"), safe="/")
        address = self._base_uri + encoded
        if version_index is not None:
            address = f"{address};{version_index}"
        return address

    def create_identity_from_resource_address(self, resource_address: str) -> str:
        """Create a resource identity from a resource address.

        Uses a single get_item() call instead of separate is_dir() + get_item().
        """
        path = self._extract_path(resource_address)

        try:
            item = self._path_mapper.get_item(path)
        except ItemNotFoundError:
            raise FileNotFoundError(f"Resource not found: {resource_address}") from None

        if item.is_folder:
            raise ValueError(f"Cannot create identity for folder: {resource_address}")

        version_idx = self._extract_version_index(resource_address)
        versions = self._client.list_versions(self._drive_id, item.id)

        if version_idx is not None:
            if version_idx >= len(versions.value):
                raise FileNotFoundError(f"Version {version_idx} not found for {resource_address}")
            version_id = versions.value[version_idx].id
        elif versions.value:
            version_id = versions.value[0].id
        else:
            version_id = "current"

        return self._create_identity(item.id, version_id, path)

    def _create_identity(self, item_id: str, version_id: str, path: str) -> str:
        """Create an identity string from components."""
        identity_data = {
            "drive_id": self._drive_id,
            "item_id": item_id,
            "version_id": version_id,
            "path": path,
        }
        encoded = base64.urlsafe_b64encode(json.dumps(identity_data).encode()).decode()
        return f"{self.IDENTITY_SCHEMA}://me/{encoded}"

    def _decode_identity(self, resource_identity: str) -> dict[str, Any]:
        """Decode an identity string to its components."""
        parsed = urlparse(resource_identity)
        if parsed.scheme != self.IDENTITY_SCHEMA:
            raise ValueError(f"Invalid identity schema: {parsed.scheme}")

        encoded = parsed.path.lstrip("/")
        try:
            decoded = base64.urlsafe_b64decode(encoded).decode()
            return json.loads(decoded)
        except Exception as e:
            raise ValueError(f"Invalid identity encoding: {resource_identity}") from e

    def address_from_identity(self, resource_identity: str) -> str:
        """Convert identity back to resource address (without version suffix)."""
        identity_data = self._decode_identity(resource_identity)
        path = identity_data["path"]
        return self._build_address(path)

    def url_from_identity(self, resource_identity: str) -> str:
        """Convert identity to URL including version suffix."""
        identity_data = self._decode_identity(resource_identity)
        path = identity_data["path"]
        version_id = identity_data.get("version_id", "current")

        if version_id != "current":
            item_id = identity_data["item_id"]
            versions = self._client.list_versions(self._drive_id, item_id)
            for idx, v in enumerate(versions.value):
                if v.id == version_id:
                    return self._build_address(path, version_index=idx)

        return self._build_address(path)

    # =========================================================================
    # File Existence and Type Checking
    # =========================================================================

    def exists(self, resource_address: str) -> bool:
        """Check if a resource exists, using PathMapper cache when available."""
        path = self._extract_path(resource_address)
        if not path:
            return True  # Root always exists
        if self._path_mapper.is_cached(path):
            return True
        try:
            self._path_mapper.get_item(path)
            return True
        except ItemNotFoundError:
            return False

    def is_file(self, resource_address: str) -> bool:
        """Check if address points to a file."""
        path = self._extract_path(resource_address)
        try:
            item = self._path_mapper.get_item(path)
            return item.is_file
        except ItemNotFoundError:
            return False

    def is_dir(self, resource_address: str) -> bool:
        """Check if address points to a directory."""
        path = self._extract_path(resource_address)
        if not path:
            return True  # Root is always a directory
        try:
            item = self._path_mapper.get_item(path)
            return item.is_folder
        except ItemNotFoundError:
            return False

    # =========================================================================
    # File Operations
    # =========================================================================

    def read_from_address(self, resource_address: str) -> Generator[bytes, None, None]:
        """Read file content from an address."""
        if not self.exists(resource_address):
            raise FileNotFoundError(f"Resource not found: {resource_address}")
        if self.is_dir(resource_address):
            raise ValueError(f"Cannot read from folder: {resource_address}")

        path = self._extract_path(resource_address)
        version_idx = self._extract_version_index(resource_address)

        item = self._path_mapper.get_item(path)

        if version_idx is not None:
            # Read specific version
            versions = self._client.list_versions(self._drive_id, item.id)
            if version_idx >= len(versions.value):
                raise FileNotFoundError(f"Version {version_idx} not found")
            version_id = versions.value[version_idx].id
            yield from self._client.download_version_content(self._drive_id, item.id, version_id)
        else:
            # Read latest version
            yield from self._client.download_content(self._drive_id, item.id)

    def read_from_identity(self, resource_identity: str) -> Generator[bytes, None, None]:
        """Read file content from a specific version by identity."""
        try:
            identity_data = self._decode_identity(resource_identity)
        except Exception as e:
            raise ValueError(f"Invalid identity: {resource_identity}") from e

        item_id = identity_data["item_id"]
        version_id = identity_data.get("version_id", "current")

        if version_id == "current":
            yield from self._client.download_content(self._drive_id, item_id)
        else:
            yield from self._client.download_version_content(self._drive_id, item_id, version_id)

    def write_version(
        self,
        resource_address: str,
        content: bytes,
        previous_version: str | None = None,
    ) -> str:
        """Write content to a resource address, creating a new version."""
        if self.is_dir(resource_address):
            raise ValueError(f"Cannot write to folder: {resource_address}")

        path = self._extract_path(resource_address)

        # Check optimistic locking
        if_match: str | None = None
        if previous_version is not None:
            try:
                current_info = self.stat(resource_address)
                if current_info.resource_identity != previous_version:
                    raise EtagMismatchError(
                        key=resource_address,
                        expected_etag=previous_version,
                        actual_etag=current_info.resource_identity,
                    )
                # Get ETag for conditional update
                item = self._path_mapper.get_item(path)
                if_match = item.e_tag
            except FileNotFoundError:
                # File doesn't exist yet, previous_version should be None
                raise EtagMismatchError(
                    key=resource_address,
                    expected_etag=previous_version,
                    actual_etag="(not found)",
                ) from None

        # Split path into parent and filename
        parent_path = os.path.dirname(path) or "/"
        filename = os.path.basename(path)

        # Upload content with retry logic for transient ETag conflicts.
        # OneDrive may return 412 on concurrent writes to the same file even
        # without an explicit If-Match header. When we are NOT doing optimistic
        # locking (previous_version is None), these are transient and retryable.
        max_retries = 5 if previous_version is None else 0
        for attempt in range(max_retries + 1):
            try:
                if len(content) > LARGE_FILE_THRESHOLD:
                    item = self._upload_large_file(parent_path, filename, content)
                else:
                    item = self._client.upload_content(self._drive_id, parent_path, filename, content, if_match)
                break  # Success
            except EtagMismatchError:
                if attempt < max_retries:
                    backoff = 1.0 * (2**attempt)  # 1, 2, 4, 8, 16 seconds
                    logger.warning(
                        f"ETag conflict on write to {resource_address}, "
                        f"retrying in {backoff}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(backoff)
                else:
                    raise

        # Update cache
        self._path_mapper.update_cache(path, item)

        # Get the version ID of the newly created version
        versions = self._client.list_versions(self._drive_id, item.id)
        version_id = versions.value[0].id if versions.value else "current"

        return self._create_identity(item.id, version_id, path)

    def _upload_large_file(self, parent_path: str, filename: str, content: bytes) -> DriveItem:
        """Upload a large file using upload sessions."""
        session = self._client.create_upload_session(self._drive_id, parent_path, filename)

        total_size = len(content)
        uploaded = 0
        result_item: DriveItem | None = None

        while uploaded < total_size:
            chunk_end = min(uploaded + UPLOAD_CHUNK_SIZE, total_size)
            chunk = content[uploaded:chunk_end]

            result = self._client.upload_chunk(
                session.upload_url,
                chunk,
                uploaded,
                chunk_end - 1,
                total_size,
            )

            if result:
                result_item = result

            uploaded = chunk_end

        if result_item is None:
            raise RuntimeError("Upload completed but no result item returned")

        return result_item

    def stat(self, resource_address: str) -> VersionInfo:
        """Get metadata for a resource address.

        Uses a single get_item() call (cached) instead of separate
        exists() + is_dir() + get_item() to minimize Graph API roundtrips.
        """
        path = self._extract_path(resource_address)

        try:
            item = self._path_mapper.get_item(path)
        except ItemNotFoundError:
            raise FileNotFoundError(f"Resource not found: {resource_address}") from None

        if item.is_folder:
            raise IsADirectoryError(f"{resource_address} is a directory")

        version_idx = self._extract_version_index(resource_address)

        versions = self._client.list_versions(self._drive_id, item.id)

        if version_idx is not None:
            if version_idx >= len(versions.value):
                raise FileNotFoundError(f"Version {version_idx} not found for {resource_address}")
            version = versions.value[version_idx]
            identity = self._create_identity(item.id, version.id, path)
            last_modified = version.last_modified_date_time or datetime.now(timezone.utc)
            return VersionInfo(
                resource_identity=identity,
                metadata=Metadata(
                    data_object_size=version.size,
                    last_modified_timestamp=last_modified,
                ),
            )

        version_id = versions.value[0].id if versions.value else "current"
        identity = self._create_identity(item.id, version_id, path)
        last_modified = item.last_modified_date_time or datetime.now(timezone.utc)
        return VersionInfo(
            resource_identity=identity,
            metadata=Metadata(
                data_object_size=item.size,
                last_modified_timestamp=last_modified,
            ),
        )

    def is_version_latest(self, resource_address: str, version_identity: str) -> bool:
        """Check whether a client-supplied version identity is the current latest.

        Extends the default (exact identity-string equality) to also accept the
        ``current`` version sentinel. ``list_stat`` entries encode
        ``version_id="current"`` instead of resolving a concrete version id per
        file (which would be a per-file Graph call / N+1). Such a token refers to
        "the latest version" by definition, so it matches the current state as
        long as it points at the same item — without this, optimistic-locking
        callers that reuse a ``list_stat`` identity would hit spurious conflicts.
        """
        current_info = self.stat(resource_address)
        current_identity = current_info.resource_identity
        if current_identity == version_identity:
            return True
        try:
            provided = self._decode_identity(version_identity)
            current = self._decode_identity(current_identity)
        except ValueError:
            return False
        if provided.get("item_id") != current.get("item_id"):
            return False
        return provided.get("version_id", "current") == "current"

    def stat_identity(self, resource_identity: str) -> VersionInfo:
        """Get metadata for a specific version by identity."""
        try:
            identity_data = self._decode_identity(resource_identity)
        except Exception as e:
            raise ValueError(f"Invalid identity: {resource_identity}") from e

        item_id = identity_data["item_id"]
        version_id = identity_data.get("version_id", "current")

        if version_id == "current":
            item = self._client.get_item_by_id(self._drive_id, item_id)
            last_modified = item.last_modified_date_time or datetime.now(timezone.utc)
            return VersionInfo(
                resource_identity=resource_identity,
                metadata=Metadata(
                    data_object_size=item.size,
                    last_modified_timestamp=last_modified,
                ),
            )
        else:
            version = self._client.get_version(self._drive_id, item_id, version_id)
            last_modified = version.last_modified_date_time or datetime.now(timezone.utc)
            return VersionInfo(
                resource_identity=resource_identity,
                metadata=Metadata(
                    data_object_size=version.size,
                    last_modified_timestamp=last_modified,
                ),
            )

    def remove_by_address(self, resource_address: str) -> None:
        """Remove the current version (moves to recycle bin)."""
        if not self.exists(resource_address):
            raise FileNotFoundError(f"Resource not found: {resource_address}")

        path = self._extract_path(resource_address)
        item = self._path_mapper.get_item(path)

        self._client.delete_item(self._drive_id, item.id)
        self._path_mapper.invalidate(path)

    def obliterate(self, resource_address: str) -> None:
        """Permanently delete resource (same as remove for OneDrive)."""
        # OneDrive doesn't have a separate "obliterate" - delete moves to recycle bin
        # For true permanent delete, would need to also delete from recycle bin
        self.remove_by_address(resource_address)

    # =========================================================================
    # Folder Operations
    # =========================================================================

    def create_folder(self, resource_address: str) -> None:
        """Create a folder."""
        path = self._extract_path(resource_address)

        if self.exists(resource_address):
            if self.is_file(resource_address):
                raise FileExistsError(f"A file exists at this path: {resource_address}")
            return  # Folder already exists

        parent_path = os.path.dirname(path) or "/"
        folder_name = os.path.basename(path)

        if parent_path and parent_path != "/":
            parent_address = self._build_address(parent_path)
            if not self.exists(parent_address):
                self.create_folder(parent_address)

        item = self._client.create_folder(self._drive_id, parent_path, folder_name)
        self._path_mapper.update_cache(path, item)

    def list(self, resource_address: str) -> tuple[_list[str], _list[str]]:
        """List immediate children of a folder."""
        if not self.exists(resource_address):
            raise FileNotFoundError(f"Folder not found: {resource_address}")
        if self.is_file(resource_address):
            raise ValueError(f"Not a folder: {resource_address}")

        path = self._extract_path(resource_address)
        children = self._client.list_children(self._drive_id, path=path or "/")

        folders: list[str] = []
        files: list[str] = []

        for item in children.value:
            if item.is_folder:
                folders.append(item.name)
            else:
                files.append(item.name)

        # Handle pagination
        while children.next_link:
            children = self._client.list_children_next_page(children.next_link)
            for item in children.value:
                if item.is_folder:
                    folders.append(item.name)
                else:
                    files.append(item.name)

        return folders, files

    def list_stat(
        self,
        resource_address: str,
        start_index: int = 0,
        limit: int | None = None,
    ) -> tuple[_list[str], _list[ListEntry]]:
        """List folder contents with metadata."""
        if not self.exists(resource_address):
            raise FileNotFoundError(f"Folder not found: {resource_address}")
        if self.is_file(resource_address):
            raise ValueError(f"Not a folder: {resource_address}")

        path = self._extract_path(resource_address)
        children = self._client.list_children(self._drive_id, path=path or "/")

        folders: list[str] = []
        file_entries: list[ListEntry] = []

        all_items: list[DriveItem] = list(children.value)

        # Handle pagination
        while children.next_link:
            children = self._client.list_children_next_page(children.next_link)
            all_items.extend(children.value)

        # Apply start_index and limit
        for item in all_items[start_index:]:
            if limit is not None and len(file_entries) + len(folders) >= limit:
                break

            if item.is_folder:
                folders.append(item.name)
            else:
                item_path = f"{path}/{item.name}" if path else item.name

                last_modified = item.last_modified_date_time or datetime.now(timezone.utc)

                # Build the identity with the sentinel "current" version rather
                # than resolving the latest version id per file. Resolving it
                # would issue one Graph `versions` call per item (an N+1 that
                # quickly trips Graph throttling when browsing large folders),
                # and read/stat of a "current" identity already targets the
                # latest version, so the concrete id adds no behavior here.
                identity = self._create_identity(item.id, "current", item_path)

                file_entries.append(
                    ListEntry(
                        resource_address=item.name,
                        metadata=Metadata(
                            data_object_size=item.size,
                            last_modified_timestamp=last_modified,
                        ),
                        resource_identity=identity,
                    )
                )

        return folders, file_entries

    def enumerate(
        self,
        resource_address: str,
        start_index: int = 0,
        limit: int | None = None,
    ) -> Generator[_list[ListEntry], None, None]:
        """Recursively enumerate all files under a directory."""
        if not self.exists(resource_address):
            raise FileNotFoundError(f"Directory not found: {resource_address}")
        if self.is_file(resource_address):
            raise ValueError(f"Not a directory: {resource_address}")
        if self.is_version_address(resource_address):
            raise ValueError(f"Cannot enumerate versioned address: {resource_address}")

        count = 0
        skipped = 0

        def _enumerate_folder(folder_path: str) -> Generator[_list[ListEntry], None, None]:
            nonlocal count, skipped

            children = self._client.list_children(self._drive_id, path=folder_path or "/")
            all_items: list[DriveItem] = list(children.value)

            while children.next_link:
                children = self._client.list_children_next_page(children.next_link)
                all_items.extend(children.value)

            batch: list[ListEntry] = []

            for item in all_items:
                if limit is not None and count >= limit:
                    if batch:
                        yield batch
                    return

                item_path = f"{folder_path}/{item.name}" if folder_path else item.name

                if item.is_folder:
                    # Recursively enumerate subfolder
                    yield from _enumerate_folder(item_path)
                else:
                    if skipped < start_index:
                        skipped += 1
                        continue

                    item_address = self._build_address(item_path)
                    last_modified = item.last_modified_date_time or datetime.now(timezone.utc)

                    batch.append(
                        ListEntry(
                            resource_address=item_address,
                            metadata=Metadata(
                                data_object_size=item.size,
                                last_modified_timestamp=last_modified,
                            ),
                        )
                    )
                    count += 1

                    # Yield batches of 100
                    if len(batch) >= 100:
                        yield batch
                        batch = []

            if batch:
                yield batch

        path = self._extract_path(resource_address)
        yield from _enumerate_folder(path)

    def remove_empty_folder(self, resource_address: str) -> bool:
        """Remove an empty folder."""
        if not self.exists(resource_address):
            raise FileNotFoundError(f"Folder not found: {resource_address}")
        if self.is_file(resource_address):
            raise ValueError(f"Not a folder: {resource_address}")

        folders, files = self.list(resource_address)
        if folders or files:
            return False  # Not empty

        path = self._extract_path(resource_address)
        item = self._path_mapper.get_item(path)

        self._client.delete_item(self._drive_id, item.id)
        self._path_mapper.invalidate(path)

        return True

    # =========================================================================
    # Versioning Operations
    # =========================================================================

    def enumerate_versions(
        self,
        resource_address: str,
        start_index: int = 0,
        limit: int | None = None,
    ) -> tuple[_list[VersionInfo], VersionsOrder]:
        """Enumerate all versions of a resource."""
        if self.is_dir(resource_address):
            raise ValueError(f"Cannot enumerate versions for folder: {resource_address}")
        if self.is_version_address(resource_address):
            raise ValueError(f"Cannot enumerate versions for versioned address: {resource_address}")

        path = self._extract_path(resource_address)

        if not self.exists(resource_address):
            raise FileNotFoundError(f"Resource not found: {resource_address}")

        item = self._path_mapper.get_item(path)
        versions = self._client.list_versions(self._drive_id, item.id)

        result: list[VersionInfo] = []

        for idx, v in enumerate(versions.value[start_index:]):
            if limit is not None and len(result) >= limit:
                break

            identity = self._create_identity(item.id, v.id, path)
            last_modified = v.last_modified_date_time or datetime.now(timezone.utc)

            result.append(
                VersionInfo(
                    resource_identity=identity,
                    metadata=Metadata(
                        data_object_size=v.size,
                        last_modified_timestamp=last_modified,
                    ),
                    sorting_key=f"{start_index + idx:010d}",
                    resource_address=f"{resource_address};{start_index + idx}",
                )
            )

        return result, VersionsOrder.NEWEST_FIRST

    # =========================================================================
    # Metadata Operations (Stubbed - OneDrive doesn't support custom metadata)
    # =========================================================================

    def get_metadata(
        self,
        metadata_uri: str,
        keys: _list[str],
    ) -> dict[str, dict[str, Any]]:
        """Get user-defined metadata - returns empty dict.

        OneDrive doesn't support custom user-defined metadata in the same way
        as the filesystem backend. We return an empty dict to indicate no
        metadata is available.
        """
        return {}

    def update_metadata(
        self,
        metadata_uri: str,
        key: str,
        value: str,
        expected_etag: str | None = None,
    ) -> str:
        """Update metadata - no-op, returns empty etag.

        OneDrive doesn't support custom user-defined metadata.
        This is a no-op that returns an empty etag.
        """
        return ""

    def delete_metadata(
        self,
        metadata_uri: str,
        key: str,
        expected_etag: str | None = None,
    ) -> None:
        """Delete metadata - no-op.

        OneDrive doesn't support custom user-defined metadata.
        This is a no-op.
        """
        pass

    # =========================================================================
    # Permission Operations
    # =========================================================================

    def check_read_permission_on_address(self, resource_address: str) -> bool:
        """Check if the current user can read this resource.

        Attempts to resolve the item via the path cache (which calls Graph).
        Returns True optimistically if the check itself fails for non-permission
        reasons (network, not-found, etc.) — the actual read will produce the
        real error. Only raises PermissionError if Graph explicitly denies access.
        """
        try:
            path = self._extract_path(resource_address)
            self._path_mapper.get_item(path)
            return True
        except PermissionError:
            raise
        except Exception:
            return True

    # =========================================================================
    # Upload/Download Support
    # =========================================================================

    def supports_redirect_download(self) -> bool:
        """OneDrive provides pre-authenticated download URLs via Graph API."""
        return True

    def supports_redirect_upload(self) -> bool:
        """Redirect uploads disabled - service layer hardcodes local /upload/ endpoint.

        The REST service constructs redirect URLs as /upload/{path} rather than
        calling backend.construct_redirect_url(), so we can't redirect to Microsoft.
        Instead, uploads go through write_version() which uses Graph API directly.
        """
        return False

    def supports_multipart_upload(self) -> bool:
        """Multipart uploads disabled for now.

        The multipart upload flow also expects local endpoints. Graph API has
        upload sessions but they work differently than the local multipart flow.
        Large files are handled via write_version() which uses Graph upload sessions.
        """
        return False

    def construct_redirect_url(
        self,
        resource_address: str,
        redirect_host: str,
        redirect_port: int,
    ) -> str:
        """Return Graph API's pre-authenticated download URL.

        Graph API provides @microsoft.graph.downloadUrl which is a pre-authenticated
        URL pointing to Microsoft's CDN. The URL expires after approximately 1 hour.

        Args:
            resource_address: Resource address to download
            redirect_host: Ignored - we redirect to Microsoft's servers
            redirect_port: Ignored - we redirect to Microsoft's servers

        Returns:
            Pre-authenticated download URL from Microsoft

        Raises:
            FileNotFoundError: If the resource doesn't exist
            ValueError: If no download URL is available
        """
        path = self._extract_path(resource_address)
        item = self._path_mapper.get_item(path)

        # Fetch fresh item to get download URL (it expires after ~1 hour)
        fresh_item = self._client.get_item_with_download_url(self._drive_id, item.id)
        if not fresh_item.download_url:
            raise ValueError(f"No download URL available for item: {resource_address}")

        return fresh_item.download_url

    def construct_redirect_url_for_identity(
        self,
        resource_identity: str,
        redirect_host: str,
        redirect_port: int,
    ) -> str:
        """Return download URL for a specific version by identity.

        Args:
            resource_identity: Resource identity to download
            redirect_host: Ignored - we redirect to Microsoft's servers
            redirect_port: Ignored - we redirect to Microsoft's servers

        Returns:
            Pre-authenticated download URL for the specific version

        Raises:
            ValueError: If the identity is invalid
            VersionNotFoundError: If the version doesn't exist
        """
        identity_data = self._decode_identity(resource_identity)
        item_id = identity_data["item_id"]
        version_id = identity_data.get("version_id", "current")

        if version_id == "current":
            # Get current version's download URL
            fresh_item = self._client.get_item_with_download_url(self._drive_id, item_id)
            if not fresh_item.download_url:
                raise ValueError(f"No download URL available for item: {item_id}")
            return fresh_item.download_url
        else:
            # Get specific version's download URL
            return self._client.get_version_download_url(self._drive_id, item_id, version_id)

    # =========================================================================
    # Multipart Upload Support
    # =========================================================================

    def create_upload_session(self, upload_id: str) -> None:
        """Create a new multipart upload session.

        The actual Graph API upload session is created lazily when the first
        part is uploaded, since we need to know the file path at that point.

        Args:
            upload_id: Unique identifier for this upload session
        """
        # Session will be created lazily in construct_upload_part_redirect
        # when we have the resource_address and can create the Graph session
        pass

    def upload_session_exists(self, upload_id: str) -> bool:
        """Check if an upload session exists.

        Args:
            upload_id: Upload session identifier

        Returns:
            True if session exists, False otherwise
        """
        return upload_id in self._upload_sessions

    def cleanup_upload_session(self, upload_id: str) -> None:
        """Clean up resources for a completed or aborted upload session.

        Args:
            upload_id: Upload session identifier
        """
        self._upload_sessions.pop(upload_id, None)

    def construct_upload_part_redirect(
        self,
        upload_id: str,
        part_number: int,
        resource_address: str,
        redirect_host: str,
        redirect_port: int,
    ) -> dict[str, Any]:
        """Construct redirect properties for uploading a multipart part.

        Returns redirect info pointing to Microsoft's Graph API upload session URL.
        The client uploads chunks directly to Microsoft's servers.

        Args:
            upload_id: Upload session identifier
            part_number: Part number (0-indexed)
            resource_address: Target resource address
            redirect_host: Ignored - we redirect to Microsoft's servers
            redirect_port: Ignored - we redirect to Microsoft's servers

        Returns:
            Dict with redirect properties for the upload part
        """
        # Get or create Graph upload session
        if upload_id not in self._upload_sessions:
            path = self._extract_path(resource_address)
            parent_path = os.path.dirname(path) or "/"
            filename = os.path.basename(path)

            session = self._client.create_upload_session(self._drive_id, parent_path, filename)
            self._upload_sessions[upload_id] = UploadSessionInfo(
                upload_url=session.upload_url,
                resource_address=resource_address,
                expiration=session.expiration_date_time,
                total_size=0,  # Will be set from Content-Range header by client
            )

        session_info = self._upload_sessions[upload_id]

        return {
            "redirect_target_url": session_info.upload_url,
            "method": "PUT",
            "additional_headers": [],  # Content-Range added by client
            "completion_header_names": [],
        }

    def complete_redirect_upload(
        self,
        destination_resource_address: str,
        completion_headers: dict[str, str],
    ) -> RedirectUploadResult:
        """Complete a redirect-based upload.

        For OneDrive, the Graph API automatically creates the item when the
        upload completes. We fetch the item to get its identity and metadata.

        Args:
            destination_resource_address: Target address for the upload
            completion_headers: Headers returned from the upload endpoint

        Returns:
            RedirectUploadResult with resource_identity and metadata
        """
        path = self._extract_path(destination_resource_address)

        # Invalidate cache since a new file was uploaded
        self._path_mapper.invalidate(path)

        # Fetch the newly created/updated item
        item = self._path_mapper.get_item(path)

        # Get version info
        versions = self._client.list_versions(self._drive_id, item.id)
        version_id = versions.value[0].id if versions.value else "current"

        identity = self._create_identity(item.id, version_id, path)
        last_modified = item.last_modified_date_time or datetime.now(timezone.utc)

        return RedirectUploadResult(
            resource_identity=identity,
            metadata=Metadata(
                data_object_size=item.size,
                last_modified_timestamp=last_modified,
            ),
        )

    # =========================================================================
    # Copy/Move Operations
    # =========================================================================

    def copy(
        self,
        source_resource_address: str,
        destination_resource_address: str,
    ) -> str:
        """Copy resource to new address.

        Implemented as read-from-source + write-to-destination to correctly
        handle overwrites, self-copy, and versioned source addresses.
        """
        if self.is_dir(source_resource_address):
            raise ValueError(f"Cannot copy folder: {source_resource_address}")
        if not self.exists(source_resource_address):
            raise FileNotFoundError(f"Source not found: {source_resource_address}")

        # Read content from source (handles versioned addresses)
        content = b"".join(self.read_from_address(source_resource_address))

        # Write to destination, creating a new version (handles overwrites)
        return self.write_version(destination_resource_address, content)

    def move(
        self,
        source_resource_address: str,
        destination_resource_address: str,
    ) -> str:
        """Move/rename resource to new address.

        Implemented as copy + delete to correctly handle overwrites and
        avoid Graph API "Name already exists" errors.
        """
        if self.is_dir(source_resource_address):
            raise ValueError(f"Cannot move folder: {source_resource_address}")
        if not self.exists(source_resource_address):
            raise FileNotFoundError(f"Source not found: {source_resource_address}")

        # If source and destination are the same, this is a no-op
        if source_resource_address == destination_resource_address:
            return self.create_identity_from_resource_address(source_resource_address)

        # Copy to destination (creates new version, handles overwrites)
        result_identity = self.copy(source_resource_address, destination_resource_address)

        # Remove source
        self.remove_by_address(source_resource_address)

        return result_identity

    # =========================================================================
    # Upload ID Encoding (for multipart uploads)
    # =========================================================================

    def encode_upload_id(
        self,
        upload_id: str,
        previous_version: str | None = None,
    ) -> str:
        """Encode upload identifier."""
        data = {"upload_id": upload_id}
        if previous_version:
            data["previous_version"] = previous_version
        return json.dumps(data)

    def decode_upload_id(self, value: str) -> tuple[str, str | None]:
        """Decode upload identifier."""
        try:
            data = json.loads(value)
            return data["upload_id"], data.get("previous_version")
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError("Invalid upload_id") from e

# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Storage Backend Interface for pluggable storage implementations.

This module defines the abstract interface that all storage backends must implement
to be used with the Storage API service layer. This allows swapping different storage
implementations (local filesystem, Git repository, cloud storage like Dropbox, etc.)
without modifying the REST or gRPC service code.

The interface abstracts away backend-specific details like:
- How files are physically stored
- How versions are managed
- How metadata is persisted
- How folders are represented
- How uploads/downloads are handled

Backend implementations must provide all methods defined in StorageBackendInterface.
"""

import builtins
from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from fastapi import FastAPI


class EtagMismatchError(Exception):
    """Raised when an ETag mismatch is detected during metadata operations."""

    def __init__(self, key: str, expected_etag: str, actual_etag: str):
        self.key = key
        self.expected_etag = expected_etag
        self.actual_etag = actual_etag
        super().__init__(f"ETag mismatch for key '{key}': expected '{expected_etag}', got '{actual_etag}'")


class MetadataKeyNotFoundError(Exception):
    """Raised when a metadata key is not found during operations that require it to exist."""

    def __init__(self, key: str):
        self.key = key
        super().__init__(f"Metadata key '{key}' does not exist")


class VersionsOrder(Enum):
    """Ordering of versions returned by enumerate_versions."""

    NEWEST_FIRST = 1
    OLDEST_FIRST = 2
    BY_KEY = 3


class FolderMode(Enum):
    """Folder handling semantics for storage backends.

    Indicates how a storage backend handles folder persistence and lifecycle.
    This is used by clients to understand whether empty folders can exist,
    whether folder creation is required before writing files, etc.
    """

    NATIVE = 1  # Real filesystem directories - empty folders persist until deleted
    NO_EMPTY = 2  # Empty folders don't exist (like S3/Git) - folders are implicit
    HYBRID = 3  # Placeholder files simulate folders - explicit folders persist, implicit may disappear


@dataclass
class OptimisticLockingSupport:
    """Information about which operations support optimistic locking (conditional execution with previous_version)."""

    write: bool = False
    delete: bool = False
    copy: bool = False
    move: bool = False


@dataclass
class Metadata:
    """File metadata (size and modification time)."""

    data_object_size: int
    last_modified_timestamp: datetime


@dataclass
class VersionInfo:
    """Information about a specific version of a resource."""

    resource_identity: str  # Opaque identifier for this version
    metadata: Metadata
    sorting_key: str | None = None  # For ordering versions
    resource_address: str | None = None  # Version-specific address (v1alpha only)


@dataclass
class ListEntry:
    """Entry in a directory listing."""

    resource_address: str
    metadata: Metadata | None = None  # None for folders, populated for files
    resource_identity: str | None = None  # Populated for files


@dataclass
class RedirectUploadResult:
    """Result of completing a redirect upload."""

    resource_identity: str
    metadata: Metadata


class StorageBackendInterface(ABC):
    """Abstract interface for storage backends.

    All storage backends must implement this interface to be compatible with
    the Storage API service layer. The interface provides operations for:
    - File operations (read, write, delete, stat)
    - Folder operations (create, list, enumerate)
    - Versioning (enumerate versions, create versions)
    - Metadata (get, update, delete user-defined metadata)
    - Identity/resource address conversion
    - Permission checking

    Resource Address Format:
        Resource addresses are URIs that identify resources in the storage system.
        Format: <scheme>://<netloc>/<path>
        Example: "file-storage://server/path/to/file"

    Resource Identity Format:
        Resource identities are opaque identifiers that uniquely identify a specific
        version of a resource. They are backend-specific but must be:
        - Encodable as strings
        - Routable (can be converted back to resource addresses)
        - Version-specific (each version has a unique identity)

    Versioning Model:
        - Each write operation creates a new immutable version
        - Versions are identified by resource identities
        - Resource addresses point to the "latest" version
        - Versioned addresses can reference specific versions: <address>;version_number
    """

    # =============================================================================
    # Configuration and Initialization
    # =============================================================================

    @property
    @abstractmethod
    def base_uri(self) -> str:
        """Base URI for this storage backend instance.

        Returns:
            Base URI string (e.g., "file-storage://server")
        """
        pass

    # =============================================================================
    # Resource Address Validation and Conversion
    # =============================================================================

    @abstractmethod
    def is_address_valid(self, resource_address: str) -> bool:
        """Check if a resource address is valid for this backend.

        Args:
            resource_address: Resource address to validate

        Returns:
            True if the address is valid, False otherwise
        """
        pass

    @abstractmethod
    def is_version_address(self, resource_address: str) -> bool:
        """Check if a resource address refers to a specific version.

        Args:
            resource_address: Resource address to check

        Returns:
            True if the address contains a version suffix (e.g., ";123")
        """
        pass

    @abstractmethod
    def create_identity_from_resource_address(self, resource_address: str) -> str:
        """Create a resource identity from a resource address.

        Creates an identity for the latest version at the given address.

        Args:
            resource_address: Resource address to create identity for

        Returns:
            Opaque resource identity string

        Raises:
            ValueError: If resource_address is invalid or points to a folder
        """
        pass

    @abstractmethod
    def address_from_identity(self, resource_identity: str) -> str:
        """Convert a resource identity back to a resource address.

        Args:
            resource_identity: Opaque resource identity string

        Returns:
            Resource address for the version identified by the identity

        Raises:
            ValueError: If resource_identity is invalid or not routable
        """
        pass

    def url_from_identity(self, resource_identity: str) -> str:
        """Convert a resource identity to the full URL including version.

        Unlike address_from_identity(), this preserves version information
        in the URL, which is needed for operations on specific versions.

        Default implementation returns the same as address_from_identity().
        Backends with versioning support should override this.

        Args:
            resource_identity: Opaque resource identity string

        Returns:
            Full URL for the version, potentially including version suffix
        """
        return self.address_from_identity(resource_identity)

    def get_optimistic_locking_support(self) -> OptimisticLockingSupport:
        """Get information about which operations support optimistic locking.

        Optimistic locking allows clients to pass a previous_version parameter
        to operations like Write, Delete, Copy, and Move to ensure the operation
        only succeeds if the resource hasn't been modified since the specified version.

        Default implementation returns no support for any operation.
        Backends that support optimistic locking should override this.

        Returns:
            OptimisticLockingSupport data class indicating which operations support conditional execution
        """
        return OptimisticLockingSupport()

    def is_version_latest(self, resource_address: str, version_identity: str) -> bool:
        """Check if the given version identity is the latest version at the resource address.

        Used for optimistic locking to verify that a client's view of a resource
        is current before performing a write, delete, copy, or move operation.

        Args:
            resource_address: Resource address to check
            version_identity: Version identity to compare against the latest

        Returns:
            True if version_identity matches the latest version at resource_address

        Raises:
            FileNotFoundError: If no version exists at the resource address
        """
        current_info = self.stat(resource_address)
        return current_info.resource_identity == version_identity

    # =============================================================================
    # File Existence and Type Checking
    # =============================================================================

    @abstractmethod
    def exists(self, resource_address: str) -> bool:
        """Check if a resource exists at the given address.

        Args:
            resource_address: Resource address to check

        Returns:
            True if resource exists, False otherwise
        """
        pass

    @abstractmethod
    def is_file(self, resource_address: str) -> bool:
        """Check if a resource address points to a file.

        Args:
            resource_address: Resource address to check

        Returns:
            True if resource is a file, False otherwise
        """
        pass

    @abstractmethod
    def is_dir(self, resource_address: str) -> bool:
        """Check if a resource address points to a directory.

        Args:
            resource_address: Resource address to check

        Returns:
            True if resource is a directory, False otherwise
        """
        pass

    # =============================================================================
    # File Operations
    # =============================================================================

    @abstractmethod
    def read_from_address(self, resource_address: str) -> Generator[bytes, None, None]:
        """Read file content from a resource address.

        Yields chunks of file content. For versioned addresses, reads the
        specific version. For non-versioned addresses, reads the latest version.

        Args:
            resource_address: Resource address to read from

        Yields:
            Bytes chunks of file content

        Raises:
            FileNotFoundError: If resource doesn't exist
            ValueError: If resource_address points to a folder
            PermissionError: If read permission is denied
        """
        pass

    @abstractmethod
    def read_from_identity(self, resource_identity: str) -> Generator[bytes, None, None]:
        """Read file content from a specific version by its identity.

        Yields chunks of file content for the specific version identified by
        the resource identity. This allows reading historical versions even
        if the address was deleted.

        Args:
            resource_identity: Resource identity to read from

        Yields:
            Bytes chunks of file content

        Raises:
            FileNotFoundError: If the specific version doesn't exist
            ValueError: If resource_identity is invalid or malformed
            PermissionError: If read permission is denied
        """
        pass

    @abstractmethod
    def write_version(self, resource_address: str, content: bytes, previous_version: str | None = None) -> str:
        """Write content to a resource address, creating a new version.

        Creates a new immutable version of the resource. If previous_version
        is provided, validates that it matches the current latest version
        (optimistic concurrency control).

        Args:
            resource_address: Resource address to write to
            content: File content to write
            previous_version: Optional resource identity of expected latest version

        Returns:
            Resource identity of the newly created version

        Raises:
            ValueError: If resource_address points to a folder
            PermissionError: If write permission is denied
            EtagMismatchError: If previous_version doesn't match current latest
        """
        pass

    @abstractmethod
    def stat(self, resource_address: str) -> VersionInfo:
        """Get metadata for a resource address.

        Returns metadata for the current/latest version at an address.
        If the address was deleted (but versions still exist), this raises FileNotFoundError.

        Args:
            resource_address: Resource address to stat

        Returns:
            VersionInfo with resource identity and metadata

        Raises:
            FileNotFoundError: If resource doesn't exist (address is deleted or never existed)
            IsADirectoryError: If resource is a directory
            PermissionError: If read permission is denied
        """
        pass

    @abstractmethod
    def stat_identity(self, resource_identity: str) -> VersionInfo:
        """Get metadata for a specific version by its identity.

        Unlike stat(), this method:
        - Takes a resource_identity instead of resource_address
        - Returns metadata even if the address was deleted
        - Only checks if the specific version exists in version history

        This is used for operations that need to access specific versions
        regardless of whether the address is currently deleted.

        Args:
            resource_identity: The resource identity to stat

        Returns:
            VersionInfo with the same resource identity and metadata

        Raises:
            FileNotFoundError: If the specific version doesn't exist
            ValueError: If the identity is invalid or malformed
        """
        pass

    @abstractmethod
    def remove_by_address(self, resource_address: str) -> None:
        """Remove the current version of a resource.

        Removes only the latest version, leaving historical versions intact.

        Args:
            resource_address: Resource address to remove

        Raises:
            FileNotFoundError: If resource doesn't exist
            PermissionError: If delete permission is denied
        """
        pass

    @abstractmethod
    def obliterate(self, resource_address: str) -> None:
        """Permanently delete a resource and all its versions.

        Args:
            resource_address: Resource address to obliterate

        Raises:
            FileNotFoundError: If resource doesn't exist
            PermissionError: If delete permission is denied
        """
        pass

    # =============================================================================
    # Folder Operations
    # =============================================================================

    @abstractmethod
    def create_folder(self, resource_address: str) -> None:
        """Create a folder at the given resource address.

        Args:
            resource_address: Resource address for the folder

        Raises:
            FileExistsError: If a file exists at the requested path
            PermissionError: If write permission is denied
        """
        pass

    @abstractmethod
    def list(self, resource_address: str) -> tuple[list[str], list[str]]:
        """List immediate children of a folder.

        Returns separate lists of subfolder names and file names.

        Args:
            resource_address: Folder address to list

        Returns:
            Tuple of (subfolder_names, file_names)

        Raises:
            FileNotFoundError: If folder doesn't exist
            ValueError: If resource_address points to a file
        """
        pass

    @abstractmethod
    def list_stat(
        self, resource_address: str, start_index: int = 0, limit: int | None = None
    ) -> tuple[builtins.list[str], builtins.list[ListEntry]]:
        """List immediate children of a folder with metadata.

        Returns folders as names and files as ListEntry objects with metadata.
        Supports pagination via start_index and limit.

        Args:
            resource_address: Folder address to list
            start_index: Index to start from (for pagination)
            limit: Maximum number of items to return

        Returns:
            Tuple of (subfolder_names, file_entries_with_metadata)

        Raises:
            FileNotFoundError: If folder doesn't exist
            ValueError: If resource_address points to a file
        """
        pass

    @abstractmethod
    def enumerate(
        self, resource_address: str, start_index: int = 0, limit: int | None = None
    ) -> Generator[builtins.list[ListEntry], None, None]:
        """Recursively enumerate all files under a directory tree.

        Yields batches of ListEntry objects for all files found recursively.
        Folders are not included in the results, only files.

        Args:
            resource_address: Directory address to enumerate
            start_index: Index to start from (for pagination)
            limit: Maximum number of items to return (early termination)

        Yields:
            Lists of ListEntry objects (batches)

        Raises:
            FileNotFoundError: If directory doesn't exist
            ValueError: If resource_address points to a file or is versioned
        """
        pass

    @abstractmethod
    def remove_empty_folder(self, resource_address: str) -> bool:
        """Remove an empty folder.

        Args:
            resource_address: Folder address to remove

        Returns:
            True if folder was removed, False if it wasn't empty

        Raises:
            FileNotFoundError: If folder doesn't exist
            ValueError: If resource_address points to a file
        """
        pass

    # =============================================================================
    # Versioning Operations
    # =============================================================================

    @abstractmethod
    def enumerate_versions(
        self, resource_address: str, start_index: int = 0, limit: int | None = None
    ) -> tuple[builtins.list[VersionInfo], VersionsOrder]:
        """Enumerate all versions of a resource.

        Args:
            resource_address: Resource address to enumerate versions for
            start_index: Index to start from (for pagination)
            limit: Maximum number of versions to return

        Returns:
            Tuple of (list of VersionInfo objects, VersionsOrder enum)

        Raises:
            ValueError: If resource_address points to a folder or is versioned
            FileNotFoundError: If resource doesn't exist
        """
        pass

    # =============================================================================
    # Metadata Operations
    # =============================================================================

    @abstractmethod
    def get_metadata(self, metadata_uri: str, keys: builtins.list[str]) -> dict[str, dict[str, Any]]:
        """Get user-defined metadata for a resource.

        metadata_uri can be either a resource_address or a resource_identity.

        Args:
            metadata_uri: Resource address or identity to get metadata for
            keys: List of metadata keys to retrieve (empty list = all keys)

        Returns:
            Dictionary mapping keys to metadata values (with 'value' and 'etag' fields)

        Raises:
            ValueError: If metadata_uri is invalid
        """
        pass

    @abstractmethod
    def update_metadata(self, metadata_uri: str, key: str, value: str, expected_etag: str | None = None) -> str:
        """Update user-defined metadata for a resource.

        Args:
            metadata_uri: Resource address or identity to update metadata for
            key: Metadata key to update
            value: New metadata value (string)
            expected_etag: Optional ETag for optimistic concurrency control

        Returns:
            New ETag for the updated metadata

        Raises:
            ValueError: If metadata_uri is invalid
            MetadataKeyNotFoundError: If key doesn't exist and expected_etag provided
            EtagMismatchError: If expected_etag doesn't match current ETag
        """
        pass

    @abstractmethod
    def delete_metadata(self, metadata_uri: str, key: str, expected_etag: str | None = None) -> None:
        """Delete user-defined metadata for a resource.

        Args:
            metadata_uri: Resource address or identity to delete metadata for
            key: Metadata key to delete
            expected_etag: Optional ETag for optimistic concurrency control

        Raises:
            ValueError: If metadata_uri is invalid
            EtagMismatchError: If expected_etag doesn't match current ETag
        """
        pass

    # =============================================================================
    # Permission Operations
    # =============================================================================

    @abstractmethod
    def check_read_permission_on_address(self, resource_address: str) -> bool:
        """Check if a resource address can be read.

        Args:
            resource_address: Resource address to check

        Returns:
            True if readable, False otherwise

        Raises:
            PermissionError: If permission check fails
        """
        pass

    # =============================================================================
    # Upload/Download Support (for redirect-based operations)
    # =============================================================================

    @abstractmethod
    def construct_redirect_url(self, resource_address: str, redirect_host: str, redirect_port: int) -> str:
        """Construct a redirect URL for upload/download operations.

        Backends that support redirect-based uploads/downloads should return
        URLs that point to backend-specific endpoints. Backends that don't
        support redirects can raise NotImplementedError.

        Args:
            resource_address: Resource address for the redirect
            redirect_host: Hostname for the redirect URL
            redirect_port: Port for the redirect URL

        Returns:
            Redirect URL string

        Raises:
            NotImplementedError: If redirects are not supported
        """
        pass

    # =============================================================================
    # Capability Query Methods
    # =============================================================================

    def supports_redirect_download(self) -> bool:
        """Return True if backend supports redirect-based downloads.

        Backends that support this will provide redirect URLs pointing
        to backend-specific endpoints for downloading files.
        Default implementation returns False.
        """
        return False

    def supports_redirect_upload(self) -> bool:
        """Return True if backend supports redirect-based uploads.

        Backends that support this will provide redirect URLs pointing
        to backend-specific endpoints for uploading files.
        Default implementation returns False.
        """
        return False

    def supports_multipart_upload(self) -> bool:
        """Return True if backend supports multipart uploads.

        Default implementation returns False.
        """
        return False

    def folder_mode(self) -> FolderMode:
        """Get the folder mode for this storage backend.

        Different storage backends handle folders differently:
        - NATIVE: Real filesystem directories (like local filesystem or Nucleus).
          Empty folders can be created and persist until explicitly deleted.
        - NO_EMPTY: Empty folders don't exist (like AWS S3 or Git).
          Folders are implicitly created when files are added and may disappear
          when the last file is removed.
        - HYBRID: Placeholder files simulate empty folders. Explicit folders
          created via create_folder persist, implicit folders may disappear.

        Returns:
            FolderMode enum indicating how this backend handles folders.

        Note:
            Default implementation returns NATIVE. Backends with different
            folder semantics should override this method.
        """
        return FolderMode.NATIVE

    # =============================================================================
    # HTTP Route Registration
    # =============================================================================

    def register_http_routes(self, app: "FastAPI") -> None:  # noqa: B027
        """Register backend-specific HTTP routes on the FastAPI application.

        Optional override. Backends that support redirect-based operations
        should register their upload/download endpoints here. The default
        implementation does nothing.

        Args:
            app: FastAPI application to register routes on
        """

    # =============================================================================
    # Redirect URL Construction for Identity-based Reads
    # =============================================================================

    def construct_redirect_url_for_identity(
        self, resource_identity: str, redirect_host: str, redirect_port: int
    ) -> str:
        """Construct a redirect URL for reading a specific version by identity.

        Args:
            resource_identity: Resource identity to read
            redirect_host: Hostname for the redirect URL
            redirect_port: Port for the redirect URL

        Returns:
            Redirect URL string

        Raises:
            NotImplementedError: If redirects are not supported
        """
        raise NotImplementedError("Redirect downloads not supported by this backend")

    # =============================================================================
    # Redirect Upload Completion
    # =============================================================================

    def complete_redirect_upload(
        self, destination_resource_address: str, completion_headers: dict[str, str]
    ) -> RedirectUploadResult:
        """Complete a redirect-based upload using completion headers from the HTTP response.

        After a client uploads via HTTP redirect, the service layer calls this method
        with the headers returned from the upload endpoint to finalize the upload
        and get the resulting resource info.

        Args:
            destination_resource_address: Target address for the upload
            completion_headers: Headers returned from the HTTP upload endpoint
                               (e.g., {'x-nvidia-storage-upload-location': 'path/to/version'})

        Returns:
            RedirectUploadResult with resource_identity and metadata

        Raises:
            NotImplementedError: If redirect uploads not supported
            ValueError: If completion_headers are invalid or missing required data
            FileNotFoundError: If the uploaded file cannot be found
        """
        raise NotImplementedError("Redirect uploads not supported by this backend")

    # =============================================================================
    # Multipart Upload Management
    # =============================================================================

    def create_upload_session(self, upload_id: str) -> None:
        """Create a new multipart upload session.

        Args:
            upload_id: Unique identifier for this upload session

        Raises:
            NotImplementedError: If multipart uploads not supported
        """
        raise NotImplementedError("Multipart uploads not supported by this backend")

    def get_upload_part_path(self, upload_id: str, part_number: int, resource_address: str) -> str:
        """Get the filesystem path where an upload part should be stored.

        Args:
            upload_id: Upload session identifier
            part_number: Part number (0-indexed)
            resource_address: Target resource address

        Returns:
            Filesystem path for storing the part

        Raises:
            NotImplementedError: If multipart uploads not supported
        """
        raise NotImplementedError("Multipart uploads not supported by this backend")

    def cleanup_upload_session(self, upload_id: str) -> None:
        """Clean up resources for a completed or aborted upload session.

        Args:
            upload_id: Upload session identifier

        Raises:
            NotImplementedError: If multipart uploads not supported
        """
        raise NotImplementedError("Multipart uploads not supported by this backend")

    def upload_session_exists(self, upload_id: str) -> bool:
        """Check if an upload session exists.

        Args:
            upload_id: Upload session identifier

        Returns:
            True if session exists, False otherwise

        Raises:
            NotImplementedError: If multipart uploads not supported
        """
        raise NotImplementedError("Multipart uploads not supported by this backend")

    def construct_upload_part_redirect(
        self, upload_id: str, part_number: int, resource_address: str, redirect_host: str, redirect_port: int
    ) -> dict[str, Any]:
        """Construct redirect properties for uploading a multipart part.

        Args:
            upload_id: Upload session identifier
            part_number: Part number (0-indexed)
            resource_address: Target resource address
            redirect_host: Host for redirect URL
            redirect_port: Port for redirect URL

        Returns:
            Dict with redirect properties:
            - redirect_target_url: URL for uploading the part
            - method: HTTP method (e.g., "POST")
            - additional_headers: Headers client must include
            - completion_header_names: Headers to collect from response

        Raises:
            NotImplementedError: If multipart uploads not supported
        """
        raise NotImplementedError("Multipart uploads not supported by this backend")

    # =============================================================================
    # Multipart Upload Support (Legacy - Upload ID Encoding)
    # =============================================================================

    @abstractmethod
    def encode_upload_id(self, upload_id: str, previous_version: str | None = None) -> str:
        """Encode an upload identifier for multipart uploads.

        Args:
            upload_id: Unique upload identifier
            previous_version: Optional resource identity of expected latest version

        Returns:
            Encoded upload identifier string
        """
        pass

    @abstractmethod
    def decode_upload_id(self, value: str) -> tuple[str, str | None]:
        """Decode an upload identifier for multipart uploads.

        Args:
            value: Encoded upload identifier string

        Returns:
            Tuple of (upload_id, previous_version)

        Raises:
            ValueError: If value is invalid
        """
        pass

    # =============================================================================
    # Copy/Move Operations (v1alpha only)
    # =============================================================================

    @abstractmethod
    def copy(self, source_resource_address: str, destination_resource_address: str) -> str:
        """Copy a resource to a new address.

        Creates a new version at the destination address.

        Args:
            source_resource_address: Source resource address
            destination_resource_address: Destination resource address

        Returns:
            Resource identity of the newly created version at destination

        Raises:
            FileNotFoundError: If source doesn't exist
            ValueError: If source is a folder
            PermissionError: If read/write permission is denied
        """
        pass

    @abstractmethod
    def move(self, source_resource_address: str, destination_resource_address: str) -> str:
        """Move/rename a resource to a new address.

        Creates a new version at the destination and removes the source.

        Args:
            source_resource_address: Source resource address
            destination_resource_address: Destination resource address

        Returns:
            Resource identity of the newly created version at destination

        Raises:
            FileNotFoundError: If source doesn't exist
            ValueError: If source is a folder
            PermissionError: If read/write/delete permission is denied
        """
        pass

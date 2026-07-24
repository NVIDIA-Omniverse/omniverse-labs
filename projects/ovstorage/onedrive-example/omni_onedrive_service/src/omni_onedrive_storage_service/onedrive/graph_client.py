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

"""Microsoft Graph API client wrapper for OneDrive operations.

Provides a high-level interface for OneDrive/SharePoint file operations
via the Microsoft Graph API, with automatic retry logic for rate limiting.
"""

import logging
import time
from collections.abc import Generator
from urllib.parse import quote

import requests
from pydantic import ValidationError
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
)

from .auth import get_authorization_header
from .exceptions import GraphApiError, ItemNotFoundError, RateLimitError, VersionNotFoundError
from .models import (
    DriveItem,
    DriveItemCollection,
    DriveItemVersion,
    DriveItemVersionCollection,
    GraphErrorResponse,
    UploadSession,
)
from .request_context import request_token

logger = logging.getLogger(__name__)


# Chunk size for streaming downloads (1 MB)
DOWNLOAD_CHUNK_SIZE = 1024 * 1024

# Threshold for large file uploads (4 MB)
LARGE_FILE_THRESHOLD = 4 * 1024 * 1024

# Upper bound on how long a single throttled request will wait before retrying,
# so one 429 can't hang a client request indefinitely. Mirrors the manual
# retry loops in download_content / download_version_content.
RATE_LIMIT_MAX_WAIT_SECONDS = 60


def _wait_for_rate_limit(retry_state: RetryCallState) -> float:
    """Tenacity wait strategy for 429s.

    Honors the Graph ``Retry-After`` hint carried by ``RateLimitError``
    (capped at ``RATE_LIMIT_MAX_WAIT_SECONDS``); falls back to exponential
    backoff if the value is unavailable. Retrying without honoring the hint
    just replays inside the throttling window and burns the retry budget.
    """
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, RateLimitError):
        # Clamp to a non-negative value: a malformed/negative Retry-After would
        # otherwise produce a negative sleep duration (ValueError in time.sleep).
        safe_retry_after = max(0, exc.retry_after)
        return min(safe_retry_after, RATE_LIMIT_MAX_WAIT_SECONDS)
    return wait_exponential(multiplier=1, min=4, max=RATE_LIMIT_MAX_WAIT_SECONDS)(retry_state)


class GraphClient:
    """Client for Microsoft Graph API operations on OneDrive/SharePoint.

    Wraps the Graph API with automatic token management, retry logic for
    rate limiting, and proper error handling.
    """

    BASE_URL = "https://graph.microsoft.com/v1.0"

    def __init__(self):
        """Initialize the Graph client."""
        self._session = requests.Session()

    def _get_headers(self, additional: dict[str, str] | None = None) -> dict[str, str]:
        """Get request headers with authorization from the current request context."""
        headers = {
            **get_authorization_header(request_token.get()),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if additional:
            headers.update(additional)
        return headers

    def _handle_response(self, response: requests.Response, url: str | None = None) -> requests.Response:
        """Handle Graph API response, raising appropriate exceptions."""
        if response.ok:
            return response

        # Handle rate limiting
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            raise RateLimitError(retry_after)

        # Parse error response
        try:
            error_data = response.json()
            error_response = GraphErrorResponse.model_validate(error_data)
            error_code = error_response.error.code
            error_message = error_response.error.message
        except (ValueError, ValidationError):
            error_code = "unknown"
            error_message = response.text or "Unknown error"

        # Include URL context in error message for debugging
        url_context = f" (URL: {url})" if url else ""
        logger.error(f"Graph API error: {response.status_code} {error_code} - {error_message}{url_context}")

        # Map to specific exceptions
        if response.status_code == 404 or error_code == "itemNotFound":
            raise ItemNotFoundError(f"{error_message}{url_context}")
        elif response.status_code == 412 or error_code == "resourceModified":
            # ETag mismatch / concurrent write conflict - raise as a retryable conflict
            from omni_onedrive_storage_service.backends.storage_backend_interface import EtagMismatchError

            raise EtagMismatchError(
                key=url or "",
                expected_etag="(request)",
                actual_etag="(server)",
            )
        elif response.status_code == 403 or error_code == "accessDenied":
            raise PermissionError(f"{error_message}{url_context}")
        elif response.status_code == 409 or error_code == "nameAlreadyExists":
            raise FileExistsError(f"{error_message}{url_context}")
        else:
            raise GraphApiError(error_message, response.status_code, error_code)  # ty: ignore[invalid-argument-type]

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=_wait_for_rate_limit,
        # Bound both the attempt count and total time so an interactive call
        # can't wait unboundedly across repeated throttling windows.
        stop=stop_after_attempt(5) | stop_after_delay(120),
        # Surface the underlying RateLimitError (not tenacity's RetryError) so
        # callers/logs see the real cause, matching the download retry loops.
        reraise=True,
    )
    def _request(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> requests.Response:
        """Make an HTTP request with retry logic for rate limiting."""
        if "headers" not in kwargs:
            kwargs["headers"] = self._get_headers()
        else:
            kwargs["headers"] = self._get_headers(kwargs["headers"])

        logger.debug(f"Graph API request: {method} {url}")
        response = self._session.request(method, url, **kwargs)
        return self._handle_response(response, url)

    # =========================================================================
    # Drive Operations
    # =========================================================================

    def get_my_drive(self) -> str:
        """Get the current user's default OneDrive drive ID.

        Uses the /me/drive endpoint which resolves to the authenticated
        user's OneDrive for Business.

        Returns:
            The drive ID string

        Raises:
            PermissionError: If the token lacks OneDrive access
        """
        url = f"{self.BASE_URL}/me/drive"
        response = self._request("GET", url)
        data = response.json()
        drive_id = data.get("id")
        if not drive_id:
            raise ValueError("No drive ID returned from /me/drive")
        logger.info(f"Resolved user drive: {data.get('name', 'Unknown')} ({drive_id})")
        return drive_id

    # =========================================================================
    # Item Operations
    # =========================================================================

    def get_item_by_path(self, drive_id: str, path: str) -> DriveItem:
        """Get a drive item by its path.

        Args:
            drive_id: The drive ID
            path: Path relative to drive root (e.g., "Documents/file.txt")

        Returns:
            DriveItem representing the file or folder

        Raises:
            ItemNotFoundError: If item doesn't exist
        """
        # Handle root path
        if not path or path == "/":
            url = f"{self.BASE_URL}/drives/{drive_id}/root"
        else:
            # Encode path components
            encoded_path = quote(path.lstrip("/"), safe="/")
            url = f"{self.BASE_URL}/drives/{drive_id}/root:/{encoded_path}"

        response = self._request("GET", url)
        return DriveItem.model_validate(response.json())

    def get_item_by_id(self, drive_id: str, item_id: str) -> DriveItem:
        """Get a drive item by its ID.

        Args:
            drive_id: The drive ID
            item_id: The item ID

        Returns:
            DriveItem representing the file or folder
        """
        url = f"{self.BASE_URL}/drives/{drive_id}/items/{item_id}"
        response = self._request("GET", url)
        return DriveItem.model_validate(response.json())

    def item_exists(self, drive_id: str, path: str) -> bool:
        """Check if an item exists at the given path.

        Args:
            drive_id: The drive ID
            path: Path relative to drive root

        Returns:
            True if item exists, False otherwise
        """
        try:
            self.get_item_by_path(drive_id, path)
            return True
        except ItemNotFoundError:
            return False

    # =========================================================================
    # Content Operations
    # =========================================================================

    def download_content(self, drive_id: str, item_id: str) -> Generator[bytes, None, None]:
        """Download file content as a stream.

        Args:
            drive_id: The drive ID
            item_id: The item ID

        Yields:
            Chunks of file content
        """
        url = f"{self.BASE_URL}/drives/{drive_id}/items/{item_id}/content"
        headers = self._get_headers({"Accept": "*/*"})

        max_retries = 5
        for attempt in range(max_retries + 1):
            try:
                with self._session.get(url, headers=headers, stream=True) as response:
                    self._handle_response(response)
                    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            yield chunk
                    return  # Success, exit generator
            except RateLimitError as e:
                if attempt < max_retries:
                    wait_time = min(e.retry_after, 60)
                    logger.warning(
                        f"Rate limited downloading content, "
                        f"retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    raise

    def download_version_content(self, drive_id: str, item_id: str, version_id: str) -> Generator[bytes, None, None]:
        """Download content of a specific version.

        Args:
            drive_id: The drive ID
            item_id: The item ID
            version_id: The version ID

        Yields:
            Chunks of file content

        Note:
            Graph API doesn't allow downloading the current version via the
            versions endpoint. If that error occurs, we fall back to the
            direct content endpoint.
        """
        url = f"{self.BASE_URL}/drives/{drive_id}/items/{item_id}/versions/{version_id}/content"
        headers = self._get_headers({"Accept": "*/*"})

        max_retries = 5
        for attempt in range(max_retries + 1):
            try:
                with self._session.get(url, headers=headers, stream=True) as response:
                    try:
                        self._handle_response(response)
                    except ItemNotFoundError:
                        raise VersionNotFoundError(item_id, version_id) from None
                    except GraphApiError as e:
                        # Graph API returns 400 "invalidRequest" when trying to get
                        # the current version's content via versions endpoint.
                        # Fall back to direct content download.
                        if "current version" in str(e).lower():
                            logger.debug(f"Version {version_id} is current, using direct download")
                            yield from self.download_content(drive_id, item_id)
                            return
                        raise
                    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            yield chunk
                    return  # Success, exit generator
            except RateLimitError as e:
                if attempt < max_retries:
                    wait_time = min(e.retry_after, 60)
                    logger.warning(
                        f"Rate limited downloading version content, "
                        f"retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    raise

    def upload_content(
        self,
        drive_id: str,
        parent_path: str,
        filename: str,
        content: bytes,
        if_match: str | None = None,
    ) -> DriveItem:
        """Upload file content (for files under 4MB).

        Args:
            drive_id: The drive ID
            parent_path: Path to parent folder
            filename: Name of the file
            content: File content bytes
            if_match: Optional ETag for conditional update

        Returns:
            DriveItem representing the uploaded file
        """
        if parent_path and parent_path != "/":
            encoded_path = quote(parent_path.lstrip("/"), safe="/")
            url = f"{self.BASE_URL}/drives/{drive_id}/root:/{encoded_path}/{quote(filename)}:/content"
        else:
            url = f"{self.BASE_URL}/drives/{drive_id}/root:/{quote(filename)}:/content"

        headers: dict[str, str] = {"Content-Type": "application/octet-stream"}
        if if_match:
            headers["If-Match"] = if_match

        response = self._request("PUT", url, data=content, headers=headers)
        return DriveItem.model_validate(response.json())

    def create_upload_session(
        self,
        drive_id: str,
        parent_path: str,
        filename: str,
    ) -> UploadSession:
        """Create an upload session for large file uploads.

        Args:
            drive_id: The drive ID
            parent_path: Path to parent folder
            filename: Name of the file

        Returns:
            UploadSession with upload URL
        """
        if parent_path and parent_path != "/":
            encoded_path = quote(parent_path.lstrip("/"), safe="/")
            url = f"{self.BASE_URL}/drives/{drive_id}/root:/{encoded_path}/{quote(filename)}:/createUploadSession"
        else:
            url = f"{self.BASE_URL}/drives/{drive_id}/root:/{quote(filename)}:/createUploadSession"

        body = {
            "item": {
                "@microsoft.graph.conflictBehavior": "replace",
                "name": filename,
            }
        }

        response = self._request("POST", url, json=body)
        return UploadSession.model_validate(response.json())

    def upload_chunk(
        self,
        upload_url: str,
        chunk: bytes,
        start_byte: int,
        end_byte: int,
        total_size: int,
    ) -> DriveItem | None:
        """Upload a chunk to an upload session.

        Args:
            upload_url: The upload URL from the session
            chunk: The chunk data
            start_byte: Start byte position (0-indexed)
            end_byte: End byte position (inclusive)
            total_size: Total file size

        Returns:
            DriveItem if upload is complete, None if more chunks needed
        """
        headers = {
            "Content-Length": str(len(chunk)),
            "Content-Range": f"bytes {start_byte}-{end_byte}/{total_size}",
        }

        # Upload URL doesn't need auth header - it has a token embedded
        response = self._session.put(upload_url, data=chunk, headers=headers)
        self._handle_response(response)

        # If status is 201 or 200, upload is complete
        if response.status_code in (200, 201):
            return DriveItem.model_validate(response.json())
        return None

    # =========================================================================
    # Folder Operations
    # =========================================================================

    def list_children(
        self,
        drive_id: str,
        item_id: str | None = None,
        path: str | None = None,
    ) -> DriveItemCollection:
        """List children of a folder.

        Args:
            drive_id: The drive ID
            item_id: The folder item ID (optional)
            path: The folder path (optional, used if item_id not provided)

        Returns:
            DriveItemCollection with children
        """
        if item_id:
            url = f"{self.BASE_URL}/drives/{drive_id}/items/{item_id}/children"
        elif path and path != "/":
            encoded_path = quote(path.lstrip("/"), safe="/")
            url = f"{self.BASE_URL}/drives/{drive_id}/root:/{encoded_path}:/children"
        else:
            url = f"{self.BASE_URL}/drives/{drive_id}/root/children"

        response = self._request("GET", url)
        return DriveItemCollection.model_validate(response.json())

    def list_children_next_page(self, next_link: str) -> DriveItemCollection:
        """Get next page of children listing.

        Args:
            next_link: The @odata.nextLink URL

        Returns:
            DriveItemCollection with next page of children
        """
        response = self._request("GET", next_link)
        return DriveItemCollection.model_validate(response.json())

    def create_folder(self, drive_id: str, parent_path: str, folder_name: str) -> DriveItem:
        """Create a folder.

        Args:
            drive_id: The drive ID
            parent_path: Path to parent folder
            folder_name: Name of the new folder

        Returns:
            DriveItem representing the created folder
        """
        if parent_path and parent_path != "/":
            encoded_path = quote(parent_path.lstrip("/"), safe="/")
            url = f"{self.BASE_URL}/drives/{drive_id}/root:/{encoded_path}:/children"
        else:
            url = f"{self.BASE_URL}/drives/{drive_id}/root/children"

        body = {
            "name": folder_name,
            "folder": {},
            "@microsoft.graph.conflictBehavior": "fail",
        }

        response = self._request("POST", url, json=body)
        return DriveItem.model_validate(response.json())

    # =========================================================================
    # Delete Operations
    # =========================================================================

    def delete_item(self, drive_id: str, item_id: str, if_match: str | None = None) -> None:
        """Delete an item (moves to recycle bin).

        Args:
            drive_id: The drive ID
            item_id: The item ID
            if_match: Optional ETag for conditional delete
        """
        url = f"{self.BASE_URL}/drives/{drive_id}/items/{item_id}"
        headers: dict[str, str] = {}
        if if_match:
            headers["If-Match"] = if_match

        self._request("DELETE", url, headers=headers)

    # =========================================================================
    # Version Operations
    # =========================================================================

    def list_versions(self, drive_id: str, item_id: str) -> DriveItemVersionCollection:
        """List versions of an item.

        Args:
            drive_id: The drive ID
            item_id: The item ID

        Returns:
            DriveItemVersionCollection with versions
        """
        url = f"{self.BASE_URL}/drives/{drive_id}/items/{item_id}/versions"
        response = self._request("GET", url)
        return DriveItemVersionCollection.model_validate(response.json())

    def get_version(self, drive_id: str, item_id: str, version_id: str) -> DriveItemVersion:
        """Get a specific version of an item.

        Args:
            drive_id: The drive ID
            item_id: The item ID
            version_id: The version ID

        Returns:
            DriveItemVersion
        """
        url = f"{self.BASE_URL}/drives/{drive_id}/items/{item_id}/versions/{version_id}"
        try:
            response = self._request("GET", url)
            return DriveItemVersion.model_validate(response.json())
        except ItemNotFoundError:
            raise VersionNotFoundError(item_id, version_id) from None

    def get_item_with_download_url(self, drive_id: str, item_id: str) -> DriveItem:
        """Get item including the @microsoft.graph.downloadUrl field.

        The download URL is a pre-authenticated URL that can be used to
        download the file content directly without additional authorization.
        It expires after approximately 1 hour.

        Args:
            drive_id: The drive ID
            item_id: The item ID

        Returns:
            DriveItem with download_url field populated
        """
        url = f"{self.BASE_URL}/drives/{drive_id}/items/{item_id}"
        response = self._request("GET", url)
        return DriveItem.model_validate(response.json())

    def get_version_download_url(self, drive_id: str, item_id: str, version_id: str) -> str:
        """Get download URL for a specific version.

        Graph API redirects /versions/{id}/content to the actual download URL.
        We follow the redirect to get the final URL.

        Args:
            drive_id: The drive ID
            item_id: The item ID
            version_id: The version ID

        Returns:
            Pre-authenticated download URL for the version

        Raises:
            VersionNotFoundError: If the version doesn't exist
            ValueError: If no redirect URL is returned

        Note:
            Graph API doesn't allow getting the current version via the versions
            endpoint. If that error occurs, we fall back to the item's download URL.
        """
        url = f"{self.BASE_URL}/drives/{drive_id}/items/{item_id}/versions/{version_id}/content"
        headers = self._get_headers()

        # Don't follow redirect - we want the redirect URL
        response = self._session.get(url, headers=headers, allow_redirects=False)

        if response.status_code in (301, 302, 307, 308):
            location = response.headers.get("Location")
            if location:
                return location
            raise ValueError("Redirect response missing Location header")

        if response.status_code == 404:
            raise VersionNotFoundError(item_id, version_id)

        # Check for "current version" error (400 invalidRequest)
        if response.status_code == 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", "")
                if "current version" in error_message.lower():
                    # Fall back to item's download URL
                    logger.debug(f"Version {version_id} is current, using item download URL")
                    item = self.get_item_with_download_url(drive_id, item_id)
                    if item.download_url:
                        return item.download_url
                    raise ValueError(f"No download URL available for item: {item_id}")
            except (ValueError, KeyError):
                pass

        # If no redirect, handle as error
        self._handle_response(response)
        raise ValueError(f"Expected redirect for version content, got {response.status_code}")

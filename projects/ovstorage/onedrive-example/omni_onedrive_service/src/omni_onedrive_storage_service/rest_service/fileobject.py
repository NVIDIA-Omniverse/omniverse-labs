# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""FileObjectService implementation for REST.

This module implements the FileObjectService from the Storage API, which
provides core file operations:
- Read: Download file content by resource identity
- ReadFromAddress: Download file content by resource address
- Write: Upload file content (body, redirect, or multipart)
- Delete: Remove a file
- Stat: Get file metadata
- Enumerate: Recursively list files in a directory tree
- Copy: Copy a file (v1alpha only)
- Move: Move/rename a file (v1alpha only)

The service supports multiple upload methods:
- Body: Small files uploaded directly in the HTTP request body
- Redirect: Medium files uploaded via HTTP redirect
- Multipart: Large files uploaded in multiple parts

Version management is automatic - each write creates a new immutable version.
"""

import asyncio
import urllib.parse
import uuid
from typing import (
    Annotated,
    Any,
)

from fastapi import (
    HTTPException,
    Request,
    Response,
)
from pydantic import WithJsonSchema
from starlette import status
from starlette.responses import (
    JSONResponse,
    StreamingResponse,
)

from omni_onedrive_storage_service.backends import (
    MAX_UPLOAD_SIZE_BYTES,
    REDIRECT_HOST,
    REDIRECT_PORT,
    get_backend,
)
from omni_onedrive_storage_service.backends.storage_backend_interface import FolderMode

from .rest_messages import (
    DOWNLOAD_PREFERENCE_SCHEMA,
    MAX_PAGE_SIZE_SCHEMA,
    PAGE_HANDLE_TYPE,
    RESOURCE_ADDRESS_TYPE,
    RESOURCE_IDENTITY_TYPE,
    UPLOAD_PREFERENCE_SCHEMA,
    AddressInfo,
    CompleteMultipartUploadRequest,
    CompleteUploadRequest,
    CopyRequest,
    CopyResponse,
    EnumerateResponse,
    HTTPHeader,
    HTTPValidationError,
    Metadata,
    MoveRequest,
    MoveResponse,
    MultipartUploadAbortRequest,
    MultipartUploadProperties,
    MultipartUploadRequest,
    MultipartUploadResponse,
    OptimisticLockingSupportResponse,
    ReadResponse,
    ResourceInfo,
    UploadOptionsResponse,
    UploadPartResponse,
    UploadPreference,
    WriteRedirectProperties,
    WriteRedirectResponse,
    WriteTypeInterval,
)
from .routes import (
    fileobject_service_alpha,
    fileobject_service_beta,
)

# =============================================================================
# Constants
# =============================================================================

# HTTP Content-Type constants
CONTENT_TYPE_APPLICATION_JSON = "application/json"
CONTENT_TYPE_APPLICATION_OCTET_STREAM = "application/octet-stream"

# Custom HTTP headers used by the Storage API
# These headers carry metadata about file objects in responses
OV_STORAGE_METADATA_HEADER = "x-nvidia-omniverse-storage-metadata"
OV_STORAGE_RESOURCE_IDENTITY_HEADER = "x-nvidia-omniverse-storage-resource-identity"

_METADATA_HEADERS_SPEC = {
    OV_STORAGE_RESOURCE_IDENTITY_HEADER: {
        "required": True,
        "description": "The identity of the data object.",
        "schema": {
            "type": "string",
        },
    },
    OV_STORAGE_METADATA_HEADER: {
        "required": True,
        "description": "Metadata for the data object.",
        "content": {
            CONTENT_TYPE_APPLICATION_JSON: {
                "schema": Metadata.model_json_schema(),
            },
        },
    },
}


enumerate_root_route_params: dict[str, Any] = {
    "path": "/data-objects",
    "responses": {
        status.HTTP_200_OK: {
            "description": "Success response.",
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "content": {"application/json": HTTPValidationError.model_json_schema()}
        },
    },
    "include_in_schema": False,
    "methods": ["GET"],
}


async def enumerate_root() -> EnumerateResponse:
    """Enumerate all data objects starting from the root.

    Special case of enumerate_() for the root directory, required because
    path parameters don't match empty strings in FastAPI routing.

    Returns:
        EnumerateResponse with recursive listing of all files

    See Also:
        enumerate_() for the main implementation
    """
    return await enumerate_("")


enumerate_route_params: dict[str, Any] = {
    "path": "/data-objects/{resource_address:path}",
    "responses": {
        status.HTTP_200_OK: {
            "description": "Success response.",
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "content": {"application/json": HTTPValidationError.model_json_schema()}
        },
    },
    "methods": ["GET"],
}


async def enumerate_(
    resource_address: RESOURCE_ADDRESS_TYPE,
    max_page_size: Annotated[int | None, WithJsonSchema(MAX_PAGE_SIZE_SCHEMA)] = 1000,
    continuation_handle: PAGE_HANDLE_TYPE = None,
) -> EnumerateResponse:
    """Recursively enumerate all data objects under a resource address.

    Unlike list() which returns immediate children, enumerate() recursively
    traverses the entire directory tree and returns all data objects (files)
    found under the specified address. Folders are not included in results.

    Args:
        resource_address: Root address to enumerate from
        max_page_size: Maximum items per page (default 1000)
        continuation_handle: Token from previous page

    Returns:
        EnumerateResponse containing:
        - items: List of AddressInfo objects with address and metadata
        - next_continuation_handle: Token for next page, or None if complete

    Raises:
        HTTPException(400): If enumerating a versioned address
        HTTPException(404): If address doesn't exist or points to a file
        HTTPException(500): On internal errors

    Note:
        This is a recursive operation that can be expensive for large directory
        trees. Consider using list() or liststat() if you only need immediate
        children. Directories are silently skipped during enumeration.

    Example:
        Enumerating 'folder1' with subfolders returns all files:
        - folder1/file1.txt
        - folder1/subfolder1/file2.txt
        - folder1/subfolder1/subfolder2/file3.txt
    """
    # Run blocking backend calls in thread pool to avoid blocking event loop
    backend = get_backend()
    if await asyncio.to_thread(backend.is_version_address, resource_address):
        raise HTTPException(status_code=400, detail="Cannot enumerate versioned addresses")
    if not await asyncio.to_thread(backend.exists, resource_address):
        raise HTTPException(status_code=404, detail="Resource address not found")
    if await asyncio.to_thread(backend.is_file, resource_address):
        raise HTTPException(status_code=404, detail="Cannot enumerate file addresses")

    start_index = int(continuation_handle) if continuation_handle else 0
    page_size = max_page_size if max_page_size is not None else 1000

    # Request one extra item to accurately detect if more pages exist
    # Run enumerate in thread pool since it's a generator of blocking operations
    def _enumerate():
        entries = []
        for entry_batch in backend.enumerate(resource_address, start_index=start_index, limit=page_size + 1):
            for item in entry_batch:
                if item.metadata is None:
                    continue
                metadata = Metadata(
                    data_object_size=item.metadata.data_object_size,
                    last_modified_timestamp=item.metadata.last_modified_timestamp.isoformat(),
                )
                entries.append(AddressInfo(resource_address=item.resource_address, metadata=metadata))
        return entries

    entries = await asyncio.to_thread(_enumerate)

    # Check if there are more items beyond this page
    has_more = len(entries) > page_size
    if has_more:
        entries = entries[:page_size]  # Trim to page_size

    next_handle = str(start_index + page_size) if has_more else None
    return EnumerateResponse(items=entries, next_continuation_handle=next_handle)


stat_route_params: dict[str, Any] = {
    "path": "/by-address/{resource_address:path}",
    "status_code": status.HTTP_204_NO_CONTENT,
    "responses": {
        status.HTTP_204_NO_CONTENT: {
            "description": "Success response.",
            "headers": _METADATA_HEADERS_SPEC,
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "content": {"application/json": HTTPValidationError.model_json_schema()}
        },
    },
    "methods": ["HEAD"],
}


async def stat_(resource_address: RESOURCE_ADDRESS_TYPE) -> Response:
    """Get metadata for a data object without downloading it.

    Returns metadata (size, modification time, resource identity) for a file
    in response headers without transferring the file content. Similar to
    HTTP HEAD request.

    Args:
        resource_address: Address of the file to stat

    Returns:
        Response with status 204 (No Content) and headers:
        - x-nvidia-omniverse-storage-metadata: JSON-encoded Metadata object
        - x-nvidia-omniverse-storage-resource-identity: Unique version identifier

    Raises:
        HTTPException(400): If resource address is invalid
        HTTPException(404): If file not found or is a directory
        HTTPException(403): If permission denied

    Note:
        This is more efficient than GET when you only need metadata.
        Use this to check if a file exists and get its current version.
    """
    backend = get_backend()
    if not await asyncio.to_thread(backend.is_address_valid, resource_address):
        raise HTTPException(status_code=400, detail="Invalid resource address")
    try:
        if not await asyncio.to_thread(backend.exists, resource_address):
            raise HTTPException(status_code=404, detail="Resource address not found")
        info = await asyncio.to_thread(backend.stat, resource_address)
        metadata = Metadata(
            data_object_size=info.metadata.data_object_size,
            last_modified_timestamp=info.metadata.last_modified_timestamp.isoformat(),
        )
        identity = await asyncio.to_thread(backend.create_identity_from_resource_address, resource_address)
        return JSONResponse(
            status_code=204,
            content=None,
            headers={
                "x-nvidia-omniverse-storage-metadata": metadata.model_dump_json(),
                "x-nvidia-omniverse-storage-resource-identity": identity,
            },
        )
    except (FileNotFoundError, IsADirectoryError) as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from None


read_from_address_route_params: dict[str, Any] = {
    "path": "/by-address/{resource_address:path}",
    "responses": {
        status.HTTP_200_OK: {
            "description": "Successful response containing data object content.",
            "headers": _METADATA_HEADERS_SPEC,
            "content": {
                CONTENT_TYPE_APPLICATION_OCTET_STREAM: {},
            },
        },
        status.HTTP_300_MULTIPLE_CHOICES: {
            "description": "Redirect to a HTTP download location.",
            "content": {
                CONTENT_TYPE_APPLICATION_JSON: {
                    "schema": ReadResponse.model_json_schema(),
                },
            },
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "content": {"application/json": HTTPValidationError.model_json_schema()}
        },
    },
    "response_class": StreamingResponse,  # The default response (200) is an application/octet-stream, not json.
    "methods": ["GET"],
}


async def read_from_address(
    resource_address: RESOURCE_ADDRESS_TYPE,
    download_preference: Annotated[str | None, WithJsonSchema(DOWNLOAD_PREFERENCE_SCHEMA)] = None,
):
    """Download a data object by its resource address.

    Downloads the current (latest) version of a file at the given address.
    Supports two download methods:
    - body: File content in response body (HTTP 200)
    - redirect: HTTP 300 redirect to a different download URL

    Args:
        resource_address: Address of the file to download
        download_preference: Preferred download method (\"body\" or \"redirect\")
                           If None, defaults to \"body\"

    Returns:
        - If preference is \"body\": StreamingResponse with file content
        - If preference is \"redirect\": JSONResponse (300) with ReadResponse

    Response Headers (for body mode):
        - x-nvidia-omniverse-storage-metadata: File metadata (size, mtime)
        - x-nvidia-omniverse-storage-resource-identity: Version identifier

    Raises:
        HTTPException(400): If address invalid, preference invalid, or is a directory
        HTTPException(403): If permission denied
        HTTPException(404): If file not found

    Note:
        To download a specific version, use read() with resource_identity instead.
        This always returns the current/latest version at the address.
    """
    backend = get_backend()
    if not await asyncio.to_thread(backend.is_address_valid, resource_address):
        raise HTTPException(status_code=400, detail="Invalid resource address")

    try:
        if await asyncio.to_thread(backend.is_dir, resource_address):
            error_code_to_use = 404 if backend.folder_mode() == FolderMode.HYBRID else 400
            raise HTTPException(
                status_code=error_code_to_use,
                detail="Resource address is folder, can't be read. No object found at that address",
            )
        info = await asyncio.to_thread(backend.stat, resource_address)
        metadata = Metadata(
            data_object_size=info.metadata.data_object_size,
            last_modified_timestamp=info.metadata.last_modified_timestamp.isoformat(),
        )
        await asyncio.to_thread(backend.check_read_permission_on_address, resource_address)
        if download_preference == "body" or download_preference is None:
            identity = await asyncio.to_thread(backend.create_identity_from_resource_address, resource_address)
            headers = {
                "x-nvidia-omniverse-storage-metadata": metadata.model_dump_json(),
                "x-nvidia-omniverse-storage-resource-identity": identity,
            }
            return StreamingResponse(
                backend.read_from_address(resource_address),
                media_type="application/octet-stream",
                headers=headers,
            )
        elif download_preference == "redirect":
            # Try to use redirect if backend supports it
            try:
                url = await asyncio.to_thread(
                    backend.construct_redirect_url, resource_address, REDIRECT_HOST, REDIRECT_PORT
                )
                redirect_data = ReadResponse(redirect_target_url=url, additional_headers={})
                return JSONResponse(content=redirect_data.model_dump(), status_code=300)
            except (NotImplementedError, AttributeError):
                # Backend doesn't support redirects (e.g., Nucleus), fall back to body download
                identity = await asyncio.to_thread(backend.create_identity_from_resource_address, resource_address)
                headers = {
                    "x-nvidia-omniverse-storage-metadata": metadata.model_dump_json(),
                    "x-nvidia-omniverse-storage-resource-identity": identity,
                }
                return StreamingResponse(
                    backend.read_from_address(resource_address),
                    media_type="application/octet-stream",
                    headers=headers,
                )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Download preference invalid: {download_preference}",
            )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except (PermissionError, IsADirectoryError) as e:
        raise HTTPException(status_code=403, detail=str(e)) from None


read_route_params: dict[str, Any] = {
    "path": "/by-identity/{resource_identity:path}",
    "responses": {
        status.HTTP_200_OK: {
            "description": "Successful response containing data object content.",
            "headers": _METADATA_HEADERS_SPEC,
            "content": {
                CONTENT_TYPE_APPLICATION_OCTET_STREAM: {},
            },
        },
        status.HTTP_300_MULTIPLE_CHOICES: {
            "description": "Redirect to a HTTP download location.",
            "content": {
                CONTENT_TYPE_APPLICATION_JSON: {
                    "schema": ReadResponse.model_json_schema(),
                },
            },
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "content": {"application/json": HTTPValidationError.model_json_schema()}
        },
    },
    "response_class": StreamingResponse,  # The default response (200) is an application/octet-stream, not json.
    "methods": ["GET"],
}


async def read(
    resource_identity: RESOURCE_IDENTITY_TYPE,
    download_preference: Annotated[str | None, WithJsonSchema(DOWNLOAD_PREFERENCE_SCHEMA)] = None,
):
    """Download a specific version of a data object by its resource identity.

    Downloads a specific, immutable version of a file using its resource identity
    (version identifier). Unlike read_from_address() which gets the latest version,
    this always returns the exact version specified.

    Args:
        resource_identity: Unique identifier for a specific file version
        download_preference: Download method (\"body\" or \"redirect\", default \"body\")

    Returns:
        - If \"body\": StreamingResponse with file content (HTTP 200)
        - If \"redirect\": JSONResponse with redirect URL (HTTP 300)

    Response Headers (for body mode):
        - x-nvidia-omniverse-storage-metadata: File metadata
        - x-nvidia-omniverse-storage-resource-identity: Same as input identity

    Raises:
        HTTPException(400): If identity is invalid or malformed
        HTTPException(403): If permission denied or is a directory
        HTTPException(404): If version not found

    Note:
        Resource identities are immutable - they always point to the same content.
        This makes them suitable for caching and content-addressed storage.
    """
    backend = get_backend()
    decoded_identity = urllib.parse.unquote_plus(resource_identity)

    # Use the backend's stat_identity to validate and get metadata
    try:
        info = await asyncio.to_thread(backend.stat_identity, decoded_identity)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except (PermissionError, IsADirectoryError) as e:
        raise HTTPException(status_code=403, detail=str(e)) from None

    metadata = Metadata(
        data_object_size=info.metadata.data_object_size,
        last_modified_timestamp=info.metadata.last_modified_timestamp.isoformat(),
    )

    if download_preference == "body" or download_preference is None:
        # Use backend's read_from_identity method
        try:
            content_generator = await asyncio.to_thread(lambda: list(backend.read_from_identity(decoded_identity)))
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(status_code=404, detail=str(e)) from None
        except (PermissionError, IsADirectoryError) as e:
            raise HTTPException(status_code=403, detail=str(e)) from None

        headers = {
            "x-nvidia-omniverse-storage-metadata": metadata.model_dump_json(),
            "x-nvidia-omniverse-storage-resource-identity": resource_identity,
        }

        async def generate():
            for chunk in content_generator:
                yield chunk

        return StreamingResponse(
            generate(),
            media_type="application/octet-stream",
            headers=headers,
        )
    elif download_preference == "redirect":
        # Check if backend supports redirect downloads
        if backend.supports_redirect_download():
            try:
                url = backend.construct_redirect_url_for_identity(decoded_identity, REDIRECT_HOST, REDIRECT_PORT)
                redirect_data = ReadResponse(redirect_target_url=url, additional_headers={})
                return JSONResponse(content=redirect_data.model_dump(), status_code=300)
            except (ValueError, OSError, NotImplementedError) as e:
                raise HTTPException(status_code=500, detail=f"Failed to construct redirect URL: {e}") from None
        else:
            # Backend doesn't support redirects, fall back to body download
            try:
                content_generator = await asyncio.to_thread(lambda: list(backend.read_from_identity(decoded_identity)))
            except (ValueError, FileNotFoundError) as e:
                raise HTTPException(status_code=404, detail=str(e)) from None
            except (PermissionError, IsADirectoryError) as e:
                raise HTTPException(status_code=403, detail=str(e)) from None

            headers = {
                "x-nvidia-omniverse-storage-metadata": metadata.model_dump_json(),
                "x-nvidia-omniverse-storage-resource-identity": resource_identity,
            }

            async def generate():
                for chunk in content_generator:
                    yield chunk

            return StreamingResponse(
                generate(),
                media_type="application/octet-stream",
                headers=headers,
            )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Download preference invalid: {str(download_preference)}",
        )


get_upload_options_route_params: dict[str, Any] = {
    "path": "/upload-options/by-address/{resource_address:path}",
    "responses": {
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"content": {"application/json": HTTPValidationError.model_json_schema()}}
    },
    "methods": ["GET"],
}


async def get_upload_options(
    resource_address: RESOURCE_ADDRESS_TYPE,
) -> UploadOptionsResponse:
    """Get recommended upload methods based on file size.

    Returns recommended upload strategies for different file size ranges.
    Clients can use this to determine the most efficient upload method.

    For Nucleus backend: Only body upload is supported.
    For Filesystem backend: body, redirect, and multipart are supported.

    Args:
        resource_address: Destination address for the upload

    Returns:
        UploadOptionsResponse with write_type_intervals based on backend:

        Filesystem backend:
        - 0-1KB: Use \"body\" (direct upload in request body)
        - 1KB-1MB: Use \"redirect\" (upload to separate endpoint)
        - 1MB+: Use \"multipart\" (split into multiple parts)

        Nucleus backend:
        - All sizes: Use \"body\"

    Raises:
        HTTPException(400): If address is a versioned address

    Note:
        These are recommendations, not requirements. Clients can choose
        a different method if needed. The thresholds are tuned for this
        implementation and may differ in other storage services.
    """
    backend = get_backend()
    if await asyncio.to_thread(backend.is_version_address, resource_address):
        raise HTTPException(400, "Versioned resource addresses cannot be written to")

    # Check if backend supports redirects and multipart uploads
    supports_redirects = False
    try:
        # Try to check if construct_redirect_url is implemented
        await asyncio.to_thread(backend.construct_redirect_url, "test://test", "http://localhost", 8011)
        supports_redirects = True
    except (NotImplementedError, AttributeError, ValueError):
        # ValueError is raised if the test URL scheme doesn't match the backend's scheme
        supports_redirects = False

    if supports_redirects:
        # Filesystem backend - supports body, redirect, and multipart
        return UploadOptionsResponse(
            write_type_intervals=[
                WriteTypeInterval(
                    minimum_data_object_size=0,
                    maximum_data_object_size=1024,
                    preferred_upload_method=UploadPreference.body,
                ),
                WriteTypeInterval(
                    minimum_data_object_size=1024,
                    maximum_data_object_size=1024 * 1024,
                    preferred_upload_method=UploadPreference.redirect,
                ),
                WriteTypeInterval(
                    minimum_data_object_size=1024 * 1024,
                    maximum_data_object_size=pow(2, 53),
                    preferred_upload_method=UploadPreference.multipart,
                ),
            ]
        )
    else:
        # Nucleus or other backends - only support body upload
        return UploadOptionsResponse(
            write_type_intervals=[
                WriteTypeInterval(
                    minimum_data_object_size=0,
                    maximum_data_object_size=pow(2, 53),
                    preferred_upload_method=UploadPreference.body,
                ),
            ]
        )


def raise_on_previous_version_not_latest(resource_address: str, previous_version: str):
    """Validate that a specified version is still the latest version.

    This implements optimistic concurrency control for write operations. When
    a client specifies a previous_version (expected current state), this
    function verifies that version is still the latest. If not, it raises
    HTTPException with PRECONDITION_FAILED, preventing lost updates.

    Args:
        resource_address: Storage API resource address to check.
        previous_version: Encoded identity string of the expected latest version.

    Raises:
        HTTPException(412): If:
            - The specified version is not the latest version
            - No version exists at the resource address

    Note:
        This is used in Write, Delete, Copy, and Move operations to ensure
        the client's view of the resource state is current.
    """
    try:
        if not get_backend().is_version_latest(resource_address, previous_version):
            raise HTTPException(412, detail="specified previous version is not latest version")
    except FileNotFoundError:
        raise HTTPException(412, detail="specified previous version, but no version found") from None


def generate_upload_part_properties(upload_id: str, part_number: str, resource_address: str) -> WriteRedirectProperties:
    """Generate redirect properties for uploading a multipart part.

    Creates a WriteRedirectProperties object with the URL and headers needed
    for a client to upload one part of a multipart upload.

    Args:
        upload_id: Unique identifier for this multipart upload session.
        part_number: Sequential number of this part (zero-based).
        resource_address: Destination resource address.

    Returns:
        WriteRedirectProperties containing:
        - redirect_target_url: URL where client should POST the part data
        - method: \"post\"
        - additional_headers: Headers client must include
        - completion_header_names: Headers to collect from response

    Note:
        The completion_header_names (\"local-file\") are used later when
        completing the multipart upload to locate and assemble the parts.
    """
    backend = get_backend()
    redirect_props = backend.construct_upload_part_redirect(
        upload_id, int(part_number), resource_address, REDIRECT_HOST, REDIRECT_PORT
    )
    headers: list[HTTPHeader] = [HTTPHeader(name=h[0], value=h[1]) for h in redirect_props["additional_headers"]]
    return WriteRedirectProperties(
        method="post",
        redirect_target_url=redirect_props["redirect_target_url"],
        completion_header_names=redirect_props["completion_header_names"],
        additional_headers=headers,
    )


async def _read_body_limited(request: Request, max_bytes: int) -> bytes:
    """Read the request body into memory, bounded by ``max_bytes``.

    The body upload path buffers the whole object in memory before writing it to
    the backend, so stream it in chunks and abort as soon as the accumulated size
    exceeds the limit. This prevents an authenticated client from exhausting
    process memory with an oversized or size-misdeclared upload (DoS).
    """
    content_length = request.headers.get("content-length")
    if content_length is not None and content_length.isdigit() and int(content_length) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Upload exceeds maximum allowed {max_bytes} bytes",
        )

    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"Upload exceeds maximum allowed {max_bytes} bytes",
            )
    return bytes(body)


write_route_params: dict[str, Any] = {
    "path": "/by-address/{resource_address:path}",
    "status_code": 201,
    "response_model": ResourceInfo,
    "responses": {
        status.HTTP_201_CREATED: {
            "description": "Successful response containing data object content.",
        },
        status.HTTP_300_MULTIPLE_CHOICES: {
            "description": "Service wants client to redirect HTTP write operation somewhere else.",
            "content": {
                CONTENT_TYPE_APPLICATION_JSON: {},
            },
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "content": {"application/json": HTTPValidationError.model_json_schema()}
        },
    },
    "methods": ["PUT"],
}


async def write(
    resource_address: RESOURCE_ADDRESS_TYPE,
    request: Request,
    data_object_size: Annotated[int, WithJsonSchema({"type": "integer", "format": "int64"})],
    previous_version: Annotated[str, WithJsonSchema({"maxLength": 4096})] | None = None,
    upload_preference: Annotated[str | None, WithJsonSchema(UPLOAD_PREFERENCE_SCHEMA)] = None,
) -> JSONResponse:
    """Upload a new version of a data object.

    Creates a new version at the specified resource address. Supports three
    upload methods with different performance characteristics:

    Upload Methods:
        - \"body\": Upload content directly in request body (HTTP 201)
          Best for: Small files < 1KB

        - \"redirect\": Server returns redirect URL for upload (HTTP 300)
          Best for: Medium files 1KB-1MB
          Client uploads to redirect URL, then calls complete_redirect_upload()

        - \"multipart\": Server initiates multipart upload (HTTP 300)
          Best for: Large files > 1MB
          Client uploads in parts, then calls complete_multipart_upload()

    Args:
        resource_address: Destination address for the new version
        request: FastAPI request (body contains file data for \"body\" mode)
        data_object_size: Size of the data in bytes (required for all modes)
        previous_version: Expected current version for conditional write (optional)
                         If specified, write fails if this isn't the latest version
        upload_preference: Preferred method (\"body\", \"redirect\", \"multipart\")
                          If None, automatically chosen based on data_object_size

    Returns:
        - If \"body\": ResourceInfo with identity and metadata (HTTP 201)
        - If \"redirect\": WriteRedirectResponse with redirect URL (HTTP 300)
        - If \"multipart\": MultipartUploadResponse with upload_id (HTTP 300)

    Raises:
        HTTPException(400): If address invalid or versioned
        HTTPException(403): If permission denied
        HTTPException(412): If previous_version doesn't match current version

    Example (body mode):
        PUT /v1alpha/fileobject/by-address/myfile.txt?data_object_size=123
        Body: <file content>
        Response (201): {\"resource_identity\": \"...\", \"metadata\": {...}}

    Example (multipart mode):
        PUT /v1alpha/fileobject/by-address/largefile.bin?data_object_size=10485760&upload_preference=multipart
        Response (300): {\"multipart\": {\"upload_id\": \"...\", \"first_part_write_redirect\": {...}}}

    Note:
        Versioning is automatic - each write creates a new immutable version.
        Previous versions remain accessible via their resource_identity.
    """
    try:
        # Check if backend supports redirect/multipart uploads
        backend = get_backend()
        supports_redirects = backend.supports_redirect_upload()
        supports_multipart = backend.supports_multipart_upload()

        if upload_preference is None:
            if 0 <= data_object_size < 1024:
                # Small file, use chunk method
                upload_preference = "body"
            elif 1024 <= data_object_size < 1024 * 1024:
                # Medium file, use redirect method
                upload_preference = "redirect" if supports_redirects else "body"
            elif data_object_size >= 1024 * 1024:
                # Large file, upload this in parts
                upload_preference = "multipart" if supports_multipart else "body"
            else:
                raise HTTPException(status_code=400, detail="Invalid data size")

        # If client requested redirect/multipart but backend doesn't support it, use body instead
        if not supports_redirects and upload_preference == "redirect":
            upload_preference = "body"
        if not supports_multipart and upload_preference == "multipart":
            upload_preference = "body"

        if previous_version is not None:
            # Check if backend supports optimistic locking for write
            if not backend.get_optimistic_locking_support().write:
                raise HTTPException(
                    status_code=501,
                    detail=f"Write operation with previous_version parameter is not supported by {type(backend).__name__} backend.",
                )
            raise_on_previous_version_not_latest(resource_address, previous_version)

        if not await asyncio.to_thread(backend.is_address_valid, resource_address):
            raise HTTPException(status_code=400, detail=f"Invalid resource address: {resource_address}")

        if await asyncio.to_thread(backend.is_version_address, resource_address):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot write to individual version address: {resource_address}",
            )

        if upload_preference is None or upload_preference == "body":
            body_data = await _read_body_limited(request, MAX_UPLOAD_SIZE_BYTES)
            await asyncio.to_thread(backend.write_version, resource_address, body_data)
            uploaded_stat = await asyncio.to_thread(backend.stat, resource_address)
            mod_time = uploaded_stat.metadata.last_modified_timestamp.isoformat()
            identity = await asyncio.to_thread(backend.create_identity_from_resource_address, resource_address)
            resource_info = ResourceInfo(
                resource_identity=identity,
                metadata=Metadata(
                    data_object_size=uploaded_stat.metadata.data_object_size, last_modified_timestamp=mod_time
                ),
            )
            return JSONResponse(content=resource_info.model_dump(), status_code=201)
        elif upload_preference == "redirect":
            url = f"{REDIRECT_HOST}:{REDIRECT_PORT}/upload/" + urllib.parse.quote_plus(
                resource_address.replace("\\", "/")
            )
            redirect_response = WriteRedirectResponse(
                redirect=WriteRedirectProperties(
                    redirect_target_url=url,
                    method="POST",
                    additional_headers=[],
                    completion_header_names=["x-nvidia-storage-upload-location"],
                )
            )
            return JSONResponse(content=redirect_response.model_dump(), status_code=300)
        elif upload_preference == "multipart":
            upload_id = str(uuid.uuid4())
            backend.create_upload_session(upload_id)
            multipart_response = MultipartUploadResponse(
                multipart=MultipartUploadProperties(
                    upload_id=backend.encode_upload_id(upload_id, previous_version),
                    first_part_write_redirect=generate_upload_part_properties(upload_id, "0", resource_address),
                )
            )
            return JSONResponse(content=multipart_response.model_dump(), status_code=300)
    except (PermissionError, IsADirectoryError, OSError) as e:
        raise HTTPException(status_code=403, detail=f"Can't write to that resource address: {e}") from None
    # Mypy
    raise HTTPException(status_code=500, detail="reached unreachable code")


complete_redirect_route_params: dict[str, Any] = {
    "path": "/by-address/{destination_resource_address:path}/redirect/complete",
    "responses": {
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"content": {"application/json": HTTPValidationError.model_json_schema()}}
    },
    "methods": ["POST"],
}


async def complete_redirect_upload(
    destination_resource_address: RESOURCE_ADDRESS_TYPE,
    completion: CompleteUploadRequest,
) -> ResourceInfo:
    """Complete a redirect-based upload and finalize the new version.

    After a client uploads via HTTP redirect, this method verifies the
    upload completion headers and returns the resource info for the new version.

    Args:
        destination_resource_address: Target address for the upload
        completion: CompleteUploadRequest containing:
                   - additional_headers: Completion headers from HTTP upload
                                       (must include 'x-nvidia-storage-upload-location')

    Returns:
        ResourceInfo containing:
        - resource_identity: Identity and metadata for the new version
        - metadata: File size and modification timestamp

    Raises:
        HTTPException(400): If address invalid or required headers missing/invalid
        HTTPException(404): If upload location header value is invalid
        HTTPException(501): If backend doesn't support redirect uploads

    Note:
        The 'x-nvidia-storage-upload-location' header contains the
        version path created by the HTTP upload endpoint.
    """
    # Check if backend supports redirect uploads
    backend = get_backend()
    if not backend.supports_redirect_upload():
        raise HTTPException(
            status_code=501,
            detail=f"Redirect uploads not supported by {type(backend).__name__} backend",
        )

    if not await asyncio.to_thread(backend.is_address_valid, destination_resource_address):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid resource address: {destination_resource_address}",
        )
    if not await asyncio.to_thread(backend.exists, destination_resource_address):
        # Do not expose the existence of a file, react with a generic error message
        raise HTTPException(
            status_code=400,
            detail=f"Invalid resource address: {destination_resource_address}",
        )

    # Check that the client provides the correct upload headers for this redirect to be valid
    if len(completion.additional_headers) == 0:
        raise HTTPException(status_code=400, detail="No additional headers received")

    # Convert headers to dict
    headers_dict = {h.name: h.value for h in completion.additional_headers}

    try:
        result = await asyncio.to_thread(backend.complete_redirect_upload, destination_resource_address, headers_dict)
        return ResourceInfo(
            resource_identity=result.resource_identity,
            metadata=Metadata(
                data_object_size=result.metadata.data_object_size,
                last_modified_timestamp=result.metadata.last_modified_timestamp.isoformat(),
            ),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None


prepare_multipart_upload_route_params: dict[str, Any] = {
    "path": "/by-address/{resource_address:path}/multipart/prepare",
    "responses": {
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"content": {"application/json": HTTPValidationError.model_json_schema()}}
    },
    "methods": ["POST"],
}


async def prepare_multipart_upload_part_request(
    resource_address: RESOURCE_ADDRESS_TYPE, upload: MultipartUploadRequest
) -> UploadPartResponse:
    """Get redirect URLs for uploading one or more multipart parts.

    Returns HTTP redirect URLs for uploading the specified part(s) of
    a multipart upload.

    Args:
        resource_address: Target address for the complete file
        upload: MultipartUploadRequest containing:
               - upload_id: Upload session identifier
               - part_number: Starting part number
               - part_count: Optional number of consecutive parts (default 1)

    Returns:
        UploadPartResponse containing:
        - part_write_redirects: List of WriteRedirectProperties,
                                 one for each part

    Raises:
        HTTPException(501): If backend doesn't support multipart uploads

    Note:
        part_count allows batch-requesting redirect URLs for efficiency.
    """
    # Check if backend supports multipart uploads
    backend = get_backend()
    if not backend.supports_multipart_upload():
        raise HTTPException(
            status_code=501,
            detail=f"Multipart uploads not supported by {type(backend).__name__} backend",
        )

    redirects = []
    upload_id, _ = backend.decode_upload_id(upload.upload_id)
    for i in range(upload.part_count if upload.part_count else 1):
        redirects.append(generate_upload_part_properties(upload_id, str(upload.part_number + i), resource_address))
    return UploadPartResponse(
        part_write_redirects=redirects,
    )


complete_multipart_upload_route_params: dict[str, Any] = {
    "path": "/by-address/{resource_address:path}/multipart/complete",
    "responses": {
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"content": {"application/json": HTTPValidationError.model_json_schema()}}
    },
    "methods": ["POST"],
}


async def complete_multipart_upload(
    resource_address: RESOURCE_ADDRESS_TYPE, completion: CompleteMultipartUploadRequest
) -> ResourceInfo:
    """Finalize a multipart upload by assembling all parts.

    Assembles the uploaded parts into the final file, creates a new version,
    and cleans up the upload session.

    Args:
        resource_address: Target address for the complete file
        completion: CompleteMultipartUploadRequest containing:
                   - upload_id: Upload session identifier
                   - parts: Ordered list of CompletedUploadPart with headers from uploads

    Returns:
        ResourceInfo containing:
        - resource_identity: Identity and metadata for the new version
        - metadata: File size and modification timestamp

    Raises:
        HTTPException(400): If address invalid, upload_id doesn't exist,
                          parts out of order, or headers missing
        HTTPException(412): If previous_version doesn't match current version
        HTTPException(501): If backend doesn't support multipart uploads

    Note:
        Parts must be provided in order (part_number 0, 1, 2, ...).
        The upload session directory is cleaned up after completion.
    """
    # Check if backend supports multipart uploads
    backend = get_backend()
    if not backend.supports_multipart_upload():
        raise HTTPException(
            status_code=501,
            detail=f"Multipart uploads not supported by {type(backend).__name__} backend",
        )

    if not await asyncio.to_thread(backend.is_address_valid, resource_address):
        raise HTTPException(status_code=400, detail=f"Invalid resource address: {resource_address}")

    upload_id, previous_version = await asyncio.to_thread(backend.decode_upload_id, completion.upload_id)
    if previous_version:
        # Check if backend supports optimistic locking for write
        if not backend.get_optimistic_locking_support().write:
            raise HTTPException(
                status_code=501,
                detail=f"Multipart upload with previous_version parameter is not supported by {type(backend).__name__} backend.",
            )
        raise_on_previous_version_not_latest(resource_address, previous_version)

    if not backend.upload_session_exists(upload_id):
        raise HTTPException(400, detail=f"No active multipart upload for upload id {upload_id}")

    assembled_data = bytearray()
    # Assemble the parts into the final file
    for i, part in enumerate(completion.parts):
        if part.part_number != i:
            raise HTTPException(status_code=400, detail=f"Invalid part number: {i}")
        header_name = "local-file"
        found = False
        for header in part.additional_headers:
            if header.name == header_name:
                found = True
                source_file = header.value
                with backend.safe_open(source_file, "rb") as input_file:  # ty: ignore[unresolved-attribute]
                    data = input_file.read()
                    assembled_data.extend(data)
                break
        if not found:
            raise HTTPException(
                status_code=400,
                detail=f"part {part.part_number} is missing the required header {header_name}",
            )

    await asyncio.to_thread(backend.write_version, resource_address, bytes(assembled_data))
    stat = await asyncio.to_thread(backend.stat, resource_address)
    backend.cleanup_upload_session(upload_id)
    identity = await asyncio.to_thread(backend.create_identity_from_resource_address, resource_address)
    return ResourceInfo(
        resource_identity=identity,
        metadata=Metadata(
            data_object_size=stat.metadata.data_object_size,
            last_modified_timestamp=stat.metadata.last_modified_timestamp.isoformat(),
        ),
    )


abort_multipart_upload_route_params: dict[str, Any] = {
    "path": "/by-address/{resource_address:path}/multipart/abort",
    "status_code": 204,
    "responses": {
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"content": {"application/json": HTTPValidationError.model_json_schema()}}
    },
    "methods": ["POST"],
}


async def abort_multipart_upload(resource_address: RESOURCE_ADDRESS_TYPE, params: MultipartUploadAbortRequest):
    """Cancel a multipart upload and clean up resources.

    Aborts an in-progress multipart upload and removes any uploaded parts.
    This operation is idempotent - aborting a non-existent upload succeeds.

    Args:
        resource_address: Target address (currently unused, but required for routing)
        params: MultipartUploadAbortRequest containing:
               - upload_id: Upload session identifier

    Returns:
        Response with status 204 (No Content).

    Raises:
        HTTPException(501): If backend doesn't support multipart uploads

    Note:
        Always returns success, even if the upload_id doesn't exist.
        This matches the behavior of cloud storage services where you
        cannot inspect abort status.
    """
    # Check if backend supports multipart uploads
    backend = get_backend()
    if not backend.supports_multipart_upload():
        raise HTTPException(
            status_code=501,
            detail=f"Multipart uploads not supported by {type(backend).__name__} backend",
        )

    upload_id, _ = backend.decode_upload_id(params.upload_id)
    backend.cleanup_upload_session(upload_id)
    # Abort always returns success, even if there was nothing to abort. The reason is the hyperscalers do not allow
    # to inspect the status of an abort, so the client can just trust it worked ot it has to inspect the status afterward
    return Response(status_code=204)


@fileobject_service_beta.delete(
    "/by-address/{resource_address:path}",
    status_code=status.HTTP_204_NO_CONTENT,
)
@fileobject_service_alpha.delete(
    "/by-address/{resource_address:path}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_data_object(
    resource_address: RESOURCE_ADDRESS_TYPE,
    previous_version: RESOURCE_IDENTITY_TYPE | None = None,
):
    """Delete a file and all its versions.

    Removes a file from the storage service, deleting all versions.
    This operation is idempotent - deleting a non-existent file succeeds.

    Args:
        resource_address: File address to delete
        previous_version: Optional expected current version (for optimistic locking)

    Returns:
        Response with status 204 (No Content).

    Raises:
        HTTPException(400): If address is invalid, versioned, or not a file
        HTTPException(412): If previous_version doesn't match current version

    Note:
        - Cannot delete individual versions, only the entire file
        - Cannot delete versioned addresses or folders
        - Supports optimistic locking via previous_version
    """
    backend = get_backend()

    if not await asyncio.to_thread(backend.is_address_valid, resource_address):
        raise HTTPException(status_code=400, detail=f"Invalid resource address: {resource_address}")

    if await asyncio.to_thread(backend.is_version_address, resource_address):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete individual version address: {resource_address}",
        )

    if not await asyncio.to_thread(backend.exists, resource_address):
        return

    if not await asyncio.to_thread(backend.is_file, resource_address):
        raise HTTPException(400, detail=f"Failed to remove {resource_address}, not a  file")
    if previous_version is not None:
        # Check if backend supports optimistic locking for delete
        if not backend.get_optimistic_locking_support().delete:
            raise HTTPException(
                status_code=501,
                detail=f"Delete operation with previous_version parameter is not supported by {type(backend).__name__} backend.",
            )
        raise_on_previous_version_not_latest(resource_address, previous_version)
    await asyncio.to_thread(backend.remove_by_address, resource_address)


@fileobject_service_alpha.post(
    "/by-identity/{resource_identity:path}/copy",
    status_code=201,
    responses={
        status.HTTP_201_CREATED: {
            "description": "Successful response",
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "content": {"application/json": HTTPValidationError.model_json_schema()}
        },
    },
)
async def copy_(
    resource_identity: RESOURCE_IDENTITY_TYPE,
    copy_request: CopyRequest,
) -> CopyResponse:
    """Copy a file version to a new destination (v1alpha only).

    Creates a copy of a specific file version at a new address, creating
    a new version at the destination.

    Args:
        resource_identity: Identity of version to copy
        copy_request: CopyRequest containing:
                     - destination_resource_address: Target address
                     - previous_version: Optional expected version at destination (for optimistic locking)

    Returns:
        CopyResponse containing:
        - resource_identity: Identity of the newly created version at destination

    Raises:
        HTTPException(400): If identities/addresses invalid, or destination is versioned
        HTTPException(404): If source identity doesn't exist
        HTTPException(412): If previous_version doesn't match destination current version
        HTTPException(403): If read permission denied on source
        HTTPException(500): If source identity refers to a directory (unexpected state)

    Note:
        This is a v1alpha-only method, not available in v1beta.
        The copy creates a new version at the destination.
    """
    backend = get_backend()

    # Convert source identity to resource address
    # Use url_from_identity to preserve version info for versioned copies
    try:
        source_address = await asyncio.to_thread(backend.url_from_identity, resource_identity)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, detail=f"Invalid source identity: {e}") from None

    # Validate source exists
    if not await asyncio.to_thread(backend.exists, source_address):
        raise HTTPException(404, detail="Source resource not found")

    # Validate source is a file, not a folder
    if await asyncio.to_thread(backend.is_dir, source_address):
        raise HTTPException(400, detail="Cannot copy folders, only files")

    # Validate destination address
    if not await asyncio.to_thread(backend.is_address_valid, copy_request.destination_resource_address):
        raise HTTPException(
            400,
            detail=f"{copy_request.destination_resource_address} is not a valid destination address",
        )

    if await asyncio.to_thread(backend.is_version_address, copy_request.destination_resource_address):
        raise HTTPException(
            400,
            detail=f"Cannot copy to individual version address: {copy_request.destination_resource_address}",
        )

    # Check previous version if specified
    if copy_request.previous_version is not None:
        # Check if backend supports optimistic locking for copy
        if not backend.get_optimistic_locking_support().copy:
            raise HTTPException(
                status_code=501,
                detail=f"Copy operation with previous_version parameter is not supported by {type(backend).__name__} backend.",
            )
        raise_on_previous_version_not_latest(copy_request.destination_resource_address, copy_request.previous_version)

    try:
        # Perform the copy using backend's copy method
        new_identity = await asyncio.to_thread(backend.copy, source_address, copy_request.destination_resource_address)
        return CopyResponse(resource_identity=new_identity)
    except PermissionError as e:
        raise HTTPException(403, detail=str(e)) from None
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, detail=str(e)) from None


for fileobject_service in [fileobject_service_alpha, fileobject_service_beta]:
    fileobject_service.add_api_route(**enumerate_root_route_params, endpoint=enumerate_root)
    fileobject_service.add_api_route(**enumerate_route_params, endpoint=enumerate_)
    fileobject_service.add_api_route(**stat_route_params, endpoint=stat_)
    fileobject_service.add_api_route(**read_from_address_route_params, endpoint=read_from_address)
    fileobject_service.add_api_route(**read_route_params, endpoint=read)
    fileobject_service.add_api_route(**get_upload_options_route_params, endpoint=get_upload_options)
    fileobject_service.add_api_route(**write_route_params, endpoint=write)
    fileobject_service.add_api_route(**complete_redirect_route_params, endpoint=complete_redirect_upload)
    fileobject_service.add_api_route(
        **prepare_multipart_upload_route_params,
        endpoint=prepare_multipart_upload_part_request,
    )
    fileobject_service.add_api_route(**complete_multipart_upload_route_params, endpoint=complete_multipart_upload)
    fileobject_service.add_api_route(**abort_multipart_upload_route_params, endpoint=abort_multipart_upload)


@fileobject_service_alpha.get(
    "/optimistic-locking-support/{resource_address:path}",
    status_code=200,
)
async def get_optimistic_locking_support(
    resource_address: RESOURCE_ADDRESS_TYPE,
) -> OptimisticLockingSupportResponse:
    """Query the server's support for optimistic locking.

    Returns information about which operations support conditional execution
    with previous_version parameter for the given resource address. This allows
    clients to determine capabilities before attempting conditional operations.

    Args:
        resource_address: The resource address to check optimistic locking support for

    Returns:
        OptimisticLockingSupportResponse containing boolean flags for:
        - write: True if Write supports previous_version
        - delete: True if Delete supports previous_version
        - copy: True if Copy supports previous_version
        - move: True if Move supports source_previous_version/destination_previous_version
    """
    # Note: Currently, the backend returns the same support for all addresses.
    # In the future, this could vary based on the resource_address.
    backend = get_backend()
    support = backend.get_optimistic_locking_support()

    return OptimisticLockingSupportResponse(
        supports_write=support.write,
        supports_delete=support.delete,
        supports_copy=support.copy,
        supports_move=support.move,
    )


@fileobject_service_alpha.post(
    "/by-address/{source_resource_address:path}/move",
    status_code=201,
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"content": {"application/json": HTTPValidationError.model_json_schema()}}
    },
)
async def move_object(
    source_resource_address: RESOURCE_ADDRESS_TYPE,
    move_request: MoveRequest,
) -> MoveResponse:
    """Move a file object from one resource address to another.

    This operation atomically moves a file to a new location, similar to a rename
    or mv operation. The move can be conditional on the source or destination being
    at a specific version (for conflict resolution).

    Args:
        source_resource_address: Current address of the file to move
        move_request: MoveRequest containing:
            - destination_resource_address: New address for the file
            - source_previous_version: Expected current version (optional, for conflict detection)
            - destination_previous_version: Expected destination version (optional)

    Returns:
        MoveResponse containing the resource_identity of the file at its new location

    Raises:
        HTTPException(400): If addresses are invalid or versioned, or if moving a directory
        HTTPException(404): If source doesn't exist or expected version not found
        HTTPException(412): If precondition fails (version mismatch)
        HTTPException(403): If permission denied

    Note:
        - Source must be a file, not a directory
        - Cannot move from or to versioned addresses (e.g., address@version)
        - If source_previous_version is specified, move fails if it's not the current version
        - The file's version history is preserved at the new location
    """
    backend = get_backend()

    if not source_resource_address:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="source_resource_address is required")
    if not await asyncio.to_thread(backend.is_address_valid, source_resource_address):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid source resource address")

    if await asyncio.to_thread(backend.is_version_address, source_resource_address):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Invalid source resource address. Versioned addresses can't be moved from.",
        )

    # Check source exists
    if not await asyncio.to_thread(backend.exists, source_resource_address):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Source not found")

    # Check source is a file, not a folder
    if await asyncio.to_thread(backend.is_dir, source_resource_address):
        error_code_to_send = (
            status.HTTP_404_NOT_FOUND if backend.folder_mode() == FolderMode.HYBRID else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(error_code_to_send, detail="Cannot move directories")

    # Check source_previous_version if specified
    if move_request.source_previous_version is not None:
        # Check if backend supports optimistic locking for move
        if not backend.get_optimistic_locking_support().move:
            raise HTTPException(
                status_code=501,
                detail=f"Move operation with source_previous_version parameter is not supported by {type(backend).__name__} backend.",
            )
        try:
            # Force a validity check of the given source identity by parsing it
            try:
                await asyncio.to_thread(backend.address_from_identity, move_request.source_previous_version)
            except ValueError:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    f"Invalid source_previous_version given: {move_request.source_previous_version}",
                ) from None
            source_is_latest = await asyncio.to_thread(
                backend.is_version_latest,
                source_resource_address,
                move_request.source_previous_version,
            )
            if not source_is_latest:
                raise HTTPException(
                    status.HTTP_412_PRECONDITION_FAILED,
                    detail="source_previous_version no longer matches current version at source_resource_address",
                )
        except FileNotFoundError:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                detail=f"source_previous_version given, but no versions found at {source_resource_address}",
            ) from None
        except (ValueError, AttributeError) as e:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid source_previous_version: {e}",
            ) from None

    # Validate destination address
    if not await asyncio.to_thread(backend.is_address_valid, move_request.destination_resource_address):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Invalid resource address given to Move as destination resource address",
        )
    if await asyncio.to_thread(backend.is_version_address, move_request.destination_resource_address):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Invalid destination resource address. Versioned addresses can't be used as move destination.",
        )

    # Check destination_previous_version if specified
    if move_request.destination_previous_version is not None:
        # Check if backend supports optimistic locking for move
        if not backend.get_optimistic_locking_support().move:
            raise HTTPException(
                status_code=501,
                detail=f"Move operation with destination_previous_version parameter is not supported by {type(backend).__name__} backend.",
            )
        raise_on_previous_version_not_latest(
            move_request.destination_resource_address,
            move_request.destination_previous_version,
        )

    try:
        # Perform the move using backend's move method
        result_identity = await asyncio.to_thread(
            backend.move, source_resource_address, move_request.destination_resource_address
        )
        return MoveResponse(resource_identity=result_identity)
    except FileNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Source not found") from None
    except PermissionError:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Permission denied") from None
    except OSError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None

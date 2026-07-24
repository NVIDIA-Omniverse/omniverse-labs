# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""FileFolderService implementation for REST.

This module implements the FileFolderService from the Storage API, which
provides operations for managing folders and listing their contents:
- List: Get immediate children of a folder (addresses only)
- ListStat: Get immediate children with full metadata
- CreateFolder: Create a new folder
- DeleteFolder: Delete an empty folder
- GetFolderMode: Query the folder mode

The behavior adapts to different folder modes (native, no_empty,
placeholder) to support various storage backend semantics via HTTP endpoints.
"""

import asyncio
from typing import (
    Annotated,
    Any,
)

from fastapi import (
    HTTPException,
    Response,
)
from pydantic import WithJsonSchema
from starlette import status

from omni_onedrive_storage_service.backends import get_backend
from omni_onedrive_storage_service.backends.path_helpers import (
    build_address,
    get_relative_path_from_address,
)
from omni_onedrive_storage_service.backends.storage_backend_interface import (
    FolderMode as BackendFolderMode,
)

from . import rest_messages
from .rest_messages import (
    FOLDER_ADDRESS_TYPE,
    MAX_PAGE_SIZE_SCHEMA,
    PAGE_HANDLE_TYPE,
    FolderMode,
    GetFolderModeResponse,
    HTTPValidationError,
    ListItem,
    ListResponse,
    ListStatResponse,
)
from .routes import (
    filefolder_service_alpha,
    filefolder_service_beta,
)


def _build_full_uri(base_uri: str, path: str, entry: str) -> str:
    """Build a full, URL-encoded outbound resource address."""
    return build_address(base_uri, path, child=entry)


@filefolder_service_alpha.put(
    "/{folder_address:path}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {
            "description": "Folder created successfully or already exists (idempotent)",
        },
        status.HTTP_400_BAD_REQUEST: {
            "description": "Invalid resource address",
        },
        status.HTTP_409_CONFLICT: {
            "description": "Conflict - resource already exists as a file",
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "content": {"application/json": HTTPValidationError.model_json_schema()}
        },
    },
)
async def create_folder_v1alpha(folder_address: FOLDER_ADDRESS_TYPE):
    """Create a folder at the given resource address.

    This operation is idempotent - creating a folder that already exists succeeds.
    The behavior depends on the folder simulation mode:
    - native: Creates actual filesystem directory
    - no_empty: No-op (folders don't exist until they contain files)
    - placeholder: Creates marker file to simulate folder

    Args:
        folder_address: Resource address where folder should be created

    Returns:
        Response with status 204 (No Content) on success

    Raises:
        HTTPException(400): If resource address is invalid
        HTTPException(409): If a file already exists at this address
        HTTPException(403): If permission denied
        HTTPException(500): If folder creation fails for other reasons

    Note:
        This is a v1alpha endpoint. Folder-mode behavior depends on the active
        backend's declared folder mode (OneDrive uses ``native`` — real folders).
    """
    backend = get_backend()

    if not await asyncio.to_thread(backend.is_address_valid, folder_address):
        raise HTTPException(status_code=400, detail=f"Invalid resource address: {folder_address}")

    if await asyncio.to_thread(backend.exists, folder_address) and await asyncio.to_thread(
        backend.is_file, folder_address
    ):
        raise HTTPException(status_code=409, detail="Conflict - resource already exists as a file")

    try:
        await asyncio.to_thread(backend.create_folder, folder_address)
        return Response(status_code=204)  # 204 No Content when creating
    except FileExistsError:
        raise HTTPException(status_code=409, detail="Conflict - resource already exists as a file") from None
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied") from None


@filefolder_service_alpha.get(
    "/get-folder-mode/{folder_address:path}",
    responses={
        status.HTTP_200_OK: {
            "description": "Success response.",
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "content": {"application/json": HTTPValidationError.model_json_schema()}
        },
    },
)
async def get_folder_mode(folder_address: FOLDER_ADDRESS_TYPE) -> GetFolderModeResponse:
    """Get the folder simulation mode used by this storage service.

    Different storage backends handle folders differently:
    - NATIVE: Real filesystem directories (like local filesystem)
    - NO_EMPTY: Empty folders don't exist (like AWS S3)
    - HYBRID: Uses placeholder files to simulate folders

    Args:
        folder_address: Resource address to check. The folder mode is typically
                       consistent across the entire backend, but the address is
                       validated to ensure it's a valid location.

    Returns:
        GetFolderModeResponse indicating the current folder_mode

    Raises:
        HTTPException(400): If folder address is invalid
        HTTPException(500): If service is using unknown folder mode

    Note:
        Knowing the folder mode helps clients understand whether:
        - Empty folders can exist
        - Folder creation is required before writing files
        - Listing empty folders is supported
    """
    backend = get_backend()

    # Validate the folder address
    if not await asyncio.to_thread(backend.is_address_valid, folder_address):
        raise HTTPException(status_code=400, detail=f"Invalid resource address: {folder_address}")

    # Get the folder mode from the backend
    backend_folder_mode = backend.folder_mode()
    if backend_folder_mode == BackendFolderMode.NATIVE:
        return GetFolderModeResponse(folder_mode=FolderMode.NATIVE)
    elif backend_folder_mode == BackendFolderMode.NO_EMPTY:
        return GetFolderModeResponse(folder_mode=FolderMode.NO_EMPTY)
    elif backend_folder_mode == BackendFolderMode.HYBRID:
        return GetFolderModeResponse(folder_mode=FolderMode.HYBRID)
    else:
        raise HTTPException(status_code=500, detail=f"Unknown folder mode: {backend_folder_mode}")


list_root_route: dict[str, Any] = {
    "path": "/",
    "responses": {
        status.HTTP_200_OK: {
            "description": "Success response.",
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "content": {"application/json": HTTPValidationError.model_json_schema()}
        },
    },
    "include_in_schema": False,
}


async def list_root_api() -> ListResponse:
    """List the contents of the root folder.

    This is a special case of list_api() for the root directory, required
    because path parameters don't match empty strings in FastAPI routing.

    Returns:
        ListResponse containing subfolder_addresses and sub_resource_addresses

    See Also:
        list_api() for the main implementation
    """
    return await list_api("")


for api in [filefolder_service_alpha, filefolder_service_beta]:
    api.add_api_route(**list_root_route, endpoint=list_root_api, methods=["GET"])

list_route: dict[str, Any] = {
    "path": "/list/{folder_address:path}",
    "responses": {
        status.HTTP_200_OK: {
            "description": "Success response.",
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "content": {"application/json": HTTPValidationError.model_json_schema()}
        },
    },
}


async def list_api(
    folder_address: FOLDER_ADDRESS_TYPE,
    max_page_size: Annotated[int | None, WithJsonSchema(MAX_PAGE_SIZE_SCHEMA)] = 1000,
    continuation_handle: PAGE_HANDLE_TYPE = None,
) -> ListResponse:
    """List contents of a folder (immediate children only, non-recursive).

    Returns immediate children of the specified folder, separated into
    subfolders and files. Results are paginated for large directories.

    Args:
        folder_address: Folder to list (can be empty string for root)
        max_page_size: Maximum number of items to return per page (default 1000)
        continuation_handle: Opaque token from previous response to get next page

    Returns:
        ListResponse containing:
        - subfolder_addresses: List of child folder addresses
        - sub_resource_addresses: List of child file addresses
        - next_continuation_handle: Token for next page, or None if complete

    Raises:
        HTTPException(400): If trying to list a versioned address
        HTTPException(404): If folder doesn't exist or address points to a file

    Note:
        This is non-recursive (only immediate children). For recursive listing,
        use the enumerate endpoint instead.

    Example:
        Listing 'folder1' might return:
        {
            \"subfolder_addresses\": [\"folder1/subfolder1\", \"folder1/subfolder2\"],
            \"sub_resource_addresses\": [\"folder1/file1.txt\", \"folder1/file2.txt\"],
            \"next_continuation_handle\": \"100\"
        }
    """
    backend = get_backend()

    if await asyncio.to_thread(backend.is_version_address, folder_address):
        raise HTTPException(status_code=400, detail="Cannot list versioned addresses")
    if await asyncio.to_thread(backend.is_file, folder_address):
        raise HTTPException(status_code=404, detail="Cannot list file addresses")
    if not await asyncio.to_thread(backend.exists, folder_address):
        raise HTTPException(status_code=404, detail="Resource address not found")
    subfolder_addresses, sub_resource_addresses = await asyncio.to_thread(backend.list, folder_address)
    subfolder_addresses = [folder_address + "/" + address for address in subfolder_addresses]
    sub_resource_addresses = [folder_address + "/" + address for address in sub_resource_addresses]

    combined_entries = [("folder", address) for address in subfolder_addresses] + [
        ("resource", address) for address in sub_resource_addresses
    ]
    total_entries = len(combined_entries)

    start_address = int(continuation_handle) if continuation_handle else 0
    start_address = max(0, min(start_address, total_entries))

    page_size = max_page_size or 1000
    end_address = min(start_address + page_size, total_entries)

    paginated_entries = combined_entries[start_address:end_address]
    paginated_subfolder_addresses = [address for entry_type, address in paginated_entries if entry_type == "folder"]
    paginated_sub_resource_addresses = [
        address for entry_type, address in paginated_entries if entry_type == "resource"
    ]

    next_continuation_handle = str(end_address) if end_address < total_entries else None
    return ListResponse(
        subfolder_addresses=paginated_subfolder_addresses,
        sub_resource_addresses=paginated_sub_resource_addresses,
        next_continuation_handle=next_continuation_handle,
    )


for api in [filefolder_service_alpha, filefolder_service_beta]:
    api.add_api_route(**list_route, endpoint=list_api, methods=["GET"])

delete_folder_route: dict[str, Any] = {
    "path": "/list/{folder_address:path}",
    "responses": {
        status.HTTP_204_NO_CONTENT: {
            "description": "Success response.",
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "content": {"application/json": HTTPValidationError.model_json_schema()}
        },
    },
    "status_code": status.HTTP_204_NO_CONTENT,
}


async def delete_folder_api(
    folder_address: FOLDER_ADDRESS_TYPE,
):
    """Delete an empty folder.

    Removes a folder if it is empty. This operation is idempotent - deleting
    a non-existent folder succeeds (returns 204).

    Args:
        folder_address: Address of the folder to delete

    Returns:
        Response with status 204 (No Content) on success

    Raises:
        HTTPException(400): If folder is not empty or address is invalid

    Note:
        Only empty folders can be deleted. To delete a folder tree, first
        delete all files and subfolders recursively.
    """
    backend = get_backend()

    if not await asyncio.to_thread(backend.is_address_valid, folder_address):
        raise HTTPException(400, detail=f"invalid folder address {folder_address}")
    elif not await asyncio.to_thread(backend.exists, folder_address):
        return Response(status_code=204)
    elif await asyncio.to_thread(backend.is_dir, folder_address):
        if not await asyncio.to_thread(backend.remove_empty_folder, folder_address):
            raise HTTPException(400, detail=f"Failed to remove {folder_address}, folder not empty")
        return Response(status_code=204)
    else:
        raise HTTPException(400, detail=f"Failed to remove {folder_address}, is not a folder")


for api in [filefolder_service_alpha, filefolder_service_beta]:
    api.add_api_route(**delete_folder_route, endpoint=delete_folder_api, methods=["DELETE"])

liststat_route: dict[str, Any] = {
    "path": "/liststat/{folder_address:path}",
    "status_code": status.HTTP_200_OK,
    "responses": {
        status.HTTP_200_OK: {
            "description": "Success response.",
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "content": {"application/json": HTTPValidationError.model_json_schema()}
        },
    },
}


async def liststat_api(
    folder_address: FOLDER_ADDRESS_TYPE,
    max_page_size: Annotated[int | None, WithJsonSchema(MAX_PAGE_SIZE_SCHEMA)] = 1000,
    continuation_handle: PAGE_HANDLE_TYPE = None,
) -> ListStatResponse:
    """List folder contents with metadata in a single operation.

    Combines listing and stat operations for efficiency - returns immediate
    children of a folder along with their metadata (size, modification time,
    resource identity). More efficient than calling list() then stat() for
    each item.

    Args:
        folder_address: Folder to list (can be empty string for root)
        max_page_size: Maximum number of items to return per page (default 1000)
        continuation_handle: Token from previous response to get next page

    Returns:
        ListStatResponse containing:
        - subfolder_addresses: List of child folder addresses
        - entries: List of ListItem objects with address, identity, and metadata
        - next_continuation_handle: Token for next page, or None if complete

    Raises:
        HTTPException(400): If trying to list a versioned address
        HTTPException(404): If folder doesn't exist or address points to a file
        HTTPException(500): On internal errors

    Note:
        Preferred over list() when you need metadata, as it avoids round-trips.
        Particularly useful for UI applications that need to display file sizes
        and timestamps.
    """
    backend = get_backend()

    if await asyncio.to_thread(backend.is_version_address, folder_address):
        raise HTTPException(status_code=400, detail="Cannot list versioned addresses")
    if await asyncio.to_thread(backend.is_file, folder_address):
        raise HTTPException(status_code=404, detail="Cannot list file addresses")
    if not await asyncio.to_thread(backend.exists, folder_address):
        raise HTTPException(status_code=404, detail="Resource address not found")
    start_index = int(continuation_handle) if continuation_handle else 0
    page_size = max_page_size if max_page_size else 1000

    # Request one extra item to accurately detect if more pages exist
    subfolder_addresses, file_metadata = await asyncio.to_thread(
        backend.list_stat, folder_address, start_index=start_index, limit=page_size + 1
    )
    base_uri = get_backend().base_uri
    path = get_relative_path_from_address(base_uri, folder_address)
    subfolder_addresses = [_build_full_uri(base_uri, path, entry) for entry in subfolder_addresses]

    # Check if there are more items beyond this page
    total_returned = len(subfolder_addresses) + len(file_metadata)
    has_more = total_returned > page_size

    # Trim to page_size if we got more
    if has_more:
        # Trim the combined list to page_size
        if len(subfolder_addresses) >= page_size:
            subfolder_addresses = subfolder_addresses[:page_size]
            file_metadata = []
        else:
            files_to_include = page_size - len(subfolder_addresses)
            file_metadata = file_metadata[:files_to_include]

    sub_resources = [
        ListItem(
            resource_address=_build_full_uri(base_uri, path, file.resource_address),
            resource_identity=file.resource_identity,  # ty: ignore[invalid-argument-type]
            metadata=rest_messages.Metadata(
                data_object_size=file.metadata.data_object_size,  # ty: ignore[unresolved-attribute]
                last_modified_timestamp=(
                    file.metadata.last_modified_timestamp.isoformat() if file.metadata.last_modified_timestamp else None  # ty: ignore[unresolved-attribute]
                ),
            ),
        )
        for file in file_metadata
    ]

    next_handle = str(start_index + page_size) if has_more else None

    return ListStatResponse(
        subfolder_addresses=subfolder_addresses,
        entries=sub_resources,
        next_continuation_handle=next_handle,
    )


for api in [filefolder_service_alpha, filefolder_service_beta]:
    api.add_api_route(**liststat_route, endpoint=liststat_api, methods=["GET"])

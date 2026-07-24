# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Versioning service implementation for REST.

This module implements the VersioningService from the Storage API, which
provides operations for enumerating file versions. The Storage API maintains
immutable versions of files, and this service allows clients to discover and
access historical versions via HTTP endpoints.
"""

import asyncio
from typing import (
    Annotated,
)

from fastapi import HTTPException
from pydantic import WithJsonSchema
from starlette import status

from omni_onedrive_storage_service.backends import get_backend
from omni_onedrive_storage_service.backends.storage_backend_interface import (
    VersionsOrder as BackendVersionsOrder,
)

from .rest_messages import (
    MAX_PAGE_SIZE_SCHEMA,
    PAGE_HANDLE_TYPE,
    RESOURCE_ADDRESS_TYPE,
    EnumerateVersionsResponse,
    HTTPValidationError,
    Metadata,
    ResourceInfo,
    VersionInfo,
    VersionsOrder,
)
from .routes import (
    versioning_service_alpha,
    versioning_service_beta,
)


@versioning_service_alpha.get(
    "/{resource_address:path}/versions",
    responses={
        status.HTTP_200_OK: {
            "description": "Success response.",
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "content": {"application/json": HTTPValidationError.model_json_schema()}
        },
    },
)
async def enumerate_versions_alpha(
    resource_address: RESOURCE_ADDRESS_TYPE,
    max_page_size: Annotated[int | None, WithJsonSchema(MAX_PAGE_SIZE_SCHEMA)] = 1000,
    continuation_handle: PAGE_HANDLE_TYPE = None,
) -> EnumerateVersionsResponse:
    """Enumerate all versions of a resource (v1alpha).

    Lists all available versions of a fileobject in the order determined by the
    storage service (newest first, oldest first, or by key). Each version
    represents an immutable snapshot of the fileobject at a point in time.

    Args:
        resource_address: Address of the resource to enumerate
        max_page_size: Maximum number of versions to return per page (default 1000)
        continuation_handle: Opaque token from previous response to get next page

    Returns:
        EnumerateVersionsResponse containing:
        - items: List of VersionInfo messages with identity, metadata, and resource_address
        - next_continuation_handle: Token for next page, or None if complete
        - versions_order: Ordering of the returned versions

    Raises:
        HTTPException(400): If resource_address is versioned or is a folder
        HTTPException(404): If resource_address doesn't exist
        HTTPException(500): On internal errors

    Note:
        - Version addresses (containing version IDs) cannot be enumerated
        - Only files (not folders) can have versions
        - The versions_order field indicates how versions are sorted
        - v1alpha includes resource_address in each VersionInfo
    """
    return await _enumerate_versions(resource_address, max_page_size, continuation_handle, True)


@versioning_service_beta.get(
    "/{resource_address:path}/versions",
    responses={
        status.HTTP_200_OK: {
            "description": "Success response.",
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "content": {"application/json": HTTPValidationError.model_json_schema()}
        },
    },
)
async def enumerate_versions_beta(
    resource_address: RESOURCE_ADDRESS_TYPE,
    max_page_size: Annotated[int | None, WithJsonSchema(MAX_PAGE_SIZE_SCHEMA)] = 1000,
    continuation_handle: PAGE_HANDLE_TYPE = None,
) -> EnumerateVersionsResponse:
    """Enumerate all versions of a resource (v1beta).

    Lists all available versions of a fileobject in the order determined by the
    storage service (newest first, oldest first, or by key). Each version
    represents an immutable snapshot of the fileobject at a point in time.

    Args:
        resource_address: Address of the resource to enumerate
        max_page_size: Maximum number of versions to return per page (default 1000)
        continuation_handle: Opaque token from previous response to get next page

    Returns:
        EnumerateVersionsResponse containing:
        - items: List of VersionInfo messages with identity and metadata
        - next_continuation_handle: Token for next page, or None if complete
        - versions_order: Ordering of the returned versions

    Raises:
        HTTPException(400): If resource_address is versioned or is a folder
        HTTPException(404): If resource_address doesn't exist
        HTTPException(500): On internal errors

    Note:
        - Version addresses (containing version IDs) cannot be enumerated
        - Only files (not folders) can have versions
        - The versions_order field indicates how versions are sorted
        - v1beta excludes resource_address from VersionInfo (unlike v1alpha)
    """
    return await _enumerate_versions(resource_address, max_page_size, continuation_handle, False)


async def _enumerate_versions(
    resource_address: str,
    max_page_size: int | None,
    continuation_handle: str | None,
    is_alpha: bool,
) -> EnumerateVersionsResponse:
    """Internal implementation of version enumeration.

    Shared implementation for both v1alpha and v1beta endpoints. Handles
    pagination, validation, and version ordering.

    Args:
        resource_address: Address of the resource to enumerate
        max_page_size: Maximum number of versions per page
        continuation_handle: Pagination token
        is_alpha: Whether to include resource_address in VersionInfo (v1alpha only)

    Returns:
        EnumerateVersionsResponse with paginated version list

    Raises:
        HTTPException(400): If address is versioned or is a folder
        HTTPException(404): If address doesn't exist
        HTTPException(500): On internal errors
    """
    backend = get_backend()

    if await asyncio.to_thread(backend.is_version_address, resource_address):
        raise HTTPException(status_code=400, detail="Cannot enumerate versions of versioned addresses")
    if not await asyncio.to_thread(backend.exists, resource_address):
        raise HTTPException(status_code=404, detail="Resource address not found")
    if await asyncio.to_thread(backend.is_dir, resource_address):
        raise HTTPException(status_code=400, detail="Cannot enumerate versions of folder addresses")

    try:
        start_index = 0 if continuation_handle is None else int(continuation_handle)
        page_size = max_page_size if max_page_size is not None else 1000

        # Request one extra item to accurately detect if more pages exist
        versions, versions_order = await asyncio.to_thread(
            backend.enumerate_versions, resource_address, start_index=start_index, limit=page_size + 1
        )
        versions_order_mapping = {
            BackendVersionsOrder.NEWEST_FIRST: VersionsOrder.NEWEST_FIRST,
            BackendVersionsOrder.OLDEST_FIRST: VersionsOrder.OLDEST_FIRST,
            BackendVersionsOrder.BY_KEY: VersionsOrder.BY_KEY,
        }

        # Check if there are more items beyond this page
        has_more = len(versions) > page_size
        if has_more:
            versions = versions[:page_size]  # Trim to page_size

        entries = []
        for item in versions:
            metadata = Metadata(
                data_object_size=item.metadata.data_object_size,
                last_modified_timestamp=item.metadata.last_modified_timestamp.isoformat(),
            )
            if is_alpha:
                # Alpha is adding the resource address field
                entries.append(
                    VersionInfo(
                        resource_info=ResourceInfo(
                            resource_identity=item.resource_identity,
                            metadata=metadata,
                        ),
                        sorting_key=item.sorting_key,
                        resource_address=item.resource_address,
                    )
                )
            else:
                entries.append(
                    VersionInfo(
                        resource_info=ResourceInfo(
                            resource_identity=item.resource_identity,
                            metadata=metadata,
                        ),
                        sorting_key=item.sorting_key,
                    )
                )

        next_handle = str(start_index + page_size) if has_more else None
        return EnumerateVersionsResponse(
            items=entries,
            next_continuation_handle=next_handle,
            versions_order=versions_order_mapping[versions_order],
        )
    except ValueError as e:
        # Re-raise ValueError as 400 if it's from invalid continuation_handle
        raise HTTPException(status_code=400, detail=str(e)) from e

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Metadata service implementation for REST.

This module implements the MetadataService from the Storage API, which allows
storing and retrieving user-defined metadata key-value pairs for resources.
This is separate from system metadata (size, modification time) and enables
applications to attach custom properties like tags, descriptions, or any
JSON-serializable data to files and folders via HTTP endpoints.

The service supports optimistic concurrency control via ETags to prevent
concurrent modification conflicts.
"""

import asyncio
import json
from json import JSONDecodeError

from fastapi import (
    Header,
    HTTPException,
    Request,
    Response,
)
from starlette import status

from omni_onedrive_storage_service.backends import (
    EtagMismatchError,
    MetadataKeyNotFoundError,
    get_backend,
)

from .rest_messages import (
    METADATA_ETAG_TYPE,
    METADATA_KEY_TYPE,
    METADATA_URI_TYPE,
    HTTPValidationError,
    UserMetadataKeys,
    UserMetadataResponse,
    UserMetadataValue,
)
from .routes import metadata_service


@metadata_service.post(
    "/{uri:path}",
    responses={
        status.HTTP_200_OK: {
            "description": "Successful Response, data in body.",
            "content": {"application/json": {"schema": UserMetadataResponse.model_json_schema()}},
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "content": {"application/json": {"schema": HTTPValidationError.model_json_schema()}}
        },
    },
)
async def get_metadata(
    uri: METADATA_URI_TYPE,
    keys: UserMetadataKeys,
) -> UserMetadataResponse:
    """Retrieve user-defined metadata key-value pairs for a resource.

    The Metadata API allows storing custom key-value data associated with
    resources (files or folders). This is separate from system metadata
    (size, modification time) and is useful for application-specific data
    like tags, descriptions, custom properties, etc.

    Args:
        uri: Resource address or identity to get metadata for
        keys: UserMetadataKeys object specifying which keys to retrieve
              If keys list is empty, returns all metadata

    Returns:
        UserMetadataResponse mapping key names to UserMetadataValue objects
        Each value includes the value itself and an etag for optimistic locking

    Raises:
        HTTPException(400): If URI is invalid
        HTTPException(500): On internal errors

    Note:
        Values are stored as JSON. Non-JSON strings are returned as-is.
        ETags can be used with update_metadata() to prevent concurrent modifications.

    Example:
        POST /v1alpha/metadata/myfile.txt
        Body: {\"keys\": [\"tag\", \"description\"]}
        Response: {\"tag\": {\"value\": \"important\", \"etag\": \"abc\"}, ...}
    """
    backend = get_backend()

    # Validate that the URI is a valid resource address or identity
    if not await asyncio.to_thread(backend.is_address_valid, uri):
        # Try to validate as identity
        try:
            await asyncio.to_thread(backend.stat_identity, uri)
        except (ValueError, FileNotFoundError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid resource address or identity: {uri}"
            ) from None

    try:
        metadata = await asyncio.to_thread(backend.get_metadata, uri, keys)

        rest_metadata = {}
        for key, value_info in metadata.items():
            try:
                parsed_value = json.loads(value_info["value"])
            except (JSONDecodeError, ValueError):
                parsed_value = value_info["value"]

            rest_metadata[key] = UserMetadataValue(value=parsed_value, etag=value_info["etag"])

        return UserMetadataResponse(rest_metadata)

    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)) from err


@metadata_service.put(
    "/{uri:path}/{key}",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {
            "description": "Successful Response",
            "headers": {
                "ETag": {
                    "description": "The ETag of the updated metadata key.",
                    "required": True,
                    "schema": {"type": "string", "format": "etag", "maxLength": 256},
                }
            },
        },
        status.HTTP_412_PRECONDITION_FAILED: {
            "description": "Precondition Failed",
            "content": {"application/json": {"schema": HTTPValidationError.model_json_schema()}},
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "content": {"application/json": {"schema": HTTPValidationError.model_json_schema()}}
        },
    },
)
async def update_metadata(
    uri: METADATA_URI_TYPE,
    key: METADATA_KEY_TYPE,
    request: Request,
    if_match: METADATA_ETAG_TYPE | None = Header(None, alias="If-Match"),
) -> Response:
    """Create or update a user-defined metadata key-value pair.

    Sets or updates a metadata key for a resource. Supports conditional
    updates via ETags to prevent lost updates in concurrent scenarios.

    Args:
        uri: Resource address or identity
        key: Metadata key name
        request: Request with JSON body containing the value
                Body can be: {\"value\": <any JSON>} or just <any JSON>
        if_match: Optional ETag for conditional update (If-Match header)
                 If provided, update only succeeds if current ETag matches

    Returns:
        Response (201 Created) with ETag header containing new ETag

    Raises:
        HTTPException(400): If URI/key invalid or JSON malformed
        HTTPException(412): If ETag doesn't match (precondition failed)
        HTTPException(500): On internal errors

    Note:
        Values are serialized as JSON. To update conditionally, first call
        get_metadata() to get current ETag, then include it in If-Match header.

    Example (unconditional):
        PUT /v1alpha/metadata/myfile.txt/tag
        Body: {\"value\": \"important\"}
        Response (201): Headers: ETag: \"xyz123\"

    Example (conditional):
        PUT /v1alpha/metadata/myfile.txt/tag
        Headers: If-Match: \"xyz123\"
        Body: {\"value\": \"very-important\"}
        Response (201) if ETag matches, (412) if not
    """
    backend = get_backend()

    # Validate that the URI is a valid resource address or identity
    if not await asyncio.to_thread(backend.is_address_valid, uri):
        # Try to validate as identity
        try:
            await asyncio.to_thread(backend.stat_identity, uri)
        except (ValueError, FileNotFoundError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid resource address or identity: {uri}"
            ) from None

    try:
        body = await request.body()
        request_data = json.loads(body.decode())

        value = request_data["value"] if isinstance(request_data, dict) and "value" in request_data else request_data

        value_str = json.dumps(value, sort_keys=True, separators=(",", ":"))

        new_etag = await asyncio.to_thread(backend.update_metadata, uri, key, value_str, if_match)

        return Response(status_code=status.HTTP_201_CREATED, headers={"ETag": new_etag})

    except JSONDecodeError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON") from err
    except EtagMismatchError as err:
        raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED, detail=str(err)) from err
    except (ValueError, MetadataKeyNotFoundError) as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)) from err


@metadata_service.delete(
    "/{uri:path}/{key}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Successful Response"},
        status.HTTP_412_PRECONDITION_FAILED: {
            "description": "Precondition Failed",
            "content": {"application/json": {"schema": HTTPValidationError.model_json_schema()}},
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "content": {"application/json": {"schema": HTTPValidationError.model_json_schema()}}
        },
    },
)
async def delete_metadata(
    uri: METADATA_URI_TYPE,
    key: METADATA_KEY_TYPE,
    if_match: METADATA_ETAG_TYPE | None = Header(None, alias="If-Match"),
) -> Response:
    """Delete a user-defined metadata key-value pair.

    Removes a metadata key from a resource. Supports conditional deletion
    via ETags to prevent accidental deletion in concurrent scenarios.

    Args:
        uri: Resource address or identity
        key: Metadata key name to delete
        if_match: Optional ETag for conditional deletion (If-Match header)
                 If provided, delete only succeeds if current ETag matches

    Returns:
        Response (204 No Content) on successful deletion

    Raises:
        HTTPException(400): If URI/key invalid or key doesn't exist
        HTTPException(412): If ETag doesn't match (precondition failed)
        HTTPException(500): On internal errors

    Note:
        Idempotent operation - deleting a non-existent key returns success.
        For conditional deletion, include the current ETag in If-Match header.

    Example (unconditional):
        DELETE /v1alpha/metadata/myfile.txt/tag
        Response (204)

    Example (conditional):
        DELETE /v1alpha/metadata/myfile.txt/tag
        Headers: If-Match: \"xyz123\"
        Response (204) if ETag matches, (412) if not
    """
    backend = get_backend()

    # Validate that the URI is a valid resource address or identity
    if not await asyncio.to_thread(backend.is_address_valid, uri):
        # Try to validate as identity
        try:
            await asyncio.to_thread(backend.stat_identity, uri)
        except (ValueError, FileNotFoundError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid resource address or identity: {uri}"
            ) from None

    try:
        await asyncio.to_thread(backend.delete_metadata, uri, key, if_match)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    except EtagMismatchError as err:
        raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED, detail=str(err)) from err
    except (ValueError, MetadataKeyNotFoundError) as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)) from err

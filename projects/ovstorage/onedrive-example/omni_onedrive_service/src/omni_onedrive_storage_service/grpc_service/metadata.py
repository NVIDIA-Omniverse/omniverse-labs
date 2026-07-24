# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Metadata service implementation for gRPC.

This module implements the MetadataService from the Storage API, which allows
storing and retrieving user-defined metadata key-value pairs for resources.
This is separate from system metadata (size, modification time) and enables
applications to attach custom properties like tags, descriptions, or any
JSON-serializable data to files and folders.

The service supports optimistic concurrency control via ETags to prevent
concurrent modification conflicts.
"""

import json

import grpc
from google.protobuf.struct_pb2 import (
    ListValue,
    NullValue,
    Struct,
    Value,
)

from nvidia.omniverse.storage.metadata.v1alpha.metadata_pb2 import (
    DeleteMetadataRequest,
    DeleteMetadataResponse,
    GetMetadataRequest,
    GetMetadataResponse,
    UpdateMetadataRequest,
    UpdateMetadataResponse,
    UserMetadataValue,
)
from nvidia.omniverse.storage.metadata.v1alpha.metadata_pb2_grpc import (
    MetadataServiceServicer,
)
from omni_onedrive_storage_service.backends import (
    EtagMismatchError,
    MetadataKeyNotFoundError,
    get_backend,
)


def _decode_value(value: Value):
    """Convert a protobuf Value to a Python native type.

    Recursively converts protobuf Value messages (which can represent any
    JSON type) into corresponding Python objects.

    Args:
        value: protobuf Value message to decode.

    Returns:
        Python native type (bool, float, str, list, dict, or None).
    """
    value_type = value.WhichOneof("kind")
    if value_type == "bool_value":
        return value.bool_value
    if value_type == "number_value":
        return value.number_value
    if value_type == "string_value":
        return value.string_value
    if value_type == "list_value":
        return [_decode_value(item) for item in value.list_value.values]
    if value_type == "struct_value":
        return {key: _decode_value(item) for key, item in value.struct_value.fields.items()}
    return None


def _encode_value(value):
    """Convert a Python native type to a protobuf Value.

    Recursively converts Python objects into protobuf Value messages that
    can represent any JSON type.

    Args:
        value: Python object to encode (bool, int, float, str, list, dict, or None).

    Returns:
        protobuf Value message.

    Note:
        Integers are converted to floats as per JSON/protobuf semantics.
    """
    if isinstance(value, bool):
        return Value(bool_value=value)
    if isinstance(value, int):
        return Value(number_value=float(value))
    if isinstance(value, float):
        return Value(number_value=value)
    if isinstance(value, str):
        return Value(string_value=value)
    if isinstance(value, list):
        return Value(list_value=ListValue(values=[_encode_value(item) for item in value]))
    if isinstance(value, dict):
        return Value(struct_value=Struct(fields={key: _encode_value(item) for key, item in value.items()}))
    return Value(null_value=NullValue.NULL_VALUE)


def _validate_uri(uri: str, context) -> None:
    """Validate that a URI is either a valid resource address or identity.

    Args:
        uri: Resource address or identity to validate
        context: gRPC ServicerContext for aborting on error

    Raises:
        grpc.RpcError: Aborts with INVALID_ARGUMENT if URI is invalid
    """
    backend = get_backend()

    # Try as address first
    if backend.is_address_valid(uri):
        return

    # If not a valid address, try as identity
    try:
        backend.address_from_identity(uri)
    except ValueError:
        # Identity parsing failed - URI is invalid
        context.abort(
            grpc.StatusCode.INVALID_ARGUMENT, f"Invalid resource URI (not a valid address or identity): {uri}"
        )


class FilesystemMetadataService(MetadataServiceServicer):
    """gRPC servicer for user-defined metadata operations.

    Implements the MetadataService interface from the Storage API v1alpha,
    providing operations to get, update, and delete custom metadata associated
    with resources (files or folders).

    Metadata values are stored as JSON and support ETags for optimistic
    concurrency control.
    """

    def GetMetadata(self, request: GetMetadataRequest, context):
        """Retrieve user-defined metadata key-value pairs for a resource.

        Gets the specified metadata keys (or all keys if none specified) for
        a resource. Each value includes an ETag for use in conditional updates.

        Args:
            request: GetMetadataRequest containing:
                    - uri: Resource address or identity
                    - user_metadata_keys: List of keys to retrieve (empty = all keys)
            context: gRPC ServicerContext for the request.

        Returns:
            GetMetadataResponse mapping key names to UserMetadataValue objects,
            each containing the value (as protobuf Value) and an etag.

        Raises:
            grpc.RpcError with:
            - INVALID_ARGUMENT: If URI is invalid
            - INTERNAL: On unexpected errors

        Note:
            Values are stored as JSON. Non-JSON strings are returned as-is.
        """
        _validate_uri(request.uri, context)

        try:
            metadata = get_backend().get_metadata(request.uri, list(request.user_metadata_keys))

            # Convert to protobuf format
            proto_metadata = {}
            for key, value_info in metadata.items():
                # Parse the stored JSON value back to Python type
                try:
                    parsed_value = json.loads(value_info["value"])
                except (json.JSONDecodeError, ValueError):
                    # If not valid JSON, use as string
                    parsed_value = value_info["value"]

                proto_metadata[key] = UserMetadataValue(value=_encode_value(parsed_value), etag=value_info["etag"])

            return GetMetadataResponse(user_metadata=proto_metadata)

        except ValueError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))

    def UpdateMetadata(self, request: UpdateMetadataRequest, context):
        """Create or update a user-defined metadata key-value pair.

        Sets or updates a metadata key for a resource. Supports conditional
        updates via ETags to prevent lost updates in concurrent scenarios.

        Args:
            request: UpdateMetadataRequest containing:
                    - uri: Resource address or identity
                    - user_metadata_key: Key name to update
                    - user_metadata: protobuf Value containing the new value
                    - expected_etag: Optional ETag for conditional update
            context: gRPC ServicerContext for the request.

        Returns:
            UpdateMetadataResponse containing the new etag for the key.

        Raises:
            grpc.RpcError with:
            - INVALID_ARGUMENT: If URI/key invalid or value can't be serialized
            - FAILED_PRECONDITION: If ETag doesn't match (precondition failed)
            - INTERNAL: On unexpected errors

        Note:
            Values are serialized as JSON. To update conditionally, first call
            GetMetadata to get current ETag, then include it in expected_etag.
        """
        _validate_uri(request.uri, context)

        try:
            python_value = _decode_value(request.user_metadata)

            value_str = json.dumps(python_value, sort_keys=True, separators=(",", ":"))

            expected_etag = request.expected_etag or None

            new_etag = get_backend().update_metadata(request.uri, request.user_metadata_key, value_str, expected_etag)

            return UpdateMetadataResponse(etag=new_etag)

        except EtagMismatchError as e:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
        except (ValueError, MetadataKeyNotFoundError) as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))

    def DeleteMetadata(self, request: DeleteMetadataRequest, context):
        """Delete a user-defined metadata key-value pair.

        Removes a metadata key from a resource. Supports conditional deletion
        via ETags to prevent accidental deletion in concurrent scenarios.

        Args:
            request: DeleteMetadataRequest containing:
                    - uri: Resource address or identity
                    - user_metadata_key: Key name to delete
                    - expected_etag: Optional ETag for conditional deletion
            context: gRPC ServicerContext for the request.

        Returns:
            DeleteMetadataResponse (empty message).

        Raises:
            grpc.RpcError with:
            - INVALID_ARGUMENT: If URI/key invalid or key doesn't exist
            - FAILED_PRECONDITION: If ETag doesn't match (precondition failed)
            - INTERNAL: On unexpected errors

        Note:
            For conditional deletion, include the current ETag in expected_etag.
        """
        _validate_uri(request.uri, context)

        try:
            expected_etag = request.expected_etag or None

            get_backend().delete_metadata(request.uri, request.user_metadata_key, expected_etag)

            return DeleteMetadataResponse()

        except EtagMismatchError as e:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
        except (ValueError, MetadataKeyNotFoundError) as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))

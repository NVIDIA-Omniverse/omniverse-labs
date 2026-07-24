# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from google.protobuf.timestamp_pb2 import Timestamp

from omni_onedrive_storage_service.backends import (
    get_backend,
)
from omni_onedrive_storage_service.backends.storage_backend_interface import (
    ListEntry,
)
from omni_onedrive_storage_service.backends.storage_backend_interface import (
    Metadata as BackendMetadata,
)


def _to_proto_metadata(fileobject_version, backend_metadata: BackendMetadata):
    """Convert backend Metadata to protobuf Metadata."""
    epoch = backend_metadata.last_modified_timestamp.timestamp()
    seconds = int(epoch)
    nanos = int((epoch - seconds) * 1_000_000_000)
    return fileobject_version.Metadata(
        data_object_size=backend_metadata.data_object_size,
        last_modified_timestamp=Timestamp(seconds=seconds, nanos=nanos),
    )


def resource_info(fileobject_version, resource_address: str):
    """Create a ResourceInfo protobuf message for a resource address.

    Constructs a complete ResourceInfo message containing both the resource
    identity (opaque identifier) and metadata (size, modification time) for
    the latest version at the given resource address.

    Args:
        fileobject_version: The fileobject protobuf module (v1alpha or v1beta)
                           containing ResourceInfo and related message types.
        resource_address: Storage API resource address (e.g., 'file-storage://server/path').

    Returns:
        A ResourceInfo protobuf message containing:
        - resource_identity: Opaque encoded identity for the latest version
        - metadata: File size and modification timestamp

    Note:
        This automatically resolves to the latest version of the resource.
    """
    backend = get_backend()
    resource_identity = fileobject_version.ResourceIdentity(
        encoded_identity=backend.create_identity_from_resource_address(resource_address)
    )
    version_info = backend.stat(resource_address)
    metadata = _to_proto_metadata(fileobject_version, version_info.metadata)
    return fileobject_version.ResourceInfo(resource_identity=resource_identity, metadata=metadata)


def resource_info_from_entry(fileobject_version, entry: ListEntry):
    """Create a ResourceInfo protobuf message from a ListEntry.

    Used by ListStat to avoid redundant per-file stat() and
    create_identity_from_resource_address() calls when the backend's
    list_stat() already provides the necessary data.
    """
    identity = None
    if entry.resource_identity:
        identity = fileobject_version.ResourceIdentity(
            encoded_identity=entry.resource_identity,
        )
    metadata = None
    if entry.metadata:
        metadata = _to_proto_metadata(fileobject_version, entry.metadata)
    return fileobject_version.ResourceInfo(resource_identity=identity, metadata=metadata)

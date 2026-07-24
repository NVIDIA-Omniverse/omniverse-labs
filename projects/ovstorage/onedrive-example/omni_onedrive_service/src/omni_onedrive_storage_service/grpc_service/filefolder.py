# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""FileFolderService implementation for gRPC.

This module implements the FileFolderService from the Storage API, which
provides operations for managing folders and listing their contents:
- List: Get immediate children of a folder (addresses only)
- ListStat: Get immediate children with full metadata
- CreateFolder: Create a new folder
- DeleteFolder: Delete an empty folder
- GetFolderMode: Query the folder simulation mode

The behavior adapts to different folder simulation modes (native, no_empty,
placeholder) to support various storage backend semantics.
"""

import grpc

from omni_onedrive_storage_service.backends import get_backend
from omni_onedrive_storage_service.backends.path_helpers import (
    build_address,
    get_relative_path_from_address,
)
from omni_onedrive_storage_service.backends.storage_backend_interface import FolderMode
from omni_onedrive_storage_service.grpc_service.message_helpers import (
    resource_info_from_entry,
)


def _build_child_addresses(folder_uri: str, subfolder_names: list, file_names: list) -> tuple[list[str], list[str]]:
    """Convert child names to full, URL-encoded child addresses."""
    base_uri = get_backend().base_uri
    parent_path = get_relative_path_from_address(base_uri, folder_uri)
    return (
        [build_address(base_uri, parent_path, child=name) for name in subfolder_names],
        [build_address(base_uri, parent_path, child=name) for name in file_names],
    )


def make_filefolder_service_servicer(fileobject_version, pb2_version, pb2_grpc_version, is_alpha):
    """Create a dynamic FileFolderService servicer for gRPC.

    This factory function creates a gRPC servicer class that implements the
    FileFolderService interface. It uses dynamic type construction to adapt
    to different API versions.

    Args:
        fileobject_version: The fileobject protobuf module containing common
                           message types (ResourceInfo, Metadata, etc.).
        pb2_version: The filefolder protobuf module containing service-specific
                    message types (ListResponse, CreateFolderResponse, etc.).
        pb2_grpc_version: The filefolder gRPC module containing the servicer
                         base class.
        is_alpha: Boolean indicating if this is v1alpha (True) or v1beta (False).
                 v1alpha includes CreateFolder and GetFolderMode methods;
                 v1beta only includes List, ListStat, and DeleteFolder.

    Returns:
        An instance of a dynamically created FileFolderServiceServicer class
        with all folder operation methods implemented for that version.
    """

    def List(self, request, context):
        """List immediate children of a folder (addresses only).

        Returns the immediate children of the specified folder, separated into
        subfolders and files, without metadata. This is a lightweight operation
        compared to ListStat.

        Args:
            request: ListRequest containing:
                    - folder: FolderAddress with uri field
            context: gRPC ServicerContext for the request.

        Yields:
            ListResponse containing:
            - subfolder_addresses: List of child folder addresses
            - sub_resource_addresses: List of child file addresses

        Raises:
            grpc.RpcError with:
            - INVALID_ARGUMENT: If folder address is invalid or is a versioned address
            - NOT_FOUND: If folder doesn't exist or address points to a file

        Note:
            This is non-recursive (only immediate children). For recursive
            listing, use the Enumerate method in FileObjectService.
        """
        if not get_backend().is_address_valid(request.folder.uri):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                details=f"invalid resource_address {request.folder.uri}",
            )
        if get_backend().is_version_address(request.folder.uri):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                details="list cannot be called for versioned addresses",
            )
        if get_backend().is_file(request.folder.uri):
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                details=f"{request.folder.uri} is not a folder",
            )
        if not get_backend().exists(request.folder.uri):
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                details=f"{request.folder.uri} does not exist",
            )

        subfolder_names, file_names = get_backend().list(request.folder.uri)
        subfolder_addresses, file_addresses = _build_child_addresses(request.folder.uri, subfolder_names, file_names)

        sub_folders = [pb2_version.FolderAddress(uri=x) for x in subfolder_addresses]
        yield pb2_version.ListResponse(subfolder_addresses=sub_folders, sub_resource_addresses=file_addresses)

    def ListStat(self, request, context):
        """List folder contents with metadata in a single operation.

        Combines listing and stat operations for efficiency - returns immediate
        children of a folder along with their metadata (size, modification time,
        resource identity). More efficient than calling List then Stat for each item.

        Args:
            request: ListStatRequest containing:
                    - folder: FolderAddress with uri field
            context: gRPC ServicerContext for the request.

        Yields:
            ListStatResponse containing:
            - subfolder_addresses: List of child folder addresses
            - entries: List of ListItem messages with address, identity, and metadata

        Raises:
            grpc.RpcError with:
            - INVALID_ARGUMENT: If folder address is invalid or is a versioned address
            - NOT_FOUND: If folder doesn't exist or address points to a file

        Note:
            Preferred over List when you need metadata, as it avoids round-trips.
            Particularly useful for UI applications displaying file sizes/timestamps.
        """
        if not get_backend().is_address_valid(request.folder.uri):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                details=f"invalid resource_address {request.folder.uri}",
            )
        if get_backend().is_version_address(request.folder.uri):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                details="liststat cannot be called for versioned addresses",
            )
        if get_backend().is_file(request.folder.uri):
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                details=f"{request.folder.uri} is not a folder",
            )
        if not get_backend().exists(request.folder.uri):
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                details=f"{request.folder.uri} does not exist",
            )

        subfolder_names, file_entries = get_backend().list_stat(request.folder.uri)
        subfolder_addresses, _ = _build_child_addresses(request.folder.uri, subfolder_names, [])

        sub_folders = [pb2_version.FolderAddress(uri=x) for x in subfolder_addresses]
        base_uri = get_backend().base_uri
        parent_path = get_relative_path_from_address(base_uri, request.folder.uri)
        entries = []
        for entry in file_entries:
            entries.append(
                pb2_version.ListItem(
                    resource_address=build_address(base_uri, parent_path, child=entry.resource_address),
                    resource_info=resource_info_from_entry(fileobject_version, entry),
                )
            )
        yield pb2_version.ListStatResponse(subfolder_addresses=sub_folders, entries=entries)

    def DeleteFolder(self, request, context):
        """Delete an empty folder.

        Removes a folder if it is empty. This operation is idempotent - deleting
        a non-existent folder succeeds (returns empty response).

        Args:
            request: DeleteFolderRequest containing:
                    - folder: FolderAddress with uri field
            context: gRPC ServicerContext for the request.

        Returns:
            DeleteFolderResponse (empty message).

        Raises:
            grpc.RpcError with:
            - INVALID_ARGUMENT: If folder address is invalid or is not a folder
            - FAILED_PRECONDITION: If folder is not empty

        Note:
            Only empty folders can be deleted. To delete a folder tree, first
            delete all files and subfolders recursively.
        """
        if not get_backend().is_address_valid(request.folder.uri):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                details=f"invalid resource_address {request.folder.uri}",
            )
        if not get_backend().exists(request.folder.uri):
            return pb2_version.DeleteFolderResponse()
        if get_backend().is_dir(request.folder.uri):
            if not get_backend().remove_empty_folder(request.folder.uri):
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    details=f"Failed to remove {request.folder.uri}, folder not empty",
                )
            return pb2_version.DeleteFolderResponse()
        else:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                details=f"Failed to remove {request.folder.uri}, is not a folder",
            )

    def CreateFolder(self, request, context):
        """Create a folder at the given address.

        This operation is idempotent - creating a folder that already exists
        succeeds. The behavior depends on the folder simulation mode:
        - native: Creates actual filesystem directory
        - no_empty: No-op (folders don't exist until they contain files)
        - placeholder: Creates marker file to simulate folder

        Args:
            request: CreateFolderRequest containing:
                    - folder: FolderAddress with uri field
            context: gRPC ServicerContext for the request.

        Returns:
            CreateFolderResponse (empty message).

        Raises:
            grpc.RpcError with:
            - INVALID_ARGUMENT: If folder address is invalid
            - FAILED_PRECONDITION: If a file already exists at this address
            - PERMISSION_DENIED: If permission denied
            - INTERNAL: If folder creation fails for other reasons

        Note:
            Folder-mode behavior depends on the active backend's declared folder
            mode (OneDrive uses ``native`` — real folders).
        """
        if not get_backend().is_address_valid(request.folder.uri):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                details=f"invalid resource_address {request.folder.uri}",
            )

        if get_backend().exists(request.folder.uri) and get_backend().is_file(request.folder.uri):
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                details="Conflict - resource already exists as a file",
            )

        try:
            get_backend().create_folder(request.folder.uri)
            return pb2_version.CreateFolderResponse()
        except FileExistsError:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                details="Conflict - resource already exists as a file",
            )
        except PermissionError:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, details="Permission denied")
        except OSError as e:
            context.abort(grpc.StatusCode.INTERNAL, details=f"Failed to create folder: {e!s}")

    def GetFolderMode(self, request, context):
        """Get the folder simulation mode used by this storage service.

        Different storage backends handle folders differently:
        - NATIVE: Real filesystem directories (like Nucleus or local filesystem)
        - NO_EMPTY: Empty folders don't exist (like AWS S3)
        - HYBRID: Uses placeholder files to simulate folders

        Args:
            request: GetFolderModeRequest containing:
                    - folder: FolderAddress with uri field to check
            context: gRPC ServicerContext for the request.

        Returns:
            GetFolderModeResponse containing the folder_mode enum value.

        Raises:
            grpc.RpcError with:
            - INVALID_ARGUMENT: If folder address is invalid
            ValueError: If service is using unknown folder mode.

        Note:
            Knowing the folder mode helps clients understand whether:
            - Empty folders can exist
            - Folder creation is required before writing files
            - Listing empty folders is supported

            The folder mode is typically consistent across an entire storage
            backend, but the address is validated to ensure it's a valid location.
        """
        # Validate the folder address
        if not get_backend().is_address_valid(request.folder.uri):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                details=f"invalid resource_address {request.folder.uri}",
            )

        # Get the folder mode from the backend
        folder_mode = get_backend().folder_mode()
        if folder_mode == FolderMode.NATIVE:
            return pb2_version.GetFolderModeResponse(folder_mode=pb2_version.FolderMode.FOLDER_MODE_NATIVE)
        elif folder_mode == FolderMode.NO_EMPTY:
            return pb2_version.GetFolderModeResponse(folder_mode=pb2_version.FolderMode.FOLDER_MODE_NO_EMPTY)
        elif folder_mode == FolderMode.HYBRID:
            return pb2_version.GetFolderModeResponse(folder_mode=pb2_version.FolderMode.FOLDER_MODE_HYBRID)
        else:
            raise ValueError(f"Unsupported folder mode: {folder_mode}")

    if is_alpha:
        cls = type(
            "FileFolderService",
            (pb2_grpc_version.FileFolderServiceServicer,),
            {
                "CreateFolder": CreateFolder,
                "GetFolderMode": GetFolderMode,
                "List": List,
                "ListStat": ListStat,
                "DeleteFolder": DeleteFolder,
            },
        )
    else:
        cls = type(
            "FileFolderService",
            (pb2_grpc_version.FileFolderServiceServicer,),
            {
                "List": List,
                "ListStat": ListStat,
                "DeleteFolder": DeleteFolder,
            },
        )
    return cls()

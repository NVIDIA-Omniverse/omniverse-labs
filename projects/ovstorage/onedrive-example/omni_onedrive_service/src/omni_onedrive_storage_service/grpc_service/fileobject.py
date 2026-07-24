# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""FileObjectService implementation for gRPC.

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
- Body: Small files streamed directly in the gRPC request
- Redirect: Medium files uploaded via HTTP redirect
- Multipart: Large files uploaded in multiple parts

Version management is automatic - each write creates a new immutable version.
"""

import contextlib
import os
import uuid
from urllib.parse import quote_plus

import grpc
from google.protobuf.timestamp_pb2 import Timestamp

from nvidia.omniverse.storage.fileobject.v1alpha import (
    fileobject_pb2 as fileobject_pb2_v1alpha,
)
from nvidia.omniverse.storage.fileobject.v1alpha import (
    fileobject_service_pb2 as fileobject_service_pb2_v1alpha,
)
from omni_onedrive_storage_service.backends import MAX_UPLOAD_SIZE_BYTES, get_backend
from omni_onedrive_storage_service.backends.storage_backend_interface import FolderMode
from omni_onedrive_storage_service.grpc_service.message_helpers import (
    _to_proto_metadata,
    resource_info,
)


def check_optimistic_locking_support(context, operation_name: str = "This operation"):
    """Check if the current backend supports optimistic locking (previous_version).

    Args:
        context: gRPC context to abort with error if not supported
        operation_name: Name of the operation for error message

    Raises:
        grpc.RpcError with UNIMPLEMENTED if backend doesn't support optimistic locking
    """
    backend = get_backend()
    support = backend.get_optimistic_locking_support()
    # Check if any optimistic locking is supported
    if not (support.write or support.delete or support.copy or support.move):
        context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            details=f"{operation_name} with previous_version parameter is not supported by {type(backend).__name__} backend. "
            f"Optimistic locking is only available with backends that support it.",
        )


def abort_on_previous_version_not_latest(resource_address, previous_version: str, context):
    """Validate that a specified version is still the latest version.

    This implements optimistic concurrency control for write operations. When
    a client specifies a previous_version (expected current state), this
    function verifies that version is still the latest. If not, it aborts
    the gRPC request with FAILED_PRECONDITION, preventing lost updates.

    Args:
        resource_address: Storage API resource address to check.
        previous_version: Encoded identity string of the expected latest version.
        context: gRPC ServicerContext for aborting the request.

    Raises:
        grpc.RpcError: Aborts with FAILED_PRECONDITION if:
            - The specified version is not the latest version
            - No version exists at the resource address

    Note:
        This is used in Write, Delete, Copy, and Move operations to ensure
        the client's view of the resource state is current.
    """
    backend = get_backend()
    # Check that the specified version is still the latest. is_version_latest
    # handles both concrete version ids and the "current" sentinel used by
    # list_stat entries (avoids spurious conflicts).
    try:
        if not backend.is_version_latest(resource_address, previous_version):
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                details="specified previous version is not latest version",
            )
    except FileNotFoundError:
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            details="specified previous version, but no version was found at the address",
        )


def make_fileobject_service_servicer(
    REDIRECT_HOST: str,
    fileobject_version,
    pb2_version,
    pb2_grpc_version,
    redirect_port: int,
    version_tag: str,
):
    """Create a dynamic FileObjectService servicer for gRPC.

    This factory function creates a gRPC servicer class that implements the
    FileObjectService interface with all core file operations. It adapts to
    different API versions (v1alpha includes Copy/Move, v1beta does not).

    Args:
        REDIRECT_HOST: Hostname for constructing redirect URLs (e.g., 'http://localhost').
        fileobject_version: The fileobject protobuf module containing message types.
        pb2_version: The fileobject service protobuf module containing request/response types.
        pb2_grpc_version: The fileobject gRPC module containing the servicer base class.
        redirect_port: Port number for HTTP redirect endpoints.
        version_tag: API version string ('v1alpha' or 'v1beta').
                    - v1alpha includes Copy and Move methods
                    - v1beta includes only stable operations

    Returns:
        An instance of a dynamically created FileObjectServiceServicer class
        with all file operation methods implemented for the specified version.
    """

    def Enumerate(self, request, context):
        """Recursively enumerate all files under a directory tree.

        Lists all files (not folders) recursively within the specified directory.
        Results are streamed in batches (yielded responses).

        Args:
            request: EnumerateRequest containing:
                    - resource_address: Directory address to enumerate
            context: gRPC ServicerContext for the request.

        Yields:
            EnumerateResponse messages, each containing a batch of AddressInfo
            items with resource addresses and metadata.

        Raises:
            grpc.RpcError with:
            - INVALID_ARGUMENT: If resource_address is versioned
            - NOT_FOUND: If resource_address doesn't exist or is a file

        Note:
            This is recursive (unlike FileFolderService.List which is non-recursive).
            Large directory trees are streamed in batches for efficiency.
        """
        if get_backend().is_version_address(request.resource_address):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Versioned addresses can not be enumerated.",
            )
        if not get_backend().exists(request.resource_address):
            context.abort(grpc.StatusCode.NOT_FOUND, "Resource address not found")
        if get_backend().is_file(request.resource_address):
            context.abort(grpc.StatusCode.NOT_FOUND, "Cannot enumerate file addresses")

        for entries in get_backend().enumerate(request.resource_address):
            # Each entry is already a ListEntry with resource_address and metadata
            address_infos = []
            for entry in entries:
                # Convert to the gRPC response format
                ai = fileobject_version.AddressInfo(
                    resource_address=entry.resource_address,
                    metadata=(
                        fileobject_version.Metadata(
                            data_object_size=entry.metadata.data_object_size,
                            last_modified_timestamp=entry.metadata.last_modified_timestamp,
                        )
                        if entry.metadata
                        else None
                    ),
                )
                address_infos.append(ai)
            yield pb2_version.EnumerateResponse(items=address_infos)

    def Stat(self, request, context):
        """Get metadata for a file resource.

        Calls backend.stat() once and builds the response directly from the
        result, which already contains the resource identity and metadata.
        """
        if not get_backend().is_address_valid(request.resource_address):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Invalid resource address: {request.resource_address}",
            )
        try:
            version_info = get_backend().stat(request.resource_address)
            identity = fileobject_version.ResourceIdentity(
                encoded_identity=version_info.resource_identity,
            )
            metadata = _to_proto_metadata(fileobject_version, version_info.metadata)
            info = fileobject_version.ResourceInfo(resource_identity=identity, metadata=metadata)
            return pb2_version.StatResponse(resource_info=info)
        except IsADirectoryError as e:
            context.abort(grpc.StatusCode.NOT_FOUND, f"{str(e)} is a directory")
        except PermissionError as e:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, str(e))
        except FileNotFoundError as e:
            context.abort(grpc.StatusCode.NOT_FOUND, str(e))

    def Read(self, request, context):
        """Download file content by resource identity.

        Retrieves a specific file version by its opaque resource identity.
        Supports both direct body streaming and HTTP redirect for downloads.

        Args:
            request: ReadRequest containing:
                    - resource_identity: Opaque identifier for specific version
                    - download_preference: BODY (stream in response) or
                                          REDIRECT (provide HTTP download URL)
            context: gRPC ServicerContext for the request.

        Yields:
            ReadResponse messages containing:
            - First response: metadata (size, modification time)
            - Subsequent responses (if BODY): chunks of file content
            - Or single redirect response (if REDIRECT): download URL

        Raises:
            grpc.RpcError with:
            - INVALID_ARGUMENT: If identity is invalid or download_preference unknown
            - NOT_FOUND: If resource identity doesn't exist
            - PERMISSION_DENIED: If read permission is denied

        Note:
            This reads a specific version by identity (immutable).
            Use ReadFromAddress to always get the latest version.
        """
        backend = get_backend()
        resource_identity_str = request.resource_identity.encoded_identity

        # Use backend's stat_identity to validate and get metadata
        try:
            version_info = backend.stat_identity(resource_identity_str)
        except ValueError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
            return
        except FileNotFoundError as e:
            context.abort(grpc.StatusCode.NOT_FOUND, str(e))
            return
        except (PermissionError, IsADirectoryError) as e:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, str(e))
            return

        # Convert backend metadata to protobuf metadata
        modification_time_epoch = version_info.metadata.last_modified_timestamp.timestamp()
        modification_time_seconds = int(modification_time_epoch)
        nanos = int((modification_time_epoch - modification_time_seconds) * 1000000000)
        timestamp = Timestamp(seconds=modification_time_seconds, nanos=nanos)
        metadata = fileobject_version.Metadata(
            data_object_size=version_info.metadata.data_object_size, last_modified_timestamp=timestamp
        )

        # Yield metadata first
        yield pb2_version.ReadResponse(metadata=metadata)

        # Handle download preference
        if request.download_preference in [
            pb2_version.DownloadPreference.DOWNLOAD_PREFERENCE_BODY,
            pb2_version.DownloadPreference.DOWNLOAD_PREFERENCE_UNSPECIFIED,
        ] or (
            request.download_preference == pb2_version.DownloadPreference.DOWNLOAD_PREFERENCE_REDIRECT
            and not backend.supports_redirect_download()
        ):
            # Stream content using backend's read_from_identity
            try:
                for chunk in backend.read_from_identity(resource_identity_str):
                    yield pb2_version.ReadResponse(chunk=fileobject_version.Chunk(chunk=chunk))
            except (FileNotFoundError, PermissionError) as e:
                context.abort(grpc.StatusCode.PERMISSION_DENIED, str(e))
        elif request.download_preference == pb2_version.DownloadPreference.DOWNLOAD_PREFERENCE_REDIRECT:
            try:
                url = backend.construct_redirect_url_for_identity(resource_identity_str, REDIRECT_HOST, redirect_port)
                yield pb2_version.ReadResponse(
                    redirect=fileobject_version.Redirect(redirect_target_url=url, additional_headers=[])
                )
            except (ValueError, OSError, NotImplementedError) as e:
                context.abort(grpc.StatusCode.INTERNAL, f"Failed to construct redirect URL: {e}")
        else:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"download preference {request.download_preference} is not a valid input",
            )

    def ReadFromAddress(self, request, context):
        """Download file content by resource address (latest version).

        Uses a single stat() call to validate existence, check type, and
        obtain resource identity + metadata, avoiding redundant Graph API calls.
        """
        if not get_backend().is_address_valid(request.resource_address):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Invalid resource address: {request.resource_address}",
            )

        backend = get_backend()
        try:
            version_info = backend.stat(request.resource_address)
        except IsADirectoryError:
            error_code_to_use = (
                grpc.StatusCode.NOT_FOUND
                if backend.folder_mode() == FolderMode.HYBRID
                else grpc.StatusCode.INVALID_ARGUMENT
            )
            context.abort(
                error_code_to_use,
                f"Resource address is folder, can't be read: {request.resource_address}. No regular object found at that address.",
            )
        except FileNotFoundError:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details="Resource not found")
        except PermissionError as e:
            context.abort(code=grpc.StatusCode.PERMISSION_DENIED, details=str(e))

        identity = fileobject_version.ResourceIdentity(
            encoded_identity=version_info.resource_identity,
        )
        metadata = _to_proto_metadata(fileobject_version, version_info.metadata)
        info = fileobject_version.ResourceInfo(resource_identity=identity, metadata=metadata)
        yield pb2_version.ReadFromAddressResponse(resource_info=info)

        try:
            if request.download_preference in [
                pb2_version.DownloadPreference.DOWNLOAD_PREFERENCE_BODY,
                pb2_version.DownloadPreference.DOWNLOAD_PREFERENCE_UNSPECIFIED,
            ] or (
                request.download_preference == pb2_version.DownloadPreference.DOWNLOAD_PREFERENCE_REDIRECT
                and not backend.supports_redirect_download()
            ):
                for chunk in backend.read_from_address(request.resource_address):
                    yield pb2_version.ReadFromAddressResponse(chunk=fileobject_version.Chunk(chunk=chunk))
            elif request.download_preference == pb2_version.DownloadPreference.DOWNLOAD_PREFERENCE_REDIRECT:
                url = backend.construct_redirect_url(request.resource_address, REDIRECT_HOST, redirect_port)
                yield pb2_version.ReadFromAddressResponse(
                    redirect=fileobject_version.Redirect(redirect_target_url=url, additional_headers=[])
                )
            else:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"download preference {request.download_preference} is not a valid input",
                )
        except (PermissionError, IsADirectoryError) as e:
            context.abort(code=grpc.StatusCode.PERMISSION_DENIED, details=str(e))

    def _create_multipart_upload(
        resource_address: str, expected_head_version: str | None
    ) -> pb2_version.CreateMultipartUploadResponse:
        """Create a new multipart upload session.

        Initializes a multipart upload by creating a unique upload ID and
        preparing storage for the parts.

        Args:
            resource_address: Destination address for the file being uploaded.
            expected_head_version: Optional expected current version (for optimistic locking).

        Returns:
            CreateMultipartUploadResponse containing:
            - upload_id: Unique identifier for this upload session
            - first_part_write_redirect: Redirect URL for uploading part 0
            - minimum_size_per_part: Minimum bytes per part (0)
            - maximum_size_per_part: Maximum bytes per part (4 MiB)
        """
        backend = get_backend()
        new_upload_id = str(uuid.uuid4())
        backend.create_upload_session(new_upload_id)
        return pb2_version.CreateMultipartUploadResponse(
            upload_id=backend.encode_upload_id(new_upload_id, expected_head_version),
            first_part_write_redirect=_redirect_for_part(new_upload_id, resource_address, 0),
            minimum_size_per_part=0,
            maximum_size_per_part=4 * 1024 * 1024,
        )

    def FetchWriteTypeInfo(self, request, context):
        """Query recommended upload methods for different file sizes.

        Returns size-based recommendations for which upload method (body,
        redirect, or multipart) should be used for files of various sizes.

        For Nucleus backend: Only BODY upload is supported.
        For Filesystem backend: BODY, REDIRECT, and MULTIPART are supported.

        Args:
            request: FetchWriteTypeInfoRequest containing:
                    - destination_resource_address: Target address for upload
            context: gRPC ServicerContext for the request.

        Returns:
            FetchWriteTypeInfoResponse with write_type_intervals based on backend:

            Filesystem backend:
            - 0-1 KiB: BODY (stream in gRPC request)
            - 1 KiB-1 MiB: REDIRECT (upload via HTTP)
            - 1 MiB+: MULTIPART (upload in multiple parts)

            Nucleus backend:
            - All sizes: BODY (stream in gRPC request)

        Raises:
            grpc.RpcError with:
            - INVALID_ARGUMENT: If destination is a versioned address

        Note:
            These are recommendations. Clients can choose different methods
            but should respect the multipart size limits.
        """
        if get_backend().is_version_address(request.destination_resource_address):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Versioned addresses can not be written to.",
            )

        # Check if backend supports redirects and multipart uploads
        backend = get_backend()
        supports_redirects = backend.supports_redirect_upload()
        backend.supports_multipart_upload()

        if supports_redirects:
            # Filesystem backend - supports body, redirect, and multipart
            intervals = [
                pb2_version.WriteTypeForSizeInterval(
                    minimum_data_object_size=0,
                    maximum_data_object_size=1024,
                    preferred_upload_method=pb2_version.UploadPreference.UPLOAD_PREFERENCE_BODY,
                ),
                pb2_version.WriteTypeForSizeInterval(
                    minimum_data_object_size=1024,
                    maximum_data_object_size=1024 * 1024,
                    preferred_upload_method=pb2_version.UploadPreference.UPLOAD_PREFERENCE_REDIRECT,
                ),
                pb2_version.WriteTypeForSizeInterval(
                    minimum_data_object_size=1024 * 1024,
                    maximum_data_object_size=pow(2, 53),
                    preferred_upload_method=pb2_version.UploadPreference.UPLOAD_PREFERENCE_MULTIPART,
                ),
            ]
        else:
            # Nucleus or other backends - only support body upload
            intervals = [
                pb2_version.WriteTypeForSizeInterval(
                    minimum_data_object_size=0,
                    maximum_data_object_size=pow(2, 53),
                    preferred_upload_method=pb2_version.UploadPreference.UPLOAD_PREFERENCE_BODY,
                ),
            ]

        return pb2_version.FetchWriteTypeInfoResponse(write_type_intervals=intervals)

    def Write(self, request_iterator, context):
        """Upload file content to create a new version.

        Handles file uploads using multiple methods:
        - BODY: Small files streamed directly in the request
        - REDIRECT: Medium files uploaded via HTTP redirect
        - MULTIPART: Large files uploaded in multiple parts

        The method is determined by the upload_preference in the first request,
        or auto-selected based on data_object_size.

        Args:
            request_iterator: Iterator of WriteRequest messages:
                - First request: WriteParameters with destination, size, preferences
                - Subsequent requests: Chunk messages with file data (for BODY method)
            context: gRPC ServicerContext for the request.

        Yields:
            WriteResponse messages:
            - For BODY: WriteChunksAccepted, then final resource_info
            - For REDIRECT: WriteRedirectProperties with upload URL
            - For MULTIPART: CreateMultipartUploadResponse

        Raises:
            grpc.RpcError with:
            - INVALID_ARGUMENT: If params invalid, address versioned, or chunks out of order
            - FAILED_PRECONDITION: If previous_version doesn't match current version
            - PERMISSION_DENIED: If write permission denied

        Note:
            Supports optimistic locking via previous_version parameter.
            Each write creates a new immutable version.
        """
        destination_address = None
        upload_preference = pb2_version.UploadPreference.UPLOAD_PREFERENCE_UNSPECIFIED
        path = None
        file = None
        try:
            with contextlib.ExitStack() as exit_stack:
                for request in request_iterator:
                    if request.HasField("params"):
                        if path is not None:
                            context.abort(
                                grpc.StatusCode.INVALID_ARGUMENT,
                                "Supply WriteRequest params only once per upload!",
                            )
                        data_size = request.params.data_object_size
                        if request.params.HasField("upload_preference"):
                            upload_preference = request.params.upload_preference

                        if upload_preference == pb2_version.UploadPreference.UPLOAD_PREFERENCE_UNSPECIFIED:
                            if 0 <= data_size < 1024:
                                # Small file, use chunk method
                                upload_preference = pb2_version.UploadPreference.UPLOAD_PREFERENCE_BODY
                            elif 1024 <= data_size < 1024 * 1024:
                                # Medium file, use redirect method
                                upload_preference = pb2_version.UploadPreference.UPLOAD_PREFERENCE_REDIRECT
                            elif data_size >= 1024 * 1024:
                                # Large file, upload this in parts
                                upload_preference = pb2_version.UploadPreference.UPLOAD_PREFERENCE_MULTIPART
                            else:
                                context.abort(
                                    grpc.StatusCode.INVALID_ARGUMENT,
                                    details="data_object_size invalid",
                                )

                        if request.params.HasField("previous_version"):
                            # Check if backend supports optimistic locking
                            check_optimistic_locking_support(context, "Write operation")

                            # Check that the specified version is still the latest!
                            abort_on_previous_version_not_latest(
                                request.params.destination_resource_address,
                                request.params.previous_version.encoded_identity,
                                context,
                            )

                        if not get_backend().is_address_valid(request.params.destination_resource_address):
                            context.abort(
                                grpc.StatusCode.INVALID_ARGUMENT,
                                "destination address is not valid!",
                            )

                        if get_backend().is_version_address(request.params.destination_resource_address):
                            context.abort(
                                grpc.StatusCode.INVALID_ARGUMENT,
                                f"Cannot write to individual version address: {request.params.destination_resource_address}",
                            )

                        # Check if backend supports redirect/multipart uploads
                        backend = get_backend()
                        supports_redirects = backend.supports_redirect_upload()
                        supports_multipart = backend.supports_multipart_upload()

                        # If client requested redirect/multipart but backend doesn't support it, use body instead
                        if (
                            not supports_redirects
                            and upload_preference == pb2_version.UploadPreference.UPLOAD_PREFERENCE_REDIRECT
                        ):
                            upload_preference = pb2_version.UploadPreference.UPLOAD_PREFERENCE_BODY
                        if (
                            not supports_multipart
                            and upload_preference == pb2_version.UploadPreference.UPLOAD_PREFERENCE_MULTIPART
                        ):
                            upload_preference = pb2_version.UploadPreference.UPLOAD_PREFERENCE_BODY

                        destination_address = request.params.destination_resource_address
                        if upload_preference == pb2_version.UploadPreference.UPLOAD_PREFERENCE_BODY:
                            # For body upload, collect chunks and write to backend
                            if supports_redirects:
                                # Backend supports redirects: use temporary file
                                relative_path = backend.force_relative_path(destination_address)  # ty: ignore[unresolved-attribute]
                                # Use a unique temp file per request to avoid race conditions
                                # when multiple uploads target the same destination concurrently
                                unique_suffix = uuid.uuid4().hex
                                path = str(
                                    os.path.join(
                                        backend._uploads_dir,  # ty: ignore[unresolved-attribute]
                                        "upload_stream",
                                        f"{relative_path}.{unique_suffix}",
                                    )
                                )
                                backend.safe_makedirs(backend.safe_dirname(path), exist_ok=True)  # ty: ignore[unresolved-attribute]
                                file = exit_stack.enter_context(backend.safe_open(path, "wb"))  # ty: ignore[unresolved-attribute]
                            else:
                                # Other backends (e.g., OneDrive): collect in memory.
                                # Reject up front if the declared size already exceeds
                                # the in-memory cap so we fail fast before buffering.
                                if data_size > MAX_UPLOAD_SIZE_BYTES:
                                    context.abort(
                                        grpc.StatusCode.RESOURCE_EXHAUSTED,
                                        details=(
                                            f"Upload size {data_size} exceeds maximum allowed "
                                            f"{MAX_UPLOAD_SIZE_BYTES} bytes"
                                        ),
                                    )
                                file = None
                                file_content = bytearray()
                            yield pb2_version.WriteResponse(write_chunks_accepted=pb2_version.WriteChunksAccepted())
                        elif upload_preference == pb2_version.UploadPreference.UPLOAD_PREFERENCE_REDIRECT:
                            url = f"{REDIRECT_HOST}:{redirect_port}/upload/" + quote_plus(
                                destination_address.replace("\\", "/")
                            )
                            yield pb2_version.WriteResponse(
                                write_redirect=pb2_version.WriteRedirectProperties(
                                    redirect_target_url=url,
                                    method=pb2_version.UploadMethod.UPLOAD_METHOD_POST,
                                    additional_headers=[],
                                    completion_header_names=["x-nvidia-storage-upload-location"],
                                )
                            )
                            return
                        elif upload_preference == pb2_version.UploadPreference.UPLOAD_PREFERENCE_MULTIPART:
                            yield pb2_version.WriteResponse(
                                multipart_upload=_create_multipart_upload(
                                    destination_address,
                                    request.params.previous_version.encoded_identity,
                                )
                            )
                            return

                    elif request.HasField("chunk"):
                        if destination_address is None:
                            context.abort(
                                code=grpc.StatusCode.INVALID_ARGUMENT,
                                details="No WriteParameters message received before Chunk",
                            )
                        if upload_preference == pb2_version.UploadPreference.UPLOAD_PREFERENCE_BODY:
                            if supports_redirects:
                                # FileSystemProvider: write to file
                                if file is None:
                                    context.abort(
                                        code=grpc.StatusCode.INVALID_ARGUMENT,
                                        details="Failed to specify upload metadata first or file could not be written to!",
                                    )
                                    return
                                file.write(request.chunk.chunk)
                            else:
                                # Other backends: accumulate in memory, bounded to
                                # prevent unbounded growth (OOM) from an oversized or
                                # size-misdeclared streaming upload.
                                file_content.extend(request.chunk.chunk)
                                if len(file_content) > MAX_UPLOAD_SIZE_BYTES:
                                    context.abort(
                                        grpc.StatusCode.RESOURCE_EXHAUSTED,
                                        details=(f"Upload exceeds maximum allowed {MAX_UPLOAD_SIZE_BYTES} bytes"),
                                    )
                        else:
                            context.abort(
                                code=grpc.StatusCode.INVALID_ARGUMENT,
                                details="Don't upload chunked data for the given upload method",
                            )
                if upload_preference == pb2_version.UploadPreference.UPLOAD_PREFERENCE_BODY:
                    # Upload complete, determine metadata and resource info and return it
                    if destination_address is None:
                        context.abort(
                            code=grpc.StatusCode.INVALID_ARGUMENT,
                            details="No WriteParameters message received before Chunk",
                        )
                        return

                    if supports_redirects:
                        # Backend supports redirects: read from temp file and write to backend
                        if file is None or path is None:
                            context.abort(
                                code=grpc.StatusCode.INVALID_ARGUMENT,
                                details="Failed to specify upload metadata first or file could not be written to!",
                            )
                            return
                        # Close the file explicitly before reading it back to ensure all data is on disk
                        file.close()
                        try:
                            # Now copy the temporary file into the content and versions directories
                            with backend.safe_open(path, "rb") as f:  # ty: ignore[unresolved-attribute]
                                data = f.read()
                                backend.write_version(destination_address, data)
                        finally:
                            # Clean up the unique temp file
                            if os.path.exists(path):
                                os.unlink(path)
                    else:
                        # Other backends (e.g., Nucleus): write accumulated content directly
                        data = bytes(file_content)
                        get_backend().write_version(destination_address, data)

                    yield pb2_version.WriteResponse(
                        resource_info=resource_info(fileobject_version, destination_address)
                    )
        except (PermissionError, IsADirectoryError, OSError) as e:
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                details=f"Can't write to that resource address: {e}",
            )
        except grpc.RpcError:
            # Re-raise intentional aborts from context.abort() - don't convert to INTERNAL
            raise

    def CompleteRedirectUpload(self, request, context):
        """Complete a redirect-based upload and finalize the new version.

        After a client uploads via HTTP redirect, this method verifies the
        upload completion headers and returns the resource info for the new version.

        Args:
            request: CompleteRedirectUploadRequest containing:
                    - destination_resource_address: Target address
                    - additional_headers: Completion headers from HTTP upload
                                        (must include 'x-nvidia-storage-upload-location')
            context: gRPC ServicerContext for the request.

        Returns:
            CompleteRedirectUploadResponse containing:
            - resource_info: Identity and metadata for the new version

        Raises:
            grpc.RpcError with:
            - UNIMPLEMENTED: If backend doesn't support redirect uploads
            - INVALID_ARGUMENT: If address invalid or required headers missing/invalid
            - NOT_FOUND: If uploaded file cannot be found

        Note:
            The 'x-nvidia-storage-upload-location' header contains the
            version path created by the HTTP upload endpoint.
        """
        # Check if backend supports redirect uploads
        backend = get_backend()
        if not backend.supports_redirect_upload():
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                details=f"Redirect uploads not supported by {type(backend).__name__} backend",
            )

        if not backend.exists(request.destination_resource_address):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                details=f"Invalid argument: {request.destination_resource_address}",
            )

        # Check that the client provides the correct upload headers for this redirect to be valid
        if len(request.additional_headers) == 0:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                details="No additional headers received",
            )

        # Convert headers to dict
        headers_dict = {h.name: h.value for h in request.additional_headers}

        try:
            result = backend.complete_redirect_upload(request.destination_resource_address, headers_dict)

            # Convert to protobuf response
            modification_time_epoch = result.metadata.last_modified_timestamp.timestamp()
            modification_time_seconds = int(modification_time_epoch)
            nanos = int((modification_time_epoch - modification_time_seconds) * 1000000000)
            timestamp = Timestamp(seconds=modification_time_seconds, nanos=nanos)

            metadata = fileobject_version.Metadata(
                data_object_size=result.metadata.data_object_size, last_modified_timestamp=timestamp
            )
            resource_identity = fileobject_version.ResourceIdentity(encoded_identity=result.resource_identity)

            return pb2_version.CompleteRedirectUploadResponse(
                resource_info=fileobject_version.ResourceInfo(resource_identity=resource_identity, metadata=metadata)
            )
        except ValueError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except FileNotFoundError as e:
            context.abort(grpc.StatusCode.NOT_FOUND, str(e))

    def _redirect_for_part(upload_id, destination_resource_address, part_number):
        """Generate redirect properties for uploading a multipart part.

        Args:
            upload_id: Unique upload session identifier.
            destination_resource_address: Target address for the complete file.
            part_number: Zero-based part number.

        Returns:
            WriteRedirectProperties with:
            - redirect_target_url: HTTP URL for uploading this part
            - method: POST
            - completion_header_names: ['local-file'] to be returned after upload
            - additional_headers: Demo headers
        """
        backend = get_backend()
        redirect_props = backend.construct_upload_part_redirect(
            upload_id, part_number, destination_resource_address, REDIRECT_HOST, redirect_port
        )
        return pb2_version.WriteRedirectProperties(
            method=pb2_version.UploadMethod.UPLOAD_METHOD_POST,
            redirect_target_url=redirect_props["redirect_target_url"],
            completion_header_names=redirect_props["completion_header_names"],
            additional_headers=[
                fileobject_version.Header(name=h[0], value=h[1]) for h in redirect_props["additional_headers"]
            ],
        )

    def UploadPart(self, request, context):
        """Get redirect URLs for uploading one or more multipart parts.

        Returns HTTP redirect URLs for uploading the specified part(s) of
        a multipart upload.

        Args:
            request: UploadPartRequest containing:
                    - upload_id: Upload session identifier
                    - destination_resource_address: Target address
                    - part_number: Starting part number
                    - part_count: Optional number of consecutive parts (default 1)
            context: gRPC ServicerContext for the request.

        Returns:
            UploadPartResponse containing:
            - part_write_redirects: List of WriteRedirectProperties,
                                   one for each part

        Note:
            part_count allows batch-requesting redirect URLs for efficiency.
        """
        # Check if backend supports multipart uploads
        backend = get_backend()
        if not backend.supports_multipart_upload():
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                details=f"Multipart uploads not supported by {type(backend).__name__} backend",
            )

        num_parts = 1
        upload_id, _ = backend.decode_upload_id(request.upload_id)
        if request.HasField("part_count"):
            num_parts = request.part_count
        redirects = []
        for p in range(num_parts):
            part_number = request.part_number + p
            redirects.append(_redirect_for_part(upload_id, request.destination_resource_address, part_number))

        return pb2_version.UploadPartResponse(part_write_redirects=redirects)

    def CompleteMultipartUpload(self, request: pb2_version.CompleteMultipartUploadRequest, context):
        """Finalize a multipart upload by assembling all parts.

        Assembles the uploaded parts into the final file, creates a new version,
        and cleans up the upload session.

        Args:
            request: CompleteMultipartUploadRequest containing:
                    - upload_id: Upload session identifier
                    - destination_resource_address: Target address
                    - parts: Ordered list of PartInfo with headers from uploads
            context: gRPC ServicerContext for the request.

        Returns:
            CompleteMultipartUploadResponse containing:
            - resource_info: Identity and metadata for the new version

        Raises:
            grpc.RpcError with:
            - UNIMPLEMENTED: If backend doesn't support multipart uploads
            - INVALID_ARGUMENT: If address invalid, parts out of order, or headers missing
            - FAILED_PRECONDITION: If upload_id doesn't exist or previous_version doesn't match

        Note:
            Parts must be provided in order (part_number 0, 1, 2, ...).
            The upload session directory is cleaned up after completion.
        """
        # Check if backend supports multipart uploads
        backend = get_backend()
        if not backend.supports_multipart_upload():
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                details=f"Multipart uploads not supported by {type(backend).__name__} backend",
            )

        if not backend.is_address_valid(request.destination_resource_address):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
        assembled_data = bytearray()

        upload_id, previous_version = backend.decode_upload_id(request.upload_id)
        if previous_version:
            # Check if backend supports optimistic locking
            check_optimistic_locking_support(context, "Multipart upload")

            # Check that the specified version is still the latest!
            abort_on_previous_version_not_latest(request.destination_resource_address, previous_version, context)

        if not backend.upload_session_exists(upload_id):
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"no multipart upload with id {upload_id} exists",
            )

        # Assemble the parts into the final file
        for i, part in enumerate(request.parts):
            if part.part_number != i:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    details=f"invalid part_number {part.part_number}, expected {i}",
                )
            header_name = "local-file"
            found = False
            for header in part.headers:
                if header.name == header_name:
                    found = True
                    source_file = header.value
                    with backend.safe_open(source_file, "rb") as input_file:  # ty: ignore[unresolved-attribute]
                        data = input_file.read()
                        assembled_data.extend(data)
                    break
            if not found:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"part {part.part_number} is missing the required header {header_name}",
                )

        backend.write_version(request.destination_resource_address, bytes(assembled_data))
        backend.cleanup_upload_session(upload_id)
        return pb2_version.CompleteMultipartUploadResponse(
            resource_info=resource_info(fileobject_version, request.destination_resource_address)
        )

    def AbortMultipartUpload(self, request, context):
        """Cancel a multipart upload and clean up resources.

        Aborts an in-progress multipart upload and removes any uploaded parts.
        This operation is idempotent - aborting a non-existent upload succeeds.

        Args:
            request: AbortMultipartUploadRequest containing:
                    - upload_id: Upload session identifier
            context: gRPC ServicerContext for the request.

        Returns:
            AbortMultipartUploadResponse (empty message).

        Note:
            Always returns success, even if the upload_id doesn't exist.
            This matches the behavior of cloud storage services where you
            cannot inspect abort status.
        """
        # Check if backend supports multipart uploads
        backend = get_backend()
        if not backend.supports_multipart_upload():
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                details=f"Multipart uploads not supported by {type(backend).__name__} backend",
            )

        upload_id, _ = backend.decode_upload_id(request.upload_id)
        backend.cleanup_upload_session(upload_id)
        # Abort always returns success, even if there was nothing to abort. The reason is the hyperscalers do not allow
        # to inspect the status of an abort, so the client can just trust it worked ot it has to inspect the status afterward
        return pb2_version.AbortMultipartUploadResponse()

    def Delete(self, request: pb2_version.DeleteRequest, context):
        """Delete a file and all its versions.

        Removes a file from the storage service, deleting all versions.
        This operation is idempotent - deleting a non-existent file succeeds.

        Args:
            request: DeleteRequest containing:
                    - resource_address: File address to delete
                    - previous_version: Optional expected current version (for optimistic locking)
            context: gRPC ServicerContext for the request.

        Returns:
            DeleteResponse (empty message).

        Raises:
            grpc.RpcError with:
            - INVALID_ARGUMENT: If address is invalid, versioned, or not a file
            - FAILED_PRECONDITION: If previous_version doesn't match current version

        Note:
            - Cannot delete individual versions, only the entire file
            - Cannot delete versioned addresses or folders
            - Supports optimistic locking via previous_version
        """
        if not get_backend().is_address_valid(request.resource_address):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                details=f"invalid resource_address: {request.resource_address}",
            )

        if get_backend().is_version_address(request.resource_address):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                details=f"Cannot delete individual version address: {request.resource_address}",
            )

        if not get_backend().exists(request.resource_address):
            return pb2_version.DeleteResponse()

        if not get_backend().is_file(request.resource_address):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                details=f"Failed to remove {request.resource_address}, not a file",
            )

        if request.HasField("previous_version"):
            # Check if backend supports optimistic locking
            check_optimistic_locking_support(context, "Delete operation")

            # Check that the specified version is still the latest!
            abort_on_previous_version_not_latest(
                request.resource_address,
                request.previous_version.encoded_identity,
                context,
            )

        get_backend().remove_by_address(request.resource_address)
        return pb2_version.DeleteResponse()

    def Copy(self, request: fileobject_service_pb2_v1alpha.CopyRequest, context):
        """Copy a file version to a new destination (v1alpha only).

        Creates a copy of a specific file version at a new address, creating
        a new version at the destination.

        Args:
            request: CopyRequest containing:
                    - source_resource_identity: Identity of version to copy
                    - destination_resource_address: Target address
                    - previous_version: Optional expected version at destination (for optimistic locking)
            context: gRPC ServicerContext for the request.

        Returns:
            CopyResponse containing:
            - resource_identity: Identity of the newly created version at destination

        Raises:
            grpc.RpcError with:
            - INVALID_ARGUMENT: If identities/addresses invalid, or destination is versioned
            - FAILED_PRECONDITION: If previous_version doesn't match destination current version
            - PERMISSION_DENIED: If read permission denied on source

        Note:
            This is a v1alpha-only method, not available in v1beta.
            The copy creates a new version at the destination.
        """
        backend = get_backend()

        # Convert source identity to resource address
        # Use url_from_identity to preserve version info for versioned copies
        try:
            source_address = backend.url_from_identity(request.source_resource_identity.encoded_identity)
        except (ValueError, KeyError) as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, details=f"Invalid source identity: {e}")

        # Validate source exists
        if not backend.exists(source_address):
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                details="Source resource not found",
            )

        # Validate source is a file, not a folder
        if backend.is_dir(source_address):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                details="Cannot copy folders, only files",
            )

        # Validate destination address
        if not backend.is_address_valid(request.destination_resource_address):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                details=f"{request.destination_resource_address} is not a valid destination address",
            )
        if backend.is_version_address(request.destination_resource_address):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                details=f"Cannot copy to individual version address: {request.destination_resource_address}",
            )

        # Check previous version if specified
        if request.HasField("previous_version"):
            # Check if backend supports optimistic locking
            check_optimistic_locking_support(context, "Copy operation")

            abort_on_previous_version_not_latest(
                request.destination_resource_address,
                request.previous_version.encoded_identity,
                context,
            )

        try:
            # Perform the copy using backend's copy method
            new_identity = backend.copy(source_address, request.destination_resource_address)
            return fileobject_service_pb2_v1alpha.CopyResponse(
                resource_identity=fileobject_pb2_v1alpha.ResourceIdentity(encoded_identity=new_identity)
            )
        except PermissionError as e:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, details=str(e))
        except (FileNotFoundError, ValueError) as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, details=str(e))
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, details=f"Copy failed: {e}")

    def Move(self, request: fileobject_service_pb2_v1alpha.MoveRequest, context):
        """Move/rename a file to a new address (v1alpha only).

        Moves a file from one address to another, preserving all versions.
        This is effectively a rename operation.

        Args:
            request: MoveRequest containing:
                    - source_resource_address: Current file address
                    - destination_resource_address: New file address
                    - source_previous_version: Optional expected version at source (for optimistic locking)
                    - destination_previous_version: Optional expected version at destination (for optimistic locking)
            context: gRPC ServicerContext for the request.

        Returns:
            MoveResponse containing:
            - resource_identity: Identity of the version at the new address

        Raises:
            grpc.RpcError with:
            - INVALID_ARGUMENT: If addresses invalid, versioned, or source is directory
            - NOT_FOUND: If source doesn't exist
            - FAILED_PRECONDITION: If source_previous_version or destination_previous_version don't match
            - PERMISSION_DENIED: If permission denied

        Note:
            - This is a v1alpha-only method, not available in v1beta
            - Cannot move directories, only files
            - Cannot move from/to versioned addresses
        """
        backend = get_backend()

        if not request.source_resource_address:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                details="source_resource_address is required",
            )

        if backend.is_version_address(request.source_resource_address):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                details=f"Cannot move from individual version address: {request.source_resource_address}",
            )
        if backend.is_version_address(request.destination_resource_address):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                details=f"Cannot move to individual version address: {request.destination_resource_address}",
            )

        if not backend.is_address_valid(request.source_resource_address):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                details="Invalid source resource address",
            )

        # Check source exists
        if not backend.exists(request.source_resource_address):
            context.abort(grpc.StatusCode.NOT_FOUND, details="Source file not found")

        # Check source is a file, not a folder
        if backend.is_dir(request.source_resource_address):
            error_code_to_send = (
                grpc.StatusCode.NOT_FOUND
                if backend.folder_mode() == FolderMode.HYBRID
                else grpc.StatusCode.INVALID_ARGUMENT
            )
            context.abort(error_code_to_send, details="Cannot move directories")

        # Check source_previous_version if specified
        if request.HasField("source_previous_version"):
            # Check if backend supports optimistic locking
            check_optimistic_locking_support(context, "Move operation")

            # Force a validity check of the given resource identity
            try:
                backend.address_from_identity(request.source_previous_version.encoded_identity)
            except ValueError:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Invalid source_previous_version given: {request.source_previous_version}",
                )

            try:
                source_is_latest = backend.is_version_latest(
                    request.source_resource_address,
                    request.source_previous_version.encoded_identity,
                )
            except FileNotFoundError:
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    details=f"Source resource not found at {request.source_resource_address}",
                )
            except Exception as e:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    details=f"Invalid source_previous_version: {e}",
                )

            if not source_is_latest:
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    details="source_previous_version no longer matches current version at source_resource_address",
                )

        # Validate destination address
        if not backend.is_address_valid(request.destination_resource_address):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                details=f"{request.destination_resource_address} is not a valid destination address",
            )

        # Check destination_previous_version if specified
        if request.HasField("destination_previous_version"):
            # Check if backend supports optimistic locking
            check_optimistic_locking_support(context, "Move operation")

            abort_on_previous_version_not_latest(
                request.destination_resource_address,
                request.destination_previous_version.encoded_identity,
                context,
            )

        try:
            # Perform the move using backend's move method
            result_identity = backend.move(
                source_resource_address=request.source_resource_address,
                destination_resource_address=request.destination_resource_address,
            )

            return fileobject_service_pb2_v1alpha.MoveResponse(
                resource_identity=fileobject_pb2_v1alpha.ResourceIdentity(encoded_identity=result_identity)
            )
        except FileNotFoundError as e:
            context.abort(grpc.StatusCode.NOT_FOUND, details=str(e))
        except PermissionError as e:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, details=str(e))
        except OSError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, details=str(e))

    def GetOptimisticLockingSupport(self, request, context):
        """Query the server's support for optimistic locking.

        Returns information about which operations support conditional execution
        with previous_version parameter for the given resource address.

        Args:
            request: GetOptimisticLockingSupportRequest containing:
                    - resource_address: The resource address to check support for
            context: gRPC ServicerContext for the request.

        Returns:
            GetOptimisticLockingSupportResponse containing boolean flags for:
            - write: True if Write supports previous_version
            - delete: True if Delete supports previous_version
            - copy: True if Copy supports previous_version
            - move: True if Move supports source_previous_version/destination_previous_version
        """
        # Note: Currently, the backend returns the same support for all addresses.
        # In the future, this could vary based on the resource_address.
        backend = get_backend()
        support = backend.get_optimistic_locking_support()

        return fileobject_service_pb2_v1alpha.GetOptimisticLockingSupportResponse(
            supports_write=support.write,
            supports_delete=support.delete,
            supports_copy=support.copy,
            supports_move=support.move,
        )

    methods = {
        "Enumerate": Enumerate,
        "Stat": Stat,
        "Read": Read,
        "ReadFromAddress": ReadFromAddress,
        "FetchWriteTypeInfo": FetchWriteTypeInfo,
        "Write": Write,
        "CompleteRedirectUpload": CompleteRedirectUpload,
        "UploadPart": UploadPart,
        "CompleteMultipartUpload": CompleteMultipartUpload,
        "AbortMultipartUpload": AbortMultipartUpload,
        "Delete": Delete,
    }
    if version_tag == "v1alpha":
        methods["Copy"] = Copy
        methods["Move"] = Move
        methods["GetOptimisticLockingSupport"] = GetOptimisticLockingSupport

    cls = type(
        "FileSystemServiceServicer",
        (pb2_grpc_version.FileObjectServiceServicer,),
        methods,
    )

    return cls()

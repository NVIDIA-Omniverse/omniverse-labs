# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Model classes for REST."""

from enum import Enum
from typing import (
    Annotated,
    Any,
)

from pydantic import (
    Field,
    RootModel,
    WithJsonSchema,
    model_validator,
)
from pydantic.main import BaseModel

RESOURCE_ADDRESS_TYPE = Annotated[
    str, WithJsonSchema({"type": "string", "maxLength": 4096, "format": "resource_address"})
]
FOLDER_ADDRESS_TYPE = Annotated[str, WithJsonSchema({"type": "string", "maxLength": 4096, "format": "folder_address"})]
RESOURCE_IDENTITY_TYPE = Annotated[
    str, WithJsonSchema({"type": "string", "maxLength": 4096, "format": "resource_identity"})
]
OPTIONAL_RESOURCE_IDENTITY_TYPE = Annotated[
    str | None, WithJsonSchema({"type": "string", "maxLength": 4096, "format": "resource_identity", "nullable": True})
]
OPTIONAL_RESOURCE_ADDRESS_TYPE = Annotated[
    str | None, WithJsonSchema({"type": "string", "maxLength": 4096, "format": "resource_address", "nullable": True})
]
PAGE_HANDLE_TYPE = Annotated[
    str | None, WithJsonSchema({"type": "string", "format": "page_handle", "maxLength": 512, "nullable": True})
]
KEY_HANDLE_TYPE = Annotated[
    str | None, WithJsonSchema({"type": "string", "format": "key_handle", "maxLength": 512, "nullable": True})
]
MAX_PAGE_SIZE_SCHEMA = {"type": "integer", "format": "int32", "minimum": 0, "maximum": 65536, "nullable": True}
DOWNLOAD_PREFERENCE_SCHEMA = {"type": "string", "nullable": True, "enum": ["body", "redirect"]}
UPLOAD_PREFERENCE_SCHEMA = {"type": "string", "nullable": True, "enum": ["body", "redirect", "multipart"]}
URL_PARAMETER_TYPE = Annotated[str, WithJsonSchema({"type": "string", "maxLength": 4096, "format": "uri"})]
HTTP_HEADER_PARAMETER_TYPE = Annotated[
    str, WithJsonSchema({"type": "string", "maxLength": 4096, "format": "http_header"})
]
UPLOAD_ID_TYPE = Annotated[str, WithJsonSchema({"type": "string", "format": "upload_id", "maxLength": 128})]
SERVICE_NAME_TYPE = Annotated[str, WithJsonSchema({"type": "string", "maxLength": 4096})]
TOP_LEVEL_ADDRESS_TYPE = Annotated[str, WithJsonSchema({"type": "string", "maxLength": 4096})]
METADATA_URI_TYPE = Annotated[str, WithJsonSchema({"type": "string", "maxLength": 4096, "format": "uri"})]
METADATA_KEY_TYPE = Annotated[str, WithJsonSchema({"type": "string", "maxLength": 1024, "format": "key"})]
METADATA_ETAG_TYPE = Annotated[str, WithJsonSchema({"type": "string", "maxLength": 256, "format": "etag"})]


class ServiceEntry(BaseModel):
    service_name: SERVICE_NAME_TYPE
    service_versions: list[SERVICE_NAME_TYPE]


class ListServicesResponse(BaseModel):
    services: list[ServiceEntry]


class TopLevelAddress(BaseModel):
    top_level_address: TOP_LEVEL_ADDRESS_TYPE


class MoveRequest(BaseModel):
    source_previous_version: OPTIONAL_RESOURCE_IDENTITY_TYPE = Field(
        default=None,
        title="Expected source resource identity",
        description=(
            "The expected resource identity of the source file object. "
            "Move will fail if this is no longer the current version or the storage doesn't support conditional updates."
        ),
    )
    destination_resource_address: RESOURCE_ADDRESS_TYPE
    destination_previous_version: OPTIONAL_RESOURCE_IDENTITY_TYPE = Field(
        default=None,
        title="Expected previous version identity at destination",
        description=(
            "The resource identity of the previous version at the destination "
            "resource address. Move will fail if this is no longer the latest "
            "version or the storage doesn't support conditional updates."
        ),
    )


class MoveResponse(BaseModel):
    resource_identity: RESOURCE_IDENTITY_TYPE


class OptimisticLockingSupportResponse(BaseModel):
    """Response indicating which operations support optimistic locking."""

    supports_write: bool = Field(
        description="True if Write operations support conditional execution with previous_version."
    )
    supports_delete: bool = Field(
        description="True if Delete operations support conditional execution with previous_version."
    )
    supports_copy: bool = Field(
        description="True if Copy operations support conditional execution with previous_version."
    )
    supports_move: bool = Field(
        description="True if Move operations support conditional execution with source/destination_previous_version."
    )


class ListTopLevelAddressesResponse(BaseModel):
    items: list[TopLevelAddress]


class Route(BaseModel):
    wildcard_pattern: str | None = Field(
        None,
        description=(
            "Simple wildcard pattern that uses '*' for an identifier "
            "[a-zA-Z0-9_-]+ and '**' for any substring. Needs to end on '**' "
            "to match sub-URLs."
        ),
        # Future route types can be added here as Optional fields.
        # For example:
        # regex_pattern: Optional[str] = Field(
        #     None,
        #     description="Regular expression pattern for more complex route matching."
        # )
    )

    @model_validator(mode="before")
    def check_only_one_route_type(cls, values):
        route_fields = [
            "wildcard_pattern",
            # Add additional route type fields here, e.g.,
            # 'regex_pattern',
        ]
        set_fields = [field for field in route_fields if values.get(field) is not None]
        if len(set_fields) != 1:
            raise ValueError(f"Exactly one of {route_fields} must be set.")
        return values

    class Config:
        # Enables Pydantic to ignore any extra fields not defined in the model
        extra = "forbid"


class ListRoutesResponse(BaseModel):
    items: list[Route]


class Metadata(BaseModel):
    """Additional metadata common to all storages."""

    data_object_size: int = Field(json_schema_extra={"format": "int64"})
    last_modified_timestamp: str | None = Field(
        default=None,
        title="Last modification timestamp in ISO format",
        json_schema_extra={"type": "string", "format": "date-time", "maxLength": 32},
    )


class ListEntry(BaseModel):
    """One entry."""

    url: str
    size: int
    last_modified: str  # ISO8601 timestamp
    version_id: str | None


class AddressInfo(BaseModel):
    """Version info of a data object."""

    resource_address: RESOURCE_ADDRESS_TYPE
    metadata: Metadata = Field(title="The data object metadata.")


class EnumerateResponse(BaseModel):
    """A page of address infos."""

    items: list[AddressInfo] = Field(description="A page of matching address info entries.")
    next_continuation_handle: PAGE_HANDLE_TYPE = Field(
        default=None, description="A token by which enumeration can be continued.", max_length=512
    )


class StatRequest(BaseModel):
    """Given one or more URLs, return a list with details about those objects."""

    url: str | list[str]


class StatResponse(BaseModel):
    """Returns a list with the detail information."""

    results: list[ListEntry]


class ResourceInfo(BaseModel):
    """A ResourceIdentity with Metadata."""

    resource_identity: RESOURCE_IDENTITY_TYPE
    metadata: Metadata


class VersionInfo(BaseModel):
    """A ResourceInfo with Optional sorting_key"""

    resource_info: ResourceInfo
    sorting_key: KEY_HANDLE_TYPE = Field(
        default=None,
        description="The optional sorting_key",
        max_length=512,
    )
    resource_address: OPTIONAL_RESOURCE_ADDRESS_TYPE = Field(
        default=None,
        description="The resource address for each version that can be used with ReadFromAddress",
    )


class ReadRequest(BaseModel):
    """Start a read request. The server can choose to reply with a redirect to a pre-signed URL or deliver the data directly."""

    url: str
    multipart: bool | None = Field(None)
    force_redirect: bool | None = Field(None)


class ReadResponse(BaseModel):
    """Redirect information."""

    redirect_target_url: str = Field(
        description="Redirect request url.", max_length=4096, json_schema_extra={"format": "uri"}
    )
    additional_headers: dict[str, str] = Field(description="Additional headers for the redirect request.")


class WriteRequest(BaseModel):
    """Directly write data from memory into an object."""

    url: str
    content: bytes


class WriteResponse(BaseModel):
    """Return result of write operation."""

    resource_identity: str
    metadata: Metadata


class HTTPHeader(BaseModel):
    name: HTTP_HEADER_PARAMETER_TYPE
    value: HTTP_HEADER_PARAMETER_TYPE


class WriteRedirectProperties(BaseModel):
    redirect_target_url: URL_PARAMETER_TYPE
    method: str
    additional_headers: list[HTTPHeader]
    completion_header_names: list[Annotated[str, WithJsonSchema({"type": "string", "maxLength": 4096})]]


class WriteRedirectResponse(BaseModel):
    redirect: WriteRedirectProperties


class MultipartUploadProperties(BaseModel):
    upload_id: UPLOAD_ID_TYPE
    first_part_write_redirect: WriteRedirectProperties
    minimum_size_per_part: int | None = None
    maximum_size_per_part: int | None = None
    maximum_parts_number: int | None = None


class MultipartUploadResponse(BaseModel):
    multipart: MultipartUploadProperties


class MultipartUploadRequest(BaseModel):
    upload_id: UPLOAD_ID_TYPE
    part_number: int
    part_count: Annotated[int | None, WithJsonSchema({"type": "integer", "format": "int32", "nullable": True})] = None


class UploadPartResponse(BaseModel):
    part_write_redirects: list[WriteRedirectProperties]


class CompletedUploadPart(BaseModel):
    part_number: int
    additional_headers: list[HTTPHeader]


class CompleteMultipartUploadRequest(BaseModel):
    upload_id: UPLOAD_ID_TYPE
    parts: list[CompletedUploadPart]


class CompleteUploadRequest(BaseModel):
    additional_headers: list[HTTPHeader]


class HTTPValidationError(BaseModel):
    """Error raised by REST endpoint for unprocessable content."""

    pass


class UploadPreference(Enum):
    body = "body"
    redirect = "redirect"
    multipart = "multipart"


class WriteTypeInterval(BaseModel):
    minimum_data_object_size: Annotated[int, WithJsonSchema({"type": "integer", "format": "int64"})]
    maximum_data_object_size: Annotated[int, WithJsonSchema({"type": "integer", "format": "int64"})]
    preferred_upload_method: UploadPreference


class UploadOptionsResponse(BaseModel):
    write_type_intervals: list[WriteTypeInterval]


class MultipartUploadAbortRequest(BaseModel):
    upload_id: UPLOAD_ID_TYPE


class CreateMultipartUploadResponse(BaseModel):
    upload_id: UPLOAD_ID_TYPE


class VersionsOrder(str, Enum):
    NEWEST_FIRST = "newest_first"
    OLDEST_FIRST = "oldest_first"
    BY_KEY = "by_key"


class EnumerateVersionsResponse(BaseModel):
    """A page of versions infos."""

    items: list[VersionInfo] = Field(description="A page of version info entries.")
    next_continuation_handle: PAGE_HANDLE_TYPE = Field(
        default=None, description="A token by which enumeration can be continued.", max_length=512
    )
    versions_order: VersionsOrder = Field(description="Returned versions order.")


class ListItem(BaseModel):
    """One entry with metadata and resource identity."""

    resource_address: RESOURCE_ADDRESS_TYPE
    resource_identity: RESOURCE_IDENTITY_TYPE
    metadata: Metadata


class ListStatResponse(BaseModel):
    """A page of address infos including metadata and identities."""

    subfolder_addresses: list[FOLDER_ADDRESS_TYPE] = Field(
        description="Batch of resource addresses that can be listed as folders."
    )
    entries: list[ListItem] = Field(description="List of items with resource address, identity, and metadata.")
    next_continuation_handle: PAGE_HANDLE_TYPE = Field(
        default=None, description="A token by which listing can be continued.", max_length=512
    )


class ListResponse(BaseModel):
    """A page of address infos."""

    subfolder_addresses: list[FOLDER_ADDRESS_TYPE] = Field(
        description="Batch of resource addresses that can be listed as folders."
    )
    sub_resource_addresses: list[RESOURCE_ADDRESS_TYPE] = Field(
        description="Batch of resource addresses for file objects."
    )
    next_continuation_handle: PAGE_HANDLE_TYPE = Field(
        default=None, description="A token by which listing can be continued.", max_length=512
    )


class CopyRequest(BaseModel):
    destination_resource_address: RESOURCE_ADDRESS_TYPE
    previous_version: OPTIONAL_RESOURCE_IDENTITY_TYPE = Field(
        default=None,
        title="The previous version expected to be latest",
        max_length=4096,
        json_schema_extra={"format": "resource_identity", "type": "string", "nullable": True},
    )


class CopyResponse(BaseModel):
    resource_identity: RESOURCE_IDENTITY_TYPE


UserMetadataKeys = Annotated[
    list[Annotated[str, WithJsonSchema({"type": "string", "maxLength": 1024})]],
    WithJsonSchema(
        {
            "type": "array",
            "maxItems": 4096,
            "items": {"type": "string", "maxLength": 1024},
            "description": "A list of requested metadata keys.",
        }
    ),
]


class UserMetadataValue(BaseModel):
    """A combination of metadata value and etag."""

    value: Any = Field(description="The metadata value")
    etag: METADATA_ETAG_TYPE = Field(description="Metadata key ETag")


class UserMetadataResponse(RootModel[dict[str, UserMetadataValue]]):
    """A map-type to store extended metadata."""

    root: dict[str, UserMetadataValue] = Field(description="A map-type to store extended metadata")


class FolderMode(Enum):
    NATIVE = "native"
    NO_EMPTY = "no_empty"
    HYBRID = "hybrid"


class GetFolderModeResponse(BaseModel):
    folder_mode: FolderMode

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

"""Pydantic models for Microsoft Graph API responses.

These models represent OneDrive/SharePoint items and their metadata
as returned by the Microsoft Graph API.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class FileFacet(BaseModel):
    """File facet indicating the item is a file."""

    mime_type: str | None = Field(None, alias="mimeType")
    hashes: dict[str, Any] | None = None


class FolderFacet(BaseModel):
    """Folder facet indicating the item is a folder."""

    child_count: int = Field(0, alias="childCount")


class ParentReference(BaseModel):
    """Reference to the parent item."""

    drive_id: str | None = Field(None, alias="driveId")
    drive_type: str | None = Field(None, alias="driveType")
    id: str | None = None
    path: str | None = None
    name: str | None = None


class DriveItem(BaseModel):
    """Represents a OneDrive or SharePoint item (file or folder).

    See: https://learn.microsoft.com/en-us/graph/api/resources/driveitem
    """

    id: str
    name: str
    size: int = 0
    created_date_time: datetime | None = Field(None, alias="createdDateTime")
    last_modified_date_time: datetime | None = Field(None, alias="lastModifiedDateTime")
    web_url: str | None = Field(None, alias="webUrl")
    e_tag: str | None = Field(None, alias="eTag")
    c_tag: str | None = Field(None, alias="cTag")

    # Facets - presence indicates item type
    file: FileFacet | None = None
    folder: FolderFacet | None = None

    # Parent reference
    parent_reference: ParentReference | None = Field(None, alias="parentReference")

    # Download URL (only present in some responses)
    download_url: str | None = Field(None, alias="@microsoft.graph.downloadUrl")

    @property
    def is_file(self) -> bool:
        """Check if this item is a file."""
        return self.file is not None

    @property
    def is_folder(self) -> bool:
        """Check if this item is a folder."""
        return self.folder is not None

    @property
    def drive_id(self) -> str | None:
        """Get the drive ID from parent reference."""
        if self.parent_reference:
            return self.parent_reference.drive_id
        return None


class DriveItemVersion(BaseModel):
    """Represents a version of a OneDrive item.

    See: https://learn.microsoft.com/en-us/graph/api/resources/driveitemversion
    """

    id: str
    last_modified_date_time: datetime | None = Field(None, alias="lastModifiedDateTime")
    size: int = 0


class UploadSession(BaseModel):
    """Represents an upload session for large file uploads.

    See: https://learn.microsoft.com/en-us/graph/api/resources/uploadsession
    """

    upload_url: str = Field(alias="uploadUrl")
    expiration_date_time: datetime | None = Field(None, alias="expirationDateTime")
    next_expected_ranges: list[str] = Field(default_factory=list, alias="nextExpectedRanges")


class DriveItemCollection(BaseModel):
    """Represents a collection of drive items (paginated response)."""

    value: list[DriveItem] = Field(default_factory=list)
    next_link: str | None = Field(None, alias="@odata.nextLink")


class DriveItemVersionCollection(BaseModel):
    """Represents a collection of drive item versions."""

    value: list[DriveItemVersion] = Field(default_factory=list)
    next_link: str | None = Field(None, alias="@odata.nextLink")


class GraphError(BaseModel):
    """Represents a Microsoft Graph API error response."""

    code: str
    message: str
    inner_error: dict[str, Any] | None = Field(None, alias="innerError")


class GraphErrorResponse(BaseModel):
    """Wrapper for Graph API error responses."""

    error: GraphError

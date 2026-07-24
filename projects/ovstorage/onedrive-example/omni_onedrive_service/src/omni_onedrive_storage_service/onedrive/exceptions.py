# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""OneDrive-specific exceptions for the storage backend."""


class OneDriveError(Exception):
    """Base exception for OneDrive backend errors."""

    pass


class GraphApiError(OneDriveError):
    """Raised when a Microsoft Graph API call fails."""

    def __init__(self, message: str, status_code: int, error_code: str | None = None):
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(f"Graph API Error [{status_code}] {error_code}: {message}")


class RateLimitError(OneDriveError):
    """Raised when rate limited by Microsoft Graph API (429 response)."""

    def __init__(self, retry_after: int):
        self.retry_after = retry_after
        super().__init__(f"Rate limited. Retry after {retry_after} seconds.")


class ItemNotFoundError(OneDriveError):
    """Raised when a OneDrive item is not found (Graph 404 / itemNotFound)."""

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(f"Item not found: {detail}")


class VersionNotFoundError(OneDriveError):
    """Raised when a specific version of a OneDrive item is not found."""

    def __init__(self, item_id: str, version_id: str):
        self.item_id = item_id
        self.version_id = version_id
        super().__init__(f"Version {version_id} not found for item {item_id}")

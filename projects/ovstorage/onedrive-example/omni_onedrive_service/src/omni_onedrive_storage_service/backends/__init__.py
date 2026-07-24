# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from .backend_factory import (
    BackendConfig,
    create_backend,
    get_backend_cli_commands,
    has_backend_cli,
    is_backend_registered,
    list_backends,
    register_backend,
    register_backend_cli,
)
from .runtime import (
    MAX_UPLOAD_SIZE_BYTES,
    REDIRECT_HOST,
    REDIRECT_PORT,
    get_backend,
    init_backend,
)
from .storage_backend_interface import (
    EtagMismatchError,
    ListEntry,
    Metadata,
    MetadataKeyNotFoundError,
    RedirectUploadResult,
    StorageBackendInterface,
    VersionInfo,
    VersionsOrder,
)

__all__ = [
    # Interface
    "StorageBackendInterface",
    "Metadata",
    "VersionInfo",
    "ListEntry",
    "VersionsOrder",
    "EtagMismatchError",
    "MetadataKeyNotFoundError",
    "RedirectUploadResult",
    # Factory
    "BackendConfig",
    "create_backend",
    "register_backend",
    "register_backend_cli",
    "list_backends",
    "is_backend_registered",
    "get_backend_cli_commands",
    "has_backend_cli",
    # Runtime (active backend + shared config)
    "get_backend",
    "init_backend",
    "REDIRECT_HOST",
    "REDIRECT_PORT",
    "MAX_UPLOAD_SIZE_BYTES",
]

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Backend-agnostic runtime state for the storage service.

Holds the process-wide "active backend" instance plus shared runtime
configuration used by the gRPC and REST service layers. This lives in the
``backends`` package - not in any concrete backend module - so the service
layer can resolve the active backend without importing a specific backend
implementation.
"""

import logging
import os

from .backend_factory import BackendConfig, create_backend
from .storage_backend_interface import StorageBackendInterface

logger = logging.getLogger(__name__)

# Shared runtime configuration for redirect-based upload/download URLs. Backends
# that redirect elsewhere (e.g. OneDrive redirects to Microsoft Graph) may ignore
# these, but the service layer still passes them when constructing redirect URLs.
REDIRECT_HOST = os.getenv("REDIRECT_HOST", "http://localhost")
REDIRECT_PORT = int(os.getenv("REDIRECT_PORT", "8011"))

# Maximum size (in bytes) the service will buffer in memory for a single "body"
# upload. Backends without redirect/multipart upload support (e.g. OneDrive)
# accumulate the entire object in memory before handing it to the backend, so
# this bounds per-request memory use and prevents an authenticated client from
# exhausting process memory with an oversized or size-misdeclared upload
# (denial of service). Configurable via the MAX_UPLOAD_SIZE_BYTES env var;
# defaults to 512 MiB.
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(512 * 1024 * 1024)))

# Process-wide active backend instance, set by init_backend().
_storage_backend: StorageBackendInterface | None = None


def init_backend(config: BackendConfig) -> StorageBackendInterface:
    """Create the active storage backend from configuration and store it.

    This should be called once at application startup with the desired backend
    configuration. Subsequent calls replace the active backend.

    Args:
        config: Configuration for the storage backend.

    Returns:
        The created storage backend instance.
    """
    global _storage_backend
    _storage_backend = create_backend(config)
    logger.info(f"Initialized storage backend: {config.backend_type}")
    return _storage_backend


def get_backend() -> StorageBackendInterface:
    """Return the active storage backend.

    Returns:
        The current storage backend.

    Raises:
        RuntimeError: If no backend has been initialized via init_backend().
    """
    if _storage_backend is None:
        raise RuntimeError("Storage backend not initialized. Call init_backend() first.")
    return _storage_backend

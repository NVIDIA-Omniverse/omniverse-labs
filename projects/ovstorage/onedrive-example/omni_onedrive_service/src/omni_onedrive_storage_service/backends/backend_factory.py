# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Storage backend factory for creating and managing storage backend instances.

This module provides a registry-based factory pattern for creating storage backends.
Backends can be registered with a name and a factory function, then instantiated
at runtime based on CLI arguments or configuration.

Example:
    # Register a backend
    @register_backend("s3")
    def create_s3_backend(config: BackendConfig) -> StorageBackendInterface:
        return S3StorageBackend(config.base_uri)

    # Create backend from configuration
    backend = create_backend(BackendConfig(backend_type="s3", base_uri="..."))
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    Any,
)

from .storage_backend_interface import StorageBackendInterface

logger = logging.getLogger(__name__)


@dataclass
class BackendConfig:
    """Configuration for creating a storage backend.

    This is the base configuration class with common fields. Backend-specific
    implementations should create their own config classes with additional fields.

    Attributes:
        backend_type: Type/name of backend to create (e.g., "filesystem", "git", "s3")
        base_uri: Base URI for resource addresses (e.g., "file-storage://fileservice")
        extra_config: Backend-specific configuration options
    """

    backend_type: str
    base_uri: str
    extra_config: dict[str, Any] | None = None


# Type for backend factory functions
BackendFactory = Callable[[BackendConfig], StorageBackendInterface]

# Global registry of backend factories
_backend_registry: dict[str, BackendFactory] = {}


def register_backend(name: str) -> Callable[[BackendFactory], BackendFactory]:
    """Decorator to register a backend factory function.

    Args:
        name: Name to register the backend under (e.g., "filesystem", "git")

    Returns:
        Decorator function that registers the backend

    Example:
        @register_backend("filesystem")
        def create_filesystem_backend(config: BackendConfig) -> StorageBackendInterface:
            return MyStorageBackend(config.base_uri)
    """

    def decorator(factory: BackendFactory) -> BackendFactory:
        if name in _backend_registry:
            logger.warning(f"Backend '{name}' is already registered, overwriting")
        _backend_registry[name] = factory
        logger.info(f"Registered storage backend: {name}")
        return factory

    return decorator


def create_backend(config: BackendConfig) -> StorageBackendInterface:
    """Create a storage backend instance from configuration.

    Args:
        config: Configuration for the backend

    Returns:
        An instance of StorageBackendInterface

    Raises:
        ValueError: If the backend type is not registered

    Example:
        config = BackendConfig(
            backend_type="filesystem",
            base_uri="file-storage://fileservice",
        )
        backend = create_backend(config)
    """
    backend_type = config.backend_type

    if backend_type not in _backend_registry:
        available = ", ".join(_backend_registry.keys())
        raise ValueError(f"Unknown backend type '{backend_type}'. Available backends: {available or 'none registered'}")

    factory = _backend_registry[backend_type]
    logger.info(f"Creating storage backend: {backend_type}")

    try:
        backend = factory(config)
        logger.info(f"Successfully created {backend_type} backend")
        return backend
    except Exception as e:
        logger.error(f"Failed to create {backend_type} backend: {e}")
        raise


def list_backends() -> list[str]:
    """List all registered backend types.

    Returns:
        List of registered backend names
    """
    return sorted(_backend_registry.keys())


def is_backend_registered(name: str) -> bool:
    """Check if a backend is registered.

    Args:
        name: Backend name to check

    Returns:
        True if the backend is registered, False otherwise
    """
    return name in _backend_registry


# Re-export CLI registry functions for convenience
from .cli_registry import (  # noqa: E402
    get_backend_cli_commands,
    has_backend_cli,
    register_backend_cli,
)

__all__ = [
    "BackendConfig",
    "create_backend",
    "register_backend",
    "register_backend_cli",
    "list_backends",
    "is_backend_registered",
    "get_backend_cli_commands",
    "has_backend_cli",
]

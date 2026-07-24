# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Registry for backend CLI commands."""

import logging
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from .backend_factory import BackendConfig

logger = logging.getLogger(__name__)

# Type for backend CLI functions
# These functions take CLI arguments and return a BackendConfig
BackendCliFunc = Callable[..., "BackendConfig"]

# Global registry of backend CLI commands
_backend_cli_registry: dict[str, BackendCliFunc] = {}


def register_backend_cli(name: str, help_text: str = "") -> Callable[[BackendCliFunc], BackendCliFunc]:
    """Decorator to register a backend CLI command.

    Args:
        name: Name of the backend command (e.g., "filesystem")
        help_text: Help text for the command

    Returns:
        Decorator function
    """

    def decorator(cli_func: BackendCliFunc) -> BackendCliFunc:
        if name in _backend_cli_registry:
            logger.warning(f"Backend CLI '{name}' is already registered, overwriting")

        # Attach docstring if provided (and not already present)
        if help_text and not cli_func.__doc__:
            cli_func.__doc__ = help_text

        _backend_cli_registry[name] = cli_func
        logger.info(f"Registered storage backend CLI: {name}")
        return cli_func

    return decorator


def get_backend_cli_commands() -> dict[str, BackendCliFunc]:
    """Get all registered backend CLI commands.

    Returns:
        Dictionary mapping backend name to CLI function
    """
    return _backend_cli_registry.copy()


def has_backend_cli(name: str) -> bool:
    """Check if a backend has a registered CLI command.

    Args:
        name: Backend name to check

    Returns:
        True if registered, False otherwise
    """
    return name in _backend_cli_registry

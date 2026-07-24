# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Helper functions for service request handlers.

Provides backend-agnostic helpers for extracting the relative path from a
resource address and for building well-formed outbound resource addresses.
"""

import os
import urllib.parse


def get_relative_path_from_address(base_url: str, resource_address: str) -> str:
    """Extract the relative, URL-decoded path from a resource address.

    Well-behaved clients percent-encode reserved characters on the wire
    (space -> ``%20``, ``%`` -> ``%25``, etc.). ``urlparse().path`` preserves
    that encoding verbatim, so we must ``unquote`` here to obtain the raw
    folder/file name that filesystem calls (or downstream backends) expect
    (``New project`` rather than ``New%20project``). Any code that turns
    this path back into an outbound URI must re-encode it with
    :func:`build_address`.

    Args:
        base_url: Base resource address (e.g., "file-storage://server/")
        resource_address: Full resource address
            (e.g., "file-storage://server/Documents/New%20project")

    Returns:
        Relative decoded path (e.g., "Documents/New project").
    """
    parsed_address = urllib.parse.urlparse(resource_address)
    base_address = urllib.parse.urlparse(base_url)
    if parsed_address.scheme != base_address.scheme or parsed_address.netloc != base_address.netloc:
        raise ValueError(f"{resource_address} is not within base address {base_url}, program error!")
    path = urllib.parse.unquote(parsed_address.path)
    if os.path.isabs(path):
        path_without_drive = os.path.splitdrive(path)[1]
        path = path_without_drive.lstrip("/\\")

    return sanitize_path(path)


def build_address(base_url: str, path: str = "", child: str | None = None) -> str:
    """Build a well-formed outbound resource address.

    Percent-encodes ``path`` (and optional ``child`` component) so the
    resulting URI is valid even when the underlying names contain reserved
    characters such as spaces. ``path`` and ``child`` must be in their raw,
    decoded form — the same form returned by
    :func:`get_relative_path_from_address`.

    Args:
        base_url: Base resource address (e.g., "file-storage://server/").
        path: Optional decoded relative path to the parent.
        child: Optional decoded child (file or folder) name to append.

    Returns:
        A URI-safe address like ``file-storage://server/Documents/New%20project/foo.usd``.
    """
    base = base_url if base_url.endswith("/") else base_url + "/"
    normalized_parent = path.replace("\\", "/").strip("/") if path else ""
    if child:
        normalized_child = child.replace("\\", "/").strip("/")
        joined = f"{normalized_parent}/{normalized_child}" if normalized_parent else normalized_child
    else:
        joined = normalized_parent
    if not joined:
        return base
    return base + urllib.parse.quote(joined, safe="/")


def sanitize_path(path: str) -> str:
    """Sanitize a path to prevent directory traversal attacks while supporting . and .. operations."""
    # Convert to use forward slashes for consistency
    normalized = path.replace("\\", "/")

    # Split into components and process them with a stack-based approach
    components: list[str] = []
    for component in normalized.split("/"):
        if component == "" or component == ".":
            # Skip empty components and current directory references
            continue
        elif component == "..":
            # Handle parent directory - only pop if we have components to pop
            if components:
                components.pop()
            else:
                # Attempting to go above the root - this is a path traversal attack
                raise ValueError(f"Path traversal attempt detected: {path}")
        else:
            # Regular component - add it to the stack
            components.append(component)

    # Rejoin the safe components
    safe_path = "/".join(components)

    # Ensure the resulting path doesn't start with / (should be relative)
    return safe_path.lstrip("/")

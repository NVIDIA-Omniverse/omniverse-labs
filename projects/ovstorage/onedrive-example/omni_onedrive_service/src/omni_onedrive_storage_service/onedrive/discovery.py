# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Discovery endpoints for the OneDrive storage service.

Serves /api/v1/services and /api/v1/auth-config matching the format expected
by the Kit Client Library and the Web Streaming Portal.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from fastapi import FastAPI

if TYPE_CHECKING:
    from . import OneDriveConfig


def _build_service_urls(http_port: int, grpc_port: int) -> tuple[str, str]:
    """Build REST and gRPC URLs for the services discovery response.

    When SERVICE_PUBLIC_URL is set (e.g. behind an ingress), both REST and
    gRPC use that host. Otherwise, fall back to localhost with the actual ports.
    """
    public_url = os.environ.get("SERVICE_PUBLIC_URL", "").rstrip("/")
    if public_url:
        parsed = urlparse(public_url)
        host = parsed.hostname or parsed.path
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        rest_url = public_url
        grpc_url = f"{host}:{port}"
    else:
        rest_url = f"http://localhost:{http_port}"
        grpc_url = f"localhost:{grpc_port}"
    return rest_url, grpc_url


def register_discovery_routes(
    app: FastAPI,
    config: OneDriveConfig,
    grpc_port: int,
    http_port: int,
) -> None:
    """Register discovery endpoints on the FastAPI app.

    Args:
        app: The main FastAPI application
        config: OneDrive backend configuration with tenant/OIDC info
        grpc_port: gRPC server port for the services response
        http_port: REST server port for the services response
    """

    openid_configuration = f"https://login.microsoftonline.com/{config.tenant_id}/v2.0/.well-known/openid-configuration"

    scope = config.oidc_scopes
    if "{client_id}" in scope:
        scope = scope.format(client_id=config.oidc_client_id)

    client_entry = {
        "client_id": config.oidc_client_id,
        "scope": scope,
    }

    auth_config_response = {
        "openid_configuration": openid_configuration,
        "clients": {
            "default": client_entry,
            "client_library": client_entry,
            "navigator": client_entry,
        },
    }

    rest_url, grpc_url = _build_service_urls(http_port, grpc_port)

    services_response = {
        "schema-version": 1,
        "services": [
            {
                "id": "onedrive-storage",
                "name": "OneDrive Storage",
                "type": "storage",
                "rest": rest_url,
                "grpc": grpc_url,
            }
        ],
    }

    @app.get("/api/v1/services")
    @app.get("/api/v1alpha/services")
    async def get_services():
        return services_response

    @app.get("/api/v1/auth-config")
    @app.get("/api/v1alpha/auth-config")
    async def get_auth_config():
        return auth_config_response

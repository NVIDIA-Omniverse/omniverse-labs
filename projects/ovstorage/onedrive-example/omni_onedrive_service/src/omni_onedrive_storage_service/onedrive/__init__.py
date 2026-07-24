# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""OneDrive storage backend for NVIDIA Omniverse Storage API.

This package provides a storage backend implementation for OneDrive for Business
using per-user Bearer token authentication. The Kit Client Library handles the
OAuth/OIDC flow; this service receives the resulting token and forwards it to
the Microsoft Graph API.

Supported:
- OneDrive for Business (per-user, via /me/drive)

Not Supported:
- OneDrive Personal (consumer accounts)
- SharePoint shared document libraries (use client-credentials flow for those)

Example:
    omni-onedrive-storage-service onedrive \\
        --tenant-id YOUR_TENANT_ID \\
        --oidc-client-id YOUR_PUBLIC_CLIENT_ID

Configuration via environment variables:
    - AZURE_TENANT_ID: Azure AD tenant ID (for OpenID configuration URL)
    - OIDC_CLIENT_ID: Public app registration client ID (advertised to Kit clients)
    - OIDC_SCOPES: OAuth scope string (default: ``openid {client_id}/.default
      offline_access``). The ``{client_id}`` placeholder is replaced with
      OIDC_CLIENT_ID at startup. Override with explicit Graph scopes if needed.
"""

from dataclasses import dataclass
from typing import Annotated

import typer

from omni_onedrive_storage_service.backends import (
    BackendConfig,
    StorageBackendInterface,
    register_backend,
    register_backend_cli,
)

from .auth import configure_obo
from .onedrive_provider import OneDriveStorageProvider
from .request_context import configure_drive_cache


@dataclass
class OneDriveConfig(BackendConfig):
    """Configuration for the OneDrive storage backend.

    Attributes:
        backend_type: Type/name of backend (always "onedrive")
        base_uri: Base URI for resource addresses (default "onedrive://me")
        tenant_id: Azure AD tenant ID (for discovery endpoint)
        oidc_client_id: Public OIDC client ID (advertised to Kit clients)
        oidc_scopes: OAuth scopes (advertised to Kit clients)
        client_secret: Azure AD client secret (enables OBO flow when set)
        cache_ttl: Path and drive cache TTL in seconds
    """

    tenant_id: str = ""
    oidc_client_id: str = ""
    oidc_scopes: str = "openid {client_id}/.default offline_access"
    use_obo_flow: bool = False
    client_secret: str = ""
    obo_timeout: float = 10.0
    obo_cache_maxsize: int = 1000
    obo_cache_ttl: int = 3000
    cache_ttl: int = 300


@register_backend("onedrive")  # ty: ignore[invalid-argument-type]
def create_onedrive_backend(config: OneDriveConfig) -> StorageBackendInterface:
    """Create a OneDrive storage backend with per-user Bearer token auth."""
    configure_drive_cache(ttl=config.cache_ttl)
    if config.use_obo_flow:
        if not config.client_secret:
            raise ValueError("--client-secret (AZURE_CLIENT_SECRET) is required when --use-obo-flow is enabled")
        configure_obo(
            tenant_id=config.tenant_id,
            client_id=config.oidc_client_id,
            client_secret=config.client_secret,
            timeout=config.obo_timeout,
            cache_maxsize=config.obo_cache_maxsize,
            cache_ttl=config.obo_cache_ttl,
        )
    return OneDriveStorageProvider(
        base_uri=config.base_uri,
        cache_ttl=config.cache_ttl,
    )


@register_backend_cli("onedrive", "Start service with OneDrive storage (per-user Bearer token auth)")
def onedrive_cli_command(
    tenant_id: Annotated[
        str,
        typer.Option(
            "--tenant-id",
            help="Azure AD tenant ID (for OpenID configuration URL in discovery)",
            envvar="AZURE_TENANT_ID",
        ),
    ],
    oidc_client_id: Annotated[
        str,
        typer.Option(
            "--oidc-client-id",
            help="Public OIDC client ID (advertised to Kit clients for auth)",
            envvar="OIDC_CLIENT_ID",
        ),
    ],
    oidc_scopes: Annotated[
        str,
        typer.Option(
            "--oidc-scopes",
            help="OAuth scope string advertised to Kit clients. Use {client_id} as a"
            " placeholder — it is replaced with --oidc-client-id at startup."
            " Requires delegated permissions (e.g. Files.ReadWrite, User.Read)"
            " configured on the Azure AD app registration.",
            envvar="OIDC_SCOPES",
        ),
    ] = "openid {client_id}/.default offline_access",
    use_obo_flow: Annotated[
        bool,
        typer.Option(
            "--use-obo-flow",
            help="Enable On-Behalf-Of (OBO) token exchange. The service exchanges"
            " incoming app-scoped user tokens for Graph API tokens via Azure AD."
            " Requires --client-secret to be set.",
            envvar="USE_OBO_FLOW",
        ),
    ] = False,
    client_secret: Annotated[
        str,
        typer.Option(
            "--client-secret",
            help="Azure AD client secret for the OBO flow.",
            envvar="AZURE_CLIENT_SECRET",
        ),
    ] = "",
    obo_timeout: Annotated[
        float,
        typer.Option(
            "--obo-timeout",
            help="Timeout in seconds for OBO token exchange requests to Azure AD.",
            envvar="OBO_TIMEOUT",
        ),
    ] = 10.0,
    obo_cache_maxsize: Annotated[
        int,
        typer.Option(
            "--obo-cache-maxsize",
            help="Maximum number of OBO exchanged tokens to cache.",
            envvar="OBO_CACHE_MAXSIZE",
        ),
    ] = 1000,
    obo_cache_ttl: Annotated[
        int,
        typer.Option(
            "--obo-cache-ttl",
            help="Upper bound (seconds) for cached OBO token lifetime. Each token is"
            " additionally capped at its real expiry minus a safety margin, so a"
            " cached token never outlives its actual validity.",
            envvar="OBO_CACHE_TTL",
        ),
    ] = 3000,
    base_uri: Annotated[
        str,
        typer.Option(
            "--base-uri",
            help="Base URI for resource addresses",
            envvar="ONEDRIVE_BASE_URI",
        ),
    ] = "onedrive://me",
    cache_ttl: Annotated[
        int,
        typer.Option(
            "--cache-ttl",
            help="Path and drive cache TTL in seconds",
        ),
    ] = 300,
) -> OneDriveConfig:
    """Configure OneDrive storage backend with per-user Bearer token auth.

    The OneDrive backend accepts Bearer tokens from Kit's Client Library and
    forwards them to the Microsoft Graph API. Each user accesses their own
    OneDrive for Business.

    Requires an Azure AD public app registration with delegated permissions
    (e.g. Files.ReadWrite, User.Read) configured and admin-consented. The
    default scope uses {client_id}/.default so Azure AD grants whatever
    permissions are on the app registration.

    Examples:
        # Basic usage (scope auto-interpolates client ID)
        ... onedrive --tenant-id abc --oidc-client-id xyz

        # Explicit Graph scopes (bypasses .default)
        ... onedrive --tenant-id abc --oidc-client-id xyz \\
            --oidc-scopes "Files.ReadWrite.All User.Read offline_access"
    """
    return OneDriveConfig(
        backend_type="onedrive",
        base_uri=base_uri,
        tenant_id=tenant_id,
        oidc_client_id=oidc_client_id,
        oidc_scopes=oidc_scopes,
        use_obo_flow=use_obo_flow,
        client_secret=client_secret,
        obo_timeout=obo_timeout,
        obo_cache_maxsize=obo_cache_maxsize,
        obo_cache_ttl=obo_cache_ttl,
        cache_ttl=cache_ttl,
    )


__all__ = [
    "OneDriveConfig",
    "OneDriveStorageProvider",
    "create_onedrive_backend",
    "onedrive_cli_command",
]

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Capabilities service implementation for REST.

This module implements the CapabilitiesService from the Storage API, which
provides service discovery functionality. It allows clients to:
- List available services and their versions
- Discover top-level addresses (root URIs) handled by this server
- Query routing patterns to determine which addresses this server handles

This enables clients to gracefully handle different service capabilities
and route requests to the appropriate service instances via HTTP endpoints.
"""

import urllib.parse
from typing import (
    Any,
)

from starlette import status

from omni_onedrive_storage_service.backends import (
    get_backend,
)

from .rest_messages import (
    HTTPValidationError,
    ListRoutesResponse,
    ListServicesResponse,
    ListTopLevelAddressesResponse,
    Route,
    ServiceEntry,
    TopLevelAddress,
)
from .routes import (
    capabilities_service,
    capabilities_service_alpha,
)

list_services_route: dict[str, Any] = {
    "path": "/services",
    "responses": {
        status.HTTP_200_OK: {
            "description": "Success response.",
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "content": {"application/json": HTTPValidationError.model_json_schema()}
        },
    },
}


async def list_services_api_beta() -> ListServicesResponse:
    """List all available API services and their supported versions for the v1beta endpoint.

    This is the primary service discovery endpoint. Clients call this to
    determine which Storage API services are available and what versions
    they support, enabling graceful feature detection.

    Returns:
        ListServicesResponse containing a list of ServiceEntry messages,
        each specifying a service name and its available versions.

    Example Response:
        {
            "services": [
                {"service_name": "fileobject", "service_versions": ["v1beta"]},
            ]
        }
    """
    return ListServicesResponse(
        services=[
            ServiceEntry(service_name="capabilities", service_versions=["v1beta"]),
            ServiceEntry(service_name="fileobject", service_versions=["v1beta"]),
            ServiceEntry(service_name="filefolder", service_versions=["v1beta"]),
            ServiceEntry(service_name="versioning", service_versions=["v1beta"]),
        ]
    )


async def list_services_api_alpha() -> ListServicesResponse:
    """List all available API services and their supported versions for the v1alpha endpoint.

    This is the primary service discovery endpoint. Clients call this to
    determine which Storage API services are available and what versions
    they support, enabling graceful feature detection.

    Returns:
        ListServicesResponse containing a list of ServiceEntry messages,
        each specifying a service name and its available versions.

    Example Response:
        {
            "services": [
                {"service_name": "fileobject", "service_versions": ["v1alpha", "v1beta"]},
                {"service_name": "versioning", "service_versions": ["v1alpha", "v1beta"]}
            ]
        }

    Note:
        Alpha versions include experimental services and test versions.
        Beta versions include only stable, production-ready services.
    """
    # Note: metadata service not advertised - OneDrive doesn't support custom metadata
    return ListServicesResponse(
        services=[
            ServiceEntry(service_name="capabilities", service_versions=["v1alpha", "v1beta"]),
            ServiceEntry(service_name="fileobject", service_versions=["v1alpha", "v1beta"]),
            ServiceEntry(service_name="filefolder", service_versions=["v1alpha", "v1beta"]),
            ServiceEntry(service_name="versioning", service_versions=["v1alpha", "v1beta"]),
        ]
    )


capabilities_service.add_api_route(**list_services_route, endpoint=list_services_api_beta, methods=["GET"])
capabilities_service_alpha.add_api_route(**list_services_route, endpoint=list_services_api_alpha, methods=["GET"])


@capabilities_service.get(
    "/top-level-addresses",
)
@capabilities_service_alpha.get(
    "/top-level-addresses",
)
async def list_top_level_addresses_api() -> ListTopLevelAddressesResponse:
    """List the top-level addresses (root URIs) served by this storage service.

    Returns the base URIs that this service handles. Clients use this to
    discover what URI schemes/prefixes are supported by this service instance.

    Returns:
        ListTopLevelAddressesResponse containing the base URIs served,
        such as ['file-storage://fileservice'] or ['omniverse://server/path'].

    Example Response:
        {
            \"items\": [
                {\"top_level_address\": \"file-storage://fileservice\"}
            ]
        }

    Note:
        For a multi-cloud service, this might return multiple URIs like
        ['s3://bucket', 'omniverse://server', 'https://azure.microsoft.com/container'].
    """
    # Get the base URI from the current backend
    backend = get_backend()
    return ListTopLevelAddressesResponse(
        items=[
            TopLevelAddress(top_level_address=backend.base_uri),
        ]
    )


@capabilities_service_alpha.get(
    "/routes",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Success response.",
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "content": {"application/json": HTTPValidationError.model_json_schema()}
        },
    },
)
async def list_routes_api() -> ListRoutesResponse:
    """List the routing patterns that this service handles.

    Returns wildcard patterns for resource addresses that this service
    can process. Clients can use this to determine if a given address
    will be handled by this service without making a request.

    Returns:
        ListRoutesResponse containing wildcard patterns:
        - Base URI pattern (e.g., 'file-storage://fileservice/**')
        - Identity schema pattern (e.g., 'omniverse-identity://fileservice/**')

    Example Response:
        {
            \"items\": [
                {\"wildcard_pattern\": \"file-storage://fileservice/**\"},
                {\"wildcard_pattern\": \"omniverse-identity://fileservice/**\"}
            ]
        }

    Note:
        This method is only available in v1alpha, not in v1beta.
        The '**' wildcard matches any path under the base.
    """
    parsed_base = urllib.parse.urlparse(get_backend().base_uri)
    return ListRoutesResponse(
        items=[
            Route(wildcard_pattern=f"{get_backend().base_uri}**"),
            Route(wildcard_pattern=f"{get_backend().IDENTITY_SCHEMA}://{parsed_base.netloc}/**"),  # ty: ignore[unresolved-attribute]
        ]
    )

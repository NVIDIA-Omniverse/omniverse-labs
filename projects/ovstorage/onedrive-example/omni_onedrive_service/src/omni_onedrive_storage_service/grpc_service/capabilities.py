# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Capabilities service implementation for gRPC.

This module implements the CapabilitiesService from the Storage API, which
provides service discovery functionality. It allows clients to:
- List available services and their versions
- Discover top-level addresses (root URIs) handled by this server
- Query routing patterns to determine which addresses this server handles

This enables clients to gracefully handle different service capabilities
and route requests to the appropriate service instances.
"""

import urllib.parse

from omni_onedrive_storage_service.backends import get_backend


def make_filesystem_capabilities_servicer(pb2_version, pb2_grpc_version, version_tag):
    """Create a dynamic CapabilitiesService servicer for gRPC.

    This factory function creates a gRPC servicer class that implements the
    CapabilitiesService interface for service discovery. It adapts to different
    API versions (v1alpha, v1beta) with version-specific features. The base URI
    advertised in discovery responses is read from the active backend
    (``get_backend().base_uri``) at request time.

    Args:
        pb2_version: The capabilities protobuf module containing message types.
        pb2_grpc_version: The capabilities gRPC module containing the servicer
                         base class.
        version_tag: API version string ('v1alpha' or 'v1beta').
                    - v1alpha includes all services and the ListRoutes method
                    - v1beta includes only stable services, no ListRoutes

    Returns:
        An instance of a dynamically created CapabilitiesServiceServicer class
        with methods implemented based on the version_tag.

    Raises:
        RuntimeError: If version_tag is not 'v1alpha' or 'v1beta'.
    """

    def ListServices(self, request, context):
        """List all available API services and their supported versions.

        This is the primary service discovery endpoint. Clients call this to
        determine which Storage API services are available and what versions
        they support, enabling graceful feature detection.

        Args:
            request: ListServicesRequest (currently empty).
            context: gRPC ServicerContext for the request.

        Returns:
            ListServicesResponse containing a list of ServiceEntry messages,
            each specifying a service name and its available versions.

        Services Returned (v1alpha):
            - fileobject: v1alpha, v1beta
            - filefolder: v1alpha, v1beta
            - versioning: v1alpha, v1beta
            - capabilities: v1alpha, v1beta

        Services Returned (v1beta):
            - fileobject: v1beta
            - filefolder: v1beta
            - versioning: v1beta
            - capabilities: v1beta

        Note:
            Alpha versions include experimental services and test versions.
            Beta versions include only stable, production-ready services.
        """
        services = [
            pb2_version.ServiceEntry(service_name="fileobject", service_versions=["v1beta"]),
            pb2_version.ServiceEntry(service_name="filefolder", service_versions=["v1beta"]),
            pb2_version.ServiceEntry(service_name="versioning", service_versions=["v1beta"]),
            pb2_version.ServiceEntry(service_name="capabilities", service_versions=["v1beta"]),
        ]
        alpha_services = [
            pb2_version.ServiceEntry(service_name="fileobject", service_versions=["v1alpha"]),
            pb2_version.ServiceEntry(service_name="filefolder", service_versions=["v1alpha"]),
            pb2_version.ServiceEntry(service_name="versioning", service_versions=["v1alpha"]),
            pb2_version.ServiceEntry(service_name="capabilities", service_versions=["v1alpha"]),
        ]
        # Note: metadata service not advertised - OneDrive doesn't support custom metadata

        if version_tag == "v1alpha":
            return pb2_version.ListServicesResponse(services=services + alpha_services)
        elif version_tag == "v1beta":
            return pb2_version.ListServicesResponse(services=services)
        else:
            raise RuntimeError(f"Unknown version_tag given for configuration of capability service: {version_tag}")

    def ListRoutes(self, request, context):
        """List the routing patterns that this service handles.

        Returns wildcard patterns for resource addresses that this service
        can process. Clients can use this to determine if a given address
        will be handled by this service without making a request.

        Args:
            request: ListRoutesRequest (currently empty).
            context: gRPC ServicerContext for the request.

        Returns:
            ListRoutesResponse containing wildcard patterns:
            - Base URI pattern (e.g., 'file-storage://fileservice/**')
            - Identity schema pattern (e.g., 'omniverse-identity://fileservice/**')

        Note:
            This method is only available in v1alpha, not in v1beta.
            The '**' wildcard matches any path under the base.
        """
        parsed_base = urllib.parse.urlparse(get_backend().base_uri)
        return pb2_version.ListRoutesResponse(
            items=[
                pb2_version.Route(wildcard_pattern=f"{get_backend().base_uri}**"),
                pb2_version.Route(wildcard_pattern=f"{get_backend().IDENTITY_SCHEMA}://{parsed_base.netloc}/**"),  # ty: ignore[unresolved-attribute]
            ]
        )

    def ListTopLevelAddresses(self, request, context):
        """List the top-level addresses (root URIs) served by this storage service.

        Returns the base URIs that this service handles. Clients use this to
        discover what URI schemes/prefixes are supported by this service instance.

        Args:
            request: ListTopLevelAddressesRequest (currently empty).
            context: gRPC ServicerContext for the request.

        Returns:
            ListTopLevelAddressesResponse containing the base URIs served,
            such as ['file-storage://fileservice'] or ['omniverse://server/path'].

        Note:
            For a multi-cloud service, this might return multiple URIs like
            ['s3://bucket', 'omniverse://server', 'https://azure.microsoft.com/container'].
        """
        # Get the base URI from the current backend
        backend = get_backend()
        return pb2_version.ListTopLevelAddressesResponse(
            items=[pb2_version.TopLevelAddressEntry(top_level_address=backend.base_uri)]
        )

    if version_tag == "v1alpha":
        cls = type(
            "FileSystemCapabilitiesServicer",
            (pb2_grpc_version.CapabilitiesServiceServicer,),
            {
                "ListServices": ListServices,
                "ListTopLevelAddresses": ListTopLevelAddresses,
                "ListRoutes": ListRoutes,
            },
        )
    elif version_tag == "v1beta":
        cls = type(
            "FileSystemCapabilitiesServicer",
            (pb2_grpc_version.CapabilitiesServiceServicer,),
            {
                "ListServices": ListServices,
                "ListTopLevelAddresses": ListTopLevelAddresses,
            },
        )
    else:
        raise RuntimeError(f"Invalid version specified, use one of v1alpha, v1beta: {version_tag}")

    return cls()

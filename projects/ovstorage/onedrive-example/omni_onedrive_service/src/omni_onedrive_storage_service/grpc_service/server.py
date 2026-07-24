# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Implementation of a Storage API for a local file system."""

import logging
import os
from concurrent import futures

import grpc
import uvicorn

from nvidia.omniverse.storage.capabilities.v1alpha import (
    capabilities_pb2 as pb2_capabilities_v1alpha,
)
from nvidia.omniverse.storage.capabilities.v1alpha import (
    capabilities_pb2_grpc as pb2_grpc_capabilities_v1alpha,
)
from nvidia.omniverse.storage.capabilities.v1beta import (
    capabilities_pb2 as pb2_capabilities_v1beta,
)
from nvidia.omniverse.storage.capabilities.v1beta import (
    capabilities_pb2_grpc as pb2_grpc_capabilities_v1beta,
)
from nvidia.omniverse.storage.filefolder.v1alpha import (
    filefolder_service_pb2 as filefolder_service_pb2_v1alpha,
)
from nvidia.omniverse.storage.filefolder.v1alpha import (
    filefolder_service_pb2_grpc as filefolder_service_pb2_grpc_v1alpha,
)
from nvidia.omniverse.storage.filefolder.v1beta import (
    filefolder_service_pb2 as filefolder_service_pb2_v1beta,
)
from nvidia.omniverse.storage.filefolder.v1beta import (
    filefolder_service_pb2_grpc as filefolder_service_pb2_grpc_v1beta,
)
from nvidia.omniverse.storage.fileobject.v1alpha import (
    fileobject_pb2 as fileobject_pb2_v1alpha,
)
from nvidia.omniverse.storage.fileobject.v1alpha import (
    fileobject_service_pb2 as fileobject_service_pb2_v1alpha,
)
from nvidia.omniverse.storage.fileobject.v1alpha import (
    fileobject_service_pb2_grpc as fileobject_service_pb2_grpc_v1alpha,
)
from nvidia.omniverse.storage.fileobject.v1beta import (
    fileobject_pb2 as fileobject_pb2_v1beta,
)
from nvidia.omniverse.storage.fileobject.v1beta import (
    fileobject_service_pb2 as fileobject_service_pb2_v1beta,
)
from nvidia.omniverse.storage.fileobject.v1beta import (
    fileobject_service_pb2_grpc as fileobject_service_pb2_grpc_v1beta,
)
from nvidia.omniverse.storage.metadata.v1alpha.metadata_pb2_grpc import (
    add_MetadataServiceServicer_to_server,
)
from nvidia.omniverse.storage.versioning.v1alpha import (
    versioning_pb2 as versioning_pb2_v1alpha,
)
from nvidia.omniverse.storage.versioning.v1alpha import (
    versioning_pb2_grpc as versioning_pb2_grpc_v1alpha,
)
from nvidia.omniverse.storage.versioning.v1beta import (
    versioning_pb2 as versioning_pb2_v1beta,
)
from nvidia.omniverse.storage.versioning.v1beta import (
    versioning_pb2_grpc as versioning_pb2_grpc_v1beta,
)
from omni_onedrive_storage_service.grpc_service.capabilities import (
    make_filesystem_capabilities_servicer,
)
from omni_onedrive_storage_service.grpc_service.filefolder import (
    make_filefolder_service_servicer,
)
from omni_onedrive_storage_service.grpc_service.fileobject import (
    make_fileobject_service_servicer,
)
from omni_onedrive_storage_service.grpc_service.interceptors import (
    ExceptionLoggingInterceptor,
)
from omni_onedrive_storage_service.grpc_service.metadata import FilesystemMetadataService
from omni_onedrive_storage_service.grpc_service.versioning import make_versioning_service

REDIRECT_HOST = os.getenv("REDIRECT_HOST", "http://localhost")

# Optional TLS for the gRPC server. When a cert + key are configured the server
# serves gRPC over TLS; otherwise it falls back to a plaintext port (only safe
# behind a TLS-terminating proxy/ingress on a trusted network).
GRPC_TLS_CERT_PATH = os.getenv("GRPC_TLS_CERT_PATH", "")
GRPC_TLS_KEY_PATH = os.getenv("GRPC_TLS_KEY_PATH", "")
GRPC_TLS_CLIENT_CA_PATH = os.getenv("GRPC_TLS_CLIENT_CA_PATH", "")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _bind_grpc_port(server: grpc.Server, port: int) -> None:
    """Bind the gRPC server port, using TLS when a cert/key are configured.

    Set ``GRPC_TLS_CERT_PATH`` and ``GRPC_TLS_KEY_PATH`` to serve gRPC over TLS
    (``add_secure_port``). Optionally set ``GRPC_TLS_CLIENT_CA_PATH`` to require
    and verify client certificates (mTLS).

    When no cert/key are configured the server falls back to an insecure
    (plaintext) port. Bearer tokens travel in gRPC metadata, so a plaintext port
    is only safe when TLS is terminated by a trusted proxy/ingress in front of
    the pod (as the bundled Helm chart does via a Contour HTTPProxy) and the pod
    network is otherwise trusted. Never expose an insecure port directly to an
    untrusted network.
    """
    if GRPC_TLS_CERT_PATH and GRPC_TLS_KEY_PATH:
        with open(GRPC_TLS_KEY_PATH, "rb") as key_file:
            private_key = key_file.read()
        with open(GRPC_TLS_CERT_PATH, "rb") as cert_file:
            certificate_chain = cert_file.read()
        root_certificates = None
        require_client_auth = False
        if GRPC_TLS_CLIENT_CA_PATH:
            with open(GRPC_TLS_CLIENT_CA_PATH, "rb") as ca_file:
                root_certificates = ca_file.read()
            require_client_auth = True
        credentials = grpc.ssl_server_credentials(
            [(private_key, certificate_chain)],
            root_certificates=root_certificates,
            require_client_auth=require_client_auth,
        )
        server.add_secure_port(f"[::]:{port}", credentials)
        logger.info(f"gRPC server using TLS (mTLS={'on' if require_client_auth else 'off'})")
    else:
        server.add_insecure_port(f"[::]:{port}")
        logger.warning(
            "gRPC server listening on an INSECURE (plaintext) port. Bearer tokens are "
            "sent in gRPC metadata; only expose this behind a TLS-terminating proxy/ingress "
            "on a trusted network, or set GRPC_TLS_CERT_PATH and GRPC_TLS_KEY_PATH to enable TLS."
        )


def startGRPCserver(port: int, http_port: int, extra_interceptors: list | None = None) -> grpc.Server:
    """Start the gRPC server with all Storage API services registered.

    Initializes and starts a gRPC server with all required services:
    - FileObjectService (v1alpha, v1beta)
    - FileFolderService (v1alpha, v1beta)
    - CapabilitiesService (v1alpha, v1beta)
    - VersioningService (v1alpha, v1beta)
    - MetadataService (v1alpha)

    Also enables gRPC reflection for service discovery.

    Args:
        port: Port number for the gRPC server.
        http_port: Port number for HTTP redirect endpoints (used for
                  constructing redirect URLs in upload/download operations).

    Returns:
        The started grpc.Server instance (non-blocking).

    Note:
        The server uses a ThreadPoolExecutor with 10 workers.
        The server is started in non-blocking mode; call server.wait_for_termination()
        or handle shutdown separately.
    """
    interceptors = list(extra_interceptors or []) + [ExceptionLoggingInterceptor()]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), interceptors=interceptors)
    (
        fileobject_service_pb2_grpc_v1alpha.add_FileObjectServiceServicer_to_server(
            make_fileobject_service_servicer(
                REDIRECT_HOST,
                fileobject_pb2_v1alpha,
                fileobject_service_pb2_v1alpha,
                fileobject_service_pb2_grpc_v1alpha,
                redirect_port=http_port,
                version_tag="v1alpha",
            ),
            server,
        ),
    )
    fileobject_service_pb2_grpc_v1beta.add_FileObjectServiceServicer_to_server(
        make_fileobject_service_servicer(
            REDIRECT_HOST,
            fileobject_pb2_v1beta,
            fileobject_service_pb2_v1beta,
            fileobject_service_pb2_grpc_v1beta,
            redirect_port=http_port,
            version_tag="v1beta",
        ),
        server,
    )

    pb2_grpc_capabilities_v1alpha.add_CapabilitiesServiceServicer_to_server(
        make_filesystem_capabilities_servicer(pb2_capabilities_v1alpha, pb2_grpc_capabilities_v1alpha, "v1alpha"),
        server,
    )
    pb2_grpc_capabilities_v1beta.add_CapabilitiesServiceServicer_to_server(
        make_filesystem_capabilities_servicer(pb2_capabilities_v1beta, pb2_grpc_capabilities_v1beta, "v1beta"),
        server,
    )
    filefolder_service_pb2_grpc_v1alpha.add_FileFolderServiceServicer_to_server(
        make_filefolder_service_servicer(
            fileobject_pb2_v1alpha,
            filefolder_service_pb2_v1alpha,
            filefolder_service_pb2_grpc_v1alpha,
            is_alpha=True,
        ),
        server,
    )
    filefolder_service_pb2_grpc_v1beta.add_FileFolderServiceServicer_to_server(
        make_filefolder_service_servicer(
            fileobject_pb2_v1beta,
            filefolder_service_pb2_v1beta,
            filefolder_service_pb2_grpc_v1beta,
            is_alpha=False,
        ),
        server,
    )
    versioning_pb2_grpc_v1alpha.add_VersioningServiceServicer_to_server(
        make_versioning_service(
            fileobject_pb2_v1alpha,
            versioning_pb2_v1alpha,
            versioning_pb2_grpc_v1alpha,
            True,
        ),
        server,
    )
    versioning_pb2_grpc_v1beta.add_VersioningServiceServicer_to_server(
        make_versioning_service(
            fileobject_pb2_v1beta,
            versioning_pb2_v1beta,
            versioning_pb2_grpc_v1beta,
            False,
        ),
        server,
    )

    # Alpha and test versions only, no need to use make_... factory functions
    add_MetadataServiceServicer_to_server(FilesystemMetadataService(), server)

    # Enable reflection
    from grpc_reflection.v1alpha import reflection

    SERVICE_NAMES = (
        pb2_capabilities_v1alpha.DESCRIPTOR.services_by_name["CapabilitiesService"].full_name,
        pb2_capabilities_v1beta.DESCRIPTOR.services_by_name["CapabilitiesService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    _bind_grpc_port(server, port)
    server.start()  # Start the server, non-blocking
    logger.info(f"gRPC Server launched on port {port}")
    return server


def createStaticServer(app, port: int) -> uvicorn.Server:
    """Create a Uvicorn server for serving HTTP endpoints.

    Creates (but doesn't start) a Uvicorn server that hosts the FastAPI
    application with upload/download endpoints used by redirect-based
    transfers.

    Args:
        app: FastAPI application instance with registered endpoints.
        port: Port number for the HTTP server.

    Returns:
        Configured uvicorn.Server instance (not yet started).

    Note:
        The server is configured with INFO log level and custom log formatting.
    """
    # Create the Uvicorn Server instance
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"

    # Ensure gRPC loggers aren't disabled by uvicorn
    # Set all loggers to propagate to root logger
    log_config["loggers"]["omni_onedrive_storage_service"] = {
        "handlers": ["default"],
        "level": "INFO",
        "propagate": False,
    }

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info", log_config=log_config)
    return uvicorn.Server(config)


def run_static_server(server: uvicorn.Server):
    """Run the FastAPI/Uvicorn server (blocking).

    Starts the Uvicorn server and blocks until it's shut down. This is
    typically run in a separate thread.

    Args:
        server: Configured uvicorn.Server instance.

    Note:
        This is a blocking call. Run in a thread if concurrent operation
        with gRPC server is needed.
    """
    logger.info("Starting static server...")
    server.run()  # Run the Uvicorn server, blocking

# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""NVIDIA Omniverse Storage API - Combined gRPC and REST Service

This module provides the main entry point for running both gRPC and REST services
together. Each storage backend has its own subcommand with backend-specific parameters.

Example Usage:
    # Start with the OneDrive backend (also the default when no subcommand is given)
    python -m omni_onedrive_storage_service onedrive --tenant-id ... --oidc-client-id ...

    # Start with custom ports
    python -m omni_onedrive_storage_service --grpc-port 50052 --http-port 8012 onedrive

    # Start only REST server
    python -m omni_onedrive_storage_service --no-grpc onedrive

    # See all available backends
    python -m omni_onedrive_storage_service --help

    # See backend-specific options
    python -m omni_onedrive_storage_service onedrive --help

Environment Variables:
    Configuration can be provided via environment variables or a .env file.
    The service automatically loads .env files from the current directory
    and parent directories.
"""

from __future__ import annotations

import inspect
import logging
import os
import signal
import socket
import sys
import threading
import time

from dotenv import load_dotenv

# Load environment variables from .env file before any other imports
# This ensures env vars are available for typer's envvar parameter
load_dotenv()

from typing import Annotated

import typer

import omni_onedrive_storage_service.onedrive  # noqa: F401 - import for backend registration
from omni_onedrive_storage_service.backends import (
    BackendConfig,
    get_backend,
    get_backend_cli_commands,
    init_backend,
    list_backends,
)
from omni_onedrive_storage_service.grpc_service.server import (
    createStaticServer,
    run_static_server,
    startGRPCserver,
)
from omni_onedrive_storage_service.rest_service import app

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FileSystemService")

# Create Typer app
cli_app = typer.Typer(
    name="omni-onedrive-storage-service",
    help="NVIDIA Omniverse Storage API - Combined gRPC and REST Service",
    add_completion=False,
)

# Global variables for server management
exiting = False
grpc_server = None
static_server = None
fastapi_thread = None


def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
            return False
        except OSError:
            return True


def handle_sigint(signum, frame):
    """Handle SIGINT to gracefully shut down servers."""
    global exiting, grpc_server, static_server
    logger.info("Received SIGINT, shutting down!")
    exiting = True
    if grpc_server:
        grpc_server.stop(0)  # Immediate shutdown
    if static_server:
        static_server.should_exit = True  # Signal exit


def _run_services(
    backend_config: BackendConfig,
    backend_name: str,
    grpc_port: int,
    http_port: int,
    enable_grpc: bool,
    enable_rest: bool,
):
    """Shared logic to initialize backend and run servers."""
    global exiting, grpc_server, static_server, fastapi_thread

    # Check for port conflicts before starting
    ports_in_use = []
    if enable_grpc and is_port_in_use(grpc_port):
        ports_in_use.append(f"gRPC port {grpc_port}")
    if enable_rest and is_port_in_use(http_port):
        ports_in_use.append(f"HTTP port {http_port}")

    if ports_in_use:
        logger.error(f"Port conflict detected: {', '.join(ports_in_use)} already in use")
        logger.error("Another service may be running. Stop it first or use different ports.")
        raise typer.Exit(1)

    # Initialize storage backend
    try:
        init_backend(backend_config)
        logger.info(f"Initialized {backend_name} storage backend")
        logger.info(f"  Base URI: {backend_config.base_uri}")
    except ValueError as e:
        logger.error(f"Failed to initialize backend: {e}")
        raise typer.Exit(1) from None

    # Register backend-specific HTTP routes AFTER backend is initialized
    backend = get_backend()
    backend.register_http_routes(app)
    logger.info("Registered backend HTTP routes")

    # Track whether request authentication was actually wired up for each
    # transport. The service must fail closed rather than ever serving an
    # unauthenticated backend (defense-in-depth so a future backend that forgets
    # to register auth cannot be exposed on 0.0.0.0 without it).
    rest_auth_enabled = False
    grpc_auth_enabled = False

    # OneDrive-specific: add Bearer token middleware and discovery routes
    if backend_name == "onedrive":
        from omni_onedrive_storage_service.onedrive import OneDriveConfig
        from omni_onedrive_storage_service.onedrive.discovery import register_discovery_routes
        from omni_onedrive_storage_service.onedrive.middleware import BearerTokenMiddleware
        from omni_onedrive_storage_service.onedrive.onedrive_provider import OneDriveStorageProvider

        if isinstance(backend, OneDriveStorageProvider) and isinstance(backend_config, OneDriveConfig):
            app.add_middleware(BearerTokenMiddleware, graph_client=backend.graph_client)
            register_discovery_routes(app, backend_config, grpc_port, http_port)
            rest_auth_enabled = True
            logger.info("Registered OneDrive Bearer token middleware and discovery endpoints")

    # CORS: allow browser-based clients (Kit, Streaming Portal) to call the API.
    # Origins are configurable via CORS_ALLOWED_ORIGINS (comma-separated); defaults
    # to "*". A wildcard MUST NOT be combined with credentials — browsers reject
    # it and Starlette would otherwise reflect any Origin, allowing any site to
    # make credentialed cross-origin requests. Credentials are therefore only
    # enabled when explicit origins are configured. The service authenticates via
    # the Bearer token in the Authorization header (set explicitly by clients),
    # which is not a CORS "credential", so this does not affect Kit clients.
    from starlette.middleware.cors import CORSMiddleware

    cors_origins = [o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
    allow_credentials = cors_origins != ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Start servers
    extra_interceptors = []
    if backend_name == "onedrive":
        from omni_onedrive_storage_service.onedrive.middleware import BearerTokenInterceptor
        from omni_onedrive_storage_service.onedrive.onedrive_provider import OneDriveStorageProvider

        if isinstance(backend, OneDriveStorageProvider):
            extra_interceptors.append(BearerTokenInterceptor(backend.graph_client))
            grpc_auth_enabled = True

    # Fail closed: refuse to expose any enabled transport without authentication.
    missing_auth = []
    if enable_rest and not rest_auth_enabled:
        missing_auth.append("REST")
    if enable_grpc and not grpc_auth_enabled:
        missing_auth.append("gRPC")
    if missing_auth:
        logger.error(
            f"Refusing to start: no request authentication configured for {', '.join(missing_auth)} "
            f"with backend '{backend_name}'. The service must not serve an unauthenticated backend."
        )
        raise typer.Exit(1)

    if enable_grpc:
        logger.info(f"Starting gRPC server on port {grpc_port}")
        grpc_server = startGRPCserver(grpc_port, http_port, extra_interceptors=extra_interceptors)
        logger.info(f"  gRPC endpoint: localhost:{grpc_port}")

    if enable_rest:
        logger.info(f"Starting HTTP/REST server on port {http_port}")
        static_server = createStaticServer(app, http_port)
        fastapi_thread = threading.Thread(target=run_static_server, args=(static_server,))
        fastapi_thread.start()
        logger.info(f"  REST endpoint: http://localhost:{http_port}")
        logger.info(f"  OpenAPI docs: http://localhost:{http_port}/docs")

    if not enable_grpc and not enable_rest:
        logger.warning("Both gRPC and REST servers are disabled. Exiting.")
        return

    # Set up signal handling and run
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)
    logger.info("Services started successfully")

    try:
        while not exiting:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, forcing shutdown...")
        exiting = True
        if grpc_server:
            grpc_server.stop(0)
        if static_server:
            static_server.should_exit = True

    logger.debug("Waiting for servers to terminate...")
    if fastapi_thread and fastapi_thread.is_alive():
        fastapi_thread.join(5)
    logger.info("Server shut down successfully.")


# Context class to pass common options to subcommands
class ServiceContext:
    def __init__(self):
        self.grpc_port = 50051
        self.http_port = 8011
        self.enable_grpc = True
        self.enable_rest = True


@cli_app.callback()
def common_options(
    ctx: typer.Context,
    grpc_port: Annotated[
        int,
        typer.Option(
            "--grpc-port",
            help="Port for gRPC server",
            envvar="GRPC_SERVER_PORT",
        ),
    ] = 50051,
    http_port: Annotated[
        int,
        typer.Option(
            "--http-port",
            help="Port for HTTP/REST server",
            envvar="HTTP_SERVER_PORT",
        ),
    ] = 8011,
    enable_grpc: Annotated[
        bool,
        typer.Option(
            "--grpc/--no-grpc",
            help="Enable gRPC server",
        ),
    ] = True,
    enable_rest: Annotated[
        bool,
        typer.Option(
            "--rest/--no-rest",
            help="Enable REST server",
        ),
    ] = True,
):
    """NVIDIA Omniverse Storage API - Combined gRPC and REST Service.

    Provides both gRPC and REST interfaces for the Storage API. Choose a backend
    subcommand to start the service with that storage backend. Omitting the
    subcommand defaults to the authenticated ``onedrive`` backend, configured via
    environment variables.

    Common options apply to all backends and control server ports.
    """
    # Store common options in context for subcommands to access
    ctx.obj = ServiceContext()
    ctx.obj.grpc_port = grpc_port
    ctx.obj.http_port = http_port
    ctx.obj.enable_grpc = enable_grpc
    ctx.obj.enable_rest = enable_rest


@cli_app.command(name="list-backends")
def list_backends_command():
    """List all available storage backends."""
    available_backends = list_backends()
    typer.echo("Available storage backends:")
    for backend_name in available_backends:
        typer.echo(f"  {backend_name}")
    typer.echo("\nUse 'python -m omni_onedrive_storage_service BACKEND --help' for backend-specific options.")


def create_backend_command(backend_name: str, backend_cli_func):
    """Create a command function for a specific backend.

    This wraps the backend's CLI function to initialize the backend and start services.
    """

    def backend_command_wrapper(ctx: typer.Context, **backend_kwargs):
        """Start service with the specified backend."""
        backend_config: BackendConfig = backend_cli_func(**backend_kwargs)
        _run_services(
            backend_config,
            backend_name,
            ctx.obj.grpc_port,
            ctx.obj.http_port,
            ctx.obj.enable_grpc,
            ctx.obj.enable_rest,
        )

    # Create a merged signature: ctx + all backend parameters
    backend_sig = inspect.signature(backend_cli_func)
    ctx_param = inspect.Parameter("ctx", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=typer.Context)
    new_params = [ctx_param] + list(backend_sig.parameters.values())
    backend_command_wrapper.__signature__ = inspect.Signature(new_params)  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    backend_command_wrapper.__annotations__ = {"ctx": typer.Context, **backend_cli_func.__annotations__}  # type: ignore[attr-defined]
    backend_command_wrapper.__doc__ = backend_cli_func.__doc__

    return backend_command_wrapper


# Dynamically register all backend CLI commands as subcommands
for backend_name, backend_cli_func in get_backend_cli_commands().items():
    command_func = create_backend_command(backend_name, backend_cli_func)
    cli_app.command(name=backend_name)(command_func)


def _default_to_onedrive(argv: list[str]) -> None:
    """Default a bare invocation to the authenticated OneDrive backend.

    If the CLI args contain no known subcommand, ``--help``/``-h``, or
    ``list-backends``, inject the ``onedrive`` subcommand so Typer resolves every
    OneDrive option from its environment variable (failing fast with a clear
    "Missing option" error if the required ``AZURE_TENANT_ID`` / ``OIDC_CLIENT_ID``
    are absent). This keeps a bare ``omni-onedrive-storage-service`` equivalent to
    ``... onedrive`` while never starting an unauthenticated backend.
    """
    known = set(get_backend_cli_commands()) | {"list-backends"}
    for arg in argv:
        if arg in ("--help", "-h") or arg in known:
            return
    argv.append("onedrive")


def main():
    """Entry point for the omni-onedrive-storage-service console script."""
    _default_to_onedrive(sys.argv)
    cli_app()


if __name__ == "__main__":
    main()

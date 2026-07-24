# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""FastAPI middleware and gRPC interceptor for Bearer token extraction and drive resolution.

Extracts the user's Bearer token from incoming requests, resolves their
OneDrive drive ID (with caching), and sets both values in request-scoped
context variables for downstream use by the storage backend.
"""

import logging
from collections.abc import Generator
from typing import Any

import grpc
from fastapi import Request
from fastapi.responses import JSONResponse
from grpc_interceptor import ServerInterceptor
from starlette.middleware.base import BaseHTTPMiddleware

from .graph_client import GraphClient
from .request_context import (
    get_cached_drive_id,
    request_drive_id,
    request_token,
    set_cached_drive_id,
)

logger = logging.getLogger(__name__)

AUTH_CONFIG_PATHS = {"/api/v1/auth-config", "/api/v1alpha/auth-config"}


def _extract_bearer_token(auth_header: str | None) -> str | None:
    """Extract Bearer token from an Authorization header value."""
    if not auth_header:
        return None
    parts = auth_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1]


def _resolve_drive_id(token: str, graph_client: GraphClient) -> str:
    """Resolve the user's drive ID, using cache if available."""
    cached = get_cached_drive_id(token)
    if cached is not None:
        return cached

    drive_id = graph_client.get_my_drive()
    set_cached_drive_id(token, drive_id)
    return drive_id


class BearerTokenMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that extracts Bearer tokens and resolves drives.

    Skips authentication only for /api/v1/auth-config (needed by the client
    library to bootstrap OIDC). All other paths including /api/v1/services
    require a Bearer token -- returning 401 is what triggers the client
    library's interactive auth flow.
    """

    def __init__(self, app, graph_client: GraphClient):
        super().__init__(app)
        self._graph_client = graph_client

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in AUTH_CONFIG_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("authorization")
        token = _extract_bearer_token(auth_header)

        if token is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authorization header with Bearer token is required"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        token_ctx = request_token.set(token)
        try:
            drive_id = _resolve_drive_id(token, self._graph_client)
            drive_ctx = request_drive_id.set(drive_id)
            try:
                return await call_next(request)
            finally:
                request_drive_id.reset(drive_ctx)
        except PermissionError:
            return JSONResponse(
                status_code=403,
                content={"detail": "Access denied to OneDrive. Check token permissions."},
            )
        except Exception as e:
            logger.error(f"Drive resolution failed: {e}")
            return JSONResponse(
                status_code=401,
                content={"detail": "Failed to resolve OneDrive. Token may be invalid or expired."},
            )
        finally:
            request_token.reset(token_ctx)


def _wrap_streaming_response(
    generator: Generator,
    token: str,
    drive_id: str,
    method_name: str,
    context: Any = None,
) -> Generator:
    """Wrap a streaming gRPC response generator so context vars stay set during iteration.

    Streaming methods (yield) return a lazy generator. The interceptor's finally
    block resets context vars before gRPC iterates the generator, so we
    re-establish them around each next() call on the inner generator.
    """
    item_count = 0
    while True:
        tok_ctx = request_token.set(token)
        drv_ctx = request_drive_id.set(drive_id)
        try:
            item = next(generator)
            item_count += 1
        except StopIteration:
            logger.info(f"gRPC {method_name}: stream completed ({item_count} items)")
            return
        except Exception as e:
            if context is not None and context.code() is not None:
                raise
            logger.error(f"gRPC {method_name}: stream error after {item_count} items: {e}", exc_info=True)
            raise
        finally:
            request_drive_id.reset(drv_ctx)
            request_token.reset(tok_ctx)
        yield item


class BearerTokenInterceptor(ServerInterceptor):
    """gRPC interceptor that extracts Bearer tokens and resolves drives."""

    def __init__(self, graph_client: GraphClient):
        self._graph_client = graph_client

    def intercept(self, method, request, context, method_name):  # ty: ignore[invalid-method-override]
        logger.info(f"gRPC request: {method_name}")
        metadata = dict(context.invocation_metadata())
        auth_value = metadata.get("authorization", "")
        token = _extract_bearer_token(auth_value)

        if token is None:
            logger.warning(f"gRPC UNAUTHENTICATED: {method_name} (no Bearer token)")
            context.abort(
                grpc.StatusCode.UNAUTHENTICATED,
                "Authorization metadata with Bearer token is required",
            )

        assert token is not None
        token_ctx = request_token.set(token)
        try:
            drive_id = _resolve_drive_id(token, self._graph_client)
            logger.info(f"gRPC {method_name}: drive resolved, proceeding")
            drive_ctx = request_drive_id.set(drive_id)
            try:
                result = method(request, context)
                if isinstance(result, Generator) or hasattr(result, "__next__"):
                    logger.info(f"gRPC {method_name}: returning streaming response")
                    return _wrap_streaming_response(result, token, drive_id, method_name, context)
                logger.info(f"gRPC {method_name}: completed successfully")
                return result
            except Exception as e:
                if context.code() is not None:
                    raise
                logger.error(f"gRPC {method_name}: method error: {e}", exc_info=True)
                raise
            finally:
                request_drive_id.reset(drive_ctx)
        except PermissionError:
            logger.warning(f"gRPC {method_name}: PERMISSION_DENIED")
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "Access denied to OneDrive")
        except Exception as e:
            if context.code() is not None:
                raise
            logger.error(f"gRPC {method_name}: drive resolution failed: {e}", exc_info=True)
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Failed to resolve OneDrive")
        finally:
            request_token.reset(token_ctx)

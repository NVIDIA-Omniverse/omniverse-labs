# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""gRPC server interceptors for cross-cutting concerns.

This module provides interceptors that handle common functionality across
all gRPC service methods, such as exception logging and error handling.
"""

import logging

import grpc
from grpc_interceptor import ServerInterceptor

logger = logging.getLogger(__name__)


class ExceptionLoggingInterceptor(ServerInterceptor):
    """Interceptor that logs unhandled exceptions from gRPC methods.

    This interceptor catches all exceptions that escape from gRPC servicer
    methods, logs them with full stack traces, and converts them to appropriate
    gRPC errors. This provides a single point for exception logging without
    needing to wrap every service method.

    Expected gRPC aborts (via context.abort()) are passed through unchanged.
    Only unexpected exceptions are caught and logged.
    """

    def intercept(self, method, request, context, method_name):  # ty: ignore[invalid-method-override]
        """Intercept a gRPC method call and handle exceptions.

        Args:
            method: The servicer method being called
            request: The request message
            context: The gRPC ServicerContext
            method_name: Name of the method being called

        Returns:
            The response from the method, or raises a gRPC error

        Raises:
            Exception: Expected abort from context.abort() or unexpected exceptions
        """
        try:
            return method(request, context)
        except Exception as e:
            # context.abort() raises a plain Exception after setting the error code
            # Check if an error code was already set on the context
            if context.code() is not None:
                # This is an expected abort - the code and details were already set
                # Just re-raise to let gRPC handle it
                raise

            # Truly unexpected exception - log with full stack trace
            logger.exception(f"Unexpected error in {method_name}: {e}")
            # Set error on context
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {type(e).__name__}")
            # Re-raise to let gRPC handle it
            raise

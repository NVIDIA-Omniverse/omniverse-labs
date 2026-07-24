# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""FastAPI application setup and route configuration for REST service.

This module sets up the main FastAPI application and mounts all service
sub-applications for different API versions. The Storage API uses semantic
versioning with alpha/beta/stable releases, where different API versions
can have different feature sets and are mounted separately.

Services are organized by version:
- v1alpha: Includes all services including experimental features (Copy, Move, ListRoutes)
- v1beta: Includes only stable, production-ready services

Each service is mounted at its versioned path (e.g., /v1alpha/fileobject).
"""

# =============================================================================
# FastAPI Application Setup
# =============================================================================
from fastapi import FastAPI

# Main FastAPI application
# HTTP routes for redirect-based operations will be registered lazily
# after backend initialization - see __main__.py
app = FastAPI()

# =============================================================================
# API Version Routing
# =============================================================================
# The Storage API uses semantic versioning with alpha/beta/stable releases.
# Different API versions can have different feature sets and are mounted separately.
# We follow channel based releases, so alpha in one specific version is a strict superset of beta, which in turn is a
# strict superset of the release version if any is available.

# Capabilities service - Advertises service capabilities and available routes
capabilities_service = FastAPI()
capabilities_service_alpha = FastAPI()
app.mount("/v1alpha/capabilities", capabilities_service_alpha)
app.mount("/v1beta/capabilities", capabilities_service)

# FileObject service - Core data object operations (read, write, delete, enumerate)
fileobject_service_beta = FastAPI()  # Beta version - no copy/move endpoints
fileobject_service_alpha = FastAPI()  # Alpha version - includes copy and move endpoints
app.mount("/v1alpha/fileobject", fileobject_service_alpha)
app.mount("/v1beta/fileobject", fileobject_service_beta)

# FileFolder service - Folder operations (list, create, delete folders)
filefolder_service_alpha = FastAPI()
filefolder_service_beta = FastAPI()
app.mount("/v1alpha/filefolder", filefolder_service_alpha)
app.mount("/v1beta/filefolder", filefolder_service_beta)

# Versioning service - Enumerate and manage file versions
versioning_service_alpha = FastAPI()
app.mount("/v1alpha/versioning", versioning_service_alpha)
versioning_service_beta = FastAPI()
app.mount("/v1beta/versioning", versioning_service_beta)

# Metadata service - User-defined key-value metadata storage
metadata_service = FastAPI()
app.mount("/v1alpha/metadata", metadata_service)

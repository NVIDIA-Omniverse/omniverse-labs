# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""gRPC service implementation for NVIDIA Omniverse Storage API.

This package provides gRPC-based access to the filesystem storage service,
implementing the NVIDIA Omniverse Storage API protocol over gRPC. It parallels
the REST service implementation but uses Protocol Buffers and gRPC for
communication.

Modules:
    server: gRPC server setup and initialization
    capabilities: Service discovery and capability advertisement
    fileobject: Core file operations (read, write, delete, enumerate)
    filefolder: Folder operations (list, create, delete)
    versioning: File version enumeration and management
    metadata: User-defined metadata storage and retrieval
    helpers: Common utility functions for gRPC handlers

The service supports multiple API versions (v1alpha, v1beta) and provides
both alpha and beta stability guarantees for different features.
"""

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# ---------------------------------------------------------------------------
# Multi-stage Dockerfile for the OneDrive Storage Service
#
# Build:  docker build -f omni_onedrive_service/Dockerfile -t onedrive-storage-service .
# Run:    docker run --env-file .env -p 8011:8011 -p 50051:50051 onedrive-storage-service
#
# Build context must be the repository root.
# ---------------------------------------------------------------------------

# --- Stage 1: install dependencies ------------------------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy only dependency metadata first for layer caching
COPY omni_onedrive_service/pyproject.toml omni_onedrive_service/uv.lock omni_onedrive_service/README.md ./

# Install production dependencies (no dev/test groups) into the system Python.
# --frozen ensures the lockfile is used exactly as committed.
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code (including gitignored protobuf stubs under src/nvidia/)
COPY omni_onedrive_service/src/ src/

# Install the project itself as a proper wheel (not editable) so the venv
# is self-contained and can be copied to the runtime stage.
RUN uv sync --frozen --no-dev --no-editable

# --- Stage 2: runtime image -------------------------------------------------
FROM python:3.12-slim

# Non-root user for defense-in-depth
RUN groupadd --system app && useradd --system --gid app app

WORKDIR /app

# Copy the virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Put the venv on PATH so entrypoints are available directly
ENV PATH="/app/.venv/bin:$PATH"

# REST and gRPC default ports
EXPOSE 8011
EXPOSE 50051

USER app

ENTRYPOINT ["omni-onedrive-storage-service"]
CMD ["onedrive"]

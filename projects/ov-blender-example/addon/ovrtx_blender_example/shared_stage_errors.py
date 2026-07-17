# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared-stage composition exceptions."""

from __future__ import annotations


class SharedStageCompositionError(RuntimeError):
    """Raised when the interactive shared-stage composition path fails."""

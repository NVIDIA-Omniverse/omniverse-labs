# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility imports; the SSP backend now belongs to :mod:`ovfmi`."""

from ovfmi._ssp_runtime import *  # noqa: F401,F403
from ovfmi._ssp_runtime import _default_system_connector_value  # noqa: F401

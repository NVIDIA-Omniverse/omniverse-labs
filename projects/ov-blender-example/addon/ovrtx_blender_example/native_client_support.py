# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Safe scalar decoding shared by current physics presentation."""
from __future__ import annotations
from typing import Any, Mapping

def coerce_mapping_int(mapping: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return int(mapping.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def coerce_mapping_float(mapping: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return float(default)

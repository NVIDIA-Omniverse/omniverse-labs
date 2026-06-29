# Adapted from usd_fmi (omni/source/extensions/fmi/source/fmi.usd.parser)
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

from dataclasses import dataclass
from enum import Enum


class FmuDirection(Enum):
    INPUT = "input"
    OUTPUT = "output"


@dataclass
class FmuParserMapping:
    fmiAttributeName: str
    usdAttributeName: str
    direction: FmuDirection
    usdMapping: tuple  # (component_offset, component_count); (0,0) = whole attribute


@dataclass
class FmuParserConnection:
    enabled: bool
    targets: list  # list[str] prim paths
    mappings: list  # list[FmuParserMapping]


@dataclass
class FmuParserInstance:
    enabled: bool
    fmu: str          # resolved path to .fmu file
    path: str         # prim path of the FmuInstance prim
    connections: list # list[FmuParserConnection]

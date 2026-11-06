# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Author PhysX schema data with OpenUSD; ovphysx loads the native schemas."""
from pxr import Sdf


def apply(prim, schema, attributes):
    # usd-core intentionally does not ship NVIDIA's Python schema bindings.
    # Author their documented USD representation; the actual ovphysx runtime
    # registers and interprets these schemas before the scene is populated.
    prim.AddAppliedSchema(schema)
    for name, (kind, value) in attributes.items():
        prim.CreateAttribute(name, Sdf.ValueTypeNames.Find(kind), custom=False).Set(value)

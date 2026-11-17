# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recorded array element layouts shared by the tree and property inspector."""
from .formatting import element_size


def array_layout(adapter, prop):
    type_name = prop.type
    # OVD stores these vector arrays as flat floats, unlike legacy structures.
    if (adapter.ovd and prop.name in ('pairsContactNormals', 'pairsContactPoints')
            and type_name in ('PvdF32', 'PxF32')):
        type_name = 'PxVec3'
    stride = element_size(type_name)
    if not stride:
        sizes = {getattr(cls, 'byte_size', 0) for cls in adapter.classes.values()
                 if cls.name == type_name}
        stride = sizes.pop() if len(sizes) == 1 else 0
    return type_name, stride


def inspection_requests(item):
    """Return bounded raw windows or legacy structure requests for one row."""
    if item is None or getattr(item, 'kind', '') == 'unresolved':
        return [], []
    if element_size(item.class_name):
        return [(item.source_instance_id, item.property_id,
                 item.element_index * item.element_size, item.element_size)], []
    return [], [(item.source_instance_id, item.property_id, item.element_index)]

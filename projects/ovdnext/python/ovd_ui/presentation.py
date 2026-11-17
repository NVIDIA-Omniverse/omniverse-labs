# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decisions based only on the native, owned presentation."""
from collections import Counter


def default_scene_index(scenes, geometry):
    """Choose a populated native scene, with the source order as a stable tie."""
    counts=Counter(map(int,geometry.scene_ids)) if geometry is not None else {}
    candidates=[i for i,scene in enumerate(scenes) if scene.alive] or list(range(len(scenes)))
    return max(candidates,key=lambda i:counts.get(scenes[i].instance_id,0),default=0)

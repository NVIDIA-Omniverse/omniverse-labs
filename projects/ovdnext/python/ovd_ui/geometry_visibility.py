# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The native geometry rows admitted by the viewport, before GPU batching."""
import numpy as np

from .render_settings import PlaneMode, display_mask


def geometry_visibility(snapshot, settings=None, *, scene_id=None, hidden=(),
                        selected=(), opacity=1., all_transparent=False):
    count = len(snapshot.object_ids)
    mask = np.ones(count, dtype=bool)
    if scene_id is not None:
        mask &= (snapshot.scene_ids == scene_id) | (snapshot.scene_ids <= 0)
    if hidden:
        ids = list(hidden)
        mask &= ~(np.isin(snapshot.object_ids, ids) | np.isin(snapshot.shape_ids, ids) |
                  np.isin(snapshot.geometry_ids, ids))
    ids = list(selected)
    selected_mask = (np.isin(snapshot.object_ids, ids) | np.isin(snapshot.shape_ids, ids) |
                     np.isin(snapshot.geometry_ids, ids))
    if settings is not None:
        sleeping = np.asarray(getattr(snapshot, "sleeping", np.zeros(count)), dtype=bool)
        mask &= ~sleeping | display_mask(settings.sleeping_objects, selected_mask)
        trigger = (np.asarray(getattr(snapshot, "shape_flags", np.zeros(count)), dtype="u4") & 4) != 0
        # The source's stored mode 2 excludes selected triggers.
        trigger_selected = ~selected_mask if settings.trigger_shapes in (2, 3) else selected_mask
        mask &= ~trigger | display_mask(settings.trigger_shapes, trigger_selected)
        if settings.planes in (PlaneMode.NONE, PlaneMode.BASIS) or all_transparent or opacity < .9999:
            mask &= snapshot.kinds != 4
        if not settings.show_particles or settings.particle_size <= 0:
            mask &= snapshot.kinds != 7
    return mask, selected_mask

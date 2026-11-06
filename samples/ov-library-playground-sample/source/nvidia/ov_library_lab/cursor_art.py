# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compact Lucide navigation cursors; provenance/license in data/cursors."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


@lru_cache(maxsize=2)
def _source(closed):
    name = 'hand-grab' if closed else 'hand'
    path = Path(__file__).resolve().parents[3]/'data'/'cursors'/(name+'.png')
    with Image.open(path) as image:
        return image.convert('RGBA')


def grab_cursor(size, closed=False):
    """Return premultiplied BGRA pixels with an identical hotspot in both poses."""
    image = _source(closed).resize((size, size), Image.Resampling.LANCZOS)
    rgba = np.asarray(image).copy()
    rgba[:, :, :3] = np.rint(rgba[:, :, :3]*(rgba[:, :, 3:4].astype(np.float32)/255))
    return rgba[:, :, [2, 1, 0, 3]], (round(size/2), round(size/2))

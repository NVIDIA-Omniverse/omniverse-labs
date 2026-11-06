# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage interpretation shared by private sublayer wrappers."""


def copy_authored_stage_metadata(source, destination):
    """Preserve source units/timeline without authoring resolved fallbacks.

    These root fields do not follow prim opinions through sublayers. The
    complete source, rather than sparse edit/device opinions, owns their values.
    """
    for key in (
        "metersPerUnit", "upAxis", "kilogramsPerUnit", "defaultPrim",
        "startTimeCode", "endTimeCode", "timeCodesPerSecond", "framesPerSecond",
    ):
        if source.pseudoRoot.HasInfo(key):
            destination.pseudoRoot.SetInfo(key, source.pseudoRoot.GetInfo(key))

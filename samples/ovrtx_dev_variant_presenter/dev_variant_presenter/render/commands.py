# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render-thread command dataclasses.

Moved out of runtime.py so StageSession / batch backends can share the protocol
without importing the full RenderRuntime.
"""
from __future__ import annotations

from dataclasses import dataclass

from dev_variant_presenter.models import QualitySpec, Selection, StageInfo


@dataclass(frozen=True)
class OpenStage:
    user_usd: str
    selection: Selection
    camera_path: str
    camera_xform: tuple          # 16 floats (row-major 4x4) of the authored camera
    focus_distance: float
    stage_info: StageInfo
    up_axis: str = "Y"
    looks: dict | None = None    # seed per-camera optics at open (project restore -> ONE reopen,
    xforms: dict | None = None   # not original-view-then-settle); None = fresh stage, no overrides
    edit_layer: str = ""         # sidecar edits layer (turntable camera) sublayered into composites


@dataclass(frozen=True)
class SetSelection:
    selection: Selection


@dataclass(frozen=True)
class SetCameraXform:
    matrix: tuple                # 16 floats, row-vector 4x4


@dataclass(frozen=True)
class SnapToCamera:
    camera_xform: tuple
    focus_distance: float
    camera_path: str = ""        # the authored camera being selected (keys its per-camera look)
    reset: bool = False          # True -> drop this camera's transform override, snap to authored
    at_s: float | None = None    # seconds into the camera's timeline clip: pose an ANIMATED
                                 # camera's rig at that stage time (timeline scrub)


@dataclass(frozen=True)
class SetQuality:
    quality: QualitySpec


@dataclass(frozen=True)
class SetCamera:
    params: dict          # any subset of focal_length / f_stop / focus_distance / exposure


@dataclass(frozen=True)
class SetResolution:
    width: int
    height: int


@dataclass(frozen=True)
class PickFocus:
    nx: float            # normalized click coords in the stream (0..1)
    ny: float
    reply: "queue.Queue"


@dataclass(frozen=True)
class PickPoint:
    nx: float            # normalized click coords -> picked prim's world bbox center + size
    ny: float
    reply: "queue.Queue"


@dataclass(frozen=True)
class ProbeOcclusion:
    point: tuple          # world point to test (the gizmo pivot)
    nx: float             # its projected screen position
    ny: float
    reply: "queue.Queue"


@dataclass(frozen=True)
class GetCameraState:
    reply: "queue.Queue"          # -> {"looks": {cam: optics}, "xforms": {cam: {m, dist}}}


@dataclass(frozen=True)
class SetCameraState:
    looks: dict                   # restore a project's per-camera looks (optics)
    xforms: dict                  # restore a project's per-camera transform overrides (framing)


@dataclass(frozen=True)
class MoveCamera:
    matrix: tuple                # 16 floats row-vector: place the FREE viewport camera
    focus_distance: float = 0.0


@dataclass(frozen=True)
class CaptureFraming:
    pass    # commit the CURRENT live view as the active camera's framing override


@dataclass(frozen=True)
class SetPlayback:
    playing: bool
    fps: float = 0.0     # 0 -> the stage's own fps


@dataclass(frozen=True)
class RestartStream:
    # rebuild the ovstream server at the current size — clears half-open WebRTC sessions.
    # reply (optional): the route blocks on it so the CLIENT only handshakes after the
    # rebuild actually happened (a fixed client-side wait loses to slow warmup drains)
    reply: "queue.Queue | None" = None


@dataclass(frozen=True)
class CancelJob:
    pass


@dataclass(frozen=True)
class Shutdown:
    pass


@dataclass(frozen=True)
class RunBatch:
    job: object
    reply: "queue.Queue"


@dataclass(frozen=True)
class RunTimeline:
    timeline: dict
    quality: QualitySpec
    out_dir: str
    reply: "queue.Queue"


@dataclass(frozen=True)
class PrerenderThumbnails:
    cameras: tuple
    out_dir: str
    reply: "queue.Queue"

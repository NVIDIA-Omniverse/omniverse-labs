# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Matrix expansion + counting + explosion guard + time estimate (pure Python, no ovrtx)
plus the render-thread execution (`run_batch`, which imports ovrtx/PIL lazily).

Every expanded Selection carries the FULL base look with the swept set(s) overridden,
so each rendered permutation is complete and order-independent.
"""
from __future__ import annotations

import itertools
import math
from typing import Protocol

from dev_variant_presenter.batch.jobs import BatchJob, MatrixMode, permutation_name
from dev_variant_presenter.models import Selection, StageInfo, VariantChoice, VariantSetInfo

DEFAULT_EXPLOSION_THRESHOLD = 500


class ExplosionError(Exception):
    """Raised when a batch exceeds the permutation guard without explicit confirmation."""

    def __init__(self, count: int, threshold: int):
        self.count = count
        self.threshold = threshold
        super().__init__(f"{count} permutations exceeds the guard of {threshold}; confirm to proceed.")


def guard_count(count: int, *, threshold: int = DEFAULT_EXPLOSION_THRESHOLD,
                confirm: bool = False) -> int:
    """Raise ExplosionError if count exceeds threshold and not confirmed."""
    if count > threshold and not confirm:
        raise ExplosionError(count, threshold)
    return count


def _set_index(sets: tuple[VariantSetInfo, ...]) -> dict[str, VariantSetInfo]:
    return {vs.set_name: vs for vs in sets}


def _base_index(base: Selection) -> dict[str, VariantChoice]:
    return {c.set_name: c for c in base}


def expand_labeled(job: BatchJob, base: Selection,
                   sets: tuple[VariantSetInfo, ...]) -> list[tuple[Selection, str]]:
    """Expand to (selection, label) pairs. Each Selection carries the FULL base look with
    the swept set(s) overridden (so the render is complete); the label names only what
    VARIES — `Carpaint-Sakura` (one-at-a-time), `Carpaint-Sakura_Wheels-B` (cartesian),
    the full selection (curated). Labels are unique within a job and folder-safe."""
    by_set = _set_index(sets)
    base_by_set = _base_index(base)

    if job.mode == MatrixMode.CURATED:
        return [(c, permutation_name(c)) for c in job.curated]

    swept = [(name, variants) for name, variants in job.included.items() if name in by_set]

    if job.mode == MatrixMode.FULL_CARTESIAN:
        if not swept:
            return [(base, "base")]
        choice_lists = [
            [VariantChoice(by_set[name].prim_path, name, v) for v in variants]
            for name, variants in swept
        ]
        out: list[tuple[Selection, str]] = []
        for combo in itertools.product(*choice_lists):
            overridden = {c.set_name: c for c in combo}
            sel = tuple(overridden.get(c.set_name, c) for c in base)
            for c in combo:
                if c.set_name not in base_by_set:
                    sel = sel + (c,)
            out.append((sel, permutation_name(combo)))   # all included sets, swept order
        return out

    if job.mode == MatrixMode.ONE_AT_A_TIME:
        out = []
        for name, variants in swept:
            for v in variants:
                override = VariantChoice(by_set[name].prim_path, name, v)
                if name in base_by_set:
                    sel = tuple(override if c.set_name == name else c for c in base)
                else:
                    sel = base + (override,)
                out.append((sel, permutation_name((override,))))  # the swept set only
        return out

    raise ValueError(f"expand_matrix: unknown mode {job.mode}")


def expand_matrix(job: BatchJob, base: Selection, sets: tuple[VariantSetInfo, ...]) -> list[Selection]:
    """Concrete Selections only (labels dropped). See expand_labeled for naming."""
    return [sel for sel, _ in expand_labeled(job, base, sets)]


def count_permutations(job: BatchJob, sets: tuple[VariantSetInfo, ...]) -> int:
    """Count without materializing every Selection (Cartesian can be huge)."""
    by_set = _set_index(sets)
    swept = [variants for name, variants in job.included.items() if name in by_set]
    if job.mode == MatrixMode.CURATED:
        return len(job.curated)
    if job.mode == MatrixMode.FULL_CARTESIAN:
        if not swept:
            return 1
        return math.prod(len(v) for v in swept)
    if job.mode == MatrixMode.ONE_AT_A_TIME:
        return sum(len(v) for v in swept)
    raise ValueError(f"count_permutations: unknown mode {job.mode}")


def estimate_seconds(job: BatchJob, sets: tuple[VariantSetInfo, ...], *,
                     per_frame_s: float, frame_count: int = 1) -> float:
    """Rough wall-clock estimate: permutations * frames-per-perm * cameras * per_frame_s.
    `per_frame_s` is a measured/assumed cost of one converged frame at the job quality.
    `frame_count` is 1 for single, the stage frame span for animation_range."""
    perms = count_permutations(job, sets)
    frames = frame_count if job.frame_mode == "animation_range" else 1
    cams = max(1, len(job.cameras))
    return perms * frames * cams * per_frame_s


# --------------------------- render-thread execution ---------------------------
# Everything below imports ovrtx/PIL lazily and MUST be called only from the render
# thread (the sole ovrtx owner). Validated by the manual integration step, not unit tests.

class RenderBackend(Protocol):
    """What the batch/timeline render loop needs from the render thread — deliberately
    narrow so this module never touches `Renderer.open_usd` / `update_from_usd_time`
    directly. `RenderRuntime` implements this via `StageSession` + `ovrtx.Renderer`."""

    def open_composite(self, path: str) -> None: ...
    def set_time(self, time_code: float) -> None: ...
    def reset(self) -> None: ...
    def step_product(self, rp: str, dt: float): ...   # same shape as ovrtx.Renderer.step()'s result


def _converge(backend: RenderBackend, rp_path, *, max_steps: int, delta_time: float,
              min_steps: int = 30, stable_needed: int = 5,
              heartbeat=None, heartbeat_every: int = 8) -> int:
    """Step until a 64x64 downsample hash is stable `stable_needed` times, or `max_steps`.
    Works for both RT2 (denoiser settles fast) and reference PathTracing (runs toward the
    sample budget). `heartbeat()` every `heartbeat_every` steps keeps WebRTC alive (~7s)."""
    import numpy as np
    import ovrtx
    floor = min(min_steps, max_steps)
    prev = None
    stable = 0
    for step_i in range(max_steps):
        products = backend.step_product(rp_path, delta_time)
        if heartbeat and step_i % heartbeat_every == 0:
            heartbeat()
        if step_i < floor:
            continue
        frame = products[rp_path].frames[0]
        if "LdrColor" not in frame.render_vars:
            continue
        with frame.render_vars["LdrColor"].map(device=ovrtx.Device.CPU) as m:
            px = np.from_dlpack(m)
            h, w = px.shape[:2]
            ds = px[:: max(1, h // 64), :: max(1, w // 64)]
            checksum = hash(ds.tobytes())
        if prev is not None and checksum == prev:
            stable += 1
        else:
            stable = 0
        prev = checksum
        if stable >= stable_needed:
            return step_i + 1
    return max_steps


def _save_frame(backend: RenderBackend, rp_path, out_path, *, delta_time: float) -> None:
    """Step once, map LdrColor (CPU), save a PNG. Render-thread use."""
    from pathlib import Path

    import numpy as np
    import ovrtx
    from PIL import Image
    products = backend.step_product(rp_path, delta_time)
    frame = products[rp_path].frames[0]
    with frame.render_vars["LdrColor"].map(device=ovrtx.Device.CPU) as m:
        arr = np.from_dlpack(m).copy()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(str(out_path))


def run_batch(job: BatchJob, base: Selection, stage_info: StageInfo, *,
              backend: RenderBackend, render_product_path: str, emit, is_cancelled,
              composer, user_usd: str, viewer_camera_path: str = "/Viewer/Camera",
              heartbeat=None, camera_look=None, apply_camera_override=None,
              camera_is_animated=None, camera_world_at=None, write_camera=None,
              extra_sublayers=()) -> str:
    """Execute a batch on the render thread. Returns the out_dir.
    `emit(event_dict)` pushes a WS event; `is_cancelled()` truthy aborts between perms.
    `heartbeat()` is called periodically to keep the WebRTC stream alive during the batch."""
    import os
    from pathlib import Path

    labeled = expand_labeled(job, base, stage_info.variant_sets)
    cameras = job.cameras or [""]
    total = len(labeled) * len(cameras)
    fps = stage_info.fps or 24.0
    delta_time = 1.0 / fps
    max_steps = max(job.quality.samples_per_pixel, 40)
    out_root = Path(job.out_dir)
    done = 0

    for sel, name in labeled:
        if is_cancelled():
            break
        for cam in cameras:
            if is_cancelled():
                break
            # one composite per (permutation, camera); cam suffix only when multi-camera
            stem = name if len(cameras) == 1 else f"{name}__{cam.rsplit('/', 1)[-1]}"
            comp_path = str(out_root / f"_comp_{stem}.usda")
            # animated camera (turntable rig): its per-frame pose is evaluated in pxr and
            # fabric-written below — update_from_usd_time does NOT re-evaluate time-sampled
            # xforms in this ovrtx build (GPU-verified), so shooting through the stage
            # camera prim just renders its default-time pose for every frame.
            animated = bool(cam) and bool(camera_is_animated) and camera_is_animated(cam)
            composer.build_composite(
                user_usd, sel, camera_path=cam,
                render_product_path=render_product_path,
                quality=job.quality, out_path=comp_path,
                viewer_camera_path=viewer_camera_path,
                camera=camera_look(cam) if camera_look else None,   # per-camera optics (looks)
                extra_sublayers=extra_sublayers)
            backend.open_composite(comp_path)
            if apply_camera_override and not animated:
                apply_camera_override(cam)          # per-camera framing (saved orbit/pan/dolly)
            if heartbeat:
                heartbeat()

            emit({"type": "batch_progress", "done": done, "total": total,
                  "name": stem, "phase": "converge"})

            if job.frame_mode == "animation_range":
                start = int(round(job.frame_start if job.frame_start is not None else stage_info.start_time))
                end = int(round(job.frame_end if job.frame_end is not None else stage_info.end_time))
                step = max(1, job.frame_step or 1)
                seq_dir = out_root / stem
                for f in range(start, end + 1, step):
                    if is_cancelled():
                        break
                    backend.set_time(float(f))   # scene-side time
                    backend.reset()  # discard prior time-sample accumulation
                    if animated and camera_world_at and write_camera:
                        m = camera_world_at(cam, float(f))    # the rig's pose THIS frame
                        if m is not None:
                            write_camera(m)
                    _converge(backend, render_product_path, max_steps=max_steps,
                              delta_time=delta_time, heartbeat=heartbeat)
                    _save_frame(backend, render_product_path,
                                str(seq_dir / f"{f:04d}.png"), delta_time=delta_time)
                    if heartbeat:
                        heartbeat()
                # assemble the sequence's mp4 HERE, where the playback rate is known
                # (stage fps / frame step) — no fps knob needed in the Results pane
                if not is_cancelled() and any(seq_dir.glob("*.png")):
                    emit({"type": "batch_progress", "done": done, "total": total,
                          "name": stem, "phase": "encode"})
                    from dev_variant_presenter.post import processing
                    processing.frames_to_video(str(seq_dir), str(out_root / f"{stem}.mp4"),
                                               fps=max(1, round(fps / step)))
            else:
                _converge(backend, render_product_path, max_steps=max_steps,
                          delta_time=delta_time, heartbeat=heartbeat)
                _save_frame(backend, render_product_path,
                            str(out_root / f"{stem}.png"), delta_time=delta_time)

            emit({"type": "batch_progress", "done": done + 1, "total": total,
                  "name": stem, "phase": "save"})
            done += 1
            if heartbeat:
                heartbeat()
            try:
                os.unlink(comp_path)
            except OSError:
                pass

    emit({"type": "batch_done", "out_dir": str(out_root), "done": done, "total": total,
          "cancelled": bool(is_cancelled())})
    return str(out_root)

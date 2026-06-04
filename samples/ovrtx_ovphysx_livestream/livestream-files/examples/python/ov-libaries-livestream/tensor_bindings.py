# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

# NOTE: This file is included verbatim in documentation.
# When editing, keep tutorial line ranges in sync.


#!/usr/bin/env python3
"""
Tensor bindings sample demonstrating simulation data exchange.

This sample demonstrates:
1. Loading a USD scene with physics objects
2. Creating tensor bindings for data exchange
3. Writing control inputs (joint drive velocity) using tensor API
4. Running extended simulation (1000 steps)
5. Reading physics outputs (link pose) using tensor API

**USD viewer:** run with ``--viewer-stream`` (and ``--usd`` / ``--articulation-root``) to emit
JSONL poses on stdout for ``examples/python/usd-viewer-example`` (Simulation menu), same wire
format as ``physx_live_worker.py``, but with alternating DOF velocity targets applied.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from ovphysx import PhysX
from ovphysx.types import TensorType

# -----------------------------------------------------------------------------
# Paths and shared helpers
# -----------------------------------------------------------------------------


def _default_usd_path() -> Path:
    """Local sample stage shipped next to this script (articulated chain + PhysX)."""
    return Path(__file__).resolve().parent / "links_chain_sample.usda"


def _import_pose_utils():
    """Load the viewer's JSON payload builder without making ovphysx depend on the viewer package.

    ``--viewer-stream`` reuses ``version1_articulation_payload`` from ``usd-viewer-example`` so the
    child process speaks the same JSON line format as ``physx_live_worker.py`` (prim paths + 4×4
    matrices the desktop viewer applies each frame).
    """
    viewer_dir = Path(__file__).resolve().parent.parent / "usd-viewer-example"
    utils = viewer_dir / "physx_pose_utils.py"
    if not utils.is_file():
        raise RuntimeError(
            "viewer-stream mode needs examples/python/usd-viewer-example/physx_pose_utils.py "
            f"(expected at {utils})."
        )
    if str(viewer_dir) not in sys.path:
        sys.path.insert(0, str(viewer_dir))
    from physx_pose_utils import version1_articulation_payload  # noqa: PLC0415

    return version1_articulation_payload


# -----------------------------------------------------------------------------
# Standalone tutorial: pattern bindings + batch stepping
# -----------------------------------------------------------------------------


def run_standalone_tutorial() -> None:
    """Exercise tensor bindings in-process: write drives, step many times, read poses to the console.

    Two ways to target prims in ``create_tensor_binding``:
    - **pattern** (here): glob-like USD path, e.g. all ``articulationLink*`` capsules under the chain.
    - **prim_paths** (used in ``run_viewer_stream``): explicit list, one articulation root; required
      when you need stable ``body_names`` for JSON export.

    CPU device ensures NumPy buffers can be filled synchronously by ``read`` / ``write``.
    """
    physx = PhysX(device="cpu")

    usd_path = _default_usd_path()
    if not usd_path.exists():
        raise RuntimeError(f"USD scene not found: {usd_path}")

    print(f"Loading USD scene: {usd_path}")
    # add_usd returns a handle so we can remove the layer later; wait_all flushes async load work.
    usd_handle, _ = physx.add_usd(str(usd_path))
    physx.wait_all()

    # --- Control tensor: desired angular velocities at each DOF (drive targets), rad/s.
    # Shape is typically (num_articulations, num_dofs). This scene has one articulation.
    print("Creating tensor binding for DOF velocity targets...")
    velocity_target_binding = physx.create_tensor_binding(
        pattern="/World/articulation/articulationLink*",
        tensor_type=TensorType.ARTICULATION_DOF_VELOCITY_TARGET,
    )
    print(f"  DOF count: {velocity_target_binding.shape[1]}")

    # --- Observation tensor: world pose per link (position + orientation).
    # Last dim is 7: [px, py, pz, qx, qy, qz, qw] with imaginary-first quaternion (xyzw).
    print("Creating tensor binding for link poses...")
    link_pose_binding = physx.create_tensor_binding(
        pattern="/World/articulation/articulationLink*",
        tensor_type=TensorType.ARTICULATION_LINK_POSE,
    )
    print(f"  Link count: {link_pose_binding.shape[1]}, Pose dims: {link_pose_binding.shape[2]}")

    # Fill drive targets: alternating signs so adjacent joints spin opposite directions (visible motion).
    num_dofs = velocity_target_binding.shape[1]
    velocity_targets = np.zeros(velocity_target_binding.shape, dtype=np.float32)
    for i in range(num_dofs):
        velocity_targets[0, i] = 25.0 if i % 2 == 0 else -25.0

    print("Setting DOF velocity targets (alternating ±25 rad/s)...")
    velocity_target_binding.write(velocity_targets)
    print(f"  Velocity targets: {velocity_targets[0, :5]}... (first 5 DOFs)")

    # Simulation loop: advance physics with fixed dt; global time argument is the *simulation time*
    # after the step (here i*dt for step index i). wait_all ensures GPU/CPU work for this step finished.
    print("\nRunning 1000 simulation steps...")
    link_poses = np.zeros(link_pose_binding.shape, dtype=np.float32)

    dt = 0.01
    for i in range(1000):
        current_time = i * dt
        physx.step(dt, current_time)
        physx.wait_all()

        # read() copies link poses into our preallocated array (no new allocation per frame).
        if i % 100 == 0 or i == 999:
            link_pose_binding.read(link_poses)

            # Inspect the first link only: translation then quaternion.
            px, py, pz = link_poses[0, 0, 0:3]
            qx, qy, qz, qw = link_poses[0, 0, 3:7]

            # Convert quaternion to roll about X (degrees) for a single readable scalar.
            roll_x_rad = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
            deg_x = roll_x_rad * 180.0 / math.pi

            print(
                f"  Step {i:4d}: pos=({px:.6f}, {py:.6f}, {pz:.6f}), "
                f"quat(xyzw)=({qx:.6f}, {qy:.6f}, {qz:.6f}, {qw:.6f}), "
                f"rotation_x={deg_x:.2f}°"
            )

    print("\nCompleted 1000 simulation steps successfully!")

    # Tear down bindings before releasing PhysX (native resources tied to tensor views).
    velocity_target_binding.destroy()
    link_pose_binding.destroy()

    physx.remove_usd(usd_handle)
    physx.release()

    print("\n[SUCCESS]", flush=True)


# -----------------------------------------------------------------------------
# Viewer subprocess: JSONL on stdout (same contract as physx_live_worker)
# -----------------------------------------------------------------------------


def run_viewer_stream(
    *,
    usd_path: Path,
    articulation_root: str,
    dt: float,
    max_steps: int,
) -> int:
    """Drive the articulation with tensor targets, then stream poses to the desktop viewer.

    **Stdout:** one JSON object per line (version 1: prim paths + ``matrix4d`` per link). The viewer
    parses the *last complete line* per frame and maps transforms onto the opened USD stage.

    **Stderr:** diagnostics only so stdout stays machine-readable JSONL.

    **Loop order:** read poses → emit JSON → step — mirrors ``physx_live_worker.py`` so the first line
    reflects state after the initial warmup step; the parent process applies poses before the next
    ``renderer.step``, keeping graphics loosely in sync with PhysX time.
    """
    version1_articulation_payload = _import_pose_utils()

    usd_path = usd_path.resolve()
    if not usd_path.is_file():
        print(f"error: USD path is not a file: {usd_path}", file=sys.stderr)
        return 2

    root = articulation_root.rstrip("/")
    max_steps = int(max_steps)

    physx = PhysX(device="cpu")
    try:
        physx.add_usd(str(usd_path))

        print(
            f"[INFO] tensor_bindings viewer-stream: {usd_path}, root={root!r} dt={dt}",
            file=sys.stderr,
            flush=True,
        )

        # Bind by articulation root prim path so ``body_names`` lists link names for JSON paths
        # like ``/World/articulation/articulationLink0`` (required by version1_articulation_payload).
        velocity_target_binding = physx.create_tensor_binding(
            prim_paths=[root],
            tensor_type=TensorType.ARTICULATION_DOF_VELOCITY_TARGET,
        )
        link_pose_binding = physx.create_tensor_binding(
            prim_paths=[root],
            tensor_type=TensorType.ARTICULATION_LINK_POSE,
        )
        try:
            num_dofs = velocity_target_binding.shape[1]
            velocity_targets = np.zeros(velocity_target_binding.shape, dtype=np.float32)
            for i in range(num_dofs):
                velocity_targets[0, i] = 25.0 if i % 2 == 0 else -25.0
            velocity_target_binding.write(velocity_targets)

            # Initial step at t=0 so joints integrate from a defined state before the first read.
            physx.step(dt, 0.0)
            physx.wait_all()

            buf = np.zeros(link_pose_binding.shape, dtype=np.float32)
            names = list(link_pose_binding.body_names)
            elapsed = dt
            step_count = 0
            while True:
                link_pose_binding.read(buf)
                payload = version1_articulation_payload(buf, names, root)
                print(json.dumps(payload, separators=(",", ":")), flush=True)
                step_count += 1
                if max_steps > 0 and step_count >= max_steps:
                    break
                physx.step(dt, elapsed)
                physx.wait_all()
                elapsed += dt
        finally:
            link_pose_binding.destroy()
            velocity_target_binding.destroy()
    except BrokenPipeError:
        # Viewer closed the pipe; exit quietly (common when stopping the stream from the UI).
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 4
    finally:
        physx.release()

    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> int:
    """Default: run the standalone tutorial. With ``--viewer-stream``, run as a viewer child process."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--viewer-stream",
        action="store_true",
        help="Emit JSONL poses on stdout for usd-viewer-example (stderr for logs).",
    )
    p.add_argument("--usd", type=Path, default=None, help="Root USD file (required with --viewer-stream)")
    p.add_argument(
        "--articulation-root",
        default="/World/articulation",
        help="Articulation root prim path (TensorBinding target; default: /World/articulation)",
    )
    p.add_argument("--dt", type=float, default=1.0 / 60.0, help="Fixed step [s] (default: 1/60)")
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop after this many emitted pose lines (0 = run until killed; default: 0)",
    )
    args = p.parse_args()

    if args.viewer_stream:
        if args.usd is None:
            print("error: --usd is required with --viewer-stream", file=sys.stderr)
            return 2
        return run_viewer_stream(
            usd_path=args.usd,
            articulation_root=args.articulation_root,
            dt=float(args.dt),
            max_steps=args.max_steps,
        )

    run_standalone_tutorial()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

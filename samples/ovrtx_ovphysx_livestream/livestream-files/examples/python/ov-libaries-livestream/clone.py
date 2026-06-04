# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

# NOTE: This file is included verbatim in documentation.
# When editing, keep tutorial line ranges in sync.

"""
Clone sample demonstrating scene replication with the clone API.

This sample demonstrates:
1. Loading a USD scene with an environment hierarchy
2. Cloning the environment to create multiple copies
3. Running simulation with all clones

**USD viewer** (``examples/python/usd-viewer-example``): open ``basic_simulation.usda``, then use
**Simulation → Clone…**. The viewer process loads USD in **ovrtx** and runs ``clone_usd`` so
``env1``–``env3`` exist for drawing; a child process runs this file with ``--viewer-stream``, loads the
same file in **PhysX**, calls ``physx.clone`` so the solver also has four envs, then streams rigid-body
poses (JSON lines on stdout — same contract as ``physx_rigid_live_worker.py``). The parent applies each
line to ``omni:xform`` / fabric so you see motion. Viewer-stream mode sleeps between steps so playback
is **quarter real-time** (easier to follow).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from ovphysx import PhysX
from ovphysx.types import TensorType

# -----------------------------------------------------------------------------
# Stage paths (must match basic_simulation.usda)
# -----------------------------------------------------------------------------
# The sample USD defines a single template environment at ENV_CLONE_SOURCE. PhysX ``clone`` copies
# that subtree into new paths (ENV_CLONE_TARGETS) so the solver simulates four independent stacks
# of physics (four “rooms”), each with its own /table rigid body.
ENV_CLONE_SOURCE = "/World/envs/env0"
ENV_CLONE_TARGETS = ["/World/envs/env1", "/World/envs/env2", "/World/envs/env3"]

# One rigid body per env: the Xform ``table`` (PhysicsRigidBodyAPI) that holds the tabletop mesh.
# Order matches TensorBinding row order (env0 … env3). The desktop viewer applies each pose to the
# same path on the ovrtx stage (after it has duplicated env0 with ``clone_usd`` — see run_viewer_stream).
RIGID_TABLE_PATHS = [f"/World/envs/env{i}/table" for i in range(4)]

# -----------------------------------------------------------------------------
# Viewer-stream pacing (wall clock vs simulation time)
# -----------------------------------------------------------------------------
# The subprocess emits one JSON line per simulation step. Without sleeping, the loop would run at
# CPU speed and the parent window would only display the latest pose each frame — motion looks instant.
# ``sleep(scale * dt)`` after each step makes wall time ≈ scale× longer than sim time for that step;
# scale=4 ⇒ about quarter real-time (four wall seconds per one simulated second).
_VIEWER_WALL_SECONDS_PER_SIM_STEP: float = 4.0


def _default_basic_simulation_path() -> Path:
    return Path(__file__).resolve().parent / "basic_simulation.usda"


def _import_rigid_pose_utils():
    """Load ``version1_rigid_body_payload`` from ``usd-viewer-example`` (no hard dep from ovphysx).

    That helper turns RIGID_BODY_POSE tensors (N,7) position+quat into the same version-1 JSON dict
    the viewer already understands: ``{"version":1,"prims":[{"path":...,"matrix4d":...},...]}``.
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
    from physx_pose_utils import version1_rigid_body_payload  # noqa: PLC0415

    return version1_rigid_body_payload


def run_viewer_stream(usd_path: Path, dt: float) -> int:
    """Run PhysX in this process and stream table poses for the desktop viewer (stdout = JSONL).

    Important: ``examples/python/usd-viewer-example`` loads the same USD in **ovrtx** and must call
    ``renderer.clone_usd`` *before* starting this subprocess so ``/World/envs/env1`` … ``env3`` exist
    for rendering. This script only drives **PhysX** (separate world) and prints poses; the parent
    maps those transforms onto the RTX stage each frame.
    """
    version1_rigid_body_payload = _import_rigid_pose_utils()
    usd_path = Path(usd_path).resolve()
    if not usd_path.is_file():
        print(f"error: USD path is not a file: {usd_path}", file=sys.stderr)
        return 2

    rigid_paths = list(RIGID_TABLE_PATHS)
    physx = PhysX(device="cpu")
    try:
        print(
            f"[INFO] clone viewer-stream: {usd_path} dt={dt} "
            f"wall≈{_VIEWER_WALL_SECONDS_PER_SIM_STEP:g}×dt per step (quarter real-time) "
            f"clone {ENV_CLONE_SOURCE!r} -> {ENV_CLONE_TARGETS!r}",
            file=sys.stderr,
            flush=True,
        )

        # --- Load the on-disk layer (only contains env0) and duplicate it inside PhysX. ------------
        physx.add_usd(str(usd_path))
        physx.wait_all()  # Ensure prims are ready before clone (async load flush).
        physx.clone(ENV_CLONE_SOURCE, ENV_CLONE_TARGETS)
        physx.wait_all()  # Clone completes asynchronously; must finish before tensors bind reliably.

        # One warm-up step at t=0 so the first ``read`` sees a post-initialize state (matches rigid live worker).
        physx.step(dt, 0.0)

        # RIGID_BODY_POSE: one row per prim, 7 floats (px,py,pz, qx,qy,qz,qw) per row.
        binding = physx.create_tensor_binding(
            prim_paths=rigid_paths,
            tensor_type=TensorType.RIGID_BODY_POSE,
        )
        buf = np.zeros(binding.shape, dtype=np.float32)
        # Global simulation time passed into ``step`` (PhysX uses it for time-dependent features).
        elapsed = dt
        try:
            while True:
                # Snapshot poses *after* the previous step, convert to JSON, emit for the parent process.
                binding.read(buf)
                payload = version1_rigid_body_payload(buf, rigid_paths)
                print(json.dumps(payload, separators=(",", ":")), flush=True)

                # Advance physics; next loop iteration will read the new poses.
                physx.step(dt, elapsed)
                elapsed += dt

                # Pace the loop so the viewport can keep up (see module constant comment above).
                time.sleep(max(0.0, _VIEWER_WALL_SECONDS_PER_SIM_STEP * dt))
        finally:
            binding.destroy()
    except BrokenPipeError:
        # Viewer closed the pipe — exit quietly.
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 4
    finally:
        physx.release()


def run_standalone_tutorial() -> None:
    """CLI-only demo: same PhysX clone + tensor read as viewer-stream, but prints to the console.

    Uses a **glob pattern** for the tensor binding (alternative to an explicit prim list) to show both
    API styles. No JSONL and no wall-clock throttle — intended for docs and quick verification.
    """
    physx = PhysX(device="cpu")

    usd_path = _default_basic_simulation_path()
    if not usd_path.is_file():
        raise RuntimeError(f"USD scene not found: {usd_path}")

    print(f"Loading USD scene: {usd_path}")
    # Keep the USD handle so we can remove_usd() and tear down cleanly (good practice for layered apps).
    usd_handle, _ = physx.add_usd(str(usd_path))
    physx.wait_all()

    print(f"Cloning {ENV_CLONE_SOURCE!r} to {len(ENV_CLONE_TARGETS)} targets...")
    physx.clone(ENV_CLONE_SOURCE, ENV_CLONE_TARGETS)
    physx.wait_all()
    print(f"  Created {len(ENV_CLONE_TARGETS)} clones successfully")

    # ``pattern`` expands to the same four prims as RIGID_TABLE_PATHS; order is defined by PhysX/USD.
    pose_binding = physx.create_tensor_binding(
        pattern="/World/envs/env*/table",
        tensor_type=TensorType.RIGID_BODY_POSE,
    )
    print(f"  Rigid body binding: count={pose_binding.count}, shape={pose_binding.shape}")

    print("Running 10 simulation steps...")
    dt = 1.0 / 60.0
    for i in range(10):
        # i*dt is the simulation time at the start of this step (fixed 60 Hz integration).
        physx.step(dt, i * dt)
    physx.wait_all()
    print("  All steps completed")

    poses = np.zeros(pose_binding.shape, dtype=np.float32)
    pose_binding.read(poses)
    for env_idx in range(poses.shape[0]):
        # First three components are world-space position of each table rigid body.
        px, py, pz = poses[env_idx, 0:3]
        print(f"  env{env_idx}: pos=({px:.4f}, {py:.4f}, {pz:.4f})")

    pose_binding.destroy()
    physx.remove_usd(usd_handle)
    physx.release()

    print("[SUCCESS]", flush=True)


def main() -> int:
    # Default: ``python clone.py`` → standalone tutorial. With ``--viewer-stream``, stdout is the
    # JSONL pose stream (keep stderr for logs so the viewer can parse stdout line-by-line).
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--viewer-stream",
        action="store_true",
        help="Emit JSONL rigid-body poses on stdout for usd-viewer-example (requires --usd).",
    )
    p.add_argument(
        "--usd",
        type=Path,
        default=None,
        help="Root USD path (required with --viewer-stream).",
    )
    p.add_argument(
        "--dt",
        type=float,
        default=1.0 / 60.0,
        help="Fixed step interval [s] for viewer-stream (default: 1/60).",
    )
    args = p.parse_args()

    if args.viewer_stream:
        if args.usd is None:
            print("error: --usd is required with --viewer-stream", file=sys.stderr)
            return 2
        return run_viewer_stream(args.usd, float(args.dt))

    run_standalone_tutorial()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run ovphysx in isolation and write rigid-body or articulation poses for the USD viewer.

Exports a small JSON file listing each prim path and a 4x4 ``matrix4d`` (USD row-vector
layout: translation on row index 3). These are **world** poses from PhysX; the viewer
maps **local** transforms via ``map_attribute`` (prefers ``omni:xform``, else
``omni:fabric:localMatrix``). For
articulations, world link matrices are converted using ``articulation_world_matrix4d``
(articulation root world matrix; defaults to identity). For **rigid bodies** (free
meshes with ``PhysicsRigidBodyAPI``), pass ``--rigid-body-paths`` and this script uses
``TensorType.RIGID_BODY_POSE`` instead of articulation tensors.

Uses ``PhysX(device="cpu")`` so ``TensorBinding.read`` can fill NumPy on CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ovphysx import PhysX
from ovphysx.types import TensorType

from physx_pose_utils import version1_articulation_payload, version1_rigid_body_payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--usd", required=True, help="Path to a local USD file")
    p.add_argument(
        "--steps",
        type=int,
        default=1200,
        help="Simulation steps at --dt (default: 1200; short runs can match rest pose)",
    )
    p.add_argument("--dt", type=float, default=1.0 / 60.0, help="Fixed step [s] (default: 1/60)")
    p.add_argument(
        "--out-json",
        required=True,
        help="Write pose payload JSON for the viewer (absolute prim paths + matrix4d)",
    )
    p.add_argument(
        "--articulation-root",
        default="/World/articulation",
        help="USD path of articulation root (ignored when --rigid-body-paths is set)",
    )
    p.add_argument(
        "--rigid-body-paths",
        nargs="*",
        default=None,
        metavar="PATH",
        help=(
            "One or more rigid-body prim paths (same order as TensorBinding). "
            "When set, exports RIGID_BODY_POSE instead of articulation link poses."
        ),
    )
    args = p.parse_args()
    usd_path = Path(args.usd).resolve()
    if not usd_path.is_file():
        print(f"error: USD path is not a file: {usd_path}", file=sys.stderr)
        return 2
    out_path = Path(args.out_json).resolve()
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"error: cannot create output dir: {exc}", file=sys.stderr)
        return 2

    rigid_paths = args.rigid_body_paths

    physx = PhysX(device="cpu")
    try:
        physx.add_usd(str(usd_path))
        physx.step_n_sync(args.steps, args.dt, 0.0)
        print(
            f"[INFO] PhysX simulation finished: {args.steps} steps at dt={args.dt} "
            "(poses will be read next; noisy USD/MaterialBinding warnings are usually harmless).",
            flush=True,
        )

        if rigid_paths:
            binding = physx.create_tensor_binding(
                prim_paths=list(rigid_paths),
                tensor_type=TensorType.RIGID_BODY_POSE,
            )
        else:
            binding = physx.create_tensor_binding(
                prim_paths=[args.articulation_root],
                tensor_type=TensorType.ARTICULATION_LINK_POSE,
            )
        try:
            buf = np.zeros(binding.shape, dtype=np.float32)
            binding.read(buf)

            if rigid_paths:
                if buf.ndim != 2 or buf.shape[1] != 7:
                    print(
                        f"error: expected RIGID_BODY_POSE shape (N, 7), got {buf.shape}",
                        file=sys.stderr,
                    )
                    return 3
                n = int(buf.shape[0])
                if n != len(rigid_paths):
                    print(
                        "error: rigid body count mismatch: "
                        f"tensor N={n} vs {len(rigid_paths)} paths",
                        file=sys.stderr,
                    )
                    return 3
                payload = version1_rigid_body_payload(buf, list(rigid_paths))
            else:
                payload = version1_articulation_payload(
                    buf, list(binding.body_names), args.articulation_root
                )
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        finally:
            binding.destroy()
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 4
    finally:
        physx.release()

    print(
        f"[SUCCESS] json={out_path} steps={args.steps} — "
        "physics ran; any further lines are often native PhysX/USD teardown (GPU→CPU broadphase, etc.).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

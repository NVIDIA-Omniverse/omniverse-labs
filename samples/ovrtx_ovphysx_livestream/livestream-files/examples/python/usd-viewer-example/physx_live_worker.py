# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live ovphysx articulation: step in a loop, emit one compact JSON line per step on stdout.

The USD viewer parses these lines and applies poses to ovrtx (same version-1 payload as
``physx_subprocess_sim.py``). Log diagnostics to stderr only so stdout stays JSONL.

Mirrors ``hello_world_physx.py`` (load + step) but runs continuously and streams poses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ovphysx import PhysX
from ovphysx.types import TensorType

from physx_pose_utils import version1_articulation_payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--usd", required=True, help="Path to a local USD file")
    p.add_argument(
        "--articulation-root",
        default="/World/articulation",
        help="USD path of articulation root (TensorBinding prim path)",
    )
    p.add_argument("--dt", type=float, default=1.0 / 60.0, help="Fixed step [s] (default: 1/60)")
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop after this many emitted pose lines (0 = run until process is killed)",
    )
    args = p.parse_args()
    usd_path = Path(args.usd).resolve()
    if not usd_path.is_file():
        print(f"error: USD path is not a file: {usd_path}", file=sys.stderr)
        return 2

    articulation_root = args.articulation_root.rstrip("/")
    dt = float(args.dt)
    max_steps = int(args.max_steps)

    physx = PhysX(device="cpu")
    try:
        physx.add_usd(str(usd_path))
        physx.step(dt, 0.0)
        print(
            f"[INFO] Live worker: loaded {usd_path}, articulation_root={articulation_root!r} dt={dt}",
            file=sys.stderr,
            flush=True,
        )

        binding = physx.create_tensor_binding(
            prim_paths=[articulation_root],
            tensor_type=TensorType.ARTICULATION_LINK_POSE,
        )
        try:
            buf = np.zeros(binding.shape, dtype=np.float32)
            names = list(binding.body_names)
            elapsed = dt
            step_count = 0
            while True:
                binding.read(buf)
                payload = version1_articulation_payload(buf, names, articulation_root)
                print(json.dumps(payload, separators=(",", ":")), flush=True)
                step_count += 1
                if max_steps > 0 and step_count >= max_steps:
                    break
                physx.step(dt, elapsed)
                elapsed += dt
        finally:
            binding.destroy()
    except BrokenPipeError:
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 4
    finally:
        physx.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal ovphysx smoke test: load ``links_chain_sample.usda`` and run one simulation step."""

from __future__ import annotations

import argparse
from pathlib import Path

import ovphysx
from ovphysx import PhysX


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--usd",
        type=Path,
        default=here / "links_chain_sample.usda",
        metavar="PATH",
        help=f"USD stage to load (default: {here / 'links_chain_sample.usda'})",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    usd_path = args.usd.resolve()

    print("Using ovphysx version: ", ovphysx.__version__)

    # Use CPU like the other ov-libaries-livestream samples so this subprocess can run while
    # usd-viewer-example holds the GPU for ovrtx (default PhysX device would contend for CUDA).
    physx = PhysX(device="cpu")

    if not usd_path.is_file():
        print(f"error: USD file not found: {usd_path}", flush=True)
        return 2

    physx.add_usd(str(usd_path))

    dt = 1.0 / 60.0
    elapsed_time = 0.0
    physx.step(dt, elapsed_time)

    physx.release()

    print("[SUCCESS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

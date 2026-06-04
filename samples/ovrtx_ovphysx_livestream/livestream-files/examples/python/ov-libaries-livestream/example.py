# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal ovrtx flow for teaching: load USD → step → read LdrColor → PNG + viewer."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Renderer: main API handle. Device: use Device.CPU / Device.CUDA (not strings) when mapping outputs.
from ovrtx import Device, Renderer

from PIL import Image

# Remote USDA that references the Robot-OVRTX sample scene (same as examples/python/minimal).
USD_URL = "https://omniverse-content-production.s3.us-west-2.amazonaws.com/Samples/Robot-OVRTX/robot-ovrtx.usda"

# Fixed output path of the last rendered image
OUTPUT_PATH = Path(__file__).resolve().parent / "last_render.png"


def open_image(path: Path) -> None:
    path = path.resolve()
    try:
        if sys.platform == "win32":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except OSError as e:
        print(f"Could not open viewer ({e}). Image saved at:\n  {path}", file=sys.stderr)


def main() -> None:
    print("Creating renderer. The first run may take a while (shader compile/cache)...", file=sys.stderr)
    # 1) Create the RTX-backed renderer (heavy first time: shader cache, etc.).
    renderer = Renderer()
    print("Renderer created.", file=sys.stderr)

    print(f"Adding {USD_URL} at root...", file=sys.stderr)
    # 2) Load a USD layer (URL or local path). Scene prims and render settings come from the stage.
    renderer.open_usd(USD_URL)
    print("USD loaded.", file=sys.stderr)

    print("Stepping renderer...", file=sys.stderr)
    # 3) Advance simulation time and produce outputs for the named render product(s).
    #    render_products must match a camera / render product path on the stage (here: /Render/Camera).
    products = renderer.step(
        render_products={"/Render/Camera"},
        delta_time=1.0 / 60.0,
    )
    print("Stepped renderer.", file=sys.stderr)

    print("Fetching results...", file=sys.stderr)
    # 4) Each step returns a dict of products → frames → render_vars (e.g. LdrColor = tonemapped RGB).
    for _product_name, product in products.items():
        for frame in product.frames:
            # 5) Map GPU output into a CPU tensor you can read. Device.CPU is required (enum, not "cpu").
            with frame.render_vars["LdrColor"].map(device=Device.CPU) as var:
                pixels = var.tensor.numpy()
                pil = Image.fromarray(pixels)
                pil.save(OUTPUT_PATH)
                print(f"Saved:\n  {OUTPUT_PATH}", file=sys.stderr)
                open_image(OUTPUT_PATH)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()

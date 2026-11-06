# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Change authored materials, render with ovrtx, and compare real output pixels."""
import json
import numpy as np
from PIL import Image
from _common import scene, output_folder, run_example


def main():
    output = output_folder()
    runtime, stage = scene()
    frames = []
    try:
        shader = stage.GetPrimAtPath('/World/Material/Surface')
        for name, roughness in (('polished', .07), ('matte', .9)):
            shader.GetAttribute('inputs:roughness').Set(roughness)
            # The adapter commits USD edits to ovstage, then calls ovrtx.step_async.
            for _ in range(16):
                pixels = runtime.render('Main')
            Image.fromarray(pixels).save(output/(name+'.png'))
            frames.append(pixels.astype(float))
        difference = float(np.abs(frames[0]-frames[1]).mean())
        assert difference > .1, difference
        (output/'render.json').write_text(json.dumps({'mean_pixel_difference': difference}, indent=2))
        print('Saved polished.png and matte.png; mean pixel difference:', difference)
    finally:
        runtime.close()


if __name__ == '__main__':
    run_example(main)

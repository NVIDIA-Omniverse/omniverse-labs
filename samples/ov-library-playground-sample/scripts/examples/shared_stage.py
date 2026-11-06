# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish a transform through ovstage and read back its authoritative value."""
import json
import numpy as np
from _common import scene, output_folder, run_example


def main():
    output = output_folder()
    runtime, stage = scene()
    try:
        # Use the public native write API directly, with a monotonically increasing ordinal.
        matrix = np.eye(4, dtype=np.float64)
        matrix[3, :3] = (1.5, 2., -.75)
        runtime.ordinal += 1
        query = runtime.query(['/World/Carton'])
        runtime.stage.write_attribute(query, 'omni:xform', runtime.ordinal,
            np.asarray([matrix]), is_array=False, semantic=runtime.ovstage.AttributeSemantic.MATRIX).wait()
        runtime.stage.advance_write_floor(runtime.ordinal).wait()
        result = runtime.read(['/World/Carton'], 'omni:xform')['/World/Carton'].reshape(4, 4)
        assert np.allclose(result, matrix), result
        (output/'transform.json').write_text(json.dumps(result.tolist(), indent=2))
        print('ovstage committed and read the transform at ordinal', runtime.ordinal)
    finally:
        runtime.close()


if __name__ == '__main__':
    run_example(main)

# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Round-trip tests for the Python API tensor-layout convention.

The Python API accepts and returns attribute data in NumPy-style shape-based
form — an N-element array of 4x4 matrices is ``shape=(N, 4, 4)``, not
``shape=(N,)`` with ``dtype.lanes=16``. The C API uses DLTensor lanes for
attribute reads and writes; Python reshapes those tensors for array-library
interop.
"""

from pathlib import Path

import numpy as np
import ovrtx

TEST_BASE_PATH = str((Path(__file__).parent / "../data/ovrtx-test-base.usda").resolve())


def _load_base(renderer):
    renderer.open_usd(TEST_BASE_PATH)
    renderer.reset()


def test_scalar_int32_shape(renderer):
    """Scalar int32: ``shape=(N,)`` — one value per prim, one prim."""
    _load_base(renderer)

    # [snippet:doc-shape-scalar-int32]
    # A scalar attribute for N prims is a 1-D array of length N.
    renderer.write_attribute(
        prim_paths=["/Render/Camera"],
        attribute_name="omni:rtx:rtpt:maxBounces",
        tensor=np.array([23], dtype=np.int32),  # shape=(1,)
    )
    tensor = renderer.read_attribute(
        attribute_name="omni:rtx:rtpt:maxBounces",
        prim_paths=["/Render/Camera"],
    )
    values = np.from_dlpack(tensor)
    assert values.shape == (1,)
    # [/snippet:doc-shape-scalar-int32]

    assert int(values[0]) == 23


def test_float3_array_shape(renderer):
    """``float3[]`` array: ``shape=(M, 3)`` per prim — M elements, 3 components each."""
    _load_base(renderer)

    # [snippet:doc-shape-float3-array]
    # point3f[] is a variable-length array of 3-component float vectors.
    # Express it as a 2-D ndarray with shape=(M, 3); the trailing 3 is the
    # vector dimension, not a lane count.
    points = np.array(
        [
            [-50.0, 0.0, -50.0],
            [50.0, 0.0, -50.0],
            [-50.0, 0.0, 50.0],
            [50.0, 0.0, 50.0],
        ],
        dtype=np.float32,
    )  # shape=(4, 3)
    renderer.write_array_attribute(
        prim_paths=["/World/Plane"],
        attribute_name="points",
        tensors=[points],
    )
    tensors = renderer.read_array_attribute(
        attribute_name="points",
        prim_paths=["/World/Plane"],
    )
    values = np.from_dlpack(tensors["/World/Plane"])
    assert values.shape == (4, 3)
    # [/snippet:doc-shape-float3-array]

    np.testing.assert_array_equal(values, points)


def test_mat4_array_shape(renderer):
    """4x4 double matrix: ``shape=(N, 4, 4)`` — N prims, one 4x4 matrix each."""
    _load_base(renderer)

    # [snippet:doc-shape-mat4-array]
    # A per-prim 4x4 transform is a 3-D ndarray with shape=(N, 4, 4).
    # Translate /World/Camera to (10, 20, 30) using USD row-vector convention
    # (translation lives in matrix[3][0..2]).
    xform = np.eye(4, dtype=np.float64)
    xform[3, 0:3] = [10.0, 20.0, 30.0]
    transforms = xform.reshape(1, 4, 4)  # shape=(1, 4, 4)
    renderer.write_attribute(
        prim_paths=["/World/Camera"],
        attribute_name="omni:xform",
        tensor=transforms,
        semantic=ovrtx.Semantic.XFORM_MAT4x4,
    )
    tensor = renderer.read_attribute(
        attribute_name="omni:xform",
        prim_paths=["/World/Camera"],
    )
    values = np.from_dlpack(tensor).reshape(1, 4, 4)
    assert values.shape == (1, 4, 4)
    # [/snippet:doc-shape-mat4-array]

    np.testing.assert_allclose(values[0, 3, 0:3], [10.0, 20.0, 30.0])

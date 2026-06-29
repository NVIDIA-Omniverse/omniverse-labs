# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Tests for the stage attribute read API (``Renderer.read_attribute`` /
``read_array_attribute`` and their async variants).

Reads target schema-known attributes the runtime exposes to Fabric:
``omni:rtx:rtpt:maxBounces`` on the RenderProduct (a 32-bit integer; the
runtime may report it as either int32 or uint32) and ``points`` on
``/World/Plane`` (a float3 array authored in ``ovrtx-test-base-geometry.usda``).
"""

from pathlib import Path

import numpy as np
import ovrtx

TEST_BASE_PATH = str((Path(__file__).parent / "../data/ovrtx-test-base.usda").resolve())


def _load_base(renderer):
    renderer.open_usd(TEST_BASE_PATH)
    renderer.reset()


def test_read_scalar_attribute_cpu(renderer):
    """Write a schema-known int32 scalar and read it back."""
    _load_base(renderer)

    # Seed a known value so the read has something deterministic to observe.
    renderer.write_attribute(
        prim_paths=["/Render/Camera"],
        attribute_name="omni:rtx:rtpt:maxBounces",
        tensor=np.array([17], dtype=np.int32),
    )

    # [snippet:doc-read-attribute-scalar]
    # Read a scalar attribute — one value per prim. The returned ManagedDLTensor
    # is DLPack-compatible; np.from_dlpack() gives a zero-copy numpy view.
    tensor = renderer.read_attribute(
        attribute_name="omni:rtx:rtpt:maxBounces",
        prim_paths=["/Render/Camera"],
    )
    values = np.from_dlpack(tensor)
    # [/snippet:doc-read-attribute-scalar]

    assert isinstance(tensor, ovrtx.ManagedDLTensor)
    # maxBounces is a 32-bit integer; the runtime may report it as int32 or
    # uint32 depending on the schema — compare the value, not the sign-ness.
    assert values.dtype.itemsize == 4 and values.dtype.kind in ("i", "u")
    assert values.shape == (1,)
    assert int(values[0]) == 17


def test_read_scalar_attribute_into_dest(renderer):
    """Pass a pre-allocated tensor as ``dest``; verify data lands in it."""
    _load_base(renderer)

    renderer.write_attribute(
        prim_paths=["/Render/Camera"],
        attribute_name="omni:rtx:rtpt:maxBounces",
        tensor=np.array([11], dtype=np.int32),
    )

    # [snippet:doc-read-attribute-dest-tensor]
    # Pre-allocate the destination. The read writes directly into `dest`; the
    # returned tensor is a handle to the same memory — both aliases are valid.
    # The dtype must match how the runtime stores the attribute.
    dest = np.empty((1,), dtype=np.uint32)
    renderer.read_attribute(
        attribute_name="omni:rtx:rtpt:maxBounces",
        prim_paths=["/Render/Camera"],
        dest=dest,
    )
    # `dest` now holds the attribute value.
    # [/snippet:doc-read-attribute-dest-tensor]

    assert int(dest[0]) == 11


def test_read_array_attribute(renderer):
    """Read ``points`` (``float3[]``) from the Plane mesh."""
    _load_base(renderer)

    # [snippet:doc-read-array-attribute]
    # Arrays are returned as dict[prim_path, ManagedDLTensor]. Iteration order
    # matches the input prim_paths so you can zip() against the request.
    tensors = renderer.read_array_attribute(
        attribute_name="points",
        prim_paths=["/World/Plane"],
    )
    for path, tensor in tensors.items():
        values = np.from_dlpack(tensor)
        print(f"{path}: {values.size} elements, dtype={values.dtype}")
    # [/snippet:doc-read-array-attribute]

    assert list(tensors.keys()) == ["/World/Plane"]
    values = np.from_dlpack(tensors["/World/Plane"])
    # ovrtx-test-base-geometry.usda authors 4 float3 points on the Plane.
    assert values.dtype == np.float32
    assert values.size == 4 * 3


def test_read_attribute_async(renderer):
    """Exercise the two-phase async lifecycle for an attribute read."""
    _load_base(renderer)

    renderer.write_attribute(
        prim_paths=["/Render/Camera"],
        attribute_name="omni:rtx:rtpt:maxBounces",
        tensor=np.array([5], dtype=np.int32),
    )

    # [snippet:doc-read-attribute-async]
    # Two-phase async: wait() → PendingFetch, then fetch() → ManagedDLTensor.
    op = renderer.read_attribute_async(
        attribute_name="omni:rtx:rtpt:maxBounces",
        prim_paths=["/Render/Camera"],
    )
    pending = op.wait()
    tensor = pending.fetch()
    values = np.from_dlpack(tensor)
    # [/snippet:doc-read-attribute-async]

    assert isinstance(op, ovrtx.Operation)
    assert isinstance(pending, ovrtx.PendingFetch)
    assert isinstance(tensor, ovrtx.ManagedDLTensor)
    assert int(values[0]) == 5


def test_read_attribute_cuda_dest(renderer):
    """Read directly into a GPU (CUDA) destination tensor via DLPack."""
    import warp as wp  # via warp-lang; provides DLPack-compatible CUDA arrays

    _load_base(renderer)

    renderer.write_attribute(
        prim_paths=["/Render/Camera"],
        attribute_name="omni:rtx:rtpt:maxBounces",
        tensor=np.array([9], dtype=np.int32),
    )

    # [snippet:doc-read-attribute-cuda-dest]
    # Allocate a CUDA destination through Warp (any DLPack-compatible CUDA
    # allocator works). The read writes directly into GPU memory; pass a CUDA
    # stream handle so the read is ordered on the caller's stream.
    dest = wp.empty(1, dtype=wp.uint32, device="cuda:0")
    stream = wp.Stream(device=dest.device)
    renderer.read_attribute(
        attribute_name="omni:rtx:rtpt:maxBounces",
        prim_paths=["/Render/Camera"],
        dest=dest,
        cuda_stream=stream.cuda_stream,
    )
    wp.synchronize_stream(stream)
    # [/snippet:doc-read-attribute-cuda-dest]

    # Copy back to host and verify the value.
    host = dest.numpy()
    assert int(host[0]) == 9

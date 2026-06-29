# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Tests for persistent attribute bindings, mapping, and CUDA write paths."""

from pathlib import Path

import numpy as np
import ovrtx
import warp as wp

TEST_BASE_PATH = str((Path(__file__).parent / "../data/ovrtx-test-base.usda").resolve())


@wp.kernel
def _set_xform_translation_x(transforms: wp.array(dtype=wp.mat44d), x: wp.float64):
    transforms[wp.tid()] = wp.mat44d(
        wp.float64(1.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(1.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(1.0),
        wp.float64(0.0),
        x,
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(1.0),
    )


def _load_base(renderer):
    renderer.open_usd(TEST_BASE_PATH)
    renderer.reset()


def _read_xform(renderer, prim="/World/Plane"):
    tensor = renderer.read_attribute("omni:xform", [prim])
    return np.from_dlpack(tensor).reshape(1, 4, 4)


def _make_xform(x):
    matrix = np.eye(4, dtype=np.float64).reshape(1, 4, 4)
    matrix[0, 3, 0] = x
    return matrix


def test_bind_attribute_write_unbind(renderer):
    """Create a scalar binding, write through it, and read back the result."""
    _load_base(renderer)
    matrix = _make_xform(12.0)

    # [snippet:doc-bind-attribute-write]
    binding = renderer.bind_attribute(
        prim_paths=["/World/Plane"],
        attribute_name="omni:xform",
        dtype="float64",
        shape=(4, 4),
        prim_mode=ovrtx.PrimMode.MUST_EXIST,
        flags=ovrtx.BindingFlag.OPTIMIZE,
    )
    binding.write(matrix)
    binding.unbind()
    # [/snippet:doc-bind-attribute-write]

    np.testing.assert_allclose(_read_xform(renderer), matrix)


def test_bind_attribute_async_and_write_async(renderer):
    """Create and write a binding asynchronously."""
    _load_base(renderer)
    matrix = _make_xform(13.0)

    # [snippet:doc-bind-attribute-async]
    op = renderer.bind_attribute_async(
        ["/World/Plane"], "omni:xform", dtype="float64", shape=(4, 4)
    )
    binding = op.wait()
    assert binding is not None
    # [/snippet:doc-bind-attribute-async]

    # [snippet:doc-binding-write-async]
    write_op = binding.write_async(matrix)
    assert write_op.wait() is True
    # [/snippet:doc-binding-write-async]

    np.testing.assert_allclose(_read_xform(renderer), matrix)
    binding.unbind()


def test_bind_array_attribute(renderer):
    """Bind a variable-length array attribute, write through it, and read it back."""
    _load_base(renderer)
    points = np.array(
        [
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    # [snippet:doc-bind-array-attribute]
    binding = renderer.bind_array_attribute(
        ["/World/Plane"], "points", dtype="float32", shape=(3,)
    )
    binding.write([points])
    # [/snippet:doc-bind-array-attribute]

    values = np.from_dlpack(renderer.read_array_attribute("points", ["/World/Plane"])["/World/Plane"])
    np.testing.assert_allclose(values, points)
    binding.unbind()


def test_map_attribute_cpu(renderer):
    """Map an attribute by name on CPU and verify the mutation."""
    _load_base(renderer)

    # [snippet:doc-map-attribute-cpu]
    with renderer.map_attribute(
        ["/World/Plane"], "omni:xform", dtype="float64", shape=(4, 4)
    ) as mapping:
        matrices = np.from_dlpack(mapping.tensor).reshape(1, 4, 4)
        matrices[0, 3, 0] = 10.0
    # [/snippet:doc-map-attribute-cpu]

    assert _read_xform(renderer)[0, 3, 0] == 10.0


def test_map_bound_attribute_cpu(renderer):
    """Map through an AttributeBinding rather than by name."""
    _load_base(renderer)

    # [snippet:doc-map-bound-attribute]
    binding = renderer.bind_attribute(["/World/Plane"], "omni:xform", dtype="float64", shape=(4, 4))
    with binding.map(device=ovrtx.Device.CPU) as mapping:
        matrices = np.from_dlpack(mapping.tensor).reshape(1, 4, 4)
        matrices[0, 3, 0] = 8.0
    # [/snippet:doc-map-bound-attribute]

    assert _read_xform(renderer)[0, 3, 0] == 8.0
    binding.unbind()


def test_unmap_attribute_async(renderer):
    """Unmap explicitly through the async API."""
    _load_base(renderer)

    # [snippet:doc-unmap-attribute-async]
    mapping = renderer.map_attribute(["/World/Plane"], "omni:xform", dtype="float64", shape=(4, 4))
    matrices = np.from_dlpack(mapping.tensor).reshape(1, 4, 4)
    matrices[0, 3, 0] = 9.0
    op = mapping.unmap_async()
    assert op.wait() is True
    # [/snippet:doc-unmap-attribute-async]

    assert _read_xform(renderer)[0, 3, 0] == 9.0


def test_map_attribute_cuda(renderer):
    """Map an attribute on CUDA, edit it with Warp, and read back on CPU."""
    _load_base(renderer)

    # [snippet:doc-map-attribute-cuda]
    mapping = renderer.map_attribute(
        ["/World/Plane"],
        "omni:xform",
        dtype="float64",
        shape=(4, 4),
        device=ovrtx.Device.CUDA,
    )
    tensor = wp.from_dlpack(mapping.tensor, dtype=wp.mat44d)
    stream = wp.Stream(device=tensor.device)
    wp.launch(_set_xform_translation_x, dim=1, inputs=[tensor, wp.float64(6.0)], stream=stream)
    mapping.unmap(stream=stream.cuda_stream)
    # [/snippet:doc-map-attribute-cuda]

    assert _read_xform(renderer)[0, 3, 0] == 6.0


def test_write_attribute_async_data_access_cuda(renderer):
    """Write a CUDA tensor with asynchronous data access and stream sync."""
    _load_base(renderer)
    cuda_tensor = wp.empty(1, dtype=wp.mat44d, device="cuda:0")
    stream = wp.Stream(device=cuda_tensor.device)
    wp.launch(_set_xform_translation_x, dim=1, inputs=[cuda_tensor, wp.float64(7.0)], stream=stream)

    # [snippet:doc-write-attribute-async-data-access]
    op = renderer.write_attribute_async(
        ["/World/Plane"],
        "omni:xform",
        cuda_tensor,
        data_access=ovrtx.DataAccess.ASYNC,
        cuda_stream=stream.cuda_stream,
    )
    assert op.wait() is True
    # [/snippet:doc-write-attribute-async-data-access]

    assert _read_xform(renderer)[0, 3, 0] == 7.0


def test_write_token_array_attribute(renderer):
    """Write string data as a token array rather than a path relationship."""
    _load_base(renderer)

    # [snippet:doc-write-token-array]
    renderer.write_array_attribute(
        ["/World/Plane"],
        "omni:docTokens",
        [["sensor", "validated"]],
        is_token=True,
        prim_mode=ovrtx.PrimMode.CREATE_NEW,
    )
    # [/snippet:doc-write-token-array]

    prims = renderer.query_prims(
        require_all=[(ovrtx.FilterKind.HAS_ATTRIBUTE, "omni:docTokens")],
        attribute_filter_mode=ovrtx.AttributeFilterMode.SPECIFIC,
        attribute_names=["omni:docTokens"],
    )
    assert "/World/Plane" in prims
    token_info = prims["/World/Plane"]["omni:docTokens"]
    assert token_info.is_array is True
    assert token_info.semantic == ovrtx.Semantic.TOKEN_ID

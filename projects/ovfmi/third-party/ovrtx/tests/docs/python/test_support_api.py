# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Tests for low-level Python support types exposed by ovrtx."""

from pathlib import Path

import numpy as np
import ovrtx

TEST_BASE_PATH = str((Path(__file__).parent / "../data/ovrtx-test-base.usda").resolve())


def test_dl_data_type_from_str():
    """Construct DLPack dtype descriptors from string aliases."""
    # [snippet:doc-dldata-type-from-str]
    point_dtype = ovrtx.DLDataType.from_str("float32", lanes=3)
    assert point_dtype.bits == 32
    assert point_dtype.lanes == 3
    # [/snippet:doc-dldata-type-from-str]


def test_managed_dl_tensor_helpers(renderer):
    """Exercise convenience methods on a CPU ManagedDLTensor."""
    renderer.open_usd(TEST_BASE_PATH)
    renderer.write_attribute(
        ["/Render/Camera"],
        "omni:rtx:rtpt:maxBounces",
        np.array([19], dtype=np.int32),
    )

    # [snippet:doc-managed-dltensor-helpers]
    tensor = renderer.read_attribute("omni:rtx:rtpt:maxBounces", ["/Render/Camera"])
    values = tensor.numpy()
    raw = tensor.to_bytes()
    device_type, device_id = tensor.__dlpack_device__()
    # [/snippet:doc-managed-dltensor-helpers]

    assert isinstance(tensor, ovrtx.ManagedDLTensor)
    assert int(values[0]) == 19
    assert len(raw) == values.nbytes
    assert (device_type, device_id) == (1, 0)  # DLPack kDLCPU, device 0


def test_renderer_config_version_and_async_stage(output_dir):
    """Cover RendererConfig, version, config echo, and void async stage ops."""
    # [snippet:doc-renderer-config]
    config = ovrtx.RendererConfig(
        sync_mode=True,
        log_file_path=str(output_dir / "config-test.log"),
        log_level="info",
    )
    renderer = ovrtx.Renderer(config=config)
    assert any(component > 0 for component in renderer.version)
    assert renderer.config.sync_mode is True
    assert renderer.config.log_level == "info"
    assert renderer.config.log_file_path == str(output_dir / "config-test.log")
    # [/snippet:doc-renderer-config]

    # [snippet:doc-open-usd-async]
    op = renderer.open_usd_async(TEST_BASE_PATH)
    # TODO: Restore the assertion once the packaged open_usd_async wrapper returns _VOID_RESULT.
    op.wait()
    assert "/World/Plane" in renderer.query_prims(attribute_filter_mode=ovrtx.AttributeFilterMode.NONE)
    # [/snippet:doc-open-usd-async]

    # [snippet:doc-open-usd-from-string-async]
    inline = '#usda 1.0\ndef Xform "World" {}\n'
    # TODO: Restore the assertion once the packaged open_usd_from_string_async wrapper returns _VOID_RESULT.
    renderer.open_usd_from_string_async(inline).wait()
    # [/snippet:doc-open-usd-from-string-async]

    # [snippet:doc-reset-async]
    assert renderer.reset_async().wait() is True
    # [/snippet:doc-reset-async]

    # [snippet:doc-reset-stage-async]
    assert renderer.reset_stage_async().wait() is True
    # [/snippet:doc-reset-stage-async]

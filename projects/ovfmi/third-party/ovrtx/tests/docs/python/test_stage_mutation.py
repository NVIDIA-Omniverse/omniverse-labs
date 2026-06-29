# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Tests for runtime stage mutation: additive USD references, removal, and clone."""

from pathlib import Path

import numpy as np
import ovrtx

TEST_BASE_PATH = str((Path(__file__).parent / "../data/ovrtx-test-base.usda").resolve())

ROOT_USDA = """#usda 1.0
def Xform "World" {
}
"""

REFERENCE_USDA = """#usda 1.0
(
    defaultPrim = "Referenced"
)

def Xform "Referenced" {
    def Cube "KnownChild" {
    }
}
"""


def _query_paths(renderer):
    return renderer.query_prims(attribute_filter_mode=ovrtx.AttributeFilterMode.NONE)


def test_add_remove_usd_reference_file(renderer, tmp_path):
    """Add a file reference, prove composed child content exists, then remove it."""
    renderer.open_usd_from_string(ROOT_USDA)
    reference_file = tmp_path / "referenced.usda"
    reference_file.write_text(REFERENCE_USDA)

    # [snippet:doc-add-remove-usd-reference]
    handle = renderer.add_usd_reference(str(reference_file), "/World/LoadedBase")
    prims = renderer.query_prims(attribute_filter_mode=ovrtx.AttributeFilterMode.NONE)
    assert "/World/LoadedBase" in prims
    assert "/World/LoadedBase/KnownChild" in prims

    renderer.remove_usd(handle)
    prims = renderer.query_prims(attribute_filter_mode=ovrtx.AttributeFilterMode.NONE)
    assert "/World/LoadedBase" not in prims
    assert "/World/LoadedBase/KnownChild" not in prims
    # [/snippet:doc-add-remove-usd-reference]


def test_add_remove_usd_reference_from_string(renderer):
    """Add an inline reference layer, prove child content exists, then remove it."""
    renderer.open_usd_from_string(ROOT_USDA)

    # [snippet:doc-add-usd-reference-from-string]
    handle = renderer.add_usd_reference_from_string(REFERENCE_USDA, "/World/Injected")
    prims = renderer.query_prims(attribute_filter_mode=ovrtx.AttributeFilterMode.NONE)
    assert "/World/Injected" in prims
    assert "/World/Injected/KnownChild" in prims

    renderer.remove_usd(handle)
    # [/snippet:doc-add-usd-reference-from-string]

    prims = _query_paths(renderer)
    assert "/World/Injected" not in prims
    assert "/World/Injected/KnownChild" not in prims


def test_clone_usd(renderer):
    """Clone a mesh subtree and verify the clone keeps mesh data."""
    renderer.open_usd(TEST_BASE_PATH)
    renderer.reset()
    source_points = np.from_dlpack(
        renderer.read_array_attribute("points", ["/World/Plane"])["/World/Plane"]
    ).copy()

    # [snippet:doc-clone-usd]
    renderer.clone_usd("/World/Plane", ["/World/PlaneCloneA", "/World/PlaneCloneB"])
    prims = renderer.query_prims(attribute_filter_mode=ovrtx.AttributeFilterMode.NONE)
    assert "/World/PlaneCloneA" in prims
    assert "/World/PlaneCloneB" in prims
    # [/snippet:doc-clone-usd]

    meshes = renderer.query_prims(require_all=[(ovrtx.FilterKind.PRIM_TYPE, "Mesh")])
    assert "/World/PlaneCloneA" in meshes
    clone_points = np.from_dlpack(
        renderer.read_array_attribute("points", ["/World/PlaneCloneA"])["/World/PlaneCloneA"]
    )
    np.testing.assert_allclose(clone_points, source_points)


def test_clone_usd_async(renderer):
    """Exercise the async clone operation for void-operation wait semantics."""
    renderer.open_usd(TEST_BASE_PATH)
    renderer.reset()

    # [snippet:doc-clone-usd-async]
    op = renderer.clone_usd_async("/World/Plane", ["/World/PlaneCloneAsync"])
    assert op.wait() is True
    # [/snippet:doc-clone-usd-async]

    prims = _query_paths(renderer)
    assert "/World/PlaneCloneAsync" in prims

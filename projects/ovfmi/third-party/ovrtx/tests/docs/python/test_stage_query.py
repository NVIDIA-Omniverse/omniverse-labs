# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Tests for the stage query API (``Renderer.query_prims`` / ``query_prims_async``)."""

from pathlib import Path

import ovrtx

TEST_BASE_PATH = str((Path(__file__).parent / "../data/ovrtx-test-base.usda").resolve())
INLINE_SUBLAYERS_PATH = str(
    (Path(__file__).parent / "../usd/data/inline_sublayers_camera_renderproduct.usda").resolve()
)


def _load_base(renderer):
    renderer.open_usd(TEST_BASE_PATH)
    renderer.reset()


def test_query_all_prims_no_attrs(renderer):
    """Query every prim on the stage without attribute descriptors."""
    _load_base(renderer)

    # [snippet:doc-query-prims-basic]
    # Fetch every prim on the stage. With AttributeFilterMode.NONE the result
    # dict maps each prim path to an empty attributes dict (lightweight).
    prims = renderer.query_prims(attribute_filter_mode=ovrtx.AttributeFilterMode.NONE)
    for prim_path, attributes in prims.items():
        print(f"{prim_path}: {len(attributes)} attributes")
    # [/snippet:doc-query-prims-basic]

    # Known prims from ovrtx-test-base-*.usda sublayers.
    for expected in [
        "/World/Camera",
        "/World/DomeLight",
        "/World/Plane",
        "/World/Looks/srf_glass",
        "/Render/Camera",
    ]:
        assert expected in prims, f"expected {expected} in query result"

    # NONE mode means no attribute descriptors per prim.
    for attrs in prims.values():
        assert attrs == {}


def test_query_by_prim_type(renderer):
    """Filter prims by USD type using ``FilterKind.PRIM_TYPE``."""
    _load_base(renderer)

    # [snippet:doc-query-prims-by-type]
    # AND filter: require prim type == "Mesh".
    meshes = renderer.query_prims(
        require_all=[(ovrtx.FilterKind.PRIM_TYPE, "Mesh")],
    )
    # [/snippet:doc-query-prims-by-type]

    assert len(meshes) > 0, "expected at least one Mesh in the base scene"
    # The flat Plane is always a Mesh; the logo leaf is a Mesh too.
    assert "/World/Plane" in meshes

    # Confirm PRIM_TYPE=Camera narrows to exactly the one camera in the scene.
    cameras = renderer.query_prims(
        require_all=[(ovrtx.FilterKind.PRIM_TYPE, "Camera")],
    )
    assert list(cameras.keys()) == ["/World/Camera"]


def test_query_with_attribute_filter(renderer):
    """Use ``AttributeFilterMode.SPECIFIC`` to request descriptors for named attributes."""
    _load_base(renderer)

    # [snippet:doc-query-prims-with-attributes]
    # Find meshes and get AttributeInfo descriptors for "points" and "material:binding".
    # AttributeInfo exposes dtype, is_array, and semantic — enough to decide how
    # to read the attribute next. Relationship attributes surface with
    # Semantic.PATH_ID; resolve those IDs through the renderer's path dictionary.
    meshes = renderer.query_prims(
        require_all=[(ovrtx.FilterKind.PRIM_TYPE, "Mesh")],
        attribute_filter_mode=ovrtx.AttributeFilterMode.SPECIFIC,
        attribute_names=["points", "material:binding"],
    )
    for prim_path, attributes in meshes.items():
        for name, info in attributes.items():
            print(f"{prim_path}.{name}: dtype={info.dtype} array={info.is_array} semantic={info.semantic.name}")
    # [/snippet:doc-query-prims-with-attributes]

    assert "/World/Plane" in meshes
    plane_attrs = meshes["/World/Plane"]

    # "points" is a per-vertex float3 array.
    assert "points" in plane_attrs
    points = plane_attrs["points"]
    assert isinstance(points, ovrtx.AttributeInfo)
    assert points.is_array is True

    # "material:binding" is a relationship; it surfaces as a PATH_ID semantic
    # (raw path handle — resolve via the renderer's path dictionary when you
    # need the actual string path).
    assert "material:binding" in plane_attrs
    binding = plane_attrs["material:binding"]
    assert binding.semantic == ovrtx.Semantic.PATH_ID


def test_query_prims_async(renderer):
    """Exercise the two-phase Operation/PendingFetch lifecycle for queries."""
    _load_base(renderer)

    # [snippet:doc-query-prims-async]
    # Two-phase async: wait() → PendingFetch, then fetch() → result dict.
    op = renderer.query_prims_async(
        require_all=[(ovrtx.FilterKind.PRIM_TYPE, "Mesh")],
    )
    pending = op.wait()
    meshes = pending.fetch()
    # [/snippet:doc-query-prims-async]

    assert isinstance(op, ovrtx.Operation)
    assert isinstance(pending, ovrtx.PendingFetch)
    assert "/World/Plane" in meshes


def test_query_require_any_exclude_all_attrs(renderer):
    """Exercise OR, NOT, and ALL-attributes query options together."""
    _load_base(renderer)

    # [snippet:doc-query-require-any-exclude]
    # Match Mesh or Camera prims, then exclude Camera. The exclusion removes
    # a prim that would otherwise match the OR clause.
    prims = renderer.query_prims(
        require_any=[
            (ovrtx.FilterKind.PRIM_TYPE, "Mesh"),
            (ovrtx.FilterKind.PRIM_TYPE, "Camera"),
        ],
        exclude=[(ovrtx.FilterKind.PRIM_TYPE, "Camera")],
        attribute_filter_mode=ovrtx.AttributeFilterMode.ALL,
    )
    # [/snippet:doc-query-require-any-exclude]

    assert "/World/Plane" in prims
    assert "/World/Camera" not in prims
    assert prims["/World/Plane"]


def test_query_specific_empty_attribute_list(renderer):
    """SPECIFIC with no requested names returns matched prims with no descriptors."""
    _load_base(renderer)

    # [snippet:doc-query-specific-empty-attributes]
    result = renderer.query_prims(
        require_all=[(ovrtx.FilterKind.PRIM_TYPE, "Mesh")],
        attribute_filter_mode=ovrtx.AttributeFilterMode.SPECIFIC,
        attribute_names=[],
    )
    assert all(attrs == {} for attrs in result.values())
    # [/snippet:doc-query-specific-empty-attributes]

    assert "/World/Plane" in result


def test_query_inline_sublayer_composition(renderer):
    """Query prims from both the inline root layer and its composed sublayer."""
    renderer.open_usd(INLINE_SUBLAYERS_PATH)
    renderer.reset()

    # [snippet:doc-query-inline-sublayer-composition]
    prims = renderer.query_prims(attribute_filter_mode=ovrtx.AttributeFilterMode.NONE)
    for expected in [
        "/World/Plane",  # from the sublayered base scene
        "/World/Camera",  # from the sublayered base scene
        "/DocsCamera",  # authored in the inline root layer
        "/Render/DocsCamera",  # authored in the inline root layer
    ]:
        assert expected in prims
    # [/snippet:doc-query-inline-sublayer-composition]

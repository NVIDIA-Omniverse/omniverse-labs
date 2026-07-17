"""Deterministic articulation profile for the supplied Home Scan kitchen."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .core import AXES, Bounds, RealToSimError, Workspace
from .gaussian import GaussianScene, load_gaussians, select_bounds
from .segmentation import PLANAR_REFINER, refine_planar_selection
from .sim import add_node, fit_joint
from .visual_completion import author_visual_completion


HOME_SCAN_WORKING_GAUSSIANS = 671_787
HOME_KITCHEN_PROFILE = "home-scan-kitchen-v2"


@dataclass(frozen=True, slots=True)
class PanelProfile:
    node_id: str
    label: str
    parent: str
    kind: str
    selection: Bounds
    collider: Bounds
    front_axis: str
    outward_sign: int
    depth: float
    hinge_side: int | None = None
    upper: float | None = None
    shelf_count: int = 1
    segmentation_inward_band: float | None = None
    segmentation_outward_band: float | None = None


def _bounds(values: tuple[float, float, float, float, float, float]) -> Bounds:
    return Bounds(values[:3], values[3:])


PARENT_PROFILES: tuple[tuple[str, str, Bounds, str], ...] = (
    (
        "kitchen_cabinet_bank",
        "Oven wall and corner cabinet shells",
        _bounds((-9.55, -2.50, -7.05, -8.10, 0.08, -4.05)),
        "oven-wall",
    ),
    (
        "refrigerator_body",
        "Refrigerator shell",
        _bounds((-8.01, -2.20, -5.28, -6.70, 0.08, -4.35)),
        "refrigerator",
    ),
    (
        "fridge_upper_cabinet_bank",
        "Cabinet shell above refrigerator",
        _bounds((-8.05, -2.52, -5.02, -6.70, -2.23, -4.30)),
        "fridge-upper",
    ),
    (
        "sink_cabinet_bank",
        "Sink wall cabinet shells",
        _bounds((-8.80, -2.52, -8.45, -5.40, 0.08, -7.05)),
        "sink-wall",
    ),
    (
        "peninsula_cabinet_bank",
        "Peninsula outer cabinet shell",
        _bounds((-4.85, -1.20, -8.50, -4.00, 0.08, -5.18)),
        "peninsula",
    ),
)


PANELS: tuple[PanelProfile, ...] = (
    # Oven-wall upper cabinets.
    PanelProfile(
        "oven_upper_left_outer",
        "Oven wall upper cabinet left outer door",
        "kitchen_cabinet_bank",
        "cabinet-door",
        _bounds((-9.37, -2.49, -6.68, -9.10, -1.45, -6.27)),
        _bounds((-9.33, -2.47, -6.67, -9.18, -1.47, -6.28)),
        "X",
        1,
        0.42,
        hinge_side=-1,
    ),
    PanelProfile(
        "oven_upper_left_inner",
        "Oven wall upper cabinet left inner door",
        "kitchen_cabinet_bank",
        "cabinet-door",
        _bounds((-9.37, -2.49, -6.25, -9.10, -1.45, -5.83)),
        _bounds((-9.33, -2.47, -6.24, -9.18, -1.47, -5.84)),
        "X",
        1,
        0.42,
        hinge_side=1,
    ),
    PanelProfile(
        "microwave_upper_left",
        "Cabinet door above microwave left",
        "kitchen_cabinet_bank",
        "cabinet-door",
        _bounds((-9.37, -2.49, -5.79, -9.10, -1.64, -5.29)),
        _bounds((-9.33, -2.47, -5.78, -9.18, -1.66, -5.30)),
        "X",
        1,
        0.38,
        hinge_side=-1,
    ),
    PanelProfile(
        "microwave_upper_right",
        "Cabinet door above microwave right",
        "kitchen_cabinet_bank",
        "cabinet-door",
        _bounds((-9.37, -2.49, -5.27, -9.10, -1.64, -4.75)),
        _bounds((-9.33, -2.47, -5.26, -9.18, -1.66, -4.76)),
        "X",
        1,
        0.38,
        hinge_side=1,
    ),
    PanelProfile(
        "oven_upper_right",
        "Oven wall upper cabinet right door",
        "kitchen_cabinet_bank",
        "cabinet-door",
        _bounds((-9.37, -2.49, -4.72, -9.08, -1.43, -4.40)),
        _bounds((-9.33, -2.47, -4.71, -9.17, -1.45, -4.41)),
        "X",
        1,
        0.38,
        hinge_side=1,
        # Closed/half/open measured-only review showed that the narrower band
        # captured trim but left most of this narrow corner front in the static
        # background. Keep twelve negative LOD5 references while spanning the
        # two depth lobes that visually form the complete front.
        segmentation_inward_band=0.14,
        segmentation_outward_band=0.14,
    ),
    # Corner upper cabinets facing the refrigerator.
    PanelProfile(
        "corner_upper_left",
        "Corner upper cabinet left door",
        "kitchen_cabinet_bank",
        "cabinet-door",
        _bounds((-9.02, -2.49, -4.37, -8.57, -1.42, -4.15)),
        _bounds((-9.00, -2.47, -4.34, -8.58, -1.44, -4.18)),
        "Z",
        -1,
        0.38,
        hinge_side=-1,
    ),
    PanelProfile(
        "corner_upper_right",
        "Corner upper cabinet right door",
        "kitchen_cabinet_bank",
        "cabinet-door",
        _bounds((-8.55, -2.49, -4.37, -8.12, -1.42, -4.15)),
        _bounds((-8.54, -2.47, -4.34, -8.13, -1.44, -4.18)),
        "Z",
        -1,
        0.38,
        hinge_side=1,
    ),
    # Three drawers left of the oven.
    PanelProfile(
        "oven_drawer_top",
        "Top drawer left of oven",
        "kitchen_cabinet_bank",
        "drawer",
        _bounds((-9.15, -1.03, -6.68, -8.74, -0.77, -5.85)),
        _bounds((-9.02, -1.02, -6.67, -8.76, -0.78, -5.86)),
        "X",
        1,
        0.50,
        upper=0.44,
        shelf_count=0,
    ),
    PanelProfile(
        "oven_drawer_middle",
        "Middle drawer left of oven",
        "kitchen_cabinet_bank",
        "drawer",
        _bounds((-9.15, -0.76, -6.68, -8.74, -0.35, -5.85)),
        _bounds((-9.02, -0.75, -6.67, -8.76, -0.36, -5.86)),
        "X",
        1,
        0.50,
        upper=0.44,
        shelf_count=0,
    ),
    PanelProfile(
        "oven_drawer_bottom",
        "Bottom drawer left of oven",
        "kitchen_cabinet_bank",
        "drawer",
        _bounds((-9.15, -0.34, -6.68, -8.74, 0.07, -5.85)),
        _bounds((-9.02, -0.33, -6.67, -8.76, 0.05, -5.86)),
        "X",
        1,
        0.50,
        upper=0.44,
        shelf_count=0,
    ),
    PanelProfile(
        "oven_right_base_door",
        "Base cabinet door right of oven",
        "kitchen_cabinet_bank",
        "cabinet-door",
        _bounds((-9.15, -1.05, -4.70, -8.74, 0.07, -4.36)),
        _bounds((-9.02, -1.03, -4.69, -8.76, 0.05, -4.37)),
        "X",
        1,
        0.50,
        hinge_side=1,
    ),
    # Perpendicular base cabinet visible left of the refrigerator.
    PanelProfile(
        "corner_base_drawer",
        "Corner base top drawer",
        "kitchen_cabinet_bank",
        "drawer",
        _bounds((-8.52, -1.05, -4.72, -8.12, -0.77, -4.50)),
        _bounds((-8.51, -1.03, -4.70, -8.13, -0.78, -4.54)),
        "Z",
        -1,
        0.50,
        upper=0.42,
        shelf_count=0,
    ),
    PanelProfile(
        "corner_base_left_door",
        "Corner base left door",
        "kitchen_cabinet_bank",
        "cabinet-door",
        _bounds((-8.52, -0.76, -4.72, -8.33, 0.07, -4.50)),
        _bounds((-8.51, -0.75, -4.70, -8.34, 0.05, -4.54)),
        "Z",
        -1,
        0.50,
        hinge_side=-1,
    ),
    PanelProfile(
        "corner_base_right_door",
        "Corner base right door",
        "kitchen_cabinet_bank",
        "cabinet-door",
        _bounds((-8.31, -0.76, -4.72, -8.12, 0.07, -4.50)),
        _bounds((-8.30, -0.75, -4.70, -8.13, 0.05, -4.54)),
        "Z",
        -1,
        0.50,
        hinge_side=1,
    ),
    # Refrigerator and cabinets above it.
    PanelProfile(
        "fridge_left_door",
        "Refrigerator left door",
        "refrigerator_body",
        "refrigerator-door",
        _bounds((-8.03, -2.22, -4.98, -7.40, 0.08, -4.82)),
        _bounds((-8.01, -2.20, -5.04, -7.41, 0.05, -4.76)),
        "Z",
        -1,
        0.52,
        hinge_side=-1,
        upper=95.0,
        shelf_count=3,
    ),
    PanelProfile(
        "fridge_right_door",
        "Refrigerator right door",
        "refrigerator_body",
        "refrigerator-door",
        _bounds((-7.38, -2.22, -4.98, -6.73, 0.08, -4.82)),
        _bounds((-7.37, -2.20, -5.04, -6.75, 0.05, -4.76)),
        "Z",
        -1,
        0.52,
        hinge_side=1,
        upper=95.0,
        shelf_count=3,
    ),
    PanelProfile(
        "fridge_upper_left",
        "Cabinet above refrigerator left door",
        "fridge_upper_cabinet_bank",
        "cabinet-door",
        _bounds((-8.02, -2.50, -4.75, -7.40, -2.20, -4.48)),
        _bounds((-8.00, -2.48, -4.70, -7.41, -2.22, -4.53)),
        "Z",
        -1,
        0.34,
        hinge_side=-1,
        shelf_count=0,
    ),
    PanelProfile(
        "fridge_upper_right",
        "Cabinet above refrigerator right door",
        "fridge_upper_cabinet_bank",
        "cabinet-door",
        _bounds((-7.38, -2.50, -4.75, -6.74, -2.20, -4.48)),
        _bounds((-7.37, -2.48, -4.70, -6.76, -2.22, -4.53)),
        "Z",
        -1,
        0.34,
        hinge_side=1,
        shelf_count=0,
    ),
    # Sink wall.
    PanelProfile(
        "sink_upper_left",
        "Sink upper cabinet left door",
        "sink_cabinet_bank",
        "cabinet-door",
        _bounds((-7.78, -2.49, -8.34, -7.12, -1.20, -8.02)),
        _bounds((-7.76, -2.47, -8.22, -7.13, -1.22, -8.04)),
        "Z",
        1,
        0.40,
        hinge_side=-1,
    ),
    PanelProfile(
        "sink_upper_right",
        "Sink upper cabinet right door",
        "sink_cabinet_bank",
        "cabinet-door",
        _bounds((-7.10, -2.49, -8.34, -6.42, -1.20, -8.02)),
        _bounds((-7.09, -2.47, -8.22, -6.44, -1.22, -8.04)),
        "Z",
        1,
        0.40,
        hinge_side=1,
    ),
    PanelProfile(
        "sink_base_drawer",
        "Sink base drawer",
        "sink_cabinet_bank",
        "drawer",
        _bounds((-7.18, -1.05, -7.92, -6.28, -0.76, -7.68)),
        _bounds((-7.17, -1.03, -7.88, -6.29, -0.77, -7.70)),
        "Z",
        1,
        0.48,
        upper=0.42,
        shelf_count=0,
    ),
    PanelProfile(
        "sink_base_left_door",
        "Sink base left door",
        "sink_cabinet_bank",
        "cabinet-door",
        _bounds((-7.18, -0.74, -7.92, -6.73, 0.06, -7.68)),
        _bounds((-7.17, -0.73, -7.88, -6.74, 0.04, -7.70)),
        "Z",
        1,
        0.48,
        hinge_side=-1,
    ),
    PanelProfile(
        "sink_base_right_door",
        "Sink base right door",
        "sink_cabinet_bank",
        "cabinet-door",
        _bounds((-6.71, -0.74, -7.92, -6.25, 0.06, -7.68)),
        _bounds((-6.70, -0.73, -7.88, -6.26, 0.04, -7.70)),
        "Z",
        1,
        0.48,
        hinge_side=1,
    ),
    # Four outer peninsula doors.
    PanelProfile(
        "peninsula_outer_door_1",
        "Peninsula outer door 1",
        "peninsula_cabinet_bank",
        "cabinet-door",
        _bounds((-4.76, -1.13, -8.35, -4.20, 0.06, -7.65)),
        _bounds((-4.58, -1.11, -8.34, -4.32, 0.04, -7.66)),
        "X",
        1,
        0.46,
        hinge_side=-1,
    ),
    PanelProfile(
        "peninsula_outer_door_2",
        "Peninsula outer door 2",
        "peninsula_cabinet_bank",
        "cabinet-door",
        _bounds((-4.76, -1.13, -7.62, -4.20, 0.06, -6.86)),
        _bounds((-4.58, -1.11, -7.61, -4.32, 0.04, -6.88)),
        "X",
        1,
        0.46,
        hinge_side=1,
    ),
    PanelProfile(
        "peninsula_outer_door_3",
        "Peninsula outer door 3",
        "peninsula_cabinet_bank",
        "cabinet-door",
        _bounds((-4.76, -1.13, -6.82, -4.20, 0.08, -6.04)),
        _bounds((-4.58, -1.11, -6.81, -4.32, 0.06, -6.05)),
        "X",
        1,
        0.46,
        hinge_side=-1,
    ),
    PanelProfile(
        "peninsula_outer_door_4",
        "Peninsula outer door 4",
        "peninsula_cabinet_bank",
        "cabinet-door",
        _bounds((-4.76, -1.13, -6.02, -4.20, 0.06, -5.27)),
        _bounds((-4.58, -1.11, -6.01, -4.32, 0.04, -5.28)),
        "X",
        1,
        0.46,
        hinge_side=1,
    ),
)


OVEN_SELECTION = _bounds((-9.15, -0.84, -5.82, -8.55, 0.08, -4.72))
OVEN_COLLIDER = _bounds((-9.00, -0.82, -5.82, -8.72, 0.055, -4.72))
OVEN_APERTURE_MASK = _bounds((-9.78, -0.82, -5.82, -8.55, 0.055, -4.72))


def _indices(scene: GaussianScene, bounds: Bounds, *, label: str) -> np.ndarray:
    selected = select_bounds(scene, bounds)
    if len(selected) < 24:
        raise RealToSimError(
            f"Home Scan panel selection is too sparse for {label}: {len(selected)} Gaussians"
        )
    return selected


def _door_axis_sign(profile: PanelProfile) -> int:
    if profile.hinge_side not in {-1, 1}:
        raise RealToSimError(f"door profile has no hinge side: {profile.node_id}")
    if profile.front_axis == "X":
        return -profile.hinge_side * profile.outward_sign
    if profile.front_axis == "Z":
        return profile.hinge_side * profile.outward_sign
    raise RealToSimError("Home Scan cabinet doors currently require X- or Z-facing fronts")


def _physical_panel_collider(profile: PanelProfile) -> Bounds:
    axis = AXES[profile.front_axis]
    thickness = {
        "cabinet-door": 0.065,
        "drawer": 0.080,
        "refrigerator-door": 0.160,
    }[profile.kind]
    center = np.asarray(profile.collider.center, dtype=np.float64)
    minimum = np.asarray(profile.collider.minimum, dtype=np.float64)
    maximum = np.asarray(profile.collider.maximum, dtype=np.float64)
    minimum[axis] = center[axis] - thickness * 0.5
    maximum[axis] = center[axis] + thickness * 0.5
    return Bounds(tuple(minimum), tuple(maximum))


def _door_origin(
    workspace: Workspace,
    profile: PanelProfile,
    collider: Bounds,
) -> tuple[float, float, float]:
    front = AXES[profile.front_axis]
    up = AXES[workspace.up_axis]
    width = next(index for index in range(3) if index not in {front, up})
    origin = np.asarray(collider.center, dtype=np.float64)
    origin[width] = (
        collider.minimum[width]
        if profile.hinge_side == -1
        else collider.maximum[width]
    )
    origin[front] = (
        collider.maximum[front]
        if profile.outward_sign > 0
        else collider.minimum[front]
    )
    return tuple(float(value) for value in origin)


def _completion_counts(kind: str) -> tuple[int, int]:
    if kind == "refrigerator-door":
        return 3200, 1450
    if kind == "drawer":
        return 1500, 1750
    return 1850, 700


def _add_parent(
    workspace: Workspace,
    scene: GaussianScene,
    *,
    node_id: str,
    label: str,
    bounds: Bounds,
    region: str,
) -> None:
    add_node(
        workspace,
        node_id=node_id,
        label=label,
        role="static",
        source_indices=_indices(scene, bounds, label=node_id),
        scene=scene,
        collider_bounds=bounds,
        collider_confidence=0.82,
        collider_provenance=f"{HOME_KITCHEN_PROFILE}:measured-shell",
        collision_mode="shell",
        tags=("measured", "kitchen", "cabinet-shell", HOME_KITCHEN_PROFILE, region),
    )


def _add_panel(
    workspace: Workspace,
    scene: GaussianScene,
    profile: PanelProfile,
) -> dict[str, Any]:
    segmentation = refine_planar_selection(
        scene,
        profile.selection,
        front_axis=profile.front_axis,
        outward_sign=profile.outward_sign,
        kind=profile.kind,
        inward_band=profile.segmentation_inward_band,
        outward_band=profile.segmentation_outward_band,
    )
    selected = segmentation.indices
    collider = _physical_panel_collider(profile)
    add_node(
        workspace,
        node_id=profile.node_id,
        label=profile.label,
        role="movable",
        source_indices=selected,
        scene=scene,
        collider_bounds=collider,
        collider_confidence=0.78,
        collider_provenance=f"{HOME_KITCHEN_PROFILE}:panel-bounds",
        tags=(
            "measured",
            "kitchen",
            "visual-refined",
            PLANAR_REFINER,
            HOME_KITCHEN_PROFILE,
            profile.kind,
        ),
        selection_mode="stable-reference",
        selection_bounds=profile.selection,
    )
    if profile.kind == "drawer":
        joint = fit_joint(
            workspace,
            node_id=profile.node_id,
            parent_id=profile.parent,
            kind="prismatic",
            axis=profile.front_axis,
            axis_sign=profile.outward_sign,
            origin=collider.center,
            lower=0.0,
            upper=profile.upper or min(profile.depth * 0.84, 0.46),
        )
    else:
        joint = fit_joint(
            workspace,
            node_id=profile.node_id,
            parent_id=profile.parent,
            kind="revolute",
            axis=workspace.up_axis,
            axis_sign=_door_axis_sign(profile),
            origin=_door_origin(workspace, profile, collider),
            lower=0.0,
            upper=profile.upper or 95.0,
        )
    static_count, moving_count = _completion_counts(profile.kind)
    completion = author_visual_completion(
        workspace,
        node_id=profile.node_id,
        kind=profile.kind,
        front_axis=profile.front_axis,
        outward_sign=profile.outward_sign,
        depth=profile.depth,
        up_sign=-1,
        shelf_count=profile.shelf_count,
        static_gaussians=static_count,
        moving_gaussians=moving_count,
        confidence=0.72,
    )
    return {
        "id": profile.node_id,
        "selected_gaussians": int(len(selected)),
        "joint": joint.kind,
        "completion": completion["id"],
        "segmentation": segmentation.diagnostics,
    }


def author_home_scan_kitchen(workspace: Workspace) -> dict[str, Any]:
    """Replace the appliance-only prototype with the full visible kitchen profile."""

    scene = load_gaussians(workspace.source_path)
    if scene.count != HOME_SCAN_WORKING_GAUSSIANS:
        raise RealToSimError(
            "the Home Scan kitchen profile is pinned to the supplied LOD5 source "
            f"({HOME_SCAN_WORKING_GAUSSIANS} Gaussians), got {scene.count}"
        )

    try:
        existing_oven = workspace.node("oven_door")
        if existing_oven.selection_mode == "stable-reference":
            oven_indices = workspace.load_selection(existing_oven).astype(np.uint32)
            oven_selection_mode = "stable-reference"
            oven_selection_bounds = existing_oven.selection_bounds or OVEN_SELECTION
            oven_segmentation: dict[str, Any] = {
                "schema_version": 1,
                "refiner": "preserved-existing-stable-reference",
                "positive_references": int(len(oven_indices)),
                "proposal_bounds": oven_selection_bounds.to_json(),
            }
        else:
            raise RealToSimError("replace bounds-selected oven with planar references")
    except RealToSimError:
        refined_oven = refine_planar_selection(
            scene,
            OVEN_SELECTION,
            front_axis="X",
            outward_sign=1,
            kind="oven-door",
        )
        oven_indices = refined_oven.indices
        oven_selection_mode = "stable-reference"
        oven_selection_bounds = OVEN_SELECTION
        oven_segmentation = refined_oven.diagnostics

    panel_ids = {profile.node_id for profile in PANELS}
    cleanup_ids = panel_ids | {
        "oven_door",
        "utility_drawer",
        *(item[0] for item in PARENT_PROFILES),
    }
    cleanup_ids.update(
        node.node_id
        for node in workspace.nodes
        if HOME_KITCHEN_PROFILE in node.tags
    )
    workspace.remove_completions_for_nodes(cleanup_ids)
    for node_id in sorted(cleanup_ids):
        try:
            workspace.remove_node(node_id)
        except RealToSimError:
            pass

    for node_id, label, bounds, region in PARENT_PROFILES:
        _add_parent(
            workspace,
            scene,
            node_id=node_id,
            label=label,
            bounds=bounds,
            region=region,
        )

    # Keep the hand-segmented measured oven front when it is available.
    add_node(
        workspace,
        node_id="oven_door",
        label="Oven door",
        role="movable",
        source_indices=oven_indices,
        scene=scene,
        collider_bounds=OVEN_COLLIDER,
        collider_confidence=0.88,
        collider_provenance=f"{HOME_KITCHEN_PROFILE}:stable-oven-front",
        tags=(
            "measured",
            "kitchen",
            "visual-refined",
            HOME_KITCHEN_PROFILE,
            "door",
            "oven",
        ),
        selection_mode=oven_selection_mode,
        selection_bounds=oven_selection_bounds,
    )
    oven_joint = fit_joint(
        workspace,
        node_id="oven_door",
        parent_id="kitchen_cabinet_bank",
        kind="revolute",
        axis="Z",
        axis_sign=1,
        origin=(-8.86, 0.055, -5.27),
        lower=0.0,
        upper=60.0,
    )
    oven_completion = author_visual_completion(
        workspace,
        node_id="oven_door",
        kind="oven-door",
        front_axis="X",
        outward_sign=1,
        depth=0.66,
        up_sign=-1,
        shelf_count=2,
        static_gaussians=3800,
        moving_gaussians=1400,
        confidence=0.82,
        background_occlusion_bounds=OVEN_APERTURE_MASK,
    )

    authored = [
        {
            "id": "oven_door",
            "selected_gaussians": int(len(oven_indices)),
            "joint": oven_joint.kind,
            "completion": oven_completion["id"],
            "segmentation": oven_segmentation,
        }
    ]
    for profile in PANELS:
        authored.append(_add_panel(workspace, scene, profile))

    report = {
        "schema_version": 1,
        "profile": HOME_KITCHEN_PROFILE,
        "source": str(Path(workspace.source_path)),
        "parents": [item[0] for item in PARENT_PROFILES],
        "articulated": authored,
        "articulated_count": len(authored),
        "removed_prototype_node": "utility_drawer",
        "provenance_contract": {
            "measured": "source selections and extracted LOD0 door/drawer fronts",
            "generated": "separate static cavity and joint-attached interior PLY assets",
        },
        "segmentation": {
            "method": PLANAR_REFINER,
            "bounds_are_proposals_only": True,
            "high_resolution_transfer": "positive and negative working-PLY references",
            "visual_feedback_required": ["closed", "half-open", "fully-open"],
        },
    }
    segmentation_root = workspace.root / "evidence" / "segmentation"
    segmentation_root.mkdir(parents=True, exist_ok=True)
    (segmentation_root / "refinement.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": HOME_KITCHEN_PROFILE,
                "parts": [
                    {
                        "id": item["id"],
                        "segmentation": item["segmentation"],
                    }
                    for item in authored
                ],
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    workspace.trace(
        "author-home-scan-kitchen",
        {"profile": HOME_KITCHEN_PROFILE},
        report,
    )
    return report

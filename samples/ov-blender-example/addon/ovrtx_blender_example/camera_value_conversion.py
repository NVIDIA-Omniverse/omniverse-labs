# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blender camera value edit conversion policy with live-honor probe classes.

Camera transform is already live (the viewport pose lane). Camera
projection/framing values are attempted as live value updates on the
composed camera prim behind a lazy per-session, per-class capability probe
(blender-live-render task04-05): the user's first edit of a class *is* the
probe — apply the edit, render one ``min_samples`` frame, and compare its
digest against the pre-edit frame at equal samples and the same snapshot
key. Honored classes stay on the value route for the session; unhonored
classes fold their values back into the OVRTX scene composition digest so
``reuse_decision`` forces a session replacement (background resync) — a
camera edit is never a silent no-op.

Probe classes (probed independently, cached per session):

- ``projection``: ``focalLength`` + ``horizontalAperture`` /
  ``verticalAperture`` on perspective cameras.
- ``clip``: ``clippingRange``.
- ``ortho``: the orthographic scale mapping — the aperture trio under an
  orthographic projection (Blender ``ortho_scale`` maps to apertures).

Everything else on the composed camera (projection kind, aperture offsets,
DOF) stays composition identity: changes take the replacement route as
before this task.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from . import render_requests
from . import usd_value_edit_support
from .value_edit_conversion import (
    BLENDER_DATABLOCK_NON_RENDER_FIELD_REASONS,
    FieldClassification,
    STATUS_SUPPORTED,
    STATUS_UNSUPPORTED,
    UsdAttributeValue,
    classify_mapped_field,
)


PROBE_CLASS_PROJECTION = "projection"
PROBE_CLASS_CLIP = "clip"
PROBE_CLASS_ORTHO = "ortho"
CAMERA_VALUE_PROBE_CLASSES = (
    PROBE_CLASS_CLIP,
    PROBE_CLASS_ORTHO,
    PROBE_CLASS_PROJECTION,
)

PROBE_STATUS_UNKNOWN = "unknown"
PROBE_STATUS_HONORED = "honored"
PROBE_STATUS_UNHONORED = "unhonored"

#: Inconclusive-probe reasons: the probe did not conclude and retries on
#: the next edit of the class (the edit itself is always applied).
PROBE_INCONCLUSIVE_POSE_CHANGED = "pose_changed_mid_probe"
PROBE_INCONCLUSIVE_NO_BASELINE = "no_pre_edit_baseline"
PROBE_INCONCLUSIVE_CONCURRENT_EDITS = "concurrent_edits_pending"
PROBE_INCONCLUSIVE_PHYSICS_ACTIVE = "physics_playback_active"
PROBE_INCONCLUSIVE_SAMPLE_MISMATCH = "probe_sample_mismatch"

#: Provenance ``source`` for camera projection value edits — distinct from
#: the pose lane's ``viewport_camera`` so planner/update-stream routing can
#: tell the attribute lane from the ``omni:xform`` transform lane.
VIEWPORT_CAMERA_PROJECTION_SOURCE = "viewport_camera_projection"

#: Composition-identity classification reasons (replacement route).
CAMERA_PROJECTION_KIND_CHANGED = "camera_projection_kind_is_composition_identity"
CAMERA_APERTURE_OFFSETS_COMPOSED = "camera_aperture_offsets_are_composition_identity"
CAMERA_DOF_COMPOSED = "camera_dof_is_composition_identity"

FINDINGS_OWNER = "Team Green"

SUPPORTED_USD_ATTRIBUTES = usd_value_edit_support.CAMERA_USD_VALUE_TYPES

_PROBED_APERTURE_ATTRIBUTES = (
    "focalLength",
    "horizontalAperture",
    "verticalAperture",
)

_TOPOLOGY_FIELD_REASONS = {
    "type": CAMERA_PROJECTION_KIND_CHANGED,
    "shift_x": CAMERA_APERTURE_OFFSETS_COMPOSED,
    "shift_y": CAMERA_APERTURE_OFFSETS_COMPOSED,
    "dof.use_dof": CAMERA_DOF_COMPOSED,
    "dof.aperture_fstop": CAMERA_DOF_COMPOSED,
    "dof.focus_distance": CAMERA_DOF_COMPOSED,
    "dof.focus_object": CAMERA_DOF_COMPOSED,
}

_NON_RENDER_FIELD_REASONS = {
    "name": "non_runtime_camera_identifier",
    "name_full": "non_runtime_camera_identifier",
    **BLENDER_DATABLOCK_NON_RENDER_FIELD_REASONS,
    "show_limits": "non_runtime_viewport_field",
    "show_passepartout": "non_runtime_viewport_field",
    "display_size": "non_runtime_viewport_field",
}

_UNSUPPORTED_FIELD_REASONS = {
    "lens_unit": "unsupported_camera_lens_unit_presentation",
    "show_composition_thirds": "unsupported_camera_composition_guide",
}


def probe_class_for_attribute(
    attribute: str,
    projection_token: str = "perspective",
) -> str:
    """Return the probe class owning a composed camera USD attribute.

    Empty string means the attribute is not value-routed: it stays in the
    composition digest and changes take the replacement route.
    """

    name = str(attribute or "")
    if name == "clippingRange":
        return PROBE_CLASS_CLIP
    if name in _PROBED_APERTURE_ATTRIBUTES:
        return (
            PROBE_CLASS_ORTHO
            if str(projection_token or "") == "orthographic"
            else PROBE_CLASS_PROJECTION
        )
    return ""


def classify_field(camera: Any, property_name: str) -> FieldClassification:
    """Classify a Blender camera data field for the live value-edit route."""

    field = str(property_name or "").strip()
    camera_type = str(getattr(camera, "type", "PERSP") or "PERSP").upper()
    orthographic = camera_type == "ORTHO"
    classification = classify_mapped_field(
        field,
        non_render=_NON_RENDER_FIELD_REASONS,
        topology=_TOPOLOGY_FIELD_REASONS,
        unsupported=_UNSUPPORTED_FIELD_REASONS,
    )
    if classification is not None:
        return classification
    if field == "lens":
        return FieldClassification(
            STATUS_SUPPORTED, "supported_camera_projection_value", ("focalLength",)
        )
    if field in {"sensor_width", "sensor_height", "sensor_fit"}:
        if orthographic:
            return FieldClassification(
                STATUS_UNSUPPORTED, "non_applicable_orthographic_sensor_field"
            )
        return FieldClassification(
            STATUS_SUPPORTED,
            "supported_camera_projection_value",
            ("horizontalAperture", "verticalAperture"),
        )
    if field == "ortho_scale":
        if not orthographic:
            return FieldClassification(
                STATUS_UNSUPPORTED, "non_applicable_perspective_ortho_scale"
            )
        return FieldClassification(
            STATUS_SUPPORTED,
            "supported_camera_ortho_value",
            ("horizontalAperture", "verticalAperture"),
        )
    if field in {"clip_start", "clip_end"}:
        return FieldClassification(
            STATUS_SUPPORTED, "supported_camera_clip_value", ("clippingRange",)
        )
    return FieldClassification(STATUS_UNSUPPORTED, "unsupported_camera_field")


def usd_attribute_values(camera_projection: Any) -> tuple[UsdAttributeValue, ...]:
    """Probed camera attribute values from a ``CameraProjectionState``.

    Parity is reuse: the values come from the exact
    ``CameraProjectionState.usd_attributes()`` mapping the OVRTX scene
    composition authors into the composed camera, so a live value update
    always carries the value a replacement session would compose.
    """

    attributes = render_requests.camera_projection_usd_attributes(camera_projection)
    if not attributes:
        return ()
    projection_token = str(attributes.get("projection", "perspective") or "perspective")
    orthographic = projection_token == "orthographic"
    values: list[UsdAttributeValue] = []
    for name in (*_PROBED_APERTURE_ATTRIBUTES, "clippingRange"):
        if name not in attributes:
            continue
        probe_class = probe_class_for_attribute(name, projection_token)
        raw = attributes[name]
        if name == "clippingRange":
            value: Any = tuple(float(item) for item in raw)
        else:
            value = float(raw)
        values.append(
            UsdAttributeValue(
                name,
                value,
                SUPPORTED_USD_ATTRIBUTES[name],
                _blender_property_path(name, orthographic=orthographic),
                {
                    "probe_class": probe_class,
                    "projection": projection_token,
                    "source": VIEWPORT_CAMERA_PROJECTION_SOURCE,
                },
            )
        )
    return tuple(values)


def _blender_property_path(attribute: str, *, orthographic: bool) -> str:
    if attribute == "clippingRange":
        return "data.clip_start,data.clip_end"
    if attribute == "focalLength":
        return "data.lens"
    if orthographic:
        return "data.ortho_scale"
    return "data.sensor_width" if attribute == "horizontalAperture" else "data.sensor_height"


class CameraValueProbe:
    """Per-session live-honor probe state, one status per probe class.

    Pure data (no threading, no RPCs): the render loop drives the probe and
    is the only mutator once the session runs; reads from other threads are
    advisory diagnostics, so :meth:`diagnostics` copies. State persists
    across background session replacements — "per session" is the viewport
    render session (the loop lifetime), so an unhonored class does not
    re-probe after its own replacement resync.
    """

    def __init__(self) -> None:
        self._status: dict[str, str] = {
            probe_class: PROBE_STATUS_UNKNOWN
            for probe_class in CAMERA_VALUE_PROBE_CLASSES
        }
        self._attempts: dict[str, int] = {
            probe_class: 0 for probe_class in CAMERA_VALUE_PROBE_CLASSES
        }
        self._inconclusive: dict[str, str] = {}
        self._evidence: dict[str, dict[str, Any]] = {}

    def status(self, probe_class: str) -> str:
        return self._status.get(str(probe_class), PROBE_STATUS_UNKNOWN)

    def value_route_classes(self) -> tuple[str, ...]:
        """Classes currently routed as live value updates.

        Unknown (not yet probed) and honored classes stay on the value
        route and out of session identity; unhonored classes are excluded,
        which folds their values back into the composition digest so
        ``reuse_decision`` forces replacement.
        """

        return tuple(
            probe_class
            for probe_class in CAMERA_VALUE_PROBE_CLASSES
            if self._status.get(probe_class) != PROBE_STATUS_UNHONORED
        )

    def begin_attempt(self, probe_class: str) -> int:
        probe_class = str(probe_class)
        self._attempts[probe_class] = self._attempts.get(probe_class, 0) + 1
        return self._attempts[probe_class]

    def record_inconclusive(self, probe_class: str, reason: str) -> None:
        """The probe could not conclude; the class retries on its next edit."""

        self._inconclusive[str(probe_class)] = str(reason)

    def record_result(
        self,
        probe_class: str,
        *,
        honored: bool,
        evidence: Mapping[str, Any] | None = None,
    ) -> str:
        probe_class = str(probe_class)
        status = PROBE_STATUS_HONORED if honored else PROBE_STATUS_UNHONORED
        self._status[probe_class] = status
        self._inconclusive.pop(probe_class, None)
        record = dict(evidence or {})
        record["probe_class"] = probe_class
        record["status"] = status
        record.setdefault("concluded_time_ns", time.time_ns())
        record["attempts"] = self._attempts.get(probe_class, 0)
        self._evidence[probe_class] = record
        return status

    def unhonored_findings(self) -> tuple[dict[str, Any], ...]:
        """One findings-shaped record per unhonored class (task04-05).

        The runtime measurements entry itself is written from a real-worker
        probe run (runtime host); fake-client outcomes only shape the
        record so the escalation hook is testable.
        """

        findings: list[dict[str, Any]] = []
        for probe_class in CAMERA_VALUE_PROBE_CLASSES:
            if self._status.get(probe_class) != PROBE_STATUS_UNHONORED:
                continue
            findings.append(
                {
                    "id": f"camera-{probe_class}-values-not-honored-live",
                    "status": "open",
                    "owner": FINDINGS_OWNER,
                    "probe_class": probe_class,
                    "evidence": dict(self._evidence.get(probe_class, {})),
                    "ask": (
                        "Document or expose live camera-projection updates for "
                        f"the {probe_class!r} camera value class: the WorldState "
                        "write was accepted but the rendered output did not "
                        "change at equal samples and an unchanged camera pose."
                    ),
                }
            )
        return tuple(findings)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "classes": {
                probe_class: {
                    "status": self._status.get(probe_class, PROBE_STATUS_UNKNOWN),
                    "attempts": self._attempts.get(probe_class, 0),
                    "last_inconclusive_reason": self._inconclusive.get(probe_class, ""),
                    "evidence": dict(self._evidence.get(probe_class, {})),
                }
                for probe_class in CAMERA_VALUE_PROBE_CLASSES
            },
            "value_route_classes": list(self.value_route_classes()),
            "unhonored_classes": [
                probe_class
                for probe_class in CAMERA_VALUE_PROBE_CLASSES
                if self._status.get(probe_class) == PROBE_STATUS_UNHONORED
            ],
            "unhonored_findings": [dict(item) for item in self.unhonored_findings()],
        }


__all__ = [
    "CAMERA_VALUE_PROBE_CLASSES",
    "CAMERA_APERTURE_OFFSETS_COMPOSED",
    "CAMERA_DOF_COMPOSED",
    "CAMERA_PROJECTION_KIND_CHANGED",
    "FINDINGS_OWNER",
    "PROBE_CLASS_CLIP",
    "PROBE_CLASS_ORTHO",
    "PROBE_CLASS_PROJECTION",
    "PROBE_INCONCLUSIVE_CONCURRENT_EDITS",
    "PROBE_INCONCLUSIVE_NO_BASELINE",
    "PROBE_INCONCLUSIVE_PHYSICS_ACTIVE",
    "PROBE_INCONCLUSIVE_POSE_CHANGED",
    "PROBE_INCONCLUSIVE_SAMPLE_MISMATCH",
    "PROBE_STATUS_HONORED",
    "PROBE_STATUS_UNHONORED",
    "PROBE_STATUS_UNKNOWN",
    "SUPPORTED_USD_ATTRIBUTES",
    "VIEWPORT_CAMERA_PROJECTION_SOURCE",
    "CameraValueProbe",
    "classify_field",
    "probe_class_for_attribute",
    "usd_attribute_values",
]

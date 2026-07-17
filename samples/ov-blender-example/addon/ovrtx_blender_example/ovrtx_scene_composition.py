# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OVRTX-only camera, render-product, and material presentation composition."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from collections.abc import Iterator
from typing import Any, Iterable, Mapping

from . import camera_value_conversion
from . import usd_paths as usd_paths
from . import render_requests
from .properties import (
    DEFAULT_RENDER_PRODUCT_PATH,
    DLSS_DISABLED_EXECMODE,
    DLSS_EXECMODE_ATTRIBUTE,
    RTPT_RENDER_SETTINGS,
)


# USD attribute type token per RTPT dtype: USD ``int`` is 32-bit (satisfies the
# ``int32`` contract), ``bool`` maps straight through.
_RTPT_USDA_TYPES = {"int32": "int", "bool": "bool"}


def _rtpt_render_product_lines(rtpt_quality: Mapping[str, Any] | None) -> list[str]:
    """Author the four RTPT quality attributes onto the render product.

    Iterates ``RTPT_RENDER_SETTINGS`` (task01-01's single source of truth) so
    the attribute names, dtypes, and documented defaults have one definition.
    Every attribute is authored on every composition regardless of whether it
    differs from its default (spec always-author determinism decision); a value
    absent from ``rtpt_quality`` falls back to the documented default.
    """

    quality = rtpt_quality or {}
    lines: list[str] = []
    for name, spec in RTPT_RENDER_SETTINGS.items():
        usda_type = _RTPT_USDA_TYPES.get(spec.dtype)
        if usda_type is None:
            raise ValueError(f"unsupported RTPT dtype: {spec.dtype!r}")
        # ``rtpt_quality`` carries artist-facing UI values; author the wire value
        # OVRTX consumes (spec.to_wire applies the Max Bounces +2 camera-ray
        # offset, sub-caps pass through) so this USD channel matches the worker
        # config and live-write channels.
        wire = spec.to_wire(quality.get(name, spec.default))
        literal = ("true" if wire else "false") if spec.dtype == "bool" else str(int(wire))
        lines.append(f"{usda_type} {spec.attribute} = {literal}")
    return lines


def _dlss_render_product_lines(dlss_enabled: bool) -> list[str]:
    """Author the DLSS execMode attribute onto the render product when OFF.

    A real-GPU A/B proved this worker honors ``omni:rtx:post:dlss:execMode`` on
    the RenderProduct at session creation (unlike the ignored omni:rtx:rtpt:*
    family), so the DLSS toggle applies via session re-key with no worker
    restart. ``dlss_enabled=True`` leaves the engine default (no line authored);
    ``False`` authors the Performance-preset execMode value. The worker exposes
    no full DLSS off, so this changes the DLSS execution mode rather than
    disabling DLSS.
    """

    if dlss_enabled:
        return []
    return [f"int {DLSS_EXECMODE_ATTRIBUTE} = {int(DLSS_DISABLED_EXECMODE)}"]


def _rtpt_digest_content(rtpt_quality: Mapping[str, Any] | None) -> dict[str, Any]:
    """Freeze-safe RTPT wire values keyed by attribute for the composition digest.

    Carries the wire values (what a (re)placed session actually authors), so the
    digest tracks the values sent to OVRTX and stays consistent with the authored
    layer body and the other channels.
    """

    quality = rtpt_quality or {}
    content: dict[str, Any] = {}
    for name, spec in RTPT_RENDER_SETTINGS.items():
        content[spec.attribute] = spec.to_wire(quality.get(name, spec.default))
    return content


class _FrozenMapping(Mapping[str, Any]):
    """Hashable immutable mapping used inside composition evidence."""

    __slots__ = ("_items",)

    def __init__(self, values: Mapping[str, Any]) -> None:
        if any(not isinstance(key, str) for key in values):
            raise TypeError("composition evidence mapping keys must be strings")
        object.__setattr__(
            self,
            "_items",
            tuple(sorted((key, _freeze(value)) for key, value in values.items())),
        )

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("frozen composition evidence cannot be changed")

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("frozen composition evidence cannot be changed")

    def __getitem__(self, key: str) -> Any:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(self._items)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(
            sorted(
                (_freeze(item) for item in value),
                key=_frozen_sort_key,
            )
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise TypeError("composition evidence floats must be finite")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        f"unsupported composition evidence value: {type(value).__name__}"
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _frozen_sort_key(value: Any) -> tuple[str, str]:
    return (
        type(value).__name__,
        json.dumps(
            _thaw(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
    )


def _freeze_records(records: Iterable[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    return tuple(_FrozenMapping(record) for record in records)


def normalize_sensor_paths(values: Iterable[str] | str) -> tuple[str, ...]:
    if isinstance(values, str):
        source = (values,)
    elif isinstance(values, (set, frozenset)):
        source = tuple(sorted(values))
    else:
        try:
            source = tuple(values)
        except TypeError as exc:
            raise TypeError("sensor paths must be a string or iterable of strings") from exc
    if any(not isinstance(path, str) for path in source):
        raise TypeError("sensor paths must be strings")
    paths = tuple(dict.fromkeys(path for path in source if path))
    return paths or (DEFAULT_RENDER_PRODUCT_PATH,)


@dataclass(frozen=True)
class OvrtxSceneComposition:
    source_scene_path: str
    composed_scene_path: str
    presentation_layers: tuple[Mapping[str, Any], ...]
    digest: str
    pass_through: bool
    session_layer_identifiers: tuple[str, ...] = ()
    conflict_records: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        for name in ("source_scene_path", "composed_scene_path", "digest"):
            if not isinstance(getattr(self, name), str):
                label = name.replace("_", " ")
                raise TypeError(f"composition {label} must be a string")
        if not isinstance(self.pass_through, bool):
            raise TypeError("composition pass through must be a boolean")
        identifiers = (
            (self.session_layer_identifiers,)
            if isinstance(self.session_layer_identifiers, str)
            else (
                tuple(sorted(self.session_layer_identifiers))
                if isinstance(self.session_layer_identifiers, (set, frozenset))
                else tuple(self.session_layer_identifiers)
            )
        )
        if any(not isinstance(identifier, str) for identifier in identifiers):
            raise TypeError("session layer identifiers must be strings")
        object.__setattr__(self, "presentation_layers", _freeze_records(self.presentation_layers))
        object.__setattr__(self, "session_layer_identifiers", identifiers)
        object.__setattr__(self, "conflict_records", _freeze_records(self.conflict_records))


@dataclass(frozen=True)
class _PresentationContribution:
    source: str
    target_path: str
    layer_body: str
    authored_properties: tuple[tuple[str, str], ...]
    digest_content: Mapping[str, Any]
    diagnostics: Mapping[str, Any]


def compose(
    *,
    source_scene_path: str,
    camera_prim_path: str,
    sensor_paths: Iterable[str],
    width: int,
    height: int,
    camera_projection: render_requests.CameraProjectionState | None,
    material_scene_layer: render_requests.MaterialPresentationLayer | None,
    light_scene_layer: render_requests.MaterialPresentationLayer | None = None,
    generate_scene_presentation: bool = False,
    scene_camera_matrix: tuple[tuple[float, ...], ...] | None = None,
    camera_value_route_classes: Iterable[str] = (),
    rtpt_quality: Mapping[str, Any] | None = None,
    rtpt_value_route: bool = False,
    dlss_enabled: bool = True,
) -> OvrtxSceneComposition:
    """Resolve the exact USD scene path an OVRTX session should open."""

    projection = render_requests.camera_projection_usd_attributes(camera_projection)
    projection_diagnostics = render_requests.camera_projection_diagnostics(camera_projection)
    source_scene_path = str(source_scene_path or "")
    camera_prim_path = str(camera_prim_path or "")
    sensor_paths = normalize_sensor_paths(sensor_paths)
    if material_scene_layer is not None and not isinstance(
        material_scene_layer,
        render_requests.MaterialPresentationLayer,
    ):
        raise TypeError("material scene layer must be a MaterialPresentationLayer or None")
    if light_scene_layer is not None and not isinstance(
        light_scene_layer,
        render_requests.MaterialPresentationLayer,
    ):
        raise TypeError("light scene layer must be a MaterialPresentationLayer or None")
    if not source_scene_path and material_scene_layer is None and light_scene_layer is None:
        return _pass_through_composition(source_scene_path)
    width = max(1, int(width))
    height = max(1, int(height))
    contributions = _presentation_contributions(
        source_scene_path=source_scene_path,
        camera_prim_path=camera_prim_path,
        sensor_paths=sensor_paths,
        width=width,
        height=height,
        projection=projection,
        projection_diagnostics=projection_diagnostics,
        material_scene_layer=material_scene_layer,
        light_scene_layer=light_scene_layer,
        generate_scene_presentation=generate_scene_presentation,
        scene_camera_matrix=scene_camera_matrix,
        camera_value_route_classes=camera_value_route_classes,
        rtpt_quality=rtpt_quality,
        rtpt_value_route=rtpt_value_route,
        dlss_enabled=dlss_enabled,
    )
    if not contributions:
        return _pass_through_composition(source_scene_path)

    source_path = Path(source_scene_path).expanduser().resolve()
    conflict_records = _overlay_conflict_records(contributions)
    if conflict_records:
        rejected_records = [
            _layer_record(layer, path="", generated=False, status="conflict_rejected")
            for layer in contributions
        ]
        return _pass_through_composition(
            source_scene_path,
            presentation_layers=tuple(rejected_records),
            conflict_records=tuple(conflict_records),
        )

    digest = _composition_digest(source_scene_path, contributions)
    # ``.resolve()`` is load-bearing on Windows: the POSIX-style default
    # expands to a drive-relative path (``\tmp\...``) whose sublayer
    # reference USD cannot resolve from the composed layer's ``file:///``
    # context — the presentation layer is then silently dropped and the
    # render product never composes.
    work_dir = Path(
        os.environ.get("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR")
        or Path(tempfile.gettempdir()) / "ov-blender-example" / "temporary-usd-layers"
    ).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    override_path = work_dir / f"ovrtx-scene-{digest}.usda"
    presentation_paths = tuple(
        work_dir / f"ovrtx-scene-{digest}-{index}-{layer.source}.usda"
        for index, layer in enumerate(contributions, start=1)
    )
    for path, layer in zip(presentation_paths, contributions, strict=True):
        _write_text_if_changed(
            path,
            "#usda 1.0\n\n" + layer.layer_body.rstrip() + "\n",
        )
    presentation_layers = [
        _layer_record(layer, path=str(path), generated=True)
        for path, layer in zip(presentation_paths, contributions, strict=True)
    ]
    _write_text_if_changed(
        override_path,
        _sublayer_composition_text((*presentation_paths, source_path)),
    )
    return OvrtxSceneComposition(
        source_scene_path=source_scene_path,
        composed_scene_path=str(override_path),
        presentation_layers=_freeze_records(presentation_layers),
        digest=digest,
        pass_through=False,
        session_layer_identifiers=(
            str(override_path),
            *(str(path) for path in presentation_paths),
        ),
    )


def diagnostics(
    composition: OvrtxSceneComposition | None,
    *,
    request: render_requests.RenderRequest | None = None,
) -> dict[str, Any]:
    """Return presentation composition diagnostics for artifacts."""

    if composition is None:
        return {"enabled": False}
    layers = [_thaw(record) for record in composition.presentation_layers]
    if request is not None:
        contributions = _contributions_from_request(request)
        _validate_artifact_layers(composition, request, contributions)
        layers = _artifact_layer_records(
            layers,
            contributions,
            composition_digest=composition.digest,
        )
    conflict_records = [_thaw(record) for record in composition.conflict_records]
    return {
        "enabled": True,
        "source_scene_path": composition.source_scene_path,
        "composed_scene_path": composition.composed_scene_path,
        "pass_through": composition.pass_through,
        "presentation_layer_count": len(layers),
        "presentation_sources": [str(record.get("source", "")) for record in layers],
        "presentation_paths": [str(record.get("path", "")) for record in layers],
        "composition_digest": composition.digest,
        "session_layer_identifiers": list(composition.session_layer_identifiers),
        "presentation_layers": layers,
        "conflict_count": len(conflict_records),
        "conflict_records": conflict_records,
    }


def _pass_through_composition(
    source_scene_path: str,
    *,
    presentation_layers: tuple[dict[str, Any], ...] = (),
    conflict_records: tuple[dict[str, Any], ...] = (),
) -> OvrtxSceneComposition:
    return OvrtxSceneComposition(
        source_scene_path=source_scene_path,
        composed_scene_path=source_scene_path,
        presentation_layers=_freeze_records(presentation_layers),
        digest=_pass_through_digest(source_scene_path),
        pass_through=True,
        conflict_records=_freeze_records(conflict_records),
    )


def _camera_override_layer_text(
    source_path: Path,
    camera_prim_path: str,
    render_product_path: str,
    *,
    width: int,
    height: int,
    focal_length: float,
    horizontal_aperture: float,
    vertical_aperture: float,
) -> str:
    return _composed_layer_text(
        source_path,
        [
            "\n".join(
                _camera_override_body_lines(
                    camera_prim_path,
                    (render_product_path,),
                    width=width,
                    height=height,
                    projection={
                        "focalLength": focal_length,
                        "horizontalAperture": horizontal_aperture,
                        "verticalAperture": vertical_aperture,
                    },
                    # Direct-USD override layers author overs only for prims
                    # the source scene already defines; a def here would
                    # re-type the source prim.
                    define_render_products=False,
                )
            ).rstrip()
        ],
    )


def _presentation_contributions(
    *,
    source_scene_path: str,
    camera_prim_path: str,
    sensor_paths: tuple[str, ...],
    width: int,
    height: int,
    projection: Mapping[str, Any],
    projection_diagnostics: Mapping[str, Any],
    material_scene_layer: render_requests.MaterialPresentationLayer | None,
    light_scene_layer: render_requests.MaterialPresentationLayer | None = None,
    generate_scene_presentation: bool = False,
    scene_camera_matrix: tuple[tuple[float, ...], ...] | None = None,
    camera_value_route_classes: Iterable[str] = (),
    rtpt_quality: Mapping[str, Any] | None = None,
    rtpt_value_route: bool = False,
    dlss_enabled: bool = True,
) -> list[_PresentationContribution]:
    contributions: list[_PresentationContribution] = []
    if material_scene_layer is not None:
        contributions.append(
            _PresentationContribution(
                source=str(material_scene_layer.digest_content.get("source", "materialx_openpbr")),
                target_path=material_scene_layer.target_path,
                layer_body=material_scene_layer.layer_body,
                authored_properties=material_scene_layer.authored_properties,
                digest_content=material_scene_layer.digest_content,
                diagnostics=material_scene_layer.diagnostics,
            )
        )
    if light_scene_layer is not None:
        contributions.append(
            _PresentationContribution(
                source=str(light_scene_layer.digest_content.get("source", "light_value_policy")),
                target_path=light_scene_layer.target_path,
                layer_body=light_scene_layer.layer_body,
                authored_properties=light_scene_layer.authored_properties,
                digest_content=light_scene_layer.digest_content,
                diagnostics=light_scene_layer.diagnostics,
            )
        )
    if source_scene_path and sensor_paths and camera_prim_path:
        contributions.append(
            _camera_projection_contribution(
                camera_prim_path,
                sensor_paths,
                width=width,
                height=height,
                projection=(
                    projection
                    if projection and usd_paths.known_usd_path(camera_prim_path)
                    else {}
                ),
                projection_diagnostics=projection_diagnostics,
                generate_scene_presentation=generate_scene_presentation,
                scene_camera_matrix=(
                    scene_camera_matrix if generate_scene_presentation else None
                ),
                camera_value_route_classes=camera_value_route_classes,
                rtpt_quality=rtpt_quality,
                rtpt_value_route=rtpt_value_route,
                dlss_enabled=dlss_enabled,
            )
        )
    return contributions


def _contributions_from_request(
    request: render_requests.RenderRequest,
) -> list[_PresentationContribution]:
    return _presentation_contributions(
        source_scene_path=str(request.input_usd_path or ""),
        camera_prim_path=str(request.camera_prim_path or ""),
        sensor_paths=normalize_sensor_paths(request.sensor_paths),
        width=max(1, int(request.width)),
        height=max(1, int(request.height)),
        projection=render_requests.camera_projection_usd_attributes(
            request.camera_projection
        ),
        projection_diagnostics=render_requests.camera_projection_diagnostics(
            request.camera_projection
        ),
        material_scene_layer=request.material_scene_layer,
        light_scene_layer=request.light_scene_layer,
        generate_scene_presentation=request.current_scene_generation,
        scene_camera_matrix=getattr(request, "scene_camera_matrix", None),
        camera_value_route_classes=tuple(
            getattr(request, "camera_value_route_classes", ()) or ()
        ),
        rtpt_quality=getattr(request, "rtpt_quality", None),
        rtpt_value_route=bool(getattr(request, "rtpt_value_route", False)),
        dlss_enabled=bool(getattr(request, "dlss_enabled", True)),
    )


def _camera_projection_contribution(
    camera_prim_path: str,
    sensor_paths: tuple[str, ...],
    *,
    width: int,
    height: int,
    projection: Mapping[str, Any],
    projection_diagnostics: Mapping[str, Any],
    generate_scene_presentation: bool = False,
    scene_camera_matrix: tuple[tuple[float, ...], ...] | None = None,
    camera_value_route_classes: Iterable[str] = (),
    rtpt_quality: Mapping[str, Any] | None = None,
    rtpt_value_route: bool = False,
    dlss_enabled: bool = True,
) -> _PresentationContribution:
    route_classes = tuple(
        sorted({str(item) for item in camera_value_route_classes if str(item)})
    )
    # ``_presentation_contributions`` only builds this contribution for a
    # non-empty camera prim path; the RenderProduct defs reference it via
    # ``rel camera`` on both routes, so it participates in the layer body,
    # the digest, and diagnostics even when no projection is authored.
    authored_camera_prim_path = camera_prim_path
    if generate_scene_presentation:
        if not authored_camera_prim_path:
            raise ValueError("live Blender presentation requires a camera prim path")
        body_lines = _generated_presentation_body_lines(
            authored_camera_prim_path,
            sensor_paths,
            width=width,
            height=height,
            projection=projection,
            camera_matrix=scene_camera_matrix,
            rtpt_quality=rtpt_quality,
            dlss_enabled=dlss_enabled,
        )
    else:
        scene_camera_matrix = None
        body_lines = _camera_override_body_lines(
            authored_camera_prim_path,
            sensor_paths,
            width=width,
            height=height,
            projection=projection,
        )
    layer_body = "\n".join(body_lines).rstrip()
    authored_properties = tuple((authored_camera_prim_path, str(name)) for name in projection)
    authored_properties += tuple((sensor_path, "resolution") for sensor_path in sensor_paths)
    authored_properties += tuple((sensor_path, "orderedVars") for sensor_path in sensor_paths)
    for sensor_path in sensor_paths:
        for render_var in ("LdrColor", "HdrColor"):
            render_var_path = f"{sensor_path.rstrip('/')}/{render_var}"
            authored_properties += ((render_var_path, "sourceName"),)
    # Both routes author the RenderProduct's ``rel camera`` (the generated
    # route defines the product, the override route re-defines it against
    # the declared sensor identity), so record the ownership unconditionally.
    authored_properties += tuple((sensor_path, "camera") for sensor_path in sensor_paths)
    if generate_scene_presentation:
        # RTPT quality attributes are authored on the generated RenderProduct
        # (task01-03) — record them so overlay-conflict detection and artifact
        # layer records account for the render-product ownership.
        authored_properties += tuple(
            (sensor_path, spec.attribute)
            for sensor_path in sensor_paths
            for spec in RTPT_RENDER_SETTINGS.values()
        )
        # DLSS execMode is authored on the generated RenderProduct only when the
        # toggle is OFF (enabled leaves the engine default). Record it for
        # overlay-conflict detection when present.
        if not dlss_enabled:
            authored_properties += tuple(
                (sensor_path, DLSS_EXECMODE_ATTRIBUTE) for sensor_path in sensor_paths
            )
    if scene_camera_matrix is not None:
        authored_properties += (
            (authored_camera_prim_path, "xformOp:transform"),
            (authored_camera_prim_path, "xformOpOrder"),
        )
    # Camera value probe routing (task04-05): attributes owned by a probe
    # class currently on the live value route stay OUT of the composition
    # digest — an honored (or not-yet-probed) camera value edit must not
    # change session identity. Unhonored classes are absent from
    # ``route_classes``, folding their values back in so ``reuse_decision``
    # forces the replacement resync. The layer body always authors the
    # current values so any (re)placed session composes them fresh.
    projection_token = str(projection.get("projection", "perspective") or "perspective")
    digest_projection = {
        name: value
        for name, value in projection.items()
        if camera_value_conversion.probe_class_for_attribute(name, projection_token)
        not in route_classes
    }
    digest_content: dict[str, Any] = {
        "source": "viewport_camera_projection",
        "camera_prim_path": authored_camera_prim_path,
        "sensor_paths": sensor_paths,
        "projection": digest_projection,
        "width": width,
        "height": height,
        "generated_scene_presentation": generate_scene_presentation,
    }
    # The RTPT quality values are session state: fold them into the composition
    # digest on the generated route so a changed value produces a distinct
    # composition and ``reuse_decision`` replaces the session that composes the
    # new opinions (task01-03). Only the generated route authors them, so the
    # fixture/direct-USD route leaves the digest untouched. When the live value
    # route is active (task01-04) the attributes are excluded from the digest
    # instead — a quality change is applied as a runtime attribute write on the
    # render thread, so it must not change session identity; the layer body
    # still authors the current values so any (re)placed session composes them
    # fresh.
    if generate_scene_presentation and not rtpt_value_route:
        digest_content["rtpt_quality"] = _rtpt_digest_content(rtpt_quality)
    # The DLSS toggle is honored on the RenderProduct at SESSION creation (real-
    # GPU A/B, runtime measurements), so fold it into the composition digest on the
    # generated route: a toggle change yields a distinct composition and
    # ``reuse_decision`` re-keys the session that composes the new execMode
    # opinion — applied with no worker restart. Included in both states so
    # toggling either direction re-keys.
    if generate_scene_presentation:
        digest_content["dlss_enabled"] = bool(dlss_enabled)
    if route_classes:
        digest_content["camera_value_route_classes"] = route_classes
    if scene_camera_matrix is not None:
        digest_content["scene_camera_matrix"] = tuple(
            tuple(float(value) for value in row) for row in scene_camera_matrix
        )
    return _PresentationContribution(
        source="viewport_camera_projection",
        target_path=authored_camera_prim_path,
        layer_body=layer_body,
        authored_properties=authored_properties,
        digest_content=digest_content,
        diagnostics={
            "source": "viewport_camera_projection",
            "camera_prim_path": authored_camera_prim_path,
            "sensor_paths": sensor_paths,
            "projection_route": str(
                projection_diagnostics.get("route", render_requests.OVRTX_SCENE_COMPOSITION_ROUTE)
            ),
            "runtime_write_status": str(
                projection_diagnostics.get("runtime_write_status", render_requests.RUNTIME_PROJECTION_UNPROVEN)
            ),
            "projection_attributes": list(projection),
            "camera_value_route_classes": list(route_classes),
            "camera_value_digest_excluded_attributes": [
                name for name in projection if name not in digest_projection
            ],
            "generated_scene_presentation": generate_scene_presentation,
            "scene_camera_pose_authored": scene_camera_matrix is not None,
            "rtpt_value_route": bool(generate_scene_presentation and rtpt_value_route),
            "rtpt_digest_excluded": bool(generate_scene_presentation and rtpt_value_route),
            "dlss_enabled": bool(dlss_enabled),
            "dlss_execmode_authored": bool(
                generate_scene_presentation and not dlss_enabled
            ),
        },
    )


def _overlay_conflict_records(
    contributions: list[_PresentationContribution],
) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str], _PresentationContribution] = {}
    conflicts: list[dict[str, Any]] = []
    for contribution in contributions:
        for target_path, property_name in contribution.authored_properties:
            key = (target_path, property_name)
            previous = seen.get(key)
            if previous is not None:
                conflicts.append(
                    {
                        "status": "rejected",
                        "reason": "same_target_property_conflict",
                        "target_path": target_path,
                        "property": property_name,
                        "sources": [previous.source, contribution.source],
                    }
                )
                continue
            seen[key] = contribution
    return conflicts


def _layer_record(
    layer: _PresentationContribution,
    *,
    path: str,
    generated: bool,
    status: str = "",
) -> dict[str, Any]:
    """Project operational layer identity without artifact-only diagnostics."""

    record = {
        "source": layer.source,
        "target_path": layer.target_path,
        "path": path,
        "generated": bool(generated),
    }
    if status:
        record["status"] = status
    return record


def _artifact_layer_records(
    operational_records: list[dict[str, Any]],
    layers: list[_PresentationContribution],
    *,
    composition_digest: str,
) -> list[dict[str, Any]]:
    if len(operational_records) != len(layers):
        raise ValueError("composition artifact layer count does not match operational state")
    records = []
    for operational, layer in zip(operational_records, layers, strict=True):
        if (
            operational.get("source") != layer.source
            or operational.get("target_path") != layer.target_path
        ):
            raise ValueError("composition artifact layer identity does not match operational state")
        record = dict(operational)
        record.update(_thaw(_freeze(layer.diagnostics)))
        for key in ("source", "target_path", "path", "generated"):
            record[key] = operational[key]
        record["composition_digest"] = composition_digest
        record.setdefault("digest", composition_digest)
        if "status" in operational:
            record["status"] = operational["status"]
        records.append(record)
    return records


def _validate_artifact_layers(
    composition: OvrtxSceneComposition,
    request: render_requests.RenderRequest,
    layers: list[_PresentationContribution],
) -> None:
    source_scene_path = str(request.input_usd_path or "")
    if composition.source_scene_path != source_scene_path:
        raise ValueError("composition artifact source path does not match operational state")
    conflicts = _overlay_conflict_records(layers)
    if conflicts:
        expected_digest = _pass_through_digest(source_scene_path)
        if conflicts != [_thaw(record) for record in composition.conflict_records]:
            raise ValueError("composition artifact conflicts do not match operational state")
    elif layers:
        expected_digest = _composition_digest(source_scene_path, layers)
    else:
        expected_digest = _pass_through_digest(source_scene_path)
    if composition.digest != expected_digest:
        raise ValueError("composition artifact digest does not match operational state")


def _composition_digest(
    source_scene_path: str,
    layers: Iterable[_PresentationContribution],
) -> str:
    return _digest_json(
        {
            "schema": 5,
            "source": str(Path(source_scene_path).expanduser().resolve()),
            "presentation": [dict(layer.digest_content) for layer in layers],
        }
    )


def _pass_through_digest(source_scene_path: str) -> str:
    return _digest_json(
        {"schema": 1, "source_scene_path": source_scene_path, "overlays": []}
    )


def _composed_layer_text(source_path: Path, layer_blocks: list[str]) -> str:
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        "    subLayers = [",
        f"        @{_usda_asset_path(str(source_path))}@",
        "    ]",
        ")",
        "",
    ]
    lines.extend(block for block in layer_blocks if block.strip())
    lines.append("")
    return "\n".join(lines)


def _sublayer_composition_text(paths: Iterable[Path]) -> str:
    values = tuple(paths)
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        "    subLayers = [",
    ]
    for index, path in enumerate(values):
        suffix = "," if index + 1 < len(values) else ""
        lines.append(f"        @{_usda_asset_path(str(path))}@{suffix}")
    lines.extend(["    ]", ")", ""])
    return "\n".join(lines)


def _write_text_if_changed(path: Path, text: str) -> None:
    try:
        existing = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        existing = None
    if existing != text:
        path.write_text(text, encoding="utf-8")


def _camera_override_body_lines(
    camera_prim_path: str,
    sensor_paths: Iterable[str],
    *,
    width: int,
    height: int,
    projection: Mapping[str, Any],
    define_render_products: bool = True,
) -> list[str]:
    """Author the fixture/direct-USD camera + render-product override layer.

    ``define_render_products=True`` (the session-composition contract) defines
    each declared sensor path as a typed ``RenderProduct`` — with typeless-def
    ancestors — so the product composes and stays traversable even when the
    source stage does not author it. ``False`` keeps the legacy pure-override
    shape used by ``_camera_override_layer_text``: overs only for the camera
    and product prims the source scene already defines (only the RenderVars
    are add-on-defined), merged into a single prim tree.
    """

    camera_attr_lines = []
    projection_token = str(projection.get("projection", "") or "")
    if projection_token in {"perspective", "orthographic"}:
        camera_attr_lines.append(f'token projection = "{_usda_string(projection_token)}"')
    for name in (
        "focalLength",
        "horizontalAperture",
        "verticalAperture",
        "horizontalApertureOffset",
        "verticalApertureOffset",
        "fStop",
        "focusDistance",
    ):
        if name in projection:
            camera_attr_lines.append(f"float {name} = {_usda_float(float(projection[name]))}")
    if "clippingRange" in projection:
        clipping_range = tuple(projection["clippingRange"])
        if len(clipping_range) == 2:
            camera_attr_lines.append(
                "float2 clippingRange = "
                f"({_usda_float(float(clipping_range[0]))}, {_usda_float(float(clipping_range[1]))})"
            )
    camera_definitions: list[tuple[str, str | None, list[str]]] = []
    if camera_prim_path:
        camera_definitions.append((camera_prim_path, "", camera_attr_lines))
    render_product_definitions: list[tuple[str, str | None, list[str]]] = []
    for sensor_path in sensor_paths:
        ldr_path = f"{sensor_path.rstrip('/')}/LdrColor"
        hdr_path = f"{sensor_path.rstrip('/')}/HdrColor"
        if define_render_products:
            product_definition: tuple[str, str | None, list[str]] = (
                sensor_path,
                "RenderProduct",
                [
                    *(
                        [f"rel camera = <{camera_prim_path}>"]
                        if camera_prim_path
                        else []
                    ),
                    'token omni:rtx:rendermode = "RealTimePathTracing"',
                    'token omni:rtx:background:source:type = "domeLight"',
                    f"rel orderedVars = [<{ldr_path}>, <{hdr_path}>]",
                    f"uniform int2 resolution = ({max(1, int(width))}, {max(1, int(height))})",
                    "bool omni:rtx:indirectDiffuse:denoiser:enabled = true",
                    "bool omni:rtx:reflections:denoiser:enabled = true",
                    "bool omni:rtx:dlss:frameGeneration = true",
                    "bool omni:rtx:autoExposure:enabled = false",
                    "bool omni:rtx:rt:ecoMode:enabled = false",
                    # No omni:rtx:rtpt:* opinions here: RTPT quality is owned
                    # by the generated route / live value writes (task01-03),
                    # and this route's digest excludes rtpt_quality, so
                    # authoring any rtpt attribute would desync body and
                    # digest. Fixture stages author their own RTPT defaults.
                    "color3f omni:rtx:rt:ambientLight:color = (0, 0, 0)",
                ],
            )
        else:
            product_definition = (
                sensor_path,
                "",
                [
                    f"rel orderedVars = [<{ldr_path}>, <{hdr_path}>]",
                    f"uniform int2 resolution = ({max(1, int(width))}, {max(1, int(height))})",
                ],
            )
        render_product_definitions.append(product_definition)
        render_product_definitions.extend(
            (
                (
                    ldr_path,
                    "RenderVar",
                    [
                        'uniform string sourceName = "LdrColor"',
                    ],
                ),
                (
                    hdr_path,
                    "RenderVar",
                    [
                        'uniform string sourceName = "HdrColor"',
                    ],
                ),
            )
        )
    if not define_render_products:
        # Pure-override shape: one merged tree so a shared root prim between
        # the camera and product paths is emitted exactly once.
        return _usda_typed_def_tree(camera_definitions + render_product_definitions)
    return _usda_typed_def_tree(camera_definitions) + _usda_typed_def_tree(
        render_product_definitions,
        define_untyped_ancestors=True,
    )


def _generated_presentation_body_lines(
    camera_prim_path: str,
    sensor_paths: Iterable[str],
    *,
    width: int,
    height: int,
    projection: Mapping[str, Any],
    camera_matrix: tuple[tuple[float, ...], ...] | None = None,
    rtpt_quality: Mapping[str, Any] | None = None,
    dlss_enabled: bool = True,
) -> list[str]:
    camera_projection = str(projection.get("projection", "perspective") or "perspective")
    camera_attributes = [
        f'token projection = "{_usda_string(camera_projection)}"',
        f"float focalLength = {_usda_float(float(projection.get('focalLength', 35.0)))}",
        "float horizontalAperture = "
        + _usda_float(float(projection.get("horizontalAperture", 36.0))),
        "float verticalAperture = "
        + _usda_float(float(projection.get("verticalAperture", 20.25))),
        "float2 clippingRange = "
        + _usda_float2(projection.get("clippingRange", (0.05, 100.0))),
        f"float fStop = {_usda_float(float(projection.get('fStop', 0.0)))}",
    ]
    # fStop without focusDistance focuses at distance zero and blurs the
    # whole frame (Junk Shop DOF regression, 2026-07-07): author them
    # together or not at all.
    for name in (
        "horizontalApertureOffset",
        "verticalApertureOffset",
        "focusDistance",
    ):
        if name in projection:
            camera_attributes.append(
                f"float {name} = {_usda_float(float(projection[name]))}"
            )
    if camera_matrix is not None:
        camera_attributes.extend(
            (
                f"matrix4d xformOp:transform = {_usda_matrix4d(camera_matrix)}",
                'uniform token[] xformOpOrder = ["xformOp:transform"]',
            )
        )
    definitions: list[tuple[str, str | None, list[str]]] = [
        (camera_prim_path, "Camera", camera_attributes)
    ]
    # The generated presentation is authored for scenes that do not contain
    # the render-product or camera hierarchy at all (live-authored scenes
    # only define /World). A `def` nested under `over` ancestors composes
    # as an undefined-ancestor prim that default stage traversal never
    # reaches — the OVRTX worker then rejects the sensor set with "Render
    # product prim not found", and camera value writes fail with "path or
    # attribute not found in stage" when the camera prim path sits outside
    # /World (e.g. a scene-setting path like /Camera/Camera). Define the
    # camera's and each sensor path's ancestor chains explicitly (typeless
    # defs: they define the prims without overriding any type a source
    # layer may author).
    defined_paths = {camera_prim_path}
    for target_path in (camera_prim_path, *sensor_paths):
        ancestor = ""
        for part in usd_paths.path_parts(target_path)[:-1]:
            ancestor = f"{ancestor}/{part}"
            if ancestor in defined_paths:
                continue
            defined_paths.add(ancestor)
            definitions.append((ancestor, None, []))
    for sensor_path in sensor_paths:
        ldr_path = f"{sensor_path.rstrip('/')}/LdrColor"
        hdr_path = f"{sensor_path.rstrip('/')}/HdrColor"
        definitions.append(
            (
                sensor_path,
                "RenderProduct",
                [
                    f"rel camera = <{camera_prim_path}>",
                    'token omni:rtx:rendermode = "RealTimePathTracing"',
                    'token omni:rtx:background:source:type = "domeLight"',
                    f"rel orderedVars = [<{ldr_path}>, <{hdr_path}>]",
                    f"uniform int2 resolution = ({max(1, int(width))}, {max(1, int(height))})",
                    "bool omni:rtx:indirectDiffuse:denoiser:enabled = true",
                    "bool omni:rtx:reflections:denoiser:enabled = true",
                    "bool omni:rtx:dlss:frameGeneration = true",
                    "bool omni:rtx:autoExposure:enabled = false",
                    "bool omni:rtx:rt:ecoMode:enabled = false",
                    *_rtpt_render_product_lines(rtpt_quality),
                    *_dlss_render_product_lines(dlss_enabled),
                    "color3f omni:rtx:rt:ambientLight:color = (0, 0, 0)",
                ],
            )
        )
        definitions.extend(
            (
                (
                    ldr_path,
                    "RenderVar",
                    [
                        'uniform string sourceName = "LdrColor"',
                    ],
                ),
                (
                    hdr_path,
                    "RenderVar",
                    [
                        'uniform string sourceName = "HdrColor"',
                    ],
                ),
            )
        )
    return _usda_typed_def_tree(definitions)


def _usda_typed_def_tree(
    definitions: Iterable[tuple[str, str | None, list[str]]],
    *,
    define_untyped_ancestors: bool = False,
) -> list[str]:
    """Emit a nested usda prim tree from ``(path, type_name, attributes)``.

    ``type_name`` semantics: a non-empty string emits a typed ``def``; the
    empty string emits an ``over`` (override an existing prim without
    defining it — the fixture-stage presentation contract); ``None`` emits
    a typeless ``def`` (define the prim without authoring a type opinion —
    used for generated sensor-path ancestor scopes). Paths never listed
    stay ``over`` intermediates, unless ``define_untyped_ancestors`` is set,
    in which case an unlisted ancestor of a defined prim is emitted as a
    typeless ``def`` so the defined descendants stay traversable.
    """

    roots: dict[str, dict[str, Any]] = {}
    for path, type_name, attributes in definitions:
        parts = usd_paths.path_parts(path)
        if not parts:
            raise ValueError("generated presentation path must be an absolute USD prim path")
        children = roots
        node: dict[str, Any] | None = None
        for part in parts:
            node = children.setdefault(
                part,
                {
                    "defined": False,
                    "type_name": "",
                    "attributes": (),
                    "children": {},
                },
            )
            children = node["children"]
        if node is None or node["defined"]:
            raise ValueError(f"duplicate generated presentation prim: {path}")
        node["defined"] = True
        node["type_name"] = type_name
        node["attributes"] = tuple(attributes)

    def emit(nodes: Mapping[str, Mapping[str, Any]], depth: int) -> list[str]:
        lines: list[str] = []
        indent = "    " * depth
        for index, (name, node) in enumerate(nodes.items()):
            if index:
                lines.append("")
            type_name = node["type_name"]
            if node["defined"] and type_name:
                specifier = f'def {type_name} "{_usda_string(name)}"'
            elif (node["defined"] and type_name is None) or (
                not node["defined"]
                and define_untyped_ancestors
                and _has_defined_descendant(node)
            ):
                specifier = f'def "{_usda_string(name)}"'
            else:
                specifier = f'over "{_usda_string(name)}"'
            lines.append(indent + specifier)
            lines.append(indent + "{")
            lines.extend(
                "    " + indent + attribute
                for attribute in node["attributes"]
            )
            child_lines = emit(node["children"], depth + 1)
            if node["attributes"] and child_lines:
                lines.append("")
            lines.extend(child_lines)
            lines.append(indent + "}")
        return lines

    return emit(roots, 0)


def _has_defined_descendant(node: Mapping[str, Any]) -> bool:
    return bool(node["defined"]) or any(
        _has_defined_descendant(child) for child in node["children"].values()
    )


def _usda_matrix4d(rows: tuple[tuple[float, ...], ...]) -> str:
    values = tuple(tuple(float(value) for value in row) for row in rows)
    if len(values) != 4 or any(len(row) != 4 for row in values):
        raise ValueError("generated presentation matrix4d requires four rows of four values")
    return (
        "( "
        + ", ".join(
            "(" + ", ".join(_usda_float(value) for value in row) + ")"
            for row in values
        )
        + " )"
    )


def _usda_float2(value: Any) -> str:
    values = tuple(value)
    if len(values) != 2:
        raise ValueError("generated presentation float2 requires two values")
    return f"({_usda_float(float(values[0]))}, {_usda_float(float(values[1]))})"


def _usda_asset_path(path: str) -> str:
    return path.replace("\\", "\\\\").replace("@", "@@")


def _usda_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _usda_float(value: float) -> str:
    return f"{float(value):.9g}"


def _digest_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()[:16]

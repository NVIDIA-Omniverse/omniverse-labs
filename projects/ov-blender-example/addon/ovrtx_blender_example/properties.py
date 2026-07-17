# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render settings for the OVRTX Blender example."""

from typing import Any, NamedTuple

from .color_presentation import (
    MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH,
    MODE_SCENE_LINEAR_HDR,
)

try:
    import bpy  # type: ignore
    from bpy.props import (  # type: ignore
        BoolProperty,
        EnumProperty,
        IntProperty,
        PointerProperty,
        StringProperty,
    )
except ModuleNotFoundError:
    bpy = None  # type: ignore[assignment]
    BoolProperty = None  # type: ignore[assignment]
    EnumProperty = None  # type: ignore[assignment]
    IntProperty = None  # type: ignore[assignment]
    PointerProperty = None  # type: ignore[assignment]
    StringProperty = None  # type: ignore[assignment]


BLENDER_AVAILABLE = bpy is not None
DEFAULT_RENDER_PRODUCT_PATH = "/Render/OmniverseKit/HydraTextures/ViewportTexture0"

# RTPT bounce-slider ranges. The UI presents artist-facing Cycles-like
# semantics: the Max Bounces slider counts indirect bounces (UI 0 = direct
# lighting only, UI 1 = one indirect bounce), and the add-on adds a fixed +2
# offset (``RTPT_MAX_BOUNCES_CAMERA_RAY_OFFSET``) to produce the wire value sent
# to OVRTX. A real-GPU A/B on this worker's launch-config channel proved that
# /rtx/rtpt/maxBounces counts the primary camera ray, so wire 0 and 1 both
# render a fully black frame while wire 2 is the first value that produces
# direct lighting (runtime measurements, "RTPT maxBounces semantics"). Mapping UI
# ``n`` -> wire ``n + 2`` therefore makes UI 0 = direct lighting, UI 1 = one
# indirect bounce (the worker's own default), matching Blender/Cycles.
#
# The two sub-caps (specular/transmission, volume) are NOT camera-ray-counted: a
# follow-up A/B (runtime measurements, sub-cap evidence) showed that setting either
# cap to 0 while maxBounces stays healthy leaves the frame fully lit (luma ~74,
# not black), only trimming the specular/transmission contribution. They are
# plain 0-based sub-budgets, so their UI value equals the wire value (offset 0).
RTPT_BOUNCE_MIN = 0
RTPT_BOUNCE_MAX = 128
# Soft minimum reverts to 0 (the yesterday-only soft_min=2 anti-black guard is
# superseded: with the +2 remap a casual drag to UI 0 lands on direct lighting,
# never a black frame).
RTPT_BOUNCE_SOFT_MIN = 0
RTPT_BOUNCE_SOFT_MAX = 32
# Fixed offset added to the Max Bounces UI value to reach the OVRTX wire value.
RTPT_MAX_BOUNCES_CAMERA_RAY_OFFSET = 2
# Max Bounces UI hard maximum, chosen so the +2 wire value stays within the
# documented 0-128 range (126 + 2 = 128).
RTPT_MAX_BOUNCES_UI_MAX = RTPT_BOUNCE_MAX - RTPT_MAX_BOUNCES_CAMERA_RAY_OFFSET
RTPT_MAX_BOUNCES_SOFT_MAX = 30

# --- DLSS Super-Resolution ---------------------------------------------------
# A real-GPU A/B on this RealTimePathTracing worker build (runtime measurements,
# "OVRTX RealTimePathTracing worker always runs DLSS ...") proved DLSS-SR is ON
# by default and that NO exposed setting fully disables it: /rtx/post/aa/op,
# /rtx/minimal/dlss/mode, /rtx/pathtracing/dlss/enabled and
# /rtx/post/scaling/staticRatio are all inert in RTPT. The ONLY honored DLSS
# knob is /rtx/post/dlss/execMode (the quality/performance preset). That knob is
# honored two ways: (1) as the carb setting in the worker-startup
# ovrtx.config.json (read once at launch) and (2) — unlike the omni:rtx:rtpt:*
# family, which this build ignores on the RenderProduct — as
# omni:rtx:post:dlss:execMode authored on the generated RenderProduct, honored at
# SESSION creation. Authoring it on the render product and folding it into the
# composition digest means a toggle change re-keys the session and applies with
# NO worker restart. The toggle therefore switches the DLSS execution mode
# (engine default vs the Performance preset execMode=0); it does not remove DLSS
# (a worker limitation, not an add-on choice).
DLSS_EXECMODE_ATTRIBUTE = "omni:rtx:post:dlss:execMode"
#: execMode value authored when the DLSS toggle is OFF. execMode 0 selects the
#: DLSS Performance preset — the strongest exposed change; this worker exposes
#: no true off, so unchecking changes the DLSS execution mode rather than
#: disabling DLSS.
DLSS_DISABLED_EXECMODE = 0


# UI-selectable color presentation modes. Identifiers are the mode constants
# from ``color_presentation`` so the enum, env override, and classification all
# agree. ``ocio_baked_display`` is intentionally absent: it stays an env-var-only
# reserved v2 seam and is never listed in the UI. The stored preference is
# always selectable; when HdrColor bindings are missing the classification
# resolves scene-linear fail-closed to LDR at request time.
COLOR_PRESENTATION_ENUM_ITEMS = (
    (
        MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH,
        "LDR Display Passthrough",
        "Present the render's display-encoded LDR color unchanged (default)",
    ),
    (
        MODE_SCENE_LINEAR_HDR,
        "Scene-Linear (Blender Color Management)",
        "Read back scene-linear HDR color and apply Blender's color management",
    ),
)
COLOR_PRESENTATION_DEFAULT = MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH


class RtptSettingSpec(NamedTuple):
    """Documented runtime contract for one RTPT quality attribute.

    ``attribute`` is the exact ``RenderProduct`` attribute name, ``dtype`` is
    the USD/runtime type string (``"int32"`` for bounces, ``"bool"`` for the
    firefly filter), and ``default`` is the artist-facing (UI) default the
    matching scene property ships with. ``offset`` is the fixed amount added to
    the artist-facing UI value to reach the value sent to OVRTX (the "wire"
    value); it is non-zero only for the camera-ray-counted Max Bounces control.

    This spec is the single conversion authority: ``to_wire`` (UI -> wire) and
    ``from_wire`` (wire -> UI) are the ONE place the UI/wire mapping is defined,
    so every channel that authors a value (worker config, USD composition, the
    live runtime write) sends identical wire values, and any path that recovers
    the artist value from a wire value (the live-write re-key fallback) inverts
    the same mapping. ``wire_default`` is the resulting documented runtime
    default (3 / 3 / 15 / true) so out-of-the-box output is unchanged.
    """

    attribute: str
    dtype: str
    default: Any
    offset: int = 0

    def to_wire(self, ui_value: Any) -> Any:
        """The value sent to OVRTX for an artist-facing UI value."""

        if self.dtype == "bool":
            return bool(ui_value)
        return int(ui_value) + self.offset

    def from_wire(self, wire_value: Any) -> Any:
        """The artist-facing UI value for a wire value (inverse of ``to_wire``)."""

        if self.dtype == "bool":
            return bool(wire_value)
        return int(wire_value) - self.offset

    @property
    def wire_default(self) -> Any:
        """Documented runtime default (the wire value of the UI default)."""

        return self.to_wire(self.default)


# Single source of truth for the RTPT quality controls: scene property name ->
# (attribute name, dtype string, UI default, ui->wire offset). Defined outside
# the ``bpy`` guard so non-Blender tests, composition authoring (task01-03),
# worker-config authoring, and live-change application (task01-04) all import
# the same mapping and the same UI/wire conversion.
RTPT_RENDER_SETTINGS: dict[str, RtptSettingSpec] = {
    # Camera-ray-counted: UI 0 = direct lighting only (wire 2), UI 1 = one
    # indirect bounce (wire 3, the worker default). UI default 1 -> wire 3.
    "rtpt_max_bounces": RtptSettingSpec(
        "omni:rtx:rtpt:maxBounces", "int32", 1, RTPT_MAX_BOUNCES_CAMERA_RAY_OFFSET
    ),
    # 0-based sub-budgets: UI value == wire value (offset 0).
    "rtpt_max_specular_and_transmission_bounces": RtptSettingSpec(
        "omni:rtx:rtpt:maxSpecularAndTransmissionBounces", "int32", 3
    ),
    "rtpt_max_volume_bounces": RtptSettingSpec(
        "omni:rtx:rtpt:maxVolumeBounces", "int32", 15
    ),
    "rtpt_firefly_filter_enabled": RtptSettingSpec(
        "omni:rtx:rtpt:fireflyFilter:enabled", "bool", True
    ),
}


def _require_blender() -> Any:
    if bpy is None:
        raise RuntimeError("ovrtx_blender_example properties require Blender's bpy module")
    return bpy


def _tag_view3d_redraw(_self: Any, context: Any) -> None:
    screen = getattr(context, "screen", None)
    for area in getattr(screen, "areas", ()):
        if getattr(area, "type", "") == "VIEW_3D":
            area.tag_redraw()


def _rtpt_setting_update(property_name: str):
    """Update callback: redraw, and route a live change to the render thread.

    A quality change on a running session is applied as a runtime attribute
    write on the active render product, executed on the session-owning
    render thread (task01-04). With no active session the redraw alone
    stands; the value reaches the next session through composition authoring
    (task01-03). Never raises out of the RNA update callback.
    """

    def _update(self: Any, context: Any) -> None:
        _tag_view3d_redraw(self, context)
        try:
            from . import rtpt_live_change

            rtpt_live_change.dispatch_render_setting_change(
                property_name, getattr(self, property_name)
            )
        except Exception:
            pass

    return _update


if bpy is not None:

    class OvrtxExampleRenderSettings(bpy.types.PropertyGroup):  # type: ignore[misc]
        render_product_path: StringProperty(  # type: ignore[valid-type]
            name="Render Product",
            description=(
                "USD render product prim path for the direct-USD validation "
                "route only; the live scene route generates its own render "
                "product and ignores this value"
            ),
            default=DEFAULT_RENDER_PRODUCT_PATH,
            options={"HIDDEN"},
        )

        min_samples: IntProperty(  # type: ignore[valid-type]
            name="Minimum Samples",
            description="First sample count used when a viewport preview starts refining",
            default=1,
            min=1,
            max=4096,
            update=_tag_view3d_redraw,
        )

        max_samples: IntProperty(  # type: ignore[valid-type]
            name="Maximum Samples",
            description="Highest sample count requested for an OVRTX render",
            default=128,
            min=1,
            max=4096,
            update=_tag_view3d_redraw,
        )

        rtpt_max_bounces: IntProperty(  # type: ignore[valid-type]
            name="Max Bounces",
            description=(
                "Maximum number of indirect light bounces "
                "(omni:rtx:rtpt:maxBounces), Cycles-like: 0 = direct lighting "
                "only, 1 (default) = one indirect bounce, higher = more "
                "indirect light. Applied when the OVRTX worker launches -- use "
                "Restart OVRTX Session for a change to take effect"
            ),
            default=RTPT_RENDER_SETTINGS["rtpt_max_bounces"].default,
            min=RTPT_BOUNCE_MIN,
            max=RTPT_MAX_BOUNCES_UI_MAX,
            soft_min=RTPT_BOUNCE_SOFT_MIN,
            soft_max=RTPT_MAX_BOUNCES_SOFT_MAX,
            update=_rtpt_setting_update("rtpt_max_bounces"),
        )

        rtpt_max_specular_and_transmission_bounces: IntProperty(  # type: ignore[valid-type]
            name="Max Specular and Transmission Bounces",
            description=(
                "Maximum specular and transmission bounces "
                "(omni:rtx:rtpt:maxSpecularAndTransmissionBounces). A 0-based "
                "sub-budget: 0 removes only the specular/transmission "
                "contribution (reflections and refractions), leaving direct and "
                "diffuse-bounce lighting intact. Applied when the OVRTX worker "
                "launches -- use Restart OVRTX Session to apply a change"
            ),
            default=RTPT_RENDER_SETTINGS[
                "rtpt_max_specular_and_transmission_bounces"
            ].default,
            min=RTPT_BOUNCE_MIN,
            max=RTPT_BOUNCE_MAX,
            soft_min=RTPT_BOUNCE_SOFT_MIN,
            soft_max=RTPT_BOUNCE_SOFT_MAX,
            update=_rtpt_setting_update(
                "rtpt_max_specular_and_transmission_bounces"
            ),
        )

        rtpt_max_volume_bounces: IntProperty(  # type: ignore[valid-type]
            name="Max Volume Bounces",
            description=(
                "Maximum volume scattering bounces "
                "(omni:rtx:rtpt:maxVolumeBounces). A 0-based sub-budget: 0 "
                "removes only volumetric scattering, leaving surface lighting "
                "intact. Applied when the OVRTX worker launches -- use Restart "
                "OVRTX Session to apply a change"
            ),
            default=RTPT_RENDER_SETTINGS["rtpt_max_volume_bounces"].default,
            min=RTPT_BOUNCE_MIN,
            max=RTPT_BOUNCE_MAX,
            soft_min=RTPT_BOUNCE_SOFT_MIN,
            soft_max=RTPT_BOUNCE_SOFT_MAX,
            update=_rtpt_setting_update("rtpt_max_volume_bounces"),
        )

        rtpt_firefly_filter_enabled: BoolProperty(  # type: ignore[valid-type]
            name="Firefly Filter",
            description=(
                "RTPT firefly filter (omni:rtx:rtpt:fireflyFilter:enabled). "
                "Applied when the OVRTX worker launches -- use Restart OVRTX "
                "Session for a change to take effect"
            ),
            default=RTPT_RENDER_SETTINGS["rtpt_firefly_filter_enabled"].default,
            update=_rtpt_setting_update("rtpt_firefly_filter_enabled"),
        )

        dlss_enabled: BoolProperty(  # type: ignore[valid-type]
            name="DLSS Super-Resolution",
            description=(
                "NVIDIA DLSS Super-Resolution (omni:rtx:post:dlss:execMode). On "
                "by default. NOTE: this OVRTX RealTimePathTracing worker build "
                "exposes no full DLSS off -- unchecking selects the DLSS "
                "Performance execution mode (execMode 0) rather than disabling "
                "DLSS. Applied at OVRTX session creation: a change re-keys the "
                "session and takes effect on the next viewport refresh with NO "
                "worker restart (also written to the worker startup config for "
                "freshly launched workers)"
            ),
            default=True,
            update=_tag_view3d_redraw,
        )

        color_presentation_mode: EnumProperty(  # type: ignore[valid-type]
            name="Color Presentation",
            description=(
                "How OVRTX viewport color is presented: LDR display "
                "passthrough (default), or scene-linear HDR readback with "
                "Blender color management. Scene-linear falls back to LDR when "
                "the native client lacks HdrColor readback"
            ),
            items=list(COLOR_PRESENTATION_ENUM_ITEMS),
            default=COLOR_PRESENTATION_DEFAULT,
            update=_tag_view3d_redraw,
        )

        camera_prim_path: StringProperty(  # type: ignore[valid-type]
            name="Preview Camera",
            description=(
                "USD camera prim path for the direct-USD validation route "
                "only; the live scene route mirrors the Blender viewport or "
                "scene camera into its generated camera and ignores this "
                "value"
            ),
            default="",
            options={"HIDDEN"},
            update=_tag_view3d_redraw,
        )

        sync_viewport_camera: BoolProperty(  # type: ignore[valid-type]
            name="Sync Blender View",
            description="Map Blender viewport orbit, pan, and zoom to the ovrtx viewport preview camera",
            default=True,
            update=_tag_view3d_redraw,
        )

        simulation_reset_token: IntProperty(  # type: ignore[valid-type]
            name="Simulation Reset Token",
            description="Internal token incremented when the OVPhysX demo simulation should restart",
            default=0,
            min=0,
            options={"HIDDEN"},
            update=_tag_view3d_redraw,
        )

else:
    OvrtxExampleRenderSettings = None  # type: ignore[assignment]


def register() -> None:
    _bpy = _require_blender()
    if not getattr(OvrtxExampleRenderSettings, "is_registered", False):
        _bpy.utils.register_class(OvrtxExampleRenderSettings)
    if not hasattr(_bpy.types.Scene, "ovrtx_example"):
        _bpy.types.Scene.ovrtx_example = PointerProperty(type=OvrtxExampleRenderSettings)


def unregister() -> None:
    _bpy = _require_blender()
    if hasattr(_bpy.types.Scene, "ovrtx_example"):
        del _bpy.types.Scene.ovrtx_example
    try:
        _bpy.utils.unregister_class(OvrtxExampleRenderSettings)
    except (RuntimeError, ValueError) as exc:
        if "missing bl_rna" not in str(exc) and "not registered" not in str(exc):
            raise


__all__ = [
    "BLENDER_AVAILABLE",
    "COLOR_PRESENTATION_DEFAULT",
    "COLOR_PRESENTATION_ENUM_ITEMS",
    "DEFAULT_RENDER_PRODUCT_PATH",
    "DLSS_DISABLED_EXECMODE",
    "DLSS_EXECMODE_ATTRIBUTE",
    "RTPT_RENDER_SETTINGS",
    "RtptSettingSpec",
    "OvrtxExampleRenderSettings",
    "register",
    "unregister",
]

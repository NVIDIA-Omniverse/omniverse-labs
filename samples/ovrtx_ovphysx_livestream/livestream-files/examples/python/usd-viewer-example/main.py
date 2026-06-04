# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live desktop viewport: ovrtx renders the USD stage; PySide6 displays HdrColor or LdrColor.

A monospace panel on the left shows the root layer USDA/ASCII (or a composed debug dump on demand).
Messages that were previously shown as modal dialogs are appended to a read-only log under the viewport.

How to read this file (for viewers / live walkthroughs)
------------------------------------------------------
- **Bootstrap:** ``parse_args()`` → ``main()`` builds ``QApplication`` and ``ViewerWindow``.
- **Render loop:** a ``QTimer`` (~30 Hz) calls ``_on_frame()`` → ``Renderer.step()`` on the chosen
  **render product** → map ``HdrColor`` or ``LdrColor`` → ``QImage`` → scaled pixmap in the viewport.
- **Loading USD:** ``load_usd()`` → ``open_usd`` (resets stage internally) when switching files;
  ``_apply_render_product_for_current_stage()`` picks which ``RenderProduct`` path steps (CLI override,
  probe order, optional ``--use-stage-dump``).
- **PhysX / samples:** optional ``QProcess`` workers stream JSON poses or run batch sim; the viewer
  writes transforms with ``map_attribute`` (``omni:xform`` / ``omni:fabric:localMatrix``), then the
  same ``_on_frame`` path draws the scene.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer
from PySide6.QtGui import QAction, QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QWidget,
    QVBoxLayout,
)

from ovrtx import Device, PrimMode, Renderer, Semantic

DEFAULT_USD_URL = (
    "https://omniverse-content-production.s3.us-west-2.amazonaws.com/"
    "Samples/Robot-OVRTX/robot-ovrtx.usda"
)

# --- Render products: USD must define a compatible RenderProduct; the viewer steps one path. ---
# clone_example default source; pre-filled for Robot-OVRTX URL in ovrtx → Clone / Play.
_DEFAULT_CLONE_SOURCE_PRIM_SAMPLE = "/World"

# When --render-product is not passed, pick from stage dump (see _render_product_from_stage_dump).
FALLBACK_RENDER_PRODUCT = "/Render/Camera"

_RENDER_PRODUCT_KIT_VIEWPORT = (
    "/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0"
)
_RENDER_PRODUCT_KIT_VT0 = "/Render/OmniverseKit/HydraTextures/ViewportTexture0"

# Default probe order for remote / unknown sources (e.g. Robot-OVRTX URL: /Render/Camera is valid).
_AUTO_PROBE_ORDER_CAMERA_FIRST: tuple[str, ...] = (
    FALLBACK_RENDER_PRODUCT,
    _RENDER_PRODUCT_KIT_VIEWPORT,
    _RENDER_PRODUCT_KIT_VT0,
)

# Kit-style root layers (e.g. three_spheres.usda) often have no /Render/Camera RenderProduct.
# Stepping /Render/Camera first spams sensor errors and probing bogus VT0 confuses the scheduler.
_AUTO_PROBE_ORDER_KIT_FIRST: tuple[str, ...] = (
    _RENDER_PRODUCT_KIT_VIEWPORT,
    FALLBACK_RENDER_PRODUCT,
    _RENDER_PRODUCT_KIT_VT0,
)

# Rigid-body prim paths for ov-libaries-livestream/boxes_falling_on_groundplane.usda (PhysX RIGID_BODY_POSE).
_DEFAULT_BOXES_FALLING_RIGID_PATHS = ",".join(f"/World/Cube{i}" for i in range(1, 12))

# Left panel: cap remote / huge files so the UI stays responsive.
_MAX_SOURCE_PANEL_CHARS = 2_000_000

# Modal prompts: fixed label width so long strings wrap vertically instead of stretching the dialog wide.
_DIALOG_LABEL_WRAP_WIDTH = 460

# --- Qt input helpers (wrap labels; stock QInputDialog would widen the window for long paths) ---


def input_text_wrapped(
    parent: QWidget,
    title: str,
    label: str,
    text: str = "",
    *,
    label_wrap_width: int = _DIALOG_LABEL_WRAP_WIDTH,
) -> tuple[str, bool]:
    """Like ``QInputDialog.getText`` but the label word-wraps to a bounded width (taller dialog, not wider)."""
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    layout = QVBoxLayout(dlg)
    lbl = QLabel(label)
    lbl.setWordWrap(True)
    lbl.setFixedWidth(label_wrap_width)
    layout.addWidget(lbl)
    edit = QLineEdit(text)
    layout.addWidget(edit)
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)
    accepted = dlg.exec() == QDialog.DialogCode.Accepted
    return edit.text(), accepted


def input_int_wrapped(
    parent: QWidget,
    title: str,
    label: str,
    value: int,
    min_value: int,
    max_value: int,
    step: int = 1,
    *,
    label_wrap_width: int = _DIALOG_LABEL_WRAP_WIDTH,
) -> tuple[int, bool]:
    """Like ``QInputDialog.getInt`` but the label word-wraps to a bounded width."""
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    layout = QVBoxLayout(dlg)
    lbl = QLabel(label)
    lbl.setWordWrap(True)
    lbl.setFixedWidth(label_wrap_width)
    layout.addWidget(lbl)
    spin = QSpinBox()
    spin.setRange(min_value, max_value)
    spin.setSingleStep(step)
    spin.setValue(value)
    layout.addWidget(spin)
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)
    accepted = dlg.exec() == QDialog.DialogCode.Accepted
    return spin.value(), accepted


def _mono_font() -> QFont:
    f = QFont("Consolas")
    if not f.exactMatch():
        f = QFont("Courier New")
    f.setStyleHint(QFont.StyleHint.Monospace)
    return f


def _is_likely_usdc_crate(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            head = fh.read(16)
    except OSError:
        return False
    return head.startswith(b"PXR-USDC")


def _read_local_root_layer_text(path_str: str) -> str:
    """Return UTF-8 text for a local root layer, or an explanatory message."""
    try:
        p = Path(path_str)
    except OSError as exc:
        return f"(invalid path: {exc})"
    if not p.is_file():
        return f"(not a regular file: {path_str})"
    if _is_likely_usdc_crate(p):
        return (
            "(This file is binary USDC / crate format. The left panel only shows text for ASCII "
            ".usda or text-compatible .usd files.)\n"
        )
    try:
        raw = p.read_bytes()
    except OSError as exc:
        return f"(could not read file: {exc})"
    if b"\x00" in raw[:4096]:
        return (
            "(File appears to be binary. Open a .usda file to inspect layer text in this panel.)\n"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
    if len(text) > _MAX_SOURCE_PANEL_CHARS:
        return (
            text[:_MAX_SOURCE_PANEL_CHARS]
            + "\n\n… [truncated: increase _MAX_SOURCE_PANEL_CHARS in main.py if needed]\n"
        )
    return text


def _fetch_http_usd_text(url: str) -> str:
    """GET a remote root layer; used for https:// .usda samples."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ovrtx-usd-viewer-example/1.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 — intentional sample fetch
            raw = resp.read(_MAX_SOURCE_PANEL_CHARS + 1)
    except urllib.error.URLError as exc:
        return f"(could not fetch URL: {exc})"
    except TimeoutError as exc:
        return f"(fetch timed out: {exc})"
    if len(raw) > _MAX_SOURCE_PANEL_CHARS:
        return (
            raw[:_MAX_SOURCE_PANEL_CHARS].decode("utf-8", errors="replace")
            + "\n\n… [truncated]\n"
        )
    return raw.decode("utf-8", errors="replace")


def _root_layer_text_for_source(usd_source: str) -> str:
    """Text shown in the left 'source' panel for the current stage identifier."""
    if not usd_source:
        return "(no stage loaded)\n"
    if usd_source.startswith(("https://", "http://")):
        return _fetch_http_usd_text(usd_source)
    if usd_source.startswith("omniverse://"):
        return (
            "(Omniverse URIs cannot be read as plain text here. "
            "Download or sync the layer to a local .usda path, or use an https:// sample URL.)\n"
        )
    return _read_local_root_layer_text(usd_source)


def _default_rigid_paths_hint(usd_source: str) -> str:
    """Pre-fill rigid-body paths in the PhysX dialog when the open file matches known samples."""
    try:
        if Path(usd_source).name.lower() == "boxes_falling_on_groundplane.usda":
            return _DEFAULT_BOXES_FALLING_RIGID_PATHS
    except OSError:
        pass
    return ""


def _default_articulation_root_hint(usd_source: str) -> str:
    """Pre-fill articulation root for live/batch PhysX when the open file matches known samples."""
    try:
        if Path(usd_source).name.lower() == "links_chain_sample.usda":
            return "/World/articulation"
    except OSError:
        pass
    return "/World/articulation"


OVLIBS_DIR = Path(__file__).resolve().parent.parent / "ov-libaries-livestream"

PlayMode = Literal[
    "ovrtx_clone",
    "ovrtx_depth",
    "physx_articulation",
    "physx_tensor",
    "physx_contact",
    "physx_clone_env",
]


def _ovlibs_usd_path(filename: str) -> Optional[Path]:
    """Return ``ov-libaries-livestream/<filename>`` if that file exists."""
    p = OVLIBS_DIR / filename
    return p.resolve() if p.is_file() else None


def _find_robot_with_depth_usda_on_disk() -> Optional[Path]:
    """Viewer-bundled copy, then ``ov-libaries-livestream`` (no open stage required)."""
    for p in (
        Path(__file__).resolve().parent / "robot_with_depth.usda",
        Path(__file__).resolve().parent.parent / "ov-libaries-livestream" / "robot_with_depth.usda",
    ):
        if p.is_file():
            return p.resolve()
    return None


def _default_clone_source_prim_text(usd_source: str) -> str:
    """Pre-fill ovrtx → Play (clone) when the open source is the Robot-OVRTX sample."""
    if not usd_source:
        return ""
    low = usd_source.lower()
    if "robot-ovrtx" in low:
        return _DEFAULT_CLONE_SOURCE_PRIM_SAMPLE
    return ""


def _probe_order_for_usd_source(path: str) -> tuple[str, ...]:
    """Pick probe order from the root layer text when we have a local .usda (no extra deps)."""
    if not path.lower().endswith(".usda"):
        return _AUTO_PROBE_ORDER_CAMERA_FIRST
    try:
        p = Path(path)
        if not p.is_file():
            return _AUTO_PROBE_ORDER_CAMERA_FIRST
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return _AUTO_PROBE_ORDER_CAMERA_FIRST
    # Thin override layers (e.g. robot_with_depth.usda) use `over "omni_kit_widget_viewport_…"`
    # without re-declaring `def RenderProduct`.
    has_kit_vp = (
        'RenderProduct "omni_kit_widget_viewport_ViewportTexture_0"' in text
        or 'over "omni_kit_widget_viewport_ViewportTexture_0"' in text
    )
    has_rp_camera = bool(
        re.search(r'(?m)^\s*def RenderProduct "Camera"\s*[(\{]', text)
    )
    if has_kit_vp and not has_rp_camera:
        return _AUTO_PROBE_ORDER_KIT_FIRST
    return _AUTO_PROBE_ORDER_CAMERA_FIRST


def _hdr_linear_rgba_to_rgba8(rgba: np.ndarray) -> np.ndarray:
    """Match vulkan-interop cuda_kernel linear→sRGB for float H,W,4 (HdrColor)."""
    x = np.clip(rgba.astype(np.float32, copy=False), 0.0, None)
    rgb = np.clip(x[:, :, :3], 0.0, 1.0)
    lo = rgb <= 0.0031308
    srgb = np.where(lo, 12.92 * rgb, 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055)
    srgb = np.clip(srgb, 0.0, 1.0)
    r8 = (srgb * 255.0 + 0.5).astype(np.uint8)
    a = np.clip(x[:, :, 3:4], 0.0, 1.0)
    a8 = (a * 255.0 + 0.5).astype(np.uint8)
    return np.ascontiguousarray(np.concatenate([r8, a8], axis=2))


def _render_product_from_stage_dump(dump: str) -> str:
    """Infer a step()-able RenderProduct path from ovrtx_debug_dump_stage USDA text."""
    # Prefer /Render/Camera when present (e.g. Robot-OVRTX). Lines must not be #-commented.
    if re.search(r'(?m)^\s*def RenderProduct "Camera"\s*(\(|$|\{)', dump):
        return FALLBACK_RENDER_PRODUCT
    if re.search(
        r'(?m)^\s*def RenderProduct "omni_kit_widget_viewport_ViewportTexture_0"\s*(\(|$|\{)',
        dump,
    ):
        return _RENDER_PRODUCT_KIT_VIEWPORT
    if re.search(r'(?m)^\s*def RenderProduct "ViewportTexture0"\s*(\(|$|\{)', dump):
        return _RENDER_PRODUCT_KIT_VT0
    return FALLBACK_RENDER_PRODUCT


def _candidate_render_products_from_dump(dump: str) -> list[str]:
    """Ordered unique paths to try; dump hint first, then common fallbacks."""
    suggested = _render_product_from_stage_dump(dump)
    out: list[str] = []
    for c in (
        suggested,
        FALLBACK_RENDER_PRODUCT,
        _RENDER_PRODUCT_KIT_VIEWPORT,
        _RENDER_PRODUCT_KIT_VT0,
    ):
        if c not in out:
            out.append(c)
    return out


def _pick_color_var_name(render_vars) -> Optional[str]:
    """Prefer HdrColor then LdrColor (same order as examples/c/vulkan-interop)."""
    if "HdrColor" in render_vars:
        return "HdrColor"
    if "LdrColor" in render_vars:
        return "LdrColor"
    return None


def parse_args() -> argparse.Namespace:
    """CLI: optional USD path/URL, forced render product, and optional stage-dump-assisted auto-detect."""
    p = argparse.ArgumentParser(
        description="View an OVRTX-ready USD stage using ovrtx as the renderer."
    )
    p.add_argument(
        "usd",
        nargs="?",
        default=DEFAULT_USD_URL,
        help="USD file path or URL (default: Robot-OVRTX sample)",
    )
    p.add_argument(
        "--render-product",
        "-r",
        default=None,
        metavar="PATH",
        help=(
            "Force this render product path (disables auto-detect). "
            "If omitted, the viewer probes common render products (/Render/Camera, Kit viewport paths)."
        ),
    )
    p.add_argument(
        "--use-stage-dump",
        action="store_true",
        help=(
            "After probes fail, use ovrtx_debug_dump_stage to infer paths (slower; unstable on some builds)."
        ),
    )
    return p.parse_args()


_CLONE_EXAMPLE_MODULE: Any = None


def _get_clone_example_module() -> Any:
    """Load ``ov-libaries-livestream/clone_example.py`` without adding that tree to ``sys.path``."""
    global _CLONE_EXAMPLE_MODULE
    if _CLONE_EXAMPLE_MODULE is not None:
        return _CLONE_EXAMPLE_MODULE
    path = Path(__file__).resolve().parent.parent / "ov-libaries-livestream" / "clone_example.py"
    if not path.is_file():
        raise FileNotFoundError(f"clone_example.py not found: {path}")
    spec = importlib.util.spec_from_file_location("ovrtx_ovlib_clone_example", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _CLONE_EXAMPLE_MODULE = mod
    return mod


class ViewerWindow(QMainWindow):
    """Main window: owns ``Renderer``, the loaded USD handle, Qt widgets, and the frame timer.

    Each timer tick runs ``_on_frame()`` — advance simulation time, optionally apply streamed PhysX
    poses, ``step()`` the active render product, then copy the color buffer into the viewport label.
    """

    def __init__(
        self,
        usd_path: str,
        render_product_override: Optional[str],
        *,
        use_stage_dump: bool = False,
    ) -> None:
        super().__init__()
        self.setWindowTitle("USD Viewer (ovrtx)")
        self.setMinimumSize(1024, 640)
        self.resize(1280, 780)

        # Render product selection, stage identity, last CPU image for resize, one-shot error flags.
        self._render_product_override = render_product_override
        self._use_stage_dump = use_stage_dump
        self._render_product = FALLBACK_RENDER_PRODUCT
        self._render_products = {self._render_product}
        self._usd_source: str = ""
        self._last_qimage: Optional[QImage] = None
        self._step_error_shown = False
        self._no_frames_warned = False

        # Live PhysX child (ovphysx only): JSONL poses on stdout, logs on stderr.
        self._live_physx_process: Optional[QProcess] = None
        self._live_physx_stdout_buffer: str = ""
        self._live_physx_last_payload: Optional[dict[str, Any]] = None

        # ▶ Play action depends on last ovrtx / ovphysx menu choice.
        self._play_mode: PlayMode = "ovrtx_clone"

        # Layout: [ left: root layer text ] | [ top: viewport label, bottom: log ]
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("background-color: #1a1a1a;")

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMinimumHeight(100)
        self._log_view.setMaximumBlockCount(8000)
        self._log_view.setPlaceholderText("Log output (warnings, PhysX, load errors)…")
        mono = _mono_font()
        self._log_view.setFont(mono)
        self._log_view.setStyleSheet(
            "QPlainTextEdit { background-color: #121212; color: #d4d4d4; border: none; }"
        )

        self._source_usd_view = QPlainTextEdit()
        self._source_usd_view.setReadOnly(True)
        self._source_usd_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._source_usd_view.setFont(_mono_font())
        self._source_usd_view.setPlaceholderText(
            "Root layer text (def Xform, prims, payloads) appears here after load…"
        )
        self._source_usd_view.setMinimumWidth(260)
        self._source_usd_view.setStyleSheet(
            "QPlainTextEdit { background-color: #1e1e1e; color: #d4d4d4; border: none; }"
        )

        viewport_column = QWidget()
        viewport_layout = QVBoxLayout(viewport_column)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.setSpacing(0)
        viewport_layout.addWidget(self._label, 1)

        clone_footer = QWidget()
        clone_footer_layout = QHBoxLayout(clone_footer)
        clone_footer_layout.setContentsMargins(8, 4, 8, 4)
        clone_footer_layout.setSpacing(12)
        self._ovrtx_play_button = QPushButton("▶ Play")
        self._refresh_play_tooltip()
        self._ovrtx_play_button.clicked.connect(self._on_play_clicked)
        clone_footer_layout.addWidget(self._ovrtx_play_button, 0)
        play_hint = QLabel(
            "Select an ovrtx/ovphysx scene to play the simulation."
        )
        play_hint.setWordWrap(True)
        play_hint.setStyleSheet("color: #a8a8a8; font-size: 11px;")
        clone_footer_layout.addWidget(play_hint, 1)
        viewport_layout.addWidget(clone_footer, 0)

        inner_splitter = QSplitter(Qt.Orientation.Vertical)
        inner_splitter.addWidget(viewport_column)
        inner_splitter.addWidget(self._log_view)
        inner_splitter.setStretchFactor(0, 1)
        inner_splitter.setStretchFactor(1, 0)
        inner_splitter.setSizes([560, 180])

        outer_splitter = QSplitter(Qt.Orientation.Horizontal)
        outer_splitter.addWidget(self._source_usd_view)
        outer_splitter.addWidget(inner_splitter)
        outer_splitter.setStretchFactor(0, 0)
        outer_splitter.setStretchFactor(1, 1)
        outer_splitter.setSizes([400, 920])
        layout.addWidget(outer_splitter)

        # Menus delegate to methods that spawn subprocesses or mutate the ovrtx stage.
        file_menu = self.menuBar().addMenu("&File")
        open_action = QAction("&Open…", self)
        open_action.triggered.connect(self._pick_file)
        file_menu.addAction(open_action)
        quit_action = QAction("&Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        ovrtx_menu = self.menuBar().addMenu("&ovrtx")
        clone_action = QAction("&Clone", self)
        clone_action.setToolTip(
            "Loads the default Robot-OVRTX sample from the web. Then use ▶ Play below the viewport to run "
            "clone_usd (clone_example: /World → /World_clone). Needs uv or OVRTX_PXR_PYTHON. Reload before re-clone."
        )
        clone_action.triggered.connect(self._open_clone_robot_sample)
        ovrtx_menu.addAction(clone_action)
        depth_png_action = QAction("&Depth map (sensor PNG)…", self)
        depth_png_action.setToolTip(
            "Loads robot_with_depth.usda from this example or ov-libaries-livestream. "
            "Then use ▶ Play below the viewport to run depth_map_example.py (writes depth_map.png beside the .usda)."
        )
        depth_png_action.triggered.connect(self._open_robot_with_depth_sample)
        ovrtx_menu.addAction(depth_png_action)

        ovphysx_menu = self.menuBar().addMenu("&ovphysx")
        physx_articulation_action = QAction("&Articulation", self)
        physx_articulation_action.setToolTip(
            "Loads links_chain_sample.usda. ▶ Play spawns physx_live_worker.py (JSONL poses → viewer; same as the old "
            "Simulation → Live PhysX articulation stream)."
        )
        physx_articulation_action.triggered.connect(self._open_links_chain_sample)
        ovphysx_menu.addAction(physx_articulation_action)
        physx_tensor_action = QAction("&Tensor binding", self)
        physx_tensor_action.setToolTip(
            "Loads links_chain_sample.usda. ▶ Play runs tensor_bindings.py --viewer-stream (JSONL poses + drives)."
        )
        physx_tensor_action.triggered.connect(self._open_links_chain_sample_tensor)
        ovphysx_menu.addAction(physx_tensor_action)
        physx_contact_action = QAction("&Contact binding", self)
        physx_contact_action.setToolTip(
            "Loads boxes_falling_on_groundplane.usda. ▶ Play runs the same as legacy Simulation → Contact Binding: "
            "physx_rigid_live_worker (falling boxes in viewport). Other stages: contact_binding tutorial via subprocess wrapper."
        )
        physx_contact_action.triggered.connect(self._open_boxes_falling_sample)
        ovphysx_menu.addAction(physx_contact_action)
        physx_clone_env_action = QAction("&Clone", self)
        physx_clone_env_action.setToolTip(
            "Loads basic_simulation.usda. ▶ Play runs clone.py --viewer-stream (clone_usd in viewer + PhysX stream)."
        )
        physx_clone_env_action.triggered.connect(self._open_basic_simulation_sample)
        ovphysx_menu.addAction(physx_clone_env_action)

        view_menu = self.menuBar().addMenu("&View")
        reload_layer_action = QAction("Reload &root layer text (left panel)", self)
        reload_layer_action.setToolTip(
            "Re-read the current root from disk or http(s). Use after editing a local .usda on disk."
        )
        reload_layer_action.triggered.connect(self._on_reload_root_layer_panel)
        view_menu.addAction(reload_layer_action)
        dump_panel_action = QAction("Left panel: &composed stage (debug dump)", self)
        dump_panel_action.setToolTip(
            "Replace left panel with ovrtx_debug_dump_stage output (includes runtime clones/overrides; "
            "can be slow; see also --use-stage-dump caveats)."
        )
        dump_panel_action.triggered.connect(self._show_composed_stage_in_source_panel)
        view_menu.addAction(dump_panel_action)
        clear_log_action = QAction("Clear &log", self)
        clear_log_action.triggered.connect(self._log_view.clear)
        view_menu.addAction(clear_log_action)
        clear_physx_action = QAction("&Reload scene (clear PhysX transforms)", self)
        clear_physx_action.setToolTip(
            "Reload the current USD from disk/URL to drop in-memory PhysX / clone overrides."
        )
        clear_physx_action.triggered.connect(self._clear_physx_overlay)
        view_menu.addAction(clear_physx_action)

        self._log("Creating renderer (first run may compile and cache shaders)…", "info")
        self._renderer = Renderer()
        self._log("Renderer ready.", "info")

        # Paths to sibling sample scripts (PhysX workers, clone/depth tutorials — not imported as modules).
        self._depth_map_script = (
            Path(__file__).resolve().parent.parent / "ov-libaries-livestream" / "depth_map_example.py"
        )
        self._physx_script = Path(__file__).resolve().parent / "physx_subprocess_sim.py"
        self._live_physx_script = Path(__file__).resolve().parent / "physx_live_worker.py"
        self._tensor_bindings_live_script = (
            Path(__file__).resolve().parent.parent / "ov-libaries-livestream" / "tensor_bindings.py"
        )
        self._clone_env_script = (
            Path(__file__).resolve().parent.parent / "ov-libaries-livestream" / "clone.py"
        )
        self._rigid_live_script = Path(__file__).resolve().parent / "physx_rigid_live_worker.py"
        self._contact_binding_script = Path(__file__).resolve().parent / "contact_binding_subprocess.py"
        self._physx_process: Optional[QProcess] = None
        self._contact_binding_process: Optional[QProcess] = None
        self._depth_map_process: Optional[QProcess] = None
        self._physx_temp_json_path: Optional[Path] = None

        self.load_usd(usd_path)

        # ~30 FPS: each timeout runs _on_frame() (step + upload color buffer to the QLabel).
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_frame)
        self._timer.start(33)

    def _log(self, message: str, level: str = "info") -> None:
        """Append to the panel below the viewport and mirror to stderr."""
        prefix = level.upper()
        text = f"[{prefix}] {message}"
        print(text, file=sys.stderr)
        self._log_view.appendPlainText(text)
        sb = self._log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _refresh_root_layer_source_panel(self, *, from_load: bool = False) -> None:
        """Fill the left panel with the root layer ASCII (local file or http(s) fetch)."""
        if not self._usd_source:
            self._source_usd_view.setPlainText("(no stage loaded)\n")
            return
        if not from_load:
            self._log("Reloading root layer text in left panel…", "info")
        body = _root_layer_text_for_source(self._usd_source)
        header = f"# Root layer: {self._usd_source}\n#\n\n"
        self._source_usd_view.setPlainText(header + body)
        if not from_load:
            self._log("Left panel updated (root layer text).", "info")

    def _on_reload_root_layer_panel(self) -> None:
        self._refresh_root_layer_source_panel(from_load=False)

    def _show_composed_stage_in_source_panel(self) -> None:
        """Replace left panel with Fabric/composed USDA from ovrtx_debug_dump_stage."""
        if not self._usd_source:
            self._log("Load a stage before requesting a composed stage dump.", "warn")
            return
        self._log("Fetching composed stage for left panel (ovrtx_debug_dump_stage)…", "info")
        try:
            products = self._renderer.step(
                render_products={"ovrtx_debug_dump_stage"},
                delta_time=1.0 / 60.0,
            )
        except Exception as exc:  # noqa: BLE001
            self._log(f"Composed stage dump step failed: {exc}", "error")
            return
        if products is None or "ovrtx_debug_dump_stage" not in products:
            self._log("No ovrtx_debug_dump_stage in step result.", "error")
            return
        frames = list(products["ovrtx_debug_dump_stage"].frames)
        if not frames or "debug" not in frames[0].render_vars:
            self._log("Composed dump missing debug payload.", "error")
            return
        try:
            with frames[0].render_vars["debug"].map(device=Device.CPU) as mapping:
                dump = mapping.tensor.to_bytes().decode("utf-8")
        except Exception as exc:  # noqa: BLE001
            self._log(f"Could not read composed dump: {exc}", "error")
            return
        if len(dump) > _MAX_SOURCE_PANEL_CHARS:
            dump = (
                dump[:_MAX_SOURCE_PANEL_CHARS]
                + "\n\n… [truncated: see _MAX_SOURCE_PANEL_CHARS in main.py]\n"
            )
        hdr = (
            "# Composed stage (ovrtx_debug_dump_stage)\n"
            f"# Current source identifier: {self._usd_source}\n#\n\n"
        )
        self._source_usd_view.setPlainText(hdr + dump)
        self._log("Left panel shows composed stage dump (runtime view of prims / xforms).", "info")

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt API
        self._stop_live_physx(uncheck_menu=False)
        if self._contact_binding_process is not None:
            if self._contact_binding_process.state() != QProcess.ProcessState.NotRunning:
                self._contact_binding_process.kill()
                self._contact_binding_process.waitForFinished(3000)
            self._contact_binding_process = None
        if self._physx_process is not None:
            if self._physx_process.state() != QProcess.ProcessState.NotRunning:
                self._physx_process.kill()
                self._physx_process.waitForFinished(3000)
            self._physx_process = None
        if self._depth_map_process is not None:
            if self._depth_map_process.state() != QProcess.ProcessState.NotRunning:
                self._depth_map_process.kill()
                self._depth_map_process.waitForFinished(3000)
            self._depth_map_process = None
        if self._physx_temp_json_path is not None:
            try:
                self._physx_temp_json_path.unlink(missing_ok=True)
            except OSError:
                pass
            self._physx_temp_json_path = None
        super().closeEvent(event)

    def _stop_live_physx(self, *, uncheck_menu: bool = True) -> None:
        """Terminate live PhysX worker (tensor / clone stream) and clear streamed pose state."""
        del uncheck_menu  # retained for call-site compatibility
        if self._live_physx_process is not None:
            if self._live_physx_process.state() != QProcess.ProcessState.NotRunning:
                self._live_physx_process.kill()
                self._live_physx_process.waitForFinished(3000)
            self._live_physx_process = None
        self._live_physx_stdout_buffer = ""
        self._live_physx_last_payload = None

    def _spawn_physx_live_articulation_stream(self) -> None:
        """▶ Play (articulation): same as legacy Simulation → Live PhysX — ``physx_live_worker.py`` JSONL stream."""
        if self._live_physx_process is not None and self._live_physx_process.state() != QProcess.ProcessState.NotRunning:
            self._log("A live pose stream is already running. Use View → Reload scene to stop it first.", "warn")
            return
        if self._contact_binding_process is not None and self._contact_binding_process.state() != QProcess.ProcessState.NotRunning:
            self._log("Wait for the Contact Binding tutorial to finish, then use ▶ Play (articulation).", "warn")
            return
        if self._physx_process is not None and self._physx_process.state() != QProcess.ProcessState.NotRunning:
            self._log("Wait for the batch PhysX subprocess to finish, then use ▶ Play (articulation).", "warn")
            return
        if not self._usd_source_is_local_file(self._usd_source):
            self._log(
                "Live PhysX needs a local USD file. Use ovphysx → Articulation to load links_chain_sample.usda.",
                "warn",
            )
            return
        if not self._live_physx_script.is_file():
            self._log(f"Missing physx_live_worker.py (expected):\n{self._live_physx_script}", "error")
            return

        articulation_root = _default_articulation_root_hint(self._usd_source).strip() or "/World/articulation"
        usd_abs = str(Path(self._usd_source).resolve())
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        proc.setProcessEnvironment(QProcessEnvironment.systemEnvironment())
        proc.setProgram(sys.executable)
        proc.setArguments(
            [
                str(self._live_physx_script),
                "--usd",
                usd_abs,
                "--articulation-root",
                articulation_root,
                "--dt",
                str(1.0 / 60.0),
            ]
        )

        def _on_error(_error: QProcess.ProcessError) -> None:
            if self._live_physx_process is proc:
                self._log(f"Live PhysX worker error: {proc.errorString()}", "error")

        def _on_finished(exit_code: int, exit_status: QProcess.ExitStatus) -> None:
            if self._live_physx_process is not proc:
                return
            self._live_physx_process = None
            self._live_physx_stdout_buffer = ""
            self._live_physx_last_payload = None
            err_txt = bytes(proc.readAllStandardError()).decode("utf-8", errors="replace").strip()
            out_txt = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace").strip()
            if exit_status != QProcess.ExitStatus.NormalExit or exit_code != 0:
                self._log(
                    f"physx_live_worker exited (code {exit_code}, status {exit_status!r}).\n"
                    f"{err_txt or '(no stderr)'}\n{out_txt or '(no stdout)'}",
                    "warn",
                )

        proc.errorOccurred.connect(_on_error)
        proc.finished.connect(_on_finished)
        self._live_physx_process = proc
        proc.start()
        if not proc.waitForStarted(10_000):
            self._live_physx_process = None
            self._log(f"physx_live_worker failed to start: {proc.errorString()}", "error")
            return

        self._live_physx_stdout_buffer = ""
        self._live_physx_last_payload = None
        self._log(
            f"Live PhysX articulation stream started: articulation_root={articulation_root!r} "
            "(physx_live_worker.py; viewer applies JSONL poses). View → Reload scene to stop.",
            "info",
        )

    def _spawn_tensor_bindings_viewer_stream(self) -> None:
        """▶ Play (tensor): tensor_bindings.py --viewer-stream with default articulation root."""
        if self._live_physx_process is not None and self._live_physx_process.state() != QProcess.ProcessState.NotRunning:
            self._log("A live pose stream is already running. Use View → Reload scene to stop it first.", "warn")
            return
        if self._contact_binding_process is not None and self._contact_binding_process.state() != QProcess.ProcessState.NotRunning:
            self._log("Wait for the Contact Binding tutorial to finish, then use ▶ Play (tensor binding).", "warn")
            return
        if self._physx_process is not None and self._physx_process.state() != QProcess.ProcessState.NotRunning:
            self._log("Wait for the batch PhysX subprocess to finish, then use ▶ Play (tensor binding).", "warn")
            return
        if not self._usd_source_is_local_file(self._usd_source):
            self._log(
                "Tensor binding stream needs a local USD file. Use ovphysx → Tensor binding to load links_chain_sample.usda.",
                "warn",
            )
            return
        if not self._tensor_bindings_live_script.is_file():
            self._log(f"Missing tensor_bindings.py (expected):\n{self._tensor_bindings_live_script}", "error")
            return

        articulation_root = _default_articulation_root_hint(self._usd_source).strip() or "/World/articulation"
        usd_abs = str(Path(self._usd_source).resolve())
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        proc.setProcessEnvironment(QProcessEnvironment.systemEnvironment())
        proc.setProgram(sys.executable)
        proc.setArguments(
            [
                str(self._tensor_bindings_live_script),
                "--viewer-stream",
                "--usd",
                usd_abs,
                "--articulation-root",
                articulation_root,
                "--dt",
                str(1.0 / 60.0),
            ]
        )

        def _on_error(_error: QProcess.ProcessError) -> None:
            if self._live_physx_process is proc:
                self._log(f"Tensor bindings process error: {proc.errorString()}", "error")

        def _on_finished(exit_code: int, exit_status: QProcess.ExitStatus) -> None:
            if self._live_physx_process is not proc:
                return
            self._live_physx_process = None
            self._live_physx_stdout_buffer = ""
            self._live_physx_last_payload = None
            err_txt = bytes(proc.readAllStandardError()).decode("utf-8", errors="replace").strip()
            out_txt = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace").strip()
            if exit_status != QProcess.ExitStatus.NormalExit or exit_code != 0:
                self._log(
                    f"tensor_bindings --viewer-stream exited (code {exit_code}, status {exit_status!r}).\n"
                    f"{err_txt or '(no stderr)'}\n{out_txt or '(no stdout)'}",
                    "warn",
                )

        proc.errorOccurred.connect(_on_error)
        proc.finished.connect(_on_finished)
        self._live_physx_process = proc
        proc.start()
        if not proc.waitForStarted(10_000):
            self._live_physx_process = None
            self._log(f"tensor_bindings failed to start: {proc.errorString()}", "error")
            return

        self._live_physx_stdout_buffer = ""
        self._live_physx_last_payload = None
        self._log(
            f"Tensor bindings stream started: articulation_root={articulation_root!r} "
            "(child ovphysx; viewer applies poses + step). View → Reload scene to stop.",
            "info",
        )

    def _drain_live_physx_stdout(self) -> None:
        """Parse complete JSON lines from the live worker; keep the latest payload for this frame."""
        proc = self._live_physx_process
        if proc is None or proc.state() != QProcess.ProcessState.Running:
            return
        chunk = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if chunk:
            self._live_physx_stdout_buffer += chunk
        parts = self._live_physx_stdout_buffer.split("\n")
        self._live_physx_stdout_buffer = parts[-1]
        for line in parts[:-1]:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("version") == 1:
                self._live_physx_last_payload = payload

        err_chunk = bytes(proc.readAllStandardError()).decode("utf-8", errors="replace")
        if err_chunk.strip():
            for eline in err_chunk.splitlines():
                if eline.strip():
                    self._log(f"[live-pose] {eline.strip()}", "info")

    def _play_clone_env_stream(self) -> None:
        """ovphysx → Clone ▶ Play: clone_usd env0→env1–3 on the RTX stage, then PhysX clone.py --viewer-stream."""
        if self._live_physx_process is not None and self._live_physx_process.state() != QProcess.ProcessState.NotRunning:
            self._log(
                "A live pose stream is already running. Use View → Reload scene to stop it first.",
                "warn",
            )
            return
        if self._contact_binding_process is not None and self._contact_binding_process.state() != QProcess.ProcessState.NotRunning:
            self._log("Wait for the Contact Binding tutorial to finish, then use ▶ Play (ovphysx Clone).", "warn")
            return
        if self._physx_process is not None and self._physx_process.state() != QProcess.ProcessState.NotRunning:
            self._log("Wait for the batch PhysX subprocess to finish, then use ▶ Play (ovphysx Clone).", "warn")
            return
        if not self._usd_source_is_local_file(self._usd_source):
            self._log(
                "ovphysx → Clone needs a local USD file. Use the menu to load basic_simulation.usda, then ▶ Play.",
                "warn",
            )
            return
        if not self._clone_env_script.is_file():
            self._log(f"Missing clone sample (expected):\n{self._clone_env_script}", "error")
            return

        try:
            if Path(self._usd_source).resolve().name.lower() != "basic_simulation.usda":
                self._log(
                    "ovphysx → Clone expects basic_simulation.usda (env0 + PhysX clone). "
                    "Continuing — clone_usd will fail if /World/envs/env0 is missing.",
                    "warn",
                )
        except OSError:
            pass

        try:
            self._renderer.clone_usd(
                "/World/envs/env0",
                ["/World/envs/env1", "/World/envs/env2", "/World/envs/env3"],
            )
        except Exception as exc:  # noqa: BLE001
            self._log(
                f"clone_usd failed: {exc}\n"
                "If /World/envs/env1–env3 already exist, use View → Reload scene, then try again.",
                "error",
            )
            return

        self._schedule_viewport_refresh_after_physx()
        self._log(
            "clone_usd: /World/envs/env0 → env1, env2, env3. Starting PhysX clone.py --viewer-stream…",
            "info",
        )

        usd_abs = str(Path(self._usd_source).resolve())
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        proc.setProcessEnvironment(QProcessEnvironment.systemEnvironment())
        proc.setProgram(sys.executable)
        proc.setArguments(
            [
                str(self._clone_env_script),
                "--viewer-stream",
                "--usd",
                usd_abs,
                "--dt",
                str(1.0 / 60.0),
            ]
        )

        def _on_error(_error: QProcess.ProcessError) -> None:
            if self._live_physx_process is proc:
                self._log(f"Clone stream process error: {proc.errorString()}", "error")

        def _on_finished(exit_code: int, exit_status: QProcess.ExitStatus) -> None:
            if self._live_physx_process is not proc:
                return
            self._live_physx_process = None
            self._live_physx_stdout_buffer = ""
            self._live_physx_last_payload = None
            err_txt = bytes(proc.readAllStandardError()).decode("utf-8", errors="replace").strip()
            out_txt = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace").strip()
            if exit_status != QProcess.ExitStatus.NormalExit or exit_code != 0:
                self._log(
                    f"clone.py --viewer-stream exited (code {exit_code}, status {exit_status!r}).\n"
                    f"{err_txt or '(no stderr)'}\n{out_txt or '(no stdout)'}",
                    "warn",
                )
            else:
                self._log("clone.py --viewer-stream finished.", "info")

        proc.errorOccurred.connect(_on_error)
        proc.finished.connect(_on_finished)
        self._live_physx_process = proc
        proc.start()
        if not proc.waitForStarted(10_000):
            self._live_physx_process = None
            self._log(f"clone.py failed to start: {proc.errorString()}", "error")
            return

        self._live_physx_stdout_buffer = ""
        self._live_physx_last_payload = None
        self._log(
            "ovphysx → Clone: streaming rigid-body poses for /World/envs/env*/table (kill process or Reload to stop).",
            "info",
        )

    def _clear_physx_overlay(self) -> None:
        if not self._usd_source:
            return
        self._log("Reloading USD to clear in-memory PhysX transform overrides…", "info")
        self.load_usd(self._usd_source)

    def _apply_local_matrices_fabric(self, paths: list[str], local_m: np.ndarray) -> str:
        """Write local 4×4 transforms for PhysX poses.

        Prefer ``omni:xform`` (Kit-style runtime overrides on geometry). Some Mesh prims with authored
        ``xformOp`` do not visibly update when only
        ``omni:fabric:localMatrix`` is written. Fall back to ``omni:fabric:localMatrix`` if
        mapping ``omni:xform`` fails for this stage/build.
        """
        if local_m.shape != (len(paths), 4, 4):
            raise ValueError(f"local_m shape {local_m.shape} != ({len(paths)}, 4, 4)")

        last_exc: Optional[BaseException] = None
        for attr_name in ("omni:xform", "omni:fabric:localMatrix"):
            try:
                with self._renderer.map_attribute(
                    prim_paths=paths,
                    attribute_name=attr_name,
                    semantic=Semantic.XFORM_MAT4x4,
                    prim_mode=PrimMode.EXISTING_ONLY,
                    device=Device.CPU,
                ) as mapping:
                    dest = np.from_dlpack(mapping.tensor)
                    if dest.shape != local_m.shape:
                        raise RuntimeError(
                            f"map_attribute tensor shape {dest.shape} != xform buffer {local_m.shape}"
                        )
                    if dest.dtype != local_m.dtype:
                        dest[...] = local_m.astype(dest.dtype, copy=False)
                    else:
                        dest[...] = local_m
                return attr_name
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        raise RuntimeError(
            "Could not map omni:xform or omni:fabric:localMatrix for PhysX poses"
        ) from last_exc

    def _apply_physx_pose_json(self, json_path: Path) -> str:
        """Apply link poses from physx_subprocess_sim JSON (version 1)."""
        raw = json_path.read_text(encoding="utf-8")
        payload: dict[str, Any] = json.loads(raw)
        return self._apply_physx_pose_payload(payload)

    def _apply_physx_pose_payload(self, payload: dict[str, Any]) -> str:
        """Apply version-1 PhysX pose dict (file, subprocess, or live JSONL line).

        PhysX exports **world** 4x4 row-vector matrices. We convert to **local** transforms
        (``inv(W_art) @ W_link``) and write them with ``_apply_local_matrices_fabric`` (tries
        ``omni:xform`` first, then ``omni:fabric:localMatrix``).

        ``articulation_world_matrix4d`` is the articulation root prim's world matrix (same
        layout). When omitted, ``W_art`` defaults to identity (typical for a fixed root with
        no authored root xform).
        """
        if payload.get("version") != 1:
            raise ValueError(f"unsupported PhysX JSON version: {payload.get('version')!r}")
        prims = payload.get("prims")
        if not isinstance(prims, list) or not prims:
            raise ValueError("PhysX JSON has no 'prims' entries")
        w_art_list = payload.get("articulation_world_matrix4d")
        if w_art_list is None:
            w_art = np.eye(4, dtype=np.float64)
        else:
            w_art = np.asarray(w_art_list, dtype=np.float64)
            if w_art.shape != (4, 4):
                raise ValueError("articulation_world_matrix4d must be a 4x4 matrix")
        paths: list[str] = []
        rows: list[list[list[float]]] = []
        for item in prims:
            if not isinstance(item, dict):
                raise ValueError("each prim entry must be an object")
            path = item.get("path")
            mat = item.get("matrix4d")
            if not isinstance(path, str) or not path:
                raise ValueError("each prim needs a non-empty string 'path'")
            if not isinstance(mat, list) or len(mat) != 4:
                raise ValueError(f"bad matrix4d for {path!r}")
            paths.append(path)
            rows.append(mat)
        world_m = np.asarray(rows, dtype=np.float64)
        if world_m.shape != (len(paths), 4, 4):
            raise ValueError(f"expected (N, 4, 4) matrices, got shape {world_m.shape}")

        w_art_inv = np.linalg.inv(w_art)
        local_m = np.matmul(w_art_inv, world_m)
        return self._apply_local_matrices_fabric(paths, local_m)

    def _schedule_viewport_refresh_after_physx(self) -> None:
        """Run one render step on the next event-loop tick after map_attribute unmaps.

        The mapping-attributes skill pairs writes with ``renderer.step()``; without an
        immediate step, the user can wait until the next ~33 ms timer tick — or, on
        some builds, see a stale frame until step runs.
        """
        QTimer.singleShot(0, self._on_frame)

    @staticmethod
    def _usd_source_is_local_file(usd_source: str) -> bool:
        if not usd_source or usd_source.startswith(("http://", "https://", "omniverse://")):
            return False
        try:
            return Path(usd_source).is_file()
        except OSError:
            return False

    def _run_physx_subprocess(self) -> None:
        if self._live_physx_process is not None and self._live_physx_process.state() != QProcess.ProcessState.NotRunning:
            self._log("Stop live PhysX first, then run the batch PhysX subprocess.", "warn")
            return
        if self._contact_binding_process is not None and self._contact_binding_process.state() != QProcess.ProcessState.NotRunning:
            self._log("Wait for ContactBinding to finish, then run batch PhysX.", "warn")
            return
        if self._physx_process is not None and self._physx_process.state() != QProcess.ProcessState.NotRunning:
            self._log("A simulation subprocess is already running — wait for it to finish.", "warn")
            return
        if not self._usd_source_is_local_file(self._usd_source):
            self._log(
                "PhysX can only load a local USD file path. Open a file from disk (not a remote URL), then try again.",
                "warn",
            )
            return
        if not self._physx_script.is_file():
            self._log(f"Missing PhysX worker script (expected):\n{self._physx_script}", "error")
            return

        steps, ok = input_int_wrapped(
            self,
            "PhysX simulation",
            "Simulation steps (dt = 1/60 s each). "
            "Articulated chains often need hundreds of steps before poses leave the rest pose; "
            "60 steps ≈ 1 s can look unchanged in the viewport.",
            1200,
            1,
            1_000_000,
            1,
        )
        if not ok:
            return

        rigid_text, ok_rigid = input_text_wrapped(
            self,
            "PhysX simulation",
            "Rigid body prim paths (comma-separated), or leave empty for articulation mode.\n"
            "Falling-boxes sample: list every /World/Cube* mesh that has a rigid body.",
            text=_default_rigid_paths_hint(self._usd_source),
        )
        if not ok_rigid:
            return
        rigid_list = [p.strip() for p in rigid_text.replace(";", ",").split(",") if p.strip()]

        articulation_root = "/World/articulation"
        if not rigid_list:
            articulation_root_in, ok_root = input_text_wrapped(
                self,
                "Articulation root",
                "USD path to the articulation prim (TensorBinding prim path):",
                text="/World/articulation",
            )
            if not ok_root:
                return
            articulation_root = articulation_root_in.strip() or "/World/articulation"

        if self._physx_temp_json_path is None:
            fd, tmp = tempfile.mkstemp(suffix=".json", text=True)
            os.close(fd)
            self._physx_temp_json_path = Path(tmp)

        out_json = str(self._physx_temp_json_path.resolve())
        usd_abs = str(Path(self._usd_source).resolve())
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        proc.setProcessEnvironment(env)
        proc.setProgram(sys.executable)
        if rigid_list:
            self._log(
                f"Starting PhysX worker (rigid bodies): steps={steps} paths={len(rigid_list)}",
                "info",
            )
            proc_args: list[str] = [
                str(self._physx_script),
                "--usd",
                usd_abs,
                "--steps",
                str(steps),
                "--out-json",
                out_json,
                "--rigid-body-paths",
                *rigid_list,
            ]
        else:
            self._log(
                f"Starting PhysX worker (articulation): steps={steps} articulation_root={articulation_root!r}",
                "info",
            )
            proc_args = [
                str(self._physx_script),
                "--usd",
                usd_abs,
                "--steps",
                str(steps),
                "--out-json",
                out_json,
                "--articulation-root",
                articulation_root,
            ]
        proc.setArguments(proc_args)

        def _on_finished(exit_code: int, exit_status: QProcess.ExitStatus) -> None:
            self._physx_process = None
            out = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace").strip()
            err_part = ""
            if exit_status != QProcess.ExitStatus.NormalExit:
                err_part = f"\nQt exit status: {exit_status!r}"
            if exit_code != 0:
                self._log(
                    f"PhysX subprocess failed (exit {exit_code}){err_part}\n{out or '(no output)'}",
                    "error",
                )
                return
            jp = Path(out_json)
            if not jp.is_file():
                self._log(
                    f"PhysX reported success but JSON was not found:\n{jp}\n{out}",
                    "error",
                )
                return
            try:
                attr_used = self._apply_physx_pose_json(jp)
            except Exception as exc:  # noqa: BLE001
                self._log(f"Could not apply PhysX poses: {exc}\n{out}", "error")
                return
            self._schedule_viewport_refresh_after_physx()
            self._log("[SUCCESS] PhysX worker finished and poses were applied in the viewport.", "info")
            self._log(
                f"PhysX poses applied via map_attribute({attr_used!r}); scheduled an immediate render step. "
                "The viewport does not reload USD from disk; transforms are pushed in-memory each time the subprocess finishes.\n"
                "View → Reload scene clears overrides.\n"
                + (
                    "Articulations: if nothing moved, try more steps (e.g. 1200–3000).\n"
                    if not rigid_list
                    else "Rigid bodies: if cubes look unchanged, try more steps so they fall before capture.\n"
                )
                + (out if out else "(no subprocess stdout)"),
                "info",
            )

        proc.finished.connect(_on_finished)
        self._physx_process = proc
        proc.start()
        if not proc.waitForStarted(10_000):
            self._physx_process = None
            self._log(f"PhysX subprocess failed to start: {proc.errorString()}", "error")

    def _set_play_mode(self, mode: PlayMode) -> None:
        self._play_mode = mode
        self._refresh_play_tooltip()

    def _refresh_play_tooltip(self) -> None:
        tips: dict[PlayMode, str] = {
            "ovrtx_clone": (
                "Runs clone_usd (clone_example) on the loaded stage: default /World → /World_clone "
                "(+5 X, scale 1,1,1). Use ovrtx → Clone to load the Robot sample, or open another OVRTX-ready stage."
            ),
            "ovrtx_depth": (
                "Runs depth_map_example.py in a subprocess (DepthSensorDistance → depth_map.png beside the .usda). "
                "Use ovrtx → Depth map to load robot_with_depth.usda first."
            ),
            "physx_articulation": (
                "Spawns physx_live_worker.py (JSONL articulation poses; same as legacy Live PhysX stream). "
                "Use ovphysx → Articulation to load links_chain_sample.usda."
            ),
            "physx_tensor": (
                "Runs tensor_bindings.py --viewer-stream (JSONL poses + drives). "
                "Use ovphysx → Tensor binding to load links_chain_sample.usda."
            ),
            "physx_contact": (
                "Boxes stage: physx_rigid_live_worker (JSONL rigid poses — same as legacy Simulation → Contact Binding). "
                "Other local stages: contact_binding_subprocess.py (contact-force tutorial)."
            ),
            "physx_clone_env": (
                "Runs clone_usd in the viewer then clone.py --viewer-stream. "
                "Use ovphysx → Clone to load basic_simulation.usda."
            ),
        }
        self._ovrtx_play_button.setToolTip(tips[self._play_mode])

    def _on_play_clicked(self) -> None:
        m = self._play_mode
        if m == "ovrtx_clone":
            self._run_clone_robot()
        elif m == "ovrtx_depth":
            self._run_depth_map_subprocess()
        elif m == "physx_articulation":
            self._spawn_physx_live_articulation_stream()
        elif m == "physx_tensor":
            self._spawn_tensor_bindings_viewer_stream()
        elif m == "physx_contact":
            self._run_contact_binding_subprocess()
        elif m == "physx_clone_env":
            self._play_clone_env_stream()
        else:
            self._log(f"Unhandled play mode: {m!r}", "error")

    def _open_links_chain_with_play_mode(
        self, mode: Literal["physx_articulation", "physx_tensor"]
    ) -> None:
        p = _ovlibs_usd_path("links_chain_sample.usda")
        if p is None:
            self._log(
                f"Could not find links_chain_sample.usda under:\n{OVLIBS_DIR}",
                "error",
            )
            return
        label = "Articulation" if mode == "physx_articulation" else "Tensor binding"
        self._log(f"ovphysx → {label}: loading…\n{p}", "info")
        if not self.load_usd(str(p), next_play_mode=mode):
            return
        if mode == "physx_articulation":
            self._log(
                "links_chain_sample.usda loaded. Click ▶ Play to start physx_live_worker.py (live articulation stream).",
                "info",
            )
        else:
            self._log(
                "links_chain_sample.usda loaded. Click ▶ Play to run tensor_bindings.py --viewer-stream.",
                "info",
            )

    def _open_links_chain_sample(self) -> None:
        self._open_links_chain_with_play_mode("physx_articulation")

    def _open_links_chain_sample_tensor(self) -> None:
        self._open_links_chain_with_play_mode("physx_tensor")

    def _open_boxes_falling_sample(self) -> None:
        """ovphysx → Contact binding: load boxes sample; ▶ Play matches legacy Simulation → Contact Binding."""
        p = _ovlibs_usd_path("boxes_falling_on_groundplane.usda")
        if p is None:
            self._log(
                f"Could not find boxes_falling_on_groundplane.usda under:\n{OVLIBS_DIR}",
                "error",
            )
            return
        self._log(f"ovphysx → Contact binding: loading…\n{p}", "info")
        if not self.load_usd(str(p), next_play_mode="physx_contact"):
            return
        self._log(
            "boxes_falling_on_groundplane.usda loaded. Click ▶ Play to stream rigid poses "
            "(physx_rigid_live_worker.py — same as legacy Contact Binding for this stage).",
            "info",
        )

    def _open_basic_simulation_sample(self) -> None:
        """ovphysx → Clone: load basic_simulation.usda; ▶ Play runs clone.py stream."""
        p = _ovlibs_usd_path("basic_simulation.usda")
        if p is None:
            self._log(
                f"Could not find basic_simulation.usda under:\n{OVLIBS_DIR}",
                "error",
            )
            return
        self._log(f"ovphysx → Clone: loading…\n{p}", "info")
        if not self.load_usd(str(p), next_play_mode="physx_clone_env"):
            return
        self._log(
            "basic_simulation.usda loaded. Click ▶ Play to run clone_usd + clone.py --viewer-stream.",
            "info",
        )

    def _start_boxes_falling_live_stream(self) -> None:
        """Legacy path for boxes_falling_on_groundplane.usda: rigid-body JSONL stream (viewport motion)."""
        if self._live_physx_process is not None and self._live_physx_process.state() != QProcess.ProcessState.NotRunning:
            self._log(
                "A live pose stream is already running. Use View → Reload scene to stop it first.",
                "warn",
            )
            return
        if not self._usd_source_is_local_file(self._usd_source):
            self._log(
                "Rigid live stream needs a local USD file. Use ovphysx → Contact binding or File → Open… "
                "boxes_falling_on_groundplane.usda.",
                "warn",
            )
            return
        if not self._rigid_live_script.is_file():
            self._log(f"Missing physx_rigid_live_worker.py (expected):\n{self._rigid_live_script}", "error")
            return

        usd_abs = str(Path(self._usd_source).resolve())
        rigid_list = [p.strip() for p in _DEFAULT_BOXES_FALLING_RIGID_PATHS.split(",") if p.strip()]
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        proc.setProcessEnvironment(QProcessEnvironment.systemEnvironment())
        proc.setProgram(sys.executable)
        proc.setArguments(
            [
                str(self._rigid_live_script),
                "--usd",
                usd_abs,
                "--rigid-body-paths",
                *rigid_list,
                "--dt",
                str(1.0 / 60.0),
            ]
        )

        def _on_error(_error: QProcess.ProcessError) -> None:
            if self._live_physx_process is proc:
                self._log(f"Rigid live worker error: {proc.errorString()}", "error")

        def _on_finished(exit_code: int, exit_status: QProcess.ExitStatus) -> None:
            if self._live_physx_process is not proc:
                return
            self._live_physx_process = None
            self._live_physx_stdout_buffer = ""
            self._live_physx_last_payload = None
            err_txt = bytes(proc.readAllStandardError()).decode("utf-8", errors="replace").strip()
            out_txt = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace").strip()
            if exit_status != QProcess.ExitStatus.NormalExit or exit_code != 0:
                self._log(
                    f"physx_rigid_live_worker exited (code {exit_code}, status {exit_status!r}).\n"
                    f"{err_txt or '(no stderr)'}\n{out_txt or '(no stdout)'}",
                    "warn",
                )

        proc.errorOccurred.connect(_on_error)
        proc.finished.connect(_on_finished)
        self._live_physx_process = proc
        proc.start()
        if not proc.waitForStarted(10_000):
            self._live_physx_process = None
            self._log(f"physx_rigid_live_worker failed to start: {proc.errorString()}", "error")
            return

        self._live_physx_stdout_buffer = ""
        self._live_physx_last_payload = None
        self._log(
            f"Falling-boxes live stream started ({len(rigid_list)} rigids, same as legacy Simulation → Contact Binding). "
            "View → Reload scene or close the viewer to stop.",
            "info",
        )

    def _run_contact_binding_subprocess(self) -> None:
        """Same as legacy Simulation → Run Contact Binding: boxes → rigid JSONL stream; else contact_binding tutorial."""
        if self._live_physx_process is not None and self._live_physx_process.state() != QProcess.ProcessState.NotRunning:
            self._log(
                "Stop the live pose stream first (View → Reload scene), then ▶ Play (contact binding).",
                "warn",
            )
            return
        if self._contact_binding_process is not None and self._contact_binding_process.state() != QProcess.ProcessState.NotRunning:
            self._log("Contact Binding tutorial is already running — wait for it to finish.", "warn")
            return
        if self._physx_process is not None and self._physx_process.state() != QProcess.ProcessState.NotRunning:
            self._log("Wait for the batch PhysX subprocess to finish, then ▶ Play (contact binding).", "warn")
            return

        is_boxes_falling = False
        if self._usd_source_is_local_file(self._usd_source):
            try:
                is_boxes_falling = (
                    Path(self._usd_source).resolve().name.lower() == "boxes_falling_on_groundplane.usda"
                )
            except OSError:
                is_boxes_falling = False

        if is_boxes_falling:
            self._start_boxes_falling_live_stream()
            return

        if not self._contact_binding_script.is_file():
            self._log(
                f"Missing contact_binding_subprocess.py (expected):\n{self._contact_binding_script}",
                "error",
            )
            return

        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.setProcessEnvironment(QProcessEnvironment.systemEnvironment())
        proc.setProgram(sys.executable)
        argv = [str(self._contact_binding_script)]
        if self._usd_source_is_local_file(self._usd_source):
            argv.extend(["--usd", str(Path(self._usd_source).resolve())])
        proc.setArguments(argv)
        self._log(
            "Starting ContactBinding tutorial via contact_binding_subprocess.py (ovphysx; log output). "
            "Open boxes_falling_on_groundplane.usda and use ▶ Play for falling-box **viewport** stream.\n"
            f"Command: {' '.join(argv)}",
            "info",
        )

        def _on_finished(exit_code: int, exit_status: QProcess.ExitStatus) -> None:
            self._contact_binding_process = None
            out = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace").strip()
            err_part = ""
            if exit_status != QProcess.ExitStatus.NormalExit:
                err_part = f"\nQt exit status: {exit_status!r}"
            if exit_code != 0:
                self._log(
                    f"ContactBinding subprocess failed (exit {exit_code}){err_part}\n{out or '(no output)'}",
                    "error",
                )
                return
            self._log(
                "ContactBinding sample finished (no viewport transform changes from this path; expected for non-boxes).\n"
                + (out if out else "(no subprocess stdout)"),
                "info",
            )

        proc.finished.connect(_on_finished)
        self._contact_binding_process = proc
        proc.start()
        if not proc.waitForStarted(10_000):
            self._contact_binding_process = None
            self._log(f"ContactBinding subprocess failed to start: {proc.errorString()}", "error")

    def _open_clone_robot_sample(self) -> None:
        """ovrtx → Clone: load the default Robot-OVRTX URL so the user can run ▶ Play to clone."""
        self._log(f"ovrtx → Clone: loading default Robot-OVRTX sample…\n{DEFAULT_USD_URL}", "info")
        if not self.load_usd(DEFAULT_USD_URL, next_play_mode="ovrtx_clone"):
            return
        self._log(
            "Robot sample loaded. Click ▶ Play below the viewport to run the clone script "
            "(clone_example: /World → /World_clone).",
            "info",
        )

    def _open_robot_with_depth_sample(self) -> None:
        """ovrtx → Depth map: load robot_with_depth.usda; ▶ Play runs depth_map_example.py."""
        path = _find_robot_with_depth_usda_on_disk()
        if path is None:
            self._log(
                "Could not find robot_with_depth.usda. Add examples/python/usd-viewer-example/"
                "robot_with_depth.usda or place a copy under ../ov-libaries-livestream/.",
                "error",
            )
            return
        self._log(f"ovrtx → Depth map: loading…\n{path}", "info")
        if not self.load_usd(str(path), next_play_mode="ovrtx_depth"):
            return
        self._log(
            "robot_with_depth.usda loaded. Click ▶ Play below the viewport to run depth_map_example.py "
            "(writes depth_map.png beside the .usda; separate process may contend for GPU).",
            "info",
        )

    def _run_clone_robot(self) -> None:
        if self._live_physx_process is not None and self._live_physx_process.state() != QProcess.ProcessState.NotRunning:
            self._log("Stop the live PhysX stream first, then use ▶ Play (clone).", "warn")
            return
        if self._contact_binding_process is not None and self._contact_binding_process.state() != QProcess.ProcessState.NotRunning:
            self._log("Wait for Contact Binding to finish, then use ▶ Play (clone).", "warn")
            return
        if self._physx_process is not None and self._physx_process.state() != QProcess.ProcessState.NotRunning:
            self._log("Wait for the batch PhysX subprocess to finish, then use ▶ Play (clone).", "warn")
            return
        if not self._usd_source:
            self._log("Open a USD stage first (File → Open… or use the default Robot-OVRTX URL).", "warn")
            return
        try:
            ce = _get_clone_example_module()
        except Exception as exc:  # noqa: BLE001
            self._log(f"Could not load clone_example.py: {exc}", "error")
            return

        text, ok = input_text_wrapped(
            self,
            "Clone prim subtree",
            "Source prim path (clear for default). Default: "
            f"{_DEFAULT_CLONE_SOURCE_PRIM_SAMPLE} → /World_clone (+5 X, scale 1,1,1). "
            "Env OVRTX_CLONE_ROBOT_SOURCE_PRIM also sets source.",
            text=_default_clone_source_prim_text(self._usd_source),
        )
        if not ok:
            return
        override = text.strip() or None
        try:
            src = ce.run_clone_robot_on_renderer(
                self._renderer,
                self._usd_source,
                robot_source_override=override,
            )
        except Exception as exc:  # noqa: BLE001
            self._log(
                f"Clone failed: {exc}\n"
                "If /World_clone already exists, use View → Reload scene to reset the stage, then try again.",
                "error",
            )
            return
        self._schedule_viewport_refresh_after_physx()
        self._log(
            f"[SUCCESS] Cloned to /World_clone (source prim {src!r}). "
            "View → Left panel: composed stage (debug dump) shows runtime prims. "
            "Reload scene to reset and clone again.",
            "info",
        )

    def _robot_with_depth_usd_path(self) -> Optional[Path]:
        """Path to robot_with_depth.usda for depth_map_example (viewer copy, open file, or ov-libaries-livestream)."""
        if self._usd_source_is_local_file(self._usd_source):
            try:
                p = Path(self._usd_source).resolve()
                if p.name.lower() == "robot_with_depth.usda":
                    return p
            except OSError:
                pass
        return _find_robot_with_depth_usda_on_disk()

    def _run_depth_map_subprocess(self) -> None:
        """Spawn depth_map_example.py: second Renderer process; PNG is written beside the chosen .usda."""
        if not self._depth_map_script.is_file():
            self._log(f"Missing depth_map_example.py (expected):\n{self._depth_map_script}", "error")
            return
        usd_path = self._robot_with_depth_usd_path()
        if usd_path is None:
            self._log(
                "Could not find robot_with_depth.usda. Add examples/python/usd-viewer-example/"
                "robot_with_depth.usda or open ../ov-libaries-livestream/robot_with_depth.usda.",
                "error",
            )
            return
        if self._depth_map_process is not None and self._depth_map_process.state() != QProcess.ProcessState.NotRunning:
            self._log("Depth map subprocess already running — wait for it to finish.", "warn")
            return

        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.setProcessEnvironment(QProcessEnvironment.systemEnvironment())
        proc.setProgram(sys.executable)
        proc.setArguments(
            [
                str(self._depth_map_script),
                "--usd",
                str(usd_path),
            ]
        )
        out_png = usd_path.parent / "depth_map.png"

        def _on_finished(exit_code: int, exit_status: QProcess.ExitStatus) -> None:
            self._depth_map_process = None
            out = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace").strip()
            err_part = ""
            if exit_status != QProcess.ExitStatus.NormalExit:
                err_part = f"\nQt exit status: {exit_status!r}"
            if exit_code != 0:
                self._log(
                    f"depth_map_example failed (exit {exit_code}){err_part}\n"
                    f"{out or '(no output)'}",
                    "error",
                )
                return
            self._log(
                f"[SUCCESS] Depth map written (sensor AOV). Output:\n{out_png}\n"
                f"{out or '(no subprocess stdout)'}",
                "info",
            )

        proc.finished.connect(_on_finished)
        self._depth_map_process = proc
        self._log(
            f"Starting depth_map_example.py with --usd {usd_path} (separate process; may contend for GPU with the viewer).",
            "info",
        )
        proc.start()
        if not proc.waitForStarted(10_000):
            self._depth_map_process = None
            self._log(f"depth_map_example failed to start: {proc.errorString()}", "error")

    def load_usd(self, path: str, *, next_play_mode: Optional[PlayMode] = None) -> bool:
        """Open or replace the USD layer on this renderer and refresh the left panel.

        On success, optionally set ``▶ Play`` intent (``next_play_mode``). Callers that load a
        sample for a specific script should pass this so Play stays aligned if load fails mid-way.
        """
        try:
            if self._usd_source:
                self._stop_live_physx(uncheck_menu=True)
                if self._contact_binding_process is not None:
                    if self._contact_binding_process.state() != QProcess.ProcessState.NotRunning:
                        self._contact_binding_process.kill()
                        self._contact_binding_process.waitForFinished(3000)
                    self._contact_binding_process = None
                if self._physx_process is not None:
                    if self._physx_process.state() != QProcess.ProcessState.NotRunning:
                        self._physx_process.kill()
                        self._physx_process.waitForFinished(3000)
                    self._physx_process = None
                if self._depth_map_process is not None:
                    if self._depth_map_process.state() != QProcess.ProcessState.NotRunning:
                        self._depth_map_process.kill()
                        self._depth_map_process.waitForFinished(3000)
                    self._depth_map_process = None
                # open_usd() resets the stage internally; reset() clears simulation history.
                self._renderer.reset(time=0.0)
            self._log(f"Loading USD: {path}", "info")
            self._renderer.open_usd(path)
            self._usd_source = path
            self.setWindowTitle(f"USD Viewer — {path}")
            self._step_error_shown = False
            self._no_frames_warned = False
            self._log("USD loaded.", "info")
            self._apply_render_product_for_current_stage()
            self._refresh_root_layer_source_panel(from_load=True)
        except Exception as exc:  # noqa: BLE001 — surface load failures in UI
            self._log(f"Failed to load USD: {exc}", "error")
            return False
        if next_play_mode is not None:
            self._set_play_mode(next_play_mode)
        return True

    # --- Which RenderProduct path to step (override, probe Kit vs camera paths, optional dump) ---

    def _apply_render_product_for_current_stage(self) -> None:
        if self._render_product_override is not None:
            self._render_product = self._render_product_override
        else:
            self._render_product = self._detect_render_product(self._usd_source)
        self._render_products = {self._render_product}
        self._log(f"Using render product: {self._render_product}", "info")

    def _detect_render_product(self, usd_source: str) -> str:
        for cand in _probe_order_for_usd_source(usd_source):
            if self._probe_render_product(cand):
                self._log(f"Auto-detect: using {cand!r} (probe succeeded).", "info")
                return cand

        self._log(
            "Auto-detect: no standard render product responded with HdrColor/LdrColor.",
            "info",
        )
        if not self._use_stage_dump:
            self._log(
                f"Auto-detect: using {FALLBACK_RENDER_PRODUCT!r} "
                f"(try --use-stage-dump or -r PATH for exotic stages).",
                "info",
            )
            return FALLBACK_RENDER_PRODUCT

        return self._detect_render_product_from_stage_dump()

    def _detect_render_product_from_stage_dump(self) -> str:
        try:
            products = self._renderer.step(
                render_products={"ovrtx_debug_dump_stage"},
                delta_time=1.0 / 60.0,
            )
        except Exception as exc:  # noqa: BLE001
            self._log(
                f"Auto-detect: debug dump step failed ({exc!r}); "
                f"using {FALLBACK_RENDER_PRODUCT!r}.",
                "info",
            )
            return FALLBACK_RENDER_PRODUCT
        if products is None or "ovrtx_debug_dump_stage" not in products:
            self._log(
                f"Auto-detect: no ovrtx_debug_dump_stage; using {FALLBACK_RENDER_PRODUCT!r}.",
                "info",
            )
            return FALLBACK_RENDER_PRODUCT
        frames = list(products["ovrtx_debug_dump_stage"].frames)
        if not frames or "debug" not in frames[0].render_vars:
            self._log(
                f"Auto-detect: missing debug dump payload; using {FALLBACK_RENDER_PRODUCT!r}.",
                "info",
            )
            return FALLBACK_RENDER_PRODUCT
        try:
            with frames[0].render_vars["debug"].map(device=Device.CPU) as mapping:
                dump = mapping.tensor.to_bytes().decode("utf-8")
        except Exception as exc:  # noqa: BLE001
            self._log(
                f"Auto-detect: could not read dump ({exc!r}); using {FALLBACK_RENDER_PRODUCT!r}.",
                "info",
            )
            return FALLBACK_RENDER_PRODUCT
        for cand in _candidate_render_products_from_dump(dump):
            if self._probe_render_product(cand):
                self._log(f"Auto-detect: using {cand!r} (probe after dump succeeded).", "info")
                return cand

        chosen = _render_product_from_stage_dump(dump)
        self._log(
            f"Auto-detect: no probe succeeded; falling back to dump hint {chosen!r}.",
            "info",
        )
        return chosen

    def _probe_render_product(self, rp: str) -> bool:
        """True if a normal step can use this render product (has color output)."""
        try:
            products = self._renderer.step(
                render_products={rp},
                delta_time=1.0 / 60.0,
            )
        except Exception as exc:  # noqa: BLE001
            self._log(f"Auto-detect: probe {rp!r} raised {exc!r}.", "info")
            return False
        if products is None or rp not in products:
            return False
        frames = list(products[rp].frames)
        if not frames:
            return False
        return _pick_color_var_name(frames[0].render_vars) is not None

    def _pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open USD",
            "",
            "USD files (*.usd *.usda *.usdc);;All files (*.*)",
        )
        if path:
            self.load_usd(path)

    def resizeEvent(self, event) -> None:  # noqa: N802 — Qt API
        super().resizeEvent(event)
        self._refresh_label_pixmap()

    def _refresh_label_pixmap(self) -> None:
        if self._last_qimage is None or self._last_qimage.isNull():
            return
        pm = QPixmap.fromImage(self._last_qimage)
        lw, lh = self._label.width(), self._label.height()
        if lw < 2 or lh < 2:
            self._label.setPixmap(pm)
            return
        pm = pm.scaled(
            self._label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(pm)

    def _on_frame(self) -> None:
        """One display tick: optional JSONL pose apply, then renderer.step and Qt blit."""
        dt = 1.0 / 60.0
        if self._live_physx_process is not None:
            self._drain_live_physx_stdout()
            if self._live_physx_last_payload is not None:
                try:
                    self._apply_physx_pose_payload(self._live_physx_last_payload)
                except Exception as exc:  # noqa: BLE001
                    self._log(f"Live PhysX pose apply failed: {exc}", "error")
                    self._stop_live_physx(uncheck_menu=True)

        # Heart of the viewer: advance time and produce GPU outputs for the active RenderProduct set.
        try:
            products = self._renderer.step(
                render_products=self._render_products,
                delta_time=dt,
            )
        except Exception as exc:  # noqa: BLE001
            if not self._step_error_shown:
                self._step_error_shown = True
                self._log(f"Render step failed: {exc}", "error")
            return

        if products is None:
            if not self._step_error_shown:
                self._step_error_shown = True
                self._log("Render step failed: step() returned no products.", "error")
            return

        # RenderProductSetOutputs supports `in` and [] but not dict.get().
        if self._render_product not in products:
            if not self._step_error_shown:
                self._step_error_shown = True
                names = ", ".join(sorted(n for n, _ in products.items())) or "(none)"
                self._log(
                    f"Missing render product: stage has no output for {self._render_product!r}. "
                    f"Available products: {names}",
                    "error",
                )
            return

        product = products[self._render_product]
        frames = list(product.frames)
        if not frames:
            if not self._no_frames_warned:
                self._no_frames_warned = True
                self._log(
                    f"No frames: render product {self._render_product!r} returned no frames this step.",
                    "warn",
                )
            return

        try:
            frame = frames[0]
            var_name = _pick_color_var_name(frame.render_vars)
            if var_name is None:
                if not self._step_error_shown:
                    self._step_error_shown = True
                    names = (
                        ", ".join(sorted(n for n, _ in frame.render_vars.items()))
                        or "(none)"
                    )
                    self._log(
                        "No color buffer: need HdrColor or LdrColor on this render product. "
                        f"Available render vars: {names}",
                        "error",
                    )
                return

            # Copy while mapped: the CPU tensor view is only valid inside this ``with`` block.
            with frame.render_vars[var_name].map(device=Device.CPU) as var:
                src = var.tensor.numpy()
                if var_name == "LdrColor":
                    if src.dtype != np.uint8:
                        raise TypeError(
                            f"LdrColor expected uint8, got dtype={src.dtype!r}"
                        )
                    if src.ndim != 3 or src.shape[2] < 4:
                        raise ValueError(
                            f"LdrColor expected HxWx>=4, got shape={src.shape!r}"
                        )
                    rgba_u8 = np.ascontiguousarray(src[:, :, :4]).copy()
                else:
                    if src.dtype not in (np.float16, np.float32):
                        raise TypeError(
                            f"HdrColor expected float16/float32, got dtype={src.dtype!r}"
                        )
                    if src.ndim != 3 or src.shape[2] < 4:
                        raise ValueError(
                            f"HdrColor expected HxWx>=4, got shape={src.shape!r}"
                        )
                    rgba_u8 = _hdr_linear_rgba_to_rgba8(src)

            h, w = int(rgba_u8.shape[0]), int(rgba_u8.shape[1])
            raw = rgba_u8.tobytes()
            qimg = QImage(
                raw,
                w,
                h,
                4 * w,
                QImage.Format.Format_RGBA8888,
            ).copy()
            self._last_qimage = qimg
            self._refresh_label_pixmap()
        except Exception as exc:  # noqa: BLE001
            if not self._step_error_shown:
                self._step_error_shown = True
                self._log(f"Failed to read frame: {exc}", "error")


def main() -> None:
    """Entry point: CLI args, Qt application, show window and run the event loop."""
    args = parse_args()
    app = QApplication(sys.argv)
    win = ViewerWindow(
        args.usd,
        args.render_product,
        use_stage_dump=args.use_stage_dump,
    )
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

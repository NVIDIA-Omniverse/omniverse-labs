# SPDX-License-Identifier: Apache-2.0
"""
nanousdview entrypoint — `python -m nanousdview file.usda`.

Loads the shared nanousd Python API, which registers its pxr compatibility
shim so `from pxr import ...` resolves to nanousd, never to OpenUSD.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def _install_nanousd_python_api():
    """Load nanousd-python and return its shared pxr compatibility module."""
    try:
        from nanousd import pxr_compat as _pxr_compat
    except ImportError as exc:  # pragma: no cover - exercised by install failures
        raise ImportError(
            "nanousdview requires the nanousd-python package. Build the "
            "workspace with ./build.sh or install nanousd-python from the "
            "sibling nanousd-python repo."
        ) from exc
    return _pxr_compat


def _install_stack_dump_signal() -> None:
    if os.environ.get("NUSD_VIEW_DUMP_STACK_SIGNAL", "").lower() not in (
        "1", "true", "yes", "on"
    ):
        return
    try:
        import faulthandler
        import signal
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    except Exception:
        pass


def _raise_file_descriptor_limit() -> None:
    """Give large composed stages enough room for layer + texture files."""
    try:
        import resource
    except Exception:
        return

    try:
        target = int(os.environ.get("NANOUSD_VIEW_NOFILE", "8192"))
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft >= target:
            return
        new_soft = target if hard == resource.RLIM_INFINITY else min(target, hard)
        if new_soft > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    except Exception:
        pass


def _stage_view(ctrl):
    """Locate the stageView widget — same lookup as the screenshot path."""
    s = getattr(ctrl, "_stageView", None)
    if s is None:
        ui = getattr(ctrl, "_ui", None)
        if ui is not None:
            s = getattr(ui, "stageView", None)
    return s


def _close_stage_renderer(ctrl) -> None:
    stage = _stage_view(ctrl)
    renderer = getattr(stage, "_renderer", None) if stage is not None else None
    if renderer is not None:
        close = getattr(renderer, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        try:
            stage._renderer = None
        except Exception:
            pass


def _install_nanousd_menu(ctrl) -> None:
    """Add a top-level "View" menu with render-mode picker + HUD toggle.

    The vendored usdviewq menubar is built from a generated .ui (File / Edit
    / Window) and intentionally doesn't expose the Hydra "Render" submenu
    here, so we install our own menu post-construction. Bound to the same
    state stageView's `T` and `H` keys toggle, so menu and keys stay in sync.
    """
    from nanousdview.usdviewq.qt import QtCore, QtGui, QtWidgets  # local
    try:
        from nanousdview._backend import (
            VIEW_RENDER_RASTER,
            VIEW_RENDER_SHADOW,
            VIEW_RENDER_RT,
            get_backend,
        )
    except ImportError:
        return  # backend not available; nothing to wire

    stage = _stage_view(ctrl)
    if stage is None:
        return
    mw = ctrl._mainWindow
    bar = mw.menuBar()

    menu = bar.addMenu("&View")

    # Render mode submenu — radio actions, kept in sync with stageView state.
    mode_menu = menu.addMenu("Render &Mode")
    grp = QtGui.QActionGroup(mw)
    grp.setExclusive(True)

    backend = get_backend()
    if backend == "opengl":
        supported_modes = {VIEW_RENDER_RASTER}
    elif backend == "ovrtx":
        supported_modes = {VIEW_RENDER_RT}
    else:
        supported_modes = {VIEW_RENDER_RT, VIEW_RENDER_SHADOW, VIEW_RENDER_RASTER}

    def _add_mode(label, mode_const, shortcut=None):
        a = QtGui.QAction(label, mw, checkable=True)
        if shortcut:
            a.setShortcut(QtGui.QKeySequence(shortcut))
        a.setData(int(mode_const))
        a.setChecked(stage._render_mode == mode_const)
        a.setEnabled(mode_const in supported_modes)
        grp.addAction(a)
        mode_menu.addAction(a)
        return a

    _add_mode("Ray Tracing", VIEW_RENDER_RT)
    _add_mode("Shadow",      VIEW_RENDER_SHADOW)
    _add_mode("Raster",      VIEW_RENDER_RASTER)

    def _on_mode_triggered(action):
        stage._render_mode = int(action.data())
        stage.update()
    grp.triggered.connect(_on_mode_triggered)

    # Keep the radio in sync if the user presses T (cycle).
    def _resync_mode():
        for a in grp.actions():
            a.setChecked(int(a.data()) == stage._render_mode)
    stage._resync_render_mode_menu = _resync_mode

    # HUD toggle.
    menu.addSeparator()
    hud_action = QtGui.QAction("Show &HUD", mw, checkable=True)
    hud_action.setShortcut(QtGui.QKeySequence("H"))
    hud_action.setChecked(stage._show_hud)
    def _on_hud_toggled(checked):
        stage._show_hud = bool(checked)
        stage.update()
    hud_action.toggled.connect(_on_hud_toggled)
    menu.addAction(hud_action)
    stage._hud_menu_action = hud_action  # so H-key handler can keep it in sync


def _qa_vec3(value) -> list[float]:
    return [float(value[i]) for i in range(3)]


def _qa_key_value(key) -> int:
    try:
        return int(key)
    except Exception:
        return int(getattr(key, "value", 0))


def _qa_camera_state(stage) -> dict:
    eye, target, up = stage._camera_eye_target_up()
    render_mode = getattr(stage, "_render_mode", None)
    return {
        "mode": "FLY" if getattr(stage, "_fly_mode", False) else "ORBIT",
        "eye": _qa_vec3(eye),
        "target": _qa_vec3(target),
        "up": _qa_vec3(up),
        "speed_m_s": float(getattr(stage, "_fly_speed", 0.0)),
        "render_mode": int(render_mode) if render_mode is not None else None,
        "held_motion_keys": sorted(
            _qa_key_value(k) for k in getattr(stage, "_wasd_keys", set())),
        "viewport": {
            "width": int(stage.width()),
            "height": int(stage.height()),
        },
    }


def _qa_camera_delta(a: dict, b: dict) -> float:
    return (
        sum(abs(a["eye"][i] - b["eye"][i]) for i in range(3))
        + sum(abs(a["target"][i] - b["target"][i]) for i in range(3))
    )


def _qa_format_vec(values: list[float]) -> str:
    return "(" + ", ".join(f"{v:.3f}" for v in values) + ")"


def _qa_print_camera_state(label: str, state: dict) -> None:
    print(
        f"[--qa] {label}: mode={state['mode']} "
        f"eye={_qa_format_vec(state['eye'])} "
        f"target={_qa_format_vec(state['target'])} "
        f"up={_qa_format_vec(state['up'])} "
        f"speed={state['speed_m_s']:.3f} m/s"
    )


def _qa_pixmap_metrics(pixmap) -> dict:
    """Small sampled viewport metrics for visual QA/nonblank checks."""
    try:
        from nanousdview.usdviewq.qt import QtGui

        try:
            fmt = QtGui.QImage.Format.Format_RGB32
        except AttributeError:
            fmt = QtGui.QImage.Format_RGB32
        image = pixmap.toImage().convertToFormat(fmt)
        width = int(image.width())
        height = int(image.height())
        if width <= 0 or height <= 0:
            return {"width": width, "height": height, "sample_count": 0}

        def sample_region(x0: int, y0: int, x1: int, y1: int) -> dict:
            region_w = max(x1 - x0, 1)
            region_h = max(y1 - y0, 1)
            step = max(1, min(region_w, region_h) // 240)
            count = 0
            sum_r = sum_g = sum_b = 0
            sum_luma = 0.0
            sum_luma2 = 0.0
            min_luma = 255.0
            max_luma = 0.0
            dark = bright = near_white = clipped = 0
            for y in range(y0, y1, step):
                for x in range(x0, x1, step):
                    color = image.pixelColor(x, y)
                    r = color.red()
                    g = color.green()
                    b = color.blue()
                    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    count += 1
                    sum_r += r
                    sum_g += g
                    sum_b += b
                    sum_luma += luma
                    sum_luma2 += luma * luma
                    min_luma = min(min_luma, luma)
                    max_luma = max(max_luma, luma)
                    if luma < 8:
                        dark += 1
                    if luma > 210:
                        bright += 1
                    if r > 235 and g > 235 and b > 235:
                        near_white += 1
                    if r >= 254 or g >= 254 or b >= 254:
                        clipped += 1
            if count == 0:
                return {"sample_count": 0}
            mean_luma = sum_luma / count
            luma_var = max(sum_luma2 / count - mean_luma * mean_luma, 0.0)
            return {
                "sample_count": count,
                "mean_rgb": [
                    round(sum_r / count, 3),
                    round(sum_g / count, 3),
                    round(sum_b / count, 3),
                ],
                "mean_luma": round(mean_luma, 3),
                "luma_stddev": round(luma_var ** 0.5, 3),
                "min_luma": round(min_luma, 3),
                "max_luma": round(max_luma, 3),
                "dark_fraction": dark / count,
                "bright_fraction": bright / count,
                "near_white_fraction": near_white / count,
                "clipped_channel_fraction": clipped / count,
            }

        return {
            "width": width,
            "height": height,
            "whole": sample_region(0, 0, width, height),
            "center": sample_region(width // 5, height // 5,
                                    (width * 4) // 5, (height * 4) // 5),
            "lower_half": sample_region(0, height // 2, width, height),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _qa_record_view(ctrl, report: dict, label: str) -> None:
    stage = _stage_view(ctrl)
    if stage is None:
        return
    state = _qa_camera_state(stage)
    _qa_print_camera_state(label, state)
    report[f"camera_{label}"] = state
    try:
        pixmap = stage.grab()
        metrics = _qa_pixmap_metrics(pixmap)
        report[f"viewport_metrics_{label}"] = metrics
        lower = metrics.get("lower_half", {})
        if lower:
            print(
                f"[--qa] viewport {label}: "
                f"{metrics.get('width')}x{metrics.get('height')} "
                f"lower_luma_std={lower.get('luma_stddev', 0.0):.3f} "
                f"lower_bright={lower.get('bright_fraction', 0.0):.4f} "
                f"lower_near_white={lower.get('near_white_fraction', 0.0):.4f} "
                f"lower_clipped={lower.get('clipped_channel_fraction', 0.0):.4f}"
            )
    except Exception as exc:  # noqa: BLE001
        report[f"viewport_metrics_{label}"] = {"error": str(exc)}


def _qa_write_report(path: str | None, report: dict) -> None:
    if not path:
        return
    report_path = Path(path).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[--qa-report] wrote {report_path}")


def _qa_assert_view_has_content(report: dict, label: str) -> None:
    metrics = report.get(f"viewport_metrics_{label}", {})
    whole = metrics.get("whole", {})
    center = metrics.get("center", {})
    whole_stddev = float(whole.get("luma_stddev", 0.0) or 0.0)
    whole_span = float(whole.get("max_luma", 0.0) or 0.0) - \
        float(whole.get("min_luma", 0.0) or 0.0)
    center_stddev = float(center.get("luma_stddev", 0.0) or 0.0)
    center_span = float(center.get("max_luma", 0.0) or 0.0) - \
        float(center.get("min_luma", 0.0) or 0.0)
    center_count = int(center.get("sample_count", 0) or 0)
    if center_count > 0 and center_stddev < 2.0 and center_span < 10.0:
        raise RuntimeError(
            f"viewport {label} center appears flat "
            f"(center_luma_stddev={center_stddev:.3f}, "
            f"center_luma_span={center_span:.3f})"
        )
    if int(whole.get("sample_count", 0) or 0) > 0 \
            and whole_stddev < 4.0 and whole_span < 40.0:
        raise RuntimeError(
            f"viewport {label} appears visually flat "
            f"(luma_stddev={whole_stddev:.3f}, "
            f"luma_span={whole_span:.3f})"
        )


def _qa_base_report(args) -> dict:
    return {
        "usd_path": args.usd_path,
        "backend": args.backend or os.environ.get("NANOUSD_VIEW_BACKEND"),
        "render_mode": (
            args.render_mode or os.environ.get("NANOUSD_VIEW_RENDER_MODE")),
        "capture_window": args.capture_window,
        "capture_delay_seconds": float(args.capture_delay),
        "width": args.width,
        "height": args.height,
        "camera_arg": args.camera,
    }


def _run_camera_qa_controls(ctrl, app, report: dict | None = None,
                            restore_view: bool = True) -> None:
    from nanousdview.usdviewq.qt import (
        QtCore,
        PySideModule,
    )
    try:
        if PySideModule == "PySide2":
            from PySide2 import QtTest
        else:
            from PySide6 import QtTest
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"QtTest unavailable: {exc}") from exc

    stage = _stage_view(ctrl)
    if stage is None:
        raise RuntimeError("stage view unavailable for camera QA")

    def pump(count: int = 2) -> None:
        for _ in range(max(count, 1)):
            app.processEvents()

    saved_view = None
    view_settings = getattr(
        getattr(stage, "_dataModel", None), "viewSettings", None)
    saved_camera_prim = getattr(view_settings, "cameraPrim", None) \
        if view_settings is not None else None
    if restore_view and hasattr(stage, "copyViewState"):
        saved_view = stage.copyViewState()

    stage.setFocus()
    pump()
    active_camera = getattr(stage, "_activeScenePrimCamera", None)
    had_scene_camera = bool(active_camera and active_camera() is not None)
    before = _qa_camera_state(stage)
    _qa_print_camera_state("camera_controls_before", before)

    QtTest.QTest.keyPress(stage, QtCore.Qt.Key_W)
    if QtCore.Qt.Key_W not in getattr(stage, "_wasd_keys", set()):
        raise RuntimeError("WASD key press was not captured by the stage view")
    stage._on_tick()
    QtTest.QTest.keyRelease(stage, QtCore.Qt.Key_W)
    after_wasd = _qa_camera_state(stage)
    moved_delta = _qa_camera_delta(before, after_wasd)
    if moved_delta <= 1e-5:
        raise RuntimeError("WASD camera input did not move the view")
    if getattr(stage, "_wasd_keys", set()):
        raise RuntimeError("WASD motion key still held after key release")
    if had_scene_camera and active_camera() is not None:
        raise RuntimeError("camera navigation left the USD scene camera pinned")

    released = _qa_camera_state(stage)
    stage._on_tick()
    stopped = _qa_camera_state(stage)
    stop_delta = _qa_camera_delta(released, stopped)
    if stop_delta >= 1e-7:
        raise RuntimeError("WASD camera kept moving after key release")
    print(
        f"[--qa] camera_controls: moved_delta={moved_delta:.8f} "
        f"release_delta={stop_delta:.8f}"
    )

    if saved_view is not None and hasattr(stage, "restoreViewState"):
        stage.restoreViewState(saved_view)
    if view_settings is not None:
        try:
            view_settings.cameraPrim = saved_camera_prim
        except Exception:
            pass
    try:
        stage.update()
    except Exception:
        pass
    if saved_view is not None or view_settings is not None:
        pump(3)

    restored = _qa_camera_state(stage)
    _qa_print_camera_state("camera_controls_after_restore", restored)
    if report is not None:
        report["camera_controls"] = {
            "before": before,
            "after_wasd": after_wasd,
            "after_release_tick": stopped,
            "after_restore": restored,
            "moved_delta": moved_delta,
            "release_delta": stop_delta,
            "restored_delta": _qa_camera_delta(before, restored),
            "held_keys_after_release": sorted(
                _qa_key_value(k) for k in getattr(stage, "_wasd_keys", set())),
            "detached_scene_camera": bool(had_scene_camera),
        }


def _run_ui_qa_interactions(ctrl, app, report: dict | None = None) -> None:
    import traceback

    from nanousdview.usdviewq.qt import (
        QtCore,
        QtGui,
        QtWidgets,
        PySideModule,
    )
    try:
        if PySideModule == "PySide2":
            from PySide2 import QtTest
        else:
            from PySide6 import QtTest
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"QtTest unavailable: {exc}") from exc

    qt_exceptions: list[str] = []
    original_excepthook = sys.excepthook

    def _qa_excepthook(exc_type, exc_value, exc_traceback):
        qt_exceptions.append("".join(traceback.format_exception(
            exc_type, exc_value, exc_traceback)).strip())
        original_excepthook(exc_type, exc_value, exc_traceback)

    sys.excepthook = _qa_excepthook

    def check_qt_exceptions() -> None:
        if qt_exceptions:
            raise RuntimeError("Qt callback raised during QA:\n" +
                               "\n".join(qt_exceptions))

    def pump(count: int = 2) -> None:
        for _ in range(max(count, 1)):
            app.processEvents()
            check_qt_exceptions()

    def check(condition: bool, message: str) -> None:
        check_qt_exceptions()
        if not condition:
            raise RuntimeError(message)

    try:
        mw = getattr(ctrl, "_mainWindow", None)
        if mw is not None:
            for w, h in ((1800, 1050), (1280, 820), (2000, 1160)):
                mw.resize(w, h)
                pump(3)

        stage = _stage_view(ctrl)
        if stage is not None:
            for w, h in ((960, 540), (1200, 675), (1440, 810)):
                stage.setFixedSize(w, h)
                pump(3)
            stage.setFocus()
            pump()

            active_camera = getattr(stage, "_activeScenePrimCamera", None)
            had_scene_camera = bool(active_camera and active_camera() is not None)
            before_wasd = _qa_camera_state(stage)
            _qa_print_camera_state("ui_interactions_before_wasd", before_wasd)
            QtTest.QTest.keyPress(stage, QtCore.Qt.Key_W)
            check(QtCore.Qt.Key_W in getattr(stage, "_wasd_keys", set()),
                  "WASD key press was not captured by the stage view")
            stage._on_tick()
            QtTest.QTest.keyRelease(stage, QtCore.Qt.Key_W)
            after_wasd = _qa_camera_state(stage)
            wasd_delta = _qa_camera_delta(before_wasd, after_wasd)
            check(wasd_delta > 1e-5,
                  "WASD camera input did not move the view")
            check(not getattr(stage, "_wasd_keys", set()),
                  "WASD motion key still held after key release")
            if had_scene_camera:
                check(active_camera() is None,
                      "camera navigation left the USD scene camera pinned")
            released_wasd = _qa_camera_state(stage)
            stage._on_tick()
            stopped_wasd = _qa_camera_state(stage)
            stop_delta = _qa_camera_delta(released_wasd, stopped_wasd)
            check(stop_delta < 1e-7,
                  "WASD camera kept moving after key release")
            print(
                f"[--qa] ui_interactions_wasd: "
                f"moved_delta={wasd_delta:.8f} release_delta={stop_delta:.8f}"
            )
            if report is not None:
                report["ui_interactions_wasd"] = {
                    "before": before_wasd,
                    "after_wasd": after_wasd,
                    "after_release_tick": stopped_wasd,
                    "moved_delta": wasd_delta,
                    "release_delta": stop_delta,
                }

            x0 = max(stage.width() // 2 - 120, 10)
            y0 = max(stage.height() // 2 - 80, 10)
            x1 = min(x0 + 180, max(stage.width() - 10, 10))
            y1 = min(y0 + 120, max(stage.height() - 10, 10))
            p0 = QtCore.QPoint(x0, y0)
            p1 = QtCore.QPoint(x1, y1)
            QtTest.QTest.mousePress(stage, QtCore.Qt.LeftButton,
                                    QtCore.Qt.NoModifier, p0)
            QtTest.QTest.mouseMove(stage, p1, 25)
            QtTest.QTest.mouseRelease(stage, QtCore.Qt.LeftButton,
                                      QtCore.Qt.NoModifier, p1)
            QtTest.QTest.keyClick(stage, QtCore.Qt.Key_PageUp)
            pump(4)

        ui = getattr(ctrl, "_ui", None)
        if ui is not None:
            prim_view = getattr(ui, "primView", None)
            if prim_view is not None:
                QtTest.QTest.mouseClick(
                    prim_view.viewport(), QtCore.Qt.LeftButton,
                    QtCore.Qt.NoModifier, QtCore.QPoint(80, 30))
                pump()
            prim_find = getattr(ui, "primViewLineEdit", None)
            if prim_find is not None:
                prim_find.setFocus()
                prim_find.clear()
                QtTest.QTest.keyClicks(prim_find, "Sphere")
                pump()
                find_btn = getattr(ui, "primViewFindNext", None)
                if find_btn is not None:
                    QtTest.QTest.mouseClick(find_btn, QtCore.Qt.LeftButton)
                    pump()
            prop_find = getattr(ui, "attrViewLineEdit", None)
            if prop_find is not None:
                prop_find.setFocus()
                prop_find.clear()
                QtTest.QTest.keyClicks(prop_find, "xform")
                pump()
                prop_btn = getattr(ui, "attrViewFindNext", None)
                if prop_btn is not None:
                    QtTest.QTest.mouseClick(prop_btn, QtCore.Qt.LeftButton)
                    pump()
            inspector = getattr(ui, "propertyInspector", None)
            if inspector is not None and inspector.count() > 1:
                inspector.setCurrentIndex(1)
                pump()
                inspector.setCurrentIndex(0)
                pump()

            def trigger(action_name: str):
                action = getattr(ui, action_name, None)
                check(action is not None, f"missing action {action_name}")
                check(action.isEnabled(), f"disabled action {action_name}")
                action.trigger()
                pump(4)
                return action

        open_path = Path(getattr(ctrl._parserData, "usdFile", "") or "test_cube.usda")
        if not open_path.is_absolute():
            open_path = (Path.cwd() / open_path).resolve()
        check(open_path.is_file(), f"Open File QA source missing: {open_path}")

        from nanousdview.usdviewq.attributeViewContextMenu import (
            CopyAllTargetPathsMenuItem,
        )
        from nanousdview.usdviewq.common import (
            PropertyViewDataRoles,
            PropertyViewIndex,
        )
        rel_item = QtWidgets.QTreeWidgetItem()
        rel_item.setData(
            PropertyViewIndex.TYPE,
            QtCore.Qt.ItemDataRole.WhatsThisRole,
            PropertyViewDataRoles.RELATIONSHIP_WITH_TARGETS,
        )
        for target_path in ("/World/Cube", "/World/Floor"):
            child = QtWidgets.QTreeWidgetItem(rel_item)
            child.setText(PropertyViewIndex.NAME, target_path)
        CopyAllTargetPathsMenuItem(ctrl._dataModel, rel_item).RunCommand()
        check(QtWidgets.QApplication.clipboard().text() ==
              "/World/Cube, /World/Floor",
              "Copy Target Path(s) context menu copied unexpected text")

        from nanousdview.usdviewq import layerStackContextMenu
        layer_item = QtWidgets.QTreeWidgetItem()
        layer_item.layerPath = str(open_path)
        layer_item.identifier = str(open_path)
        layer_item.path = "/World/Cube"

        layerStackContextMenu.CopyLayerPathMenuItem(layer_item).RunCommand()
        check(QtWidgets.QApplication.clipboard().text() == str(open_path),
              "Copy Layer Path context menu copied unexpected text")
        layerStackContextMenu.CopyLayerIdentifierMenuItem(layer_item).RunCommand()
        check(QtWidgets.QApplication.clipboard().text() == str(open_path),
              "Copy Layer Identifier context menu copied unexpected text")
        layerStackContextMenu.CopyPathMenuItem(layer_item).RunCommand()
        check(QtWidgets.QApplication.clipboard().text() == "/World/Cube",
              "Copy Object Path context menu copied unexpected text")

        launched_layers = []
        original_spawn_nanousdview = layerStackContextMenu.SpawnNanousdview

        def _qa_spawn_nanousdview(path, extraArgs=()):
            launched_layers.append((str(path), tuple(extraArgs)))
            return None

        layerStackContextMenu.SpawnNanousdview = _qa_spawn_nanousdview
        try:
            layerStackContextMenu.UsdviewLayerMenuItem(layer_item).RunCommand()
        finally:
            layerStackContextMenu.SpawnNanousdview = original_spawn_nanousdview
        check(launched_layers == [(str(open_path), ())],
              "Open Layer In nanousdview launched unexpected command")

        original_get_open_file_name = QtWidgets.QFileDialog.getOpenFileName

        def _qa_open_file_name(*_args, **_kwargs):
            return (str(open_path), "")

        QtWidgets.QFileDialog.getOpenFileName = _qa_open_file_name
        try:
            trigger("actionOpen")
        finally:
            QtWidgets.QFileDialog.getOpenFileName = original_get_open_file_name
        check(str(ctrl._parserData.usdFile) == str(open_path),
              "Open File did not update parserData.usdFile")

        exclusive_action_sets = (
            ("actionWireframe", "actionWireframeOnSurface",
             "actionSmooth_Shaded", "actionFlat_Shaded", "actionPoints",
             "actionGeom_Only", "actionGeom_Smooth", "actionGeom_Flat",
             "actionHidden_Surface_Wireframe"),
            ("actionNoColorCorrection", "actionSRGBColorCorrection",
             "actionOpenColorIO"),
            ("actionPick_Prims", "actionPick_Models", "actionPick_Instances",
             "actionPick_Prototypes"),
            ("actionNever", "actionOnly_when_paused", "actionAlways"),
            ("actionSelYellow", "actionSelCyan", "actionSelWhite"),
            ("actionBlack", "actionGrey_Dark", "actionGrey_Light",
             "actionWhite"),
            ("actionLow", "actionMedium", "actionHigh", "actionVery_High"),
            ("actionLevel_1", "actionLevel_2", "actionLevel_3",
             "actionLevel_4", "actionLevel_5", "actionLevel_6",
             "actionLevel_7", "actionLevel_8"),
            ("actionCameraMask_Full", "actionCameraMask_Partial",
             "actionCameraMask_None"),
        )
        for action_names in exclusive_action_sets:
            for action_name in action_names:
                action = getattr(ui, action_name, None)
                if action is None or not action.isEnabled():
                    continue
                action.trigger()
                pump(2)
                if action.isCheckable():
                    check(action.isChecked(),
                          f"{action_name} did not become checked")

        reversible_toggles = (
            "showBBoxes", "showAABBox", "showOBBox", "showBBoxPlayback",
            "useExtentsHint", "actionDisplay_Guide", "actionDisplay_Proxy",
            "actionDisplay_Render", "actionEnable_Scene_Materials",
            "actionEnable_Scene_Lights", "actionCull_Backfaces",
            "actionDomeLightTexturesVisible", "actionAmbient_Only",
            "actionDomeLight", "actionHUD", "actionHUD_Info",
            "actionHUD_Complexity", "actionHUD_Performance",
            "actionHUD_GPUstats", "actionRollover_Prim_Info",
            "actionAuto_Compute_Clipping_Planes", "actionCameraMask_Outline",
            "actionCameraReticles_Inside", "actionCameraReticles_Outside",
        )
        for action_name in reversible_toggles:
            action = getattr(ui, action_name, None)
            if action is None or not action.isEnabled() or not action.isCheckable():
                continue
            was_checked = action.isChecked()
            action.trigger()
            pump(2)
            action.trigger()
            pump(2)
            check(action.isChecked() == was_checked,
                  f"{action_name} did not restore checked state")

        for action_name in ("actionReset_View", "actionFrame_Selected",
                            "actionToggle_Framed_View"):
            action = getattr(ui, action_name, None)
            if action is not None and action.isEnabled():
                action.trigger()
                pump(4)

        viewer_mode_action = getattr(ui, "actionToggle_Viewer_Mode", None)
        if viewer_mode_action is not None and viewer_mode_action.isEnabled():
            was_viewer_mode = ctrl.isViewerMode()
            viewer_mode_action.trigger()
            pump(4)
            check(ctrl.isViewerMode() != was_viewer_mode,
                  "Toggle Viewer-Only Mode did not change viewer mode")
            viewer_mode_action.trigger()
            pump(4)
            check(ctrl.isViewerMode() == was_viewer_mode,
                  "Toggle Viewer-Only Mode did not restore viewer mode")

        original_get_color = QtWidgets.QColorDialog.getColor

        def _qa_get_mask_color(*_args, **_kwargs):
            return QtGui.QColor(51, 102, 153, 204)

        QtWidgets.QColorDialog.getColor = _qa_get_mask_color
        try:
            mask_color_action = getattr(ui, "actionCameraMask_Color", None)
            if mask_color_action is not None and mask_color_action.isEnabled():
                mask_color_action.trigger()
                pump(2)
                check(tuple(round(v, 3) for v in
                            ctrl._dataModel.viewSettings.cameraMaskColor) ==
                      (0.2, 0.4, 0.6, 0.8),
                      "Camera mask color action did not update settings")

            def _qa_get_reticle_color(*_args, **_kwargs):
                return QtGui.QColor(204, 153, 102, 51)

            QtWidgets.QColorDialog.getColor = _qa_get_reticle_color
            reticle_color_action = getattr(
                ui, "actionCameraReticles_Color", None)
            if (reticle_color_action is not None
                    and reticle_color_action.isEnabled()):
                reticle_color_action.trigger()
                pump(2)
                check(tuple(round(v, 3) for v in
                            ctrl._dataModel.viewSettings.cameraReticlesColor) ==
                      (0.8, 0.6, 0.4, 0.2),
                      "Camera reticle color action did not update settings")
        finally:
            QtWidgets.QColorDialog.getColor = original_get_color

        other_aov = getattr(ui, "aovOtherAction", None)
        if other_aov is not None:
            check(not other_aov.isEnabled(),
                  "Custom AOV action should be disabled for nanousd backends")

        interpreter_action = getattr(ui, "showInterpreter", None)
        if interpreter_action is not None and interpreter_action.isEnabled():
            interpreter_action.trigger()
            pump(4)
            interpreter = getattr(ctrl, "_interpreter", None)
            check(interpreter is not None and interpreter.isVisible(),
                  "Interpreter action did not show interpreter")
            interpreter.close()
            pump(2)

        validation_action = getattr(ui, "showUsdValidation", None)
        if validation_action is not None and validation_action.isEnabled():
            validation_action.trigger()
            pump(4)
            validation = getattr(ctrl, "_usdValidationWidget", None)
            check(validation is not None and validation.isVisible(),
                  "USD Validation action did not show validation widget")
            validation.close()
            pump(2)

        trigger("actionCopy_Viewer_Image")
        clipboard_image = QtWidgets.QApplication.clipboard().image()
        check(not clipboard_image.isNull(), "Copy Viewer Image produced null image")

        original_get_save_file_name = QtWidgets.QFileDialog.getSaveFileName
        tmp_dir = Path(tempfile.mkdtemp(prefix="nanousdview_qa_"))

        def run_save_action(action_name: str, output_name: str) -> Path:
            output_path = tmp_dir / output_name
            try:
                output_path.unlink()
            except FileNotFoundError:
                pass

            def _qa_save_file_name(*_args, **_kwargs):
                return (str(output_path), "")

            QtWidgets.QFileDialog.getSaveFileName = _qa_save_file_name
            try:
                trigger(action_name)
            finally:
                QtWidgets.QFileDialog.getSaveFileName = original_get_save_file_name
            pump(4)
            check(output_path.is_file(), f"{action_name} did not write {output_path}")
            check(output_path.stat().st_size > 0,
                  f"{action_name} wrote empty file {output_path}")
            return output_path

        run_save_action("actionSave_Viewer_Image",
                        "nanousdview_qa_viewer_image.png")
        run_save_action("actionSave_Flattened_As",
                        "nanousdview_qa_flattened.usda")
        run_save_action("actionSave_Overrides_As",
                        "nanousdview_qa_overrides.usda")

        state_save_action = getattr(ui, "actionSave_State_To", None)
        if (state_save_action is not None and state_save_action.isVisible()
                and state_save_action.isEnabled()):
            state_save_action.trigger()
            pump(4)

            state_save_new = getattr(
                ui, "actionSave_State_As_New_Config", None)
            if (state_save_new is not None and state_save_new.isVisible()
                    and state_save_new.isEnabled()):
                state_save_new.trigger()
                pump(2)
                dialogs = [
                    w for w in app.topLevelWidgets()
                    if isinstance(w, QtWidgets.QDialog)
                    and w.windowTitle() == "Save State As"
                ]
                check(bool(dialogs), "Save State As dialog did not open")
                state_dialog = dialogs[-1]
                field = state_dialog.findChild(QtWidgets.QLineEdit)
                check(field is not None, "Save State As dialog missing field")
                field.setText("nanousdview_qa")
                button_box = state_dialog.findChild(QtWidgets.QDialogButtonBox)
                check(button_box is not None,
                      "Save State As dialog missing buttons")
                save_button = button_box.button(
                    QtWidgets.QDialogButtonBox.StandardButton.Save)
                check(save_button is not None,
                      "Save State As dialog missing Save button")
                save_button.click()
                pump(4)
                check(not state_dialog.isVisible(),
                      "Save State As dialog stayed visible after Save")

                config_names = {
                    action.text()
                    for menu_name in ("menuLoad_New_State",
                                      "menuSave_State_As")
                    for action in getattr(ui, menu_name).actions()
                }
                check("nanousdview_qa" in config_names,
                      "Saved state config did not appear in state menus")

        free_cam_action = trigger("actionAdjust_Free_Camera")
        free_cam_dialog = getattr(ctrl, "_adjustFreeCameraDlg", None)
        check(free_cam_dialog is not None and free_cam_dialog.isVisible(),
              "Free Camera Settings did not open")
        free_cam_ui = getattr(free_cam_dialog, "_ui", None)
        check(free_cam_ui is not None, "Free Camera Settings missing UI")
        free_cam_ui.overrideNear.setChecked(True)
        free_cam_ui.nearSpinBox.setValue(0.25)
        free_cam_ui.overrideFar.setChecked(True)
        free_cam_ui.farSpinBox.setValue(5000.0)
        free_cam_ui.lockFreeCamAspect.setChecked(True)
        free_cam_ui.freeCamAspect.setValue(1.777)
        free_cam_ui.freeCamFov.setValue(45.0)
        pump(4)
        free_cam_dialog.close()
        pump(4)
        check(not free_cam_action.isChecked(),
              "Free Camera Settings action stayed checked after close")

        material_action = trigger("actionAdjust_Default_Material")
        material_dialog = getattr(ctrl, "_adjustDefaultMaterialDlg", None)
        check(material_dialog is not None and material_dialog.isVisible(),
              "Default Material Settings did not open")
        material_ui = getattr(material_dialog, "_ui", None)
        check(material_ui is not None, "Default Material Settings missing UI")
        material_ui.ambientIntSpinBox.setValue(0.4)
        material_ui.specularIntSpinBox.setValue(0.6)
        material_ui.resetButton.click()
        material_ui.doneButton.click()
        pump(4)
        check(not material_action.isChecked(),
              "Default Material Settings action stayed checked after Done")

        prefs_action = trigger("actionPreferences")
        prefs_dialog = getattr(ctrl, "_preferencesDlg", None)
        check(prefs_dialog is not None and prefs_dialog.isVisible(),
              "Preferences did not open")
        prefs_ui = getattr(prefs_dialog, "_ui", None)
        check(prefs_ui is not None, "Preferences missing UI")
        current_font_size = prefs_ui.fontSizeSpinBox.value()
        prefs_ui.fontSizeSpinBox.setValue(current_font_size + 1)
        apply_button = prefs_ui.buttonBox.button(
            QtWidgets.QDialogButtonBox.StandardButton.Apply)
        check(apply_button is not None, "Preferences missing Apply button")
        apply_button.click()
        pump(2)
        ok_button = prefs_ui.buttonBox.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok)
        check(ok_button is not None, "Preferences missing OK button")
        ok_button.click()
        pump(4)
        check(not prefs_action.isChecked(),
              "Preferences action stayed checked after OK")
    finally:
        sys.excepthook = original_excepthook


def main(argv: list[str] | None = None) -> int:
    _install_stack_dump_signal()
    _raise_file_descriptor_limit()

    p = argparse.ArgumentParser(prog="nanousdview")
    p.add_argument("usd_path", nargs="?", default=None,
                   help="USD file to open (.usd, .usda, .usdc, .usdz)")
    p.add_argument("--backend", choices=("vulkan", "opengl", "metal", "ovrtx"),
                   default=None,
                   help="Rendering backend. Linux default: vulkan. "
                        "macOS default: metal. opengl is the cross-platform "
                        "raster fallback (Linux: GL ES 3.2; macOS: GL 4.1). "
                        "ovrtx routes through the NVIDIA OmniRTX runtime "
                        "(Linux only; requires `ovrtx` pip wheel).")
    p.add_argument("--screenshot", metavar="OUTFILE", default=None,
                   help="Non-interactive: render one frame, save PPM to "
                        "OUTFILE, exit. For CI / pixel verification across "
                        "backends without needing keyboard input.")
    p.add_argument("--width", type=int, default=None,
                   help="Window/viewport width (used with --screenshot).")
    p.add_argument("--height", type=int, default=None,
                   help="Window/viewport height (used with --screenshot).")
    p.add_argument("--camera", default="",
                   help="Initial camera prim path or camera name.")
    p.add_argument("--frame", "--cf", dest="currentframe", type=float,
                   default=None,
                   help="Initial frame/time code to render.")
    p.add_argument("--ff", dest="firstframe", type=float, default=None,
                   help="First frame for the viewer timeline.")
    p.add_argument("--lf", dest="lastframe", type=float, default=None,
                   help="Last frame for the viewer timeline.")
    p.add_argument("--render-mode", choices=("rt", "raytrace", "raytraced",
                                             "shadow", "raster"),
                   default=None,
                   help="Initial render mode. Vulkan supports rt, shadow, "
                        "and raster; OpenGL uses raster.")
    p.add_argument("--aov", choices=("color", "depth", "normals", "segmentation"),
                   default="color",
                   help="Renderer AOV to capture with --screenshot. Only color "
                        "is currently supported by the nanousdview OVRTX path.")
    p.add_argument("--envmap", default=None,
                   help="HDR environment map for deterministic captures. "
                        "Currently unsupported by the nanousdview OVRTX path.")
    p.add_argument("--envmap-intensity", type=float, default=None,
                   help="Environment intensity multiplier for --envmap.")
    p.add_argument("--capture-window", metavar="OUTFILE", default=None,
                   help="Show the full nanousdview window for "
                        "--capture-delay seconds, then save a PNG of the "
                        "main window (chrome + side panels + viewport) to "
                        "OUTFILE and exit. For QA — proves the interactive "
                        "GUI renders correctly, not just headless pixels.")
    p.add_argument("--qa-interactions", action="store_true",
                   help="With --capture-window, exercise resize, camera drag, "
                        "prim search, and property search before capture.")
    p.add_argument("--qa-camera-controls", action="store_true",
                   help="Exercise viewport camera controls, verify WASD motion "
                        "stops after key release, then restore the view.")
    p.add_argument("--qa-report", metavar="OUTFILE", default=None,
                   help="Write a JSON QA report with camera state and sampled "
                        "viewport metrics. Useful with --capture-window and "
                        "--qa-camera-controls.")
    p.add_argument("--capture-delay", type=float, default=4.0,
                   help="Seconds to display the window before --capture-window.")
    p.add_argument("--config", default="",
                   help="Load a saved nanousdview state config.")
    p.add_argument("--clearsettings", action="store_true",
                   help="Reset saved nanousdview settings before launch.")
    p.add_argument("--defaultsettings", action="store_true",
                   help="Launch with default settings without saving state.")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if args.clearsettings and args.defaultsettings:
        print("nanousdview: cannot supply both --clearsettings and "
              "--defaultsettings.", file=sys.stderr)
        return 2
    if args.config and args.defaultsettings:
        print("nanousdview: cannot supply both --config and "
              "--defaultsettings.", file=sys.stderr)
        return 2

    # Reject missing input file early — AppController doesn't surface a
    # clean error (raises mid-construct) and the user gets a stack trace
    # that buries the actual problem.
    if args.usd_path and not Path(args.usd_path).is_file():
        print(f"nanousdview: USD file not found: {args.usd_path}",
              file=sys.stderr)
        return 2

    # --backend takes precedence over environment selection. Set the viewer
    # env so downstream imports of nanousdview._backend.get_backend() see it.
    if args.backend:
        os.environ["NANOUSD_VIEW_BACKEND"] = args.backend
    if args.aov != "color":
        print(f"nanousdview: AOV {args.aov!r} unsupported by the current "
              "OVRTX viewport path", file=sys.stderr)
        return 2
    if args.envmap or args.envmap_intensity is not None:
        print("nanousdview: --envmap is unsupported by the current OVRTX "
              "viewport path", file=sys.stderr)
        return 2
    if args.render_mode:
        render_mode = (
            "rt" if args.render_mode in ("raytrace", "raytraced")
            else args.render_mode
        )
        os.environ["NANOUSD_VIEW_RENDER_MODE"] = render_mode

    # Ensure the nanousd C API is loadable. Prefer standalone nanousd
    # libraries; if this tree only has renderer-exported symbols, use the
    # selected renderer library so RTLD_GLOBAL cannot bind another backend.
    workspace = Path(__file__).resolve().parent.parent.parent.parent
    if "NANOUSD_LIB" not in os.environ and "AIUSD_LIB_PATH" not in os.environ:
        selected_backend = (
            args.backend
            or os.environ.get("NANOUSD_VIEW_BACKEND")
            or os.environ.get("NANOUSD_OVRTX_BACKEND")
            or ("metal" if sys.platform == "darwin" else "vulkan")
        ).strip().lower()
        selected_backend = {
            "gles": "opengl",
            "opengles": "opengl",
            "opengl-es": "opengl",
            "opengl_es": "opengl",
        }.get(selected_backend, selected_backend)
        renderer_api_candidates = {
            "vulkan": (
                workspace / "nanousd-vulkan-renderer" / "build" / "libnusd_renderer.so",
                workspace / "nanousd-vulkan-renderer" / "build" / "libnusd_renderer.dylib",
            ),
            "opengl": (
                workspace / "nanousd-opengl-renderer" / "build" / "libnusd_renderer_opengl.so",
                workspace / "nanousd-opengl-renderer" / "build" / "libnusd_renderer_opengl.dylib",
            ),
            "metal": (
                workspace / "nanousd-metal-renderer" / "build" / "libnusd_renderer.dylib",
                workspace / "nanousd-metal-renderer" / "build" / "libnusd_renderer.so",
            ),
        }.get(selected_backend, ())
        # Per-platform default: macOS ships .dylib, Linux ships .so.
        for cand in (
            workspace / ".local" / "lib" / "libnanousdapi.dylib",
            workspace / ".local" / "lib" / "libnanousdapi.so",
            workspace / "nanousd" / "build" / "libnanousdapi.dylib",
            workspace / "nanousd" / "build" / "Release" / "libnanousdapi.dylib",
            workspace / "nanousd" / "build" / "libnanousdapi.so",
            workspace / "nanousd" / "build" / "Release" / "libnanousdapi.so",
            workspace / "nanousd" / "_install" / "release" / "lib" / "libnanousdapi.so",
            workspace / "nanousd" / "_build" / "release" / "Release" / "libnanousdapi.so",
            *renderer_api_candidates,
            workspace / ".local" / "lib" / "libnanousd.dylib",
            workspace / ".local" / "lib" / "libnanousd.so",
            workspace / "nanousd" / "build" / "libnanousd.dylib",
            workspace / "nanousd" / "build" / "Release" / "libnanousd.dylib",
            workspace / "nanousd" / "build" / "libnanousd.so",
            workspace / "nanousd" / "build" / "Release" / "libnanousd.so",
            workspace / "nanousd" / "_install" / "release" / "lib" / "libnanousd.so",
            workspace / "nanousd" / "_build" / "release" / "Release" / "libnanousd.so",
        ):
            if cand.exists():
                os.environ["NANOUSD_LIB"] = str(cand)
                os.environ.setdefault("AIUSD_LIB_PATH", str(cand))
                break

    # Configure the selected OVRTX implementation. For nanousd backends this
    # means putting the nanousd OVRTX facade on sys.path and selecting its
    # concrete implementation via NANOUSD_OVRTX_BACKEND.
    try:
        from nanousdview._backend import (
            configure_backend as _configure_backend,
            validate_renderer as _validate_renderer,
        )
        _chosen = _configure_backend(args.backend, workspace)
        _validate_renderer()
    except Exception as _e:  # noqa: BLE001
        _chosen = args.backend or os.environ.get("NANOUSD_VIEW_BACKEND") or "default"
        print(f"nanousdview: backend {_chosen!r} unavailable: {_e}",
              file=sys.stderr)
        return 2

    # Load nanousd-python BEFORE importing usdviewq; importing it registers
    # the synthetic `pxr` modules used throughout the vendored frontend.
    _pxr_compat = _install_nanousd_python_api()

    # Pull Qt via usdviewq's qt.py module, which decides which binding to load.
    from nanousdview.usdviewq.qt import QtWidgets, QtCore, QtGui  # noqa: F401

    # Now import the vendored Qt frontend.
    from nanousdview.usdviewq.appController import AppController
    from nanousdview.usdviewq.settings import ConfigManager
    # appController.py expects an argparse-like Namespace, not raw argv.
    # Build the minimum viable launch params.

    # Need pxr.Sdf to make camera a real Sdf.Path
    from pxr import Sdf
    if args.config:
        configs = ConfigManager(AppController._outputBaseDirectory()).getConfigs()
        if args.config not in configs:
            print(f"nanousdview: unknown config {args.config!r}",
                  file=sys.stderr)
            return 2
    # Build a permissive parserData stand-in. Any field usdview's
    # AppController reads but we don't define returns None.
    default_settings = bool(
        args.defaultsettings
        or ((args.screenshot is not None or args.capture_window is not None)
            and not args.config and not args.clearsettings)
    )

    explicit = {
        "usdFile": args.usd_path,
        "primPath": "/",
        "camera": Sdf.Path(args.camera or ""),
        "renderer": "GL", "rendererPlugin": "GL",
        "imageType": "png",
        "clearSettings": args.clearsettings, "defaultSettings": default_settings,
        # complexity must be a RefinementComplexity object, not a string —
        # usdview's _refreshComplexityMenu reads `complexity.name`.
        "complexity": _pxr_compat._RefinementComplexities.MEDIUM,
        "defaultLight": True,
        "norender": False, "noRender": False,
        "unloaded": False,
        "timing": False, "memstats": "none",
        "numThreads": 1, "traceToFile": None, "traceFormat": "chrome",
        "mallocTagStats": "none", "bboxstandin": "include",
        "quitAfterStartup": False,
        "sessionLayer": None, "mute": [], "mask": [],
        "purposes": ["default", "proxy"],
        "cameraLightOnly": False,
        "rendererSettings": None, "layerEditor": None,
        "first": None, "last": None,
        "firstframe": args.firstframe, "lastframe": args.lastframe,
        # AppController's parser-time currentframe validation assumes an
        # authored frame range. Apply --frame after construction so static
        # stages can still be rendered at arbitrary time codes.
        "currentframe": None,
        "config": args.config,
    }

    class _Opts:
        def __init__(self):
            for k, v in explicit.items():
                setattr(self, k, v)
        def __getattr__(self, name):
            return None

    opts = _Opts()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    # AppController takes (parserData, resolverContextFn). Resolver context is
    # an Ar concept we don't need; pass a no-op factory.
    def _resolverContextFn(usdFile):
        return None
    if args.clearsettings:
        AppController.clearSettings()
    ctrl = AppController(opts, _resolverContextFn)

    if args.currentframe is not None:
        from pxr import Usd
        try:
            ctrl._dataModel.currentFrame = Usd.TimeCode(args.currentframe)
            if hasattr(ctrl, "setFrameField"):
                ctrl.setFrameField(args.currentframe)
            stage = _stage_view(ctrl)
            if stage is not None and hasattr(stage, "updateView"):
                stage.updateView()
        except Exception as e:  # noqa: BLE001
            print(f"nanousdview: failed to set frame {args.currentframe}: {e}",
                  file=sys.stderr)
            return 2

    if hasattr(ctrl, "_mainWindow") and ctrl._mainWindow is not None:
        _install_nanousd_menu(ctrl)
        # Default interactive size: 1920x1200 (1.5x the legacy 1280x800).
        # The legacy size was small on modern displays — viewport ended
        # up <800x600 once side panels were laid out. Override via Qt's
        # standard --geometry CLI if needed.
        ctrl._mainWindow.resize(2400, 1400)
        ctrl._mainWindow.show()

    if args.qa_camera_controls and not args.capture_window:
        from nanousdview.usdviewq.qt import QtCore

        qa_report = _qa_base_report(args)
        deadline = QtCore.QElapsedTimer()
        deadline.start()
        delay_ms = int(max(args.capture_delay, 0.5) * 1000)
        while deadline.elapsed() < delay_ms:
            app.processEvents()
        _qa_record_view(ctrl, qa_report, "settled")
        try:
            _qa_assert_view_has_content(qa_report, "settled")
            _run_camera_qa_controls(ctrl, app, qa_report, restore_view=True)
            _qa_record_view(ctrl, qa_report, "after_camera_controls")
        except Exception as e:  # noqa: BLE001
            qa_report["error"] = str(e)
            _qa_write_report(args.qa_report, qa_report)
            print(f"[--qa-camera-controls] failed: {e}", file=sys.stderr)
            return 1
        _qa_write_report(args.qa_report, qa_report)
        _close_stage_renderer(ctrl)
        app.quit()
        return 0

    # Non-interactive window-capture path: show the full window (chrome,
    # docks, viewport) for a few seconds so the renderer settles, then
    # grab a PNG of the entire main window via QScreen.grabWindow and
    # exit. Useful for QA verification that the interactive GUI renders.
    if args.capture_window:
        from nanousdview.usdviewq.qt import QtCore
        qa_report = _qa_base_report(args)
        # Suppress the selection-bbox wireframe overlay during headless
        # captures: a default pseudo-root selection puts a yellow box
        # over the rendered pixels in vulkan/opengl (where stageView's
        # QPainter overlay shows through) but is occluded by ovrtx's
        # full-frame path-traced output, breaking apples-to-apples.
        os.environ["NUVIEW_NO_SEL_HIGHLIGHT"] = "1"
        if ctrl._mainWindow is not None:
            ctrl._mainWindow.resize(2400, 1400)
            ctrl._mainWindow.raise_()
            ctrl._mainWindow.activateWindow()
        # Lock the central stageView widget to a deterministic size so the
        # captured render aspect ratio is the same regardless of the
        # backend's per-process side-panel layout finalization timing —
        # without this, vulkan + opengl ended up at 1017x605 while ovrtx
        # ended up at 853x444, which made each renderer produce a
        # different aspect-ratio projection and shifted the framing of
        # the same scene.
        try:
            from nanousdview.usdviewq.qt import QtWidgets  # noqa: F401
            sv = getattr(ctrl, "_stageView", None)
            if sv is None:
                ui = getattr(ctrl, "_ui", None)
                if ui is not None:
                    sv = getattr(ui, "stageView", None)
            if sv is not None:
                sv.setFixedSize(1440, 810)  # 16:9, 50% bump from 960x540
        except Exception:
            pass
        # Pump events long enough for shaders to compile, IBL to load,
        # auto-frame to settle. Use a real timer so we don't spin.
        deadline = QtCore.QElapsedTimer()
        deadline.start()
        delay_ms = int(max(args.capture_delay, 0.5) * 1000)
        while deadline.elapsed() < delay_ms:
            app.processEvents()
        if args.qa_report or args.qa_camera_controls:
            _qa_record_view(ctrl, qa_report, "settled")
        if args.qa_camera_controls:
            try:
                _qa_assert_view_has_content(qa_report, "settled")
                _run_camera_qa_controls(ctrl, app, qa_report, restore_view=True)
                _qa_record_view(ctrl, qa_report, "after_camera_controls")
            except Exception as e:  # noqa: BLE001
                qa_report["error"] = str(e)
                _qa_write_report(args.qa_report, qa_report)
                print(f"[--qa-camera-controls] failed: {e}", file=sys.stderr)
                return 1
        if args.qa_interactions:
            try:
                _run_ui_qa_interactions(ctrl, app, qa_report)
            except Exception as e:  # noqa: BLE001
                qa_report["error"] = str(e)
                _qa_write_report(args.qa_report, qa_report)
                print(f"[--qa-interactions] failed: {e}", file=sys.stderr)
                return 1
        if ctrl._mainWindow is not None:
            # Use the widget's `grab()` rather than `screen.grabWindow(wid)`
            # so the capture isn't clipped by the physical screen size —
            # bumping the viewport to 1440x810 takes the outer window past
            # 1680x1050 (this workstation's :1 virtual display), and the
            # screen path returned a black-banded PNG with the bottom
            # ~366 rows clipped. widget.grab() renders to a QPixmap matched
            # to the widget's current size regardless of where it lands on
            # screen.
            capture_window_path = Path(args.capture_window).expanduser()
            capture_window_path.parent.mkdir(parents=True, exist_ok=True)
            pix = ctrl._mainWindow.grab()
            if pix.save(str(capture_window_path)):
                qa_report["capture_window_written"] = str(capture_window_path)
                qa_report["capture_window_size"] = {
                    "width": int(pix.width()),
                    "height": int(pix.height()),
                }
                print(f"[--capture-window] wrote {capture_window_path}")
            else:
                print(f"[--capture-window] save failed for {args.capture_window}",
                      file=sys.stderr)
                return 1
        if args.qa_report or args.qa_camera_controls:
            _qa_record_view(ctrl, qa_report, "captured")
            _qa_write_report(args.qa_report, qa_report)
        _close_stage_renderer(ctrl)
        app.quit()
        return 0

    # Non-interactive screenshot path: pump the event loop a couple of
    # times so the stageView gets a paint event and the renderer is
    # constructed at the right size, then ask stageView for its current
    # framebuffer and dump it to a PPM. Exits cleanly afterwards.
    if args.screenshot:
        try:
            sw = args.width or 800
            sh = args.height or 600
            # Locate the stageView widget — it's where the renderer lives.
            stage = getattr(ctrl, "_stageView", None)
            if stage is None:
                stage = getattr(ctrl, "_ui", None)
                if stage is not None:
                    stage = getattr(stage, "stageView", None)
            # Resize the *stageView widget* to the requested dimensions
            # so the renderer initializes at the right size. Also frame
            # the scene so far-away camera defaults don't put geometry
            # off-screen (Kitchen_set + similar large scenes).
            #
            # Use setFixedSize (== setMinimumSize + setMaximumSize) so Qt's
            # layout doesn't grow the widget when the parent window's side
            # panels claim more space than the requested width — at small
            # --width values (≤ ~600), setMinimumSize alone let the docks'
            # min widths bump the central stage up by ~57px.
            if stage is not None:
                stage.setFixedSize(sw, sh)
                tick = getattr(stage, "_tick", None)
                if tick is not None:
                    try:
                        tick.stop()
                    except Exception:
                        pass
            if ctrl._mainWindow is not None:
                ctrl._mainWindow.resize(sw + 280, sh + 80)  # add room for side panels
            # Pump first so widgets get sized + first paintEvent constructs
            # the renderer + scene bounds become available.
            for _ in range(4):
                app.processEvents()
            # Force-frame the scene from scene bounds. stageView's
            # _auto_frame runs once on first paint, but for screenshot
            # mode we re-call it here unconditionally so far-from-origin
            # scenes (e.g. Kitchen_set) frame correctly.
            if stage is not None:
                stage._auto_framed_once = False
                skip_frame = os.environ.get("NUSD_VIEW_SKIP_INITIAL_FRAME", "").lower() in (
                    "1", "true", "yes", "on"
                )
                if not skip_frame and hasattr(stage, "_auto_frame"):
                    try: stage._auto_frame()
                    except Exception as e:  # noqa: BLE001
                        print(f"[--screenshot] _auto_frame failed: {e}",
                              file=sys.stderr)
                if getattr(stage, "_frame_pixels", None) is None:
                    stage.update()  # force a repaint only if no frame exists
            # Pump again so the framing + paint settle.
            for _ in range(1):
                app.processEvents()
            ok = False
            if stage is not None:
                renderer = getattr(stage, "_renderer", None)
                if renderer is not None:
                    try:
                        # Prefer the pixels stageView's paintEvent already
                        # produced. Fall back to one explicit OVRTX step only
                        # if no paint has produced a frame yet, or if the
                        # first paint happened before the screenshot size was
                        # locked in.
                        pix = getattr(stage, "_frame_pixels", None)
                        if pix is None or pix.shape[:2] != (sh, sw):
                            if hasattr(renderer, "set_size"):
                                renderer.set_size(sw, sh)
                            mode = getattr(stage, "_render_mode", None)
                            pix = renderer.render_ldr(render_mode=mode) \
                                if mode is not None else renderer.render_ldr()
                        h, w = pix.shape[:2]
                        screenshot_path = Path(args.screenshot).expanduser()
                        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(screenshot_path, "wb") as f:
                            f.write(f"P6\n{w} {h}\n255\n".encode())
                            f.write(pix[:, :, :3].astype("uint8").tobytes())
                        print(f"[--screenshot] wrote {screenshot_path} ({w}x{h})")
                        ok = True
                    except Exception as e:  # noqa: BLE001
                        print(f"[--screenshot] screenshot failed: {e}",
                              file=sys.stderr)
            if not ok:
                print("[--screenshot] could not retrieve framebuffer "
                      "(renderer not constructed?)", file=sys.stderr)
                return 1
        finally:
            _close_stage_renderer(ctrl)
            app.quit()
        return 0

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

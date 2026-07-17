# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blender add-on entry point for the OVRTX render example.

The module is importable in plain Python. Blender-specific classes are defined
only when ``bpy`` is available.
"""

from pathlib import Path
import os
import queue
import threading
from typing import Any

from . import bundled_runtime, ovrtx_gpu_lease, runtime_services, user_messages
from .preflight import (
    PreflightCheck,
    check_addon_prerequisites,
    ensure_native_client_path,
    preflight_summary,
)
from .runtime_manifest import RuntimeManifestError, load_manifest_pin
from .runtime_materializer import (
    RuntimeMaterializerCancelled,
    RuntimeMaterializerError,
    materialize_runtime,
    runtime_source_uses_network,
)
from .runtime_store import remove_runtime, status as runtime_store_status
from .runtime_warmup import warm_shader_cache

bl_info = {
    "name": "ovrtx Blender Example",
    "author": "OVRTX Blender Example contributors",
    "version": (0, 1, 0),
    "blender": (5, 1, 0),
    "location": "Render Properties",
    "description": "ovrtx render shell",
    "category": "Render",
}

try:
    import bpy  # type: ignore
    from bpy.props import BoolProperty, StringProperty  # type: ignore
except ModuleNotFoundError:
    bpy = None  # type: ignore[assignment]
    BoolProperty = None  # type: ignore[assignment]
    StringProperty = None  # type: ignore[assignment]


BLENDER_AVAILABLE = bpy is not None
_PACKAGE_LEAF = "ovrtx_blender_example"
_RUNTIME_INSTALL_STATE: dict[str, object] = {
    "active": False,
    "operation": "",
    "progress": 0.0,
    "message": "",
    "error": "",
}
_RUNTIME_INSTALL_CANCEL: threading.Event | None = None


def _runtime_install_error_lines(message: str) -> list[str]:
    return [line for line in message.splitlines() if line] or [message]


def _set_runtime_install_state(**values: object) -> None:
    _RUNTIME_INSTALL_STATE.update(values)


def runtime_install_status() -> dict[str, object]:
    return dict(_RUNTIME_INSTALL_STATE)


def _addon_preferences_id() -> str:
    package = __package__ or _PACKAGE_LEAF
    extension_subpackage_suffix = f".{_PACKAGE_LEAF}.{_PACKAGE_LEAF}"
    if package.endswith(extension_subpackage_suffix):
        return package[: -len(f".{_PACKAGE_LEAF}")]
    return package


ADDON_PREFERENCES_ID = _addon_preferences_id()


def addon_build_sha(root: Path | None = None) -> str:
    path = (root or Path(__file__).resolve().parents[1]) / "build-sha"
    return path.read_text(encoding="utf-8").strip()[:8] if path.is_file() else ""


def addon_build_generation(root: Path | None = None) -> str:
    path = (root or Path(__file__).resolve().parents[1]) / "build-generation"
    value = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
    return value if value.isdecimal() and int(value) > 0 else ""


def addon_build_label(root: Path | None = None) -> str:
    build_sha = addon_build_sha(root)
    if build_sha and (build_generation := addon_build_generation(root)):
        return f"Build {build_generation} ({build_sha})"
    return f"Build SHA: {build_sha}" if build_sha else ""


def _require_blender() -> Any:
    if bpy is None:
        raise RuntimeError("ovrtx_blender_example registration requires Blender's bpy module")
    return bpy


def _is_class_registered(cls: type[Any]) -> bool:
    return bool(getattr(cls, "is_registered", False))


def _register_class_once(cls: type[Any]) -> None:
    _bpy = _require_blender()
    if _is_class_registered(cls):
        return
    # A different Python class owning the same RNA identifier is not equivalent
    # to this class being registered. Hiding that error leaves Blender pointing
    # at the stale class and later produces "unable to get Python class".
    _bpy.utils.register_class(cls)


def _unregister_class_once(cls: type[Any]) -> None:
    _bpy = _require_blender()
    try:
        _bpy.utils.unregister_class(cls)
    except (RuntimeError, ValueError) as exc:
        if "missing bl_rna" not in str(exc) and "not registered" not in str(exc):
            raise


def get_addon_preferences(context: Any | None = None) -> Any | None:
    """Return add-on preferences when running inside Blender."""

    if bpy is None:
        return None
    context = context or bpy.context
    addon = context.preferences.addons.get(ADDON_PREFERENCES_ID)
    return addon.preferences if addon else None


def _online_access_state(context: Any) -> tuple[bool, bool]:
    enabled = bool(getattr(bpy.app, "online_access", False))
    try:
        locked = bool(context.preferences.system.is_property_readonly("use_online_access"))
    except Exception:
        locked = bool(getattr(bpy.app, "online_access_override", False))
    return enabled, locked


def preflight_preferences(preferences: Any) -> dict[str, object]:
    runtime = runtime_bundle_status()
    runtime_root = Path(str(runtime["current_root"])) if runtime.get("state") == "ready" else None
    checks = check_addon_prerequisites(
        worker_command=str(getattr(preferences, "worker_command", "") or ""),
        native_client_path=str(getattr(preferences, "native_client_path", "") or ""),
        native_client_module=str(
            getattr(preferences, "native_client_module", "")
            or bundled_runtime.DEFAULT_OVRTX_NATIVE_CLIENT_MODULE
        ),
        runtime_root=runtime_root,
    )
    checks.append(_gpu_lease_check())
    if runtime.get("state") == "ready":
        services = runtime_services.owner.diagnostics()
        checks.append(
            PreflightCheck(
                "runtime_services",
                "Runtime services",
                services["status"] == "ready",
                (
                    "SERVING"
                    if services["status"] == "ready"
                    else str(services["error"] or services["status"])
                ),
            )
        )
    summary = preflight_summary(checks)
    summary["runtime"] = runtime
    return summary


def _gpu_lease_check() -> PreflightCheck:
    if runtime_services.owner.diagnostics()["status"] in {"starting", "ready"}:
        return PreflightCheck(
            "ovrtx_gpu_lease", "OVRTX GPU lease", True, "held by this Blender"
        )
    status = ovrtx_gpu_lease.probe()
    owner = status.get("owner") if isinstance(status.get("owner"), dict) else {}
    if status["status"] == "busy" and owner.get("pid") == os.getpid():
        return PreflightCheck(
            "ovrtx_gpu_lease", "OVRTX GPU lease", True, "held by this Blender"
        )
    return PreflightCheck(
        "ovrtx_gpu_lease",
        "OVRTX GPU lease",
        status["status"] == "available",
        "available" if status["status"] == "available" else str(status.get("error", status["status"])),
    )


def runtime_bundle_status(storage_root: Path | None = None) -> dict[str, object]:
    try:
        manifest_sha256 = load_manifest_pin(bundled_runtime.addon_root())
    except RuntimeManifestError as exc:
        return {
            "state": "missing_manifest",
            "current_root": "",
            "message": str(exc),
            "manifest_sha256": "",
        }
    try:
        root = storage_root or _extension_storage_root(create=False)
    except RuntimeError as exc:
        return {
            "state": "unavailable",
            "current_root": "",
            "message": str(exc),
            "manifest_sha256": manifest_sha256,
        }
    platform_id = bundled_runtime.current_platform_id()
    runtime_status = runtime_store_status(root, platform_id, manifest_sha256)
    return {
        "state": runtime_status.state,
        "current_root": str(runtime_status.current_root),
        "message": runtime_status.message,
        "manifest_sha256": manifest_sha256,
        "installed_manifest_sha256": runtime_status.installed_manifest_sha256,
    }


_RUNTIME_GENERATION = 0
_RUNTIME_SERVICE_THREAD: threading.Thread | None = None


def runtime_generation() -> int:
    """Counter bumped whenever the materialized runtime changes (install/retry/
    remove), so hot-path callers can cache resolved runtime paths and recheck with
    a cheap integer compare instead of re-reading the manifest pin."""
    return _RUNTIME_GENERATION


def _bump_runtime_generation() -> None:
    global _RUNTIME_GENERATION
    _RUNTIME_GENERATION += 1


def start_runtime_services_async() -> bool:
    """Start a verified installed runtime after Blender file readiness."""

    global _RUNTIME_SERVICE_THREAD
    runtime = runtime_bundle_status()
    if runtime.get("state") != "ready":
        return False
    if _RUNTIME_SERVICE_THREAD is not None and _RUNTIME_SERVICE_THREAD.is_alive():
        return True
    root = Path(str(runtime["current_root"]))
    _RUNTIME_SERVICE_THREAD = threading.Thread(
        target=_start_runtime_services,
        args=(root,),
        name="ovrtx-runtime-services",
        daemon=True,
    )
    _RUNTIME_SERVICE_THREAD.start()
    return True


def _start_runtime_services(root: Path) -> None:
    try:
        runtime_services.owner.start(root)
    except RuntimeError:
        pass


def _extension_storage_root(*, create: bool) -> Path:
    _bpy = _require_blender()
    try:
        value = _bpy.utils.extension_path_user(ADDON_PREFERENCES_ID, path="", create=create)
    except Exception as exc:
        raise RuntimeError(f"Could not resolve extension user storage: {exc}") from exc
    return Path(value).expanduser().resolve()


def _load_runtime_pin_and_storage(*, create: bool) -> tuple[str, Path]:
    manifest_sha256 = load_manifest_pin(bundled_runtime.addon_root())
    storage_root = _extension_storage_root(create=create)
    return manifest_sha256, storage_root


def _apply_runtime_defaults(preferences: Any, root: Path | None = None) -> None:
    defaults = bundled_runtime.defaults(root=root)
    preferences.worker_command = defaults.worker_command
    preferences.native_client_path = defaults.native_client_path
    preferences.native_client_module = bundled_runtime.DEFAULT_OVRTX_NATIVE_CLIENT_MODULE


def _populate_runtime_preferences() -> None:
    preferences = get_addon_preferences()
    if preferences is None:
        return
    preferences.runtime_source = ""
    runtime = runtime_bundle_status()
    if runtime.get("state") != "ready":
        return
    defaults = bundled_runtime.defaults(root=Path(str(runtime["current_root"])))
    if not str(getattr(preferences, "worker_command", "") or ""):
        preferences.worker_command = defaults.worker_command
    if not str(getattr(preferences, "native_client_path", "") or ""):
        preferences.native_client_path = defaults.native_client_path
    if not str(getattr(preferences, "native_client_module", "") or ""):
        preferences.native_client_module = bundled_runtime.DEFAULT_OVRTX_NATIVE_CLIENT_MODULE


def write_viewport_session_outputs() -> int:
    """Write viewport session outputs for active render-engine instances."""

    from .engine import write_viewport_session_outputs as write

    return write()


if bpy is not None:

    class _RuntimeInstallMixin:
        _enable_online_access: bool = False
        _events: Any = None
        _thread: Any = None
        _timer: Any = None
        _cancel_event: threading.Event | None = None

        @classmethod
        def poll(cls, context: Any) -> bool:
            preferences = get_addon_preferences(context)
            return not bool(_RUNTIME_INSTALL_STATE["active"]) and (
                runtime_bundle_status().get("state") != "ready"
            ) and bool(str(getattr(preferences, "runtime_source", "") or "").strip())

        def invoke(self, context: Any, event: Any) -> set[str]:
            preferences = get_addon_preferences(context)
            source = str(getattr(preferences, "runtime_source", "") or "")
            if not source.strip():
                return self._cancel_with_error(context, "Install Runtime From is empty")
            if not runtime_source_uses_network(source):
                return self.execute(context)
            enabled, locked = _online_access_state(context)
            if enabled:
                return self.execute(context)
            if locked:
                return self._cancel_with_error(
                    context,
                    "Blender is in offline mode. Restart without --offline-mode to install the runtime.",
                )
            self._enable_online_access = True
            return context.window_manager.invoke_confirm(
                self,
                event,
                title="Enable Online Access",
                message="Enable Blender online access to download and install the runtime?",
                confirm_text="Enable and Install",
            )

        def execute(self, context: Any) -> set[str]:
            preferences = get_addon_preferences(context)
            if preferences is None:
                return self._cancel_with_error(context, "Add-on preferences are unavailable")
            source = str(getattr(preferences, "runtime_source", "") or "")
            if not source.strip():
                return self._cancel_with_error(context, "Install Runtime From is empty")
            if runtime_bundle_status().get("state") == "ready":
                return self._cancel_with_error(
                    context,
                    "Remove the installed runtime before installing from another location.",
                )
            if getattr(self, "_enable_online_access", False):
                try:
                    context.preferences.system.use_online_access = True
                except Exception as exc:
                    return self._cancel_with_error(
                        context,
                        f"Could not enable online access: {exc}",
                    )
            if runtime_source_uses_network(source) and not bool(
                getattr(bpy.app, "online_access", False)
            ):
                return self._cancel_with_error(
                    context,
                    "Blender online access is disabled.",
                )
            try:
                manifest_sha256, storage_root = _load_runtime_pin_and_storage(create=True)
            except (RuntimeManifestError, RuntimeMaterializerError, RuntimeError, OSError) as exc:
                return self._cancel_with_error(context, str(exc))

            global _RUNTIME_INSTALL_CANCEL
            self._events = queue.SimpleQueue()
            self._cancel_event = threading.Event()
            _RUNTIME_INSTALL_CANCEL = self._cancel_event
            self._thread = threading.Thread(
                target=self._materialize,
                args=(
                    manifest_sha256,
                    storage_root,
                    source,
                ),
                name="ovrtx-runtime-install",
                daemon=True,
            )
            _set_runtime_install_state(
                active=True,
                operation="installing",
                progress=0.0,
                message="Preparing runtime installation",
                error="",
            )
            try:
                self._timer = context.window_manager.event_timer_add(
                    0.1,
                    window=context.window,
                )
                context.window_manager.modal_handler_add(self)
                self._thread.start()
            except Exception as exc:
                self._finish(context)
                return self._cancel_with_error(context, str(exc))
            self._tag_redraw(context)
            return {"RUNNING_MODAL"}

        def _materialize(
            self,
            manifest_sha256: str,
            storage_root: Path,
            source: str,
        ) -> None:
            current_root: Path | None = None
            try:
                current_root = materialize_runtime(
                    manifest_sha256,
                    storage_root,
                    source=source,
                    progress=lambda message, completed, total: self._events.put(
                        ("progress", message, completed, total)
                    ),
                    cancelled=self._cancel_event.is_set,
                )
                if self._cancel_event.is_set():
                    remove_runtime(storage_root, bundled_runtime.current_platform_id())
                    raise RuntimeMaterializerCancelled("Runtime installation cancelled")
                serving: list[str] = []

                def report_serving(health: runtime_services.ServiceHealth) -> None:
                    serving.append(health.name)
                    self._events.put(
                        ("progress", f"{health.name} SERVING", len(serving), 2)
                    )

                self._events.put(("progress", "Starting runtime services", 0, 2))
                runtime_services.owner.start(
                    current_root,
                    on_serving=report_serving,
                    cancelled=self._cancel_event.is_set,
                )
                try:
                    warm_shader_cache(
                        current_root,
                        storage_root,
                        progress=lambda message: self._events.put(
                            ("progress", message, 1, 1)
                        ),
                    )
                except Exception as exc:
                    raise RuntimeError(f"Shader cache warmup failed: {exc}") from exc
                if self._cancel_event.is_set():
                    runtime_services.owner.close()
                    remove_runtime(storage_root, bundled_runtime.current_platform_id())
                    raise RuntimeMaterializerCancelled("Runtime installation cancelled")
            except RuntimeMaterializerCancelled:
                runtime_services.owner.cancel()
                if current_root is not None:
                    remove_runtime(storage_root, bundled_runtime.current_platform_id())
                self._events.put(("cancelled", None))
            except Exception as exc:
                runtime_services.owner.cancel()
                if current_root is not None:
                    remove_runtime(storage_root, bundled_runtime.current_platform_id())
                self._events.put(
                    ("cancelled", None)
                    if self._cancel_event.is_set()
                    else ("error", str(exc))
                )
            else:
                self._events.put(("complete", current_root))

        def modal(self, context: Any, event: Any) -> set[str]:
            if event.type != "TIMER":
                return {"PASS_THROUGH"}

            terminal: tuple[str, Any] | None = None
            while True:
                try:
                    update = self._events.get_nowait()
                except queue.Empty:
                    break
                if update[0] == "progress":
                    _kind, message, completed, total = update
                    factor = min(1.0, max(0.0, completed / total)) if total else 0.0
                    _set_runtime_install_state(
                        progress=factor,
                        message=message,
                    )
                elif terminal is None:
                    terminal = (update[0], update[1])

            self._tag_redraw(context)
            if terminal is None:
                return {"PASS_THROUGH"}

            self._finish(context)
            if terminal[0] == "cancelled":
                message = "Runtime installation cancelled."
                _set_runtime_install_state(
                    active=False,
                    operation="",
                    message=message,
                    error="",
                )
                self.report({"INFO"}, message)
                self._tag_redraw(context)
                return {"CANCELLED"}
            if terminal[0] == "error":
                return self._cancel_with_error(context, str(terminal[1]))

            current_root = Path(terminal[1])
            _bump_runtime_generation()
            try:
                preferences = get_addon_preferences(context)
                if preferences is not None:
                    _apply_runtime_defaults(preferences, current_root)
            except Exception as exc:
                return self._cancel_with_error(context, str(exc))
            message = f"Runtime installed: {current_root}"
            _set_runtime_install_state(
                active=False,
                operation="",
                progress=1.0,
                message=message,
                error="",
            )
            self.report({"INFO"}, message)
            self._tag_redraw(context)
            return {"FINISHED"}

        def _finish(self, context: Any) -> None:
            global _RUNTIME_INSTALL_CANCEL
            if self._timer is not None:
                context.window_manager.event_timer_remove(self._timer)
                self._timer = None
            _RUNTIME_INSTALL_CANCEL = None

        def _cancel_with_error(self, context: Any, detail: str) -> set[str]:
            message = str(_RUNTIME_INSTALL_STATE["error"] or "")
            first_error = not message
            if not message:
                message = detail
            _set_runtime_install_state(
                active=False,
                operation="",
                message=message,
                error=message,
            )
            if first_error:
                user_messages.mirror_to_console(user_messages.ERROR, message)
            self._tag_redraw(context)
            return {"CANCELLED"}

        @staticmethod
        def _tag_redraw(context: Any) -> None:
            area = getattr(context, "area", None)
            if area is not None:
                area.tag_redraw()

    class OvrtxExampleInstallRuntime(_RuntimeInstallMixin, bpy.types.Operator):  # type: ignore[misc]
        bl_idname = "ovrtx_example.install_runtime"
        bl_label = "Install Runtime"


    class OvrtxExampleCancelRuntimeInstall(bpy.types.Operator):  # type: ignore[misc]
        bl_idname = "ovrtx_example.cancel_runtime_install"
        bl_label = "Cancel"

        @classmethod
        def poll(cls, context: Any) -> bool:
            return bool(_RUNTIME_INSTALL_STATE["active"]) and (
                _RUNTIME_INSTALL_STATE["operation"] == "installing"
            )

        def execute(self, context: Any) -> set[str]:
            if _RUNTIME_INSTALL_CANCEL is None:
                return {"CANCELLED"}
            _RUNTIME_INSTALL_CANCEL.set()
            runtime_services.owner.cancel()
            _set_runtime_install_state(message="Cancelling runtime installation")
            area = getattr(context, "area", None)
            if area is not None:
                area.tag_redraw()
            return {"FINISHED"}


    class OvrtxExampleRemoveRuntime(bpy.types.Operator):  # type: ignore[misc]
        bl_idname = "ovrtx_example.remove_runtime"
        bl_label = "Remove Runtime"

        _events: Any = None
        _thread: Any = None
        _timer: Any = None

        def execute(self, context: Any) -> set[str]:
            _set_runtime_install_state(
                active=True,
                operation="removing",
                progress=0.0,
                message="Preparing runtime removal",
                error="",
            )
            try:
                from . import scene_generation_sessions
                from .engine import stop_viewport_sessions_for_unregister

                if not stop_viewport_sessions_for_unregister():
                    raise RuntimeError(
                        "viewport teardown deadline exceeded; restart Blender before removing the runtime"
                    )
                scene_generation_sessions.close()
                runtime_services.owner.close()
                _manifest_sha256, storage_root = _load_runtime_pin_and_storage(create=True)
            except (RuntimeManifestError, RuntimeError, OSError) as exc:
                return self._cancel_with_error(context, str(exc))

            self._events = queue.SimpleQueue()
            self._thread = threading.Thread(
                target=self._remove,
                args=(storage_root, bundled_runtime.current_platform_id()),
                name="ovrtx-runtime-remove",
                daemon=True,
            )
            try:
                self._timer = context.window_manager.event_timer_add(
                    0.1,
                    window=context.window,
                )
                context.window_manager.modal_handler_add(self)
                self._thread.start()
            except Exception as exc:
                self._finish(context)
                return self._cancel_with_error(context, str(exc))
            self._tag_redraw(context)
            return {"RUNNING_MODAL"}

        def _remove(self, storage_root: Path, platform: str) -> None:
            try:
                self._events.put(("progress", "Removing runtime files", 0.25))
                remove_runtime(storage_root, platform)
            except Exception as exc:
                self._events.put(("error", str(exc)))
            else:
                self._events.put(("complete", "Runtime removed."))

        def modal(self, context: Any, event: Any) -> set[str]:
            if event.type != "TIMER":
                return {"PASS_THROUGH"}

            terminal: tuple[str, str] | None = None
            while True:
                try:
                    update = self._events.get_nowait()
                except queue.Empty:
                    break
                if update[0] == "progress":
                    _set_runtime_install_state(progress=update[2], message=update[1])
                else:
                    terminal = (update[0], update[1])

            self._tag_redraw(context)
            if terminal is None:
                return {"PASS_THROUGH"}

            self._finish(context)
            if terminal[0] == "error":
                return self._cancel_with_error(context, terminal[1])
            _bump_runtime_generation()
            preferences = get_addon_preferences(context)
            if preferences is not None:
                preferences.worker_command = ""
                preferences.native_client_path = ""
                preferences.native_client_module = ""
            _set_runtime_install_state(
                active=False,
                operation="",
                progress=1.0,
                message=terminal[1],
                error="",
            )
            user_messages.report_for_operator(self, {"INFO"}, terminal[1])
            self._tag_redraw(context)
            return {"FINISHED"}

        def _finish(self, context: Any) -> None:
            if self._timer is not None:
                context.window_manager.event_timer_remove(self._timer)
                self._timer = None

        def _cancel_with_error(self, context: Any, detail: str) -> set[str]:
            message = f"Runtime remove failed: {detail}"
            _set_runtime_install_state(
                active=False,
                operation="",
                message=message,
                error=message,
            )
            user_messages.report_for_operator(self, {"ERROR"}, message)
            self._tag_redraw(context)
            return {"CANCELLED"}

        @staticmethod
        def _tag_redraw(context: Any) -> None:
            area = getattr(context, "area", None)
            if area is not None:
                area.tag_redraw()


    class OvrtxExampleResetRuntimeOverrides(bpy.types.Operator):  # type: ignore[misc]
        bl_idname = "ovrtx_example.reset_runtime_overrides"
        bl_label = "Reset to Installed Runtime"

        def execute(self, context: Any) -> set[str]:
            preferences = get_addon_preferences(context)
            if preferences is None:
                return {"CANCELLED"}
            runtime = runtime_bundle_status()
            root = (
                Path(str(runtime["current_root"]))
                if runtime.get("state") == "ready"
                else None
            )
            _apply_runtime_defaults(preferences, root)
            return {"FINISHED"}

    class OvrtxExamplePreferences(bpy.types.AddonPreferences):  # type: ignore[misc]
        bl_idname = ADDON_PREFERENCES_ID

        show_advanced: BoolProperty(  # type: ignore[valid-type]
            name="Advanced Runtime Overrides",
            default=False,
        )
        worker_command: StringProperty(  # type: ignore[valid-type]
            name="Worker Command",
            default="",
        )
        native_client_path: StringProperty(  # type: ignore[valid-type]
            name="Native Client Path",
            subtype="DIR_PATH",
            default="",
        )
        native_client_module: StringProperty(  # type: ignore[valid-type]
            name="Native Client Module",
            default=bundled_runtime.DEFAULT_OVRTX_NATIVE_CLIENT_MODULE,
        )
        runtime_source: StringProperty(  # type: ignore[valid-type]
            name="Install Runtime From",
            subtype="DIR_PATH",
            options={"SKIP_SAVE"},
            default="",
        )

        def draw(self, context: Any) -> None:
            layout = self.layout
            if build_label := addon_build_label():
                layout.label(text=build_label)
            status = preflight_preferences(self)
            runtime = status["runtime"]
            install = runtime_install_status()
            runtime_box = layout.box()
            runtime_state = install["operation"] if install["active"] else runtime["state"]
            runtime_box.label(text=f"Runtime: {runtime_state}")
            if install["active"]:
                runtime_box.progress(
                    factor=float(install["progress"]),
                    text=str(install["message"]),
                )
            else:
                runtime_box.label(text=str(runtime["message"]))
                if install["error"]:
                    for index, line in enumerate(
                        _runtime_install_error_lines(str(install["error"]))
                    ):
                        runtime_box.label(
                            text=line,
                            **({"icon": "ERROR"} if index == 0 else {}),
                        )
            runtime_box.prop(self, "runtime_source")
            actions = runtime_box.row(align=True)
            install_action = actions.row(align=True)
            install_action.enabled = (
                not bool(install["active"])
                and runtime["state"] != "ready"
                and bool(str(self.runtime_source).strip())
            )
            install_action.operator("ovrtx_example.install_runtime")
            if install["active"] and install["operation"] == "installing":
                actions.operator("ovrtx_example.cancel_runtime_install", text="Cancel")
            else:
                remove_action = actions.row(align=True)
                remove_action.enabled = not bool(install["active"])
                remove_action.operator("ovrtx_example.remove_runtime")
            advanced = layout.box()
            advanced.prop(self, "show_advanced")
            if self.show_advanced:
                advanced.prop(self, "worker_command")
                advanced.prop(self, "native_client_path")
                advanced.prop(self, "native_client_module")
                advanced.operator("ovrtx_example.reset_runtime_overrides")
            box = layout.box()
            box.label(text=f"Preflight: {status['status']}")
            for check in status["checks"]:
                icon = "CHECKMARK" if check["ok"] else "ERROR"
                box.label(text=f"{check['label']}: {check['message']}", icon=icon)

else:
    OvrtxExampleInstallRuntime = None  # type: ignore[assignment]
    OvrtxExampleCancelRuntimeInstall = None  # type: ignore[assignment]
    OvrtxExampleRemoveRuntime = None  # type: ignore[assignment]
    OvrtxExampleResetRuntimeOverrides = None  # type: ignore[assignment]
    OvrtxExamplePreferences = None  # type: ignore[assignment]


def register() -> None:
    """Register the Blender add-on classes."""

    _require_blender()
    from . import (
        authoring_properties,
        properties,
        scene_generation_sessions,
        ui,
        worker_process_reaper,
    )
    from .engine import (
        OvrtxExampleRenderEngine,
        register_final_render_handlers,
        register_interactive_edit_bridge,
    )
    from .viewport_presentation import register_viewport_presentation_monitor

    # Exit-time backstop for the ovphysx-bridge-server orphan: Blender does not
    # reliably run unregister() on quit, so this atexit hook hard-kills any of
    # our still-running worker processes. Registered once; removed in
    # unregister() so a reload cannot leak or double-register it.
    worker_process_reaper.install()
    user_messages.register(bpy)
    _register_class_once(OvrtxExampleInstallRuntime)
    _register_class_once(OvrtxExampleCancelRuntimeInstall)
    _register_class_once(OvrtxExampleRemoveRuntime)
    _register_class_once(OvrtxExampleResetRuntimeOverrides)
    _register_class_once(OvrtxExamplePreferences)
    _populate_runtime_preferences()
    properties.register()
    authoring_properties.register()
    scene_generation_sessions.register_handlers(bpy)
    _register_class_once(OvrtxExampleRenderEngine)
    register_interactive_edit_bridge()
    register_final_render_handlers()
    register_viewport_presentation_monitor(bpy)
    ui.register()
    # After the engine class is registered: join the stock property panels
    # (spec task03-01). Idempotent set-add, so add-on reload is safe.
    ui.register_stock_panel_compat()
    timers = getattr(getattr(bpy, "app", None), "timers", None)
    if timers is not None:
        timers.register(lambda: (start_runtime_services_async(), None)[1], first_interval=0.0)
    scene_generation_sessions.schedule_initial_generation(bpy)


def unregister() -> None:
    """Unregister the Blender add-on classes in reverse order."""

    _require_blender()
    from . import (
        authoring_properties,
        properties,
        scene_generation_sessions,
        ui,
        worker_process_reaper,
    )
    from .engine import (
        OvrtxExampleRenderEngine,
        stop_viewport_sessions_for_unregister,
        unregister_final_render_handlers,
        unregister_interactive_edit_bridge,
    )
    from .viewport_presentation import unregister_viewport_presentation_monitor

    # Symmetric discard first, restoring stock panel visibility rules exactly
    # (idempotent, so a partial earlier unregister is safe).
    ui.unregister_stock_panel_compat()
    ui.unregister()
    unregister_viewport_presentation_monitor(bpy)
    unregister_final_render_handlers()
    unregister_interactive_edit_bridge()
    viewport_stopped = stop_viewport_sessions_for_unregister()
    scene_generation_sessions.unregister_handlers(bpy)
    if _RUNTIME_INSTALL_CANCEL is not None:
        _RUNTIME_INSTALL_CANCEL.set()
    if viewport_stopped:
        runtime_services.owner.close()
    # Belt-and-suspenders: the disable path has the same orphan gap as quit
    # (owner.close() is skipped when viewport teardown is unconfirmed, and the
    # ovphysx worker can outlive a graceful shutdown). Hard-kill any surviving
    # worker children, then drop the exit hook so a re-enable re-installs a
    # fresh one instead of leaking this registration.
    worker_process_reaper.reap()
    worker_process_reaper.uninstall()
    _unregister_class_once(OvrtxExampleRenderEngine)
    authoring_properties.unregister()
    properties.unregister()
    _unregister_class_once(OvrtxExamplePreferences)
    _unregister_class_once(OvrtxExampleResetRuntimeOverrides)
    _unregister_class_once(OvrtxExampleRemoveRuntime)
    _unregister_class_once(OvrtxExampleCancelRuntimeInstall)
    _unregister_class_once(OvrtxExampleInstallRuntime)
    user_messages.unregister(bpy)


__all__ = [
    "BLENDER_AVAILABLE",
    "ADDON_PREFERENCES_ID",
    "OvrtxExampleCancelRuntimeInstall",
    "OvrtxExampleInstallRuntime",
    "OvrtxExamplePreferences",
    "OvrtxExampleRemoveRuntime",
    "OvrtxExampleResetRuntimeOverrides",
    "PreflightCheck",
    "bl_info",
    "check_addon_prerequisites",
    "ensure_native_client_path",
    "write_viewport_session_outputs",
    "get_addon_preferences",
    "preflight_preferences",
    "preflight_summary",
    "register",
    "runtime_bundle_status",
    "runtime_generation",
    "runtime_install_status",
    "start_runtime_services_async",
    "unregister",
]

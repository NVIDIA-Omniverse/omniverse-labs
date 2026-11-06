# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blender add-on entry point for the OVRTX render example.

The module is importable in plain Python. Blender-specific classes are defined
only when ``bpy`` is available.
"""
from pathlib import Path
import json
import os
import queue
import threading
import sys
from typing import Any
from . import bundled_runtime, performance_profile, user_messages
from .preflight import PreflightCheck, check_addon_prerequisites, preflight_summary
bl_info = {'name': 'ovrtx Blender Example', 'author': 'OVRTX Blender Example contributors', 'version': (0, 1, 0), 'blender': (5, 1, 0), 'location': 'Render Properties', 'description': 'ovrtx render shell', 'category': 'Render'}
try:
    import bpy
    from bpy.props import BoolProperty, StringProperty
except ModuleNotFoundError:
    bpy = None
    BoolProperty = None
    StringProperty = None
BLENDER_AVAILABLE = bpy is not None

def _exit_pre(_unused: Any=None) -> None:
    """Close owner threads and native resources while Blender data is valid."""
    from . import scene_generation_sessions, worker_process_reaper, render_sequence
    render_sequence.shutdown()
    from .engine import stop_viewport_sessions_for_unregister
    from . import public_sdk_install
    public_sdk_install.close()
    stopped = stop_viewport_sessions_for_unregister()
    scene_generation_sessions.close()
    worker_process_reaper.reap()
try:
    from bpy.app.handlers import persistent as _persistent
except ImportError:
    pass
else:
    _exit_pre = _persistent(_exit_pre)
_PACKAGE_LEAF = 'ovrtx_blender_example'
_RUNTIME_INSTALL_STATE: dict[str, object] = {'active': False, 'operation': '', 'progress': 0.0, 'message': '', 'error': ''}
_PUBLIC_SDK_SETUP_ERROR = ''
_PUBLIC_SDK_LAST_STATE = ''

def _runtime_install_error_lines(message: str) -> list[str]:
    return [line for line in message.splitlines() if line] or [message]

def _set_runtime_install_state(**values: object) -> None:
    _RUNTIME_INSTALL_STATE.update(values)

def runtime_install_status() -> dict[str, object]:
    status = _public_sdk_status()
    return {'active': status['state'] in {'installing', 'removing'}, 'operation': status['state'], 'progress': status.get('progress', 0.0), 'message': status.get('message', ''), 'error': status.get('error', '')}

def _public_sdk_status() -> dict[str, object]:
    configured = os.environ.get('OV_BLENDER_PUBLIC_SDK_PYTHON', '').strip()
    if configured:
        ready = Path(configured).expanduser().is_file()
        return {'state': 'ready' if ready else 'failed', 'progress': 1.0 if ready else 0.0, 'message': 'Rendering environment is ready' if ready else 'Configured rendering environment is unavailable', 'error': '' if ready else 'The configured interpreter does not exist'}
    if _PUBLIC_SDK_SETUP_ERROR:
        return {'state': 'failed', 'progress': 0.0, 'message': 'Rendering setup needs attention', 'error': _PUBLIC_SDK_SETUP_ERROR}
    from . import public_sdk_install
    return public_sdk_install.status()

def _public_sdk_local_package_mode() -> bool:
    package = bundled_runtime.public_sdk_package_root()
    if package is None:
        return False
    try:
        marker = json.loads((package / 'public-sdk.json').read_text())
        return marker.get('install_mode', 'automatic') == 'local_package'
    except (OSError, ValueError):
        return True

def _addon_preferences_id() -> str:
    package = __package__ or _PACKAGE_LEAF
    extension_subpackage_suffix = f'.{_PACKAGE_LEAF}.{_PACKAGE_LEAF}'
    if package.endswith(extension_subpackage_suffix):
        return package[:-len(f'.{_PACKAGE_LEAF}')]
    return package
ADDON_PREFERENCES_ID = _addon_preferences_id()

def addon_build_sha(root: Path | None=None) -> str:
    path = (root or Path(__file__).resolve().parents[1]) / 'build-sha'
    return path.read_text(encoding='utf-8').strip()[:8] if path.is_file() else ''

def addon_build_generation(root: Path | None=None) -> str:
    path = (root or Path(__file__).resolve().parents[1]) / 'build-generation'
    value = path.read_text(encoding='utf-8').strip() if path.is_file() else ''
    return value if value.isdecimal() and int(value) > 0 else ''

def addon_build_label(root: Path | None=None) -> str:
    build_sha = addon_build_sha(root)
    if build_sha and (build_generation := addon_build_generation(root)):
        return f'Build {build_generation} ({build_sha})'
    return f'Build SHA: {build_sha}' if build_sha else ''

def _require_blender() -> Any:
    if bpy is None:
        raise RuntimeError("ovrtx_blender_example registration requires Blender's bpy module")
    return bpy

def _is_class_registered(cls: type[Any]) -> bool:
    return bool(getattr(cls, 'is_registered', False))

def _register_class_once(cls: type[Any]) -> None:
    _bpy = _require_blender()
    if _is_class_registered(cls):
        return
    _bpy.utils.register_class(cls)

def _unregister_class_once(cls: type[Any]) -> None:
    _bpy = _require_blender()
    try:
        _bpy.utils.unregister_class(cls)
    except (RuntimeError, ValueError) as exc:
        if 'missing bl_rna' not in str(exc) and 'not registered' not in str(exc):
            raise

def get_addon_preferences(context: Any | None=None) -> Any | None:
    """Return add-on preferences when running inside Blender."""
    if bpy is None:
        return None
    context = context or bpy.context
    addon = context.preferences.addons.get(ADDON_PREFERENCES_ID)
    return addon.preferences if addon else None

def _online_access_state(context: Any) -> tuple[bool, bool]:
    enabled = bool(getattr(bpy.app, 'online_access', False))
    try:
        locked = bool(context.preferences.system.is_property_readonly('use_online_access'))
    except Exception:
        locked = bool(getattr(bpy.app, 'online_access_override', False))
    return (enabled, locked)

def preflight_preferences(preferences):
    summary = preflight_summary(check_addon_prerequisites())
    summary['runtime'] = runtime_bundle_status()
    return summary

def runtime_bundle_status(storage_root=None):
    package = bundled_runtime.public_sdk_package_root()
    status = _public_sdk_status()
    return {**status, 'current_root': str(package or bundled_runtime.public_sdk_worker_script().parent.parent), 'manifest_sha256': ''}
_RUNTIME_GENERATION = 0

def runtime_generation() -> int:
    """Counter bumped whenever the materialized runtime changes (install/retry/
    remove), so hot-path callers can cache resolved runtime paths and recheck with
    a cheap integer compare instead of re-reading the manifest pin."""
    return _RUNTIME_GENERATION

def _bump_runtime_generation() -> None:
    global _RUNTIME_GENERATION
    _RUNTIME_GENERATION += 1

def _extension_storage_root(*, create: bool) -> Path:
    _bpy = _require_blender()
    try:
        value = _bpy.utils.extension_path_user(ADDON_PREFERENCES_ID, path='', create=create)
    except Exception as exc:
        raise RuntimeError(f'Could not resolve extension user storage: {exc}') from exc
    return Path(value).expanduser().resolve()

def _bundled_python_path() -> Path:
    """Resolve Blender's shipped interpreter on the main thread, not blender.exe."""
    app = _require_blender().app
    version = f'{app.version[0]}.{app.version[1]}'
    directory = Path(app.binary_path).resolve().parent / version / 'python' / 'bin'
    names = ('python.exe',) if sys.platform == 'win32' else (f'python{sys.version_info.major}.{sys.version_info.minor}', 'python3', 'python')
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise RuntimeError("Blender's bundled Python is missing; reinstall this Blender version before retrying setup")

def _public_sdk_storage_root() -> Path:
    from .public_sdk_sidecar import runtime_storage_root
    # Blender resolves the normal profile even when a desktop launch has no
    # BLENDER_USER_CONFIG override. Use the same profile-owned location that
    # the paired installer selects with its explicit launch environment.
    config_root = _require_blender().utils.user_resource('CONFIG')
    if not config_root:
        raise RuntimeError('Could not resolve Blender profile configuration directory')
    return runtime_storage_root(_extension_storage_root(create=True), config_root=config_root, local_appdata=os.environ.get('LOCALAPPDATA'))

def _public_sdk_setup_poll() -> float | None:
    global _PUBLIC_SDK_LAST_STATE
    state = str(_public_sdk_status()['state'])
    if state != _PUBLIC_SDK_LAST_STATE:
        _PUBLIC_SDK_LAST_STATE = state
        _bump_runtime_generation()
    for window in getattr(getattr(bpy.context, 'window_manager', None), 'windows', ()):
        for area in getattr(window.screen, 'areas', ()):
            area.tag_redraw()
    return 0.5 if state in {'installing', 'removing'} else None

def _close_public_sdk_runtime_owners() -> None:
    """Main-thread owner shutdown before changing the selected installation."""
    from . import engine, public_sdk_install, scene_generation_sessions
    if public_sdk_install.status()['state'] in {'installing', 'removing'}:
        raise RuntimeError('Runtime setup or removal is still active')
    if engine._FINAL_RENDER_ACTIVE:
        raise RuntimeError('Finish or cancel the final render before removing its runtime')
    if not engine.stop_viewport_sessions_for_unregister():
        raise RuntimeError('Viewport teardown was not confirmed; restart Blender before removal')
    scene_generation_sessions.close()
    if scene_generation_sessions.close_status().get('status') != 'closed':
        raise RuntimeError('Scene runtime teardown was not confirmed; restart Blender before removal')

def _reset_public_sdk_override() -> None:
    _close_public_sdk_runtime_owners()
    os.environ.pop('OV_BLENDER_PUBLIC_SDK_PYTHON', None)
    _start_public_sdk_setup(configure_only=True)
    _bump_runtime_generation()

def _remove_public_sdk_runtime() -> None:
    """Close owners then remove only the addon-managed packaged candidate."""
    from . import public_sdk_install
    if os.environ.get('OV_BLENDER_PUBLIC_SDK_PYTHON', '').strip():
        raise RuntimeError('An external interpreter is selected; its files are not managed by this add-on')
    _close_public_sdk_runtime_owners()
    public_sdk_install.remove()
    timers = getattr(bpy.app, 'timers', None)
    if timers is not None and (not timers.is_registered(_public_sdk_setup_poll)):
        timers.register(_public_sdk_setup_poll, first_interval=0.1)

def _start_public_sdk_setup(context: Any | None=None, *, source: str | None=None, configure_only: bool=False) -> bool:
    """Configure setup; local packages wait for Install. Never start a worker."""
    global _PUBLIC_SDK_SETUP_ERROR
    package = bundled_runtime.public_sdk_package_root()
    if package is None or os.environ.get('OV_BLENDER_PUBLIC_SDK_PYTHON', '').strip():
        return False
    _PUBLIC_SDK_SETUP_ERROR = ''
    try:
        marker = json.loads((package / 'public-sdk.json').read_text())
        if marker.get('schema') != 1 or marker.get('backend') != 'public_sdk':
            raise RuntimeError('Unsupported rendering package; reinstall the add-on ZIP')
        from . import public_sdk_install
        public_sdk_install.configure(_public_sdk_storage_root(), _bundled_python_path(), package / 'runtime' / 'requirements-public-sdk.txt')
        if configure_only:
            started = False
        elif source is not None or marker.get('install_mode', 'automatic') == 'automatic':
            started = public_sdk_install.start(online_access=bool(getattr(bpy.app, 'online_access', False)), source=source or "")
        elif marker.get('install_mode') == 'local_package':
            started = False
        else:
            raise RuntimeError('Unsupported rendering setup mode; reinstall the add-on ZIP')
    except (OSError, ValueError, RuntimeError) as error:
        _PUBLIC_SDK_SETUP_ERROR = str(error)
        started = False
    timers = getattr(bpy.app, 'timers', None)
    if timers is not None and (not timers.is_registered(_public_sdk_setup_poll)):
        timers.register(_public_sdk_setup_poll, first_interval=0.1)
    return started

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
            return _public_sdk_status()['state'] not in {'installing', 'ready'} and (not _public_sdk_local_package_mode() or bool(str(getattr(preferences, 'runtime_source', '')).strip()))

        def invoke(self, context: Any, event: Any) -> set[str]:
            preferences = get_addon_preferences(context)
            source = str(getattr(preferences, 'runtime_source', ''))
            local_package = _public_sdk_local_package_mode()
            if not source.strip() and local_package:
                return self._cancel_with_error(context, 'Install Runtime From is empty')
            if local_package:
                return self.execute(context)
            enabled, locked = _online_access_state(context)
            if enabled:
                return self.execute(context)
            if locked:
                return self._cancel_with_error(context, 'Blender is in offline mode. Restart without --offline-mode to install the runtime.')
            self._enable_online_access = True
            return context.window_manager.invoke_confirm(self, event, title='Enable Online Access', message='Enable Blender online access to download and install the runtime?', confirm_text='Enable and Install')

        def execute(self, context: Any) -> set[str]:
            preferences = get_addon_preferences(context)
            if preferences is None:
                return self._cancel_with_error(context, 'Add-on preferences are unavailable')
            source = str(getattr(preferences, 'runtime_source', ''))
            local_package = _public_sdk_local_package_mode()
            if not source.strip() and local_package:
                return self._cancel_with_error(context, 'Install Runtime From is empty')
            if runtime_bundle_status().get('state') == 'ready':
                return self._cancel_with_error(context, 'Remove the installed runtime before installing from another location.')
            if getattr(self, '_enable_online_access', False):
                try:
                    context.preferences.system.use_online_access = True
                except Exception as exc:
                    return self._cancel_with_error(context, f'Could not enable online access: {exc}')
            if not local_package and (not bool(getattr(bpy.app, 'online_access', False))):
                return self._cancel_with_error(context, 'Blender online access is disabled.')
            started = _start_public_sdk_setup(context, source=source if local_package else '')
            if not started and _public_sdk_status()['state'] not in {'installing', 'ready'}:
                status = _public_sdk_status()
                return self._cancel_with_error(context, str(status.get("error") or status.get("message") or "Setup could not start"))
            return {'FINISHED'}

        def _cancel_with_error(self, context: Any, detail: str) -> set[str]:
            message = str(_RUNTIME_INSTALL_STATE['error'])
            first_error = not message
            if not message:
                message = detail
            _set_runtime_install_state(active=False, operation='', message=message, error=message)
            if first_error:
                user_messages.mirror_to_console(user_messages.ERROR, message)
            self._tag_redraw(context)
            return {'CANCELLED'}

        @staticmethod
        def _tag_redraw(context: Any) -> None:
            area = getattr(context, 'area', None)
            if area is not None:
                area.tag_redraw()

    class OvrtxExampleInstallRuntime(_RuntimeInstallMixin, bpy.types.Operator):
        bl_idname = 'ovrtx_example.install_runtime'
        bl_label = 'Install Runtime'
        bl_description = 'Install and verify the runtime from the selected release or local folder'

    class OvrtxExampleCancelRuntimeInstall(bpy.types.Operator):
        bl_idname = 'ovrtx_example.cancel_runtime_install'
        bl_label = 'Cancel'
        bl_description = 'Cancel the current runtime installation'

        @classmethod
        def poll(cls, context: Any) -> bool:
            return _public_sdk_status()['state'] == 'installing'

        def execute(self, context: Any) -> set[str]:
            from . import public_sdk_install
            public_sdk_install.cancel()
            return {'FINISHED'}

    class OvrtxExampleRemoveRuntime(bpy.types.Operator):
        bl_idname = 'ovrtx_example.remove_runtime'
        bl_label = 'Remove Runtime'
        bl_description = 'Remove the installed OVRTX runtime from extension storage'
        _events: Any = None
        _thread: Any = None
        _timer: Any = None

        def execute(self, context: Any) -> set[str]:
            try:
                _remove_public_sdk_runtime()
            except (RuntimeError, ValueError, OSError) as exc:
                return self._cancel_with_error(context, str(exc))
            self._tag_redraw(context)
            return {'FINISHED'}

        def _cancel_with_error(self, context: Any, detail: str) -> set[str]:
            message = f'Runtime remove failed: {detail}'
            _set_runtime_install_state(active=False, operation='', message=message, error=message)
            user_messages.report_for_operator(self, {'ERROR'}, message)
            self._tag_redraw(context)
            return {'CANCELLED'}

        @staticmethod
        def _tag_redraw(context: Any) -> None:
            area = getattr(context, 'area', None)
            if area is not None:
                area.tag_redraw()

    class OvrtxExampleResetRuntimeOverrides(bpy.types.Operator):
        bl_idname = 'ovrtx_example.reset_runtime_overrides'
        bl_label = 'Reset to Installed Runtime'
        bl_description = 'Clear advanced overrides and use the verified installed runtime'

        def execute(self, context: Any) -> set[str]:
            try:
                _reset_public_sdk_override()
            except (RuntimeError, ValueError, OSError) as exc:
                user_messages.report_for_operator(self, {'ERROR'}, str(exc))
                return {'CANCELLED'}
            return {'FINISHED'}

    class OvrtxExamplePreferences(bpy.types.AddonPreferences):
        bl_idname = ADDON_PREFERENCES_ID
        show_advanced: BoolProperty(name='Advanced Runtime Overrides', default=False)
        runtime_source: StringProperty(name='Install Runtime From', subtype='FILE_PATH', options={'SKIP_SAVE'}, default='')

        def draw(self, context: Any) -> None:
            layout = self.layout
            if (build_label := addon_build_label()):
                layout.label(text=build_label)
            setup = _public_sdk_status()
            box = layout.box()
            box.label(text='Rendering setup')
            local_package = _public_sdk_local_package_mode()
            if local_package and setup['state'] == 'unconfigured':
                box.label(text='Choose the rendering package, then click Install.')
            else:
                box.label(text=str(setup.get('message') or setup['state']))
            if local_package:
                box.prop(self, 'runtime_source')
            if os.environ.get('OV_BLENDER_PUBLIC_SDK_PYTHON', '').strip():
                box.operator('ovrtx_example.reset_runtime_overrides', text='Use Installed Runtime')
            if setup['state'] == 'installing':
                box.progress(factor=float(setup.get('progress', 0.0)), text='Preparing rendering components')
                box.operator('ovrtx_example.cancel_runtime_install', text='Cancel')
            elif setup['state'] == 'removing':
                box.progress(factor=float(setup.get('progress', 0.0)), text='Removing rendering components')
            elif setup['state'] != 'ready':
                box.operator('ovrtx_example.install_runtime', text='Retry Setup' if setup['state'] in {'failed', 'cancelled'} else 'Install')
            if setup.get('error'):
                for line in _runtime_install_error_lines(str(setup['error'])):
                    box.label(text=line, icon='ERROR')
            if setup['state'] == 'ready':
                box.label(text='Ready. Select OVRTX to render.')
                if local_package and (not os.environ.get('OV_BLENDER_PUBLIC_SDK_PYTHON', '').strip()):
                    box.operator('ovrtx_example.remove_runtime')
            elif setup['state'] == 'offline' and (not local_package):
                box.label(text='Installation needs Blender online access.')
            if setup['state'] in {'unconfigured', 'failed', 'cancelled'} and setup.get('incomplete_root') and (not os.environ.get('OV_BLENDER_PUBLIC_SDK_PYTHON', '').strip()):
                # An unfinished install's mapped runtime folder: always offer a way to delete it.
                box.operator('ovrtx_example.remove_runtime')
            return
else:
    OvrtxExampleInstallRuntime = None
    OvrtxExampleCancelRuntimeInstall = None
    OvrtxExampleRemoveRuntime = None
    OvrtxExampleResetRuntimeOverrides = None
    OvrtxExamplePreferences = None

@performance_profile.timed('addon.register')
def register() -> None:
    """Register the Blender add-on classes."""
    performance_profile.start_main()
    _require_blender()
    from . import authoring_properties, properties, scene_generation_sessions, ui, worker_process_reaper
    from .engine import OvrtxExampleRenderEngine, register_final_render_handlers, register_interactive_edit_bridge
    from .viewport_presentation import register_viewport_presentation_monitor
    worker_process_reaper.install()
    user_messages.register(bpy)
    _register_class_once(OvrtxExampleInstallRuntime)
    _register_class_once(OvrtxExampleCancelRuntimeInstall)
    _register_class_once(OvrtxExampleRemoveRuntime)
    _register_class_once(OvrtxExampleResetRuntimeOverrides)
    _register_class_once(OvrtxExamplePreferences)
    properties.register()
    authoring_properties.register()
    from . import physics_bake_ui
    physics_bake_ui.register()
    scene_generation_sessions.register_handlers(bpy)
    exit_handlers = getattr(bpy.app.handlers, 'exit_pre', None)
    if exit_handlers is not None and _exit_pre not in exit_handlers:
        exit_handlers.append(_exit_pre)
    _register_class_once(OvrtxExampleRenderEngine)
    register_interactive_edit_bridge()
    register_final_render_handlers()
    register_viewport_presentation_monitor(bpy)
    ui.register()
    from . import render_sequence
    render_sequence.register()
    from . import hdr_output
    hdr_output.register()
    ui.register_stock_panel_compat()
    timers = getattr(getattr(bpy, 'app', None), 'timers', None)
    _start_public_sdk_setup()

@performance_profile.timed('addon.unregister')
def unregister() -> None:
    """Unregister the Blender add-on classes in reverse order."""
    _require_blender()
    from . import public_sdk_install
    public_sdk_install.close()
    timers = getattr(bpy.app, 'timers', None)
    if timers is not None and timers.is_registered(_public_sdk_setup_poll):
        timers.unregister(_public_sdk_setup_poll)
    from . import authoring_properties, properties, scene_generation_sessions, ui, worker_process_reaper
    from .engine import OvrtxExampleRenderEngine, stop_viewport_sessions_for_unregister, unregister_final_render_handlers, unregister_interactive_edit_bridge
    from .viewport_presentation import unregister_viewport_presentation_monitor
    ui.unregister_stock_panel_compat()
    from . import render_sequence
    render_sequence.unregister()
    from . import hdr_output
    hdr_output.unregister()
    ui.unregister()
    unregister_viewport_presentation_monitor(bpy)
    unregister_final_render_handlers()
    unregister_interactive_edit_bridge()
    viewport_stopped = stop_viewport_sessions_for_unregister()
    exit_handlers = getattr(bpy.app.handlers, 'exit_pre', None)
    if exit_handlers is not None and _exit_pre in exit_handlers:
        exit_handlers.remove(_exit_pre)
    scene_generation_sessions.unregister_handlers(bpy)
    
    worker_process_reaper.reap()
    worker_process_reaper.uninstall()
    _unregister_class_once(OvrtxExampleRenderEngine)
    from . import physics_bake_ui
    physics_bake_ui.unregister()
    authoring_properties.unregister()
    properties.unregister()
    _unregister_class_once(OvrtxExamplePreferences)
    _unregister_class_once(OvrtxExampleResetRuntimeOverrides)
    _unregister_class_once(OvrtxExampleRemoveRuntime)
    _unregister_class_once(OvrtxExampleCancelRuntimeInstall)
    _unregister_class_once(OvrtxExampleInstallRuntime)
    user_messages.unregister(bpy)
    performance_profile.stop_main()
__all__ = ['BLENDER_AVAILABLE', 'ADDON_PREFERENCES_ID', 'OvrtxExampleCancelRuntimeInstall', 'OvrtxExampleInstallRuntime', 'OvrtxExamplePreferences', 'OvrtxExampleRemoveRuntime', 'OvrtxExampleResetRuntimeOverrides', 'PreflightCheck', 'bl_info', 'check_addon_prerequisites', 'write_viewport_session_outputs', 'get_addon_preferences', 'preflight_preferences', 'preflight_summary', 'register', 'runtime_bundle_status', 'runtime_generation', 'runtime_install_status', 'unregister']

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

import ovrtx_blender_example as addon  # noqa: E402
from ovrtx_blender_example import check_addon_prerequisites, preflight_summary
from ovrtx_blender_example import bundled_runtime
from ovrtx_blender_example import preflight as preflight_module
from ovrtx_blender_example.preflight import _worker_command_parts
from ovrtx_blender_example.runtime_materializer import RuntimeMaterializerError


def test_addon_build_sha_reads_packaged_commit(tmp_path: Path) -> None:
    (tmp_path / "build-sha").write_text("a" * 40 + "\n", encoding="utf-8")

    assert addon.addon_build_sha(tmp_path) == "a" * 8


def test_addon_build_sha_is_empty_when_not_packaged(tmp_path: Path) -> None:
    assert addon.addon_build_sha(tmp_path) == ""


def test_addon_build_label_shows_generation(tmp_path: Path) -> None:
    (tmp_path / "build-sha").write_text("a" * 40 + "\n", encoding="utf-8")
    (tmp_path / "build-generation").write_text("1842\n", encoding="utf-8")

    assert addon.addon_build_label(tmp_path) == f"Build 1842 ({'a' * 8})"


def test_addon_build_label_falls_back_to_sha(tmp_path: Path) -> None:
    (tmp_path / "build-sha").write_text("b" * 40 + "\n", encoding="utf-8")

    assert addon.addon_build_label(tmp_path) == f"Build SHA: {'b' * 8}"


@pytest.fixture(autouse=True)
def _skip_host_driver_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight_module, "_detect_nvidia_driver_version", lambda: "")
    monkeypatch.setattr(
        addon.ovrtx_gpu_lease,
        "probe",
        lambda: {"status": "available", "gpu_id": "GPU-test"},
    )


def test_register_class_once_does_not_hide_stale_rna_registration(monkeypatch) -> None:
    class Candidate:
        is_registered = False

    def reject_stale_registration(_cls) -> None:
        raise ValueError("already registered as a subclass")

    monkeypatch.setattr(addon, "bpy", SimpleNamespace(utils=SimpleNamespace(register_class=reject_stale_registration)))

    try:
        addon._register_class_once(Candidate)
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("stale RNA registration must not be ignored")


def test_windows_worker_command_parts_remove_list2cmdline_quotes() -> None:
    executable = r"C:\ProgramData\Blender Foundation\ovrtx-bridge-server.exe"
    package_root = r"C:\ProgramData\Blender Foundation\runtime\ovrtx-bridge-server"
    command = subprocess.list2cmdline(
        [
            executable,
            "--address",
            "127.0.0.1",
            "--port",
            "50051",
            "--package-root",
            package_root,
        ]
    )

    parts, error = _worker_command_parts(command, windows=True)

    assert error == ""
    assert parts[0] == executable
    assert parts[-1] == package_root


_HOST_PLATFORM_ID = "windows-x64" if os.name == "nt" else "linux-x64"
_HOST_EXECUTABLE_SUFFIX = ".exe" if os.name == "nt" else ""


def _write_bundled_worker(addon_root: Path) -> Path:
    worker = addon_root / "bin" / f"ovrtx-bridge-server{_HOST_EXECUTABLE_SUFFIX}"
    worker.parent.mkdir(parents=True, exist_ok=True)
    worker.write_text("#!/bin/sh\n", encoding="utf-8")
    worker.chmod(0o755)
    return worker


def _make_package_root(package_root: Path) -> Path:
    package_root.mkdir(parents=True, exist_ok=True)
    for marker in preflight_module._PACKAGE_ROOT_MARKERS:
        (package_root / marker).mkdir(exist_ok=True)
    return package_root


def _write_native_client(path: Path, module_name: str = "fake_native_client") -> str:
    module = path / f"{module_name}.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(
        """
def capabilities():
    return {
        "rpcs": ["CreateSimulation", "WriteWorldState", "ReadWorldState"],
        "generic_builders": ["build_WriteWorldState_columns", "build_ReadWorldState_ldr_color"],
    }

def start_worker(_request): return {"status": "ok"}
class Client:
    def CreateSimulation(self, _request): return {"simulation_id": "sim"}
    def WriteWorldState(self, _request): return {"status": "ok"}
    def ReadWorldState(self, _request): return {"response_handle": "response"}
def build_WriteWorldState_columns(_request): return {"handle": "write"}
def build_ReadWorldState_ldr_color(_request): return {"handle": "read"}
def decode_ldr_color_frame(_request, _response): return {"frames": []}
""".lstrip(),
        encoding="utf-8",
    )
    return module_name


def _fake_bpy(*, online_access: bool, locked: bool = False, readonly_raises: bool = False) -> SimpleNamespace:
    def is_property_readonly(_name: str) -> bool:
        if readonly_raises:
            raise RuntimeError("missing bl_rna")
        return locked

    system = SimpleNamespace(is_property_readonly=is_property_readonly)
    return SimpleNamespace(
        app=SimpleNamespace(online_access=online_access, online_access_override=locked),
        context=SimpleNamespace(preferences=SimpleNamespace(system=system)),
    )


def _load_addon_with_fake_bpy(monkeypatch: pytest.MonkeyPatch) -> object:
    package_name = "ovrtx_blender_example_operator_test"
    package_root = ROOT / "addon" / "ovrtx_blender_example"
    fake_bpy = ModuleType("bpy")

    class Operator:
        def report(self, levels, message):
            reports = getattr(self, "reports", [])
            reports.append((set(levels), message))
            self.reports = reports

    fake_bpy.types = SimpleNamespace(Operator=Operator, AddonPreferences=object)
    fake_bpy.app = SimpleNamespace(online_access=True, online_access_override=False)
    props = ModuleType("bpy.props")
    props.BoolProperty = lambda **_kwargs: None
    fake_bpy.string_properties = []
    props.StringProperty = lambda **kwargs: fake_bpy.string_properties.append(kwargs)
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "bpy.props", props)
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package_name, module)
    spec.loader.exec_module(module)
    return module


class _FakeWindowManager:
    def __init__(self) -> None:
        self.timer = object()
        self.timer_removed = False
        self.modal_operator = None
        self.popups: list[tuple[str, str, list[str]]] = []

    def event_timer_add(self, _interval: float, *, window) -> object:
        return self.timer

    def event_timer_remove(self, timer: object) -> None:
        assert timer is self.timer
        self.timer_removed = True

    def modal_handler_add(self, operator) -> None:
        self.modal_operator = operator

    def popup_menu(self, draw, *, title: str, icon: str) -> None:
        labels: list[str] = []
        layout = SimpleNamespace(label=lambda *, text: labels.append(text))
        draw(SimpleNamespace(layout=layout), None)
        self.popups.append((title, icon, labels))


def _operator_context() -> SimpleNamespace:
    area = SimpleNamespace(redraw_count=0)
    area.tag_redraw = lambda: setattr(area, "redraw_count", area.redraw_count + 1)
    return SimpleNamespace(
        area=area,
        preferences=SimpleNamespace(system=SimpleNamespace(use_online_access=True)),
        window=object(),
        window_manager=_FakeWindowManager(),
    )


def test_preflight_reports_foreign_gpu_lease_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(addon, "runtime_bundle_status", lambda: {"state": "missing"})
    monkeypatch.setattr(
        addon.ovrtx_gpu_lease,
        "probe",
        lambda: {
            "status": "busy",
            "error": "OVRTX GPU lease is busy (pid=42, entrypoint=owner)",
        },
    )

    summary = addon.preflight_preferences(SimpleNamespace())

    check = next(item for item in summary["checks"] if item["key"] == "ovrtx_gpu_lease")
    assert not check["ok"]
    assert "pid=42" in check["message"]


def test_runtime_preferences_expose_one_install_source_and_two_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_addon_with_fake_bpy(monkeypatch)
    assert not hasattr(module, "OvrtxExampleRetryRuntime")
    assert not hasattr(module, "OvrtxExampleVerifyRuntime")
    assert {
        "name": "Install Runtime From",
        "subtype": "DIR_PATH",
        "options": {"SKIP_SAVE"},
        "default": "",
    } in module.bpy.string_properties
    source = (ROOT / "addon" / "ovrtx_blender_example" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert 'install_action.operator("ovrtx_example.install_runtime")' in source
    assert 'actions.operator("ovrtx_example.cancel_runtime_install", text="Cancel")' in source
    assert 'remove_action.operator("ovrtx_example.remove_runtime")' in source
    assert "ovrtx_example.verify_runtime" not in source
    assert "ovrtx_example.retry_runtime" not in source


def test_runtime_install_operator_runs_async_and_reports_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_addon_with_fake_bpy(monkeypatch)
    started = threading.Event()
    release = threading.Event()
    preferences = SimpleNamespace(runtime_source=str(tmp_path / "artifact-set"))
    applied: list[Path] = []
    selected_sources: list[str] = []
    warmed: list[tuple[Path, Path]] = []
    warm_started = threading.Event()
    release_warm = threading.Event()

    def materialize(_manifest, _storage_root, *, source=None, progress=None, cancelled=None):
        selected_sources.append(source)
        assert cancelled is not None and not cancelled()
        started.set()
        if progress is not None:
            progress("Downloading test component", 1, 2)
        release.wait(timeout=0.25)
        return tmp_path / "runtime"

    monkeypatch.setattr(module, "_load_runtime_pin_and_storage", lambda **_kwargs: ("a" * 64, tmp_path))
    monkeypatch.setattr(module, "materialize_runtime", materialize)
    def warm_shader_cache(root, storage, *, progress=None):
        warmed.append((root, storage))
        assert progress is not None
        progress("Warming shader cache (can take several minutes) — 0:01 elapsed")
        warm_started.set()
        release_warm.wait(timeout=0.25)

    monkeypatch.setattr(module, "warm_shader_cache", warm_shader_cache)
    monkeypatch.setattr(
        module.runtime_services.owner,
        "start",
        lambda root, **_kwargs: applied.append(Path(root)),
    )
    monkeypatch.setattr(module, "get_addon_preferences", lambda _context=None: preferences)
    monkeypatch.setattr(module, "_apply_runtime_defaults", lambda _preferences, _root: None)
    module.bpy.app.online_access = False
    context = _operator_context()
    operator = module.OvrtxExampleInstallRuntime()

    assert operator.execute(context) == {"RUNNING_MODAL"}
    assert started.wait(timeout=0.1)
    assert module._RUNTIME_INSTALL_STATE["active"] is True

    operator.modal(context, SimpleNamespace(type="TIMER", timer=object()))
    assert module._RUNTIME_INSTALL_STATE["progress"] == 0.5
    assert module._RUNTIME_INSTALL_STATE["message"] == "Downloading test component"
    release.set()
    assert warm_started.wait(timeout=0.2)
    operator.modal(context, SimpleNamespace(type="TIMER", timer=operator._timer))
    assert "0:01 elapsed" in module._RUNTIME_INSTALL_STATE["message"]
    release_warm.set()
    operator._thread.join(timeout=1.0)

    assert operator.modal(context, SimpleNamespace(type="TIMER", timer=operator._timer)) == {"FINISHED"}
    assert applied == [tmp_path / "runtime"]
    assert selected_sources == [str(tmp_path / "artifact-set")]
    assert warmed == [(tmp_path / "runtime", tmp_path)]
    assert module._RUNTIME_INSTALL_STATE["active"] is False
    assert module._RUNTIME_INSTALL_STATE["progress"] == 1.0
    assert context.window_manager.timer_removed
    assert operator.reports[-1][0] == {"INFO"}


def test_runtime_install_operator_elevates_background_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_addon_with_fake_bpy(monkeypatch)

    def materialize(_manifest, _storage_root, *, source=None, progress=None, cancelled=None):
        raise RuntimeMaterializerError("download exploded")

    monkeypatch.setattr(module, "_load_runtime_pin_and_storage", lambda **_kwargs: ("a" * 64, tmp_path))
    monkeypatch.setattr(module, "materialize_runtime", materialize)
    monkeypatch.setattr(
        module,
        "get_addon_preferences",
        lambda _context=None: SimpleNamespace(runtime_source=str(tmp_path)),
    )
    context = _operator_context()
    operator = module.OvrtxExampleInstallRuntime()

    assert operator.execute(context) == {"RUNNING_MODAL"}
    operator._thread.join(timeout=1.0)

    assert operator.modal(context, SimpleNamespace(type="TIMER", timer=operator._timer)) == {"CANCELLED"}
    assert module._RUNTIME_INSTALL_STATE["active"] is False
    assert module._RUNTIME_INSTALL_STATE["error"]
    assert not hasattr(operator, "reports")
    assert context.window_manager.popups == []


def test_runtime_install_error_lines_fit_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_addon_with_fake_bpy(monkeypatch)

    lines = module._runtime_install_error_lines("observation\nexplanation\naction")

    assert lines == ["observation", "explanation", "action"]


def test_runtime_install_keeps_first_terminal_error(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_addon_with_fake_bpy(monkeypatch)
    context = _operator_context()
    operator = module.OvrtxExampleInstallRuntime()
    operator._events = module.queue.SimpleQueue()
    operator._timer = context.window_manager.timer

    class Marker:
        def __init__(self) -> None:
            self.string_calls = 0

        def __str__(self) -> str:
            self.string_calls += 1
            return "failure"

    first = Marker()
    second = Marker()
    operator._events.put(("error", first))
    operator._events.put(("error", second))

    assert operator.modal(context, SimpleNamespace(type="TIMER")) == {"CANCELLED"}
    assert first.string_calls == 1
    assert second.string_calls == 0
    assert context.window_manager.popups == []


def test_runtime_install_removes_promoted_runtime_when_health_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_addon_with_fake_bpy(monkeypatch)
    removed = []
    monkeypatch.setattr(
        module,
        "_load_runtime_pin_and_storage",
        lambda **_kwargs: ("a" * 64, tmp_path),
    )
    monkeypatch.setattr(
        module,
        "materialize_runtime",
        lambda *_args, **_kwargs: tmp_path / "current",
    )
    monkeypatch.setattr(
        module.runtime_services.owner,
        "start",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("health failed")),
    )
    monkeypatch.setattr(
        module,
        "remove_runtime",
        lambda root, platform: removed.append((root, platform)),
    )
    monkeypatch.setattr(
        module,
        "get_addon_preferences",
        lambda _context=None: SimpleNamespace(runtime_source=str(tmp_path)),
    )
    context = _operator_context()
    operator = module.OvrtxExampleInstallRuntime()

    assert operator.execute(context) == {"RUNNING_MODAL"}
    operator._thread.join(timeout=1.0)

    assert operator.modal(context, SimpleNamespace(type="TIMER")) == {"CANCELLED"}
    assert removed == [(tmp_path, bundled_runtime.current_platform_id())]
    assert "health failed" in module._RUNTIME_INSTALL_STATE["error"]


def test_runtime_install_cancel_button_cancels_without_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_addon_with_fake_bpy(monkeypatch)
    started = threading.Event()
    release = threading.Event()

    def materialize(_manifest, _storage_root, *, source=None, progress=None, cancelled=None):
        started.set()
        release.wait(timeout=0.25)
        if cancelled is not None and cancelled():
            raise module.RuntimeMaterializerCancelled("Runtime installation cancelled")
        return tmp_path / "runtime"

    monkeypatch.setattr(
        module,
        "_load_runtime_pin_and_storage",
        lambda **_kwargs: ("a" * 64, tmp_path),
    )
    monkeypatch.setattr(module, "materialize_runtime", materialize)
    monkeypatch.setattr(
        module,
        "get_addon_preferences",
        lambda _context=None: SimpleNamespace(runtime_source=str(tmp_path)),
    )
    context = _operator_context()
    install = module.OvrtxExampleInstallRuntime()

    assert install.execute(context) == {"RUNNING_MODAL"}
    assert started.wait(timeout=0.1)
    cancel = module.OvrtxExampleCancelRuntimeInstall()
    assert cancel.bl_label == "Cancel"
    assert cancel.execute(context) == {"FINISHED"}
    assert module._RUNTIME_INSTALL_STATE["message"] == "Cancelling runtime installation"
    release.set()
    install._thread.join(timeout=1.0)

    assert install.modal(context, SimpleNamespace(type="TIMER")) == {"CANCELLED"}
    assert module._RUNTIME_INSTALL_STATE["active"] is False
    assert module._RUNTIME_INSTALL_STATE["error"] == ""
    assert module._RUNTIME_INSTALL_STATE["message"] == "Runtime installation cancelled."
    assert context.window_manager.popups == []


def test_runtime_remove_operator_reports_progress_and_clears_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_addon_with_fake_bpy(monkeypatch)
    started = threading.Event()
    release = threading.Event()
    removed: list[tuple[Path, str]] = []
    preferences = SimpleNamespace(
        worker_command="installed-worker",
        native_client_path="installed-native",
        native_client_module=bundled_runtime.DEFAULT_OVRTX_NATIVE_CLIENT_MODULE,
    )

    def remove(storage_root: Path, platform: str) -> None:
        started.set()
        release.wait(timeout=0.25)
        removed.append((storage_root, platform))

    sessions = ModuleType(f"{module.__name__}.scene_generation_sessions")
    sessions.close = lambda: None
    engine = ModuleType(f"{module.__name__}.engine")
    engine.stop_viewport_sessions_for_unregister = lambda: True
    monkeypatch.setitem(sys.modules, sessions.__name__, sessions)
    monkeypatch.setitem(sys.modules, engine.__name__, engine)
    monkeypatch.setattr(module.runtime_services.owner, "close", lambda: None)
    monkeypatch.setattr(
        module,
        "_load_runtime_pin_and_storage",
        lambda **_kwargs: ("a" * 64, tmp_path),
    )
    monkeypatch.setattr(module, "remove_runtime", remove)
    monkeypatch.setattr(module, "get_addon_preferences", lambda _context=None: preferences)
    module._RUNTIME_INSTALL_STATE["error"] = "stale install error"
    context = _operator_context()
    operator = module.OvrtxExampleRemoveRuntime()

    assert operator.execute(context) == {"RUNNING_MODAL"}
    assert started.wait(timeout=0.1)
    assert module._RUNTIME_INSTALL_STATE["operation"] == "removing"
    assert module._RUNTIME_INSTALL_STATE["error"] == ""

    operator.modal(context, SimpleNamespace(type="TIMER"))
    assert module._RUNTIME_INSTALL_STATE["progress"] == 0.25
    assert module._RUNTIME_INSTALL_STATE["message"] == "Removing runtime files"
    release.set()
    operator._thread.join(timeout=1.0)

    assert operator.modal(context, SimpleNamespace(type="TIMER")) == {"FINISHED"}
    assert removed == [(tmp_path, bundled_runtime.current_platform_id())]
    assert module._RUNTIME_INSTALL_STATE["active"] is False
    assert module._RUNTIME_INSTALL_STATE["progress"] == 1.0
    assert module._RUNTIME_INSTALL_STATE["error"] == ""
    assert context.window_manager.timer_removed
    assert preferences.worker_command == ""
    assert preferences.native_client_path == ""
    assert preferences.native_client_module == ""


def test_runtime_install_is_disabled_when_runtime_is_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_addon_with_fake_bpy(monkeypatch)
    monkeypatch.setattr(
        module,
        "get_addon_preferences",
        lambda _context=None: SimpleNamespace(runtime_source=str(tmp_path)),
    )
    monkeypatch.setattr(
        module,
        "runtime_bundle_status",
        lambda: {"state": "ready", "current_root": str(tmp_path / "old")},
    )
    context = _operator_context()
    operator = module.OvrtxExampleInstallRuntime()

    assert not module.OvrtxExampleInstallRuntime.poll(context)
    assert operator.execute(context) == {"CANCELLED"}
    assert module._RUNTIME_INSTALL_STATE["error"]
    assert not hasattr(operator, "reports")
    assert context.window_manager.popups == []


def test_preflight_reports_missing_external_prerequisites() -> None:
    summary = preflight_summary(
        check_addon_prerequisites(
            worker_command="",
            native_client_path="",
            native_client_module="missing_ovrtx_native_client",
        )
    )

    assert summary["status"] == "blocked"
    blockers = {item["key"]: item["message"] for item in summary["blockers"]}
    assert "user_scene_path" not in blockers
    assert blockers["worker_command"] == "Set the ovrtx-bridge-server command."
    assert blockers["worker_package_root"] == "Set the worker command with --package-root."
    assert blockers["native_client_path"] == "Set the built native client output path."
    assert blockers["native_client_module"] == "Module not importable: missing_ovrtx_native_client"


def test_preflight_passes_when_configured_paths_are_available(tmp_path: Path) -> None:
    package_root = tmp_path / "package-root"
    native_client_path = tmp_path / "native-client"
    _make_package_root(package_root)
    module_name = _write_native_client(native_client_path)

    summary = preflight_summary(
        check_addon_prerequisites(
            worker_command=f"{sys.executable} --package-root {package_root}",
            native_client_path=str(native_client_path),
            native_client_module=module_name,
        )
    )

    assert summary["status"] == "pass"
    assert summary["blockers"] == []


def test_preflight_rejects_package_root_missing_ovrtx_layout(tmp_path: Path) -> None:
    package_root = tmp_path / "package-root"
    package_root.mkdir()
    (package_root / "plugins").mkdir()  # only one of the required markers present

    summary = preflight_summary(
        check_addon_prerequisites(
            worker_command=f"{sys.executable} --package-root={package_root}",
            native_client_path="",
            native_client_module="missing_ovrtx_native_client",
        )
    )

    blockers = {item["key"]: item["message"] for item in summary["blockers"]}
    assert "worker_package_root" in blockers
    assert "mdl" in blockers["worker_package_root"]
    assert "usd_plugins" in blockers["worker_package_root"]


def test_preflight_accepts_module_from_configured_native_client_path(tmp_path: Path) -> None:
    package_root = tmp_path / "package-root"
    native_client_path = tmp_path / "native-client"
    _make_package_root(package_root)
    module_name = _write_native_client(native_client_path)

    summary = preflight_summary(
        check_addon_prerequisites(
            worker_command=f"{sys.executable} --package-root={package_root}",
            native_client_path=str(native_client_path),
            native_client_module=module_name,
        )
    )

    assert summary["status"] == "pass"


def test_preflight_rejects_module_discovery_without_required_surface(tmp_path: Path) -> None:
    package_root = tmp_path / "package-root"
    native_client_path = tmp_path / "native-client"
    _make_package_root(package_root)
    native_client_path.mkdir()
    (native_client_path / "bad_native_client.py").write_text("VALUE = 1\n", encoding="utf-8")

    summary = preflight_summary(
        check_addon_prerequisites(
            worker_command=f"{sys.executable} --package-root={package_root}",
            native_client_path=str(native_client_path),
            native_client_module="bad_native_client",
        )
    )

    assert summary["status"] == "blocked"
    blockers = {item["key"]: item["message"] for item in summary["blockers"]}
    assert "Native client surface invalid" in blockers["native_client_module"]


def test_addon_preflight_checks_only_runtime_inputs(tmp_path: Path) -> None:
    package_root = tmp_path / "package-root"
    _make_package_root(package_root)
    native_client_path = tmp_path / "native-client"
    module_name = _write_native_client(native_client_path)

    summary = preflight_summary(
        check_addon_prerequisites(
            worker_command=f"{sys.executable} --package-root={package_root}",
            native_client_path=str(native_client_path),
            native_client_module=module_name,
        )
    )

    assert summary["status"] == "pass"
    assert all(item["key"] != "user_scene_path" for item in summary["checks"])


def test_preflight_uses_bundled_runtime_defaults(tmp_path: Path, monkeypatch) -> None:
    addon_root = tmp_path / "addon"
    package_root = addon_root / "runtime" / "ovrtx-bridge-server"
    native_client_path = addon_root / "native"
    _write_bundled_worker(addon_root)
    _make_package_root(package_root)
    _write_native_client(native_client_path, module_name="ovsensors_worker_client")
    monkeypatch.setattr(bundled_runtime, "addon_root", lambda: addon_root)
    monkeypatch.setattr(bundled_runtime, "current_platform_id", lambda: _HOST_PLATFORM_ID)

    summary = preflight_summary(
        check_addon_prerequisites(
            worker_command="",
            native_client_path="",
            native_client_module="ovsensors_worker_client",
        )
    )

    assert summary["status"] == "pass"


def test_preflight_preferences_uses_installed_runtime(tmp_path: Path, monkeypatch) -> None:
    runtime_root = tmp_path / "runtime"
    package_root = runtime_root / "runtime" / "ovrtx-bridge-server"
    native_client_path = runtime_root / "native"
    _write_bundled_worker(runtime_root)
    _make_package_root(package_root)
    _write_native_client(native_client_path, module_name="ovsensors_worker_client")
    monkeypatch.setattr(bundled_runtime, "current_platform_id", lambda: _HOST_PLATFORM_ID)
    monkeypatch.setattr(
        addon,
        "runtime_bundle_status",
        lambda: {
            "state": "ready",
            "current_root": str(runtime_root),
            "message": "Runtime is installed.",
            "manifest_sha256": "manifest",
            "installed_manifest_sha256": "manifest",
        },
    )
    monkeypatch.setattr(
        addon.runtime_services.owner,
        "diagnostics",
        lambda: {"status": "ready", "error": "", "health": {}},
    )

    summary = addon.preflight_preferences(
        SimpleNamespace(
            worker_command="",
            native_client_path="",
            native_client_module="ovsensors_worker_client",
        )
    )

    assert summary["status"] == "pass"
    assert summary["runtime"]["state"] == "ready"
    assert "scene_input" not in summary
    assert all(check["key"] != "user_scene_path" for check in summary["checks"])


def test_runtime_override_reset_uses_installed_defaults(tmp_path: Path, monkeypatch) -> None:
    worker = tmp_path / "bin" / "ovrtx-bridge-server"
    native = tmp_path / "native"
    package = tmp_path / "runtime" / "ovrtx-bridge-server"
    worker.parent.mkdir(parents=True)
    worker.write_text("worker", encoding="utf-8")
    native.mkdir()
    package.mkdir(parents=True)
    preferences = SimpleNamespace(
        worker_command="override",
        native_client_path="override",
        native_client_module="override",
    )
    monkeypatch.setattr(bundled_runtime, "current_platform_id", lambda: "linux-x64")

    addon._apply_runtime_defaults(preferences, tmp_path)

    assert preferences.native_client_path == str(native)
    assert preferences.native_client_module == bundled_runtime.DEFAULT_OVRTX_NATIVE_CLIENT_MODULE


def test_runtime_preference_population_preserves_partial_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    worker = tmp_path / "bin" / "ovrtx-bridge-server"
    package = tmp_path / "runtime" / "ovrtx-bridge-server"
    native = tmp_path / "native"
    worker.parent.mkdir(parents=True)
    worker.write_text("worker", encoding="utf-8")
    package.mkdir(parents=True)
    native.mkdir()
    monkeypatch.setattr(bundled_runtime, "current_platform_id", lambda: "linux-x64")
    monkeypatch.setattr(
        addon,
        "runtime_bundle_status",
        lambda: {"state": "ready", "current_root": str(tmp_path)},
    )

    worker_override = SimpleNamespace(
        worker_command="custom-worker",
        native_client_path="",
        native_client_module="custom-module",
    )
    monkeypatch.setattr(addon, "get_addon_preferences", lambda: worker_override)
    addon._populate_runtime_preferences()
    assert worker_override.native_client_path == str(native)
    assert worker_override.native_client_module == "custom-module"

    native_override = SimpleNamespace(
        worker_command="",
        native_client_path="/custom/native",
        native_client_module="",
    )
    monkeypatch.setattr(addon, "get_addon_preferences", lambda: native_override)
    addon._populate_runtime_preferences()
    assert native_override.native_client_path == "/custom/native"
    assert native_override.native_client_module == bundled_runtime.DEFAULT_OVRTX_NATIVE_CLIENT_MODULE


def test_runtime_preference_population_resets_session_install_source(monkeypatch) -> None:
    preferences = SimpleNamespace(runtime_source="/previous/session")
    monkeypatch.setattr(addon, "get_addon_preferences", lambda: preferences)
    monkeypatch.setattr(addon, "runtime_bundle_status", lambda: {"state": "missing"})

    addon._populate_runtime_preferences()

    assert preferences.runtime_source == ""



def test_addon_ui_surfaces_no_scene_input_setting() -> None:
    addon_package = ROOT / "addon" / "ovrtx_blender_example"
    for name in ("__init__.py", "ui.py", "properties.py"):
        source = (addon_package / name).read_text(encoding="utf-8")
        assert "user_scene_path" not in source, f"{name} still exposes a scene-input setting"

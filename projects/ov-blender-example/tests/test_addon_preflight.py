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
sys.path.insert(0, str(ROOT / 'addon'))
import ovrtx_blender_example as addon
from ovrtx_blender_example import check_addon_prerequisites, preflight_summary
from ovrtx_blender_example import bundled_runtime
from ovrtx_blender_example import preflight as preflight_module

def test_addon_build_sha_reads_packaged_commit(tmp_path: Path) -> None:
    (tmp_path / 'build-sha').write_text('a' * 40 + '\n', encoding='utf-8')
    assert addon.addon_build_sha(tmp_path) == 'a' * 8

def test_addon_build_sha_is_empty_when_not_packaged(tmp_path: Path) -> None:
    assert addon.addon_build_sha(tmp_path) == ''

def test_addon_build_label_shows_generation(tmp_path: Path) -> None:
    (tmp_path / 'build-sha').write_text('a' * 40 + '\n', encoding='utf-8')
    (tmp_path / 'build-generation').write_text('1842\n', encoding='utf-8')
    assert addon.addon_build_label(tmp_path) == f"Build 1842 ({'a' * 8})"

def test_addon_build_label_falls_back_to_sha(tmp_path: Path) -> None:
    (tmp_path / 'build-sha').write_text('b' * 40 + '\n', encoding='utf-8')
    assert addon.addon_build_label(tmp_path) == f"Build SHA: {'b' * 8}"

def test_register_class_once_does_not_hide_stale_rna_registration(monkeypatch) -> None:

    class Candidate:
        is_registered = False

    def reject_stale_registration(_cls) -> None:
        raise ValueError('already registered as a subclass')
    monkeypatch.setattr(addon, 'bpy', SimpleNamespace(utils=SimpleNamespace(register_class=reject_stale_registration)))
    try:
        addon._register_class_once(Candidate)
    except ValueError as exc:
        assert 'already registered' in str(exc)
    else:
        raise AssertionError('stale RNA registration must not be ignored')
_HOST_PLATFORM_ID = 'windows-x64' if os.name == 'nt' else 'linux-x64'
_HOST_EXECUTABLE_SUFFIX = '.exe' if os.name == 'nt' else ''

def _fake_bpy(*, online_access: bool, locked: bool=False, readonly_raises: bool=False) -> SimpleNamespace:

    def is_property_readonly(_name: str) -> bool:
        if readonly_raises:
            raise RuntimeError('missing bl_rna')
        return locked
    system = SimpleNamespace(is_property_readonly=is_property_readonly)
    return SimpleNamespace(app=SimpleNamespace(online_access=online_access, online_access_override=locked), context=SimpleNamespace(preferences=SimpleNamespace(system=system)))

def _load_addon_with_fake_bpy(monkeypatch: pytest.MonkeyPatch) -> object:
    package_name = 'ovrtx_blender_example_operator_test'
    package_root = ROOT / 'addon' / 'ovrtx_blender_example'
    fake_bpy = ModuleType('bpy')

    class Operator:

        def report(self, levels, message):
            reports = getattr(self, 'reports', [])
            reports.append((set(levels), message))
            self.reports = reports
    fake_bpy.types = SimpleNamespace(Operator=Operator, AddonPreferences=object)
    fake_bpy.app = SimpleNamespace(online_access=True, online_access_override=False)
    props = ModuleType('bpy.props')
    props.BoolProperty = lambda **_kwargs: None
    fake_bpy.string_properties = []
    props.StringProperty = lambda **kwargs: fake_bpy.string_properties.append(kwargs)
    monkeypatch.setitem(sys.modules, 'bpy', fake_bpy)
    monkeypatch.setitem(sys.modules, 'bpy.props', props)
    spec = importlib.util.spec_from_file_location(package_name, package_root / '__init__.py', submodule_search_locations=[str(package_root)])
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package_name, module)
    spec.loader.exec_module(module)
    for value in vars(module).values():
        if isinstance(value, type) and value.__module__ == module.__name__:
            getattr(value, '__annotations__', {})
    return module

def _operator_context() -> SimpleNamespace:
    area = SimpleNamespace(redraw_count=0)
    area.tag_redraw = lambda: setattr(area, 'redraw_count', area.redraw_count + 1)
    return SimpleNamespace(area=area, preferences=SimpleNamespace(system=SimpleNamespace(use_online_access=True)), window=object(), window_manager=_FakeWindowManager())

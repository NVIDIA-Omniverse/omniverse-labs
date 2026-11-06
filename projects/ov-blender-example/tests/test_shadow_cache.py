# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace
from ovrtx_blender_example import scene_generation


def test_shadow_toggle_changes_source_cache_key_even_without_dirty_flag(tmp_path, monkeypatch):
    source = tmp_path / "saved.blend"
    source.write_bytes(b"unchanged saved source")
    obj = SimpleNamespace(name_full="Glass", type="MESH", visible_shadow=True)
    scene = SimpleNamespace(name_full="Scene", objects=[obj])
    module = SimpleNamespace(data=SimpleNamespace(filepath=str(source), is_dirty=False))
    monkeypatch.setattr(scene_generation, "_blender_module_provider", lambda: module)
    before, reason = scene_generation._source_lookup(scene)
    assert before and not reason
    obj.visible_shadow = False
    after, reason = scene_generation._source_lookup(scene)
    assert after and after != before and not reason
    obj.visible_shadow = True
    assert scene_generation._source_lookup(scene)[0] == before

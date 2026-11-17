# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ovd_ui.palette import (ACTOR_KEYS, GIZMO_KEYS, ColorPalette, DEFAULT_PALETTE_PATH,
                            MAGENTA, PROFILE_FUNCTION_DEFAULT, color_hex, parse_color)


@pytest.fixture
def palette(tmp_path):
    path = tmp_path / "colors.json"
    path.write_bytes(DEFAULT_PALETTE_PATH.read_bytes())
    return ColorPalette(path, poll_interval=0)


@pytest.mark.parametrize("value,expected", [
    ("#123", (1/15, 2/15, 3/15, 1)),
    ("#80123456", (18/255, 52/255, 86/255, 128/255)),
    ("#123456789", (4657/65535, 17764/65535, 30871/65535, 1)),
    ("#123456789abc", (0x1234/65535, 0x5678/65535, 0x9abc/65535, 1)),
    ("light blue", (173/255, 216/255, 230/255, 1)),
    (" ReD\t", (1, 0, 0, 1)),
    ("transparent", (0, 0, 0, 0)),
])
def test_qcolor_compatible_forms(value, expected):
    assert parse_color(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", [None, 17, [], {}, "#1234", "#12345G", "#123 ", "rgb(1,2,3)", "rebeccapurple", "light\nblue"])
def test_invalid_qcolor_forms(value):
    with pytest.raises(ValueError):
        parse_color(value)


def test_all_shipped_keys_and_independent_profile_color(palette):
    assert len(ACTOR_KEYS) == len(GIZMO_KEYS) == 11
    assert not palette.issues
    assert parse_color(PROFILE_FUNCTION_DEFAULT) == (0, 1, 0, 1)
    assert "ProfileFunctionColor" not in palette.document
    assert palette.actor("rigidStatic") == parse_color("#E2E2E2")
    assert palette.gizmo("selectionTint") == parse_color("#787878")


def test_invalid_missing_files_and_keys_clear_old_colors(palette):
    palette.path.write_text('{"actors": {"rigidStatic": "red", "cloth": 1}}')
    palette.poll()
    assert palette.actor("rigidStatic") == (1, 0, 0, 1)
    assert palette.actor("cloth") == MAGENTA
    assert palette.gizmo("velocity") == MAGENTA
    palette.path.write_text('{broken')
    palette.poll()
    assert palette.load_error
    assert palette.actor("rigidStatic") == MAGENTA
    palette.path.unlink()
    palette.poll()
    assert palette.actor("rigidStatic") == MAGENTA


def test_repeated_atomic_replacement_reloads_and_preserves_unknowns(palette):
    for value in ("red", "blue", "lime"):
        updated = deepcopy(palette.document)
        updated["actors"]["rigidStatic"] = value
        updated["extra"] = {"retained": True}
        replacement = palette.path.with_suffix(".temporary")
        replacement.write_text(json.dumps(updated))
        os.replace(replacement, palette.path)
        assert palette.poll()
        assert palette.actor("rigidStatic") == parse_color(value)
        assert not palette.poll()
    palette.edit({("gizmos", "velocity"): "#112233"})
    assert palette.document["extra"] == {"retained": True}
    assert not list(palette.path.parent.glob(".colors-*"))
    before = palette.path.read_bytes()
    with pytest.raises(ValueError):
        palette.edit({("actors", "cloth"): "invalid"})
    assert palette.path.read_bytes() == before


def test_private_palette_does_not_write_source_or_legacy_preferences(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA" if os.name == "nt" else "XDG_DATA_HOME",
                       str(tmp_path / "data"))
    source = tmp_path / "source.json"
    source.write_bytes(DEFAULT_PALETTE_PATH.read_bytes())
    original = source.read_bytes()
    legacy = tmp_path / "legacy.ini"
    legacy.write_text("ProfileFunctionColor=legacy\n")
    palette = ColorPalette(defaults_path=source)
    palette.edit({("actors", "cloth"): "red"})
    assert palette.path != source
    assert source.read_bytes() == original
    assert legacy.read_text() == "ProfileFunctionColor=legacy\n"
    palette.reset_defaults()
    assert palette.actor("cloth") == parse_color("#F4C2AB")
    explicit = ColorPalette(tmp_path / "absent.json")
    assert not explicit.path.exists()
    assert explicit.actor("cloth") == MAGENTA


def test_actor_dispatch_preserves_reference_sleeping_kinematic_rule(palette):
    geometry = SimpleNamespace(object_ids=np.arange(6), actor_classes=np.array([1, 1, 1, 2, 3, 4]),
        actor_flags=np.array([0, 1, 4, 0, 0, 0]), sleeping=np.array([0, 0, 1, 0, 0, 0]))
    names = {1: "PxRigidDynamic", 2: "PxRigidStatic", 3: "PxCloth", 4: "PxArticulationLink"}
    keys = ("rigidDynamicActive", "kinematicActive", "rigidDynamicSleep", "rigidStatic", "cloth", "articulation")
    colors = palette.geometry_colors(geometry, names)
    for color, key in zip(colors, keys):
        assert np.array_equal(color, np.asarray(palette.actor(key), dtype="f4"))



def test_particle_blue_requires_exact_ovd_buffer_witnesses(palette):
    # Adjacent rows vary one native witness at a time. Legacy particles, cloth,
    # rigid points and incomplete metadata must retain their existing dispatch.
    count = 10
    geometry = SimpleNamespace(object_ids=np.arange(10, 10 + count),
        actor_classes=np.array([1, 1, 1, 1, 1, 1, 2, 3, 4, 5]),
        kinds=np.array([7, 5, 7, 7, 7, 7, 7, 7, 7, 7]),
        particle_radius_modes=np.array([2, 2, 1, 2, 2, 2, 1, 1, 1, 2]),
        shape_ids=np.array([100, 101, 102, 0, 14, 105, 106, 107, 108, 109]),
        geometry_ids=np.array([0, 0, 0, 0, 0, 200, 0, 0, 0, 0]))
    names = {1: "PxPBDParticleSystem", 2: "PxParticleSystem", 3: "PxParticleFluid",
             4: "PxCloth", 5: "PxRigidDynamic"}
    palette.edit({("actors", "articulation"): "red", ("actors", "particle"): "yellow"})
    colors = palette.geometry_colors(geometry, names)
    assert np.array_equal(colors[0], np.asarray((.2, .6, 1., 1.), dtype="f4"))
    assert np.all(colors[1:8] == np.asarray(palette.actor("articulation"), dtype="f4"))
    assert np.array_equal(colors[8], np.asarray(palette.actor("cloth"), dtype="f4"))
    assert np.array_equal(colors[9], np.asarray(palette.actor("rigidDynamicActive"), dtype="f4"))
    del geometry.particle_radius_modes
    assert np.array_equal(palette.geometry_colors(geometry, names)[0],
                          np.asarray(palette.actor("articulation"), dtype="f4"))


def test_real_ovd_particle_buffer_uses_reference_blue(palette, tmp_path):
    from _pvd_native import Session
    root = Path(__file__).resolve().parents[2]
    session = Session(str(root / "test/data/multi_geom.ovd"), str(tmp_path / "particles"))
    assert session.seek(session.scenes()[0].id, 30)
    snapshot = session.geometry_snapshot()
    names = {cls.id: cls.name for cls in session.classes()}
    points = np.flatnonzero(snapshot.kinds == 7)
    assert len(points) == 1
    row = points[0]
    assert names[int(snapshot.actor_classes[row])] == "PxPBDParticleSystem"
    assert snapshot.particle_radius_modes[row] == 2
    assert snapshot.shape_ids[row] != snapshot.object_ids[row]
    assert len(snapshot.override_colors) == 0
    colors = palette.geometry_colors(snapshot, names)
    assert np.array_equal(colors[row], np.asarray((.2, .6, 1., 1.), dtype="f4"))
    for other in np.flatnonzero(snapshot.kinds != 7):
        cls = names[int(snapshot.actor_classes[other])]
        expected = "articulation"
        if cls == "PxRigidStatic":
            expected = "rigidStatic"
        elif cls == "PxRigidDynamic":
            expected = "kinematicActive" if snapshot.actor_flags[other] else "rigidDynamicActive"
            if snapshot.sleeping[other]:
                expected = "rigidDynamicSleep"
        assert np.array_equal(colors[other], np.asarray(palette.actor(expected), dtype="f4"))


def test_real_legacy_cloth_points_keep_palette_color(palette, tmp_path):
    from _pvd_native import Session
    root = Path(__file__).resolve().parents[2]
    session = Session(str(root / "test/data/cloth_physx34.pxd2"), str(tmp_path / "cloth-colors"))
    assert session.seek(session.scenes()[0].id, 30)
    snapshot = session.geometry_snapshot()
    names = {cls.id: cls.name for cls in session.classes()}
    points = np.flatnonzero(snapshot.kinds == 7)
    assert len(points)
    colors = palette.geometry_colors(snapshot, names)
    for row in points:
        assert names[int(snapshot.actor_classes[row])] == "PxCloth"
        assert snapshot.particle_radius_modes[row] == 1
        assert np.array_equal(colors[row], np.asarray(palette.actor("cloth"), dtype="f4"))

def test_roles_preserve_recorded_colors_and_source_com_alpha(palette):
    original = np.arange(5 * 3 * 4, dtype="f4").reshape(5, 3, 4) / 100
    snapshot = SimpleNamespace(kinds=np.ones(5), color_roles=np.array([0, 4, 7, 8, 2]), colors=original)
    palette.edit({("gizmos", "centerOfMass"): "#00112233"})
    colors = palette.overlay_colors(snapshot)
    assert np.array_equal(original, snapshot.colors)
    assert np.array_equal(colors[0], original[0])
    for row, key in ((1, "axisX"), (2, "jointParentFrame"), (3, "jointChildFrame")):
        assert np.allclose(colors[row], palette.gizmo(key))
    assert np.allclose(colors[4, :, :3], parse_color("#112233")[:3])
    assert np.all(colors[4, :, 3] == 1)


def test_native_roles_cover_generated_axes_and_preserve_queries(palette, tmp_path):
    from _pvd_native import Session
    root = Path(__file__).resolve().parents[2]
    session = Session(str(root / "test/data/multi_geom.ovd"), str(tmp_path / "native"))
    session.seek(session.scenes()[0].id, 1)
    snapshot = session.overlay_snapshot(1 | 2 | 16 | 256)
    assert len(snapshot.color_roles) == len(snapshot.kinds)
    assert {4, 5, 6}.issubset(set(snapshot.color_roles.tolist()))
    assert set(snapshot.color_roles[snapshot.categories == 2]) == {4, 5, 6}
    session = Session(str(root / "test/data/cloth_physx34.pxd2"), str(tmp_path / "legacy"))
    session.seek(session.scenes()[0].id, 1)
    snapshot = session.overlay_snapshot(64 | 128)
    assert np.all(snapshot.color_roles == 0)
    assert np.array_equal(palette.overlay_colors(snapshot), snapshot.colors)


def test_hex_round_trip_byte_colors():
    for value in ("#80123456", "#FFFFFF", "#000000", "#FF00FF"):
        assert parse_color(color_hex(parse_color(value))) == parse_color(value)


def test_palette_atomic_reload_updates_real_gpu_without_new_snapshot(palette, tmp_path):
    import moderngl
    from _pvd_native import Session
    from ovd_ui.renderer import ViewportRenderer

    class Consumer:
        def set_image_data(self, *args):
            pass

        def destroy(self):
            pass

    root = Path(__file__).resolve().parents[2]
    session = Session(str(root / "test/data/test_scene.ovd"), str(tmp_path / "gpu"))
    session.seek(session.scenes()[0].id, 1)
    options = {} if sys.platform == "win32" else {"backend": "egl"}
    context = moderngl.create_standalone_context(require=330, **options)
    renderer = ViewportRenderer(context, Consumer())
    try:
        palette.edit({("actors", key): "red" for key in ACTOR_KEYS})
        renderer.set_palette(palette, session.classes())
        renderer.set_snapshot(session.geometry_snapshot(), True)
        renderer.render(400, 300)
        red = renderer.framebuffer.read(viewport=(200, 150, 1, 1), components=3)
        assert red[0] > red[2] + 20
        updated = deepcopy(palette.document)
        updated["actors"] = {key: "blue" for key in ACTOR_KEYS}
        replacement = palette.path.with_suffix(".replace")
        replacement.write_text(json.dumps(updated))
        os.replace(replacement, palette.path)
        renderer.render(400, 300)
        blue = renderer.framebuffer.read(viewport=(200, 150, 1, 1), components=3)
        assert blue[2] > blue[0] + 20
        assert renderer.context.error == "GL_NO_ERROR"
    finally:
        renderer.close()
        context.release()

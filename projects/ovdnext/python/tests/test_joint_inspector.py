# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derived joints remain selectable through native scene ownership and roles."""
from pathlib import Path
import subprocess

import pytest
from _pvd_native import Session

from native_fixtures import executable
from ovd_ui.adapters import PvdDataAdapter
from ovd_ui.models import ObjectTreeModel


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def joint_writer():
    return executable("pvd_geometry_fixture", "PVD_GEOMETRY_FIXTURE")


def capture(tmp_path, joint_writer, mode="--joint-inspector", frame=None):
    path = tmp_path / f"{mode[2:]}.ovd"
    subprocess.run([str(joint_writer), str(path), mode], check=True)
    session = Session(str(path), str(tmp_path))
    if frame is not None:
        scene = next(scene for scene in session.scenes() if scene.alive)
        session.seek(scene.id, frame)
    adapter = PvdDataAdapter(session)
    model = ObjectTreeModel()
    model.populate(adapter)
    return session, adapter, model


def d6_joints(adapter, model):
    return [instance.id for instance in adapter.instances.values()
            if instance.alive and adapter.class_name(instance.id) == "PxD6Joint"
            and model.canonical_item(instance.id) is not None]


def ancestor_rows(model, instance_id):
    return [(item.values[0].as_string, item.class_name, item.instance_id)
            for item in model.ancestors(instance_id)]


def test_world_connected_d6_joint_is_under_scene_joints_with_base_and_derived_properties(tmp_path, joint_writer):
    _, adapter, model = capture(tmp_path, joint_writer)
    joint, = d6_joints(adapter, model)
    assert adapter.instances[joint].scene_joint
    scene = adapter.instances[joint].scene_id
    assert ancestor_rows(model, joint)[-2:] == [
        (f"First scene [id:{scene}]", "PxScene", scene), ("Joints", "", None)]
    properties = {prop.name: prop for prop in adapter.properties(joint)}
    assert {"name", "actor0", "actor1", "breakForce", "motions"} <= properties.keys()
    assert adapter.name(joint) == "First world joint"
    assert adapter.references(joint, "actor0") == [-1]  # World endpoint, not an object row.
    actor, = adapter.references(joint, "actor1")
    assert adapter.name(actor) == "First box"
    assert adapter.value(joint, properties["breakForce"]) == "125"
    assert adapter.session.property_size(joint, properties["actor1"].id) == 8


def test_reused_handles_keep_joint_references_in_their_recording_segment(tmp_path, joint_writer):
    _, adapter, model = capture(tmp_path, joint_writer, "--joint-inspector-segments")
    scenes = [instance for instance in adapter.instances.values()
              if instance.alive and adapter.class_name(instance.id) == "PxScene"]
    assert len(scenes) == 2
    for scene in scenes:
        joints = [joint for joint in d6_joints(adapter, model)
                  if adapter.instances[joint].scene_id == scene.id]
        assert len(joints) == 1
        joint = joints[0]
        assert ancestor_rows(model, joint)[-2:] == [
            (f"{adapter.name(scene.id)} [id:{scene.id}]", "PxScene", scene.id),
            ("Joints", "", None)]
        first = adapter.name(scene.id) == "First scene"
        assert adapter.name(joint) == ("First world joint" if first else "Second world joint")
        actor, = adapter.references(joint, "actor1")
        assert adapter.name(actor) == ("First box" if first else "Second box")


def test_recreated_scene_keeps_the_active_joint_reachable(tmp_path, joint_writer):
    _, adapter, model = capture(tmp_path, joint_writer, "--joint-inspector-recreated")
    scenes = [instance for instance in adapter.instances.values()
              if instance.alive and adapter.class_name(instance.id) == "PxScene"]
    active, = [scene for scene in scenes if adapter.name(scene.id) == "First scene"]
    joint, = [joint for joint in d6_joints(adapter, model)
              if adapter.instances[joint].scene_id == active.id]
    assert ancestor_rows(model, joint)[-2:] == [
        (f"First scene [id:{active.id}]", "PxScene", active.id), ("Joints", "", None)]


def test_joint_follows_creation_death_and_rewind(tmp_path, joint_writer):
    session, adapter, model = capture(tmp_path, joint_writer, "--joint-inspector-lifetime", 2)
    scene = next(scene for scene in session.scenes() if scene.alive)
    for frame, expected in ((2, 0), (3, 1), (4, 1), (5, 0), (6, 0), (4, 1)):
        session.seek(scene.id, frame)
        adapter.reload()
        model.populate(adapter)
        assert len(d6_joints(adapter, model)) == expected


def test_articulation_joint_stays_under_its_link(tmp_path):
    session = Session(str(ROOT / "test/data/multi_geom.ovd"), str(tmp_path))
    session.seek(session.scenes()[0].id, 1)
    adapter = PvdDataAdapter(session)
    model = ObjectTreeModel()
    model.populate(adapter)
    joints = [instance.id for instance in adapter.instances.values()
              if instance.alive and adapter.class_name(instance.id) == "PxArticulationJointReducedCoordinate"]
    assert joints
    for joint in joints:
        assert not adapter.instances[joint].scene_joint
        parent = model.canonical_item(joint).parent
        assert parent.class_name == "PxArticulationLink"
        assert all(item.values[0].as_string != "Joints" for item in model.ancestors(joint))

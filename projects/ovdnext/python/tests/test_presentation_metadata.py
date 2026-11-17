# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publication metadata parity against validated native reads and retained views.

The optional CLI emits deterministic owned-value evidence for comparing two
separately built native modules in isolated processes. No GUI is initialized.
"""
from capture_semantics import test_scene_roles, unique_instance_id
from pathlib import Path
import argparse
import gc
import hashlib
import json
import socket
import tempfile

import numpy as np
import pytest
from _pvd_native import Session, SnapshotDataUnavailable
from test_native_presentation import HANDSHAKE, wait_until

ROOT = Path(__file__).resolve().parents[2]


def attrs(value, names):
    return tuple(getattr(value, name) for name in names.split())


def array_signature(value):
    assert not value.flags.writeable
    return (value.dtype.str, value.shape, hashlib.sha256(value.tobytes()).hexdigest())


def geometry_signature(view):
    geometry = view.geometry_snapshot()
    names = "object_ids shape_ids geometry_ids scene_ids actor_classes kinds sleeping double_sided particle_radius_modes shape_flags actor_flags resource_ids transforms dimensions bounds override_colors"
    result = {name: array_signature(getattr(geometry, name)) for name in names.split()}
    result["meshes"] = [(attrs(mesh, "id topology_id source_id"), array_signature(mesh.vertices),
                         array_signature(mesh.indices)) for mesh in geometry.meshes]
    result["diagnostics"] = geometry.diagnostics
    return result


def metadata_fields(classes, instance, properties):
    """All fields owned even without explicitly selecting the instance."""
    name = classes[instance.class_id].name
    for prop in properties:
        reference = prop.type == "ObjectRef"
        string = prop.name in ("Name", "name") and prop.type == "StringHandle"
        camera = ((name == "Camera" and prop.name in ("Origin", "Target", "Up")) or
                  (name == "PxScene" and prop.name in ("gravity", "Gravity")) or
                  (name in ("PxScene", "PxPhysics") and prop.name in
                   ("tolerancesScale", "TolerancesScale.Length", "TolerancesScale.Speed")))
        if reference or string or camera or (name == "PxScene" and prop.array):
            yield prop, reference, string, camera


def owned_metadata(view, direct=None):
    classes = {c.id: c for c in view.classes()}
    values = []
    for instance in view.instances():
        if not instance.alive:
            continue
        for prop, reference, string, camera in metadata_fields(
                classes, instance, view.properties(instance.class_id)):
            size = view.property_size(instance.id, prop.id)
            row = [instance.id, prop.id, size]
            if direct is not None:
                assert size == direct.property_size(instance.id, prop.id)
            if reference:
                refs = view.references(instance.id, prop.id)
                row.append(refs)
                if direct is not None:
                    assert refs == direct.references(instance.id, prop.id)
            if string:
                text = view.string_value(instance.id, prop.id)
                row.append(text)
                if direct is not None:
                    assert text == direct.string_value(instance.id, prop.id)
            if camera or string:
                limit = 256 if string else 64
                raw = view.read(instance.id, prop.id, 0, limit)
                row.append(raw.hex())
                if direct is not None:
                    assert raw == direct.read(instance.id, prop.id, 0, limit)
            values.append(row)
    return values


def owned_signature(view, direct=None):
    result = dict(frame=view.frame, scene=view.scene_id, epoch=view.segment_epoch,
                  copied_bytes=view.copied_bytes, metadata=owned_metadata(view, direct),
                  instances=[attrs(i, "id class_id scene_id physics_id alive") for i in view.instances()],
                  classes=[attrs(c, "id base_id byte_size name") for c in view.classes()],
                  properties={c.id: [attrs(p, "id name type native_type semantic array field_count")
                                     for p in view.properties(c.id)] for c in view.classes()},
                  geometry=geometry_signature(view), inspected=view.inspected_ids,
                  unavailable=view.unavailable_ids)
    selected = []
    for instance in view.instances():
        if instance.id not in view.inspected_ids:
            continue
        for prop in view.properties(instance.class_id):
            raw = view.read(instance.id, prop.id)
            selected.append((instance.id, prop.id, view.property_size(instance.id, prop.id), raw.hex()))
            if direct is not None:
                assert raw == direct.read(instance.id, prop.id)
    result["selected"] = selected
    return result


def collect_case(tmp_path, filename, frames):
    source = Session(str(ROOT / "test/data" / filename), str(tmp_path))
    scene = source.scenes()[-1] if filename == "ue4_shooter.pxd2" else source.scenes()[0]
    results, retained = [], []
    for frame in frames:
        first = source.presentation(scene.id, frame)
        ids = [int(first.geometry_snapshot().object_ids[0])] if len(first.geometry_snapshot().object_ids) else []
        view = source.presentation(scene.id, frame, inspect_ids=ids)
        assert view.request_applied and view.ready and view.frame == frame
        signature = owned_signature(view, source)
        results.append(signature)
        retained.append((view, signature))
    del source, first
    gc.collect()
    for view, signature in retained:
        assert owned_signature(view) == signature
    return results


@pytest.mark.parametrize("filename,frames", [
    ("test_scene.ovd", [1, 2, 3, 1]),
    ("multi_geom.ovd", [1, 2, 30, 31]),
    ("cloth_physx34.pxd2", [1, 2, 30, 31]),
    ("ue4_shooter.pxd2", [1, 2, 30, 31]),
    ("resample_two_segment.ovd", [1, 2]),
])
def test_metadata_matches_validated_reads_and_survives_future_publications(tmp_path, filename, frames):
    collect_case(tmp_path, filename, frames)


def test_overlapping_windows_keep_original_accounting_and_availability(tmp_path):
    source = Session(str(ROOT / "test/data/cloth_physx34.pxd2"), str(tmp_path))
    scene = source.scenes()[0]
    base = source.presentation(scene.id, 1)
    a = (15, 649, 64, 64)
    contained = (15, 649, 80, 16)
    overlapping = (15, 649, 96, 64)
    view = source.presentation(scene.id, 1, read_requests=[a, contained, overlapping])
    assert view.copied_bytes == base.copied_bytes + 128
    assert view.read(*a) == source.read(*a)
    assert view.read(*contained) == source.read(*contained)
    assert view.read(*overlapping) == source.read(*overlapping)
    # Two overlapping windows do not implicitly authorize their union.
    assert not view.has_read(15, 649, 64, 96)
    with pytest.raises(SnapshotDataUnavailable):
        view.read(15, 649, 64, 96)
    preserved = view.read(*a)
    source.presentation(scene.id, 30)
    assert view.read(*a) == preserved


def test_unique_list_raw_size_and_converted_ids_remain_distinct(tmp_path):
    source = Session(str(ROOT / "test/data/test_scene.ovd"), str(tmp_path))
    view = source.presentation(frame=1)
    classes = {c.id: c.name for c in view.classes()}
    actor = next(i for i in view.instances() if classes[i.class_id] == "PxRigidDynamic")
    shapes = next(p for p in view.properties(actor.class_id) if p.name == "shapes")
    assert view.references(actor.id, shapes.id) == source.references(actor.id, shapes.id)
    assert view.references(actor.id, shapes.id)
    assert view.property_size(actor.id, shapes.id) == source.property_size(actor.id, shapes.id) == 0
    assert view.read(actor.id, shapes.id) == b""


def collect_live_segments(tmp_path):
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    source = Session.listen(port, str(tmp_path / "live-metadata.ovd"))
    connection = None
    def connect():
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=1)
        except ConnectionRefusedError:
            return None
    try:
        connection = wait_until(connect)
        payload = (ROOT / "test/data/resample_two_segment.ovd").read_bytes()
        first_end, second_end = 42117, 98924
        connection.sendall(HANDSHAKE + payload[:first_end])
        wait_until(lambda: source.info().completed_serial == 120)
        first = source.presentation(frame=1)
        before = owned_signature(first, source)
        # The native unique lists expose references despite zero raw payload.
        assert any(row[2] == 0 and len(row) > 3 and isinstance(row[3], list) and row[3]
                   for row in before["metadata"])
        connection.sendall(payload[first_end:first_end + 1024])
        wait_until(lambda: source.info().unpublished_changes)
        deferred = source.presentation(frame=1)
        assert deferred.ready and not deferred.request_applied
        assert owned_signature(deferred) == before
        connection.sendall(payload[first_end + 1024:second_end])
        wait_until(lambda: source.info().completed_serial == 241)
        second = source.presentation(frame=1)
        after = owned_signature(second, source)
        assert second.segment_epoch != first.segment_epoch
        assert owned_signature(first) == before
        return [before, after]
    finally:
        if connection is not None:
            connection.close()
        source.stop_live()


def test_live_unique_lists_and_new_schema_keep_metadata_on_one_publication(tmp_path):
    collect_live_segments(tmp_path)


def test_public_validation_and_failed_request_frame_restore_remain_intact(tmp_path):
    source = Session(str(ROOT / "test/data/test_scene.ovd"), str(tmp_path))
    first = source.presentation(frame=1)
    identity=unique_instance_id(first,'PxRigidDynamic')
    actor = next(i for i in first.instances() if i.id == identity)
    props = {p.name: p for p in first.properties(actor.class_id)}
    mass = props["mass"].id
    for action in (lambda: source.read(999999, mass), lambda: source.property_size(actor.id, 999999)):
        with pytest.raises(IndexError): action()
    with pytest.raises(ValueError, match="reference"):
        source.references(actor.id, mass)
    with pytest.raises(ValueError, match="string"):
        source.string_value(actor.id, mass)
    with pytest.raises(IndexError):
        source.presentation(first.scene_id, 2, read_requests=[(actor.id, 999999, 0, 16)])
    assert source.scenes()[0].current_frame == 1
    with pytest.raises(ValueError, match="8 MiB"):
        source.presentation(first.scene_id, 2, read_requests=[(actor.id, mass, 0, 8*1024*1024+1)])
    assert source.scenes()[0].current_frame == 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as folder:
        result = {name: collect_case(Path(folder), name, frames) for name, frames in [
            ("test_scene.ovd", [1, 2, 3, 1]), ("multi_geom.ovd", [1, 2, 30, 31]),
            ("cloth_physx34.pxd2", [1, 2, 30, 31]), ("ue4_shooter.pxd2", [1, 2, 30, 31]),
            ("resample_two_segment.ovd", [1, 2]) ]}
        result["live_two_segment"] = collect_live_segments(Path(folder))
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")

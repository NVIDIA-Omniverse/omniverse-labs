# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Required GPU-family evidence uses the same pinned real SDK, never a fake peer."""
import json
import math
import os
from pathlib import Path

import pytest
import numpy as np
from _pvd_native import Session

from test_sdk_tcp_selection import sdk_capture, payloads, wait_for


def deformable_snapshots(session, frame):
    session.refresh()
    scene = session.scenes()[-1]
    session.seek(scene.id, frame)
    return session.geometry_snapshot(), session.overlay_snapshot(4), scene.instance_id


def assert_deformable_samples(snapshots, obj, frame):
    geometry, overlay, scene_id = snapshots
    family = obj["type"]
    fields = ["collisionPositions", "simulationPositions"] if family == "PxDeformableVolume" else ["positions"]
    expected = [np.array(payloads(obj, family + "." + field, "f")[frame - 1]).reshape(-1, 4)[:, :3]
                for field in fields]
    meshes = {mesh.id: mesh for mesh in geometry.meshes}
    resources = geometry.resource_ids[geometry.scene_ids == scene_id]
    arrows = overlay.positions[overlay.scene_ids == scene_id]
    assert len(resources) == len(expected)
    for resource, positions in zip(resources, expected):
        np.testing.assert_allclose(meshes[int(resource)].vertices, positions, atol=1e-6)
    velocities = np.array(payloads(obj, family + ".velocities", "f")[frame - 1]).reshape(-1, 4)[:, :3]
    np.testing.assert_allclose(arrows[:, 0], expected[-1], atol=1e-6)
    np.testing.assert_allclose(arrows[:, 1], expected[-1] + velocities / 10, atol=1e-6)
    assert not geometry.diagnostics and not overlay.diagnostics


@pytest.fixture
def gpu_sender():
    value = os.environ.get("OVDNEXT_SDK_GPU_SENDER")
    if not value:
        if os.environ.get("OVDNEXT_REQUIRE_SDK_GPU_SENDER") == "1":
            pytest.fail("Required genuine GPU SDK sender is missing")
        pytest.skip("Genuine GPU-family interoperability not run: OVDNEXT_SDK_GPU_SENDER is unset")
    path = Path(value)
    assert path.is_file(), "Genuine pinned GPU sender has not been built"
    return path


@pytest.mark.parametrize("secondary", [False, True])
@pytest.mark.parametrize("kind", ["PxPBDParticleSystem", "PxParticleBuffer", "PxParticleAndDiffuseBuffer",
                                  "PxDeformableSurface", "PxDeformableVolume"])
def test_authentic_gpu_family_selection_and_payloads(tmp_path, gpu_sender, kind, secondary):
    with sdk_capture(tmp_path, gpu_sender, "gpu-families") as capture:
        session = capture.session
        info = session.capture_control_info()
        rows = session.capture_catalog(info["generation"])["entries"]
        targets = {row["type_name"]: row["target_handle"] for row in rows if row["can_select"]}
        assert all(family in targets for family in ("PxPBDParticleSystem", "PxParticleBuffer",
            "PxParticleAndDiffuseBuffer", "PxDeformableSurface", "PxDeformableVolume"))
        assert not session.refresh_live().ready
        assert session.request_capture([], [targets[kind]], secondary, False, info["generation"]) == 1
        assert capture.settle()["actual_secondary"] == secondary
        deformable = kind.startswith("PxDeformable")
        capture.send("step")
        wait_for(lambda: session.info().completed_serial >= 2)
        if deformable:
            # ACK precedes publication; inspect only after STOP_FRAME.
            live_first = deformable_snapshots(session, 1)
            live_second = deformable_snapshots(session, 2)
        detailed = capture.finish()["segments"][1]
        producer = json.loads(capture.report.read_text())
        assert producer["sdk_diagnostics"] == 0
        assert producer["gpu_device"] and producer["cuda_driver_version"] > 0
    selected = detailed[str(targets[kind])]
    assert selected["type"] == kind
    if deformable:
        assert_deformable_samples(live_first, selected, 1)
        assert_deformable_samples(live_second, selected, 2)
        assert not np.array_equal(live_first[1].positions[:, 0], live_second[1].positions[:, 0])
        # The file must render the same sampled state as the live connection.
        reopened = Session(str(tmp_path / "capture.ovd"), str(tmp_path / "reopened"))
        for frame in (1, 2, 1):
            assert_deformable_samples(deformable_snapshots(reopened, frame), selected, frame)
    families = ["PxParticleBuffer", "PxParticleAndDiffuseBuffer"] if kind == "PxPBDParticleSystem" else [kind]
    for family in families:
        obj = detailed[str(targets[family])]
        if family.startswith("PxParticle"):
            for attribute, fmt in (("positionInvMasses", "f"), ("velocities", "f"), ("phases", "I")):
                samples = payloads(obj, "PxParticleBuffer." + attribute, fmt)
                assert any(len(sample) == (256 if fmt == "f" else 64) for sample in samples)
                assert all(math.isfinite(value) for sample in samples for value in sample)
            if family == "PxParticleAndDiffuseBuffer":
                for attribute in ("diffusePositionLifeTime", "diffuseVelocities"):
                    samples = payloads(obj, family + "." + attribute, "f")
                    assert bool(samples) == secondary
                    if secondary:
                        assert any(len(sample) >= 4 for sample in samples)
                        assert all(math.isfinite(value) for sample in samples for value in sample)
        elif family == "PxDeformableSurface":
            for attribute in ("positions", "velocities"):
                samples = payloads(obj, family + "." + attribute, "f")
                assert any(len(sample) >= 12 for sample in samples)
                assert all(math.isfinite(value) for sample in samples for value in sample)
        elif family == "PxDeformableVolume":
            for attribute in ("collisionPositions", "simulationPositions", "velocities"):
                samples = payloads(obj, family + "." + attribute, "f")
                assert any(len(sample) >= 16 for sample in samples)
                assert all(math.isfinite(value) for sample in samples for value in sample)
    for family in ("PxParticleBuffer", "PxParticleAndDiffuseBuffer", "PxDeformableSurface", "PxDeformableVolume"):
        if family not in families:
            assert str(targets[family]) not in detailed


@pytest.mark.parametrize("kind", ["PxDeformableSurface", "PxDeformableVolume"])
def test_selected_deformable_segments_and_reconnect_keep_samples_isolated(tmp_path, gpu_sender, kind):
    for connection in range(2):
        folder = tmp_path / str(connection)
        with sdk_capture(folder, gpu_sender, "gpu-families") as capture:
            session = capture.session
            info = session.capture_control_info()
            rows = session.capture_catalog(info["generation"])["entries"]
            target = next(row["target_handle"] for row in rows if row["type_name"] == kind and row["can_select"])
            snapshots = []
            for segment in range(2):
                assert session.request_capture([], [target], False, False, info["generation"]) == segment + 1
                assert capture.settle()["capture_epoch"] == segment + 1
                capture.send("step")
                wait_for(lambda: session.info().completed_serial >= 2 + segment * 3)
                snapshots.append(deformable_snapshots(session, 2))
            assert snapshots[0][0].segment_epoch != snapshots[1][0].segment_epoch
            decoded = capture.finish()["segments"]
            assert json.loads(capture.report.read_text())["sdk_diagnostics"] == 0
        for segment, snapshot in enumerate(snapshots, 1):
            assert_deformable_samples(snapshot, decoded[segment][str(target)], 2)
        reopened = Session(str(folder / "capture.ovd"), str(folder / "reopened"))
        assert_deformable_samples(deformable_snapshots(reopened, 2), decoded[2][str(target)], 2)
        assert_deformable_samples(snapshots[0], decoded[1][str(target)], 2)

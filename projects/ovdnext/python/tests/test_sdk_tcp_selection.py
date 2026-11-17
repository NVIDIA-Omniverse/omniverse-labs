# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real pinned SDK producer interoperability (required lane never silently skips)."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from contextlib import contextmanager
import struct

import pytest
from _pvd_native import Session


@pytest.fixture
def sdk_sender():
    path = os.environ.get("OVDNEXT_SDK_TCP_SENDER")
    if not path:
        if os.environ.get("OVDNEXT_REQUIRE_SDK_TCP_SENDER") == "1":
            pytest.fail("Required genuine SDK sender is missing; set OVDNEXT_SDK_TCP_SENDER")
        pytest.skip("Genuine SDK TCP interoperability not run: OVDNEXT_SDK_TCP_SENDER is unset")
    sender = Path(path)
    assert sender.is_file(), "Genuine pinned SDK sender has not been built"
    return sender


def wait_for(check, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(.01)
    pytest.fail("Genuine SDK capture did not reach the expected state")


class SdkCapture:
    def __init__(self, session, process, sender, recording, report):
        self.session, self.process, self.sender = session, process, sender
        self.recording, self.report = recording, report
        self.observations = []

    def send(self, command):
        assert self.process.poll() is None, self.process.communicate()
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def wait(self, check, seconds=10):
        def running():
            assert self.process.poll() is None, self.process.communicate()
            return check()
        return wait_for(running, seconds)

    def settle(self, expected="applied", command="poll"):
        wait_for(lambda: self.session.capture_control_info()["state"] != "sending")
        def poll():
            self.send(command)
            info = self.session.capture_control_info()
            assert info["state"] in ("awaiting_ack", expected), info
            return info if info["state"] == expected else None
        info = wait_for(poll)
        self.observations.append(info)
        return info

    def rows(self):
        info = self.session.capture_control_info()
        page = self.session.capture_catalog(info["generation"])
        self.observations.append(page)
        return {r["name"]: r for r in page["entries"]}

    def finish(self):
        output, error = self.process.communicate("quit\n", timeout=30)
        self.report.with_name("sender.stdout.log").write_text(output)
        self.report.with_name("sender.stderr.log").write_text(error)
        assert self.process.returncode == 0, output + error
        wait_for(lambda: self.session.info().ended)
        self.session.refresh()
        assert self.session.info().error == ""
        self.session.stop_live()
        decoded = json.loads(subprocess.check_output(
            [str(self.sender), "--inspect", str(self.recording)], text=True))
        evidence = {"sender_sha256": hashlib.sha256(self.sender.read_bytes()).hexdigest(),
                    "ovd_sha256": hashlib.sha256(self.recording.read_bytes()).hexdigest(),
                    "decoded": decoded, "producer": json.loads(self.report.read_text()),
                    "observations": self.observations}
        self.report.with_name("evidence.json").write_text(json.dumps(evidence, indent=2))
        return decoded


@contextmanager
def sdk_capture(tmp_path, sender, scenario="schema-lifetimes"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    recording, report = tmp_path / "capture.ovd", tmp_path / "sender.json"
    session = Session.listen(port, str(recording))
    process = subprocess.Popen([str(sender), "--host", "127.0.0.1", "--port", str(port),
                                "--scenario", scenario, "--report", str(report)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_for(lambda: session.capture_control_info()["catalog_ready"] or process.poll() is not None)
        assert process.poll() is None, process.communicate()
        yield SdkCapture(session, process, sender, recording, report)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
        session.stop_live()


def payloads(obj, field, fmt):
    return [struct.unpack("<" + fmt * (len(raw) // struct.calcsize("<" + fmt)), raw)
            for raw in map(bytes.fromhex, obj["attributes"].get(field, []))]


def test_authentic_selected_then_all_ack_frames_and_saved_payloads(tmp_path, sdk_sender):
    with sdk_capture(tmp_path, sdk_sender) as capture:
        session = capture.session
        info, rows = session.capture_control_info(), capture.rows()
        assert info["capabilities"] == 12 and info["issued_high_water"] == 0
        assert not session.refresh_live().ready
        box, ground = rows["box"]["target_handle"], rows["ground"]["target_handle"]
        assert session.request_capture([], [box], False, False, info["generation"]) == 1
        first = capture.settle()
        assert first["ack_id"] == 1 and first["capture_epoch"] == 1
        assert not first["actual_all"] and not first["actual_secondary"]
        assert not session.refresh_live().ready
        capture.send("step")
        wait_for(lambda: session.info().completed_serial == 2)
        assert session.refresh_live().segment_transition_serial == 1
        assert session.request_capture([], [], True, True, 0) == 2
        second = capture.settle()
        assert second["ack_id"] == 2 and second["capture_epoch"] == 2
        assert second["actual_all"] and second["actual_secondary"]
        # Genuine resetCaptureObjects closes the old segment's open frame.
        assert session.info().completed_serial == 3
        assert session.capture_control_info()["segment_transition_serial"] == 1
        capture.send("step")
        wait_for(lambda: session.info().completed_serial == 5)
        assert session.refresh_live().segment_transition_serial == 2
        decoded = capture.finish()
    assert decoded["stop_frames"] == 5 and decoded["reader_eof"]
    discovery, selected, all_objects = decoded["segments"]
    assert str(box) not in discovery and str(ground) not in selected
    assert selected[str(box)]["type"] == "PxRigidDynamic"
    assert payloads(selected[str(box)], "PxRigidActor.globalPose", "f")
    assert payloads(selected[str(box)], "PxRigidBody.linearVelocity", "f")
    assert "PxRigidBody.force" not in selected[str(box)]["attributes"]
    assert all_objects[str(ground)]["type"] == "PxRigidStatic"
    exported = tmp_path / "exported.ovd"
    assert session.save(str(exported)) == capture.recording.stat().st_size
    assert exported.read_bytes() == capture.recording.read_bytes()
    reopened = Session(str(exported), str(tmp_path / "reopened"))
    assert len(reopened.scenes()) == 2


@pytest.mark.parametrize("secondary", [False, True])
@pytest.mark.parametrize("selection", ["box", "ground", "arm", "scene_union"])
def test_authentic_cpu_families_policy_and_contact_endpoints(tmp_path, sdk_sender, secondary, selection):
    with sdk_capture(tmp_path, sdk_sender, "cpu-selection") as capture:
        session, rows = capture.session, capture.rows()
        handles = {name: row["target_handle"] for name, row in rows.items()}
        generation = session.capture_control_info()["generation"]
        scenes = [handles["other scene"]] if selection == "scene_union" else []
        objects = [handles["box" if selection == "scene_union" else selection]]
        assert session.request_capture(scenes, objects, secondary, False, generation) == 1
        info = capture.settle()
        assert info["actual_secondary"] == secondary and not info["actual_all"]
        capture.send("force")
        capture.send("step")
        wait_for(lambda: session.info().completed_serial >= 2)
        detailed = capture.finish()["segments"][1]
    if selection in ("box", "scene_union"):
        box = detailed[str(handles["box"])]
        assert payloads(box, "PxRigidActor.globalPose", "f")
        assert payloads(box, "PxRigidBody.linearVelocity", "f")
        forces = payloads(box, "PxRigidBody.force", "f")
        assert ((3., 0., 0.) in forces) == secondary
        assert payloads(box, "add:PxRigidActor.shapes", "Q")
        assert any(o["type"] == "PxMaterial" for o in detailed.values())
    if selection == "scene_union":
        assert str(handles["other box"]) in detailed and str(handles["arm"]) not in detailed
    elif selection == "arm":
        arm = detailed[str(handles["arm"])]
        links = [o for o in detailed.values() if o["type"] == "PxArticulationLink"]
        joints = [o for o in detailed.values() if o["type"] == "PxArticulationJointReducedCoordinate"]
        assert len(links) == 2 and len(joints) == 1
        assert len(payloads(arm, "add:PxArticulationReducedCoordinate.links", "Q")) == 2
        for link in links:
            assert payloads(link, "PxRigidActor.globalPose", "f")
            assert payloads(link, "PxRigidBody.linearVelocity", "f")
        assert payloads(joints[0], "PxArticulationJointReducedCoordinate.jointPosition", "f")
        assert payloads(joints[0], "PxArticulationJointReducedCoordinate.jointVelocity", "f")
        # CPU solver reactions are recorded; applied joint-force arrays are not.
        reactions = payloads(joints[0], "PxArticulationJointReducedCoordinate.incomingJointForce", "f")
        assert bool(reactions) == secondary
        if secondary:
            assert all(len(value) == 6 for value in reactions)
        assert ("PxArticulationJointReducedCoordinate.driveTarget" in joints[0]["attributes"]) == secondary
        assert "PxArticulationJointReducedCoordinate.jointForce" not in joints[0]["attributes"]
    if selection in ("box", "ground"):
        scene = detailed[str(handles["selection fixture"])]
        pairs = payloads(scene, "PxScene.pairsActors", "Q")
        points = payloads(scene, "PxScene.pairsContactPoints", "f")
        if secondary:
            assert any(set(pair) == {handles["box"], handles["ground"]} for pair in pairs)
            assert any(len(value) >= 3 for value in points)
            for field in ("pairsContactNormals", "pairsContactSeparations", "pairsContactImpulses"):
                assert any(payloads(scene, "PxScene." + field, "f"))
            for actor in ("box", "ground"):
                assert payloads(detailed[str(handles[actor])], "PxRigidActor.globalPose", "f")
            assert payloads(detailed[str(handles["box"])], "PxRigidBody.linearVelocity", "f")
        else:
            assert not pairs and not points
            assert str(handles["ground" if selection == "box" else "box"]) not in detailed


def test_authentic_busy_denied_narrowed_and_retired_requests(tmp_path, sdk_sender):
    with sdk_capture(tmp_path, sdk_sender, "cpu-selection") as capture:
        session, rows = capture.session, capture.rows()
        generation = session.capture_control_info()["generation"]
        box, ground = rows["box"]["target_handle"], rows["ground"]["target_handle"]
        assert session.request_capture([], [box], True, False, generation) == 1
        assert capture.settle("busy", "busy")["capture_epoch"] == 0
        capture.send("deny")
        assert session.request_capture([], [box], True, False, generation) == 2
        assert capture.settle("denied")["capture_epoch"] == 0
        capture.send("narrow")
        assert session.request_capture([], [], True, True, 0) == 3
        narrowed = capture.settle()
        assert not narrowed["actual_secondary"] and not narrowed["actual_all"]
        capture.send("step")
        wait_for(lambda: session.info().completed_serial >= 2)
        capture.send("allow")
        capture.send("retire")  # SDK lifetime invalidates an otherwise current catalog handle.
        assert session.request_capture([], [box], False, False, generation) == 4
        assert capture.settle("invalid")["capture_epoch"] == 1
        assert session.request_capture_catalog() == 5
        refreshed = capture.settle()
        assert refreshed["generation"] > generation and refreshed["capture_epoch"] == 1
        assert "box" not in capture.rows()
        with pytest.raises((ValueError, RuntimeError)):
            session.request_capture([], [box], False, False, generation)
        detailed = capture.finish()["segments"][1]
    assert str(ground) in detailed and str(box) not in detailed


def test_authentic_scope_context_refresh_and_reconnect(tmp_path, sdk_sender):
    for connection in range(2):
        with sdk_capture(tmp_path / str(connection), sdk_sender, "scoped") as capture:
            session, rows = capture.session, capture.rows()
            scene = rows["selection fixture"]
            assert not scene["can_select"] and rows["box"]["can_select"]
            assert "ground" not in rows and "other scene" not in rows
            assert session.request_capture_catalog() == 1  # New connection owns fresh IDs.
            refreshed = capture.settle()
            assert refreshed["generation"] == 2 and refreshed["capture_epoch"] == 0
            assert not session.refresh_live().ready
            with pytest.raises((ValueError, RuntimeError)):
                session.request_capture([scene["target_handle"]], [], False, False, 2)
            capture.send("escape")
            assert session.request_capture([], [], True, True, 0) == 2
            assert capture.settle("denied")["capture_epoch"] == 0
            capture.send("allow")
            assert session.request_capture([], [], False, True, 0) == 3
            assert not capture.settle()["actual_all"]  # All permitted, not all SDK objects.
            capture.send("step")
            wait_for(lambda: session.info().completed_serial == 2)
            detailed = capture.finish()["segments"][1]
        assert detailed[str(rows["box"]["target_handle"])]["type"] == "PxRigidDynamic"
        assert sum(o["type"] == "PxRigidDynamic" for o in detailed.values()) == 1


def test_authentic_local_restart_drops_pending_ack_until_explicit_supersession(tmp_path, sdk_sender):
    with sdk_capture(tmp_path, sdk_sender) as capture:
        session = capture.session
        assert session.request_capture([], [], False, True, 0) == 1
        wait_for(lambda: session.capture_control_info()["state"] == "awaiting_ack")
        capture.send("drop-and-restart")
        capture.wait(lambda: session.capture_control_info()["metadata_boundary_serial"] == 1)
        capture.send("step")
        wait_for(lambda: session.info().completed_serial == 2)
        wait_for(lambda: session.capture_control_info()["state"] == "unknown", seconds=12)
        info = session.capture_control_info()
        assert info["ack_id"] == 0 and info["forward_bytes"] > 0
        with pytest.raises(RuntimeError):
            session.request_capture([], [], True, True, 0)
        assert session.request_capture([], [], True, True, 0, supersede_unknown=True) == 2
        settled = capture.settle()
        assert settled["ack_id"] == 2 and settled["results"][0]["state"] == "superseded"
        capture.send("step")
        wait_for(lambda: session.info().completed_serial == 5)
        assert len(capture.finish()["segments"]) == 3


@pytest.mark.parametrize("restart", [False, True])
def test_authentic_discovery_and_same_stream_schema_lifetimes(tmp_path, sdk_sender, restart):
    """Catch incompatible schema registration, dropped restart frames and corrupt saves."""
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    recording, report = tmp_path / "authentic.ovd", tmp_path / "sender.json"
    session = Session.listen(port, str(recording))
    commands = "start\nstep\nstep\nstep\nstep\nstep\nrestart\nstep\nstep\nquit\n" if restart else "quit\n"
    try:
        result = subprocess.run([str(sdk_sender), "--host", "127.0.0.1", "--port", str(port),
                                 "--scenario", "schema-lifetimes", "--report", str(report)],
                                input=commands, text=True, capture_output=True, timeout=45)
        assert result.returncode == 0, result.stdout + result.stderr
        deadline = time.monotonic() + 10
        while not session.info().ended and time.monotonic() < deadline:
            time.sleep(0.01)
        assert session.info().ended
        session.refresh()
        assert session.info().recording_size > 0
        assert session.info().error == ""
        # The SDK records pre/post frames; no-frame EOF publishes a final serial,
        # but it must not invent a viewport frame.
        assert session.info().completed_serial == (14 if restart else 1)
        if restart:
            assert len(session.scenes()) >= 2
        else:
            assert session.info().live_max_frame == 0
            assert not session.refresh_live().ready
    finally:
        session.stop_live()
    decoded = json.loads(subprocess.check_output([str(sdk_sender), "--inspect", str(recording)], text=True))
    assert decoded["reader_eof"] is True
    assert decoded["duplicate_attributes"] == 0
    assert decoded["duplicate_classes"] == 0
    assert decoded["stop_frames"] == (14 if restart else 0)
    assert len(decoded["metadata_class_handles"]) == (3 if restart else 1)
    assert len(decoded["catalog_class_handles"]) == (3 if restart else 1)
    assert len(decoded["metadata_instances"]) == (3 if restart else 1)
    if restart:
        reopened = Session(str(recording), str(tmp_path / "reopened"))
        assert len(reopened.scenes()) >= 2
    provenance = {"sender_sha256": hashlib.sha256(sdk_sender.read_bytes()).hexdigest(),
                  "ovd_sha256": hashlib.sha256(recording.read_bytes()).hexdigest(),
                  "decoded": decoded, "producer": json.loads(report.read_text())}
    (tmp_path / "evidence.json").write_text(json.dumps(provenance, indent=2))

# SPDX-License-Identifier: Apache-2.0
from dataclasses import FrozenInstanceError
from types import SimpleNamespace as NS
import threading
import pytest
from ovrtx_blender_example import physics_kinematic_inputs as inputs
from ovrtx_blender_example.shared_stage_composition import BodyPose


def test_copied_input_is_immutable_and_finite():
    values = [BodyPose("/Kin", (1., 2., 3.), (0., 0., 0., 1.))]
    copied = inputs.KinematicInputs("gen", 1.25, values)
    values.clear()
    assert len(copied.poses) == 1
    with pytest.raises(FrozenInstanceError):
        copied.frame = 2
    with pytest.raises(ValueError, match="finite"):
        inputs.KinematicInputs("gen", float("nan"))
    nonunit = inputs.KinematicInputs("gen", 1, (BodyPose("/Kin", (0., 0., 0.), (0., 0., 0., 2.)),))
    assert nonunit.poses[0].orient == (0., 0., 0., 1.)
    with pytest.raises(ValueError, match="typed"):
        inputs.KinematicInputs("gen", 1, (NS(translate=(0, 0, 0)),))


@pytest.mark.parametrize("case", ("stale", "missing", "dynamic", "duplicate", "closed"))
def test_bad_input_cannot_reach_native_transport(monkeypatch, case):
    monkeypatch.setattr(inputs, "kinematic_targets", lambda generation: (NS(body_path="/Kin"),))
    controller = NS(config=NS(body_prims=("/Dynamic",)), started=True, failed=False, closed=False)
    pose = BodyPose("/Kin", (0., 0., 0.), (0., 0., 0., 1.))
    generation = NS(digest="gen")
    values, digest = (pose,), "gen"
    if case == "stale":
        digest = "previous"
    elif case == "missing":
        values = ()
    elif case == "dynamic":
        controller.config.body_prims = ("/Kin",)
    elif case == "duplicate":
        values = (pose, pose)
    else:
        controller.closed = True
    with pytest.raises(ValueError):
        inputs.apply_kinematic_inputs(controller, inputs.KinematicInputs(digest, 2, values), generation)


def test_off_thread_evaluation_rejected_before_reading_rna():
    errors = []
    def attempt():
        try:
            inputs.capture_kinematic_inputs(None, None)
        except ValueError as error:
            errors.append(str(error))
    thread = threading.Thread(target=attempt)
    thread.start()
    thread.join()
    assert errors == ["Kinematic evaluation requires the main thread"]


def test_native_schema_must_be_kinematic_even_if_descriptor_claims_it(tmp_path, monkeypatch):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdPhysics
    path = tmp_path / "dynamic.usda"
    stage = Usd.Stage.CreateNew(str(path))
    rigid = UsdPhysics.RigidBodyAPI.Apply(stage.DefinePrim("/Kin", "Xform"))
    rigid.CreateKinematicEnabledAttr().Set(False)
    stage.GetRootLayer().Save()
    monkeypatch.setattr(inputs, "kinematic_targets", lambda generation: (NS(body_path="/Kin"),))
    controller = NS(config=NS(body_prims=(), input_usd_path=str(path)), started=True, failed=False, closed=False)
    copied = inputs.KinematicInputs("gen", 2, (BodyPose("/Kin", (0., 0., 0.), (0., 0., 0., 1.)),))
    with pytest.raises(ValueError, match="not an enabled native kinematic"):
        inputs.apply_kinematic_inputs(controller, copied, NS(digest="gen"))


def test_partial_native_write_requires_reset_not_retry(monkeypatch):
    monkeypatch.setattr(inputs, "kinematic_targets", lambda generation: (NS(body_path="/Kin"),))
    calls = []
    def write(poses, **kwargs):
        calls.append((poses, kwargs))
        return {"body_count": 0, "grpc_status": "OK"}
    controller = NS(config=NS(body_prims=()), started=True, failed=False, closed=False,
        current_simulation_time_ns=100, _validated_kinematic_roster=("gen", ("/Kin",)),
        _physics_client=NS(write_body_poses=write))
    copied = inputs.KinematicInputs("gen", 2, (BodyPose("/Kin", (0., 0., 0.), (0., 0., 0., 1.)),))
    with pytest.raises(ValueError, match="accepted 0 of 1"):
        inputs.apply_kinematic_inputs(controller, copied, NS(digest="gen"))
    assert controller.failed and controller.completed and "accepted 0" in controller.last_error
    with pytest.raises(ValueError, match="active simulation"):
        inputs.apply_kinematic_inputs(controller, copied, NS(digest="gen"))
    assert len(calls) == 1 and calls[0][1] == {"simulation_time_ns": 101, "reset": False}

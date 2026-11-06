from ovrtx_blender_example.physics_capabilities import joint_capability, _admitted_devices
import pytest
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from ovrtx_blender_example import physics_capabilities as admission


@pytest.mark.parametrize('metadata', [
    {'metersPerUnit': 1.0, 'upAxis': 'Z', 'kilogramsPerUnit': 0.5,
     'defaultPrim': 'World', 'startTimeCode': -3.0, 'endTimeCode': 57.5,
     'timeCodesPerSecond': 48.0, 'framesPerSecond': 24.0},
    {'metersPerUnit': 0.025, 'upAxis': 'Y', 'framesPerSecond': 30.0},
    {},
])
def test_gpu_device_wrapper_preserves_source_interpretation_and_retires_old_cache(tmp_path, metadata):
    import hashlib
    from pxr import Usd, UsdGeom, UsdPhysics
    from ovrtx_blender_example import scene_generation

    exported = tmp_path / 'export.usda'
    source = Usd.Stage.CreateNew(str(exported))
    source.DefinePrim('/World')
    UsdPhysics.Scene.Define(source, '/World/Physics')
    for key, value in metadata.items():
        source.SetMetadata(key, value)
    source.GetRootLayer().Save()
    binding = tmp_path / 'identity.usda'
    binding.write_text('#usda 1.0\n')
    stock = tmp_path / 'stock.usda'
    scene_generation._write_stock_root(stock, binding, exported)
    composed = tmp_path / 'composed.usda'
    scene_generation._write_layered_generation(composed, (), (), stock)
    original = composed.read_bytes()
    legacy_digest = hashlib.sha256(str(composed.resolve()).encode()).hexdigest()[:16]
    legacy = tmp_path / f'physics-device-gpu-{legacy_digest}.usda'
    legacy.write_text('#usda 1.0\n')

    assert admission.physics_device_stage(str(composed), 'cpu') == str(composed)
    gpu_path = Path(admission.physics_device_stage(str(composed), 'gpu'))
    assert gpu_path != legacy
    gpu = Usd.Stage.Open(str(gpu_path))
    assert UsdGeom.GetStageMetersPerUnit(gpu) == UsdGeom.GetStageMetersPerUnit(source)
    assert UsdGeom.GetStageUpAxis(gpu) == UsdGeom.GetStageUpAxis(source)
    assert gpu.GetTimeCodesPerSecond() == source.GetTimeCodesPerSecond()
    assert gpu.GetFramesPerSecond() == source.GetFramesPerSecond()
    for key in ('metersPerUnit', 'upAxis', 'kilogramsPerUnit', 'defaultPrim',
                'startTimeCode', 'endTimeCode', 'timeCodesPerSecond', 'framesPerSecond'):
        assert gpu.GetRootLayer().pseudoRoot.HasInfo(key) == (key in metadata)
        assert gpu.GetMetadata(key) == source.GetMetadata(key)
    assert gpu.GetPrimAtPath('/World/Physics').GetAttribute('physxScene:enableGPUDynamics').Get() is True
    assert gpu.GetPrimAtPath('/World/Physics').GetAttribute('physxScene:broadphaseType').Get() == 'GPU'
    before = gpu_path.stat().st_mtime_ns
    assert admission.physics_device_stage(str(composed), 'gpu') == str(gpu_path)
    assert gpu_path.stat().st_mtime_ns == before
    assert composed.read_bytes() == original
    assert legacy.read_text() == '#usda 1.0\n'


@pytest.mark.parametrize("server_digest,client_digest,accepted", [
    (admission.SERVER_SHA256, admission.CLIENT_SHA256, True),
    (admission.INSTALLED_SERVER_SHA256, admission.INSTALLED_CLIENT_SHA256, True),
    (admission.CURRENT_SERVER_SHA256, admission.CURRENT_CLIENT_SHA256, True),
    (admission.SERVER_SHA256, admission.INSTALLED_CLIENT_SHA256, False),
    (admission.INSTALLED_SERVER_SHA256, admission.CLIENT_SHA256, False),
    (admission.CURRENT_SERVER_SHA256, admission.INSTALLED_CLIENT_SHA256, False),
    (admission.INSTALLED_SERVER_SHA256, admission.CURRENT_CLIENT_SHA256, False),
    ("0" * 64, admission.INSTALLED_CLIENT_SHA256, False),
])
def test_joint_runtime_admission_requires_a_measured_binary_pair(
    monkeypatch, server_digest, client_digest, accepted,
):
    server = str(Path(sys.executable).resolve())
    client = SimpleNamespace(
        __file__="client.pyd",
        capabilities=lambda: {"rpcs": (
            "CreateSimulation", "ReadWorldState", "WriteWorldState", "DeleteSimulation")},
    )
    config = SimpleNamespace(
        server=server, ovphysx_worker_command=subprocess.list2cmdline([server]), device="cpu")
    monkeypatch.setattr(admission, "binary_digest",
                        lambda path: server_digest if Path(path).resolve() == Path(server) else client_digest)
    if accepted:
        admission.validate_fixed_runtime(config, lambda: client)
    else:
        with pytest.raises(ValueError, match="no measured joint/device admission"):
            admission.validate_fixed_runtime(config, lambda: client)


def test_generic_rpc_availability_does_not_admit_live_motors():
    _admitted_devices.clear()
    status = joint_capability("D6", "gpu")
    assert status.authorable and not status.restart_simulation and not status.live
    _admitted_devices.add("gpu")
    try:
        status = joint_capability("D6", "gpu")
        assert status.authorable and status.restart_simulation and not status.live
        assert not joint_capability("D6", "cpu").restart_simulation
        assert not joint_capability("SPHERICAL", "gpu").authorable
        assert not joint_capability("DISTANCE", "gpu").restart_simulation
    finally:
        _admitted_devices.clear()


def test_gpu_rotated_frame_policy_uses_actual_usd_for_every_producer():
    pxr = pytest.importorskip("pxr")
    from pxr import Usd, UsdPhysics, Gf
    from ovrtx_blender_example.physics_capabilities import validate_gpu_joint_frames
    stage = Usd.Stage.CreateInMemory()
    joint = UsdPhysics.FixedJoint.Define(stage, "/Fixed")
    joint.CreateBody1Rel().SetTargets(["/Body"])
    joint.CreateLocalRot1Attr(Gf.Quatf(.9238795, 0, 0, .3826834))
    with pytest.raises(ValueError, match="select CPU explicitly"):
        validate_gpu_joint_frames(joint.GetPrim())
    joint.CreateLocalRot0Attr(Gf.Quatf(.9238795, 0, 0, .3826834))
    with pytest.raises(ValueError, match="select CPU explicitly"):
        validate_gpu_joint_frames(joint.GetPrim())
    joint.CreateLocalRot0Attr(Gf.Quatf(1))
    joint.CreateBody0Rel().SetTargets(["/Other"])
    with pytest.raises(ValueError, match="select CPU explicitly"):
        validate_gpu_joint_frames(joint.GetPrim())
    joint.CreateLocalRot1Attr(Gf.Quatf(-1))  # same identity rotation, antipodal sign
    with pytest.raises(ValueError, match="body-to-body"):
        validate_gpu_joint_frames(joint.GetPrim())
    joint.GetBody0Rel().ClearTargets(True)
    validate_gpu_joint_frames(joint.GetPrim())

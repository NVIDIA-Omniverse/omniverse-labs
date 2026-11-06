# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measured joint admission, separate from generic RPC transport capabilities."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import shlex
import json

# Blender-native acceptance: public/tests/blender_native_joints.py. These are
# exact Windows worker/client pairs measured for CPU and GPU joint execution.
SERVER_SHA256 = "da1d1daa7c671f3db9282255c8dc1f2cac1e13b8645ce5790a292ae17dee1e39"
CLIENT_SHA256 = "669897ffc5b0c5def898e33985b2e28a716b94bf93557c78eb75a3d0d24ec4fe"
INSTALLED_SERVER_SHA256 = "f68df0704acaa3a8f1e484f9a960863b805c0cd127697e8fc20082adcb03fae6"
INSTALLED_CLIENT_SHA256 = "4f561e39260899b784dfe15a2c4a876fa379b1953b2c3f445e4ea297f59de991"
CURRENT_SERVER_SHA256 = "fc9dd7430429c65a9057c2229918bd2b28471f71cd09133a510fa48874fb21b9"
CURRENT_CLIENT_SHA256 = "c92f9d02a387a1f6dac32c1ae7350c4052a7973d44dddfb45993206371f2a80e"
MEASURED_RUNTIME_PAIRS = frozenset({
    (SERVER_SHA256, CLIENT_SHA256),
    (INSTALLED_SERVER_SHA256, INSTALLED_CLIENT_SHA256),
    (CURRENT_SERVER_SHA256, CURRENT_CLIENT_SHA256),
})
JOINT_TYPES = {"PhysicsFixedJoint": "FIXED", "PhysicsRevoluteJoint": "REVOLUTE",
               "PhysicsPrismaticJoint": "PRISMATIC", "PhysicsJoint": "D6"}


@dataclass(frozen=True)
class JointCapability:
    authorable: bool
    restart_simulation: bool
    live: bool
    reason: str


_admitted_devices = set()
_body_admitted_devices = set()
_kinematic_admitted_devices = set()


def kinematic_capability(device=None):
    device = device or os.environ.get("OV_BLENDER_EXAMPLE_OVPHYSX_DEVICE", "cpu")
    admitted = device in _kinematic_admitted_devices
    return JointCapability(True, admitted, admitted,
        "Evaluated pose targets before each step; dynamic output remains separate" if admitted
        else "Authorable; kinematic input requires validated installed runtime")


def body_control_capability(name, device=None):
    device = device or os.environ.get("OV_BLENDER_EXAMPLE_OVPHYSX_DEVICE", "cpu")
    supported = {"BOX", "CAPSULE", "mass", "mass_mode", "mass_kg", "density", "center_of_mass",
                 "inertia", "diagonal_inertia", "principal_axes", "velocities", "initial_linear_velocity",
                 "initial_angular_velocity", "damping", "linear_damping", "angular_damping", "material",
                 "static_friction", "dynamic_friction", "restitution", "gravity", "gravity_direction", "gravity_magnitude"}
    if name not in supported:
        return JointCapability(False, False, False, "Unavailable: this field has no validated runtime admission")
    return JointCapability(True, device in _body_admitted_devices, False,
        "Simulation on explicit restart" if device in _body_admitted_devices else "Authorable; installed runtime admission pending")


def joint_capability(kind, device=None):
    device = device or os.environ.get("OV_BLENDER_EXAMPLE_OVPHYSX_DEVICE", "cpu")
    if kind not in JOINT_TYPES.values():
        return JointCapability(False, False, False, "Unsupported joint type; runtime validation required")
    if device not in _admitted_devices:
        return JointCapability(True, False, False, "Authorable; installed runtime admission pending")
    return JointCapability(True, True, False, "Simulation on explicit restart; live motors unavailable")


@lru_cache(maxsize=16)
def _digest(path, modified, size):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binary_digest(path):
    path = Path(path).resolve()
    info = path.stat()
    return _digest(str(path), info.st_mtime_ns, info.st_size)


def physics_device_stage(usd_path, device):
    """Private immutable device opinions; --device gpu alone does not enable GPU."""
    source = Path(usd_path).resolve()
    if device != "gpu" or not source.is_file():
        return str(usd_path)
    from pxr import Sdf, Usd, UsdPhysics
    from .usd_stage_metadata import copy_authored_stage_metadata
    stage = Usd.Stage.Open(str(source))
    scenes = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if not scenes:
        raise ValueError("GPU simulation requires an authored PhysicsScene")
    # Retire old cached wrappers which omitted root stage interpretation.
    digest = hashlib.sha256(("stage-metadata-v1:" + str(source)).encode()).hexdigest()[:16]
    destination = source.parent / f"physics-device-gpu-{digest}.usda"
    if destination.is_file():
        return str(destination)
    layer = Sdf.Layer.CreateAnonymous("physics-device.usda")
    copy_authored_stage_metadata(stage.GetRootLayer(), layer)
    layer.subLayerPaths = [str(source)]
    device_stage = Usd.Stage.Open(layer)
    for path in scenes:
        prim = device_stage.OverridePrim(path)
        prim.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["PhysxSceneAPI"]))
        prim.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute("physxScene:broadphaseType", Sdf.ValueTypeNames.Token).Set("GPU")
    layer.Export(str(destination))
    return str(destination)


def validate_joint_runtime(usd_path, config, import_native_client, public_sdk_identity=None):
    """Fail closed before simulation creation; generic body writes are not motors."""
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise ValueError("Cannot inspect joint simulation input")
    if public_sdk_identity is not None:
        # Basic sphere/contact scenes also need admission for this exact adapter.
        validate_fixed_runtime(config, import_native_client, public_sdk_identity, family="body_controls")
    if any(p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetAttribute("physics:kinematicEnabled").Get() is True
           for p in stage.Traverse()):
        device = str(config.device).lower()
        _kinematic_admitted_devices.discard(device)
        validate_fixed_runtime(config, import_native_client, public_sdk_identity, family="kinematic")
        _kinematic_admitted_devices.add(device)
    # New body controls/shapes share the exact fixed-worker admission identity.
    # Generic transport availability alone never proves schema behavior.
    expanded = any(
        (p.GetTypeName() in {"Cube", "Capsule"} and p.HasAPI(UsdPhysics.CollisionAPI))
        or any(p.GetAttribute(name).HasAuthoredValueOpinion() for name in (
            "physics:velocity", "physics:angularVelocity", "physxRigidBody:linearDamping",
            "physxRigidBody:angularDamping", "physics:centerOfMass", "physics:diagonalInertia",
            "physics:principalAxes", "physics:density") if p.GetAttribute(name))
        for p in stage.Traverse())
    if expanded:
        _body_admitted_devices.discard(str(config.device).lower())
        validate_fixed_runtime(config, import_native_client, public_sdk_identity, family="body_controls")
        _body_admitted_devices.add(str(config.device).lower())
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)
              and p.GetAttribute("physics:jointEnabled").Get() is not False]
    if not joints:
        return
    device = str(config.device).lower()
    _admitted_devices.discard(device)
    for prim in joints:
        if prim.GetTypeName() not in JOINT_TYPES:
            raise ValueError(f"{prim.GetPath()}: joint type has no validated runtime admission")
        if device == "gpu":
            validate_gpu_joint_frames(prim)
        for name in ("physics:body0", "physics:body1"):
            for path in prim.GetRelationship(name).GetTargets():
                body = stage.GetPrimAtPath(path)
                if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI) or body.GetAttribute("physics:rigidBodyEnabled").Get() is False:
                    raise ValueError(f"{prim.GetPath()}: connected static/disabled endpoint is not admitted")
        for schema in prim.GetAppliedSchemas():
            if schema.startswith("PhysicsDriveAPI:"):
                instance = schema.split(":", 1)[1]
                if prim.GetAttribute(f"drive:{instance}:physics:type").Get() != "force":
                    raise ValueError("Acceleration joint drives are not validated")
    if device not in {"cpu", "gpu"}:
        raise ValueError(f"Joint simulation is not validated on device {device!r}")
    validate_fixed_runtime(config, import_native_client, public_sdk_identity, family="joints")
    _admitted_devices.add(device)


def validate_fixed_runtime(config, import_native_client, public_sdk_identity=None, *, family="joints"):
    if str(config.device).lower() not in {"cpu", "gpu"}:
        raise ValueError("Expanded physics controls require validated CPU or GPU device")
    if public_sdk_identity is not None:
        identity = public_sdk_identity()
        # Windows and other measured installations may add local evidence;
        # they never inherit another platform's binary or adapter identity.
        binaries = identity.get("native_sha256") if isinstance(identity, dict) else None
        if not isinstance(binaries, dict) or not binaries:
            raise ValueError(f"Public SDK runtime identity has no measured {family}/{config.device} admission: missing native binaries")
        manifests = [Path(__file__).with_name("public_sdk_physics_admission.json")]
        local = os.environ.get("OV_BLENDER_PUBLIC_SDK_PHYSICS_ADMISSION", "").strip()
        if local:
            manifests.append(Path(local).expanduser())
        measurements = []
        for manifest in manifests:
            if not manifest.is_file() and not (local and manifest == manifests[-1]):
                continue
            try:
                evidence = json.loads(manifest.read_text(encoding="utf-8"))
                rows = evidence["measurements"]
                if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
                    raise ValueError("measurements must be a list of objects")
                if evidence.get("schema_version", 1) != 1:
                    raise ValueError("unsupported admission schema")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid public SDK admission manifest {manifest}: {exc}") from exc
            measurements.extend((row, bool(local and manifest == manifests[-1]), str(manifest.resolve())) for row in rows)
        from .public_sdk_physics_policy import admits
        for measurement, is_local, manifest_path in measurements:
            if admits(measurement, identity, str(config.device).lower(), family, local=is_local):
                return {"manifest_path": manifest_path, "status": measurement["status"],
                        "local_override": is_local, "family": family,
                        "device": str(config.device).lower(),
                        "identity_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()}
        raise ValueError(f"Public SDK runtime identity has no measured {family}/{config.device} admission")
    command = shlex.split(config.ovphysx_worker_command, posix=False)
    if not command or Path(command[0].strip('"')).resolve() != Path(config.server).resolve():
        raise ValueError("Joint admission requires the identified local worker executable")
    module = import_native_client()
    capabilities = module.capabilities()
    required = {"CreateSimulation", "ReadWorldState", "WriteWorldState", "DeleteSimulation"}
    if not required.issubset(set(capabilities.get("rpcs", ()))):
        raise ValueError("Installed native client lacks required simulation transport")
    if (binary_digest(config.server), binary_digest(module.__file__)) not in MEASURED_RUNTIME_PAIRS:
        raise ValueError("Installed runtime identity has no measured joint/device admission")


def validate_gpu_joint_frames(prim):
    """Conservative boundary for measured GPU body-to-body/frame failures.

    CPU preserves compatible rotated chains. The fixed GPU binary can move a
    stationary rotated chain even with zero gravity (see bone native fixture).
    Single WORLD joints with identity local rotations remain admitted.
    Body-to-body and other rotated configurations remain authorable and run on
    CPU; never silently select a different device.
    """
    rotations = []
    world = []
    for index in (0, 1):
        value = prim.GetAttribute(f"physics:localRot{index}").Get()
        rotations.append(value is not None and any(abs(float(v)) > 1e-6 for v in value.GetImaginary()))
        world.append(not prim.GetRelationship(f"physics:body{index}").GetTargets())
    if not any(world):
        raise ValueError(
            f"{prim.GetPath()}: body-to-body joints are not validated on the installed GPU runtime; "
            "select CPU explicitly for this mechanism")
    if not any(rotations):
        return
    raise ValueError(
        f"{prim.GetPath()}: rotated joint frames are not validated on the installed GPU runtime; "
        "select CPU explicitly for this mechanism")

"""Real external physics state survives an unrelated visual generation edit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example.generation_runtime_adapters import OvphysxGenerationAdapter
from ovrtx_blender_example.shared_stage_composition import BodyVelocity

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
p.add_argument("--measure-unadmitted", action="store_true")
p.add_argument("--qualification-mode", choices=('production', 'measurement', 'candidate'), default='production')
a = p.parse_args(sys.argv[sys.argv.index("--") + 1:])
a.public_sdk = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_sdk_physics_fixture import configure
configure(a)
a.output.mkdir(parents=True, exist_ok=True)
(a.output / "results.json").unlink(missing_ok=True)
fixture_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
os.environ["OV_BLENDER_EXAMPLE_OVPHYSX_DEVICE"] = a.device
source = (Path(__file__).parent / "fixtures/public_sdk_kinematic.usda").read_text()
paths = []
mass_source = "physics:mass = 2".join(source.rsplit("physics:mass = 1", 1))
for i, content in enumerate((source, source + '\ndef Cube "VisualOnly" {}\n', mass_source,
                             mass_source.replace("gravityMagnitude = 0", "gravityMagnitude = 1")), 1):
    path = a.output / f"scene-{i}.usda"
    path.write_text(content)
    paths.append(path)
def generation(index):
    return SimpleNamespace(number=index, materialize_usd=lambda: str(paths[index - 1]))

adapter = OvphysxGenerationAdapter()
try:
    assert adapter.activate(generation(1)), adapter.last_error
    original = adapter.controller
    client = original._physics_client
    client.write_body_velocities((BodyVelocity("/World/Dynamic", (2, 0, 0), (0, 0, 1)),), simulation_time_ns=0)
    for _ in range(10):
        original.tick()
    stamp = original.current_simulation_time_ns
    visual_stamp = stamp
    before, _ = client.read_body_states(stamp)
    old_pid = client.runtime.pid
    old_step = original.step_count
    assert old_step > 0
    assert adapter.activate(generation(2)), adapter.last_error
    assert adapter.controller is original
    assert original.step_count == old_step and original.current_simulation_time_ns == stamp
    after, _ = client.read_body_states(stamp)
    assert after == before
    assert client.runtime.pid == old_pid
    assert adapter.physics_generation == 1 and adapter.reused_generation_count == 1
    original.tick()
    assert original.current_simulation_time_ns > stamp
    health = client.runtime.request("health", {})
    stamp = original.current_simulation_time_ns
    before_mass, _ = client.read_body_states(stamp)
    assert adapter.activate(generation(3)), adapter.last_error
    assert adapter.controller is original and client.runtime.pid == old_pid
    assert client.read_body_states(stamp)[0] == before_mass
    after_health = client.runtime.request("health", {})
    assert health["stage_identity"] == after_health["stage_identity"]
    assert health["physics_identity"] == after_health["physics_identity"]
    assert original.last_body_mass_write["masses"] == {"/World/Dynamic": 2.0}
    inertia = original.last_body_mass_write["inertias"]["/World/Dynamic"]
    assert max(abs(inertia[i] - .032) for i in (0, 4, 8)) < 1e-6
    assert original.current_simulation_time_ns == stamp and adapter.live_mass_generation_count == 1
    assert adapter.activate(generation(4)), adapter.last_error
    assert adapter.controller is not original and original.closed
    assert adapter.controller._physics_client.runtime.pid != old_pid
    assert adapter.reset(), adapter.last_error
    adapter.deactivate()
    inertia_probe = Path(__file__).resolve().parents[1] / "scripts/probe_public_sdk_mass_inertia.py"
    inertia_digest = hashlib.sha256(inertia_probe.read_bytes()).hexdigest()
    with (a.output / "inertia-parity.log").open("w") as log:
        subprocess.run([os.environ["OV_BLENDER_PUBLIC_SDK_PYTHON"], str(inertia_probe),
                        "--device", a.device, "--output", str(a.output / "inertia-parity")],
                       stdout=log, stderr=subprocess.STDOUT, timeout=120, check=True)
    inertia_report = json.loads((a.output / "inertia-parity/results.json").read_text())
    identity = json.loads((a.output / "runtime-identity.json").read_text())["identity"]
    assert inertia_report["status"] == "pass" and inertia_report["runtime_identity"] == identity
    report = {"device": a.device, "fixture_sha256": fixture_digest, "preserved_pid": old_pid, "preserved_steps": old_step,
              "preserved_time_ns": visual_stamp, "mass_update_time_ns": stamp, "exact_state_preserved": True,
              "visual_reuse_count": adapter.reused_generation_count,
              "mass_update_preserved_stage_state_time": True,
              "automatic_and_explicit_inertia_parity": True, "inertia_fixture_sha256": inertia_digest,
              "other_physics_changes_and_explicit_reset_replaced": True}
    (a.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PUBLIC_PHYSICS_GENERATION_REUSE_PASS", json.dumps(report), flush=True)
finally:
    adapter.deactivate()

"""Bounded public API proof: mutate population without replacing PhysX or Stage."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from public_sdk_backend import PhysicsBackend

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
p.add_argument("--replace-source", action="store_true")
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
b = PhysicsBackend(a.output)
ovs = b.ovstage
fixture = Path(__file__).resolve().parents[1] / "tests/fixtures/public_sdk_kinematic.usda"
source = fixture.read_text()
body = '''#usda 1.0
(defaultPrim = "Added")
def Sphere "Added" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMassAPI"]) {
    double radius = 0.2
    float physics:mass = 2
    double3 xformOp:translate = (10, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}
'''
decor = '''#usda 1.0
(defaultPrim = "Decor")
def Cube "Decor" {
    double size = 0.1
}
'''
report = {"device": a.device, "pid": os.getpid(), "events": []}

def snapshot():
    return {key: b._columns(attr) for key, attr in (("position", "position"), ("velocity", "linearVelocity"), ("angular", "angularVelocity"))}

def apply(label, previous, source_path=None):
    b.ordinal += 1
    if source_path is None:
        ovs.population.apply_usd_changes(b.stage, ordinal=b.ordinal)
    else:
        ovs.population.open_usd(b.stage, str(source_path), ordinal=b.ordinal, domains=ovs.PopulationDomain.ALL)
    b.stage.advance_write_floor(b.ordinal).wait()
    b.physx.update_from_ovstage(b.ordinal, b.ordinal)
    b.physx.wait_all()
    current = snapshot()
    preserved = all(current[field]["/World/Dynamic"] == previous[field]["/World/Dynamic"] for field in current)
    assert b.time_ns == 83333330
    assert id(b.stage) == report["stage_identity"]
    assert id(b.physx) == report["physx_identity"]
    report["events"].append({"edit": label, "ordinal": b.ordinal, "time_ns": b.time_ns, "state": current, "preserved": preserved})
    print(label, "preserved", preserved, flush=True)
    return current

try:
    b.dispatch("load", {"scene_path": str(fixture), "device": a.device})
    report.update(stage_identity=id(b.stage), physx_identity=id(b.physx))
    b.dispatch("write_velocities", {"velocities": [{"prim_path": "/World/Dynamic", "linear": [2, 0, 0], "angular": [0, 0, 1]}]})
    for i in range(10):
        b.dispatch("step", {"dt_ns": 8333333})
    before = snapshot()
    with ovs.PathDictionary(b.stage) as paths:
        with paths.create_path_list_from_strings(["/World/Dynamic"]) as plist:
            with b.stage.query_from_path_list(plist) as query:
                b.stage.write_attribute(query, "physics:mass", b.ordinal + 1,
                    b.np.asarray([2.0], dtype=b.np.float32), is_array=False).wait()
    current = apply("change_body_mass", before)
    report["mass_after_edit"] = b._columns("mass")["/World/Dynamic"]
    report["mass_edit_applied"] = report["mass_after_edit"] == 2.0
    b.dispatch("write_masses", {"values": [{"prim_path": "/World/Dynamic", "mass": 2.0}]})
    report["direct_mass_after_edit"] = b._columns("mass")["/World/Dynamic"]
    assert report["direct_mass_after_edit"] == 2.0
    assert snapshot() == before
    last_ordinal = b.ordinal
    for invalid in ([{"prim_path": "/Missing", "mass": 2}],
                    [{"prim_path": "/World/Dynamic", "mass": 2}, {"prim_path": "/World/Kinematic", "mass": -1}],
                    [{"prim_path": "/World/Dynamic", "mass": 1e-300}]):
        try:
            b.dispatch("write_masses", {"values": invalid})
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid mass batch accepted")
        assert b.ordinal == last_ordinal and snapshot() == before
    report["mass_dispatch"] = b.dispatch("write_masses", {"values": [{"prim_path": "/World/Dynamic", "mass": 3.0}]})
    assert report["mass_dispatch"]["masses"] == {"/World/Dynamic": 3.0}
    assert report["mass_dispatch"]["simulation_time_ns"] == b.time_ns == 83333330
    assert snapshot() == before
    visual = ovs.population.add_usd_reference_from_string(b.stage, decor, "/World/Decor")
    current = apply("add_render_only_prim", before)
    new = ovs.population.add_usd_reference_from_string(b.stage, body, "/World/Added")
    added = apply("add_body", current)
    assert "/World/Added" in added["position"], added
    ovs.population.remove_usd(b.stage, new)
    removed = apply("remove_body", added)
    assert "/World/Added" not in removed["position"], removed
    ovs.population.remove_usd(b.stage, visual)
    current = apply("remove_render_only_prim", removed)
    if a.replace_source:
        path = a.output / "replacement.usda"
        path.write_text(source.replace("physics:mass = 1", "physics:mass = 2") + '\n' + body[body.index('def Sphere'):])
        current = apply("open_replacement_source", current, path)
        assert "/Added" in current["position"]
        path = a.output / "restored.usda"
        path.write_text(source)
        current = apply("open_original_source", current, path)
        assert "/Added" not in current["position"], current
    report["status"] = "measured"
finally:
    (a.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    b.unload()
print(json.dumps(report), flush=True)

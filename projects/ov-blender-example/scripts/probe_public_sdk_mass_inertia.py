"""Compare live mass/inertia edits with freshly authored driven-joint dynamics."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from public_sdk_backend import PhysicsBackend

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
base = (Path(__file__).resolve().parents[1] / "tests/fixtures/public_sdk_kinematic.usda").read_text()
joint = '''
    def PhysicsRevoluteJoint "Joint" (prepend apiSchemas = ["PhysicsDriveAPI:angular"]) {
        rel physics:body1 = </World/Dynamic>
        uniform token physics:axis = "Z"
        point3f physics:localPos0 = (0, 0, 1)
        point3f physics:localPos1 = (0, 0, 0)
        token drive:angular:physics:type = "force"
        float drive:angular:physics:targetVelocity = 90
        float drive:angular:physics:damping = 1
        float drive:angular:physics:maxForce = 0.01
    }
'''
base = base.rsplit("}", 1)[0] + joint + "}\n"
report = {"device": a.device, "cases": {}}
for explicit in (False, True):
    results = {}
    for live in (False, True):
        source = base
        if explicit:
            source = source.replace("float physics:mass = 1", "float physics:mass = 1\n        float3 physics:diagonalInertia = (0.12, 0.13, 0.14)")
        if not live:
            source = source.replace("physics:mass = 1", "physics:mass = 2")
        path = a.output / f"explicit-{explicit}-live-{live}.usda"
        path.write_text(source)
        backend = PhysicsBackend(a.output)
        try:
            health = backend.dispatch("load", {"scene_path": str(path), "device": a.device})
            report["runtime_identity"] = health["runtime_identity"]
            if live:
                old = backend._columns("inertia")["/World/Dynamic"]
                backend.dispatch("write_masses", {"values": [{"prim_path": "/World/Dynamic", "mass": 2}]})
            inertia = backend._columns("inertia")["/World/Dynamic"]
            if live:
                assert backend.np.allclose(inertia, backend.np.asarray(old) * (1 if explicit else 2), rtol=1e-6, atol=1e-8)
            for _ in range(60):
                backend.dispatch("step", {"dt_ns": 4166667})
            results["live" if live else "fresh"] = {
                "inertia": inertia, "orientation": backend._columns("orientation")["/World/Dynamic"],
                "angular_velocity": backend._columns("angularVelocity")["/World/Dynamic"]}
        finally:
            backend.unload()
    for field, tolerance in (("inertia", 1e-7), ("orientation", 2e-5), ("angular_velocity", 2e-4)):
        error = max(abs(x-y) for x, y in zip(results["live"][field], results["fresh"][field]))
        assert error < tolerance, (explicit, field, error, results)
    assert abs(results["fresh"]["angular_velocity"][2]) > .001, results
    report["cases"]["explicit" if explicit else "automatic"] = results
report["status"] = "pass"
(a.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
print("MASS_INERTIA_PARITY_PASS", json.dumps(report), flush=True)

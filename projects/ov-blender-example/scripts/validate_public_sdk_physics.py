#!/usr/bin/env python3
"""Run baseline Blender-authored physics acceptance sequentially on public SDKs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--blender", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--devices", nargs="+", choices=("cpu", "gpu"), default=["cpu", "gpu"])
parser.add_argument("--families", nargs="+", choices=("body_controls", "joints", "kinematic", "parented", "bones", "bake_contact"), default=["body_controls", "joints", "kinematic", "parented", "bones", "bake_contact"])
parser.add_argument("--measure-unadmitted", action="store_true")
parser.add_argument("--timeout", type=float, default=3600,
                    help="Per family/device timeout including repeated verified cold worker launches")
args = parser.parse_args()
if not os.environ.get("OV_BLENDER_PUBLIC_SDK_PYTHON"):
    raise SystemExit("OV_BLENDER_PUBLIC_SDK_PYTHON must select the public SDK environment")
public = Path(__file__).resolve().parents[1]
root = args.output.resolve()
root.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, OV_BLENDER_PUBLIC_SDK_OUTPUT=str(root / "workers"))
reports = []
scripts = {"body_controls": "blender_native_body_controls.py", "joints": "blender_native_joints.py",
           "kinematic": "blender_native_kinematic.py", "parented": "blender_native_parented_poses.py",
           "bones": "blender_bone_mechanisms.py", "bake_contact": "blender_bake_contact_standalone.py"}
for family in args.families:
    for device in args.devices:
        output = root / f"{family}-{device}"
        output.mkdir(exist_ok=True)
        acceptance_path = output / "acceptance.json"
        acceptance_path.unlink(missing_ok=True)
        command = [str(args.blender), "--background", "--factory-startup", "--python-exit-code", "17",
                   "--python", str(public / "tests" / scripts[family]), "--", "--public-sdk",
                   "--output", str(output), "--device", device]
        if args.measure_unadmitted:
            command.append("--measure-unadmitted")
        if family == "bones" and device == "gpu":
            command.extend(["--aligned", "--world-only"])
        started = time.monotonic()
        print(f"START {family} {device}", flush=True)
        with (output / "run.log").open("w") as log:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
        report = {"family": family, "device": device, "exit_code": completed.returncode,
                  "seconds": time.monotonic() - started, "log": str(output / "run.log")}
        reports.append(report)
        (root / "suite.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        if completed.returncode:
            raise SystemExit(completed.returncode)
        if not (output / "results.json").is_file():
            raise RuntimeError(f"Missing fixture acceptance report: {output}")
        if family == "bake_contact":
            offline = [str(args.blender), "--background", "--factory-startup", "--python-exit-code", "17",
                       "--python", str(public / "tests" / scripts[family]), "--", "--offline", "--output", str(output)]
            offline_env = {key: value for key, value in env.items() if not key.startswith("OV_BLENDER_PUBLIC_SDK")}
            with (output / "offline.log").open("w") as log:
                subprocess.run(offline, env=offline_env, stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True)
        metadata = json.loads((output / "runtime-identity.json").read_text())
        acceptance = {"family": family, "device": device,
                      "measurement_only": metadata["measurement_only"],
                      "identity_sha256": hashlib.sha256(json.dumps(metadata["identity"], sort_keys=True).encode()).hexdigest(),
                      "fixture_sha256": hashlib.sha256((public / "tests" / scripts[family]).read_bytes()).hexdigest(),
                      "results_sha256": hashlib.sha256((output / "results.json").read_bytes()).hexdigest()}
        if family == "bake_contact":
            acceptance["offline_results_sha256"] = hashlib.sha256((output / "offline-results.json").read_bytes()).hexdigest()
        acceptance_path.write_text(json.dumps(acceptance, indent=2) + "\n")
print("PUBLIC_SDK_PHYSICS_ACCEPTANCE_PASS", len(reports), flush=True)

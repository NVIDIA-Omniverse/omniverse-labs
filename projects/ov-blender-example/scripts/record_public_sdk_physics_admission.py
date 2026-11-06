#!/usr/bin/env python3
"""Record exact public SDK admission only after baseline numerical fixtures pass.

Measurement creates a candidate row. Explicit complete candidate qualification
can promote its identity while preserving the actual candidate-run provenance.
Fresh shipped-default production verification remains a separate required run.
"""
import argparse
import hashlib
import json
import re
import os
import tempfile
from pathlib import Path
from public_sdk_physics_evidence import validate_directory

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--evidence-root", type=Path, required=True)
parser.add_argument("--destination", type=Path, required=True)
parser.add_argument("--live-mass-evidence", nargs=2, type=Path, metavar=("CPU_DIR", "GPU_DIR"), required=True,
                    help="Successful actual generation-reuse/mass-update probes, CPU then GPU")
parser.add_argument("--production-evidence-root", type=Path,
                    help="Promote only after all fixtures rerun through admission without measurement bypass")
parser.add_argument("--production-live-mass-evidence", nargs=2, type=Path, metavar=("CPU_DIR", "GPU_DIR"))
parser.add_argument("--candidate-evidence-root", type=Path, help="Explicit full candidate qualification, not a production execution")
parser.add_argument("--candidate-live-mass-evidence", nargs=2, type=Path)
args = parser.parse_args()
if args.production_evidence_root and args.candidate_evidence_root:
    raise ValueError("Choose candidate qualification or production revalidation")
if bool(args.candidate_evidence_root) != bool(args.candidate_live_mass_evidence):
    raise ValueError("Candidate promotion requires both full family and live mass evidence")
qualification_mode = "candidate" if args.candidate_evidence_root else "production"
if args.candidate_evidence_root:
    args.production_evidence_root = args.candidate_evidence_root
    args.production_live_mass_evidence = args.candidate_live_mass_evidence
root = args.evidence_root.resolve()
scripts = {"body_controls": "blender_native_body_controls.py", "joints": "blender_native_joints.py",
           "kinematic": "blender_native_kinematic.py", "parented": "blender_native_parented_poses.py",
           "bones": "blender_bone_mechanisms.py", "bake_contact": "blender_bake_contact_standalone.py"}
identity = None
evidence = {}
public = Path(__file__).resolve().parents[1]
for family, script in scripts.items():
    for device in ("cpu", "gpu"):
        directory = root / f"{family}-{device}"
        validate_directory(directory, family, device, "measurement")
        metadata = json.loads((directory / "runtime-identity.json").read_text())
        acceptance = json.loads((directory / "acceptance.json").read_text())
        if metadata["device"] != device:
            raise ValueError(f"Wrong measured device: {directory}")
        if identity is None:
            identity = metadata["identity"]
        if metadata["identity"] != identity:
            raise ValueError("Fixture results came from different SDK/adapter identities")
        if acceptance["identity_sha256"] != hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest():
            raise ValueError("The successful run belongs to a different runtime identity")
        if acceptance.get('family') != family or acceptance.get('device') != device:
            raise ValueError('The successful run belongs to a different family/device')
        # Results files are written only after the fixture's final assertions.
        result_path = directory / "results.json"
        result = json.loads(result_path.read_text())
        if not result:
            raise ValueError(f"Empty fixture result: {result_path}")
        record = {"fixture": f"public/tests/{script}",
                  "fixture_sha256": hashlib.sha256((public / "tests" / script).read_bytes()).hexdigest(),
                  "results_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest()}
        for key in ("fixture_sha256", "results_sha256"):
            if acceptance.get(key) != record[key]:
                raise ValueError(f"Successful-run evidence changed: {directory}/{key}")
        if family == "bake_contact":
            offline = directory / "offline-results.json"
            if not json.loads(offline.read_text()):
                raise ValueError("Offline Blender bake validation is absent")
            record["offline_results_sha256"] = hashlib.sha256(offline.read_bytes()).hexdigest()
            if acceptance.get("offline_results_sha256") != record["offline_results_sha256"]:
                raise ValueError("Offline acceptance evidence changed")
        evidence[f"{family}/{device}"] = record
mass_fixture = public / "tests/blender_public_physics_generation_reuse.py"
for device, directory in zip(("cpu", "gpu"), args.live_mass_evidence, strict=True):
    validate_directory(directory, "live_mass", device, "measurement")
    metadata = json.loads((directory / "runtime-identity.json").read_text())
    if metadata["identity"] != identity or metadata["device"] != device:
        raise ValueError(f"Live mass proof came from a different runtime/device: {directory}")
    result_path = directory / "results.json"
    result = json.loads(result_path.read_text())
    required = ("exact_state_preserved", "mass_update_preserved_stage_state_time", "automatic_and_explicit_inertia_parity",
                "other_physics_changes_and_explicit_reset_replaced")
    if result.get("device") != device or any(result.get(key) is not True for key in required):
        raise ValueError(f"Incomplete live mass acceptance: {directory}")
    if result.get("preserved_steps", 0) <= 0 or result.get("visual_reuse_count", 0) < 1:
        raise ValueError("Live mass acceptance must include an already advancing simulation")
    if result.get("fixture_sha256") != hashlib.sha256(mass_fixture.read_bytes()).hexdigest():
        raise ValueError("Live mass acceptance fixture changed after the successful run")
    inertia_fixture = public / "scripts/probe_public_sdk_mass_inertia.py"
    if result.get("inertia_fixture_sha256") != hashlib.sha256(inertia_fixture.read_bytes()).hexdigest():
        raise ValueError("Inertia acceptance fixture changed after the successful run")
    inertia_path = directory / "inertia-parity/results.json"
    inertia_result = json.loads(inertia_path.read_text())
    if inertia_result.get("status") != "pass" or inertia_result.get("runtime_identity") != identity:
        raise ValueError("Inertia acceptance belongs to a different runtime")
    evidence[f"live_mass/{device}"] = {
        "fixture": "public/tests/blender_public_physics_generation_reuse.py",
        "fixture_sha256": hashlib.sha256(mass_fixture.read_bytes()).hexdigest(),
        "results_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "inertia_fixture_sha256": result["inertia_fixture_sha256"],
        "inertia_results_sha256": hashlib.sha256(inertia_path.read_bytes()).hexdigest(),
    }
if not isinstance(identity, dict) or not isinstance(identity.get("native_sha256"), dict) or not identity["native_sha256"]:
    raise ValueError("Admission requires nonempty measured native binary hashes")
measurements = []
if args.destination.exists():
    previous = json.loads(args.destination.read_text(encoding="utf-8"))
    if previous.get("schema_version") != 1 or not isinstance(previous.get("measurements"), list):
        raise ValueError("Invalid existing admission manifest; refusing to overwrite")
    if not all(isinstance(row, dict) for row in previous["measurements"]):
        raise ValueError("Invalid existing admission measurements")
    measurements = [row for row in previous["measurements"] if row.get("identity") != identity]
if bool(args.production_evidence_root) != bool(args.production_live_mass_evidence):
    raise ValueError("Production promotion requires both full family and live mass reruns")
status = 'candidate'

def _portable_name(path):
    """Return a manifest file name without the recording host's directory."""
    if not path:
        return path
    return re.split(r'[\\/]', str(path))[-1]

if args.production_evidence_root:
    qualification_manifest = None
    for cell, record in evidence.items():
        family, device = cell.split('/')
        directory = (args.production_live_mass_evidence[('cpu', 'gpu').index(device)]
                     if family == 'live_mass' else args.production_evidence_root / f'{family}-{device}')
        validated = validate_directory(directory, family, device, qualification_mode)
        metadata = validated['metadata']
        selected_manifest = metadata.get('local_manifest_override')
        if qualification_mode == 'candidate':
            if qualification_manifest is None: qualification_manifest = selected_manifest
            elif selected_manifest != qualification_manifest: raise ValueError('Mixed candidate qualification manifests')
            if args.destination.resolve()==Path(selected_manifest).resolve():raise ValueError('Production destination must preserve the immutable candidate qualification manifest')
        if metadata.get('identity') != identity or metadata.get('device') != device or metadata.get('measurement_only') is not False:
            raise ValueError(f"Production acceptance requires the same identity/device without measurement bypass: {cell}")
        result_path = directory / 'results.json'
        result = json.loads(result_path.read_text())
        if family == 'live_mass':
            if any(result.get(key) is not True for key in required):
                raise ValueError(f"Incomplete production live mass acceptance: {cell}")
            if result.get('fixture_sha256') != record['fixture_sha256']:
                raise ValueError('Production mass fixture changed')
            if result.get('inertia_fixture_sha256') != record['inertia_fixture_sha256']:
                raise ValueError('Production inertia fixture changed')
            inertia_path = directory / 'inertia-parity/results.json'
            inertia_result = json.loads(inertia_path.read_text())
            if inertia_result.get('status') != 'pass' or inertia_result.get('runtime_identity') != identity:
                raise ValueError('Production inertia acceptance is incomplete')
            record['production_inertia_results_sha256'] = hashlib.sha256(inertia_path.read_bytes()).hexdigest()
        else:
            accepted = json.loads((directory / 'acceptance.json').read_text())
            if (accepted.get('measurement_only') is not False
                    or accepted.get('identity_sha256') != hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                    or accepted.get('fixture_sha256') != record['fixture_sha256']
                    or accepted.get('family') != family or accepted.get('device') != device
                    or accepted.get('results_sha256') != hashlib.sha256(result_path.read_bytes()).hexdigest()):
                raise ValueError(f"Production successful-run evidence changed: {cell}")
            if family == 'bake_contact':
                offline = directory / 'offline-results.json'
                if (not json.loads(offline.read_text()) or accepted.get('offline_results_sha256')
                        != hashlib.sha256(offline.read_bytes()).hexdigest()):
                    raise ValueError('Production offline bake acceptance changed')
                record['production_offline_results_sha256'] = hashlib.sha256(offline.read_bytes()).hexdigest()
        if not result:
            raise ValueError(f"Empty production result: {cell}")
        record['qualification_mode'] = qualification_mode
        # Published evidence names the manifest file only; host paths stay local.
        record['qualification_admission_checks'] = [
            dict(check, manifest_path=_portable_name(check.get('manifest_path'))) if isinstance(check, dict) else check
            for check in metadata['admission_checks']]
        record['qualification_local_manifest_override'] = _portable_name(metadata.get('local_manifest_override'))
        record['qualification_acceptance_sha256'] = hashlib.sha256((directory/'acceptance.json').read_bytes()).hexdigest()
        # Legacy production_* keys are consumed as qualification-proof digests;
        # they do not relabel an actual candidate execution as production.
        record['production_results_sha256'] = hashlib.sha256(result_path.read_bytes()).hexdigest()
        record['production_measurement_only'] = False
    status = 'production'
measurements.append({"identity": identity,
    "status": status, "qualification_mode": qualification_mode if args.production_evidence_root else "measurement",
    "devices": ["cpu", "gpu"], "families": ["body_controls", "joints", "kinematic", "live_mass"],
    "evidence": evidence})
manifest = {"schema_version": 1, "measurements": measurements}
args.destination.parent.mkdir(parents=True, exist_ok=True)
# Close before replace for Windows; a failed proof or write retains old evidence.
with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.destination.parent,
                                 prefix=args.destination.name + ".", suffix=".tmp", delete=False) as stream:
    temporary = Path(stream.name)
    try:
        stream.write(json.dumps(manifest, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    except BaseException:
        stream.close()
        temporary.unlink(missing_ok=True)
        raise
try:
    os.replace(temporary, args.destination)
finally:
    temporary.unlink(missing_ok=True)
print(f"Recorded {len(evidence)} passing fixture/device combinations: {args.destination}")

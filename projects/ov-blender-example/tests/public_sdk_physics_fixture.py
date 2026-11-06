"""Observe numerical fixtures through authenticated current SDK diagnostics.

Production admission is the default. Explicit diagnostic modes are restricted
to these numerical fixtures and recorded as such; production has no bypass.
"""
import json
import copy
import hashlib
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

_state = None


def require_production_admission(receipt):
    if receipt.get('status') != 'production' or receipt.get('local_override') is not False:
        raise AssertionError('Current qualification requires shipped production admission')


def add_arguments(parser):
    parser.add_argument("--public-sdk", action="store_true")
    parser.add_argument("--measure-unadmitted", action="store_true")
    parser.add_argument("--qualification-mode", choices=('production', 'measurement', 'candidate'), default='production')


def qualification_mode(args):
    mode = getattr(args, 'qualification_mode', 'production')
    if args.measure_unadmitted:
        raise ValueError('Use explicit --qualification-mode measurement; normal production admission is the default')
    override = os.environ.get('OV_BLENDER_PUBLIC_SDK_PHYSICS_ADMISSION', '').strip()
    if mode not in ('production', 'measurement', 'candidate'):
        raise ValueError('Unknown qualification mode')
    if mode != 'production' and not args.public_sdk:
        raise ValueError('Diagnostic qualification requires --public-sdk')
    if mode in ('production', 'measurement') and override:
        raise ValueError('Current qualification forbids a local admission override')
    if mode == 'candidate' and not override:
        raise ValueError('Candidate qualification requires an explicit local admission manifest')
    return mode


def configure(args):
    global _state
    mode = qualification_mode(args)
    if not args.public_sdk:
        return None
    from ovrtx_blender_example import bundled_runtime
    selected = bundled_runtime.public_sdk_python()
    if not selected or not Path(selected).is_file():
        raise ValueError("An installed or explicitly selected public SDK runtime is required")
    from ovrtx_blender_example import physics_capabilities
    original = physics_capabilities.validate_fixed_runtime
    state = SimpleNamespace(client=None, identities=[], admissions=[], validate=original,
                            selected_python=selected, output=args.output, mode=mode)
    _state = state

    def validate(config, import_native_client, public_sdk_identity=None, *, family="joints"):
        if public_sdk_identity is None:
            return original(config, import_native_client, family=family)
        identity = public_sdk_identity()
        if identity.get("backend") != "public_sdk" or not identity.get("native_sha256") or not identity.get("adapter_sha256"):
            raise ValueError("Acceptance fixture requires an identified external SDK worker")
        state.client = public_sdk_identity.__self__
        if identity not in state.identities:
            state.identities.append(identity)
        destination = args.output / "runtime-identity.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if mode == 'measurement':
            if str(config.device).lower() not in ('cpu', 'gpu'):
                raise ValueError('Measurement requires an explicit supported device')
            receipt = dict(status='measurement-only', local_override=False, family=family,
                           device=str(config.device).lower())
        else:
            receipt = original(config, import_native_client, public_sdk_identity, family=family)
            if mode == 'production':
                require_production_admission(receipt)
            elif receipt.get('status') != 'candidate' or receipt.get('local_override') is not True:
                raise AssertionError('Candidate qualification requires local candidate admission')
        if receipt not in state.admissions:
            state.admissions.append(receipt)
        destination.write_text(json.dumps({"identity": identity, "device": args.device,
                                           "measurement_only": mode == 'measurement',
                                           "qualification_mode": mode,
                                           "admission_checks": state.admissions,
                                           "local_manifest_override": os.environ.get('OV_BLENDER_PUBLIC_SDK_PHYSICS_ADMISSION')}, indent=2) + "\n")
        return receipt

    physics_capabilities.validate_fixed_runtime = validate

    # Observe every real read at the existing client boundary without replacing
    # simulation, admission, or typed pose conversion. Keep a complete trace.
    from ovrtx_blender_example.direct_sdk_client import PublicSdkPhysicsClient
    original_read = PublicSdkPhysicsClient.read_body_states
    device_policies = {}
    def checked_read(client, simulation_time_ns):
        states, result = original_read(client, simulation_time_ns)
        if client.runtime.pid not in device_policies:
            health = client.runtime.request('health', {})
            assert health['loaded'] is True and health['cpu_only_policy'] is (args.device == 'cpu'), health
            device_policies[client.runtime.pid] = health['cpu_only_policy']
        expected = tuple(client.config.body_prims)
        paths = [row['prim_path'] for row in states]
        if len(paths) != len(set(paths)) or set(paths) != set(expected):
            raise AssertionError(f'Incomplete/duplicate native pose roster: {paths} != {expected}')
        ordered = sorted(states, key=lambda row: row['prim_path'])
        normalized = []
        for row in ordered:
            values = {}
            for name, axes in (('translate', 'xyz'), ('orient', 'ijkr'),
                               ('linear_velocity', 'xyz'), ('angular_velocity', 'xyz')):
                field = row[name]
                vector = [float(field[axis]) for axis in axes]
                assert field['found'] is True and all(math.isfinite(v) for v in vector), (name, row)
                values[name] = vector
            assert abs(math.hypot(*values['orient']) - 1) < 1e-4, row
            assert row['simulation_time_ns'] == result['simulation_time_ns'], row
            normalized.append({'prim_path': row['prim_path'], **values})
        record = {'pid': client.runtime.pid, 'simulation_time_ns': result['simulation_time_ns'],
                  'cpu_only_policy': device_policies[client.runtime.pid],
                  'expected_paths': expected, 'states': normalized,
                  'state_sha256': hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()}
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / 'pose-samples.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, sort_keys=True) + '\n')
        return states, result
    PublicSdkPhysicsClient.read_body_states = checked_read

    return state


def current_health():
    if _state is None or _state.client is None:
        raise RuntimeError("No active public SDK fixture worker")
    result = _state.client.runtime.request("health", {})
    assert result['pid'] == _state.client.runtime.identity.pid
    return {"worker_pid": result["pid"], **result}


def current_capabilities():
    if _state is None or _state.client is None:
        raise RuntimeError("No active public SDK fixture worker")
    return dict(_state.client.runtime.identity.capabilities)


def current_provenance():
    from ovrtx_blender_example import direct_sdk_client, bundled_runtime
    if _state is None or not _state.identities:
        raise RuntimeError("No authenticated SDK identity recorded")
    client = Path(direct_sdk_client.__file__)
    worker = bundled_runtime.public_sdk_worker_script()
    return dict(selected_python=_state.selected_python,
                client_path=str(client), client_sha256=hashlib.sha256(client.read_bytes()).hexdigest(),
                worker_path=str(worker), worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
                identities=_state.identities, admissions=_state.admissions,
                qualification_mode=_state.mode)


def reject_changed_identities(config):
    """Call the unmodified production validator on malformed real identity copies.

    This invokes no RPC, process constructor, or legacy native-client supplier.
    Preserve the live identity and every observation list exactly.
    """
    if _state is None or _state.client is None or not _state.identities:
        raise RuntimeError("No authenticated SDK identity available for rejection tests")
    runtime = _state.client.runtime
    before = runtime.identity
    observations = copy.deepcopy((_state.identities, _state.admissions))
    original = copy.deepcopy(_state.identities[-1])
    errors = []
    def forbidden_native():
        raise AssertionError("Public SDK validation attempted legacy native import")
    for label, key, value in (
        ('missing native identity', 'native_sha256', {}),
        ('unknown adapter identity', 'adapter_sha256', '0' * 64),
    ):
        candidate = copy.deepcopy(original)
        assert candidate[key] != value
        candidate[key] = value
        calls = []
        def supplied():
            calls.append(1)
            return candidate
        try:
            _state.validate(config, forbidden_native, supplied, family='joints')
        except ValueError as error:
            errors.append(dict(label=label, error=str(error)))
        else:
            raise AssertionError(label + ' was admitted')
        assert calls == [1], calls
        assert runtime.identity == before
        assert (_state.identities, _state.admissions) == observations
    (_state.output / 'identity-refusals.json').write_text(json.dumps(errors, indent=2) + '\n')
    return errors

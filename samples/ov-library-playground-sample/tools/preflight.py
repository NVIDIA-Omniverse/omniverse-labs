# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only installation diagnostics. No SDK imports or administrator access."""
import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def inspect(runtime_root, packages=True):
    errors, warnings = [], []
    report = {'platform': platform.platform(), 'python': platform.python_version(),
              'profile': 'public', 'gpus': [], 'packages': {}, 'errors': errors, 'warnings': warnings}
    if sys.platform != 'win32' or struct.calcsize('P') != 8:
        errors.append('This preview requires Windows x64. Linux and macOS have not been validated.')
    if sys.version_info[:2] != (3, 12):
        errors.append('Use CPython 3.12 x64. The launcher installs its own runtime automatically.')
    smi = shutil.which('nvidia-smi')
    if smi:
        try:
            result = subprocess.run([smi, '--query-gpu=name,driver_version,memory.total',
                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
            if result.returncode:
                raise RuntimeError('nvidia-smi could not communicate with the NVIDIA driver.')
            for line in result.stdout.splitlines():
                name, driver, memory = [part.strip() for part in line.rsplit(',', 2)]
                report['gpus'].append({'name': name, 'driver': driver, 'memory_mib': int(memory)})
            rtx = [gpu for gpu in report['gpus'] if 'RTX' in gpu['name'].upper()]
            if not rtx:
                errors.append('No RTX GPU was detected. Run this preview on an NVIDIA RTX workstation.')
            for gpu in rtx:
                if gpu['memory_mib'] < 12000:
                    warnings.append('Less than 12 GB of GPU memory: begin with Performance quality. This configuration is not yet validated.')
                elif gpu['memory_mib'] < 24000:
                    warnings.append('The full campus with robot LiDAR and two cameras has only been tested on a 24 GB GPU. This smaller-memory sensor configuration is unvalidated.')
                if gpu['driver'] not in ('595.97', '596.58'):
                    warnings.append('Driver '+gpu['driver']+' differs from the locally tested 595.97 / 596.58 configurations. Check the release support matrix.')
        except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            errors.append(str(exc))
    else:
        errors.append('NVIDIA driver tools were not found. Install the appropriate NVIDIA RTX driver, restart Windows, and retry.')
    root = Path(runtime_root)
    probe = root
    while not probe.exists():
        probe = probe.parent
    report['free_gib'] = round(shutil.disk_usage(probe).free/1024**3, 1)
    if report['free_gib'] < 16:
        warnings.append('Allow 16 GB of free disk space for Python, libraries, campus assets, downloads and shader caches.')
    if packages:
        import re
        for line in (ROOT/'config/requirements.txt').read_text().splitlines():
            match = re.match(r'([\w-]+)==([^\s]+)', line)
            if not match:
                if line.startswith('ovrtx @'):
                    name = 'ovrtx'
                    expected = re.search(r'ovrtx-([0-9.]+)-', line).group(1)
                else:
                    continue
            else:
                name, expected = match.groups()
            try:
                actual = metadata.version(name)
                report['packages'][name] = actual
                if actual != expected:
                    errors.append(f'{name}: expected {expected}, found {actual}. Run the launcher to repair dependencies.')
            except metadata.PackageNotFoundError:
                errors.append(name+' is missing. Run the launcher to install the public libraries.')
        report['room_environments'] = {}
        for worker in ('newton', 'package'):
            required = dict(re.findall(r'^([\w-]+)(?:\[[^]]+\])?==([^\s]+)',
                (ROOT/f'config/requirements-{worker}.txt').read_text(), re.MULTILINE))
            interpreter = root/f'{worker}-env/Scripts/python.exe'
            try:
                probe = subprocess.run([str(interpreter),'-c',
                    'import json,sys; from importlib.metadata import version; '
                    'print(json.dumps({name:version(name) for name in sys.argv[1:]}))',*required],
                    capture_output=True,text=True,timeout=20,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                actual = json.loads(probe.stdout) if probe.returncode==0 else {}
                report['room_environments'][worker] = actual
                if actual != required:
                    errors.append(worker+' room dependencies do not match the tested pins. Run Repair in the launcher.')
            except (OSError,ValueError,subprocess.TimeoutExpired):
                errors.append(worker+' room environment is missing or unavailable. Run the launcher to install it.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--hardware-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = inspect(args.runtime_root, not args.hardware_only)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    sys.exit(main())

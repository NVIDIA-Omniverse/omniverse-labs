#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Launch Blender with an external public SDK environment and isolated artist config."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from public_sdk_environment import DEFAULT_CONFIG, PUBLIC, check_blender, read_config, worker_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--blender')
    parser.add_argument('--sdk-python')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--demo', action='store_true', help='Open the small procedural viewport demo')
    args = parser.parse_args()
    config = read_config(args.config)
    for key, value in (('blender', args.blender), ('python', args.sdk_python), ('gpu', args.gpu)):
        if value is not None:
            config[key] = value
    if config['gpu'] < 0:
        parser.error('--gpu must be nonnegative')
    if args.scene and args.demo:
        parser.error('--scene and --demo are mutually exclusive')
    blender = check_blender(config['blender'])
    env = worker_environment(config, args.output)
    command = [str(blender), '--factory-startup']
    if args.scene:
        if not args.scene.is_file():
            parser.error('--scene does not exist')
        command.append(str(args.scene.absolute()))
    if args.demo:
        command += ['--python', str(PUBLIC / 'scripts/probe_public_sdk_ui_edits.py')]
    else:
        command += ['--python-expr', "import bpy, ovrtx_blender_example; ovrtx_blender_example.register(); bpy.context.scene.render.engine = 'OVRTX_EXAMPLE'"]
    raise SystemExit(subprocess.call(command, env=env))


if __name__ == '__main__':
    main()

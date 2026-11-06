# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install prepared SimReady models; no geometry tools or SDK imports are needed."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import urllib.error
import uuid

from scene_assets import download, ensure_asset, install_archive, progress, safe_member

ROOT = Path(__file__).resolve().parents[1]


def setup(runtime, repair=False, archive=None):
    release = json.loads((ROOT/'config/prepared-exhibit-release.json').read_text(encoding='utf-8'))
    pin = release['bundle']
    output = ROOT/'data/models/simready'
    if output.is_symlink() or not output.resolve().is_relative_to(output.parent.resolve()):
        raise ValueError('Exhibit destination must stay inside data/models.')
    cache = Path(runtime).resolve()/'cache/exhibits'
    bundled = safe_member(ROOT/'data/scene-archives', pin['filename'])
    archive = Path(archive).resolve() if archive else (bundled if bundled.is_file() else None)
    try:
        ensure_asset(pin, release, output, cache, archive, repair)
    except (OSError, ValueError) as error:
        if not output.is_dir():
            raise
        # Recover a legacy, interrupted or edited installation only after the
        # replacement is fully verified. Keep the previous artwork recoverable.
        progress('Checking a replacement for the existing SimReady assets')
        archive = archive or download(pin, release, cache)
        parent = output.parent.resolve()
        if output.is_symlink() or not output.resolve().is_relative_to(parent):
            raise ValueError('Exhibit destination must stay inside data/models.') from error
        with tempfile.TemporaryDirectory(prefix='.simready-install-', dir=parent) as temporary:
            staged = Path(temporary)/'simready'
            install_archive(archive, staged, pin)
            backups = parent/'.simready-backups'
            if backups.is_symlink() or not backups.resolve().is_relative_to(parent):
                raise ValueError('Exhibit backup directory must stay inside data/models.')
            backups.mkdir(exist_ok=True)
            backup = backups/uuid.uuid4().hex
            output.rename(backup)
            try:
                staged.rename(output)
            except OSError:
                backup.rename(output)
                raise
            progress('Previous SimReady files preserved in '+str(backup))
        ensure_asset(pin, release, output, cache, archive, repair=True)
    progress('Prepared SimReady exhibits ready.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--archive', type=Path, help='Pinned prepared-model ZIP for offline installation')
    parser.add_argument('--repair', '--force', dest='repair', action='store_true')
    args = parser.parse_args()
    setup(args.runtime_root, args.repair, args.archive)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        message = 'Download failed; check your connection and Retry.' if isinstance(error, urllib.error.URLError) else str(error)
        print('Exhibit setup failed: '+message, file=sys.stderr)
        sys.exit(1)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install the current campus and guide from pinned public or offline archives."""
import argparse
import json
import os
from pathlib import Path
import sys
import urllib.error

from scene_assets import ensure_asset, progress, safe_member

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root',type=Path,default=ROOT/'.ov_library_lab_runtime')
    parser.add_argument('--profile',choices=('public',),default='public')
    parser.add_argument('--archive',type=Path,help='Pinned campus archive for offline installation')
    parser.add_argument('--guide-archive',type=Path,help='Pinned robot-guide archive for offline installation')
    parser.add_argument('--repair',action='store_true')
    args=parser.parse_args()
    if os.environ.get('OVLAB_SCENE')=='classic':
        progress('Using the procedural development scene by explicit request')
        return
    release=json.loads((ROOT/'config/scene-release.json').read_text(encoding='utf8'))
    runtime=args.runtime_root.resolve()
    cache=runtime/'cache/scene-assets'
    pin=release['campus']
    campus=safe_member(ROOT/'data/campus',pin['directory'])
    guide=runtime/'assets/lab-guide'
    for key,destination,explicit in (('campus',campus,args.archive),('guide',guide,args.guide_archive)):
        item=release[key]
        bundled=safe_member(ROOT/'data/scene-archives',item['filename'])
        archive=explicit or (bundled if bundled.is_file() else None)
        ensure_asset(item,release,destination,cache,archive,args.repair)
    registry=json.loads((ROOT/'config/campus-revisions.json').read_text(encoding='utf8'))['revisions']
    spec=registry[pin['revision']]
    if spec['directory']!=campus.name or spec['manifest_sha256']!=pin['manifest_sha256']:
        raise ValueError('Campus release pin and revision registry disagree')
    # Select the authored scene only after both it and the guide are verified.
    manifest=json.loads((campus/pin['manifest']).read_text(encoding='utf8'))
    if manifest['revision']!=pin['revision'] or manifest['layout']!=spec['layout']:
        raise ValueError('Scene and navigation layout disagree')
    selected=campus.parent/'active-revision.json'
    temporary=selected.with_suffix('.tmp')
    temporary.write_text(json.dumps({'revision':pin['revision']},indent=2)+'\n',encoding='utf8')
    os.replace(temporary,selected)
    progress('Current campus and robot guide ready: '+pin['revision'])


if __name__=='__main__':
    try:main()
    except (OSError,ValueError,RuntimeError) as error:
        message='Scene download failed; check your connection and Retry.' if isinstance(error,urllib.error.URLError) else str(error)
        print('Scene setup failed: '+message,file=sys.stderr)
        sys.exit(1)

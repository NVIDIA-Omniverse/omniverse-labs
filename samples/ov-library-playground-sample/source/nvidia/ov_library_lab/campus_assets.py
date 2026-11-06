# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Location and selection of the separately delivered campus art package."""
import os
import json
import hashlib
from .extensions import value
from pathlib import Path

DATA = Path(__file__).resolve().parents[3] / 'data'
R45_DIR = DATA / 'campus' / 'nvidia-campus-r45-portable'
R45_SCENE = R45_DIR / 'campus-lab.usdc'
R45_CACHE = DATA / 'campus' / 'runtime'
CAMPUS_DIR, CAMPUS_SCENE, CAMPUS_CACHE = R45_DIR, R45_SCENE, R45_CACHE
CAMPUS_REVISION = 'r45'
CAMPUS_LAYOUT = 'eight-gallery'
CAMPUS_REFINED = False
CAMPUS_VISITOR_POLISH = False
# Activation is local, explicit, and written only after manifest verification.
# Source-only installations continue to use the pinned original asset package.
selection = DATA/'campus/active-revision.json'
# PowerShell can leave an empty environment entry when clearing an override.
# Treat it as absent so the installed revision remains authoritative.
requested = os.environ.get('OVLAB_CAMPUS_REVISION', '').strip() or None
if requested is None and selection.is_file():
    requested = json.loads(selection.read_text(encoding='utf-8'))['revision']
if requested and requested != 'r45':
    registry = json.loads((DATA.parent/'config/campus-revisions.json').read_text(encoding='utf-8'))
    spec = registry['revisions'][requested]
    candidate = (DATA/'campus'/spec['directory']).resolve()
    if not candidate.is_relative_to((DATA/'campus').resolve()):
        raise ValueError('Campus revision must be inside the campus asset directory')
    if not all((candidate/p).is_file() for p in ('campus-lab.usda','runtime/environment.usda','runtime/navigation.json','manifest.json')):
        raise RuntimeError('Selected campus revision is incomplete; activate a verified package or select r45')
    manifest_bytes=(candidate/'manifest.json').read_bytes()
    if spec['manifest_sha256'] and hashlib.sha256(manifest_bytes).hexdigest()!=spec['manifest_sha256']:
        raise RuntimeError('Campus revision manifest differs from its pin; verify the installed revision')
    manifest=json.loads(manifest_bytes)
    if manifest['revision']!=requested or manifest['layout']!=spec['layout']:
        raise RuntimeError('Campus scene and application layout disagree')
    if not spec['manifest_sha256'] and not os.environ.get('OVLAB_CAMPUS_REVISION'):
        raise RuntimeError('An unreviewed campus cannot be the default scene')
    CAMPUS_DIR, CAMPUS_SCENE, CAMPUS_CACHE = candidate, candidate/'campus-lab.usda', candidate/'runtime'
    CAMPUS_REVISION = requested
    CAMPUS_LAYOUT = spec['layout']
    CAMPUS_REFINED = spec.get('refined_interiors', False)
    CAMPUS_VISITOR_POLISH = spec.get('visitor_polish', False)
CAMPUS_TEN_GALLERIES = CAMPUS_LAYOUT == 'ten-gallery'
CAMPUS_ENABLED = os.environ.get('OVLAB_SCENE', 'auto') != 'classic' and CAMPUS_SCENE.is_file()
# Preserve the complete catalog and implementations while this building has
# only eight galleries. Developer opt-in restores the previous station layout.
CAMPUS_HIDDEN_ROOMS = (frozenset({'package', *value('CAMPUS_HIDDEN', ())})
                       if CAMPUS_ENABLED and not CAMPUS_TEN_GALLERIES and os.environ.get('OVLAB_CAMPUS_EXHIBITS') != 'all'
                       else frozenset())

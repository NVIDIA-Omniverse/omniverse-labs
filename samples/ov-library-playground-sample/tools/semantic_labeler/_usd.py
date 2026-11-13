# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the semantic labeling tools: pxr import, taxonomy, rules, prim facts."""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
try:
    from pxr import Pcp, Usd, UsdGeom, UsdSemantics, UsdShade
except ImportError:  # the sample's Windows runtime supplies pxr through its libraries
    sys.path.insert(0, str(ROOT/'source'))
    from nvidia.ov_library_lab.bootstrap import configure
    configure()
    from pxr import Pcp, Usd, UsdGeom, UsdSemantics, UsdShade

TAXONOMY_FILE = ROOT/'config/semantic-taxonomy.json'
RULES_FILE = ROOT/'config/semantic-rules.json'
OVERRIDES_FILE = ROOT/'config/semantic-overrides.json'
SEMANTICS_DIR = ROOT/'data/semantics'
QCODE = re.compile(r'Q[0-9]+')
RENDERED = ('default', 'render')


def unit_id(path):
    """Filesystem-safe id for a prim path; crops are stored as <crops>/<unit_id>.png."""
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', path.strip('/'))[-120:]+'_'+hashlib.sha1(path.encode()).hexdigest()[:8]


def group_units(units):
    """{group: members}, largest member first: it is the representative the VLM sees."""
    groups = {}
    for unit in units:
        groups.setdefault(unit['group'], []).append(unit)
    for members in groups.values():
        members.sort(key=lambda u: (-max(u['size_m']), u['path']))
    return groups


def load_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default


def load_taxonomy():
    taxonomy = load_json(TAXONOMY_FILE)
    for name, entry in taxonomy['classes'].items():
        if not QCODE.fullmatch(entry['qcode']):
            raise SystemExit('Malformed Q-code for %s: %s' % (name, entry['qcode']))
    return taxonomy


def open_stage(path):
    stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    if stage is None:
        raise SystemExit('Cannot open stage: '+str(path))
    return stage


def is_rendered_gprim(prim):
    return prim.IsA(UsdGeom.Gprim) and UsdGeom.Imageable(prim).ComputePurpose() in RENDERED


def bound_material(prim):
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    return material.GetPrim() if material else None


def rendered_materials(gprim):
    """Materials a rendered gprim shows: its own binding plus its materialBind GeomSubsets
    (the scope SimReady SL.MAT.001 checks)."""
    found = [bound_material(gprim)]
    for subset in UsdShade.MaterialBindingAPI(gprim).GetMaterialBindSubsets():
        found.append(bound_material(subset.GetPrim()))
    return [m for m in found if m is not None]


def authored_assets(prim, stage_path):
    """Asset paths of this prim's direct reference/payload arcs (default prim: the file itself)."""
    assets = []
    if prim.HasAuthoredReferences() or prim.HasAuthoredPayloads():
        for arc in Usd.PrimCompositionQuery(prim).GetCompositionArcs():
            if arc.GetArcType() in (Pcp.ArcTypeReference, Pcp.ArcTypePayload):
                assets.append(arc.GetTargetLayer().identifier.replace('\\', '/'))
    if prim == prim.GetStage().GetDefaultPrim():
        assets.append(str(stage_path).replace('\\', '/'))
    return assets


def name_tokens(name):
    """'Pav_R43Walls_002' -> ['pav', 'r', 'walls']: split underscores, digits and CamelCase."""
    name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
    return [t.lower() for t in re.split(r'[\W_\d]+', name) if t]


def compile_rules(rules):
    compiled = []
    for rule in rules:
        tests = {k: re.compile(rule[k], re.I) for k in ('asset', 'name', 'path', 'type') if k in rule}
        for key in ('tokens', 'material'):
            if key in rule:
                tests[key] = re.compile('(?:%s)' % rule[key], re.I)
        compiled.append((tests, rule['label']))
    return compiled


def _words_match(rx, name):
    return any(rx.fullmatch(t) for t in name_tokens(name))


def match_rule(compiled, name, path, type_name, assets=(), materials=()):
    """First rule whose every test passes. 'material' needs all bound materials to match."""
    for tests, label in compiled:
        ok = True
        for key, rx in tests.items():
            if key == 'asset':
                ok = any(rx.search(a) for a in assets)
            elif key == 'tokens':
                ok = _words_match(rx, name)
            elif key == 'material':
                ok = bool(materials) and all(_words_match(rx, m) for m in materials)
            elif key == 'name':
                ok = bool(rx.search(name))
            elif key == 'path':
                ok = bool(rx.search(path))
            else:
                ok = bool(rx.search(type_name))
            if not ok:
                break
        if ok:
            return label
    return None

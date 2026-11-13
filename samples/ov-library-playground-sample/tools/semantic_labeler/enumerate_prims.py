# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage 1+2: walk a USD stage, apply deterministic label rules, list what is left for the VLM.

Writes data/semantics/<stage>.labels.json:
  labels     {prim path: {label, source}} for the highest labeled prims (descendants inherit)
  materials  {material path: {label, source}} for rendered Material prims (SimReady SL.MAT.001)
  existing   {prim path: [Q-codes]} already labeled in the source (kept, never relabeled)
  units      unlabeled object-sized prims that still need a class (input to vlm_label.py)
  unlabeled_materials  rendered materials no rule matched (fix with config/semantic-overrides.json)
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _usd import (OVERRIDES_FILE, RULES_FILE, SEMANTICS_DIR, Usd, UsdGeom, UsdSemantics, authored_assets,
                  compile_rules, rendered_materials, is_rendered_gprim, load_json, load_taxonomy,
                  match_rule, name_tokens, open_stage)


def group_key(name, assets, ignore, materials, extent):
    """Instances of one object share a key so the VLM labels it once.

    Drops room, material, direction and revision words so that, for example,
    every room's copy of the same trim profile becomes one group. Names with no
    meaningful words left group by bound materials and size (0.1 m, axis order
    ignored), since repeated geometry has repeated dimensions.
    """
    words = sorted({t for t in name_tokens(name) if not ignore.fullmatch(t)})
    if not words:
        words = sorted(materials)+['%.1f' % v for v in sorted(extent)]
    return (Path(assets[0]).name if assets else '')+'|'+'_'.join(words)


def existing_labels(prim, instance):
    """Labels already authored on the prim (SimReady assets ship their own)."""
    if not prim.HasAPI(UsdSemantics.LabelsAPI, instance):
        return None
    values = UsdSemantics.LabelsAPI(prim, instance).GetLabelsAttr().Get()
    return list(values) if values else None


def subtree_facts(prim, meshes, materials, uncovered, covered=False):
    """Post-order: count meshes, collect bound material paths, and count rendered gprims
    that no existing (source-authored) wikidata_qcode label covers, for every prim."""
    covered = covered or bool(existing_labels(prim, 'wikidata_qcode'))
    count, names, open_ = (1 if prim.IsA(UsdGeom.Mesh) else 0), set(), 0
    if is_rendered_gprim(prim):
        open_ += 0 if covered else 1
        names.update(m.GetPath().pathString for m in rendered_materials(prim))
    for child in prim.GetChildren():
        c, n, o = subtree_facts(child, meshes, materials, uncovered, covered)
        count += c
        names |= n
        open_ += o
    path = prim.GetPath().pathString
    meshes[path], materials[path], uncovered[path] = count, names, open_
    return count, names, open_


def enumerate_stage(stage, stage_path, max_meshes=150, min_size=0.05):
    rules_doc = load_json(RULES_FILE)
    rules = compile_rules(rules_doc['rules'])
    material_rules = compile_rules(rules_doc['material_rules'])
    skip_types = set(rules_doc.get('skip_types', ()))
    ignore = re.compile(rules_doc.get('ignore_tokens', '$^'), re.I)
    taxonomy = load_taxonomy()
    classes, material_vocab = set(taxonomy['classes']), set(taxonomy['materials'])
    overrides = load_json(OVERRIDES_FILE, {})
    for kind, vocab in (('labels', classes), ('materials', material_vocab)):
        for path, label in overrides.get(kind, {}).items():
            if label not in vocab:
                raise SystemExit('Override %s uses a label outside the taxonomy: %s' % (path, label))
    object_overrides = overrides.get('labels', {})

    meshes, bound, uncovered = {}, {}, {}
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    subtree_facts(root, meshes, bound, uncovered)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    labels, units = {}, []
    stats = {'skipped': 0, 'rule': 0, 'override': 0, 'existing': 0, 'units': 0}
    existing = {}

    def visit(prim):
        path = prim.GetPath().pathString
        if path in object_overrides:
            labels[path] = {'label': object_overrides[path], 'source': 'override'}
            stats['override'] += 1
            return
        if meshes.get(path, 0) and not uncovered.get(path, 0):
            existing[path] = existing_labels(prim, 'wikidata_qcode') or ['(labeled below)']
            stats['existing'] += 1            # source labels already cover this subtree; keep them
            return
        type_name = str(prim.GetTypeName())
        if type_name in skip_types or prim.GetName() == 'Looks' or meshes.get(path, 0) == 0 and not prim.IsA(UsdGeom.Gprim):
            stats['skipped'] += 1            # shading, lights, cameras, empty rig helpers
            return
        assets = authored_assets(prim, stage_path)
        material_names = [p.rsplit('/', 1)[1] for p in bound.get(path, ())]
        label = match_rule(rules, prim.GetName(), path, type_name, assets, material_names)
        if label:
            labels[path] = {'label': label, 'source': 'rule'}
            stats['rule'] += 1
            return
        children = prim.GetChildren()
        if meshes.get(path, 0) <= max_meshes or not children:
            size = cache.ComputeWorldBound(prim).ComputeAlignedRange().GetSize()
            extent = [round(float(v)*mpu, 4) for v in size]
            if max(extent, default=0) >= min_size:
                units.append({'path': path, 'name': prim.GetName(), 'type': type_name,
                              'parent': prim.GetParent().GetPath().pathString,
                              'assets': [Path(a).name for a in assets], 'materials': sorted(material_names),
                              'meshes': meshes.get(path, 0), 'size_m': extent,
                              'group': group_key(prim.GetName(), assets, ignore, material_names, extent)})
                stats['units'] += 1
            else:
                stats['skipped'] += 1
            return
        for child in children:
            visit(child)

    visit(root)

    # Rendered materials (SimReady SL.MAT.001): only those bound by render/default gprims.
    material_labels, unlabeled_materials = {}, []
    material_overrides = overrides.get('materials', {})
    for path in sorted(bound.get(root.GetPath().pathString, ())):
        name = path.rsplit('/', 1)[1]
        if existing_labels(stage.GetPrimAtPath(path), taxonomy['material_taxonomy']):
            continue
        if path in material_overrides:
            material_labels[path] = {'label': material_overrides[path], 'source': 'override'}
        else:
            label = match_rule(material_rules, name, path, 'Material')
            if label:
                material_labels[path] = {'label': label, 'source': 'rule'}
            else:
                unlabeled_materials.append(path)
    return {'stage': str(stage_path).replace('\\', '/'), 'labels': labels, 'existing': existing, 'units': units,
            'materials': material_labels, 'unlabeled_materials': unlabeled_materials,
            'stats': stats, 'total_meshes': meshes.get(root.GetPath().pathString, 0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('stage', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--max-meshes', type=int, default=150,
                    help='Largest unlabeled subtree treated as one object for the VLM')
    ap.add_argument('--min-size', type=float, default=0.05, help='Ignore units smaller than this (meters)')
    args = ap.parse_args()
    stage = open_stage(args.stage)
    result = enumerate_stage(stage, args.stage.resolve(), args.max_meshes, args.min_size)
    out = args.out or SEMANTICS_DIR/(args.stage.stem+'.labels.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    groups = {u['group'] for u in result['units']}
    print('%s: %d meshes | rule %d override %d existing %d | %d unlabeled units in %d groups | materials %d labeled, %d not -> %s' % (
        args.stage.name, result['total_meshes'], result['stats']['rule'], result['stats']['override'], result['stats']['existing'],
        len(result['units']), len(groups), len(result['materials']), len(result['unlabeled_materials']), out))


if __name__ == '__main__':
    main()

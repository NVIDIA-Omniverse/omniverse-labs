# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check a stage against the SimReady semantic-label requirements and report coverage.

  SL.001       every rendered gprim resolves a label: own, ancestor, or bound material (or its ancestors)
  SL.QCODE.001 every wikidata_qcode value is well formed (Q + digits)
  SL.MAT.001   every material bound by rendered geometry carries a material label
  SL.TIME.001  no label attribute is time-sampled
SL.001 is reported for the object taxonomy (--taxonomy, default wikidata_qcode); a material
label alone does not count as an object class here, so the class breakdown stays honest.
This is a fast local check, not a replacement for the official simready-validate tool.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _usd import QCODE, Usd, UsdGeom, UsdSemantics, bound_material, is_rendered_gprim, load_taxonomy, open_stage, rendered_materials


def own_labels(prim, instance):
    if not prim.HasAPI(UsdSemantics.LabelsAPI, instance):
        return None
    values = UsdSemantics.LabelsAPI(prim, instance).GetLabelsAttr().Get()
    return list(values) if values else None


def inherited(prim, instance):
    while prim and not prim.IsPseudoRoot():
        values = own_labels(prim, instance)
        if values:
            return values
        prim = prim.GetParent()
    return None


def measure(stage, instance, material_instance):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    counts, volume, unlabeled = Counter(), Counter(), []
    bad_qcodes, time_sampled = set(), set()
    materials = {}
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    for prim in Usd.PrimRange(root):
        for name in UsdSemantics.LabelsAPI.GetDirectTaxonomies(prim):
            attr = UsdSemantics.LabelsAPI(prim, name).GetLabelsAttr()
            if attr and attr.GetNumTimeSamples():
                time_sampled.add(prim.GetPath().pathString)
            if name == 'wikidata_qcode':
                for value in attr.Get() or ():
                    if not QCODE.fullmatch(value):
                        bad_qcodes.add('%s=%s' % (prim.GetPath(), value))
        if not is_rendered_gprim(prim):
            continue
        for shown in rendered_materials(prim):
            materials[shown.GetPath().pathString] = bool(inherited(shown, material_instance))
        material = bound_material(prim)
        labels = inherited(prim, instance) or (inherited(material, instance) if material else None)
        size = cache.ComputeWorldBound(prim).ComputeAlignedRange().GetSize()
        vol = max(float(size[0]*size[1]*size[2]), 0.0)
        key = ','.join(labels) if labels else '(unlabeled)'
        counts[key] += 1
        volume[key] += vol
        if not labels:
            unlabeled.append((vol, prim.GetPath().pathString))
    return counts, volume, sorted(unlabeled, reverse=True), bad_qcodes, time_sampled, materials


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('stage', type=Path)
    ap.add_argument('--layer', type=Path, action='append', default=[], help='Sidecar layer(s) to compose on top')
    ap.add_argument('--taxonomy', default='wikidata_qcode')
    ap.add_argument('--top', type=int, default=10, help='List the N largest unlabeled gprims')
    ap.add_argument('--min-count', type=float, default=95.0, help='Fail below this %% of rendered gprims labeled')
    args = ap.parse_args()
    stage = open_stage(args.stage)
    for layer in args.layer:
        stage.GetRootLayer().subLayerPaths.append(str(layer.resolve()))
    taxonomy = load_taxonomy()
    names = {c['qcode']: k for k, c in taxonomy['classes'].items()}
    counts, volume, unlabeled, bad, timed, materials = measure(stage, args.taxonomy, taxonomy['material_taxonomy'])
    total, miss = sum(counts.values()), counts['(unlabeled)']
    pct = 100.0*(total-miss)/total if total else 100.0
    for label, n in counts.most_common():
        print('  %-22s %6d gprims %12.1f m^3 bbox' % ('%s %s' % (label, names.get(label, '')), n, volume[label]))
    labeled_mats = sum(materials.values())
    print('SL.001       %.1f%% of %d rendered gprims labeled (unlabeled listed largest first)' % (pct, total))
    print('SL.QCODE.001 %d malformed' % len(bad))
    print('SL.MAT.001   %d/%d rendered materials labeled' % (labeled_mats, len(materials)))
    print('SL.TIME.001  %d time-sampled label attributes' % len(timed))
    for vol, path in unlabeled[:args.top]:
        print('  unlabeled %10.3f  %s' % (vol, path))
    for path in sorted(p for p, ok in materials.items() if not ok)[:args.top]:
        print('  material without label: '+path)
    ok = pct >= args.min_count and not bad and not timed and labeled_mats == len(materials)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

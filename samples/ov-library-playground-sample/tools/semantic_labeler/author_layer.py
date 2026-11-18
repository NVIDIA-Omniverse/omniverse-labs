# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage 4: write a sparse USD layer of SimReady semantic labels for the labeled prims.

Objects get SemanticsLabelsAPI:wikidata_qcode (Q-code, SimReady SL.003/SL.QCODE.001) plus
SemanticsLabelsAPI:wikidata_class (the Wikidata English label), the convention SimReady
assets use; rendered materials get SemanticsLabelsAPI:material
(SL.MAT.001). Values are static (SL.TIME.001). The layer goes to data/scene-overrides/
beside the other corrections, so campus.py composes it the same way. Source assets are
never edited.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _usd import ROOT, Usd, UsdSemantics, load_taxonomy

HEADER = ('# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. '
          'All rights reserved.\n# SPDX-License-Identifier: Apache-2.0\n')


def apply_labels(prim, instance, values):
    UsdSemantics.LabelsAPI.Apply(prim, instance).CreateLabelsAttr(list(values))


def author(labels_doc, base_revision='r48-wayfinding-landscape'):
    taxonomy = load_taxonomy()
    classes, material_vocab = taxonomy['classes'], set(taxonomy['materials'])
    stage = Usd.Stage.CreateInMemory()
    for path in sorted(labels_doc['labels']):
        label = labels_doc['labels'][path]['label']
        if label not in classes:                     # 'unknown' is never authored
            raise SystemExit('Label outside the taxonomy for %s: %s' % (path, label))
        prim = stage.OverridePrim(path)
        apply_labels(prim, 'wikidata_qcode', [classes[label]['qcode']])
        apply_labels(prim, 'wikidata_class', [classes[label]['wikidata_label']])
    for path in sorted(labels_doc.get('materials', {})):
        label = labels_doc['materials'][path]['label']
        if label not in material_vocab:
            raise SystemExit('Material label outside the taxonomy for %s: %s' % (path, label))
        apply_labels(stage.OverridePrim(path), taxonomy['material_taxonomy'], [label])
    layer = stage.GetRootLayer()
    layer.customLayerData = {'baseRevision': base_revision,
                             'purpose': 'SimReady semantic labels (wikidata_qcode, wikidata_class, material) for ovannotators datasets'}
    text = layer.ExportToString()
    first, rest = text.split('\n', 1)           # keep '#usda 1.0' on line 1 for check_licensing.py
    return first+'\n'+HEADER+rest


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('labels', type=Path, nargs='+', help='*.labels.json from enumerate_prims.py / vlm_label.py')
    ap.add_argument('--out', type=Path, default=ROOT/'data/scene-overrides/r52-semantic-labels.usda')
    ap.add_argument('--base-revision', default='r48-wayfinding-landscape')
    args = ap.parse_args()
    merged = {'labels': {}, 'materials': {}}
    for file in args.labels:
        doc = json.loads(file.read_text(encoding='utf-8'))
        merged['labels'].update(doc['labels'])
        merged['materials'].update(doc.get('materials', {}))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(author(merged, args.base_revision), encoding='utf-8', newline='\n')
    print('%d objects, %d materials -> %s (%.1f KB)' % (
        len(merged['labels']), len(merged['materials']), args.out, args.out.stat().st_size/1024))


if __name__ == '__main__':
    main()

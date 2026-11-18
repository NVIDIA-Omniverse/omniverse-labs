# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Review gate: write an HTML sheet of VLM proposals and a starter overrides file.

VLM labels are proposals. A reviewer checks the sheet, then records corrections
in config/semantic-overrides.json ({"labels": {prim path: class}}). Overrides win
over rules and the VLM on the next enumerate_prims.py run.
"""
import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _usd import OVERRIDES_FILE, SEMANTICS_DIR, unit_id

STYLE = '''body{font:14px system-ui,sans-serif;margin:16px;background:#fff;color:#111}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:6px;text-align:left;vertical-align:top}
th{position:sticky;top:0;background:#f4f4f4}img{max-width:160px;max-height:120px}
.low{background:#fff4e5}.path{font-family:ui-monospace,monospace;font-size:12px;word-break:break-all}
@media (prefers-color-scheme:dark){body{background:#151515;color:#eee}th{background:#222}td,th{border-color:#333}.low{background:#3a2a10}}'''


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('labels', type=Path)
    ap.add_argument('--crops', type=Path, default=SEMANTICS_DIR/'crops')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    doc = json.loads(args.labels.read_text(encoding='utf-8'))
    rows = [dict(v, path=k) for k, v in doc['labels'].items() if v['source'] == 'vlm']
    rows += [dict(r, low=True) for r in doc.get('needs_review', [])]
    seen, groups = set(), []
    for row in sorted(rows, key=lambda r: (not r.get('low'), r['confidence'], r['group'])):
        if row['group'] not in seen:                 # one row per instance group
            seen.add(row['group'])
            groups.append((row, sum(1 for r in rows if r['group'] == row['group'])))
    body = []
    for row, count in groups:
        crop = args.crops/(unit_id(row['path'])+'.png')
        img = '<img src="%s">' % html.escape(crop.resolve().as_uri()) if crop.is_file() else ''
        body.append('<tr class="%s"><td>%s</td><td><b>%s</b><br>%.2f</td><td class="path">%s</td><td>%d</td><td>%s</td></tr>' % (
            'low' if row.get('low') else '', img, html.escape(row['label']), row['confidence'],
            html.escape(row['path']), count, html.escape(row.get('reason', ''))))
    out = args.out or args.labels.with_suffix('.review.html')
    out.write_text('<!doctype html><meta charset="utf-8"><title>Semantic label review</title><style>%s</style>'
                   '<h1>Semantic label review</h1><p>%s: %d groups (%d need review, highlighted first). '
                   'Record corrections in config/semantic-overrides.json.</p>'
                   '<table><tr><th>View</th><th>Label</th><th>Prim (group representative)</th><th>Instances</th><th>Reason</th></tr>%s</table>'
                   % (STYLE, html.escape(args.labels.name), len(groups),
                      sum(1 for r, _ in groups if r.get('low')), ''.join(body)), encoding='utf-8')
    if not OVERRIDES_FILE.is_file():
        OVERRIDES_FILE.write_text(json.dumps({'schema_version': 1, 'labels': {}}, indent=2)+'\n', encoding='utf-8')
    print('%d groups -> %s' % (len(groups), out))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Test-only Pillow: taken from the consumer/test environment (scripts/requirements-public-sdk-consumer.txt), never the shipped SDK runtime.
"""Labeled evidence contact sheet; never mix Workbench and native frames."""
import argparse
import json
from pathlib import Path
from PIL import Image,ImageDraw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kind',choices=('native','authoring'),required=True)
    p.add_argument('--limit',type=int,default=24)
    args=p.parse_args()
    paths=sorted(args.input.glob('*/00-baseline.png') if args.kind=='native' else args.input.glob('*/blender-authoring-preview.png'))[:min(args.limit,36)]
    if not paths:raise SystemExit('No matching evidence images; nothing synthesized')
    columns=min(6,len(paths));rows=(len(paths)+columns-1)//columns
    sheet=Image.new('RGB',(columns*256,rows*240+40),'#20242a');draw=ImageDraw.Draw(sheet)
    draw.text((12,12),'NATIVE OVRTX MEASUREMENTS' if args.kind=='native' else 'BLENDER WORKBENCH AUTHORING PREVIEWS — NOT OVRTX',fill='white')
    for i,path in enumerate(paths):
        image=Image.open(path).convert('RGB');image.thumbnail((256,192))
        x=(i%columns)*256;y=(i//columns)*240+40
        sheet.paste(image,(x,y));draw.text((x+4,y+195),path.parent.name[:38],fill='white')
        if args.kind=='native':
            report=json.loads((path.parent/'result.json').read_text())
            weak=any(f.get('pixel_mae_from_previous') is not None and not f.get('changed_above_noise') for f in report.get('frames',[]))
            label='GAP' if report['status']!='native_frames_measured' else 'REVIEW: weak edit signal' if weak else 'Measured; semantics need review'
            draw.text((x+4,y+213),label,fill='#ffc576' if weak or report['status']!='native_frames_measured' else '#a5d8ff')
    args.output.parent.mkdir(parents=True,exist_ok=True);sheet.save(args.output)
    args.output.with_suffix('.json').write_text(json.dumps({'kind':args.kind,'images':[str(p.resolve()) for p in paths]},indent=2)+'\n')
    print(args.output)


if __name__=='__main__':main()

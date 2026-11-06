#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a local, offline gallery of actual native evidence and recorded gaps."""
import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,nargs='+',required=True)
    p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    collection=json.loads(args.collection.read_text());native={}
    for directory in args.native:
        for path in sorted(directory.glob('*/result.json')):
            native[path.parent.name]=(json.loads(path.read_text()),path)
    def link(path):return html.escape(os.path.relpath(path,args.output.parent))
    def esc(value):return html.escape(str(value))
    cards=[];states=Counter()
    for case in collection['cases']:
        name=case['case'];entry=native.get(name);images='';metrics='';record=case.get('result')
        if entry:
            data,path=entry;record=str(path)
            frames=data.get('frames',[]);changes=[f for f in frames if f.get('pixel_mae_from_previous') is not None]
            uncertain=[f['operation'] for f in changes if not f.get('changed_above_noise')]
            state=('native gap' if data['status']!='native_frames_measured' else
                   'review: weak edit signal' if uncertain else 'native images measured')
            details=data.get('error','')
            if uncertain:details='Edit difference did not exceed noise gate: '+', '.join(uncertain)
            for i,frame in enumerate(frames):
                image=path.parent/f'{i:02d}-{frame["operation"]}.png'
                if image.is_file():images+=f'<figure><img loading="lazy" src="{link(image)}"><figcaption>{esc(frame["operation"])} · MAE {esc(frame.get("pixel_mae_from_previous"))}</figcaption></figure>'
            metrics=f'<p>Same session: {esc(data.get("same_session_for_all_edits",False))}; no-edit MAE: {esc(data.get("no_edit_pixel_mae"))}.</p>'
        else:
            state=('export only' if case['status']=='export_passed' else 'author/export gap' if case['status']!='not_run' else 'not run')
            details=case.get('error') or case['status']
        states[state]+=1
        cards.append(f'<article data-native="{str(bool(entry)).lower()}" data-state="{esc(state)}"><h2>{esc(name)}</h2><b class="badge">{esc(state)}</b><p>{esc(details)}</p>{metrics}<div class="frames">{images}</div><p>{"<a href="+chr(34)+link(record)+chr(34)+">Measured record</a>" if record else "No record"}</p></article>')
    summary=' · '.join(f'{count} {name}' for name,count in states.items())
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Blender / OVRTX test zoo</title>
<style>body{font:16px system-ui;background:#15191e;color:#eee;margin:24px}h1{font-size:32px}p{max-width:1100px;line-height:1.45}input,select{font:inherit;padding:10px;margin:8px}article{border:1px solid #49515c;padding:18px;border-radius:10px;margin:16px 0}h2{font-size:20px;margin-top:0}.badge{background:#334657;padding:4px 8px}.frames{display:flex;gap:12px;overflow:auto}figure{margin:0;flex:0 0 384px}img{width:384px;height:288px;object-fit:contain;background:#080a0c}figcaption{font-size:13px;margin:5px 0}a{color:#9acaff}article[data-state*="gap"] .badge,article[data-state*="review"] .badge{background:#745226}</style>
<h1>Blender / OVRTX test zoo</h1><p>Actual OVRTX images from Blender-authored or explicitly direct-USD scenes. Baseline and edits appear side by side. Every measured edit checks that the renderer process, session epoch and native stage remain unchanged. Export-only records and gaps are shown separately.</p><p><strong>This is a capability investigation, not a BRDF or physics certification.</strong> A changed image establishes an observable effect; a weak difference needs a targeted view or stronger test. An accepted write or export never counts as a native pass. Direct OpenPBR cases bypass Blender's Principled converter.</p>'''
    page+=f'<p>{esc(summary)}</p><input id="search" placeholder="Filter case, e.g. coat or texture"><select id="state"><option value="__native__">Native evidence</option><option value="">All evidence</option>'+''.join(f'<option>{esc(s)}</option>' for s in states)+'</select>'
    page+=''.join(cards)+'''<script>function filter(){let q=document.querySelector('#search').value.toLowerCase(),s=document.querySelector('#state').value;document.querySelectorAll('article').forEach(a=>a.hidden=!(a.textContent.toLowerCase().includes(q)&&(!s||(s==='__native__'?a.dataset.native==='true':a.dataset.state===s))))}document.querySelector('#search').oninput=filter;document.querySelector('#state').onchange=filter;filter();</script></html>'''
    args.output.write_text(page)
    args.output.with_suffix('.json').write_text(json.dumps({'states':dict(states),'native_runs':[str(x.resolve()) for x in args.native],'collection':str(args.collection.resolve())},indent=2)+'\n')
    print(json.dumps(dict(states)))


if __name__=='__main__':main()

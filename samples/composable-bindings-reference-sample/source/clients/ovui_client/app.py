# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone 2D view of the existing shared client; no Kit, USD or RTX imports.

The service supplies bindings, display definitions, values and alarm decisions.
This adapter owns only local widgets. New configured metrics appear automatically.
"""
import argparse
import asyncio
import ctypes
import json
import os
from pathlib import Path
import sys
import time

from composable_bindings.reference.client import Client
from composable_bindings.reference.config import source_key


_DLL_DIRECTORIES = []

def load_ui():
    # The ovui 0.2.0 wheel dynamically links CUDA runtime even for its 2D UI.
    # Load the pip dependency explicitly, without taking libraries from Kit.
    if sys.platform.startswith('linux'):
        import site
        for directory in site.getsitepackages():
            runtime = Path(directory) / 'nvidia/cuda_runtime/lib/libcudart.so.12'
            if runtime.is_file():
                ctypes.CDLL(str(runtime), mode=ctypes.RTLD_GLOBAL)
                break
    elif sys.platform == 'win32' and not _DLL_DIRECTORIES:
        site = Path(sys.prefix) / 'Lib/site-packages'
        for relative in ('omni/ui', 'omni/ui_scene', 'nvidia/cuda_runtime/bin'):
            directory = site / relative
            if directory.is_dir():
                # Keep handles alive while the native UI libraries are loaded.
                _DLL_DIRECTORIES.append(os.add_dll_directory(str(directory)))
    import omni.ui as ui
    from omni.ui import scene
    return ui, scene


def color(hex_value):
    r, g, b = bytes.fromhex(hex_value)
    return 0xff000000 | b << 16 | g << 8 | r


TONES = {'normal': (.015, .48, .36, 1.), 'alarm': (.9, .08, .04, 1.),
         'unknown': (.45, .52, .58, 1.)}


class Gauge:
    """One reusable flat vector gauge for every configured numeric metric."""
    def __init__(self, ui, sc, spec, height=178):
        import math
        self.spec, self.previous = spec, None
        self.start, self.sweep = -math.pi * 1.25, math.pi * 1.5
        projection = [1/72,0,0,0, 0,1/72,0,0, 0,0,-.01,0, 0,0,0,1]
        self.view = sc.SceneView(sc.CameraModel(projection, sc.Matrix44()), height=height,
            aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT)
        with self.view.scene:
            def label(text, y, size, tone=(.06,.12,.17,1.)):
                with sc.Transform(transform=sc.Matrix44.get_translation_matrix(0,y,.1)):
                    return sc.Label(text, size=size, color=tone, alignment=ui.Alignment.CENTER)
            self.title = label(spec['title'], 62, 18)
            self.track = sc.Arc(44, begin=self.start, end=self.start+self.sweep,
                axis=2, thickness=5, color=(.34,.43,.49,1.), wireframe=True,
                sector=False, tesselation=64)
            self.arc = sc.Arc(44, begin=self.start, end=self.start+.001,
                axis=2, thickness=5, color=TONES['unknown'], wireframe=True,
                sector=False, tesselation=64)
            self.value = label('Waiting', 3, 21)
            self.state = label('WAITING', -20, 11)
            limits = spec.get('range')
            label(f'{limits[0]:g} .. {limits[1]:g}' if limits else '', -52, 11)
            self.track.visible = self.arc.visible = spec['kind']=='gauge'
            if limits:
                for i in range(9):
                    a = self.start+self.sweep*i/8
                    sc.Line([49*math.cos(a),49*math.sin(a),0],
                            [53*math.cos(a),53*math.sin(a),0],
                            color=(.34,.43,.49,1.), thickness=1)

    def update(self, display, stale=False):
        state = (display['text'], display['state'], display['fraction'], stale)
        if state == self.previous:
            return
        self.previous = state
        tone = TONES['unknown' if stale else display['state']]
        self.value.text = display['text']
        self.state.text = 'STALE' if stale else display['state'].upper()
        self.value.color = self.state.color = self.arc.color = tone
        self.arc.end = self.start + max(.001,self.sweep*display['fraction'])
        self.arc.visible = self.spec['kind']=='gauge' and display['sample'] is not None


class Dashboard:
    """Config-driven line and machine cards consuming Client.snapshot()."""
    def __init__(self, client, ui, sc):
        self.client, self.ui, self.sc = client, ui, sc
        self.signature = None
        self.gauges = {}
        self.machine_labels = {}
        self.last_revision = None
        self.last_tick = 0.
        self.window = ui.Window('Production line | Live 2D', width=1260, height=860)
        self.window.frame.set_style({
            'Window': {'background_color': color('EDF3F7')},
            'Label': {'color': color('18313F'), 'font_size': 16},
            'ScrollingFrame': {'background_color': color('EDF3F7')},
        })
        with self.window.frame:
            with ui.VStack(spacing=10):
                ui.Label('PRODUCTION LINE  /  LIVE OPERATIONS', height=42,
                         style={'font_size':26, 'color':color('123C4E')})
                self.status = ui.Label('Connecting to the shared service…',height=26)
                with ui.ScrollingFrame():
                    self.content = ui.Frame()
                ui.Label('Standalone ovui · Shared Python client · Live values and source or service-evaluated alarms',
                         height=26, style={'font_size':13, 'color':color('537080')})

    @staticmethod
    def layout_signature(rows):
        return json.dumps([{'binding':r['binding'],
                            'displays':[d['spec'] for d in r['displays']]} for r in rows],sort_keys=True)

    def rebuild(self, rows):
        ui, sc = self.ui, self.sc
        self.gauges.clear()
        self.machine_labels.clear()
        line_rows = [r for r in rows if any(d['spec'].get('placement')=='hud-top-left' for d in r['displays'])]
        machines = [r for r in rows if r not in line_rows]
        with self.content:
            with ui.HStack(spacing=16):
                with ui.VStack(width=275, spacing=12):
                    ui.Label('LINE OVERVIEW',height=30,style={'font_size':18})
                    for row in line_rows:
                        ui.Label(row['binding']['label'],height=30)
                        for d in row['displays']:
                            key=(source_key(row['binding']),d['spec']['id'])
                            self.gauges[key]=Gauge(ui,sc,d['spec'],height=235)
                    ui.Label(f'{len(line_rows)} production line\n{len(machines)} connected machines',height=65,word_wrap=True)
                    ui.Label('Gauges show measured values.\nGreen: normal   Red: alarm\nGray: missing or stale',height=85,word_wrap=True,
                             style={'font_size':14,'color':color('537080')})
                    ui.Spacer()
                with ui.VStack(spacing=10):
                    with ui.HStack(spacing=16):
                        for row in machines:
                            with ui.VStack(spacing=5):
                                ui.Label(row['binding']['label'],height=34,alignment=ui.Alignment.CENTER,
                                         style={'font_size':22})
                                # Temperature is nearest the machine; new metrics grow upwards.
                                for d in reversed(row['displays']):
                                    key=(source_key(row['binding']),d['spec']['id'])
                                    self.gauges[key]=Gauge(ui,sc,d['spec'])
                                with ui.ZStack(height=64):
                                    ui.Rectangle(style={'background_color':color('C4E3E9'),
                                                        'border_radius':12,'border_width':2,'border_color':color('6FA4B1')})
                                    self.machine_labels[source_key(row['binding'])]=ui.Label(
                                        row['binding']['label'],alignment=ui.Alignment.CENTER,style={'font_size':19})
                    with ui.ZStack(height=46):
                        ui.Rectangle(style={'background_color':color('D1DFE7'),'border_radius':6})
                        ui.Label('PRODUCTION LINE    •    '+'  >  '.join(r['binding']['label'] for r in machines),alignment=ui.Alignment.CENTER)
                    ui.Spacer()

    def tick(self, force=False):
        now=time.monotonic()
        if not force and now-self.last_tick < .05:
            return
        self.last_tick=now
        snapshot=self.client.snapshot()
        if snapshot['revision']==self.last_revision and not force:
            return
        self.last_revision=snapshot['revision']
        rows=snapshot['rows']
        signature=self.layout_signature(rows)
        if signature!=self.signature:
            self.signature=signature
            self.content.clear()
            self.rebuild(rows)
        stale=not snapshot['status'].startswith('Live')
        samples=[d['sample'] for row in rows for d in row['displays'] if d['sample']]
        observed=max((s.observed for s in samples),default=None)
        stamp=time.strftime('%H:%M:%S',time.localtime(observed)) if observed else 'waiting'
        self.status.text=f"{snapshot['status']}  ·  Last observation {stamp}  ·  {len(self.gauges)} metrics"
        for row in rows:
            for d in row['displays']:
                self.gauges[(source_key(row['binding']),d['spec']['id'])].update(d,stale)
            label=self.machine_labels.get(source_key(row['binding']))
            if label:
                alarm=any(d['state']=='alarm' for d in row['displays'])
                label.style={'font_size':19,'color':color('B91C1C' if alarm and not stale else '18313F')}
        # This visual client doesn't use the shell log queue.
        self.client.drain_logs()

    def close(self):
        self.window.destroy()


def main(argv=None):
    root=Path(__file__).resolve().parents[3]
    parser=argparse.ArgumentParser(description='Standalone ovui live 2D production-line client')
    parser.add_argument('--client-config',type=Path,default=root/'.runtime/default/client.json')
    parser.add_argument('--view',type=Path,help='Optional local display configuration override')
    args=parser.parse_args(argv)
    if not args.client_config.is_file():
        parser.error('Start the existing service first, or provide --client-config.')
    ui,sc=load_ui()
    client=Client(args.client_config,args.view).start()
    ui.init('Composable Bindings — Live 2D',width=1280,height=900,max_fps=30)
    async def run():
        dashboard=Dashboard(client,ui,sc)
        try:
            while dashboard.window.visible:
                dashboard.tick()
                await ui.next_frame()
        finally:
            dashboard.close()
    try:
        ui.run(run())
    finally:
        client.stop()
        ui.shutdown()


if __name__=='__main__':
    main()

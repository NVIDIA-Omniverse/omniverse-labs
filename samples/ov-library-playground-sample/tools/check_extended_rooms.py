# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native integration and visual checks for Newton and Package; RTX required."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
if '--desktop' in sys.argv:
    os.environ.pop('OMNIUI_HEADLESS',None)
    os.environ.pop('OMNIUI_BACKEND',None)
else:
    os.environ['OMNIUI_HEADLESS']='1'
    os.environ['OMNIUI_BACKEND']='vulkan'
from nvidia.ov_library_lab.app import LibraryLab,ui
from nvidia.ov_library_lab.content import ROOMS
from nvidia.ov_library_lab.layout import PLACEMENTS,LOBBY_POSE,FRONT_Z,pose,room_at,contains
from omni.ui import _ui


async def run(args):
    pump = asyncio.create_task(ui.run_async())
    lab = LibraryLab(args.runtime_root,'public','Performance')
    lab.settings.text_scale = args.text_scale
    args.output.mkdir(parents=True,exist_ok=True)
    report = {'checks':[]}
    async def frames(n=5):
        for _ in range(n):
            frame=await ui.next_frame()
            lab.update(frame.dt)
    async def until(predicate,seconds=90):
        limit=time.monotonic()+seconds
        while not predicate():
            assert time.monotonic()<limit,'Timed out waiting for the native experiment'
            await frames(2)
    async def capture(name):
        lab.viewport.update(lab.native.render('Main'))
        await ui.next_frame()
        assert _ui._schedule_screenshot(str(args.output/(name+'.png')))
        await ui.next_frame()
        assert _ui._get_screenshot_result()['success']
    def passed(name):
        report['checks'].append(name)
        print('PASS',name,flush=True)
    async def click(widget):
        dpi=(_ui._standalone_get_window_size()[0] or args.width)/ui.Workspace.get_main_window_width()
        # Reach controls naturally in a shorter laptop panel via wheel input.
        panel = lab.panel_scroll
        for _ in range(20):
            top,bottom = panel.screen_position_y,panel.screen_position_y+panel.computed_height
            y = widget.screen_position_y+widget.computed_height/2
            if top+10 < y < bottom-10:
                break
            _ui._inject_mouse_move((panel.screen_position_x+panel.computed_width/2)*dpi,
                                  ((top+bottom)/2)*dpi)
            await frames(3)
            _ui._inject_mouse_scroll(0.,-2. if y>=bottom-10 else 2.)
            await frames(5)
        assert top<widget.screen_position_y+widget.computed_height/2<bottom,'Control cannot be reached by scrolling'
        _ui._inject_mouse_move((widget.screen_position_x+widget.computed_width/2)*dpi,
                              (widget.screen_position_y+widget.computed_height/2)*dpi)
        await frames(3)
        _ui._inject_mouse_button(0,True);await frames(2)
        _ui._inject_mouse_button(0,False);await frames(4)
    try:
        await lab.start()
        report['display'] = [ui.Workspace.get_main_window_width(),ui.Workspace.get_main_window_height()]
        from nvidia.ov_library_lab.extensions import value
        from nvidia.ov_library_lab.campus_assets import CAMPUS_ENABLED,CAMPUS_TEN_GALLERIES
        assert len(ROOMS)==8+len(value('ROOMS',{})) and FRONT_Z==-10
        if CAMPUS_ENABLED:
            assert CAMPUS_TEN_GALLERIES, 'Use the ten-gallery campus or classic layout for both exhibits'
            from nvidia.ov_library_lab.campus_assets import CAMPUS_VISITOR_POLISH
            assert LOBBY_POSE[:2]==((0.,7.8) if CAMPUS_VISITOR_POLISH else (-3.5,7.8))
        else:
            assert LOBBY_POSE==(0.,10.4,0.,.09)
        for key in ROOMS:
            x,z,_,_=pose(key)
            assert room_at(x,z)==key and contains(x,z,.25),key
        passed(f'{len(ROOMS)} accessible bays; initial position and front-wall distance preserved')
        await frames(10);await capture('01-atrium')
        n,p = lab.gallery.newton_room,lab.gallery.package_room
        await until(lambda:n.worker.ready or n.worker.error)
        assert not n.worker.error,n.worker.error
        lab.go_to('newton');await frames(8)
        for mode in ('gentle','fast'):
            await click(lab.extended_ui[mode])
            assert n.mode==mode
            await click(lab.extended_ui['run'])
            assert n.busy or n.result
            await until(lambda:not n.busy)
            assert not n.error,n.error
            assert n.result['source']=='ovnewton / ovstage / Newton XPBD'
            assert n.result['sealed_ordinal']>400
            n.playing=False
            n.elapsed=1.75 if mode=='fast' else 3.
            n.playing=True;n.advance(.001);n.playing=False
            await frames(8);await capture('02-newton-'+mode)
            labels=n.result['labels']
            trolley=labels.index('/Gantry/Trolley')
            assert abs(n.result['frames'][0][trolley][0]+1.6)<.01
            assert abs(n.result['frames'][-1][trolley][0]-1.6)<.015
            passed(mode+' drive reaches destination with native computed poses')
        assert n.comparisons['gentle']<n.comparisons['fast']
        report['sway_degrees']=n.comparisons.copy()
        passed('Measured sway distinguishes the two native motion profiles')
        n.run();assert n.busy
        n.reset();await frames(80)
        assert n.result is None and not n.playing
        passed('Reset rejects an older in-flight Newton result')
        await until(lambda:p.worker.ready or p.worker.error)
        assert not p.worker.error,p.worker.error
        lab.go_to('package');await frames(8)
        async def operation(action):
            p.submit(action)
            assert p.busy,action
            await until(lambda:not p.busy)
            assert not p.error,p.error
        # Follow the visible guided action. The older publish/install controls
        # are inside collapsed history and are not the visitor's primary flow.
        await click(lab.extended_ui['primary'])
        assert p.busy or p.versions
        await until(lambda:not p.busy)
        assert not p.error,p.error
        assert p.versions==['1.0.0']
        await click(lab.extended_ui['primary'])
        await until(lambda:not p.busy)
        assert not p.error,p.error
        assert p.receipt['verified'] and p.receipt['file_count']>=3
        assert lab.stage.GetPrimAtPath(p.destination+'/Load').GetAttribute('visibility').Get()=='inherited'
        await frames(8);await capture('03-package-logistics')
        passed('ovpackage publishes, installs and verifies USD with bundled texture dependencies')
        await click(lab.extended_ui['inspection'])
        assert p.profile=='inspection'
        await click(lab.extended_ui['primary'])
        await until(lambda:not p.busy)
        assert not p.error and len(p.versions)==2,p.error
        await click(lab.extended_ui['primary'])
        await until(lambda:not p.busy)
        assert not p.error,p.error
        assert p.installed=='1.0.1' and p.receipt['verified']
        assert lab.stage.GetPrimAtPath(p.destination+'/Inspection').GetAttribute('visibility').Get()=='inherited'
        assert lab.stage.GetPrimAtPath(p.destination+'/Load').GetAttribute('visibility').Get()=='invisible'
        await frames(8);await capture('04-package-inspection')
        passed('The installed revision changes the receiving robot from its USD data')
        await operation('edit')
        assert not p.receipt['verified'] and p.receipt['changed']==['robot.usda']
        assert lab.stage.GetPrimAtPath(p.destination+'/Inspection').GetAttribute('visibility').Get()=='invisible'
        await frames(8);await capture('05-package-drift')
        p.select(-1)
        p.submit('install');await until(lambda:not p.busy)
        assert p.error and p.installed=='1.0.1','Installing over an edited file must report a conflict'
        p.select(1)
        await operation('restore')
        assert p.receipt['verified']
        assert lab.stage.GetPrimAtPath(p.destination+'/Inspection').GetAttribute('visibility').Get()=='inherited'
        passed('A visible working-copy edit is detected by file hashes and repaired by native install')
        p.select(-1);await operation('install')
        assert p.installed=='1.0.0' and p.receipt['verified']
        p.reset()
        assert len(p.versions)==2 and p.installed=='1.0.0'
        passed('Earlier revision installs correctly; reset preserves published files')
        report['package_receipt']=p.receipt
        worker_pids=[n.worker.process.process.pid,p.worker.process.process.pid]
        report['worker_pids']=worker_pids
        # The snapshot includes independently reopenable material dependencies.
        from pxr import Usd, Sdf, UsdGeom, UsdShade
        installed=Usd.Stage.Open(p.receipt['asset'])
        texture_paths=[]
        for prim in Usd.PrimRange(installed.GetDefaultPrim(),Usd.TraverseInstanceProxies()):
            for attr in prim.GetAttributes():
                if attr.GetTypeName()!=Sdf.ValueTypeNames.Asset or not attr.Get():continue
                value=attr.Get()
                anchor=attr.GetPropertyStack()[0].layer
                path=Path(value.resolvedPath or value.path)
                if not path.is_absolute():path=(Path(anchor.realPath).parent/path).resolve()
                if '<UDIM>' in path.name:
                    tiles=list(path.parent.glob(path.name.replace('<UDIM>','*')))
                    assert tiles,'Missing bundled UDIM tiles'
                    texture_paths.extend(tiles)
                else:texture_paths.append(path)
        assert texture_paths and all(path.is_file() for path in texture_paths)
        assert all(Path(p.receipt['asset']).parent in path.parents for path in texture_paths),'Package still depends on the source checkout'
        receiving=Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(receiving,'/Placed').GetPrim().GetReferences().AddReference(p.receipt['asset'])
        for prim in receiving.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                assert material and material.GetPath().HasPrefix('/Placed/Looks'),str(prim.GetPath())
        passed('Published USD reopens independently with all relative texture dependencies resolved')
        report['status']='passed'
    except BaseException:
        report.update(status='failed',error=traceback.format_exc())
        if lab.ready:
            await capture('failure')
        raise
    finally:
        await lab.close()
        if lab.gallery:
            for room in (lab.gallery.newton_room,lab.gallery.package_room):
                if room.worker and room.worker.process:
                    assert room.worker.process.poll() is not None,'Orphan library worker'
        (args.output/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        pump.cancel();await asyncio.gather(pump,return_exceptions=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--width',type=int,default=1800)
    parser.add_argument('--height',type=int,default=1150)
    parser.add_argument('--text-scale',type=float,default=1.)
    parser.add_argument('--desktop',action='store_true')
    args=parser.parse_args()
    ui.init('New library room validation',width=args.width,height=args.height,max_fps=60)
    try:
        asyncio.run(run(args))
    finally:
        ui.shutdown()

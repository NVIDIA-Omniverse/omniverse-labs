# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native integration checks. API-driven checks, not mouse/keyboard UI evidence."""
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
import omni.ui as ui
from PIL import Image


async def run_checks(lab):
    # API checks wait for frames. Live cadence has a separate measurement.
    lab.sync_render = True
    out=lab.runtime_root/'native-proof'
    out.mkdir(parents=True,exist_ok=True)
    checks=[]
    def passed(name, **details):
        checks.append({'name':name,'passed':True,**details})
        print('PASS',name,details,flush=True)
        (out/'checks-progress.json').write_text(json.dumps(checks,indent=2))
    async def frames(n=8):
        for _ in range(n):
            lab.update(1/60)
            await ui.next_frame()
    async def capture(name):
        await frames()
        Image.fromarray(lab.viewport.pixels).save(out/(name+'-scene.png'))
        from omni.ui import _ui
        assert _ui._schedule_screenshot(str(out/(name+'-ui.png')))
        await ui.next_frame()
        assert _ui._poll_screenshot_done()
        return lab.viewport.pixels.copy()
    rt,g,w=lab.native,lab.gallery,lab.gallery.workshops
    await frames(35)
    assert lab.viewport.pixels[:,:,:3].std()>15
    await capture('atrium')
    passed('ovrtx rendered atrium', frames=rt.frames)
    assert rt.active_products == {'Main'}
    assert all(tuple(p.GetResolutionAttr().Get()) == (640,360) for n,p in rt.products.items() if n != 'Main')
    physics_source = rt.physics_source()
    from pxr import UsdGeom, UsdPhysics
    expected_colliders = {str(p.GetPath()) for p in lab.stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)}
    actual_colliders = {str(p.GetPath()) for p in physics_source.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)}
    assert expected_colliders == actual_colliders
    for path in expected_colliders | set(g.dynamic_paths):
        assert UsdGeom.Xformable(physics_source.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0) == UsdGeom.Xformable(lab.stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)
    assert not physics_source.GetPrimAtPath('/World/CarDisplay')
    assert physics_source.GetPrimAtPath('/World/Looks/Bounce')
    passed('physics population preserves all colliders and world transforms', colliders=len(actual_colliders), prims=sum(1 for _ in physics_source.Traverse()))
    del physics_source

    lab.go_to('physics')
    from .physics_workshop import configure, nudge
    heights={}
    for low in (False,True):
        configure(lab,'drop')
        lab.set_gravity(low)
        lab.toggle_physics()
        lab.step_physics(18)
        heights[str(low)]=float(rt.pose_reads[g.dynamic_paths[0]][1])
        lab.running=False
    assert heights['True']>heights['False']+.25, heights
    passed('ovphysx gravity changes measured poses',heights=heights)
    lab.set_gravity(False)
    lab.toggle_physics()
    lab.step_physics(180)
    resting=[float(v[1]) for v in rt.pose_reads.values()]
    assert len(resting)==6 and all(1.15<y<1.27 for y in resting),resting
    lab.running=False
    await capture('physics')
    passed('ovphysx contacts support six cartons',heights=resting)
    configure(lab,'stack')
    lab.toggle_physics()
    lab.step_physics(180)
    assert max(float(v[1]) for v in rt.pose_reads.values())>2.25
    nudge(lab)
    lab.step_physics(20)
    lab.running=False
    passed('ovphysx stack and write-session impulse',steps=rt.physics_steps)
    slides={}
    for friction in (.05,.65):
        configure(lab,'slide',friction=friction)
        lab.toggle_physics()
        initial=rt.pose_reads[g.dynamic_paths[0]].copy()
        lab.step_physics(30)
        slides[str(friction)]=float(np.linalg.norm(rt.pose_reads[g.dynamic_paths[0]][[0,2]]-initial[[0,2]]))
    assert slides['0.05']>slides['0.65']+.03,slides
    lab.reset_physics()
    passed('ovphysx friction affects stopping distance',distances=slides)

    lab.go_to('rendering')
    assert w.render_subject=='lidar'
    await capture('rtx-default-robot')
    lab.set_render_subject('robots')
    await capture('rtx-robots')
    lab.set_render_subject('car')
    car=await capture('rtx-car')
    lab.set_roughness(.9)
    w.car.paint('Ocean')
    w.car.set_clearcoat(.1)
    w.set_light(angle=70,size=.5)
    changed=await capture('rtx-materials')
    difference=float(np.abs(car.astype(float)-changed).mean())
    assert difference>1,difference
    passed('ovrtx material and lighting controls change rendered pixels',mean_difference=difference)
    lab.panel_modes['rendering']='camera'
    lab.request_panel()
    await frames()
    from .panels import snapshot
    file=await snapshot(lab)
    assert Image.open(file).size==(640,360)
    passed('ovrtx inspection-camera PNG',file=str(file))

    lab.go_to('scene')
    w.set_fleet(6)
    before=rt.read(['/World/Shared/AMR'],'omni:xform')
    lab.move_shared()
    await frames(90)
    after=rt.read(['/World/Shared/AMR'],'omni:xform')
    assert not np.allclose(before['/World/Shared/AMR'],after['/World/Shared/AMR'])
    lab.recolor_shared()
    for robot in w.fleet:
        path=str(robot.GetPath())
        root=rt.ovstage.instancing.get_prototype_root(rt.stage,path) if robot.GetPrim().IsInstance() else path
        cargo=rt.read([root+'/Load'],'visibility')
        assert rt.paths.token_to_string(int(cargo[root+'/Load'][0]))=='invisible',path
    await capture('stage')
    assert rt.active_products == {'Main', 'Fleet'}
    assert all(0 < n <= 2 for n in rt.last_frame_counts.values()), rt.last_frame_counts
    passed('ovstage tensor writes and reads drive shared scene',ordinal=rt.ordinal,writes=rt.writes)

    from .extensions import value
    optional_check = value('native_checks')
    optional_report = await optional_check(lab, frames, capture, passed) if optional_check else {}

    lab.go_to('storage')
    from .native_storage import RackStorage
    original=lab.storage
    original_path=g.extra.snapshot_path
    with tempfile.TemporaryDirectory(prefix='ovlab-native-storage-') as folder:
        g.extra.snapshot_path=Path(folder)/'saved-asset.usda'
        storage=RackStorage(g.extra)
        await storage.open()
        g.extra.set_asset_variant(2)
        await storage.save()
        g.extra.set_asset_variant(0)
        await storage.save()
        await storage.select(-1)
        await storage.restore()
        assert g.extra.storage_variant==2 and len(g.extra.revisions)==2
        await capture('storage')
        passed('ovstorage write/list/read/restore',operations=storage.operations)
    g.extra.snapshot_path=original_path
    await original.refresh()

    lab.go_to('interface')
    for value in (0.,.5,1.):
        lab.lift_model.set_value(value)
        await capture('ui-lift-'+str(round(value*100)))
    lab.lift_model.set_value(.8)
    await frames()
    assert abs(g.extra.lift_value-.8)<.001
    lift=rt.read(['/World/Interface/Lift'],'omni:xform')
    assert abs(float(lift['/World/Interface/Lift'].reshape(4,4)[3,1])-(g.extra.lift_table.load_base[1]+1.51*.8 if g.factory.simready.assets else 2.43))<.001
    await capture('ui')
    passed('standalone ovui value model drives ovstage scene transform')

    lab.go_to('streaming')
    await frames(4)  # The room reuses the connection prepared at startup.
    assert lab.stream_view.transport is not None,lab.stream_view.error
    lab.stream_view.send('dispatch')
    sources=set()
    delivered=False
    for _ in range(30):
        pixels=rt.render('Stream')
        sources.add(hashlib.sha256(pixels.tobytes()).hexdigest())
        received=lab.stream_view.transport.send(pixels)
        if received is not None:
            delivered |= hashlib.sha256(received.tobytes()).hexdigest() in sources
        await frames(1)
    assert delivered
    assert lab.stream_view.command_count>=1
    await capture('stream')
    assert rt.active_products == {'Main', 'Stream'}
    assert all(0 < n <= 2 for n in rt.last_frame_counts.values()), rt.last_frame_counts
    passed('ovstream delivers exact RTX pixels and return commands',frames=lab.stream_view.transport.received,commands=lab.stream_view.command_count)

    lab.go_to('lobby')
    await capture('atrium-final')
    assert rt.active_products == {'Main'}
    assert rt.max_frames_per_step <= 2, rt.max_frames_per_step
    assert rt.resident_products == {'Main', 'Preview'}
    passed('resident cameras have no frame backlog; hidden previews are not copied',max_frames_per_step=rt.max_frames_per_step)
    kit=[name for name in sys.modules if name=='omni.kit' or name.startswith('omni.kit.') or name=='omni.usd' or name=='omni.physx']
    assert not kit,kit
    packages={name:importlib.metadata.version(name) for name in ('ovrtx','ovphysx','ovstage','ovui','ovstorage','ovstream','usd-core')}
    report={'checks':checks,'packages':packages,**optional_report,'kit_modules':kit,'validation':'Programmatic native API and rendered-frame checks; UI interactions validated separately.'}
    (out/'native-validation.json').write_text(json.dumps(report,indent=2))
    print('NATIVE VALIDATION COMPLETE',flush=True)

"""CPU proof of evaluated animation requests; diagnostic pixels are not native."""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import bpy
from pxr import Usd, UsdGeom

source=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(source/'addon'),str(source/'tests')]
import ovrtx_blender_example as addon
from ovrtx_blender_example import engine,scene_generation_sessions
from blender_render_output_fixture import create_scene

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
tempfile.tempdir=str(out/'temp');Path(tempfile.tempdir).mkdir()
scene,cube,camera=create_scene(addon)
scene.frame_set(3)  # Animation starts at1; discard render_init's frame3 request.
records=[]

def diagnostic_render(self,depsgraph):
    request,_=engine._consume_final_render_request(depsgraph.scene)
    prebuilt=request is not None
    if request is None:
        request=engine._render_callback_adapter('cpu-final-frame').final_render(depsgraph)
    owner=scene_generation_sessions._owners[int(scene.session_uid)]
    generation=owner.reuse()
    stage=Usd.Stage.Open(request.input_usd_path)
    stage.Flatten().Export(str(out/f'frame-{depsgraph.scene.frame_current}.usda'))
    prim=stage.GetPrimAtPath('/World/Animated_orange_subject')
    exported=UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractTranslation()[0]
    evaluated=depsgraph.objects.get(cube.name).matrix_world.translation.x
    assert exported==evaluated,(depsgraph.scene.frame_current,exported,evaluated)
    records.append(dict(frame=depsgraph.scene.frame_current,evaluated_x=evaluated,
        exported_x=exported,generation=generation.number,prebuilt=prebuilt,
        pending=generation is owner.pending_generation,
        retained_replay_counts=list(map(len,owner.retained_values_for(generation)))))
    if owner.pending_generation is not None:
        owner.accept(generation)  # Model ACK; no native runtime runs in this CPU fixture.
    result=self.begin_result(0,0,321,241)
    result.layers[0].passes['Combined'].rect=[(.2,.3,.4,1)]*(321*241)
    self.end_result(result)

engine.OvrtxExampleRenderEngine.render=diagnostic_render
scene.render.image_settings.file_format='PNG'
scene.render.filepath=str(out/'diagnostic_')
bpy.ops.render.render(animation=True)
assert [row['frame'] for row in records]==[1,2,3]
assert [row['exported_x'] for row in records]==[-2,0,2]
assert len({row['generation'] for row in records})==3
assert all(row['retained_replay_counts']==[0,0,0] for row in records)
forbidden=[name for name in ('ovrtx','ovphysx','ovstage','ovstream','warp') if name in sys.modules]
assert not forbidden,forbidden
report=dict(status='passed',records=records,sdk_modules=forbidden,
    scope='Actual stock Blender Render Animation callback and USD capture; substitute diagnostic RenderResult, simulated ACK; native pixels qualified separately',
    blender=bpy.app.version_string,
    source_sha256={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (source/'addon').rglob('*.py')})
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
print('FINAL_FRAME_REQUESTS_PASS',json.dumps(records))

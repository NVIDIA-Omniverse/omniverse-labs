# SPDX-License-Identifier: Apache-2.0
"""Stock Blender RNA -> translator -> client admission; no worker/render launch."""
from pathlib import Path
import json
import sys
import traceback

import bpy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'addon'))
from ovrtx_blender_example import properties
from ovrtx_blender_example.blender_signals import BlenderRenderSignal,BlenderRenderSignalSource,BlenderRenderIntent
from ovrtx_blender_example.blender_signal_translation import RenderRequestTranslator
from ovrtx_blender_example.direct_sdk_client import PublicSdkRenderClient,RenderClientError

out=Path(sys.argv[sys.argv.index('--')+1])
report={'status':'running','blender_version':bpy.app.version_string,'cases':[]}
properties.register()
scene=bpy.context.scene
client=PublicSdkRenderClient.__new__(PublicSdkRenderClient)


def translate(intent):
    return RenderRequestTranslator().translate(BlenderRenderSignal(
        BlenderRenderSignalSource.VIEW_UPDATE,intent,scene,'',context=bpy.context,
        camera_prim_path='/World/Camera',render_product_path=properties.DEFAULT_RENDER_PRODUCT_PATH,
        current_scene_generation=True))


try:
    for intent in (BlenderRenderIntent.VIEWPORT,BlenderRenderIntent.FINAL_RENDER):
        request=translate(intent)
        assert request.native_exposure.enabled is False
        assert request.native_exposure.white_point_scale==10.0
        client.configure_session(request)
        assert client.request is request
        report['cases'].append({'intent':str(intent),'default_admitted':True,
            'native_exposure':request.native_exposure.diagnostics()})
    accepted=client.request
    for fields in ({'native_auto_exposure':True},
                   {'native_auto_exposure':False,'native_exposure_white_point_scale':2.0}):
        for key,value in fields.items():setattr(scene.ovrtx_example,key,value)
        try:client.configure_session(translate(BlenderRenderIntent.VIEWPORT))
        except RenderClientError as error:
            assert 'not admitted' in str(error)
        else:raise AssertionError('Explicit unadmitted scene setting accepted')
        assert client.request is accepted
    scene.ovrtx_example.native_auto_exposure=False
    scene.ovrtx_example.native_exposure_white_point_scale=10
    scene.camera.data.ovrtx_exposure.use_scene_defaults=False
    scene.camera.data.ovrtx_exposure.native_auto_exposure=True
    try:client.configure_session(translate(BlenderRenderIntent.FINAL_RENDER))
    except RenderClientError as error:assert 'not admitted' in str(error)
    else:raise AssertionError('Explicit unadmitted camera setting accepted')
    assert client.request is accepted
    report['sdk_modules']=[name for name in ('ovrtx','ovphysx','ovstage','ovstream','warp') if name in sys.modules]
    assert not report['sdk_modules']
    report['status']='passed'
except Exception:
    report['status']='failed';report['error']=traceback.format_exc()
finally:
    properties.unregister()
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print('RENDER_ADMISSION_DEFAULTS',json.dumps(report),flush=True)
if report['status']!='passed':raise RuntimeError(report['error'])

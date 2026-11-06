from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as N
import pytest
from ovrtx_blender_example.native_exposure import NativeExposure, from_scene, ENABLED_ATTRIBUTE, TARGET_ATTRIBUTE
from ovrtx_blender_example.render_requests import RenderRequest
from ovrtx_blender_example import ovrtx_session, ovrtx_session_controller as controllers
from test_ovrtx_session_controller import _Client, _factory


def test_saved_scene_defaults_and_camera_override_round_trip():
    scene=N(ovrtx_example={'native_auto_exposure':1,'native_exposure_white_point_scale':2.0},camera=None)
    assert from_scene(scene)==NativeExposure(True,2.0,'scene')
    override=N(use_scene_defaults=False,native_auto_exposure=True,native_exposure_white_point_scale=4.0)
    scene.camera=N(data=N(ovrtx_exposure=override))
    assert from_scene(scene)==NativeExposure(True,4.0,'camera')
    override.use_scene_defaults=True
    assert from_scene(scene)==NativeExposure(True,2.0,'scene')
    scene.camera=N(data=N())
    assert from_scene(scene)==NativeExposure(True,2.0,'scene')


@pytest.mark.parametrize('value',[0,-1,float('nan'),float('inf')])
def test_invalid_target_fails_closed(value):
    with pytest.raises(ValueError):NativeExposure(True,value)


def request(tmp_path,exposure,*,generated=True):
    source=tmp_path/'source.usda'
    source.write_text('#usda 1.0\ndef Xform "World" { def Camera "Camera" {} }\n')
    return RenderRequest(input_usd_path=str(source),current_scene_generation=generated,
        camera_prim_path='/World/Camera',sensor_paths=('/Render/Product',),selected_sensor_paths=('/Render/Product',),
        native_exposure=exposure)


@pytest.mark.parametrize('generated',[True,False])
def test_native_product_attrs_and_fingerprint_follow_effective_exposure(tmp_path,generated):
    from pxr import Usd
    base=request(tmp_path,NativeExposure(True,2.0,'scene'),generated=generated)
    states=[base,replace(base,native_exposure=NativeExposure(True,4.0,'camera')),
            replace(base,native_exposure=NativeExposure(False,2.0,'camera'))]
    specs=[ovrtx_session.build_spec(row) for row in states]
    assert len({spec.ovrtx_scene_composition.digest for spec in specs})==3
    for req,spec in zip(states,specs):
        stage=Usd.Stage.Open(spec.ovrtx_scene_composition.composed_scene_path)
        product=stage.GetPrimAtPath('/Render/Product')
        assert product.GetAttribute(ENABLED_ATTRIBUTE).Get()==req.native_exposure.enabled
        assert product.GetAttribute(TARGET_ATTRIBUTE).Get()==req.native_exposure.white_point_scale
        assert product.GetRelationship('camera').GetTargets()==[stage.GetPrimAtPath('/World/Camera').GetPath()]
        camera=stage.GetPrimAtPath('/World/Camera')
        assert camera.GetAttribute(ENABLED_ATTRIBUTE).Get()==req.native_exposure.enabled
        assert camera.GetAttribute(TARGET_ATTRIBUTE).Get()==req.native_exposure.white_point_scale
    # Provenance changes with identical effective settings do not re-key exposure.
    same=ovrtx_session.build_spec(replace(base,native_exposure=NativeExposure(True,2.0,'camera')))
    assert same.ovrtx_scene_composition.digest==specs[0].ovrtx_scene_composition.digest


def test_exposure_2_4_2_reuses_supported_native_stage(tmp_path,monkeypatch):
    class Client(_Client):
        def __init__(self):super().__init__('native-stage');self.updated=[]
        def can_update_scene(self,current,desired):return True
        def update_scene(self,simulation_id,desired):
            assert simulation_id=='native-stage';self.updated.append(desired)
    client=Client();_factory(monkeypatch,[client]);controller=controllers.OvrtxSessionController()
    base=request(tmp_path,NativeExposure(True,2.0,'scene'))
    try:
        assert controller.ensure(base).session_started
        for scale in (4.0,2.0):
            result=controller.ensure(replace(base,native_exposure=NativeExposure(True,scale,'camera')))
            assert not result.session_started and controller._reuse['reason']=='stage_updated'
        assert client.starts==1 and client.deletes==0 and len(client.updated)==2
        assert [row['event'] for row in controller._events]==['created','stage_updated','stage_updated']
    finally:controller.shutdown()


def test_translator_resolves_camera_then_scene_for_both_render_intents():
    from test_rtpt_render_product_authoring import _render_scene
    from ovrtx_blender_example.blender_signal_translation import RenderRequestTranslator
    from ovrtx_blender_example.blender_signals import BlenderRenderSignal, BlenderRenderSignalSource, BlenderRenderIntent
    scene=_render_scene(native_auto_exposure=True,native_exposure_white_point_scale=2.0)
    override=N(use_scene_defaults=False,native_auto_exposure=True,native_exposure_white_point_scale=4.0)
    scene.camera=N(data=N(ovrtx_exposure=override))
    for intent in (BlenderRenderIntent.VIEWPORT,BlenderRenderIntent.FINAL_RENDER):
        signal=BlenderRenderSignal(BlenderRenderSignalSource.VIEW_UPDATE,intent,scene,'',
            camera_prim_path='/Generated/Camera',render_product_path='/Generated/Product',
            context=N(),current_scene_generation=True)
        assert RenderRequestTranslator().translate(signal).native_exposure==NativeExposure(True,4.0,'camera')
        override.use_scene_defaults=True
        assert RenderRequestTranslator().translate(signal).native_exposure==NativeExposure(True,2.0,'scene')
        override.use_scene_defaults=False


def test_scene_camera_rna_defaults_and_pointer_registration_order(monkeypatch):
    import importlib.util,sys,types
    events=[]
    bpy=types.ModuleType('bpy');props=types.ModuleType('bpy.props')
    for name in ('BoolProperty','EnumProperty','FloatProperty','IntProperty','PointerProperty','StringProperty'):
        setattr(props,name,lambda **kwargs:kwargs)
    bpy.types=N(PropertyGroup=type('PropertyGroup',(),{}),Scene=type('Scene',(),{}),Camera=type('Camera',(),{}))
    def register(cls):cls.is_registered=True;events.append(('register',cls.__name__))
    def unregister(cls):
        assert not hasattr(bpy.types.Camera,'ovrtx_exposure') and not hasattr(bpy.types.Scene,'ovrtx_example')
        cls.is_registered=False;events.append(('unregister',cls.__name__))
    bpy.utils=N(register_class=register,unregister_class=unregister)
    monkeypatch.setitem(sys.modules,'bpy',bpy);monkeypatch.setitem(sys.modules,'bpy.props',props)
    path=Path(__file__).parents[1]/'addon/ovrtx_blender_example/properties.py'
    spec=importlib.util.spec_from_file_location('ovrtx_blender_example.exposure_property_fixture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for cls in (module.OvrtxExampleRenderSettings,module.OvrtxCameraExposureSettings):
        assert cls.__annotations__['native_auto_exposure']['default'] is False
        assert cls.__annotations__['native_exposure_white_point_scale']['default']==10.0
    assert module.OvrtxCameraExposureSettings.__annotations__['use_scene_defaults']['default'] is True
    module.register()
    assert bpy.types.Camera.ovrtx_exposure['type'] is module.OvrtxCameraExposureSettings
    assert module.OvrtxCameraExposureSettings.is_registered
    module.unregister()
    assert [event[0] for event in events]==['register','register','unregister','unregister']

"""Shared production PublicSdkRenderClient value recovery fixture."""
from types import SimpleNamespace
from copy import deepcopy
from test_direct_sdk_client import FakeRuntime
from ovrtx_blender_example import direct_sdk_client
from ovrtx_blender_example.ovrtx_runtime_client import RenderClientError

class FaultRuntime(FakeRuntime):
    def __init__(self,path):
        super().__init__(path)
        self.remote_fault=''
        self.write_attempts=0
        self.accepted={}
        self.shutdown_called=False
    def request(self,op,params=None,**kwargs):
        params=params or {}
        if op=='update':
            self.write_attempts+=1
            if self.remote_fault=='prefix':
                self.calls.append((op,deepcopy(params)))
                return {'_runtime_epoch':self.epoch,'_runtime_pid':self.pid,'updated_count':1}
            if self.remote_fault=='ambiguous':
                raise RenderClientError('injected ambiguous update transport failure')
            self.accepted.update({(v['prim_path'],v['attribute_name']):deepcopy(v['value']) for v in params['values']})
        return super().request(op,params,**kwargs)
    def recover(self):
        super().recover()
        self.accepted.clear()
    def close(self):
        self.shutdown_called=True
        super().close()

def fault_client(tmp_path,monkeypatch):
    runtime=FaultRuntime(tmp_path)
    monkeypatch.setattr(direct_sdk_client,'_session',lambda kind:runtime)
    client=direct_sdk_client.PublicSdkRenderClient()
    monkeypatch.setattr(client,'_snapshot_source',lambda path,**kw:str(path))
    monkeypatch.setattr(client,'_snapshot_inline',lambda text:('fixture-inline',text))
    spec=SimpleNamespace(ovrtx_scene_composition=SimpleNamespace(composed_scene_path='/scene.usda'),stage_metadata={},sensor_paths=('/Render/Product',),width=1,height=2,render_var='LdrColor')
    identity=client.start_session(spec)
    return client,runtime,identity


#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only native OVStage proof of schema-preserving renderer updates."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import ovstage as ovs
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from public_sdk_backend import RendererBackend
stage = ovs.Stage('renderer-schema-proof')
try:
    ovs.population.open_usd_from_string(stage, '#usda 1.0\ndef Xform "World" {\n def Xform "Light" {\n }\n}\n', ordinal=1)
    with ovs.PathDictionary(stage) as paths:
        with paths.create_path_list_from_strings(['/World/Light']) as plist:
            with stage.query_from_path_list(plist) as query:
                stage.write_attribute(query, 'inputs:color', 2, np.asarray([[1.,1.,1.]], dtype=np.float32), semantic=ovs.AttributeSemantic.VECTOR, is_array=False).wait()
                stage.write_attribute(query, 'inputs:width3', 2, np.asarray([[1.,2.,3.]], dtype=np.float32), is_array=False).wait()
    stage.advance_write_floor(2).wait()
    backend = RendererBackend.__new__(RendererBackend)
    backend.np, backend.ovstage, backend.stage, backend.ordinal = np, ovs, stage, 2
    backend.renderer = SimpleNamespace(update_from_stage=lambda ordinal: None, reset=lambda **kw: None)
    backend.desired_values, backend.simulation_time_ns = {}, 0
    def read(name):
        rows=[]
        with ovs.PathDictionary(stage) as paths:
            with paths.create_path_list_from_strings(['/World/Light']) as plist:
                with stage.query_from_path_list(plist) as query:
                    with stage.read_attributes(query, [paths.intern_token(name)], ovs.OrdinalRange.latest(backend.ordinal)) as result:
                        result.wait()
                        for group in result.groups():
                            try:
                                if not group.is_delete:
                                    rows.append((int(group.raw.semantic),np.array(group.array(0)).tolist()))
                            finally:
                                stage.release_group(group)
        return rows
    def color(v):
        return dict(prim_path='/World/Light', attribute_name='inputs:color', value_type='Color3f', value=v)
    backend.update([color([.1,.2,.8])])
    before = read('inputs:color')
    assert before[0][0] == int(ovs.AttributeSemantic.VECTOR), before
    assert np.allclose(before[0][1], [.1,.2,.8]), before
    ordinal = backend.ordinal
    try:
        backend.update([color([1.,0.,0.]), dict(prim_path='/World/Light', attribute_name='inputs:width3', value_type='Float2', value=[0,0])])
    except ValueError as error:
        assert 'tuple width' in str(error), error
    else:
        raise AssertionError('Incompatible native schema was accepted')
    assert backend.ordinal == ordinal
    assert read('inputs:color') == before, 'Valid prefix changed before incompatible suffix rejection'
    try:
        backend.update([color([1.,0.,0.]), dict(prim_path='/World/Missing', attribute_name='inputs:new', value_type='Float', value=1.)])
    except ValueError as error:
        assert 'existing typed USD prim' in str(error), error
    else:
        raise AssertionError('Missing prim was silently acknowledged')
    assert backend.ordinal == ordinal and read('inputs:color') == before
    print({'status':'passed','preserved_semantic':'VECTOR','color':before,'batch_rejected_before_mutation':True,'missing_target_rejected':True})
finally:
    stage.destroy()

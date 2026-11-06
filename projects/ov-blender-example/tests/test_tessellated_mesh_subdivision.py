# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Current-scene export preserves the retained add-on's mesh shading policy."""
from hashlib import sha256
import json
import pytest
from pxr import Usd,UsdGeom
from ovrtx_blender_example import scene_generation as generation


@pytest.mark.parametrize('original_scheme',[None,'none','catmullClark'])
def test_normalization_preserves_baseline_fallback_and_polygon_corner_data(tmp_path,original_scheme):
    path=tmp_path/'panel.usda'
    stage=Usd.Stage.CreateNew(str(path))
    mesh=UsdGeom.Mesh.Define(stage,'/World/Panel')
    mesh.CreatePointsAttr([(0,0,0),(1,0,0),(1,1,0),(0,1,0)])
    mesh.CreateFaceVertexCountsAttr([3,3])
    mesh.CreateFaceVertexIndicesAttr([0,1,2,0,2,3])
    mesh.CreateNormalsAttr([(0,0,1)]*3+[(0,1,0)]*3)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.CreateCreaseIndicesAttr([0,1]);mesh.CreateCreaseLengthsAttr([2]);mesh.CreateCreaseSharpnessesAttr([1])
    if original_scheme is not None:mesh.CreateSubdivisionSchemeAttr(original_scheme)
    preserved_names=('points','faceVertexCounts','faceVertexIndices','normals')
    preserved={name:mesh.GetPrim().GetAttribute(name).Get() for name in preserved_names}
    stage.GetRootLayer().Save()
    generation._normalize_stock_export_geometry(path)
    # Keep a stage reference: pxr prim/schema handles do not own their stage.
    result=Usd.Stage.Open(str(path));reopened=UsdGeom.Mesh(result.GetPrimAtPath('/World/Panel'))
    assert not reopened.GetSubdivisionSchemeAttr().HasAuthoredValueOpinion()
    assert reopened.GetSubdivisionSchemeAttr().Get()==UsdGeom.Tokens.catmullClark
    assert reopened.GetNormalsInterpolation()==UsdGeom.Tokens.faceVarying
    assert {name:reopened.GetPrim().GetAttribute(name).Get() for name in preserved_names}==preserved
    assert all(not reopened.GetPrim().GetAttribute(name).HasAuthoredValueOpinion()
               for name in ('creaseIndices','creaseLengths','creaseSharpnesses'))


@pytest.mark.parametrize('old_contract', ['blender-usd-v3-retained-visibility-shadow',
    'blender-usd-v4-explicit-tessellated-polygons', 'blender-usd-v5-physical-widths-owned-particle-hair'])
def test_cache_from_previous_subdivision_contract_is_rejected(tmp_path,old_contract):
    manifest=json.dumps({'schema':generation._REUSABLE_BASE_SCHEMA,
        'export_contract':old_contract,'lookup_key':'saved-source'}).encode()
    (tmp_path/'manifest.json').write_bytes(manifest)
    (tmp_path/'manifest.sha256').write_text(sha256(manifest).hexdigest())
    cache,reason=generation._load_reusable_base(object(),tmp_path,'saved-source')
    assert cache is None and 'incompatible_manifest' in reason

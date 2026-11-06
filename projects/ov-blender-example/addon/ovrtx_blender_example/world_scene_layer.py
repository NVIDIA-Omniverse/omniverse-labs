# SPDX-License-Identifier: Apache-2.0
"""Canonical World dome opinions: textured, discrete mixed HDR and flat-colour skies."""
from .world_dome_conversion import (DEFAULT_DOME_OWNER_PATH, DOME_DIFFUSE_SCALE, environment_texture_dome,
                                    environment_rotation_matrix, world_dome_spec)
from .usd_opinion_records import AddOnUsdOpinionRecord
from .value_edit_conversion import STATUS_SUPPORTED


def opinion_records(scene, stock_usd_path, *, stock_stage_provider=None):
    world = getattr(scene, 'world', None)
    description = environment_texture_dome(world)
    if description is None:
        return _flat_world_records(world, stock_usd_path, stock_stage_provider)
    from pxr import Gf, Sdf, Usd, UsdLux, UsdGeom
    layer = Sdf.Layer.CreateAnonymous('.usda')
    stage = Usd.Stage.Open(layer)
    dome = UsdLux.DomeLight.Define(stage, DEFAULT_DOME_OWNER_PATH)
    dome.CreateTextureFileAttr().Set(Sdf.AssetPath(description['texture_file']))
    dome.CreateTextureFormatAttr().Set('latlong')
    UsdGeom.Xformable(dome.GetPrim()).AddTransformOp().Set(Gf.Matrix4d(*sum(environment_rotation_matrix(description.get('rotation_z', 0.0)),())))
    dome.CreateIntensityAttr().Set(description['intensity'])
    dome.CreateColorAttr().Set(description['color'])
    dome.CreateDiffuseAttr().Set(DOME_DIFFUSE_SCALE)
    records = [AddOnUsdOpinionRecord(DEFAULT_DOME_OWNER_PATH, layer,
               diagnostics={'source': 'world_environment'})]
    stock = stock_stage_provider() if stock_stage_provider is not None else Usd.Stage.Open(str(stock_usd_path))
    for prim in stock.Traverse() if stock else ():
        path = str(prim.GetPath())
        if prim.GetTypeName() != 'DomeLight' or path == DEFAULT_DOME_OWNER_PATH:
            continue
        disabled = Sdf.Layer.CreateAnonymous('.usda')
        Sdf.CreatePrimInLayer(disabled, path).active = False
        records.append(AddOnUsdOpinionRecord(path, disabled,
                       diagnostics={'source': 'world_environment'}))
    return tuple(records)


def _flat_world_records(world, stock_usd_path, stock_stage_provider):
    """Convert the stock flat-colour dome with the live value-edit policy.

    Blender exports a flat World as a solid-colour texture with
    ``intensity = strength``. Without this opinion the first frame is lit
    in Blender units, about a thousand times too dark, until a live edit
    applies the conversion. The texture is cleared so the converted colour
    is not applied twice.
    """
    if world is None:
        return ()
    spec = world_dome_spec(world)
    if spec.status != STATUS_SUPPORTED:
        return ()
    from pxr import Gf, Sdf, Usd, UsdLux
    stock = stock_stage_provider() if stock_stage_provider is not None else Usd.Stage.Open(str(stock_usd_path))
    domes = [prim for prim in (stock.Traverse() if stock else ()) if prim.GetTypeName() == 'DomeLight']
    if len(domes) != 1:
        return ()
    path = str(domes[0].GetPath())
    layer = Sdf.Layer.CreateAnonymous('.usda')
    stage = Usd.Stage.Open(layer)
    dome = UsdLux.DomeLight(stage.OverridePrim(path))
    dome.CreateTextureFileAttr().Set(Sdf.AssetPath(''))
    dome.CreateIntensityAttr().Set(spec.intensity)
    dome.CreateColorAttr().Set(Gf.Vec3f(*spec.color))
    dome.CreateDiffuseAttr().Set(DOME_DIFFUSE_SCALE)
    return (AddOnUsdOpinionRecord(path, layer, diagnostics={'source': 'world_flat_color'}),)

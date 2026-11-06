# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bundle the visible robot and apply values read from the installed USD."""
import base64
import hashlib
from pathlib import Path
from pxr import Sdf, Usd, UsdGeom


def snapshot(stage, robot_path):
    layer = Sdf.Layer.CreateAnonymous('robot.usda')
    Sdf.CopySpec(stage.GetRootLayer(), Sdf.Path(robot_path), layer, Sdf.Path('/Robot'))
    asset = Usd.Stage.Open(layer)
    asset.SetDefaultPrim(asset.GetPrimAtPath('/Robot'))
    UsdGeom.SetStageUpAxis(asset, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(asset, 1.)
    materials = set()
    for prim in Usd.PrimRange(asset.GetPrimAtPath('/Robot')):
        for rel in prim.GetRelationships():
            if rel.GetName().startswith('material:binding'):
                materials.update(rel.GetTargets())
    # Material networks belong inside the default prim so an external USD
    # reference includes them. Define ancestors, rather than undefined overs.
    asset.DefinePrim('/Robot/Looks','Scope')
    materials={path for path in materials if not path.HasPrefix('/Robot')}
    remapped = {path:Sdf.Path('/Robot/Looks').AppendChild(path.name) for path in materials}
    for prim in Usd.PrimRange(asset.GetPrimAtPath('/Robot')):
        for rel in prim.GetRelationships():
            if rel.GetName().startswith('material:binding'):
                rel.SetTargets([remapped.get(path,path) for path in rel.GetTargets()])
    for path in materials:
        Sdf.CopySpec(stage.GetRootLayer(), path, layer, remapped[path])
    return layer.ExportToString()


def bundle(stage, robot_path):
    return bundle_snapshot(snapshot(stage,robot_path))


def bundle_snapshot(text):
    """Collect referenced USD and tiled textures without expanding instanced meshes."""
    layer=Sdf.Layer.CreateAnonymous('robot.usda')
    if not layer.ImportFromString(text):raise ValueError('Invalid robot snapshot')
    files={}
    assets={}
    def copy_asset(source):
        source=Path(source).resolve()
        if source in assets:return assets[source]
        if '<UDIM>' in source.name:
            tiles=sorted(source.parent.glob(source.name.replace('<UDIM>','*')))
            if not tiles:raise RuntimeError('Missing tiled texture '+source.name)
            prefix=hashlib.sha256(str(source).encode()).hexdigest()[:16]
            dest='textures/'+prefix+'.<UDIM>.png'
            assets[source]=dest
            for tile in tiles:
                name=dest.replace('<UDIM>',tile.stem.split('.')[-1])
                files[name]=base64.b64encode(tile.read_bytes()).decode('ascii')
            return dest
        if not source.is_file():raise RuntimeError('Missing package dependency '+source.name)
        if source.suffix in ('.usd','.usda','.usdc'):
            dest='assets/'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+'.usda'
            assets[source]=dest
            original=Sdf.Layer.FindOrOpen(str(source))
            # Transfer composed file content without flattening all instances.
            clone=Sdf.Layer.CreateAnonymous();clone.TransferContent(original)
            rewrite(clone,source.parent,'../')
            files[dest]=base64.b64encode(clone.ExportToString().encode()).decode('ascii')
        else:
            dest='textures/'+hashlib.sha256(source.read_bytes()).hexdigest()[:20]+source.suffix
            assets[source]=dest
            files[dest]=base64.b64encode(source.read_bytes()).decode('ascii')
        return dest
    def rewrite(current,anchor,prefix=''):
        specs=[]
        current.Traverse(Sdf.Path.absoluteRootPath,lambda p:specs.append(current.GetObjectAtPath(p)))
        for spec in specs:
            if isinstance(spec,Sdf.PrimSpec):
                for name in ('referenceList','payloadList'):
                    editor=getattr(spec,name)
                    for kind in ('prependedItems','appendedItems','explicitItems'):
                        items=list(getattr(editor,kind))
                        if not items:continue
                        updated=[]
                        for item in items:
                            if not item.assetPath:updated.append(item);continue
                            path=Path(item.assetPath)
                            if not path.is_absolute():path=anchor/path
                            cls=Sdf.Reference if name=='referenceList' else Sdf.Payload
                            updated.append(cls(prefix+copy_asset(path),item.primPath,item.layerOffset))
                        setattr(editor,kind,updated)
            elif isinstance(spec,Sdf.AttributeSpec) and spec.typeName==Sdf.ValueTypeNames.Asset:
                value=spec.default
                if value and value.path:
                    source=Path(value.resolvedPath or value.path)
                    if not source.is_absolute():source=anchor/source
                    spec.default=Sdf.AssetPath(prefix+copy_asset(source))
    rewrite(layer,Path.cwd())
    files['robot.usda']=base64.b64encode(layer.ExportToString().encode()).decode('ascii')
    files['README.txt']=base64.b64encode(b'Omniverse Libraries Playground cargo robot. Metres, Y up. Scripted travel.\nUSD, PBR materials and texture dependencies travel together. See asset attribution.\n').decode('ascii')
    data=Path(__file__).resolve().parents[3]/'data'
    for directory,label in ((data/'surfaces','surfaces'),(data/'models/simready','simready')):
        for filename in ('ATTRIBUTION.md','CC0-1.0.txt','CC-BY-4.0.txt'):
            notice=directory/filename
            if notice.is_file():files[f'licenses/{label}-{filename}']=base64.b64encode(notice.read_bytes()).decode('ascii')
    return files


def apply_installed(stage, destination_root, filename):
    """Apply the installed asset's data to matching, already-populated geometry.

    Both profiles share one topology. Copying its authored transform/visibility
    channels avoids rebuilding the renderer, while the file is authoritative.
    """
    layer = Sdf.Layer.FindOrOpen(str(filename))
    if not layer:
        raise RuntimeError('The installed USD could not be read')
    layer.Reload(force=True)
    source = Usd.Stage.Open(layer)
    root = source.GetPrimAtPath('/Robot')
    if not root:
        raise RuntimeError('The package contains no /Robot asset')
    changes = 0
    for prim in Usd.PrimRange(root):
        if prim.GetPath().HasPrefix('/Robot/Looks'):
            continue
        relative = str(prim.GetPath())[len('/Robot'):]
        destination = stage.GetPrimAtPath(destination_root+relative)
        if not destination or destination.GetTypeName() != prim.GetTypeName():
            raise RuntimeError('Installed asset topology does not match this exhibit')
        for attribute in prim.GetAuthoredAttributes():
            name = attribute.GetName()
            if name != 'visibility' and not name.startswith('xformOp'):
                continue
            value = attribute.Get()
            target = destination.GetAttribute(name)
            if not target:
                target = destination.CreateAttribute(name, attribute.GetTypeName())
            if target.Get() != value:
                target.Set(value)
                changes += 1
    return changes

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rack revision persistence through the actual ovstorage Stack API."""
import asyncio
import hashlib
from pathlib import Path
import ovstorage
from ovstorage.file import FileBackend


class RackStorage:
    def __init__(self,extra):
        self.e=extra
        self.root=extra.snapshot_path.parent
        self.stack=None
        self.task=None
        self.operations={'write':0,'read':0,'list':0}
        self.last_error=None
        self.guided_saved = False
        self.guided_restored = False

    async def open(self):
        self.root.mkdir(parents=True,exist_ok=True)
        request=ovstorage.ConnectionRequest('file')
        request.add_config('root',ovstorage.ConfigValue.string(str(self.root.resolve())))
        request.set_display_name('Omniverse Libraries Playground rack revisions')
        self.stack=await ovstorage.Stack(root='files').backend(FileBackend('files')).connection('files',request).build()
        await self.refresh()
        if self.e.revisions:
            await self.select(0)

    async def refresh(self):
        page=await self.stack.list(self.root.resolve().as_uri()+'/',max_results=10000)
        self.operations['list']+=1
        from urllib.parse import urlparse,unquote
        self.e.revisions=sorted(self.root/Path(unquote(urlparse(item.address).path)).name for item in page.items if Path(unquote(urlparse(item.address).path)).name.startswith('rack-'))
        self.e.selected_revision=max(0,len(self.e.revisions)-1)

    def submit(self,operation,*args):
        if self.task and not self.task.done():
            return self.task
        async def run():
            try:
                self.last_error = None
                self.e.storage_status='ovstorage: '+operation+'...'
                return await getattr(self,operation)(*args)
            except Exception as exc:
                self.last_error=str(exc)
                self.e.storage_status='ovstorage error: '+str(exc)
                raise
        self.task=asyncio.create_task(run())
        # Retrieve exceptions so UI errors are visible without abandoned task warnings.
        self.task.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
        return self.task

    async def read(self,path):
        from pxr import Usd,Sdf
        data,info=await self.stack.read_bytes(Path(path).resolve().as_uri(),max_bytes=32*1024*1024)
        self.operations['read']+=1
        layer=Sdf.Layer.CreateAnonymous('.usda')
        if not layer.ImportFromString(data.decode('utf-8')):
            raise ValueError('Saved rack is not valid USD')
        stage=Usd.Stage.Open(layer)
        value=stage.GetPrimAtPath('/Asset').GetAttribute('lab:variant').Get()
        if value not in (0,1,2):
            raise ValueError('Saved rack has an invalid configuration')
        return value

    async def save(self):
        from pxr import Usd,UsdGeom,Sdf,Gf
        e=self.e
        snapshot=Usd.Stage.CreateInMemory()
        source=e.stage.GetRootLayer()
        Sdf.CopySpec(source,e.working.GetPath(),snapshot.GetRootLayer(),Sdf.Path('/Asset'))
        UsdGeom.Xform.Define(snapshot,'/World')
        Sdf.CopySpec(source,Sdf.Path('/World/Looks'),snapshot.GetRootLayer(),Sdf.Path('/World/Looks'))
        snapshot.GetPrimAtPath('/Asset').GetAttribute('xformOp:translate').Set(Gf.Vec3d(0))
        snapshot.SetDefaultPrim(snapshot.GetPrimAtPath('/Asset'))
        UsdGeom.SetStageUpAxis(snapshot,'Y')
        UsdGeom.SetStageMetersPerUnit(snapshot,1.)
        data=snapshot.GetRootLayer().ExportToString().encode('utf-8')
        number=max([int(p.stem.split('-')[-1]) for p in e.revisions] or [0])+1
        path=self.root/f'rack-{number:04}.usda'
        await self.stack.write(path.resolve().as_uri(),data)
        self.operations['write']+=1
        # Verify the library can read the persisted bytes before showing Saved.
        variant=await self.read(path)
        await self.refresh()
        e.saved_variant=variant
        e._set_asset(e.saved,e.saved_blocks,variant)
        e.saved.MakeVisible()
        e.snapshot_hash=hashlib.sha256(data).hexdigest()
        e.storage_revision+=1
        e.storage_status=f'Saved {path.name} with ovstorage / {len(data):,} bytes'
        self.guided_saved, self.guided_restored = True, False
        return path

    async def select(self,step):
        e=self.e
        if not e.revisions:
            e.storage_status='Save a revision to create a baseline.'
            return
        e.selected_revision=(e.selected_revision+step)%len(e.revisions)
        path=e.revisions[e.selected_revision]
        e.saved_variant=await self.read(path)
        e._set_asset(e.saved,e.saved_blocks,e.saved_variant)
        e.saved.MakeVisible()
        e.storage_status=f'Read {path.name} through ovstorage'

    async def restore(self):
        if not self.e.revisions:
            self.e.storage_status='Save a revision first.'
            return False
        path=self.e.revisions[self.e.selected_revision]
        variant=await self.read(path)
        self.e.set_asset_variant(variant)
        self.e.storage_status=f'Restored {path.name} through ovstorage'
        self.guided_restored = True
        return True

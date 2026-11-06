# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""References to prepared NVIDIA SimReady props; no downloads on the UI thread."""
import json
import os
from pxr import Gf, UsdGeom


class SimReadyAssets:
    def __init__(self,gallery):
        self.g=gallery
        self.directory=gallery.data_dir/'models/simready'
        path=self.directory/'index.json'
        self.assets=json.loads(path.read_text())['assets'] if path.is_file() and os.environ.get('OVLAB_SIMREADY','1')!='0' else {}

    def reference(self,path,key,size=None,scale=1.,instance=True,pos=(0,0,0)):
        info=self.assets[key]
        wrapper=UsdGeom.Xform.Define(self.g.stage,path)
        wrapper.AddTranslateOp().Set(Gf.Vec3d(*pos))
        wrapper.AddScaleOp().Set(Gf.Vec3f(*(tuple(size[i]/info['size'][i] for i in range(3)) if size else (scale,)*3)))
        child=UsdGeom.Xform.Define(self.g.stage,path+'/Asset').GetPrim()
        child.GetReferences().AddReference(str((self.directory/info['file']).resolve()))
        child.SetInstanceable(instance)
        return wrapper

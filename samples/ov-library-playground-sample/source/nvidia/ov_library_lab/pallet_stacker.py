# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An operator-controlled pallet stacker; kinematic visualization, not load simulation."""
from pxr import Gf, UsdGeom


class PalletStacker:
    def __init__(self,g):
        self.g=g
        self.rise=0.
        with g.factory.assembly('/World/Interface/Stacker',(0,.30,-5.5),yaw=-25):
            g.factory.simready.reference('/World/Interface/Stacker/Visual','stacker',instance=False)
            base='/World/Interface/Stacker/Visual/Asset/Axis/Model/Geometry/'
            self.forks=[]
            for name in ('sm_fork_a01_obj_00','sm_forkwheels1_a01_obj_00','sm_forkwheels2_a01_obj_00'):
                p=g.stage.GetPrimAtPath(base+name)
                # Source parts remain in their Z-up authored frames. An extra
                # parent-space Z translation follows the source's prismatic axis.
                x=UsdGeom.Xformable(p)
                ops=x.GetOrderedXformOps()
                rise=x.AddTranslateOp(opSuffix='operatorRise')
                x.SetXformOpOrder([rise,*ops])
                self.forks.append(rise)
            with g.factory.assembly('/World/Interface/Lift',(0,.14,.20)) as self.platform:
                g.factory.pallet('/World/Interface/Lift/Pallet',(0,0,0),(.95,1.10))
                for i in range(2):
                    g.factory.carton('/World/Interface/Lift/Carton'+str(i),((i-.5)*.43,.46,0),(.4,.6,.75))
                self.tag=g.box('/World/Interface/Lift/InspectionTag',(0,.50,.39),(.24,.22,.012),'blue')
        # Lift lives at a stable top-level path, so reproduce the stacker frame
        # while retaining that contract for the ovui/ovstage checks.
        frame=UsdGeom.Xformable(g.stage.GetPrimAtPath('/World/Interface/Stacker')).ComputeLocalToWorldTransform(0)
        self.load_base=frame.Transform(Gf.Vec3d(0,.14,.20))
        rotation=g.stage.GetPrimAtPath('/World/Interface/Stacker').GetAttribute('xformOp:rotateY').Get()
        self.platform.GetPrim().GetAttribute('xformOp:rotateY').Set(rotation)
        self.set_height(.25)

    def set_height(self,value):
        # 1.51 m is the supplied fork-joint travel limit. Stay within its mast.
        self.rise=max(0.,min(1.,float(value)))*1.51
        for fork in self.forks:fork.Set(Gf.Vec3d(0,0,self.rise))
        self.platform.GetPrim().GetAttribute('xformOp:translate').Set(self.load_base+Gf.Vec3d(0,self.rise,0))

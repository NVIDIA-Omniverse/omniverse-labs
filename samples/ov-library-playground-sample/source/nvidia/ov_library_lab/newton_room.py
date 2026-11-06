# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detailed gantry whose moving assemblies display native solver poses."""
from pxr import Gf, UsdGeom
from .library_worker import LibraryWorker


class NewtonRoom:
    def __init__(self, gallery, build=True):
        self.g = gallery
        self.worker = None
        self.error = None
        self.busy = self.playing = False
        self.result = None
        self.elapsed = 0.
        self.mode = 'gentle'
        self.comparisons = {}
        self.status = 'Choose a motion profile, then run the transfer.'
        self.bodies = {}
        if build:
            self._build()

    def _build(self):
        g, p = self.g, '/World/NewtonCell'
        g.active_room = 'newton'
        g.factory.boundary(p+'/Boundary', 0, -5.8, 8.8, 4.5)
        g.box(p+'/Slab', (0,.045,-5.8),(8.5,.09,4.1),'black',obstacle=True)
        with g.factory.assembly(p+'/Rig',(0,0,-5.8)):
            r = p+'/Rig'
            for i,x in enumerate((-3.7,3.7)):
                for j,z in enumerate((-.60,.60)):
                    q = r+f'/Column{i}{j}'
                    g.box(q+'/Foot',(x,.12,z),(.66,.16,.60),'steel')
                    g.box(q+'/Post',(x,1.90,z),(.22,3.55,.24),'safety')
                    g.box(q+'/Cap',(x,3.64,z),(.48,.18,.48),'steel')
                    for k,dx in enumerate((-.23,.23)):
                        for l,dz in enumerate((-.20,.20)):
                            g.cylinder(q+f'/Anchor{k}{l}',(x+dx,.22,z+dz),.033,.055,'steel')
                    g.box(q+'/Guard',(x,.40,z),(.30,.52,.32),'black')
            for i,z in enumerate((-.50,.50)):
                g.box(r+f'/BeamWeb{i}',(0,3.83,z),(7.9,.38,.08),'safety')
                for j,y in enumerate((3.62,4.04)):
                    g.box(r+f'/BeamFlange{i}{j}',(0,y,z),(7.9,.06,.32),'safety')
                g.box(r+f'/Rail{i}',(0,3.665,z),(7.3,.036,.095),'steel')
            for i,x in enumerate((-3.3,3.3)):
                g.box(r+f'/Buffer{i}',(x,3.79,0),(.14,.25,1.30),'rubber')
            g.box(r+'/RearBrace',(0,4.06,-.65),(7.5,.10,.12),'steel')
            for name, position in (('Trolley',(-1.6,3.6,0)),('Load',(-1.6,1.9,0))):
                q = r+'/'+name
                with g.factory.assembly(q, position) as root:
                    rotate = root.AddOrientOp()
                    self.bodies['/Gantry/'+name] = (root.GetOrderedXformOps()[0],rotate)
                    if name == 'Trolley':
                        g.box(q+'/Housing',(0,.07,0),(.88,.26,1.08),'rack_blue')
                        g.box(q+'/ServiceCover',(0,.214,0),(.72,.025,.8),'aluminum')
                        for i,x in enumerate((-.29,.29)):
                            for j,z in enumerate((-.5,.5)):
                                g.cylinder(q+f'/Wheel{i}{j}',(x,.16,z),.14,.10,'steel',axis='Z')
                                g.cylinder(q+f'/Hub{i}{j}',(x,.16,z+(.056 if z>0 else -.056)),.045,.025,'black',axis='Z')
                        g.cylinder(q+'/DriveMotor',(.08,.14,.69),.16,.40,'blue',axis='X')
                        for i in range(6):
                            g.box(q+f'/CoolingFin{i}',(-.08+i*.055,.14,.69),(.018,.34,.34),'galvanized')
                        g.cylinder(q+'/Pivot',(0,-.04,0),.11,.44,'steel',axis='Z')
                    else:
                        # One rigid body: link, fixture and load travel together.
                        for i,z in enumerate((-.22,.22)):
                            g.box(q+f'/RigidLink{i}',(0,.93,z),(.065,1.55,.045),'steel')
                            g.cylinder(q+f'/PivotCollar{i}',(0,1.70,z),.12,.06,'steel',axis='Z')
                        g.box(q+'/LiftingCrossbar',(0,.42,0),(1.24,.08,.56),'steel')
                        g.box(q+'/Crate',(0,0,0),(1.05,.70,.80),'rack_blue')
                        g.box(q+'/Lid',(0,.365,0),(1.10,.05,.85),'aluminum')
                        g.box(q+'/Base',(0,-.40,0),(1.24,.13,.95),'steel')
                        for i,x in enumerate((-.48,.48)):
                            g.box(q+f'/Corner{i}',(x,0,.413),(.065,.68,.026),'galvanized')
                            g.box(q+f'/Latch{i}',(x,.22,.439),(.11,.14,.04),'black')
                        for i in range(7):
                            g.box(q+f'/Rib{i}',(-.39+i*.13,0,.411),(.038,.5,.026),'rack_blue')
                        g.box(q+'/Label',(0,.18,.431),(.44,.16,.008),'paper')
                        for i in range(9):
                            g.box(q+f'/Barcode{i}',(-.16+i*.039,.18,.437),(.015,.10,.003),'black')
            for i,x in enumerate((-1.6,1.6)):
                g.box(r+f'/Dock{i}',(x,.14,0),(1.60,.19,1.20),'galvanized')
                g.box(r+f'/DockStripe{i}',(x,.245,.62),(1.60,.018,.08),'newton_glow')
            with g.factory.assembly(r+'/Cabinet',(3.0,.10,-1.48)):
                g.box(r+'/Cabinet/Case',(0,.85,0),(.85,1.7,.48),'ivory')
                g.box(r+'/Cabinet/Door',(0,.86,.25),(.76,1.57,.025),'aluminum')
                g.box(r+'/Cabinet/HMI',(0,1.28,.28),(.53,.32,.045),'black')
                for i in range(3):
                    g.box(r+f'/Cabinet/Line{i}',(-.04,1.36-i*.07,.306),(.32,.018,.008),'newton_glow')
                g.cylinder(r+'/Cabinet/Stop',(.19,.93,.31),.048,.07,'rack_orange',axis='Z')
        g.placard('newton_label',(0,.62,-2.95),2.2,.70,'newton_label.png',obstacle=True)
        if not g.campus:
            g.sign('newton_docks', (0,.48,-4.70), 3.5,.48,'newton_docks.png')
        g.active_room = None
        self._initial_pose()

    def attach(self, root):
        try:
            self.worker = LibraryWorker(root,'newton')
        except Exception as exc:
            self.error = str(exc)

    def _initial_pose(self):
        for name,(translation,rotation) in self.bodies.items():
            translation.Set(Gf.Vec3d(-1.6,3.6 if name.endswith('Trolley') else 1.9,0))
            rotation.Set(Gf.Quatf(1.))

    def run(self):
        if self.busy or not self.worker or not self.worker.ready or self.worker.error:
            return
        self.playing = False
        self.busy = True
        self.error = None
        self.elapsed = 0.
        self.result = None
        self._initial_pose()
        self.status = 'Computing the transfer with Newton...'
        self.worker.submit(mode=self.mode)

    def advance(self, dt):
        if self.worker:
            reply = self.worker.poll()
            if reply:
                self.busy = False
                if 'error' in reply:
                    self.error = reply['error']
                    self.status = 'Simulation unavailable. See newton-worker.log.'
                else:
                    self.result, self.elapsed, self.playing = reply, 0., True
                    self.comparisons[reply['mode']] = reply['peak_sway']
                    self.status = 'Showing the computed motion at simulation speed.'
        if not self.result or not self.playing or dt <= 0:
            return
        self.elapsed = min(self.elapsed+dt,(len(self.result['frames'])-1)/self.result['fps'])
        n = self.elapsed*self.result['fps']
        a,b = int(n), min(int(n)+1,len(self.result['frames'])-1)
        for index,name in enumerate(self.result['labels']):
            if name not in self.bodies:
                continue
            first,second = self.result['frames'][a][index],self.result['frames'][b][index]
            weight = n-a
            position = [x+(y-x)*weight for x,y in zip(first[:3],second[:3])]
            q1,q2 = Gf.Quatf(first[6],Gf.Vec3f(*first[3:6])), Gf.Quatf(second[6],Gf.Vec3f(*second[3:6]))
            translation,rotation = self.bodies[name]
            translation.Set(Gf.Vec3d(*position))
            rotation.Set(Gf.Slerp(weight,q1,q2))
        if a == len(self.result['frames'])-1:
            self.playing = False
            self.status = 'Transfer complete. Compare the other motion profile.'

    def reset(self):
        if self.worker:
            self.worker.invalidate()
        self.busy = self.playing = False
        self.result = None
        self.comparisons.clear()
        self.elapsed = 0.
        self.mode = 'gentle'
        self.status = 'Choose a motion profile, then run the transfer.'
        self._initial_pose()

    def close(self):
        if self.worker:
            self.worker.close()

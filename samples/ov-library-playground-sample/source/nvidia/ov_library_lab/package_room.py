# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Asset handoff stations, backed by real ovpackage revisions and file hashes."""
from pxr import UsdGeom
from .library_worker import LibraryWorker
from .package_asset import snapshot, bundle_snapshot, apply_installed
from concurrent.futures import ThreadPoolExecutor


class PackageRoom:
    def __init__(self, gallery, build=True):
        self.g = gallery
        self.worker = None
        self.error = None
        self.busy = False
        self.action = None
        self.bundle_task = None
        self.bundle_executor = None
        self.profile = 'logistics'
        self.versions = []
        self.published_profiles = {}
        self.publishing_profile = None
        self.selected = 0
        self.installed = None
        self.receipt = None
        self.status = 'Choose a source configuration, then publish your first revision.'
        self.source = '/World/PackageCell/Source/Robot'
        self.destination = '/World/PackageCell/Receiving/Robot'
        if build:
            self._build()

    def _build(self):
        g,p = self.g,'/World/PackageCell'
        g.active_room = 'package'
        for name,x in (('Source',-2.55),('Receiving',2.55)):
            path = p+'/'+name
            g.box(path+'Plinth',(x,.105,-5.55),(3.45,.21,3.30),'black',obstacle=True)
            g.box(path+'Inlay',(x,.221,-5.55),(3.22,.022,3.08),'galvanized')
            g.factory.boundary(path+'Markings',x,-5.55,3.55,3.40)
            with g.factory.assembly(path,(x,.24,-5.55),yaw=18 if name=='Source' else -18):
                robot = path+'/Robot'
                g.factory.amr(robot,(0,0,0))
                UsdGeom.Imageable(g.stage.GetPrimAtPath(robot+'/Load')).CreateVisibilityAttr().Set('inherited')
                with g.factory.assembly(robot+'/Inspection',(0,-.46 if g.factory.simready.assets else 0,0)) as inspection:
                    g.box(robot+'/Inspection/Column',(-.35,1.25,0),(.11,1.16,.12),'aluminum')
                    g.box(robot+'/Inspection/Foot',(-.35,.76,0),(.5,.09,.48),'steel')
                    g.box(robot+'/Inspection/Head',(-.35,1.85,0),(.43,.25,.28),'ivory')
                    for i,xoffset in enumerate((-.48,-.22)):
                        g.cylinder(robot+f'/Inspection/Lens{i}',(xoffset,1.86,.155),.065,.06,'sensor_lens',axis='Z')
                    g.box(robot+'/Inspection/Computer',(.12,.87,0),(.50,.27,.67),'blue')
                    for i in range(7):
                        g.box(robot+f'/Inspection/Fin{i}',(-.08+i*.064,.99,0),(.022,.06,.62),'steel')
                    g.cylinder(robot+'/Inspection/Loom',(-.415,1.20,0),.02,.85,'rubber')
                    UsdGeom.Imageable(inspection.GetPrim()).CreateVisibilityAttr().Set('invisible')
            g.box(path+'LabelMount',(x,.34,-3.82),(2.84,.48,.08),'black')
            g.sign('package_'+name.lower(),(x,.34,-3.769),2.7,.42,'package_'+name.lower()+'.png')
        # Visible link between workstations: publishing is an asset handoff,
        # never a simulation of network packets or cloud delivery.
        for i,x in enumerate((-.55,0,.55)):
            g.box(p+f'/TransferLight{i}',(x,.013,-5.55),(.27,.018,.80),'package_glow')
        g.box(p+'/Repository',(0,.7,-8.3),(1.3,1.4,.65),'black',obstacle=True)
        for i in range(4):
            g.box(p+f'/Drawer{i}',(0,.28+i*.28,-7.961),(1.15,.22,.035),'galvanized')
            g.box(p+f'/Handle{i}',(0,.28+i*.28,-7.922),(.35,.036,.06),'steel')
            g.sphere(p+f'/LED{i}',(.45,.28+i*.28,-7.923),.022,'package_glow')
        g.sign('package_repository',(0,1.8,-7.93),2.35,.50,'package_repository.png')
        g.active_room = None
        self.receiving_visibility = UsdGeom.Imageable(g.stage.GetPrimAtPath(p+'/Receiving')).CreateVisibilityAttr()
        self.receiving_visibility.Set('invisible')

    def attach(self, root):
        try:
            self.worker = LibraryWorker(root,'package')
        except Exception as exc:
            self.error = str(exc)

    def choose(self, profile):
        if self.busy or profile not in ('logistics','inspection'):
            return
        self.profile = profile
        for name,shown in (('Load',profile=='logistics'),('Inspection',profile=='inspection')):
            UsdGeom.Imageable(self.g.stage.GetPrimAtPath(self.source+'/'+name)).GetVisibilityAttr().Set(
                'inherited' if shown else 'invisible')

    def select(self, delta):
        if not self.busy and self.versions:
            self.selected = (self.selected+delta)%len(self.versions)

    def submit(self, action):
        if self.busy or not self.worker or not self.worker.ready or self.worker.error:
            return
        if action in ('install','restore') and not self.versions:
            return
        if action in ('edit','verify') and not self.installed:
            return
        request = {'action': action}
        if action == 'publish':
            text=snapshot(self.g.stage,self.source)
            if self.bundle_executor is None:self.bundle_executor=ThreadPoolExecutor(max_workers=1)
            self.bundle_task=self.bundle_executor.submit(bundle_snapshot,text)
            self.publishing_profile = self.profile
        if action in ('install','restore'):
            request['version'] = self.versions[self.selected]
        self.busy, self.action, self.error = True, action, None
        self.status = {'publish':'Publishing the source asset...', 'install':'Installing the selected revision...',
                       'restore':'Restoring the selected published revision...', 'verify':'Verifying file contents...',
                       'edit':'Editing the installed working copy and checking its files...'}[action]
        if action != 'publish':self.worker.submit(**request)

    def advance(self):
        if self.bundle_task:
            if not self.bundle_task.done():return
            task,self.bundle_task=self.bundle_task,None
            try:self.worker.submit(action='publish',files=task.result())
            except Exception as exc:
                self.busy=False
                self.error=str(exc)
                self.status='Could not prepare package: '+str(exc)
                return
        reply = self.worker.poll() if self.worker else None
        if not reply:
            return
        self.busy = False
        if 'error' in reply:
            self.error = reply['error']
            self.status = ('The working copy was edited. Restore selected to replace it with the published revision.'
                           if self.action=='install' and self.receipt and not self.receipt['verified']
                           else 'Operation failed. Your published files are preserved.')
            return
        self.versions = reply['versions']
        if reply['action'] == 'publish':
            self.published_profiles[reply['version']] = self.publishing_profile
            self.selected = self.versions.index(reply['version'])
            self.status = f"Published revision {reply['version']} / {reply['file_count']} files. Install it on the receiving station."
        else:
            try:
                apply_installed(self.g.stage,self.destination,reply['asset'])
                self.receiving_visibility.Set('inherited')
                self.installed = reply['version']
                self.receipt = reply
                self.status = (f"Verified revision {self.installed} / {reply['file_count']} payload files match."
                               if reply['verified'] else 'Working copy differs: '+', '.join(reply['changed']+reply['missing']+reply['extra']))
            except Exception as exc:
                self.error = str(exc)
                self.status = 'Package operation completed, but the installed scene could not be displayed.'

    def reset(self):
        if self.busy:
            raise RuntimeError('Wait for the package operation to finish, then reset.')
        self.choose('logistics')
        self.error = None
        self.status = 'Source reset. Published revisions and the installed workspace are preserved.'

    def close(self):
        if self.bundle_executor:self.bundle_executor.shutdown(wait=False,cancel_futures=True)
        if self.worker:
            self.worker.close()

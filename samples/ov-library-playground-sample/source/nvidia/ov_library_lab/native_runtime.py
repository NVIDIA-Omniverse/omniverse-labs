# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real ovstage, ovrtx and ovphysx runtime. No Kit APIs or fallback engines."""
import os
import time
from pathlib import Path
import numpy as np


class NativeRuntime:
    FIRST_FRAME_TIMEOUT_SECONDS = 600.
    FRAME_TIMEOUT_SECONDS = 180.

    def __init__(self):
        os.environ.setdefault('OVRTX_SKIP_USD_CHECK', '1')
        import ovphysx
        import ovstage
        import ovrtx
        self.ovstage, self.ovrtx, self.ovphysx = ovstage, ovrtx, ovphysx
        ovstage.population.register_usd_schemas([str(ovphysx.codeless_schema_root())])
        # Optional native logging for startup/driver investigations. Disabled
        # by default; callers choose a local diagnostic path.
        log_path = os.environ.get('OVLAB_RTX_LOG')
        config = ovrtx.RendererConfig(log_file_path=log_path, log_level='info') if log_path else None
        self.renderer = ovrtx.Renderer(config)
        self.stage = ovstage.Stage('ov-library-lab')
        self.renderer.attach_ovstage(self.stage)
        self.paths = ovstage.PathDictionary(self.stage)
        self.ordinal = 1
        self.queries = {}
        self.dirty = set()
        self.notice = None
        self.usd = None
        self.frames = self.writes = self.physics_steps = 0
        self.physx = None
        self.physics_stage = None
        self._physics_source_stage = None
        self.physics_paths = None
        self.original_poses = {}
        self.pose_reads = {}
        self.products = {}
        self.prototype_aliases = {}
        self._transform_members = {}
        self.active_products = None
        self.last_frame_counts = {}
        self.max_frames_per_step = 0
        self.render_time = 0.
        self.pending_render = None
        self.pending_names = None
        self.pending_outputs = None
        self.pending_mappings = {}
        self.render_started = 0.
        self.render_paused_at = None
        self.last_render_ms = 0.
        self.quality_spec = None
        self.resident_products = frozenset({'Main', 'Preview'})
        self.sensor_output = 'LdrColor'
        self.sensor_fisheye = False
        self.sensor_raw = None
        self.sensor_raw_name = None
        self.sensor_labels = {}

    def populate(self, usd):
        from pxr import Sdf, UsdRender, UsdGeom, Tf, Usd, Gf
        self.usd = usd
        from .sensor_views import author_labels, OUTPUTS
        author_labels(usd)
        # A stable preview camera avoids tearing down RTX camera resources when
        # opening/closing room panels. Its pose and lens follow the selected
        # authored camera; its product, camera path and resolution stay fixed.
        self.preview_camera = UsdGeom.Camera.Define(usd, '/World/LabPreviewCamera')
        self.preview_transform = self.preview_camera.AddTransformOp()
        self.preview_attributes = [name for name in UsdGeom.Camera.GetSchemaAttributeNames()
                                   if not name.startswith('xformOp')]
        from .sensor_views import author_lenses
        author_lenses(self.preview_camera)
        for name, camera, resolution in (
            ('Main', '/World/TourCamera', (1824, 1191)),
            ('Inspection', '/World/InspectionCamera', (640, 360)),
            ('Fleet', '/World/ObserverCamera', (640, 360)),
            ('Stream', '/World/StreamCamera', (640, 360)),
            ('Preview', '/World/LabPreviewCamera', (640, 360)),
        ):
            if not usd.GetPrimAtPath(camera):
                raise RuntimeError(f'Missing camera {camera}')
            path = '/Render/'+name
            product = UsdRender.Product.Define(usd, path)
            product.CreateCameraRel().SetTargets([camera])
            product.CreateResolutionAttr(Gf.Vec2i(*resolution))
            product.GetPrim().AddAppliedSchema('OmniRtxDebugSettingsAPI_1')
            product.GetPrim().CreateAttribute('omni:rtx:rendermode', Sdf.ValueTypeNames.Token).Set('RealTimePathTracing')
            product.GetPrim().CreateAttribute('omni:rtx:rtpt:maxBounces', Sdf.ValueTypeNames.UInt).Set(6)
            from .campus_assets import CAMPUS_ENABLED, CAMPUS_REFINED
            if CAMPUS_ENABLED and CAMPUS_REFINED:
                # Solid architectural panes require two interfaces per window.
                # Keep the normal shading budget, but allow glass/reflections
                # to finish their paths instead of turning the exterior black.
                product.GetPrim().CreateAttribute(
                    'omni:rtx:rtpt:extraSpecularAndTransmissiveBounces',
                    Sdf.ValueTypeNames.UInt).Set(4)
            var = UsdRender.Var.Define(usd, path+'/LdrColor')
            var.CreateSourceNameAttr('LdrColor')
            product.CreateOrderedVarsRel().SetTargets([var.GetPath()])
            if name == 'Preview':
                self.preview_product = product
                for output in (*OUTPUTS.values(), 'SemanticIdMap'):
                    UsdRender.Var.Define(usd, path+'/'+output).CreateSourceNameAttr(output)
            else:
                self.products[name] = product
        self._configure_preview('Fleet')
        self.ovstage.population.open_usd_from_string(self.stage, usd.GetRootLayer().ExportToString(),
                                                    ordinal=self.ordinal, domains=self.ovstage.PopulationDomain.ALL)
        self.stage.advance_write_floor(self.ordinal).wait()
        # OpenUSD prototype names are local to each stage and need not agree
        # with ovstage population's names. Resolve them through the library.
        for prim in usd.Traverse():
            if prim.IsInstance():
                source=str(prim.GetPrototype().GetPath())
                target=self.ovstage.instancing.get_prototype_root(self.stage,str(prim.GetPath()))
                self.prototype_aliases[source]=target
        self.notice = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self._changed, usd)

    def _changed(self, notice, sender):
        self.dirty.update(str(p) for p in notice.GetChangedInfoOnlyPaths())
        for path in notice.GetResyncedPaths():
            # Cached handles are valid only while the composed topology and
            # transform schemas stay unchanged. Ordinary pose edits retain it.
            self._transform_members.clear()
            if path.IsPropertyPath():
                self.dirty.add(str(path))
                continue
            # New/removed bodies or schemas change the simulation population.
            self._physics_source_stage = None
            prim = sender.GetPrimAtPath(path.GetPrimPath())
            if prim:
                self.dirty.update(str(a.GetPath()) for a in prim.GetAttributes())
                self.dirty.update(str(a.GetPath()) for a in prim.GetRelationships())

    def query(self, paths):
        key = tuple(paths)
        if key not in self.queries:
            handle = self.paths.create_path_list_from_strings(list(key))
            self.queries[key] = handle, self.stage.query_from_path_list(handle)
        return self.queries[key][1]

    def write(self, paths, attribute, values, *, array=False, semantic=0):
        mapped=[]
        for path in paths:
            for source,target in self.prototype_aliases.items():
                if path==source or path.startswith(source+'/'):
                    path=target+path[len(source):]
                    break
            mapped.append(path)
        self.stage.write_attribute(self.query(mapped), attribute, self.ordinal, values,
                                   is_array=array, semantic=semantic).wait()
        self.writes += 1

    def flush(self):
        from pxr import Usd, UsdGeom, Sdf
        changes, self.dirty = self.dirty, set()
        if not changes:
            return
        self.ordinal += 1
        transforms = set()
        for path in sorted(changes):
            p = Sdf.Path(path)
            prim_path = str(p.GetPrimPath())
            name = p.name
            if name.startswith('xformOp'):
                transforms.add(prim_path)
                continue
            prim = self.usd.GetPrimAtPath(prim_path)
            if not prim:
                continue
            rel = prim.GetRelationship(name)
            if rel and name in ('material:binding', 'camera', 'orderedVars'):
                ids = np.array([self.paths.intern_path(str(x)) for x in rel.GetTargets()], dtype=np.uint64)
                self.write([prim_path], name, [ids], array=True,
                           semantic=self.ovstage.AttributeSemantic.RELATIONSHIP_PATH_ID)
                continue
            attr = prim.GetAttribute(name)
            if not attr or name.startswith('lab:'):
                continue
            value = attr.Get()
            if value is None:
                continue
            typ = str(attr.GetTypeName())
            if typ in ('token', 'string'):
                self.write([prim_path], name, np.array([self.paths.intern_token(str(value))], dtype=np.uint64),
                           semantic=self.ovstage.AttributeSemantic.TOKEN_ID)
            elif typ.endswith('[]'):
                if typ == 'token[]':
                    values = np.array([self.paths.intern_token(str(v)) for v in value],dtype=np.uint64)
                    self.write([prim_path], name, [values], array=True, semantic=self.ovstage.AttributeSemantic.TOKEN_ID)
                elif typ in ('point3f[]', 'float3[]', 'vector3f[]', 'color3f[]'):
                    values = np.ascontiguousarray(value, dtype=np.float32)
                    tensor = self.ovstage.make_dltensor(values, dtype=self.ovstage.DLDataType(self.ovstage.DLDataTypeCode.kDLFloat,32,3), shape=[len(values)], ndim=1)
                    self.write([prim_path],name,[tensor],array=True)
            elif typ not in ('asset',):
                dtype = np.float64 if 'double' in typ or 'matrix' in typ else np.uint32 if typ=='uint' else np.int32 if typ.startswith('int') else np.bool_ if typ=='bool' else np.float32
                self.write([prim_path],name,np.asarray([value],dtype=dtype))
        if transforms:
            # Refresh affected descendants and referenced prototypes as a batch.
            # Updating only the parent can leave rendered child poses cached,
            # even when reading the parent's matrix reports the new value.
            members = {}
            for root in sorted(transforms, key=len):
                if root in members:
                    continue
                cached = self._transform_members.get(root)
                if cached is None:
                    cached = {}
                    pending, visited = [root], set()
                    while pending:
                        path = pending.pop()
                        if path in visited:
                            continue
                        visited.add(path)
                        prim = self.usd.GetPrimAtPath(path)
                        if not prim:
                            continue
                        for child in Usd.PrimRange(prim):
                            xform = UsdGeom.Xformable(child)
                            if xform:
                                cached[str(child.GetPath())] = xform
                            if child.IsInstance():
                                pending.append(str(child.GetPrototype().GetPath()))
                    self._transform_members[root] = cached
                members.update(cached)
            paths=sorted(members)
            matrices=np.asarray([members[p].GetLocalTransformation() for p in paths],dtype=np.float64)
            self.write(paths,'omni:xform',matrices,semantic=self.ovstage.AttributeSemantic.MATRIX)
            # Publish both parts of the transform contract. In particular,
            # effective-world instance poses must stop inheriting their parent.
            self.write(paths, 'omni:resetXformStack', np.array([
                members[p].GetResetXformStack()
                for p in paths], dtype=np.bool_))
        self.stage.advance_write_floor(self.ordinal).wait()

    def render(self, name='Main'):
        # Capturing a view must not suspend the other currently visible views.
        return self.render_all((self.active_products or frozenset()) | {name})[name]

    def render_all(self, names=None):
        """Render fresh requested views; synchronous exports may request several."""
        # API checks and exports request a fresh frame after their edits. The
        # interactive UI uses begin_render/poll_render and never waits here.
        while self.pending_render is not None:
            self.poll_render(1_000_000)
        names = frozenset(self.products if names is None else names)
        if not names or not names.issubset(self.products):
            raise ValueError(f'Unknown or empty render product set: {names}')
        secondary = sorted(names-{'Main'})
        if len(secondary) > 1:
            # Interactive panels display one preview. Exports requesting more
            # reuse that same native camera sequentially, with fresh outputs.
            images, counts = {}, {}
            for name in secondary:
                images.update(self.render_all({'Main', name}))
                counts.update(self.last_frame_counts)
            self.active_products = names
            self.last_frame_counts = {name:counts[name] for name in names}
            return {name:images[name] for name in names}
        self.begin_render(names)
        while True:
            images = self.poll_render(1_000_000)
            if images is not None:
                return images

    def set_quality(self, resolution, bounces):
        """Author the next frame's settings only between native render jobs."""
        if self.pending_render is not None:
            return False
        spec = tuple(resolution), int(bounces)
        if self.quality_spec == spec:
            return True
        from pxr import Gf, Sdf
        self.products['Main'].GetResolutionAttr().Set(Gf.Vec2i(*resolution))
        for product in (*self.products.values(), self.preview_product):
            product.GetPrim().GetAttribute('omni:rtx:rtpt:maxBounces').Set(bounces)
        from .campus_assets import CAMPUS_ENABLED
        if CAMPUS_ENABLED:
            # At the laptop's output size "auto" selected expensive native
            # sampling. Explicit reconstruction retains the same viewport
            # resolution, geometry, physical materials and ray-traced lights.
            studio = resolution[0] >= 1824
            main = self.products['Main'].GetPrim()
            main.CreateAttribute('omni:rtx:post:dlss:execMode', Sdf.ValueTypeNames.Token).Set('quality' if studio else 'performance')
            main.CreateAttribute('omni:rtx:rtpt:usePathSplitting', Sdf.ValueTypeNames.Bool).Set(studio)
        self.quality_spec = spec
        return True

    def begin_render(self, names=None):
        if self.pending_render is not None:
            return False
        names = frozenset(self.products if names is None else names)
        if not names or not names.issubset(self.products):
            raise ValueError(f'Unknown or empty render product set: {names}')
        secondary = names-{'Main'}
        if len(secondary) > 1:
            raise ValueError('Asynchronous presentation supports one preview; use render_all for multi-camera exports.')
        if secondary:
            self._configure_preview(next(iter(secondary)))
        elif self.preview_source is not None:
            # Retain the small camera's GPU resources without tracing the
            # gallery when its image has no visible consumer.
            from pxr import Gf
            self.preview_transform.Set(Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(0,10000,0)))
            self.preview_camera.GetClippingRangeAttr().Set(Gf.Vec2f(.1,1.))
            self.preview_source = None
        output = self.sensor_output if secondary == {'Inspection'} else 'LdrColor'
        targets = ['/Render/Preview/LdrColor']
        if output != 'LdrColor':
            targets.append('/Render/Preview/'+output)
        if output == 'SemanticSegmentation':
            targets.append('/Render/Preview/SemanticIdMap')
        from pxr import Sdf
        if self.preview_product.GetOrderedVarsRel().GetTargets() != [Sdf.Path(p) for p in targets]:
            self.preview_product.GetOrderedVarsRel().SetTargets(targets)
        self.pending_sensor_output = output
        self.pending_sensor_camera = {
            'projection': 'opencvFisheye' if self.sensor_fisheye and secondary == {'Inspection'} else 'opencvPinhole',
            'resolution': [640,360], 'cx': 320., 'cy': 180.,
            'fx': 220. if self.sensor_fisheye and secondary == {'Inspection'} else 640*self.preview_camera.GetFocalLengthAttr().Get()/self.preview_camera.GetHorizontalApertureAttr().Get(),
            'camera_to_world': [list(row) for row in self.preview_transform.Get()]}
        self.pending_sensor_camera.update(fy=self.pending_sensor_camera['fx'],
                                          distortion_coefficients=[0., 0., 0., 0.],
                                          calibration='Illustrative, not a measured physical camera')
        self.flush()
        self.active_products = names
        self.render_started = time.perf_counter()
        self.pending_names = names
        self.pending_render = self.renderer.step_async(render_products={'/Render/'+name for name in self.resident_products},
                                             delta_time=1/60, ordinal=self.ordinal)
        return True

    def _configure_preview(self, name):
        from pxr import Usd, UsdGeom
        source = UsdGeom.Camera(self.usd.GetPrimAtPath(self.products[name].GetCameraRel().GetTargets()[0]))
        matrix = source.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        if self.preview_transform.Get() != matrix:
            self.preview_transform.Set(matrix)
        for attribute in self.preview_attributes:
            source_attr = source.GetPrim().GetAttribute(attribute)
            target_attr = self.preview_camera.GetPrim().GetAttribute(attribute)
            value = source_attr.Get()
            if value is not None and target_attr.Get() != value:
                target_attr.Set(value)
        self.preview_source = name
        from .sensor_views import configure_lens
        configure_lens(self.preview_camera, self.sensor_fisheye and name == 'Inspection')

    def poll_render(self, timeout_ns=0, gpu_presenters=None):
        """Poll once; ovui continues pumping input while the GPU is working."""
        if self.pending_render is None:
            return None
        if self.pending_outputs is None:
            pending = self.pending_render.wait(timeout_ns=timeout_ns)
            if pending is None:
                self._check_render_timeout()
                return None
            # Both phases have independent timeout control. Keep the operation
            # alive until its outputs and mapped image buffers are ready.
            self.pending_outputs = pending.fetch(timeout_ns=timeout_ns)
            if self.pending_outputs is None:
                self._check_render_timeout()
                return None
        products = self.pending_outputs
        names = self.pending_names
        paths = {name:'/Render/'+('Main' if name == 'Main' else 'Preview') for name in names}
        if gpu_presenters and not self.pending_mappings:
            import warp
            for name, path in paths.items():
                # Color readiness does not imply that a sensor buffer is ready.
                # Poll the selected producer before its CPU readback, keeping
                # input live while normals/semantics finish on the GPU.
                source = self.pending_sensor_output if name == 'Inspection' else 'LdrColor'
                output = products[path].frames[-1].render_vars[path+'/'+source]
                mapped = output.map(device=self.ovrtx.Device.CUDA)
                event = warp.Event(device='cuda:0', cuda_event=mapped.wait_event) if mapped.wait_event else None
                self.pending_mappings[name] = mapped, event
        # A completed step/fetch can still expose a CUDA buffer whose producer
        # is running. Poll its borrowed event instead of mapped.wait(), which
        # blocked the desktop input loop for up to 90 ms per campus frame.
        if any(event is not None and not event.is_complete for _,event in self.pending_mappings.values()):
            self._check_render_timeout()
            return None
        images={}
        self.last_frame_counts = {name: len(products[path].frames) for name,path in paths.items()}
        self.max_frames_per_step = max(self.max_frames_per_step, *self.last_frame_counts.values())
        try:
            for name in names:
                path=paths[name]
                frame=products[path].frames[-1]
                presenter = (gpu_presenters or {}).get(name)
                if presenter:
                    mapped, _ = self.pending_mappings[name]
                    presenter(mapped)
                    images[name] = None  # Already delivered; avoid CPU readback.
                    continue
                if name in self.pending_mappings:
                    self.pending_mappings.pop(name)[0].unmap()
                source = self.pending_sensor_output if name == 'Inspection' else 'LdrColor'
                mapped=frame.render_vars[path+'/'+source].map(device=self.ovrtx.Device.CPU)
                try:
                    view=np.from_dlpack(mapped)
                    images[name]=view.copy()
                    del view
                finally:
                    mapped.unmap()
                if name == 'Inspection':
                    from .sensor_views import colorize, decode_labels
                    self.sensor_raw, self.sensor_raw_name = images[name], source
                    self.sensor_raw_camera = self.pending_sensor_camera
                    if source == 'SemanticSegmentation':
                        mapping = frame.render_vars[path+'/SemanticIdMap'].map(device=self.ovrtx.Device.CPU)
                        try:
                            self.sensor_labels = decode_labels(np.from_dlpack(mapping).copy())
                        finally:
                            mapping.unmap()
                    images[name] = colorize(source, images[name], self.sensor_labels)
            self.frames+=1
            self.render_time += 1/60
            self.last_render_ms = (time.perf_counter()-self.render_started)*1000
            return images
        finally:
            for mapped, _ in self.pending_mappings.values():
                mapped.unmap()
            self.pending_mappings.clear()
            self.pending_outputs = None
            self.pending_render = self.pending_names = None
            del products

    def set_render_paused(self, paused):
        """Exclude time spent in the separate arcade renderer from the watchdog."""
        now = time.perf_counter()
        if paused and self.render_paused_at is None:
            self.render_paused_at = now
        elif not paused and self.render_paused_at is not None:
            if self.pending_render is not None:
                self.render_started += max(0., now-max(self.render_started, self.render_paused_at))
            self.render_paused_at = None

    def _check_render_timeout(self):
        # Check only after a native wait reports unfinished work. Restoring a
        # minimized window must still accept an image completed while UI slept.
        first_frame = self.frames == 0
        timeout = self.FIRST_FRAME_TIMEOUT_SECONDS if first_frame else self.FRAME_TIMEOUT_SECONDS
        now = self.render_paused_at if self.render_paused_at is not None else time.perf_counter()
        if now-self.render_started > timeout:
            if first_frame:
                raise TimeoutError('RTX did not finish its first image within ten minutes. First-launch shader preparation can be slow; restart to reuse completed shader work and check the runtime logs if this repeats.')
            raise TimeoutError('RTX did not finish a frame within three minutes. Restart the lab and try Performance quality.')

    def read(self, paths, attribute):
        """Read authoritative values from ovstage (used by integration checks)."""
        self.flush()
        read = self.stage.read_attributes(self.query(paths), [self.paths.intern_token(attribute)], self.ovstage.OrdinalRange.latest(self.ordinal))
        read.wait(timeout=5_000_000_000)
        values = {}
        try:
            for group in read.groups(timeout=5_000_000_000):
                try:
                    names = self.paths.get_path_strings(group.prim_list)
                    # Lane-canonical tensors expose each matrix as 16 scalars.
                    # Reconstitute one row per logical prim before pairing paths.
                    data = group.array(0).copy().reshape(len(names),-1)
                    values.update(zip(names, data))
                finally:
                    self.stage.release_group(group)
        finally:
            read.release().wait()
        return values

    def prepare_physics(self):
        """Pay library/tensor initialization once, while the gallery is loading."""
        if self.physx is not None:
            return
        self.ovphysx.PhysX.set_cpu_mode(True)
        self.physx=self.ovphysx.PhysX()
        import warp
        warp.init()
        self.physics_source()

    def physics_source(self):
        """A live masked view of the physical scene, without visual siblings.

        Masked stages retain ancestor properties: selecting a carton collider
        also retains its parent's rigid body, mass and transform. Selecting the
        body itself would unnecessarily pull in the detailed visual shell.
        """
        from pxr import Usd, UsdPhysics
        source = getattr(self, '_physics_source_stage', None)
        if source is None:
            paths = [prim.GetPath() for prim in self.usd.Traverse()
                     if (prim.IsA(UsdPhysics.Scene) or prim.HasAPI(UsdPhysics.RigidBodyAPI)
                         or prim.HasAPI(UsdPhysics.CollisionAPI) or prim.IsA(UsdPhysics.Joint))]
            mask = Usd.StagePopulationMask()
            for path in sorted(paths, key=lambda p: p.pathElementCount, reverse=True):
                if not any(child.HasPrefix(path) for child in mask.GetPaths()):
                    mask.Add(path)
            # Both stages read the same live layer: gravity, poses and friction
            # edits remain current. The notice handler invalidates structural edits.
            source = Usd.Stage.OpenMasked(self.usd.GetRootLayer(), mask)
            self._physics_source_stage = source
        # Follow newly changed material/joint relationships as well.
        source.ExpandPopulationMask()
        return source

    def start_physics(self, dynamic_paths):
        from pxr import UsdGeom
        self.prepare_physics()
        source = self.physics_source()
        self.physics_stage=self.ovstage.Stage('ov-library-lab-physics')
        self.physics_paths=self.ovstage.PathDictionary(self.physics_stage)
        self.ovstage.population.open_usd_from_string(self.physics_stage,source.Flatten().ExportToString(),
            ordinal=1,domains=self.ovstage.PopulationDomain.ALL)
        self.physics_stage.advance_write_floor(1).wait()
        self.physx.attach_ovstage(self.physics_stage,read_ordinal=1)
        self.original_poses={p: [(op.GetOpName(),op.Get()) for op in UsdGeom.Xformable(self.usd.GetPrimAtPath(p)).GetOrderedXformOps()] for p in dynamic_paths}

    def read_bodies(self, attribute):
        from ovphysx.types import SimObjectType, ObjectScope
        result={}
        with self.physx.read(SimObjectType.RIGID_BODY,[attribute],scope=ObjectScope.ALL) as values:
            for group in values.groups:
                if group.tensors:
                    paths=self.physics_paths.get_path_strings(group.prim_list)
                    data=group.tensors[0].numpy().copy()
                    result.update(zip(paths,data))
        return result

    def step_physics(self, count=1):
        from pxr import Gf, UsdGeom, Sdf
        self.physx.step_n_sync(count,1/120)
        self.physics_steps+=count
        positions=self.read_bodies('position')
        orientations=self.read_bodies('orientation')
        if not positions:
            raise RuntimeError('ovphysx returned no rigid-body poses')
        self.pose_reads=positions
        with Sdf.ChangeBlock():
            for path,position in positions.items():
                q=orientations[path]
                matrix=Gf.Matrix4d().SetRotate(Gf.Quatd(float(q[3]),Gf.Vec3d(*map(float,q[:3]))))
                matrix.SetTranslateOnly(Gf.Vec3d(*map(float,position)))
                prim=self.usd.GetPrimAtPath(path)
                attr=prim.GetAttribute('xformOp:transform') or UsdGeom.Xformable(prim).AddTransformOp().GetAttr()
                attr.Set(matrix)
                prim.GetAttribute('xformOpOrder').Set(['xformOp:transform'])

    def pulse(self, selected, velocity):
        from ovphysx.types import SimObjectType
        current=self.read_bodies('linearVelocity')
        with self.physx.write(SimObjectType.RIGID_BODY,'linearVelocity') as session:
            for group in session.groups:
                names=self.physics_paths.get_path_strings(group.prim_list)
                for tensor in group.tensors:
                    data=np.asarray([current[p] for p in names],dtype=np.float32)
                    for i,path in enumerate(names):
                        if path in selected:
                            data[i,2]+=velocity
                    tensor.assign(data)
                session.commit(group)
        self.physx.wait_all()

    def reset_physics(self):
        from pxr import Sdf
        if self.physics_stage is not None:
            self.physx.detach_ovstage()
            self.physics_paths.destroy()
            self.physics_stage.destroy()
            self.physics_paths=self.physics_stage=None
        with Sdf.ChangeBlock():
            for path,ops in self.original_poses.items():
                prim=self.usd.GetPrimAtPath(path)
                for name,value in ops:
                    prim.GetAttribute(name).Set(value)
                prim.GetAttribute('xformOpOrder').Set([name for name,_ in ops])
        self.original_poses={}

    def close(self):
        while self.pending_render is not None:
            self.poll_render(1_000_000)
        self.reset_physics()
        if self.physx is not None:
            self.physx.destroy()
            self.physx = None
        if self.notice:
            self.notice.Revoke()
        for handle,query in self.queries.values():
            query.release().wait()
            self.paths.destroy_path_list(handle)
        self.paths.destroy()
        self.renderer.detach_ovstage()
        self.stage.destroy()
        self.renderer.destroy()

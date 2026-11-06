# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gallery with procedural equipment and a bundled, attributed robot reference."""
import math
from pathlib import Path

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

from .content import ROOMS
from .extensions import call, value
from .campus_assets import CAMPUS_ENABLED, CAMPUS_HIDDEN_ROOMS
from .layout import (PLACEMENTS, world, ATRIUM_WIDTH, ATRIUM_DEPTH, HALF_WIDTH,
                     FRONT_Z, REAR_Z, ATRIUM_CENTER_Z, EMPTY_BAYS, BAY_WIDTH, WALK_FOCAL_LENGTH)


class Gallery:
    def __init__(self, stage, data_dir):
        self.stage = stage
        self.data_dir = Path(data_dir)
        self.obstacles = []
        self.active_room = None
        self.dynamic_paths = []
        self.static_paths = []
        self.initial_positions = {}
        self.shared_offset = 0.0
        self.shared_speed = 1.2
        self.revision = 0
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        stage.SetTimeCodesPerSecond(60)
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        self.materials = {}
        for name, color, metal, rough in [
            ("floor", (0.34, 0.36, 0.38), .0, .42),
            ("grout", (0.035, 0.045, 0.055), .0, .7),
            ("wall", (0.57, 0.60, 0.61), .0, .72),
            ("ivory", (0.73, 0.74, 0.70), .0, .38),
            ("black", (0.028, 0.035, 0.043), .0, .38),
            ("green", (.44, .80, .12), .25, .26),
            ("blue", (.035, .19, .32), .0, .31),
            ("purple", (.46, .22, .78), .55, .26),
            ("gold", (.83, .49, .15), 1.0, .24),
            ("silver", (.66, .68, .70), 1.0, .24),
            ("frozen", (.22, .28, .32), .0, .55),
        ]:
            self.materials[name] = self.material(name, color, metal, rough)
        for key, room in ROOMS.items():
            self.materials[key + "_glow"] = self.material(key+"_glow", room["color"], .0, .5, .65)
        from .factory import FactoryAssets
        self.factory = FactoryAssets(self)
        from .surfaces import install
        install(self)
        self.campus = CAMPUS_ENABLED
        self.campus_blocker = None
        if self.campus:
            from .campus import build
            build(self)
        else:
            self._architecture()
            from .architecture_detail import build as build_details
            build_details(self)
            from .atrium import build as build_atrium
            build_atrium(self)
        self.active_room = "physics"
        self._physics()
        from .physics_inspection import build as build_physics_inspection
        build_physics_inspection(self)
        self.active_room = "rendering"
        self._rendering()
        self.active_room = "scene"
        self._shared_scene()
        from .exhibits import ExtraExhibits
        self.extra = ExtraExhibits(self)
        from .workshops import Workshops
        self.workshops = Workshops(self)
        call('build_scene', self)
        from .newton_room import NewtonRoom
        from .package_room import PackageRoom
        self.newton_room = NewtonRoom(self, build='newton' not in CAMPUS_HIDDEN_ROOMS)
        self.package_room = PackageRoom(self, build='package' not in CAMPUS_HIDDEN_ROOMS)
        self.active_room = None
        self.camera = self.make_camera("/World/TourCamera", (0, 2.1, 13), (0, 2, -4), WALK_FOCAL_LENGTH)
        self.observer = self.make_camera("/World/ObserverCamera", self.room_position("scene", (15.1, 4.1, -2.5)), self.room_position("scene", (11, .7, -5.7)), 22)

    def room_position(self, key, pos):
        x, z = world(key, pos[0] - PLACEMENTS[key][3], pos[2] - 1.2)
        return x, pos[1], z

    def _position(self, pos):
        return self.room_position(self.active_room, pos) if self.active_room else pos

    def material(self, name, color, metallic=0., roughness=.4, emission=0.):
        material = UsdShade.Material.Define(self.stage, "/World/Looks/" + name)
        shader = UsdShade.Shader.Define(self.stage, material.GetPath().AppendChild("Surface"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        if emission:
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color) * emission)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    def bind(self, prim, mat):
        binding=UsdShade.MaterialBindingAPI(prim) if prim.HasAPI(UsdShade.MaterialBindingAPI) else UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(self.materials[mat] if isinstance(mat, str) else mat)

    def box(self, path, pos, size, mat="wall", collider=False, obstacle=False):
        pos = self._position(pos)
        if self.active_room and abs(math.sin(PLACEMENTS[self.active_room][2])) > .5:
            size = size[2], size[1], size[0]
        from .surfaces import box_mesh
        radius = min(.012, min(size)*.08) if min(size)>.025 else 0.
        cube = box_mesh(self.stage,path,size,radius,2. if mat=='floor' else 1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(*pos))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        self.bind(cube.GetPrim(), mat)
        if collider:
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(cube.GetPrim()).CreateApproximationAttr('convexHull')
        if obstacle:
            x, _, z = pos
            sx, _, sz = size
            self.obstacles.append((x-sx/2, x+sx/2, z-sz/2, z+sz/2))
        return cube

    def sphere(self, path, pos, radius, mat):
        pos = self._position(pos)
        sphere = UsdGeom.Sphere.Define(self.stage, path)
        sphere.GetPrim().CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool).Set(True)
        sphere.GetPrim().CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int).Set(3)
        sphere.CreateRadiusAttr(radius)
        sphere.AddTranslateOp().Set(Gf.Vec3d(*pos))
        self.bind(sphere.GetPrim(), mat)
        return sphere

    def cylinder(self, path, pos, radius, height, mat, axis='Y'):
        pos = self._position(pos)
        from .surfaces import cylinder_mesh
        segments = max(16, min(96, math.ceil(32*math.sqrt(radius/.08)/8)*8))
        c = cylinder_mesh(self.stage,path,radius,height,axis,segments=segments)
        c.AddTranslateOp().Set(Gf.Vec3d(*pos))
        self.bind(c.GetPrim(), mat)
        return c

    def sign(self, name, pos, width, height, texture):
        pos = self._position(pos)
        path = "/World/Signs/" + name
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreatePointsAttr([(-width/2, -height/2, 0), (width/2, -height/2, 0),
                               (width/2, height/2, 0), (-width/2, height/2, 0)])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        mesh.AddTranslateOp().Set(Gf.Vec3d(*pos))
        if self.active_room:
            mesh.AddRotateYOp().Set(-math.degrees(PLACEMENTS[self.active_room][2]))
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(
            [(0, 0), (1, 0), (1, 1), (0, 1)])
        mat = UsdShade.Material.Define(self.stage, "/World/Looks/sign_" + name)
        surface = UsdShade.Shader.Define(self.stage, mat.GetPath().AppendChild("Surface"))
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.)
        uv = UsdShade.Shader.Define(self.stage, mat.GetPath().AppendChild("UV"))
        uv.CreateIdAttr("UsdPrimvarReader_float2")
        uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(self.stage, mat.GetPath().AppendChild("Texture"))
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(self.data_dir / texture))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
        surface.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
        if self.campus:
            surface.GetInput('diffuseColor').ConnectToSource(tex.ConnectableAPI(), 'rgb')
        mat.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
        self.bind(mesh.GetPrim(), mat)
        return mesh

    def rect_light(self, name, pos, target, color, intensity, width=4., height=2.):
        pos, target = self._position(pos), self._position(target)
        light = UsdLux.RectLight.Define(self.stage, "/World/Lights/" + name)
        light.CreateWidthAttr(width)
        light.CreateHeightAttr(height)
        light.CreateIntensityAttr(intensity)
        light.CreateColorAttr(Gf.Vec3f(*color))
        matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*pos), Gf.Vec3d(*target), Gf.Vec3d(0, 0, -1) if pos[0] == target[0] and pos[2] == target[2] else Gf.Vec3d(0, 1, 0)).GetInverse()
        light.AddTransformOp().Set(matrix)
        return light

    def placard(self, name, pos, width, height, texture, floor_y=0., obstacle=False):
        """A freestanding, upright label placed beside the exhibit sightline."""
        # The campus uses architectural plaques; explanations are in the UI.
        # Do not leave invisible stands or walking blockers behind.
        if self.campus:
            return None
        x,y,z=pos
        p='/World/Placards/'+name
        self.box(p+'/Back',(x,y,z-.035),(width+.08,height+.08,.06),'black',obstacle=obstacle)
        bottom=y-height/2
        self.box(p+'/Stem',(x,(bottom+floor_y)/2,z-.065),(.06,bottom-floor_y,.06),'steel')
        self.box(p+'/Foot',(x,floor_y+.025,z-.065),(.50,.05,.34),'black')
        return self.sign(name,pos,width,height,texture)

    def _architecture(self):
        self.box("/World/Building/AtriumFloor", (0, -.2, ATRIUM_CENTER_Z), (ATRIUM_WIDTH, .4, ATRIUM_DEPTH), "grout", collider=True)
        for i in range(round(ATRIUM_WIDTH/2.75)):
            for j in range(8):
                self.box(f"/World/Building/Tiles/t_{i}_{j}", (-HALF_WIDTH+1.375+i*2.75, .006, FRONT_Z+1.375+j*2.75), (2.738, .018, 2.738), "floor")
        self.box("/World/Building/AtriumRoof", (0, 7.5, ATRIUM_CENTER_Z), (ATRIUM_WIDTH, .18, ATRIUM_DEPTH), "wall")
        self.box("/World/Building/RearFascia", (0, 6.35, REAR_Z+.05), (ATRIUM_WIDTH, 2.3, .2), "wall")
        self.box("/World/Building/AtriumFascia", (0, 6.35, FRONT_Z), (ATRIUM_WIDTH, 2.3, .3), "wall")
        self.sign("Welcome", (0, 6.3, -9.8), 19.0, 2.0, "welcome.png")
        for i, (x, z) in enumerate(EMPTY_BAYS):
            self.box(f'/World/Building/ExpansionPier{i}', (x, 2.6, z), (BAY_WIDTH, 5.2, .22), 'wall', obstacle=True)
        for key, room in ROOMS.items():
            self.active_room = key
            x, glow = room["center"], key+"_glow"
            prefix = "/World/Building/" + key
            self.box(prefix+"/Floor", (x, -.2, -4.4), (11, .4, 11.2), "floor", collider=True)
            self.box(prefix+"/Back", (x, 2.6, -10), (11, 5.2, .22), obstacle=True)
            self.box(prefix+"/Roof", (x, 5.35, -4.4), (11, .18, 11.2), "wall")
            for side in (-1, 1):
                suffix = "L" if side < 0 else "R"
                self.box(prefix+"/Wall"+suffix, (x+side*5.5, 2.6, -4.4), (.22, 5.2, 11.2), obstacle=True)
                self.box(prefix+"/Front"+suffix, (x+side*4.3, 2.5, 1.2), (2.4, 5, .3), obstacle=True)
                self.box(prefix+"/Trim"+suffix, (x+side*3.08, 1.9, 1.40), (.055, 3.8, .055), glow)
            self.box(prefix+"/Lintel", (x, 4.5, 1.2), (6.3, 1.0, .3))
            self.box(prefix+"/TopTrim", (x, 3.9, 1.4), (6.2, .055, .055), glow)
            self.box(prefix+"/Threshold", (x, .014, 1.3), (6.2, .02, .12), glow)
            self.box(prefix+"/Runner", (x, .012, 3.2), (.05, .02, 4.), glow)
            for k in range(3):
                self.box(prefix+f"/Waypoint{k}", (x, .015, 2.0+k*.9), (.38, .02, .10), glow)
            self.sign(key+"_entry", (x, 4.45, 1.40), 5.7, .82, key+"_entry.png")
            self.sign(key+"_lesson", (x, 3.95, -9.83), 7.0, 1.55, key+"_lesson.png")
            self.box(prefix+"/BackStripe", (x, .24, -9.80), (10.4, .045, .03), glow)
            for k, z in enumerate((-7.5, -2.0)):
                self.box(prefix+f"/Beam{k}", (x, 5.0, z), (10.8, .22, .24), "black")
                self.box(prefix+f"/BeamLight{k}", (x, 4.86, z), (6, .035, .13), "ivory")
            studio = key == 'rendering'
            industrial = key in ('physics', *value('INDUSTRIAL_ROOMS', ()))
            self.rect_light(key+"_overhead", (x, 4.8, -4.0), (x, 0, -4.0), (1., .97, .92), 2800. if studio else 4800. if industrial else 6500., 6, 4)
            self.rect_light(key+"_fill", (x, 3.7, .5), (x, 1.0, -6.0), (.91, .96, 1.), 1000. if studio else 1800. if industrial else 3200., 5, 2)
            self.rect_light(key+"_wallwash", (x, 4.5, -8.7), (x, 2., -9.8), (1., .97, .92), 1400., 7, .5)
            if studio or industrial:
                self.rect_light(key+'_edge', (x-4.5, 3.6, -7.8), (x, 1.2, -5.2), (.82,.90,1.), 2600., .6, 3.)
        self.active_room = None
        dome = UsdLux.DomeLight.Define(self.stage, "/World/Lights/Ambient")
        dome.CreateIntensityAttr(600.)
        dome.CreateColorAttr(Gf.Vec3f(.90, .95, 1.))
        self.rect_light("Lobby", (0, 7.1, 1), (0, 0, 1), (1., .97, .91), 6500., ATRIUM_WIDTH-6, 12)
        for k, z in enumerate((-7., 0., 7.)):
            self.box(f"/World/Building/LobbyLight{k}", (0, 7.35, z), (ATRIUM_WIDTH-2, .04, .13), "ivory")

    def _physics(self):
        scene = UsdPhysics.Scene.Define(self.stage, "/World/PhysicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, -1, 0))
        self.gravity_attr = scene.CreateGravityMagnitudeAttr(9.81)
        from .physics_schema import apply
        apply(scene.GetPrim(), "PhysxSceneAPI", {"physxScene:solverType": ("token", "TGS"), "physxScene:bounceThreshold": ("float", .25)})
        self.bounce_material = UsdShade.Material.Define(self.stage, "/World/Looks/Bounce")
        pm = UsdPhysics.MaterialAPI.Apply(self.bounce_material.GetPrim())
        pm.CreateRestitutionAttr(.035)
        pm.CreateStaticFrictionAttr(.48)
        pm.CreateDynamicFrictionAttr(.38)
        apply(self.bounce_material.GetPrim(), "PhysxMaterialAPI", {"physxMaterial:restitutionCombineMode": ("token", "min")})
        for side, x in (("Frozen", -13.15), ("Simulated", -8.85)):
            p = "/World/Physics/"+side
            self.factory.conveyor(p+"/ReceivingConveyor", (x, 0, -5.5), 3.65, 3.1, .85, collider=True)
            self.box(p+"/Footprint", (x, .06, -5.5), (3.8, .12, 3.6), "black", obstacle=True)
            self.factory.boundary(p+"/Safety", x, -5.5, 3.9, 3.8)
            for n, dx in enumerate((-1.68, 1.68)):
                self.box(p+f"/DropTower{n}", (x+dx, 1.9, -7.), (.09, 3.8, .13), "steel")
            self.box(p+"/Gantry", (x, 3.8, -7.), (3.5, .12, .2), "safety")
            label_x = x + (.45 if side == 'Frozen' else -.45)
            if self.campus:
                from .campus_signs import physics_label
                physics_label(self,side,(x-.4,.62,-3.68))
            else:
                self.placard(side+"_label", (label_x, .53, -3.12), 2.3, .50, side.lower()+".png", obstacle=True)
            for i in range(6):
                pos = (x + ((i % 3)-1)*.95, 2.25, -6.2+(i//3)*1.4)
                path = p+f"/Body{i}"
                obj = self.factory.carton(path, pos, (.7, .56, .70), dynamic=side == "Simulated")
                obj.AddRotateXYZOp().Set(Gf.Vec3f(*((4, -5, 3), (-3, 7, -5), (6, 2, -3), (-5, -6, 4), (3, 4, 6), (-4, 3, -6))[i]))
                self.initial_positions[path] = pos
                if side == "Simulated":
                    self.dynamic_paths.append(path)
                else:
                    self.static_paths.append(path)
        if not self.campus:
            self.factory.rack("/World/Physics/DispatchRack", (-11, 0, -8.7), 2.5, 2.8, 2)

    def _rendering(self):
        for name, color, metallic in (("inspection_blue", (.045, .24, .48), .05),
                                      ("inspection_orange", (.82, .20, .025), .05),
                                      ("inspection_alloy", (.58, .62, .66), 1.)):
            self.materials[name] = self.material(name, color, metallic, .24)
        self.box("/World/Rendering/Platform", (0, .05, -5.5), (8.7, .1, 3.8), "black", obstacle=True)
        self.factory.boundary("/World/Rendering/Safety", 0, -5.5, 9.1, 4.1)
        for i, (x, mat) in enumerate(((-2.5, "inspection_blue"), (0., "inspection_orange"), (2.5, "inspection_alloy"))):
            p = f"/World/Rendering/Robot{i}"
            with self.factory.assembly(p, (x-.45, .1, -5.5), yaw=-25):
                self.box(p+"/Fixture", (0, .15, 0), (1.05, .30, 1.05), "steel")
                for j,(dx,dz) in enumerate(((-.43,-.43),(-.43,.43),(.43,-.43),(.43,.43))):
                    self.cylinder(p+f'/Anchor{j}',(dx,.314,dz),.023,.028,'black')
                mount = UsdGeom.Xform.Define(self.stage, p+"/DisplayScale")
                mount.AddTranslateOp().Set(Gf.Vec3d(0, .30, 0))
                mount.AddScaleOp().Set(Gf.Vec3f(1.))
                robot = self.stage.DefinePrim(p+"/DisplayScale/IRB4600")
                robot.GetReferences().AddReference(str(self.data_dir/"models/irb4600/irb4600.usdc"))
                self.bind(robot, mat)
                # Preserve the recognizable machined wrist/tool flange.
                wrist = robot.GetPath().AppendPath("base_link/link_1/link_2/link_3/link_4/link_5/link_6/Visual")
                self.bind(self.stage.GetPrimAtPath(wrist), "steel")
            if i != 1:
                self.box(f"/World/Rendering/CameraPost{i}", (x, 1.9, -6.65), (.08, 3.8, .1), "steel")
            self.box(f"/World/Rendering/CameraArm{i}", (x, 3.6, -5.8), (.12, .12, 1.9), "steel")
            self.box(f"/World/Rendering/Camera{i}", (x, 3.38, -4.95), (.34, .36, .38), "black")
            self.cylinder(f"/World/Rendering/Lens{i}", (x, 3.13, -4.95), .115, .16, "silver")
            self.box(f"/World/Rendering/LightBar{i}", (x, 2.98, -4.88), (.9, .055, .18), "rendering_glow")
        self.key_light = self.rect_light("Key", (-2.8, 4, -3.3), (0, 1.7, -5.5), (1., .77, .48), 3500., 2, 2)
        self.rect_light("Rim", (3.6, 3.5, -7.4), (0, 1.8, -5.5), (.78, .88, 1.), 2200., 1, 3)
        if not self.campus:
            self.sign("MaterialLabels", (0, .39, -3.55), 7.3, .62, "materials.png")

    def _shared_scene(self):
        self.box("/World/Shared/Route", (11, .018, -5.7), (8.8, .03, 5.1), "black", obstacle=True)
        self.factory.boundary("/World/Shared/Safety", 11, -5.7, 9., 5.5)
        for i in range(15):
            self.box(f"/World/Shared/RouteDash{i}", (7.15+i*.55, .039, -4.9), (.26, .01, .06), "scene_glow")
        for i, x in enumerate((8.6, 13.4)):
            self.factory.rack(f"/World/Shared/Rack{i}", (x, 0, -9.), 2.8, 3., 2)
            self.factory.bollard(f"/World/Shared/Bollard{i}", (x, 0, -7.8 if self.factory.simready.assets else -7.))
        self.shared_floor_y = .033
        self.shared = self.factory.amr("/World/Shared/AMR", (8.6, self.shared_floor_y, -5.7))
        self.shared_translate = self.shared.GetPrim().GetAttribute("xformOp:translate")
        self.shared_load = UsdGeom.Xform(self.stage.GetPrimAtPath("/World/Shared/AMR/Load"))
        self.shared_offset = self.shared_target = -2.4
        self.shared_loaded = True
        self.placard("SharedLabel", (7.3, 1.05, -5.4), 2., .70, "shared.png", floor_y=self.shared_floor_y)

    def set_roughness(self, value):
        for name in ("inspection_blue", "inspection_orange", "inspection_alloy"):
            self.stage.GetPrimAtPath("/World/Looks/"+name+"/Surface").GetAttribute("inputs:roughness").Set(float(value))

    def set_shared(self, offset=None, toggle_load=False):
        if offset is not None:
            self.shared_target = float(offset)
        if toggle_load:
            self.shared_loaded = not self.shared_loaded
            (self.shared_load.MakeVisible if self.shared_loaded else self.shared_load.MakeInvisible)()
        self.revision += 1

    def advance_shared(self, dt, shuttle=False):
        difference = self.shared_target-self.shared_offset
        if abs(difference) < .001 and shuttle:
            self.shared_target = -2.4 if self.shared_target > 0 else 2.4
            difference = self.shared_target-self.shared_offset
        if difference == 0 or dt == 0:
            return
        step = math.copysign(min(abs(difference), dt*self.shared_speed), difference)
        self.shared_offset += step
        self.shared_translate.Set(Gf.Vec3d(*self.room_position("scene", (11+self.shared_offset, self.shared_floor_y, -5.7))))
        if hasattr(self, 'workshops'):
            self.workshops.update_fleet()

    def position(self, path):
        return UsdGeom.Xformable(self.stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0).ExtractTranslation()

    def make_camera(self, path, eye, target, focal):
        camera = UsdGeom.Camera.Define(self.stage, path)
        if self.campus:
            from .campus import expose
            expose(camera.GetPrim())
        camera.CreateFocalLengthAttr(focal)
        camera.CreateHorizontalApertureAttr(36.)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.05, 500))
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 1, 0)).GetInverse())
        return camera

    def set_camera(self, x, z, yaw, pitch, height=1.85):
        pose = (x, z, yaw, pitch, height)
        if getattr(self, '_camera_pose', None) == pose:
            return
        self._camera_pose = pose
        eye = Gf.Vec3d(x, height, z)
        direction = Gf.Vec3d(math.sin(yaw)*math.cos(pitch), math.sin(pitch), -math.cos(yaw)*math.cos(pitch))
        self.camera.GetPrim().GetAttribute("xformOp:transform").Set(
            Gf.Matrix4d().SetLookAt(eye, eye+direction, Gf.Vec3d(0, 1, 0)).GetInverse())

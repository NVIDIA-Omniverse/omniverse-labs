# SPDX-License-Identifier: Apache-2.0
"""Author a disposable ordinary foreground scene using the enabled extension.

This helper creates content only. It never registers an addon or drives a solver.
"""
import importlib
from mathutils import Matrix, Vector


def build(bpy, addon):
    initialize_owner = importlib.import_module(addon.__name__ + '.physics_endpoints').initialize_owner
    # A new scene avoids inheriting units, world nodes or simulation settings
    # from the disposable file used to enter the normal project frontend.
    scene = bpy.data.scenes.new('Foreground physics regression')
    bpy.context.window.scene = scene
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1
    scene.world = bpy.data.worlds.new('Foreground neutral world')
    scene.world.color = (.05, .05, .05)
    scene.frame_start, scene.frame_end = 1, 241
    scene.frame_set(1)
    scene.render.resolution_x, scene.render.resolution_y = 640, 480
    scene.render.resolution_percentage = 100
    scene.render.fps, scene.render.fps_base = 30, 1.0
    scene.ovrtx_example.min_samples = scene.ovrtx_example.max_samples = 16

    def collider(obj, identity):
        initialize_owner(obj)
        record = obj.ov.physics.colliders.add()
        record.name = record.persistent_id = identity
        record.schema_opinion, record.shape = 'APPLY', 'CONVEX_HULL'

    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 3))
    cube = bpy.context.object
    cube.name = 'SyncCube'
    cube.data.transform(Matrix.Diagonal((.5, .75, .25, 1)))
    cube.rotation_euler.z = .3
    cube.ov.physics.rigid_body.schema_opinion = 'APPLY'
    cube.ov.physics.rigid_body.mass_kg = 1
    collider(cube, '11111111111111111111111111111111')
    material = bpy.data.materials.new('Foreground red marker')
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (1, 0, 0, 1)
    bsdf.inputs['Emission Color'].default_value = (1, 0, 0, 1)
    bsdf.inputs['Emission Strength'].default_value = 1
    cube.data.materials.append(material)
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, -.5))
    floor = bpy.context.object
    floor.name = 'Floor'
    floor.scale = (5, 5, .5)
    collider(floor, '22222222222222222222222222222222')
    scene.ov.physics_scene.schema_opinion = 'APPLY'
    scene.ov.physics_scene.gravity_direction = (0, 0, -1)
    scene.ov.physics_scene.gravity_magnitude = 9.81
    bpy.ops.object.light_add(type='SUN')
    bpy.context.object.data.energy = 2
    bpy.ops.object.camera_add(location=(7, -7, 5))
    camera = bpy.context.object
    camera.rotation_euler = (Vector((0, 0, 1.5)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
    scene.camera = camera
    bpy.context.view_layer.update()
    return {'bodies': {'SyncCube': '/World/SyncCube'}, 'frame_range': [1, 241],
            'initial_world': [list(row) for row in cube.matrix_world],
            'half_extents': [.5, .75, .25], 'gravity': 9.81}

# SPDX-License-Identifier: Apache-2.0
"""Deterministic procedural case catalog; no Blender import required."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Case:
    id: str
    family: str
    variant: str
    observable: str
    edits: tuple[str, ...]
    classification: str = "probe"
    reference: str = ""


def cases():
    result = []
    def add(family, variants, observable, edits, classification="probe", reference=""):
        for variant in variants.split():
            result.append(Case(f"{family}.{variant}", family, variant, observable,
                               tuple(edits), classification, reference))
    add("mesh", "cube sphere torus ngon loose_edges negative_scale nonuniform_parent multi_material",
        "Evaluated and exported geometry, finite bounds, material subsets; silhouette changes after edit",
        ["translate", "topology", "delete_add"], reference="tests/blender_single_compiler.py")
    add("modifier", "bevel subdivision mirror array solidify boolean triangulate decimate weld displacement",
        "Evaluated modifier result survives USD with changed point or polygon counts",
        ["modifier", "translate"])
    add("modifier_stack", "array_bevel subdivision_displacement mirror_solidify",
        "Ordered modifier stacks evaluate/export correctly before and after stack edits",["modifier","translate"])
    add("instances", "linked_mesh collection nested_collection",
        "Instance count and world transforms match authored placement, no missing copies",
        ["instance_count", "translate"])
    add("geometry_nodes", "transform grid instances realized_instances named_attribute material_assignment curve_to_mesh points_to_vertices animated_field",
        "Node graph evaluates nonempty geometry/instances; edited inputs alter evaluated/exported output",
        ["node_input", "frame"], reference="skills/geometry-nodes-for-ovrtx/SKILL.md")
    add("curve", "bezier poly nurbs hair_curves text",
        "Curve points/radii or evaluated surface survive export; thickness edit changes silhouette",
        ["curve_radius", "translate"])
    add("material", "base_color roughness metallic alpha transmission ior specular anisotropy coat coat_roughness coat_ior coat_tint subsurface subsurface_radius sheen sheen_tint emission emission_strength thin_film diffuse glass transparent principled_hair mix_shader add_shader",
        "Material converter diagnostics plus native color/BRDF change; authored input alone is insufficient",
        ["material_value", "material_graph"], reference="tests/test_materialx_openpbr_conversion.py")
    add("texture", "image_srgb image_linear uv_mapping uv_second_set udim normal_map coat_normal bump displacement roughness metallic alpha emission noise checker gradient color_ramp mix_color packed_image",
        "Texture dependencies resolve, UV pattern/normal response survives conversion and edits",
        ["texture_mapping", "material_graph"], reference="tests/test_texture_materialization.py")
    add("primvar", "point_color corner_color face_float point_float uv_edit normal_edit material_index",
        "Named attribute values and interpolation survive USD and changes; pixels must respond where consumed",
        ["primvar_value", "topology"])
    add("openpbr_direct", "base_diffuse_roughness specular_weight specular_color specular_roughness_anisotropy specular_ior transmission_depth transmission_scatter transmission_scatter_anisotropy transmission_dispersion_scale transmission_dispersion_abbe_number subsurface_weight subsurface_radius subsurface_radius_scale subsurface_scatter_anisotropy coat_weight coat_color coat_roughness_anisotropy coat_ior coat_darkening fuzz_weight fuzz_color fuzz_roughness emission_luminance thin_film_weight thin_film_thickness thin_film_ior geometry_opacity geometry_thin_walled geometry_coat_normal coated_metal frosted_glass pearlescent_coat",
        "Direct USD OpenPBR parameter/combination affects native pixels; distinct from Blender Principled conversion support",
        ["openpbr_value"], "direct_usd", "https://academysoftwarefoundation.github.io/OpenPBR/")
    add("light", "point spot sun area_square area_rectangle area_disk area_ellipse temperature",
        "USD light schema/energy/color and changed image illumination agree",
        ["light_energy", "light_color", "translate"], reference="tests/test_light_value_conversion.py")
    add("world", "background_strength background_color procedural_sky environment_image hdri_studio",
        "World/dome authoring and image illumination/rotation change; no background-only false pass",
        ["world_value"], reference="tests/test_world_dome_usd_prim.py")
    add("camera", "perspective orthographic focal_length shift clipping depth_of_field panoramic",
        "Camera projection/framing/focus measured in native pixels with unchanged renderer identity",
        ["camera_value", "translate"], reference="scripts/run_ovrtx_orthographic_camera_probe.py")
    add("animation", "object_transform shape_key armature constraint driver material_keyframe topology",
        "Frames1/12/24 evaluate different expected poses/values; native output tracks each frame",
        ["frame"], reference="tests/test_frame_range.py")
    add("visibility", "object_hidden collection_hidden view_layer_exclude camera_ray holdout",
        "Expected subject disappears while control object remains; no whole-frame empty false pass",
        ["visibility"], reference="tests/test_render_visibility.py")
    add("physics", "dynamic static kinematic disabled convex_hull analytic_sphere analytic_box capsule joint_fixed joint_revolute joint_prismatic joint_d6",
        "Authored Physics schemas plus existing native motion/contact/joint constraints; frames alone not a physics pass",
        ["physics_value"], reference="tests/blender_native_joints.py")
    add("volume", "mesh_to_volume volume_to_mesh", "Actual evaluated volume/grid or isosurface export, density/shape edits",["volume_value"])
    add("render", "ldr hdr exposure agx standard transparent_film",
        "Correct dimensions/origin/color mode, finite exact HDR; presentation edit preserves scene linear values",
        ["presentation", "resize"], reference="tests/blender_presentation_exact_channels.py")
    add("challenge", "volume_shader geometry_simulation_zone geometry_bake_node grease_pencil compositor shader_to_rgb osl_script",
        "Record actual author/export/conversion/native diagnostic; never infer support from creation",
        ["frame"], "challenge")
    return tuple(result)


CASES = cases()
BY_ID = {case.id: case for case in CASES}
SMOKE = ("mesh.cube", "modifier.bevel", "instances.collection", "geometry_nodes.realized_instances",
         "material.base_color", "texture.image_srgb", "light.spot", "camera.orthographic",
         "animation.shape_key", "physics.joint_fixed", "curve.bezier", "challenge.volume_shader")


def manifest():
    return {"schema": 1, "seed": 3317, "cases": [asdict(case) for case in CASES],
            "states": ["not_run", "authoring_passed", "export_passed", "conversion_rejected",
                       "native_unverified", "native_passed", "failed", "blocked"],
            "budgets": {"downloads_bytes": 1073741824, "generated_bytes": 1073741824,
                        "baseline_resolution": [512,384], "maximum_texture_dimension": 2048},
            "smoke": SMOKE}

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import materialx_openpbr_conversion  # noqa: E402
from ovrtx_blender_example import usd_preview_emission_layer  # noqa: E402
from ovrtx_blender_example.sky_texture_radiance import sky_texture_radiance  # noqa: E402
from ovrtx_blender_example.materialx_openpbr_conversion import (  # noqa: E402
    MaterialSceneConversionResult,
    MaterialSceneConversionStatus,
    scene_layer_from_materials,
)


@dataclass(frozen=True)
class _Binding:
    material: object
    binding_targets: tuple[str, ...]


def _conversion_result(materials, identity):
    input_usd_path = "test.usda"
    materialx_openpbr_conversion._MATERIALX_BINDING_IDENTITY_CACHE.clear()
    materialx_openpbr_conversion._MATERIALX_BINDING_IDENTITY_CACHE[input_usd_path] = identity
    return scene_layer_from_materials(materials, input_usd_path)


def test_preview_emission_layer_uses_portable_asset_separators() -> None:
    layer = usd_preview_emission_layer._layer_from_records(
        (
            usd_preview_emission_layer._EmissionRecord(
                "Light",
                "/World/Materials/Light",
                "/World/Materials/Light/PreviewSurface",
                r"C:\tmp\textures\light.png",
            ),
        )
    )

    assert layer is not None
    assert "asset inputs:file = @C:/tmp/textures/light.png@" in layer.layer_body


def _convert_bindings(bindings):
    material_paths = []
    binding_records = []
    materials = []
    for binding in bindings:
        material = binding.material
        materials.append(material)
        material_path = material.get("ovrtx:sourceUsdPath", "")
        if not material_path:
            material_path = "/World/Looks/" + str(material.name).replace(" ", "_")
        material_paths.append(material_path)
        binding_records.extend(
            {"material_path": material_path, "binding_target": target}
            for target in binding.binding_targets
        )
    return _conversion_result(
        materials,
        {
            "available": True,
            "reason": "",
            "material_paths": tuple(material_paths),
            "bindings": tuple(binding_records),
        },
    )


class _Inputs:
    def __init__(self, sockets):
        self._sockets = list(sockets)
        self._by_name = {socket.name: socket for socket in self._sockets}

    def __iter__(self):
        return iter(self._sockets)

    def __getitem__(self, name):
        return self._by_name[name]

    def __setitem__(self, name, socket):
        self._by_name[name] = socket
        for index, existing in enumerate(self._sockets):
            if existing.name == name:
                self._sockets[index] = socket
                return
        self._sockets.append(socket)

    def get(self, name, default=None):
        return self._by_name.get(name, default)

    def values(self):
        return list(self._sockets)


class _Socket:
    def __init__(self, name: str, default_value=None, *, socket_type: str = "VALUE") -> None:
        self.name = name
        self.default_value = default_value
        self.type = socket_type
        self.links = []

    @property
    def is_linked(self) -> bool:
        return bool(self.links)


class _BlenderStyleUnlinkedSocket:
    is_linked = False

    @property
    def links(self):
        raise AssertionError("unlinked Blender sockets must not enumerate node-tree links")


class _BlenderStyleLinkedSocket:
    is_linked = True

    def __init__(self, name: str, pointer: int) -> None:
        self.name = name
        self.default_value = None
        self.type = "SHADER"
        self._pointer = pointer

    def as_pointer(self) -> int:
        return self._pointer

    @property
    def links(self):
        raise AssertionError("linked Blender sockets must use the per-node-tree link lookup")


class _Node:
    def __init__(
        self,
        node_type: str,
        name: str,
        inputs: list[_Socket],
        *,
        image=None,
        outputs: list[_Socket] | None = None,
    ) -> None:
        self.type = node_type
        self.name = name
        self.inputs = _Inputs(inputs)
        self.outputs = _Inputs(outputs or [])
        self.image = image
        self.is_active_output = True


class _Material:
    def __init__(self, name: str, nodes: list[_Node], *, source_usd_path: str = "") -> None:
        self.name = name
        self.name_full = name
        self.node_tree = SimpleNamespace(nodes=nodes)
        self._source_usd_path = source_usd_path
        self.custom_properties = {}

    def get(self, key: str, default=None):
        if key == "ovrtx:sourceUsdPath":
            return self._source_usd_path
        return self.custom_properties.get(key, default)


def _principled_material(
    *,
    name: str = "Paint",
    base_color=(0.2, 0.4, 0.6, 1.0),
    roughness=0.35,
    metallic=0.75,
    alpha=0.5,
    extra_principled_inputs: list[_Socket] | None = None,
    extra_nodes: list[_Node] | None = None,
) -> _Material:
    principled = _Node(
        "BSDF_PRINCIPLED",
        "Principled BSDF",
        [
            _Socket("Base Color", base_color),
            _Socket("Roughness", roughness),
            _Socket("Metallic", metallic),
            _Socket("Alpha", alpha),
            *(extra_principled_inputs or []),
        ],
    )
    output = _Node("OUTPUT_MATERIAL", "Material Output", [_Socket("Surface")])
    output.inputs["Surface"].links.append(SimpleNamespace(from_node=principled))
    return _Material(
        name,
        [principled, output, *(extra_nodes or [])],
        source_usd_path="/World/Looks/" + name.replace(" ", "_"),
    )


def _link(socket: _Socket, node: _Node, *, from_socket_name: str = "") -> _Socket:
    from_socket = SimpleNamespace(name=from_socket_name) if from_socket_name else None
    socket.links.append(SimpleNamespace(from_node=node, from_socket=from_socket))
    return socket


def _texture_node(name: str, path: Path, *, colorspace: str = "sRGB") -> _Node:
    path.write_bytes(b"fake")
    image = SimpleNamespace(
        name=path.name,
        filepath=str(path),
        filepath_raw=str(path),
        colorspace_settings=SimpleNamespace(name=colorspace),
        has_data=False,
        packed_file=None,
    )
    return _Node("TEX_IMAGE", name, [_Socket("Vector")], image=image)


def test_unlinked_blender_socket_does_not_enumerate_node_tree_links() -> None:
    socket = _BlenderStyleUnlinkedSocket()

    assert materialx_openpbr_conversion._socket_linked(socket) is False
    assert materialx_openpbr_conversion._socket_link_target_node(socket) is None


def test_material_classification_enumerates_node_tree_links_once() -> None:
    material = _principled_material()
    principled = material.node_tree.nodes[0]
    output = material.node_tree.nodes[1]
    surface = _BlenderStyleLinkedSocket("Surface", 42)
    output.inputs["Surface"] = surface
    material.node_tree.links = [SimpleNamespace(to_socket=surface, from_node=principled)]

    result = materialx_openpbr_conversion._classify_material(
        material,
        ("/World/Cube",),
        (),
        {},
    )

    assert result["status"] == "generated"


def test_first_link_reuses_active_node_tree_lookup() -> None:
    socket = _BlenderStyleLinkedSocket("Roughness", 42)
    link = SimpleNamespace(from_node=object(), from_socket=SimpleNamespace(name="Red"))
    previous = materialx_openpbr_conversion._ACTIVE_INPUT_LINKS
    materialx_openpbr_conversion._ACTIVE_INPUT_LINKS = {42: link}
    try:
        assert materialx_openpbr_conversion._first_link(socket) is link
    finally:
        materialx_openpbr_conversion._ACTIVE_INPUT_LINKS = previous


def test_material_scene_conversion_result_enforces_success_and_error_invariants() -> None:
    assert MaterialSceneConversionResult(MaterialSceneConversionStatus.OK).value is None
    with pytest.raises(ValueError, match="cannot have an error reason"):
        MaterialSceneConversionResult(MaterialSceneConversionStatus.OK, error_reason="bad")
    with pytest.raises(TypeError, match="value must be a presentation layer"):
        MaterialSceneConversionResult(MaterialSceneConversionStatus.OK, value=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires an error reason"):
        MaterialSceneConversionResult(MaterialSceneConversionStatus.ERROR)


def test_material_scene_conversion_with_no_materials_is_vacuously_ok() -> None:
    result = scene_layer_from_materials((), "unused.usda")

    assert result.status is MaterialSceneConversionStatus.OK
    assert result.value is None
    assert result.error_reason is None


def test_material_scene_conversion_caches_usd_binding_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    material = _principled_material()
    load_calls: list[str] = []

    def load_identity(path: str) -> dict[str, object]:
        load_calls.append(str(path))
        return {
            "available": True,
            "material_paths": ("/World/Looks/Paint",),
            "bindings": (),
        }

    materialx_openpbr_conversion._MATERIALX_BINDING_IDENTITY_CACHE.clear()
    monkeypatch.setattr(
        materialx_openpbr_conversion,
        "_load_materialx_binding_identity",
        load_identity,
    )

    first = scene_layer_from_materials((material,), "scene.usda")
    second = scene_layer_from_materials((material,), "scene.usda")

    assert first.status is MaterialSceneConversionStatus.OK
    assert second.status is MaterialSceneConversionStatus.OK
    assert load_calls == ["scene.usda"]


def test_material_scene_conversion_is_atomic_for_unsupported_bound_material() -> None:
    supported = _principled_material(name="Supported")
    unsupported = _principled_material(
        name="Unsupported",
        extra_principled_inputs=[_Socket("Unsupported Fancy", 0.3)],
    )
    result = _conversion_result(
        [supported, unsupported],
        {
            "available": True,
            "material_paths": ("/World/Looks/Supported", "/World/Looks/Unsupported"),
            "bindings": (
                {
                    "material_path": "/World/Looks/Supported",
                    "binding_target": "/World/Geom/Supported",
                },
                {
                    "material_path": "/World/Looks/Unsupported",
                    "binding_target": "/World/Geom/Unsupported",
                },
            ),
        },
    )

    assert result.status is MaterialSceneConversionStatus.ERROR
    assert result.value is None
    assert result.error_reason and "Unsupported" in result.error_reason
    assert len(result.diagnostics["materials"]) == 2


def test_material_scene_conversion_can_keep_stock_material_for_unsupported_overlay() -> None:
    supported = _principled_material(name="Supported")
    unsupported = _principled_material(
        name="Unsupported",
        extra_principled_inputs=[_Socket("Unsupported Fancy", 0.3)],
    )
    identity = {
        "available": True,
        "material_paths": ("/World/Looks/Supported", "/World/Looks/Unsupported"),
        "bindings": (
            {
                "material_path": "/World/Looks/Supported",
                "binding_target": "/World/Geom/Supported",
            },
            {
                "material_path": "/World/Looks/Unsupported",
                "binding_target": "/World/Geom/Unsupported",
            },
        ),
    }
    materialx_openpbr_conversion._MATERIALX_BINDING_IDENTITY_CACHE["scene.usda"] = identity

    result = scene_layer_from_materials(
        (supported, unsupported),
        "scene.usda",
        allow_stock_fallback=True,
    )

    assert result.status is MaterialSceneConversionStatus.OK
    assert result.value is not None
    assert result.diagnostics["stock_fallback_materials"] == ["Unsupported"]
    assert 'over "Supported"' in result.value.layer_body
    assert 'over "Unsupported"' not in result.value.layer_body


def test_materialx_openpbr_conversion_generates_scalar_principled_material() -> None:
    material = _principled_material(name="Paint Red")

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Can",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    assert result.value is not None
    assert result.value.diagnostics["source"] == "materialx_openpbr"
    assert result.value.target_path == "/World/Geom/Can"
    assert result.value.authored_properties == (("/World/Geom/Can", "material:binding"),)
    assert result.value.digest_content["layer_body"].rstrip() == result.value.layer_body
    assert result.value.digest_content["digest"] == result.diagnostics["digest"]
    assert result.diagnostics["generated_material_paths"] == ["/OVRTX_Materials/Paint_Red"]
    assert result.diagnostics["binding_targets"] == ["/World/Geom/Can"]
    text = result.value.layer_body
    assert 'def Scope "OVRTX_Materials"' in text
    assert 'def Material "Paint_Red"' in text
    assert "uniform token info:id = \"ND_open_pbr_surface_surfaceshader\"" in text
    assert "color3f inputs:base_color = (0.2, 0.4, 0.6)" in text
    assert "float inputs:base_metalness = 0.75" in text
    assert "float inputs:geometry_opacity = 0.5" in text
    assert "float inputs:specular_roughness = 0.35" in text
    assert "rel material:binding = </OVRTX_Materials/Paint_Red>" in text
    material_record = result.diagnostics["materials"][0]
    assert material_record["source_usd_path"] == "/World/Looks/Paint_Red"
    assert material_record["node_inventory"][0]["classification"] == "supported"


def test_overlay_digest_hashes_authored_layer_not_diagnostic_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest_inputs: list[object] = []
    original_digest = materialx_openpbr_conversion._digest_json

    def capture_digest(value: object) -> str:
        digest_inputs.append(value)
        return original_digest(value)

    monkeypatch.setattr(materialx_openpbr_conversion, "_digest_json", capture_digest)

    _convert_bindings([_Binding(_principled_material(), ("/World/Geom/Can",))])

    assert list(digest_inputs[-1]) == ["source", "layer_body"]


def test_materialx_openpbr_conversion_groups_binding_overs_by_path_tree() -> None:
    paint = _principled_material(name="Paint")
    metal = _principled_material(name="Metal")

    result = _convert_bindings(
        [
            _Binding(paint, ("/World/Geom/Paint",)),
            _Binding(metal, ("/World/Geom/Metal",)),
        ]
    )

    text = result.value.layer_body
    assert text.count('over "World"') == 1
    assert text.count('over "Geom"') == 1
    assert 'over "Paint"' in text
    assert 'over "Metal"' in text
    assert "rel material:binding = </OVRTX_Materials/Paint>" in text
    assert "rel material:binding = </OVRTX_Materials/Metal>" in text


def test_materialx_openpbr_conversion_generates_add_shader_emission_texture(
    tmp_path: Path,
) -> None:
    base_texture = _texture_node("Base Color Texture", tmp_path / "base.png")
    metal_texture = _texture_node("Metal Texture", tmp_path / "metal.png", colorspace="Non-Color")
    rough_texture = _texture_node("Rough Texture", tmp_path / "rough.png", colorspace="Non-Color")
    normal_texture = _texture_node("Normal Texture", tmp_path / "normal.png", colorspace="Non-Color")
    emission_texture = _texture_node("Emission Texture", tmp_path / "emission.png")
    normal_map = _Node(
        "NORMAL_MAP",
        "Normal Map",
        [_Socket("Color"), _Socket("Strength", 0.75)],
    )
    normal_map.inputs["Color"].links.append(SimpleNamespace(from_node=normal_texture))
    principled = _Node(
        "BSDF_PRINCIPLED",
        "Principled BSDF",
        [
            _link(_Socket("Base Color", (0.8, 0.8, 0.8, 1.0)), base_texture),
            _link(_Socket("Metallic", 0.0), metal_texture),
            _link(_Socket("Roughness", 0.5), rough_texture),
            _link(_Socket("Normal", None), normal_map),
            _Socket("Alpha", 1.0),
            _Socket("Emission Color", (0.0, 0.0, 0.0, 1.0)),
            _Socket("Emission Strength", 0.0),
        ],
    )
    emission = _Node(
        "EMISSION",
        "Emission",
        [
            _link(_Socket("Color", (0.8, 0.8, 0.8, 1.0)), emission_texture),
            _Socket("Strength", 3.0),
        ],
    )
    add = _Node(
        "ADD_SHADER",
        "Add Shader",
        [
            _link(_Socket("Shader", socket_type="SHADER"), principled),
            _link(_Socket("Shader", socket_type="SHADER"), emission),
        ],
    )
    output = _Node("OUTPUT_MATERIAL", "Material Output", [_link(_Socket("Surface"), add)])
    material = _Material(
        "Stoplight",
        [
            principled,
            emission,
            add,
            output,
            base_texture,
            metal_texture,
            rough_texture,
            normal_map,
            normal_texture,
            emission_texture,
        ],
        source_usd_path="/World/Materials/Stoplight",
    )

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Stoplight",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    assert result.diagnostics["generated_material_paths"] == ["/OVRTX_Materials/Stoplight"]
    text = result.value.layer_body
    expected_luminance = 3.0 * 120.0 * math.pi * math.pi
    assert f"float inputs:emission_luminance = {expected_luminance:.9g}" in text
    assert (
        "color3f inputs:emission_color.connect = "
        "</OVRTX_Materials/Stoplight/ND_image_color3_emission_color.outputs:out>"
    ) in text
    assert "color3f inputs:base_color.connect = </OVRTX_Materials/Stoplight/ND_image_color3_base_color.outputs:out>" in text
    assert "float inputs:base_metalness.connect = </OVRTX_Materials/Stoplight/ND_extract_color3_base_metalness.outputs:out>" in text
    assert "float inputs:specular_roughness.connect = </OVRTX_Materials/Stoplight/ND_extract_color3_specular_roughness.outputs:out>" in text
    assert 'uniform token info:id = "ND_extract_color3"' in text
    assert "float3 inputs:geometry_normal.connect = </OVRTX_Materials/Stoplight/ND_normalmap_float_geometry_normal.outputs:out>" in text
    assert "colorSpace = \"sRGB\"" in text
    assert "colorSpace = \"raw\"" in text
    record = result.diagnostics["materials"][0]
    assert record["status"] == "generated"
    assert math.isclose(record["openpbr_values"]["emission_luminance"], expected_luminance)


def test_materialx_openpbr_conversion_generates_emission_surface_without_specular() -> None:
    emission = _Node(
        "EMISSION",
        "Emission",
        [
            _Socket("Color", (0.2, 0.4, 0.8, 1.0)),
            _Socket("Strength", 2.0),
        ],
    )
    output = _Node("OUTPUT_MATERIAL", "Material Output", [_link(_Socket("Surface"), emission)])
    material = _Material("Glow", [emission, output], source_usd_path="/World/Looks/Glow")

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Glow",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    expected_luminance = 2.0 * 120.0 * math.pi * math.pi
    assert f"float inputs:emission_luminance = {expected_luminance:.9g}" in text
    assert "color3f inputs:emission_color = (0.2, 0.4, 0.8)" in text
    assert "float inputs:specular_weight = 0" in text


def test_materialx_openpbr_conversion_uses_emission_fallback_for_add_shader_without_surface() -> None:
    emission = _Node(
        "EMISSION",
        "Emission",
        [
            _Socket("Color", (1.0, 0.5, 0.25, 1.0)),
            _Socket("Strength", 1.5),
        ],
    )
    add = _Node(
        "ADD_SHADER",
        "Add Shader",
        [
            _link(_Socket("Shader", socket_type="SHADER"), emission),
            _Socket("Shader", socket_type="SHADER"),
        ],
    )
    output = _Node("OUTPUT_MATERIAL", "Material Output", [_link(_Socket("Surface"), add)])
    material = _Material(
        "GlowAdd",
        [emission, add, output],
        source_usd_path="/World/Looks/GlowAdd",
    )

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/GlowAdd",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    expected_luminance = 1.5 * 120.0 * math.pi * math.pi
    assert f"float inputs:emission_luminance = {expected_luminance:.9g}" in text
    assert "color3f inputs:emission_color = (1, 0.5, 0.25)" in text
    assert "float inputs:specular_weight = 0" in text


def test_materialx_openpbr_conversion_generates_bump_height_normal(
    tmp_path: Path,
) -> None:
    height_texture = _texture_node("Height Texture", tmp_path / "height.png", colorspace="Non-Color")
    bump = _Node(
        "BUMP",
        "Bump",
        [
            _Socket("Strength", 0.25),
            _link(_Socket("Height", 0.0), height_texture),
        ],
    )
    bump.invert = True
    material = _principled_material(
        name="Bumpy Paint",
        extra_principled_inputs=[_link(_Socket("Normal", None), bump)],
        extra_nodes=[bump, height_texture],
    )

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Bumpy",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    assert 'uniform token info:id = "ND_image_float"' in text
    assert 'uniform token info:id = "ND_heighttonormal_vector3"' in text
    assert (
        "float inputs:in.connect = "
        "</OVRTX_Materials/Bumpy_Paint/ND_image_float_geometry_normal.outputs:out>"
    ) in text
    assert "float inputs:scale = -0.25" in text
    assert (
        "float3 inputs:in.connect = "
        "</OVRTX_Materials/Bumpy_Paint/ND_heighttonormal_vector3_geometry_normal.outputs:out>"
    ) in text
    assert "ND_convert_color3_vector3_geometry_normal" not in text

    normal_texture = result.diagnostics["materials"][0]["openpbr_values"]["textures"]["geometry_normal"]
    assert normal_texture["bump"] is True
    assert normal_texture["info_id"] == "ND_image_float"
    assert normal_texture["image_output_type"] == "float"
    assert normal_texture["scale"] == -0.25


def test_materialx_openpbr_conversion_scales_bump_height_by_distance(
    tmp_path: Path,
) -> None:
    height_texture = _texture_node("Height Texture", tmp_path / "height.png", colorspace="Non-Color")
    bump = _Node(
        "BUMP",
        "Bump",
        [
            _Socket("Strength", 0.2),
            _Socket("Distance", 0.1),
            _link(_Socket("Height", 0.0), height_texture),
        ],
    )
    material = _principled_material(
        name="Distance Bump",
        extra_principled_inputs=[_link(_Socket("Normal", None), bump)],
        extra_nodes=[bump, height_texture],
    )

    result = _convert_bindings([_Binding(material, ("/World/Geom/Bumpy",))])

    normal_texture = result.diagnostics["materials"][0]["openpbr_values"]["textures"]["geometry_normal"]
    assert abs(normal_texture["scale"] - 0.02) < 1.0e-9


def test_materialx_openpbr_conversion_uses_image_raw_colorspace_over_color_slot_default(
    tmp_path: Path,
) -> None:
    texture = _texture_node(
        "Base Color Texture",
        tmp_path / "base.exr",
        colorspace="Utility - Raw",
    )
    material = _principled_material(name="Poster", extra_nodes=[texture])
    material.node_tree.nodes[0].inputs["Base Color"] = _link(
        _Socket("Base Color", (0.8, 0.8, 0.8, 1.0)),
        texture,
    )

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Poster",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    assert (
        "color3f inputs:base_color.connect = "
        "</OVRTX_Materials/Poster/ND_image_color3_base_color.outputs:out>"
    ) in text
    assert "colorSpace = \"raw\"" in text
    assert "colorSpace = \"sRGB\"" not in text


def test_materialx_openpbr_conversion_uses_image_srgb_colorspace_over_scalar_slot_default(
    tmp_path: Path,
) -> None:
    texture = _texture_node("Roughness Texture", tmp_path / "rough.png", colorspace="sRGB")
    material = _principled_material(name="Paint", extra_nodes=[texture])
    material.node_tree.nodes[0].inputs["Roughness"] = _link(
        _Socket("Roughness", 0.5),
        texture,
    )

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Paint",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    assert (
        "float inputs:specular_roughness.connect = "
        "</OVRTX_Materials/Paint/ND_extract_color3_specular_roughness.outputs:out>"
    ) in text
    assert "colorSpace = \"sRGB\"" in text
    assert "colorSpace = \"raw\"" not in text


def test_materialx_openpbr_conversion_preserves_separate_color_scalar_channel(
    tmp_path: Path,
) -> None:
    texture = _texture_node("ORM Texture", tmp_path / "orm.png", colorspace="Non-Color")
    separate = _Node(
        "SEPARATE_COLOR",
        "Separate Color",
        [_link(_Socket("Color", socket_type="RGBA"), texture)],
    )
    material = _principled_material(name="Packed", extra_nodes=[texture, separate])
    material.node_tree.nodes[0].inputs["Roughness"] = _link(
        _Socket("Roughness", 0.5),
        separate,
        from_socket_name="Green",
    )

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Packed",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    assert "int inputs:index = 1" in text
    assert (
        "float inputs:specular_roughness.connect = "
        "</OVRTX_Materials/Packed/ND_extract_color3_specular_roughness.outputs:out>"
    ) in text
    roughness_texture = result.diagnostics["materials"][0]["openpbr_values"]["textures"]["specular_roughness"]
    assert roughness_texture["extract_channel"] == 1


def test_materialx_openpbr_conversion_ports_gamma_between_color_texture_and_socket(
    tmp_path: Path,
) -> None:
    texture = _texture_node("Base Color Texture", tmp_path / "base.png")
    gamma = _Node(
        "GAMMA",
        "Gamma",
        [
            _link(_Socket("Color", (1.0, 1.0, 1.0, 1.0), socket_type="RGBA"), texture),
            _Socket("Gamma", 2.2),
        ],
    )
    material = _principled_material(name="Poster", extra_nodes=[texture, gamma])
    material.node_tree.nodes[0].inputs["Base Color"] = _link(
        _Socket("Base Color", (0.8, 0.8, 0.8, 1.0), socket_type="RGBA"),
        gamma,
    )

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Poster",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    assert (
        "color3f inputs:base_color.connect = "
        "</OVRTX_Materials/Poster/ND_power_color3FA_base_color.outputs:out>"
    ) in text
    assert 'uniform token info:id = "ND_power_color3FA"' in text
    assert (
        "color3f inputs:in1.connect = "
        "</OVRTX_Materials/Poster/ND_image_color3_base_color.outputs:out>"
    ) in text
    assert "float inputs:in2 = 2.2" in text


def test_materialx_openpbr_conversion_ports_math_between_scalar_texture_and_socket(
    tmp_path: Path,
) -> None:
    texture = _texture_node("Roughness Texture", tmp_path / "rough.png", colorspace="Non-Color")
    separate = _Node(
        "SEPARATE_COLOR",
        "Separate Color",
        [_link(_Socket("Color", socket_type="RGBA"), texture)],
    )
    multiply = _Node(
        "MATH",
        "Multiply",
        [
            _link(_Socket("Value", 0.5), separate, from_socket_name="Blue"),
            _Socket("Value", 0.8),
        ],
    )
    multiply.operation = "MULTIPLY"
    material = _principled_material(name="Paint", extra_nodes=[texture, separate, multiply])
    material.node_tree.nodes[0].inputs["Roughness"] = _link(
        _Socket("Roughness", 0.5),
        multiply,
    )

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Paint",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    assert (
        "float inputs:specular_roughness.connect = "
        "</OVRTX_Materials/Paint/ND_multiply_float_specular_roughness.outputs:out>"
    ) in text
    assert 'uniform token info:id = "ND_multiply_float"' in text
    assert (
        "float inputs:in1.connect = "
        "</OVRTX_Materials/Paint/ND_extract_color3_specular_roughness.outputs:out>"
    ) in text
    assert "int inputs:index = 2" in text
    assert "float inputs:in2 = 0.8" in text


def test_materialx_openpbr_conversion_uses_linked_rgb_principled_emission_color() -> None:
    rgb = _Node(
        "RGB",
        "Emission Tint",
        [],
        outputs=[_Socket("Color", (0.2, 0.4, 0.8, 1.0), socket_type="RGBA")],
    )
    material = _principled_material(name="Glow", extra_nodes=[rgb])
    material.node_tree.nodes[0].inputs["Emission Color"] = _link(
        _Socket("Emission Color", (1.0, 1.0, 1.0, 1.0), socket_type="RGBA"),
        rgb,
        from_socket_name="Color",
    )
    material.node_tree.nodes[0].inputs["Emission Strength"] = _Socket("Emission Strength", 2.0)

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Glow",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    expected_luminance = 2.0 * 120.0 * math.pi * math.pi
    assert f"float inputs:emission_luminance = {expected_luminance:.9g}" in text
    assert "color3f inputs:emission_color = (0.2, 0.4, 0.8)" in text
    assert "ND_image_color3_emission_color" not in text
    record = result.diagnostics["materials"][0]
    assert record["openpbr_values"]["emission_color"] == (0.2, 0.4, 0.8)
    assert record["openpbr_values"]["textures"].get("emission_color") is None


def test_materialx_openpbr_conversion_uses_linked_rgb_emission_bsdf_color() -> None:
    rgb = _Node(
        "RGB",
        "Emission Tint",
        [],
        outputs=[_Socket("Color", (1.0, 0.5, 0.25, 1.0), socket_type="RGBA")],
    )
    emission = _Node(
        "EMISSION",
        "Emission",
        [
            _link(
                _Socket("Color", (1.0, 1.0, 1.0, 1.0), socket_type="RGBA"),
                rgb,
                from_socket_name="Color",
            ),
            _Socket("Strength", 3.0),
        ],
    )
    output = _Node("OUTPUT_MATERIAL", "Material Output", [_link(_Socket("Surface"), emission)])
    material = _Material("Lamp", [emission, output, rgb], source_usd_path="/World/Looks/Lamp")

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Lamp",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    expected_luminance = 3.0 * 120.0 * math.pi * math.pi
    assert f"float inputs:emission_luminance = {expected_luminance:.9g}" in text
    assert "color3f inputs:emission_color = (1, 0.5, 0.25)" in text
    assert "ND_image_color3_emission_color" not in text
    record = result.diagnostics["materials"][0]
    assert record["openpbr_values"]["emission_color"] == (1.0, 0.5, 0.25)
    assert record["openpbr_values"]["textures"].get("emission_color") is None


def test_materialx_openpbr_conversion_uses_linked_value_emission_color() -> None:
    value = _Node(
        "VALUE",
        "Emission Value",
        [],
        outputs=[_Socket("Value", 0.3)],
    )
    emission = _Node(
        "EMISSION",
        "Emission",
        [
            _link(_Socket("Color", (1.0, 1.0, 1.0, 1.0), socket_type="RGBA"), value),
            _Socket("Strength", 4.0),
        ],
    )
    output = _Node("OUTPUT_MATERIAL", "Material Output", [_link(_Socket("Surface"), emission)])
    material = _Material("Lamp", [emission, output, value], source_usd_path="/World/Looks/Lamp")

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Lamp",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    expected_luminance = 4.0 * 120.0 * math.pi * math.pi
    assert f"float inputs:emission_luminance = {expected_luminance:.9g}" in text
    assert "color3f inputs:emission_color = (0.3, 0.3, 0.3)" in text
    record = result.diagnostics["materials"][0]
    assert record["openpbr_values"]["emission_color"] == (0.3, 0.3, 0.3)
    assert record["openpbr_values"]["textures"].get("emission_color") is None


def test_materialx_openpbr_conversion_walks_static_emission_color_passthrough() -> None:
    rgb = _Node(
        "RGB",
        "Emission Tint",
        [],
        outputs=[_Socket("Color", (0.1, 0.6, 0.9, 1.0), socket_type="RGBA")],
    )
    hue_sat = _Node(
        "HUE_SAT",
        "Hue Saturation",
        [_link(_Socket("Color", (1.0, 1.0, 1.0, 1.0), socket_type="RGBA"), rgb)],
    )
    emission = _Node(
        "EMISSION",
        "Emission",
        [
            _link(_Socket("Color", (1.0, 1.0, 1.0, 1.0), socket_type="RGBA"), hue_sat),
            _Socket("Strength", 2.0),
        ],
    )
    output = _Node("OUTPUT_MATERIAL", "Material Output", [_link(_Socket("Surface"), emission)])
    material = _Material("Lamp", [emission, output, hue_sat, rgb], source_usd_path="/World/Looks/Lamp")

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Lamp",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    assert "color3f inputs:emission_color = (0.1, 0.6, 0.9)" in text
    record = result.diagnostics["materials"][0]
    assert record["openpbr_values"]["emission_color"] == (0.1, 0.6, 0.9)


def test_materialx_openpbr_conversion_averages_static_mix_rgb_emission_color() -> None:
    warm = _Node(
        "RGB",
        "Warm",
        [],
        outputs=[_Socket("Color", (1.0, 0.2, 0.0, 1.0), socket_type="RGBA")],
    )
    cool = _Node(
        "RGB",
        "Cool",
        [],
        outputs=[_Socket("Color", (0.0, 0.4, 1.0, 1.0), socket_type="RGBA")],
    )
    mix = _Node(
        "MIX_RGB",
        "Mix Color",
        [
            _link(_Socket("Color1", (1.0, 1.0, 1.0, 1.0), socket_type="RGBA"), warm),
            _link(_Socket("Color2", (1.0, 1.0, 1.0, 1.0), socket_type="RGBA"), cool),
        ],
    )
    emission = _Node(
        "EMISSION",
        "Emission",
        [
            _link(_Socket("Color", (1.0, 1.0, 1.0, 1.0), socket_type="RGBA"), mix),
            _Socket("Strength", 2.0),
        ],
    )
    output = _Node("OUTPUT_MATERIAL", "Material Output", [_link(_Socket("Surface"), emission)])
    material = _Material(
        "Lamp",
        [emission, output, mix, warm, cool],
        source_usd_path="/World/Looks/Lamp",
    )

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Lamp",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    assert "color3f inputs:emission_color = (0.5, 0.3, 0.5)" in text
    record = result.diagnostics["materials"][0]
    assert all(
        math.isclose(actual, expected)
        for actual, expected in zip(record["openpbr_values"]["emission_color"], (0.5, 0.3, 0.5))
    )


def test_materialx_openpbr_conversion_resolves_blackbody_emission_color() -> None:
    blackbody = _Node(
        "BLACKBODY",
        "Blackbody",
        [_Socket("Temperature", 3000.0)],
    )
    emission = _Node(
        "EMISSION",
        "Emission",
        [
            _link(_Socket("Color", (1.0, 1.0, 1.0, 1.0), socket_type="RGBA"), blackbody),
            _Socket("Strength", 1.0),
        ],
    )
    output = _Node("OUTPUT_MATERIAL", "Material Output", [_link(_Socket("Surface"), emission)])
    material = _Material(
        "Warm Lamp",
        [emission, output, blackbody],
        source_usd_path="/World/Looks/Warm_Lamp",
    )

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Lamp",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    color = result.diagnostics["materials"][0]["openpbr_values"]["emission_color"]
    assert math.isclose(color[0], 1.0)
    assert math.isclose(color[1], 0.694903, rel_tol=1.0e-6)
    assert math.isclose(color[2], 0.431048, rel_tol=1.0e-6)
    assert result.diagnostics["materials"][0]["openpbr_values"]["textures"].get("emission_color") is None


def test_materialx_openpbr_conversion_resolves_sky_texture_emission_color() -> None:
    sky = _Node("TEX_SKY", "Sky Texture", [])
    emission = _Node(
        "EMISSION",
        "Emission",
        [
            _link(_Socket("Color", (1.0, 1.0, 1.0, 1.0), socket_type="RGBA"), sky),
            _Socket("Strength", 1.0),
        ],
    )
    output = _Node("OUTPUT_MATERIAL", "Material Output", [_link(_Socket("Surface"), emission)])
    material = _Material(
        "Sky Lamp",
        [emission, output, sky],
        source_usd_path="/World/Looks/Sky_Lamp",
    )

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Lamp",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    assert result.diagnostics["materials"][0]["openpbr_values"]["emission_color"] == pytest.approx(sky_texture_radiance(sky)[0])
    # Same Sky Texture value on the MaterialX route and the light-proxy route.
    assert result.diagnostics["materials"][0]["openpbr_values"]["textures"].get("emission_color") is None


def test_materialx_openpbr_conversion_generates_alpha_texture(tmp_path: Path) -> None:
    alpha_texture = _texture_node("Alpha Texture", tmp_path / "alpha.png", colorspace="Non-Color")
    material = _principled_material(name="Screen", extra_nodes=[alpha_texture])
    material.node_tree.nodes[0].inputs["Alpha"] = _link(_Socket("Alpha", 1.0), alpha_texture)

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Screen",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    assert (
        "float inputs:geometry_opacity.connect = "
        "</OVRTX_Materials/Screen/ND_extract_color3_geometry_opacity.outputs:out>"
    ) in text
    assert 'uniform token info:id = "ND_extract_color3"' in text
    assert "colorSpace = \"raw\"" in text


def test_materialx_openpbr_conversion_generates_inverted_transmission_texture(
    tmp_path: Path,
) -> None:
    opacity_texture = _texture_node("Opacity Texture", tmp_path / "opacity.png", colorspace="Non-Color")
    ramp = _Node("VALTORGB", "ColorRamp", [_link(_Socket("Fac"), opacity_texture)])
    invert = _Node("INVERT", "Invert", [_Socket("Fac", 1.0), _link(_Socket("Color"), ramp)])
    material = _principled_material(
        name="Cutout",
        extra_principled_inputs=[
            _link(_Socket("Transmission Weight", 0.0), invert),
        ],
        extra_nodes=[opacity_texture, ramp, invert],
    )

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Cutout",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    assert "float inputs:transmission_weight = 0" in text
    assert (
        "float inputs:transmission_weight.connect = "
        "</OVRTX_Materials/Cutout/ND_invert_float_transmission_weight.outputs:out>"
    ) in text
    assert (
        "float inputs:in.connect = "
        "</OVRTX_Materials/Cutout/ND_extract_color3_transmission_weight.outputs:out>"
    ) in text
    assert 'uniform token info:id = "ND_invert_float"' in text
    assert "colorSpace = \"raw\"" in text
    record = result.diagnostics["materials"][0]
    assert record["node_inventory"][2]["classification"] == "supported"
    assert record["node_inventory"][3]["classification"] == "supported"


def test_materialx_openpbr_conversion_generates_scalar_transmission_color() -> None:
    material = _principled_material(
        name="Glass",
        base_color=(0.6, 0.8, 1.0, 1.0),
        extra_principled_inputs=[_Socket("Transmission Weight", 0.75)],
    )

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Glass",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    assert "float inputs:transmission_weight = 0.75" in text
    assert "color3f inputs:transmission_color = (0.6, 0.8, 1)" in text


def test_materialx_openpbr_conversion_generates_coat_lobe() -> None:
    material = _principled_material(
        name="Eye",
        extra_principled_inputs=[
            _Socket("Coat Weight", 0.25),
            _Socket("Coat Roughness", 0.2),
            _Socket("Coat IOR", 1.4),
            _Socket("Coat Tint", (0.9, 0.8, 0.7, 1.0)),
        ],
    )

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Eye",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    assert "float inputs:coat_weight = 0.25" in text
    assert "float inputs:coat_roughness = 0.2" in text
    assert "float inputs:coat_ior = 1.4" in text
    assert "color3f inputs:coat_color = (0.9, 0.8, 0.7)" in text


def test_materialx_openpbr_conversion_preserves_default_coat_lobe_values() -> None:
    material = _principled_material(
        name="Coated Default",
        extra_principled_inputs=[
            _Socket("Coat Weight", 0.25),
            _Socket("Coat Roughness", 0.03),
            _Socket("Coat IOR", 1.5),
        ],
    )

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/CoatedDefault",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    assert "float inputs:coat_weight = 0.25" in text
    assert "float inputs:coat_roughness = 0.03" in text
    assert "float inputs:coat_ior = 1.5" in text


def test_materialx_openpbr_conversion_generates_subsurface_lobe() -> None:
    material = _principled_material(
        name="Skin",
        base_color=(0.8, 0.6, 0.5, 1.0),
        extra_principled_inputs=[
            _Socket("Subsurface Weight", 1.0),
            _Socket("Subsurface Radius", (1.0, 0.4, 0.2)),
            _Socket("Subsurface Scale", 0.05),
            _Socket("Subsurface Anisotropy", 0.3),
        ],
    )

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Skin",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    assert "float inputs:subsurface_weight = 1" in text
    assert "color3f inputs:subsurface_color = (0.8, 0.6, 0.5)" in text
    assert "float inputs:subsurface_radius = 0.05" in text
    assert "color3f inputs:subsurface_radius_scale = (1, 0.4, 0.2)" in text
    assert "float inputs:subsurface_scatter_anisotropy = 0.3" in text


def test_materialx_openpbr_conversion_generates_default_principled_specular_ior() -> None:
    material = _principled_material(name="Plastic")

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Plastic",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    assert "float inputs:specular_ior = 1.45" in text
    assert "float inputs:specular_weight = 1" not in text


def test_materialx_openpbr_conversion_generates_principled_specular_scalars() -> None:
    material = _principled_material(
        name="Brushed",
        metallic=0.0,
        extra_principled_inputs=[
            _Socket("IOR", 1.33),
            _Socket("Specular IOR Level", 0.25),
            _Socket("Specular Tint", (0.8, 0.9, 1.0, 1.0)),
            _Socket("Anisotropic", 0.4),
        ],
    )

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Brushed",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    assert "float inputs:specular_ior = 1.33" in text
    assert "float inputs:specular_weight = 0.5" in text
    assert "color3f inputs:specular_color = (0.8, 0.9, 1)" in text
    assert "float inputs:specular_roughness_anisotropy = 0.4" in text


def test_materialx_openpbr_conversion_keeps_scalar_metal_specular_weight_default() -> None:
    material = _principled_material(
        name="Gold",
        metallic=1.0,
        extra_principled_inputs=[
            _Socket("Specular IOR Level", 0.25),
        ],
    )

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Gold",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    assert "float inputs:base_metalness = 1" in text
    assert "inputs:specular_weight" not in text


def test_materialx_openpbr_conversion_neutralizes_disabled_specular_color() -> None:
    material = _principled_material(
        name="Matte Red Specular",
        metallic=0.0,
        extra_principled_inputs=[
            _Socket("Specular IOR Level", 0.0),
            _Socket("Specular Tint", (1.0, 0.0, 0.0, 1.0)),
        ],
    )

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/MatteRedSpecular",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    assert "float inputs:specular_weight = 0" in text
    assert "inputs:specular_color" not in text


def test_materialx_openpbr_conversion_uses_mix_color_input_texture(
    tmp_path: Path,
) -> None:
    base_texture = _texture_node("Base Texture", tmp_path / "shirt_base.png")
    mask_texture = _texture_node(
        "SSS Mask",
        tmp_path / "shirt_sss.png",
        colorspace="Non-Color",
    )
    mix = _Node(
        "MIX",
        "Mix",
        [
            _link(_Socket("Factor", 0.0), mask_texture),
            _link(
                _Socket("A", (1.0, 1.0, 1.0, 1.0), socket_type="RGBA"),
                base_texture,
            ),
            _Socket("B", (0.7, 0.1, 0.1, 1.0), socket_type="RGBA"),
        ],
    )
    material = _principled_material(
        name="Mixed Shirt",
        extra_nodes=[base_texture, mask_texture, mix],
    )
    material.node_tree.nodes[0].inputs["Base Color"] = _link(
        _Socket("Base Color", (0.8, 0.8, 0.8, 1.0), socket_type="RGBA"),
        mix,
    )

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Shirt",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    base_asset_path = materialx_openpbr_conversion._usda_asset_path(base_texture.image.filepath)
    mask_asset_path = materialx_openpbr_conversion._usda_asset_path(mask_texture.image.filepath)
    # The Mix is translated as a node network, so the mask image drives the blend
    # instead of being dropped (it used to be omitted, with only the first image kept).
    assert f"asset inputs:file = @{base_asset_path}@" in text
    assert f"asset inputs:file = @{mask_asset_path}@" in text
    assert 'info:id = "ND_mix_color3"' in text


def test_materialx_openpbr_conversion_accepts_zero_linked_subsurface_scale(
    tmp_path: Path,
) -> None:
    scale_texture = _texture_node("Scale Texture", tmp_path / "scale.png", colorspace="Non-Color")
    mix = _Node("MIX", "Mix", [_link(_Socket("Factor"), scale_texture)])
    material = _principled_material(
        name="Skin",
        extra_principled_inputs=[
            _Socket("Subsurface Weight", 1.0),
            _Socket("Subsurface Radius", (1.0, 1.0, 1.0)),
            _link(_Socket("Subsurface Scale", 0.0), mix),
        ],
        extra_nodes=[scale_texture, mix],
    )

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Skin",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    assert "inputs:subsurface_weight" not in text
    record = result.diagnostics["materials"][0]
    assert record["node_inventory"][-1]["classification"] == "supported"


def test_materialx_openpbr_conversion_generates_diffuse_surface() -> None:
    diffuse = _Node(
        "BSDF_DIFFUSE",
        "Diffuse BSDF",
        [_Socket("Color", (0.1, 0.2, 0.3, 1.0)), _Socket("Roughness", 0.4)],
    )
    output = _Node("OUTPUT_MATERIAL", "Material Output", [_link(_Socket("Surface"), diffuse)])
    material = _Material("Matte", [diffuse, output], source_usd_path="/World/Looks/Matte")

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Matte",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    assert "color3f inputs:base_color = (0.1, 0.2, 0.3)" in text
    assert "float inputs:specular_roughness = 0.9" in text
    assert "float inputs:specular_weight = 0" in text


def test_materialx_openpbr_conversion_generates_hair_surface() -> None:
    hair = _Node(
        "BSDF_HAIR",
        "Hair BSDF",
        [_Socket("Color", (0.4, 0.2, 0.1, 1.0))],
    )
    output = _Node("OUTPUT_MATERIAL", "Material Output", [_link(_Socket("Surface"), hair)])
    material = _Material("Hair", [hair, output], source_usd_path="/World/Looks/Hair")

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Hair",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    assert "color3f inputs:base_color = (0.4, 0.2, 0.1)" in text
    assert "float inputs:specular_roughness = 0.7" in text
    assert "float inputs:specular_weight = 0" in text


def test_materialx_openpbr_conversion_generates_transparent_surface_without_specular() -> None:
    transparent = _Node(
        "BSDF_TRANSPARENT",
        "Transparent BSDF",
        [_Socket("Color", (0.7, 0.8, 0.9, 1.0))],
    )
    output = _Node("OUTPUT_MATERIAL", "Material Output", [_link(_Socket("Surface"), transparent)])
    material = _Material(
        "Ghost",
        [transparent, output],
        source_usd_path="/World/Looks/Ghost",
    )

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Ghost",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    assert "color3f inputs:base_color = (0.7, 0.8, 0.9)" in text
    assert "float inputs:geometry_opacity = 0" in text
    assert "float inputs:specular_weight = 0" in text


def test_materialx_openpbr_conversion_generates_transparent_mix_shader() -> None:
    transparent = _Node("BSDF_TRANSPARENT", "Transparent BSDF", [_Socket("Color", (1, 1, 1, 1))])
    principled = _Node(
        "BSDF_PRINCIPLED",
        "Principled BSDF",
        [
            _Socket("Base Color", (0.3, 0.4, 0.5, 1.0)),
            _Socket("Roughness", 0.25),
            _Socket("Metallic", 0.0),
            _Socket("Alpha", 1.0),
        ],
    )
    mix = _Node(
        "MIX_SHADER",
        "Mix Shader",
        [
            _Socket("Factor", 0.25),
            _link(_Socket("Shader", socket_type="SHADER"), principled),
            _link(_Socket("Shader", socket_type="SHADER"), transparent),
        ],
    )
    output = _Node("OUTPUT_MATERIAL", "Material Output", [_link(_Socket("Surface"), mix)])
    material = _Material(
        "Cristal",
        [transparent, principled, mix, output],
        source_usd_path="/World/Looks/Cristal",
    )

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Cristal",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    text = result.value.layer_body
    assert "color3f inputs:base_color = (0.3, 0.4, 0.5)" in text
    assert "float inputs:geometry_opacity = 0.75" in text
    record = result.diagnostics["materials"][0]
    assert record["node_inventory"][0]["classification"] == "supported"
    assert record["node_inventory"][2]["classification"] == "supported"


def test_material_scene_conversion_includes_plain_supported_materials() -> None:
    plain = _principled_material(name="Paint", alpha=1.0, extra_principled_inputs=[])
    emission = _Node(
        "EMISSION",
        "Emission",
        [_Socket("Color", (1.0, 0.9, 0.8, 1.0)), _Socket("Strength", 1.0)],
    )
    output = _Node("OUTPUT_MATERIAL", "Material Output", [_link(_Socket("Surface"), emission)])
    glow = _Material("Glow", [emission, output], source_usd_path="/World/Looks/Glow")
    identity = {
        "available": True,
        "material_paths": ("/World/Looks/Paint", "/World/Looks/Glow"),
        "bindings": (
            {"material_path": "/World/Looks/Paint", "binding_target": "/World/Geom/Paint"},
            {"material_path": "/World/Looks/Glow", "binding_target": "/World/Geom/Glow"},
        ),
    }

    result = _conversion_result([plain, glow], identity)

    assert result.diagnostics["selection_policy"] == "all_supported"
    assert result.diagnostics["selected_materials"] == ["Paint", "Glow"]
    assert result.status is MaterialSceneConversionStatus.OK
    assert result.diagnostics["material_count"] == 2
    assert result.diagnostics["binding_targets"] == ["/World/Geom/Paint", "/World/Geom/Glow"]
    assert "Paint" in result.value.layer_body
    assert "Glow" in result.value.layer_body


def test_material_scene_conversion_skips_unbound_materials() -> None:
    bound = _principled_material(name="Bound", alpha=1.0, extra_principled_inputs=[])
    unbound = _principled_material(name="Unbound", alpha=1.0, extra_principled_inputs=[])
    identity = {
        "available": True,
        "material_paths": ("/World/Looks/Bound", "/World/Looks/Unbound"),
        "bindings": (
            {"material_path": "/World/Looks/Bound", "binding_target": "/World/Geom/Bound"},
        ),
    }

    result = _conversion_result([bound, unbound], identity)
    assert result.status is MaterialSceneConversionStatus.OK
    assert result.value is not None
    assert result.diagnostics["selection_policy"] == "all_supported"
    assert result.diagnostics["selected_materials"] == ["Bound", "Unbound"]
    assert result.status is MaterialSceneConversionStatus.OK
    assert result.diagnostics["material_count"] == 1
    assert result.diagnostics["binding_targets"] == ["/World/Geom/Bound"]
    assert "Bound" in result.value.layer_body
    assert "Unbound" not in result.value.layer_body
    assert "skipped_materials" not in result.value.diagnostics
    assert "selected_materials" not in result.value.diagnostics
    assert result.diagnostics["skipped_materials"] == [
        {
            "material_name": "Unbound",
            "status": "skipped_unbound_material",
            "reason": "no_usd_binding_targets",
            "identity": {
                "status": "no_binding_targets",
                "reason": "no_usd_binding_targets",
                "material_name": "Unbound",
                "source_usd_path": "/World/Looks/Unbound",
                "material_path": "/World/Looks/Unbound",
                "binding_targets": [],
                "raw_binding_targets": [],
                "candidate_count": 0,
                "match_source": "source_usd_path",
            },
        }
    ]


def test_material_scene_conversion_skips_local_material_without_usd_identity() -> None:
    bound = _principled_material(name="Bound")
    local = _principled_material(name="Local")
    local._source_usd_path = ""
    identity = {
        "available": True,
        "material_paths": ("/World/Looks/Bound",),
        "bindings": (
            {"material_path": "/World/Looks/Bound", "binding_target": "/World/Geom/Bound"},
        ),
    }

    result = _conversion_result([bound, local], identity)

    assert result.status is MaterialSceneConversionStatus.OK
    assert result.diagnostics["material_count"] == 1
    assert result.diagnostics["skipped_materials"][0]["material_name"] == "Local"
    assert result.diagnostics["skipped_materials"][0]["reason"] == "missing_source_usd_path"


def test_materialx_openpbr_conversion_rejects_untranslated_uv_vector_routing(tmp_path: Path) -> None:
    uv_map = _Node("UVMAP", "UV Map", [])
    vector_math = _Node(
        "VECT_MATH",
        "Vector Math",
        [_link(_Socket("Vector", (0.0, 0.0, 0.0)), uv_map)],
    )
    texture = _texture_node("Image Texture", tmp_path / "base.png")
    texture.inputs["Vector"] = _link(texture.inputs["Vector"], vector_math)
    material = _principled_material(extra_nodes=[uv_map, vector_math, texture])
    material.node_tree.nodes[0].inputs["Base Color"] = _link(
        _Socket("Base Color", (0.8, 0.8, 0.8, 1.0)),
        texture,
    )

    result = _convert_bindings([_Binding(material, ("/World/Geom/Can",))])

    assert result.status is MaterialSceneConversionStatus.ERROR
    assert result.value is None
    assert "image coordinate graph is not translated: Image Texture" in str(result.diagnostics)


def test_materialx_openpbr_conversion_fails_loud_for_linked_supported_input() -> None:
    texture = _Node("TEX_IMAGE", "Image Texture", [])
    linked_base_color = _Socket("Base Color", (0.8, 0.8, 0.8, 1.0))
    linked_base_color.links.append(SimpleNamespace(from_node=texture))
    material = _principled_material(
        extra_principled_inputs=[],
        extra_nodes=[texture],
    )
    material.node_tree.nodes[0].inputs["Base Color"] = linked_base_color

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Can",))]
    )

    assert result.status is MaterialSceneConversionStatus.ERROR
    assert result.value is None
    assert result.error_reason
    record = result.diagnostics["materials"][0]
    assert record["status"] == "unsupported_no_entry"
    assert "linked_supported_input:Base Color" in record["blocking_reasons"]
    assert "unsupported_node:TEX_IMAGE" in record["blocking_reasons"]


def test_materialx_openpbr_conversion_ignores_nodes_outside_active_surface_graph() -> None:
    unused = _Node("TEX_IMAGE", "Unused Image Texture", [])
    material = _principled_material(extra_nodes=[unused])

    result = _convert_bindings([_Binding(material, ("/World/Geom/Can",))])

    assert result.status is MaterialSceneConversionStatus.OK
    inventory = result.diagnostics["materials"][0]["node_inventory"]
    assert {record["name"] for record in inventory} == {
        "Principled BSDF",
        "Material Output",
    }


def test_materialx_openpbr_conversion_fails_loud_without_binding_targets() -> None:
    material = _principled_material()
    result = _conversion_result(
        [material],
        {
            "available": True,
            "material_paths": ("/World/Looks/Paint",),
            "bindings": (),
        },
    )

    assert result.status is MaterialSceneConversionStatus.OK
    assert result.value is None
    assert result.error_reason is None
    assert result.diagnostics["skipped_materials"][0]["material_name"] == "Paint"


def test_materialx_openpbr_conversion_fails_loud_for_invalid_binding_targets() -> None:
    material = _principled_material()

    result = _convert_bindings(
        [_Binding(material, ("/", "/World/Geom.inputs:x"))]
    )

    assert result.status is MaterialSceneConversionStatus.ERROR
    assert result.value is None
    assert result.error_reason
    assert result.diagnostics["materials"][0]["invalid_binding_targets"] == ["/", "/World/Geom.inputs:x"]
    assert result.diagnostics["materials"][0]["blocking_reasons"] == [
        "invalid_binding_targets",
        "missing_binding_targets",
    ]


def test_materialx_openpbr_conversion_fails_loud_for_unsupported_non_default_socket() -> None:
    material = _principled_material(extra_principled_inputs=[_Socket("Unsupported Fancy", 0.3)])

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Can",))]
    )

    assert result.status is MaterialSceneConversionStatus.ERROR
    assert result.value is None
    assert result.error_reason
    assert (
        "unsupported_non_default_input:Unsupported Fancy"
        in result.diagnostics["materials"][0]["blocking_reasons"]
    )


def test_materialx_openpbr_conversion_accepts_passive_principled_defaults() -> None:
    material = _principled_material(
        extra_principled_inputs=[
            _Socket("Emission Color", (1.0, 1.0, 1.0, 1.0)),
            _Socket("Emission Strength", 0.0),
            _Socket("Subsurface Radius", (1.0, 0.2, 0.1)),
            _Socket("IOR", 1.45),
        ]
    )

    result = _convert_bindings(
        [_Binding(material, ("/World/Geom/Can",))]
    )

    assert result.status is MaterialSceneConversionStatus.OK
    principled_inputs = result.diagnostics["materials"][0]["node_inventory"][0]["inputs"]
    passive = {
        record["name"]: record
        for record in principled_inputs
        if record["name"] in {"Emission Color", "Emission Strength", "Subsurface Radius"}
    }
    assert {record["classification"] for record in passive.values()} == {"fallback"}
    supported = {record["name"]: record for record in principled_inputs if record["name"] == "IOR"}
    assert supported["IOR"]["classification"] == "supported"


def _principled_input_records(result, names):
    inputs = result.diagnostics["materials"][0]["node_inventory"][0]["inputs"]
    return {record["name"]: record for record in inputs if record["name"] in names}


def test_materialx_openpbr_conversion_accepts_dispersion_sockets_at_their_defaults() -> None:
    # Blender 5.3 adds these two Principled sockets; a 5.2 file never has them.
    material = _principled_material(
        extra_principled_inputs=[
            _Socket("Transmission Dispersion Scale", 0.0),
            _Socket("Transmission Dispersion Abbe Number", 20.0),
        ]
    )

    result = _convert_bindings([_Binding(material, ("/World/Geom/Can",))])

    assert result.status is MaterialSceneConversionStatus.OK
    record = result.diagnostics["materials"][0]
    assert record["blocking_reasons"] == []
    assert "transmission_dispersion_scale" not in record["openpbr_values"]
    assert "transmission_dispersion_abbe_number" not in record["openpbr_values"]


def test_materialx_openpbr_conversion_translates_non_default_dispersion() -> None:
    material = _principled_material(
        extra_principled_inputs=[
            _Socket("Transmission Weight", 1.0),
            _Socket("Transmission Dispersion Scale", 1.0),
            _Socket("Transmission Dispersion Abbe Number", 35.0),
        ]
    )

    result = _convert_bindings([_Binding(material, ("/World/Geom/Can",))])

    assert result.status is MaterialSceneConversionStatus.OK
    record = result.diagnostics["materials"][0]
    assert record["blocking_reasons"] == []
    assert record["openpbr_values"]["transmission_dispersion_scale"] == 1.0
    assert record["openpbr_values"]["transmission_dispersion_abbe_number"] == 35.0
    classified = _principled_input_records(
        result, {"Transmission Dispersion Scale", "Transmission Dispersion Abbe Number"})
    assert {item["classification"] for item in classified.values()} == {"supported"}


def test_materialx_openpbr_conversion_rejects_linked_dispersion() -> None:
    noise = _Node("TEX_NOISE", "Noise Texture", [_Socket("Vector")], outputs=[_Socket("Fac", 0.5)])
    scale = _link(_Socket("Transmission Dispersion Scale", 0.0), noise, from_socket_name="Fac")
    material = _principled_material(extra_principled_inputs=[scale], extra_nodes=[noise])

    result = _convert_bindings([_Binding(material, ("/World/Geom/Can",))])

    assert result.status is MaterialSceneConversionStatus.ERROR
    assert (
        "linked_supported_input:Transmission Dispersion Scale"
        in result.diagnostics["materials"][0]["blocking_reasons"]
    )


def test_materialx_openpbr_conversion_resolves_explicit_usd_binding_targets() -> None:
    material = _principled_material(name="Paint", extra_principled_inputs=[])
    identity = {
        "available": True,
        "bindings": (
            {"material_path": "/World/Looks/Paint", "binding_target": "/World/Geom/A"},
            {"material_path": "/World/Looks/Other", "binding_target": "/World/Geom/B"},
        ),
    }
    material._source_usd_path = "/World/Looks/Paint"

    result = _conversion_result([material], identity)

    assert result.status is MaterialSceneConversionStatus.OK
    assert result.diagnostics["binding_targets"] == ["/World/Geom/A"]
    assert result.diagnostics["materials"][0]["identity"]["status"] == "resolved"


def test_materialx_openpbr_conversion_resolves_usd_binding_targets_by_material_name() -> None:
    material = _principled_material(name="Stoplight", extra_principled_inputs=[])
    material._source_usd_path = ""
    identity = {
        "available": True,
        "material_paths": ("/World/Materials/Stoplight",),
        "bindings": (
            {"material_path": "/World/Materials/Stoplight", "binding_target": "/World/Geom/Stoplight"},
        ),
    }

    result = _conversion_result([material], identity)

    assert result.status is MaterialSceneConversionStatus.OK
    assert result.diagnostics["binding_targets"] == ["/World/Geom/Stoplight"]
    assert result.diagnostics["materials"][0]["identity"]["match_source"] == "material_name"
    assert result.diagnostics["materials"][0]["identity"]["material_path"] == "/World/Materials/Stoplight"


def test_thin_walled_glass_is_opt_in_and_changes_overlay_digest() -> None:
    material = _principled_material(
        name="ShowerGlass",
        metallic=0.0,
        alpha=1.0,
        extra_principled_inputs=[_Socket("Transmission Weight", 1.0), _Socket("IOR", 1.5)],
    )
    bulk = _convert_bindings([_Binding(material, ("/World/ShowerGlass",))])
    assert bulk.status is MaterialSceneConversionStatus.OK
    assert "geometry_thin_walled" not in bulk.value.layer_body

    material.custom_properties["ovrtx_thin_walled"] = True
    thin = _convert_bindings([_Binding(material, ("/World/ShowerGlass",))])
    assert thin.status is MaterialSceneConversionStatus.OK
    assert "bool inputs:geometry_thin_walled = true" in thin.value.layer_body
    assert thin.value.digest_content["digest"] != bulk.value.digest_content["digest"]
    assert thin.diagnostics["materials"][0]["openpbr_values"]["geometry_thin_walled"] is True

    material.custom_properties["ovrtx_thin_walled"] = False
    reset = _convert_bindings([_Binding(material, ("/World/ShowerGlass",))])
    assert reset.value.digest_content["digest"] == bulk.value.digest_content["digest"]


@pytest.mark.parametrize("flag,reason", [
    ("true", "ovrtx_thin_walled_must_be_boolean"),
    (1, "ovrtx_thin_walled_must_be_boolean"),
])
def test_thin_walled_flag_rejects_invalid_values(flag, reason) -> None:
    material = _principled_material(
        name="Glass", extra_principled_inputs=[_Socket("Transmission Weight", 1.0)]
    )
    material.custom_properties["ovrtx_thin_walled"] = flag
    result = _convert_bindings([_Binding(material, ("/World/Glass",))])
    assert result.status is MaterialSceneConversionStatus.ERROR
    assert reason in result.diagnostics["materials"][0]["blocking_reasons"]


def test_thin_walled_flag_rejects_opaque_material() -> None:
    material = _principled_material(name="Wood")
    material.custom_properties["ovrtx_thin_walled"] = True
    result = _convert_bindings([_Binding(material, ("/World/Wood",))])
    assert result.status is MaterialSceneConversionStatus.ERROR
    assert "ovrtx_thin_walled_requires_principled_transmission" in (
        result.diagnostics["materials"][0]["blocking_reasons"]
    )



def test_thin_walled_rejects_connected_volume_and_allows_disconnected_node():
    material = _principled_material(extra_principled_inputs=[_Socket("Transmission Weight", 1.0)])
    material.custom_properties['ovrtx_thin_walled'] = True
    output = next(node for node in material.node_tree.nodes if node.type == 'OUTPUT_MATERIAL')
    volume = _Node('VOLUME_ABSORPTION', 'Absorption', [])
    output.inputs['Volume'] = _Socket('Volume', 0.0)
    material.node_tree.nodes.append(volume)
    valid = _convert_bindings([_Binding(material, ('/World/Glass',))])
    assert valid.status is MaterialSceneConversionStatus.OK
    _link(output.inputs['Volume'], volume)
    invalid = _convert_bindings([_Binding(material, ('/World/Glass',))])
    assert invalid.status is MaterialSceneConversionStatus.ERROR
    assert 'ovrtx_thin_walled_unsupported_volume_absorption' in invalid.diagnostics['materials'][0]['blocking_reasons']


def _uv_textured_material(tmp_path: Path, coordinate_chain):
    """Base Color image whose Vector input is fed by ``coordinate_chain(image)``."""
    texture = _texture_node("Image Texture", tmp_path / "base.png")
    extra = coordinate_chain(texture)
    material = _principled_material(extra_nodes=[texture, *extra])
    material.node_tree.nodes[0].inputs["Base Color"] = _link(
        _Socket("Base Color", (0.8, 0.8, 0.8, 1.0)), texture, from_socket_name="Color")
    return material


def _mapping(vector_type="TEXTURE", *, location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)):
    node = _Node("MAPPING", "Mapping", [_Socket("Vector", (0.0, 0.0, 0.0)), _Socket("Location", location),
                                        _Socket("Rotation", rotation), _Socket("Scale", scale)])
    node.vector_type = vector_type
    return node


def _tex_coord_into(target_socket, output="UV"):
    coordinate = _Node("TEX_COORD", "Texture Coordinate", [])
    _link(target_socket, coordinate, from_socket_name=output)
    return coordinate


@pytest.mark.parametrize("vector_type", ["TEXTURE", "POINT"])
def test_identity_mapping_from_uv_is_the_default_coordinate(tmp_path: Path, vector_type: str) -> None:
    def chain(texture):
        mapping = _mapping(vector_type)
        _link(texture.inputs["Vector"], mapping, from_socket_name="Vector")
        return [mapping, _tex_coord_into(mapping.inputs["Vector"])]

    result = _convert_bindings([_Binding(_uv_textured_material(tmp_path, chain), ("/World/Geom/Can",))])

    assert result.status is MaterialSceneConversionStatus.OK
    assert result.diagnostics["materials"][0]["status"] == "generated"
    assert "ND_image_color3_base_color" in result.value.layer_body


def test_direct_uv_and_unnamed_uv_map_are_the_default_coordinate(tmp_path: Path) -> None:
    def direct(texture):
        return [_tex_coord_into(texture.inputs["Vector"])]

    def uv_map(texture):
        node = _Node("UVMAP", "UV Map", [])
        node.uv_map = ""
        _link(texture.inputs["Vector"], node, from_socket_name="UV")
        return [node]

    for chain in (direct, uv_map):
        result = _convert_bindings([_Binding(_uv_textured_material(tmp_path, chain), ("/World/Geom/Can",))])
        assert result.status is MaterialSceneConversionStatus.OK, chain.__name__


@pytest.mark.parametrize("case", ["named_uv_map", "tilted_mapping", "linked_mapping_value",
                                  "generated_coordinates", "unfed_mapping"])
def test_other_image_coordinates_stay_untranslated(tmp_path: Path, case: str) -> None:
    def chain(texture):
        if case == "named_uv_map":
            node = _Node("UVMAP", "UV Map", [])
            node.uv_map = "Decals"
            _link(texture.inputs["Vector"], node, from_socket_name="UV")
            return [node]
        if case == "generated_coordinates":
            return [_tex_coord_into(texture.inputs["Vector"], output="Generated")]
        # Tilting about both X and Y shears the UV plane (a lone tilt or a Z rotation is translated).
        mapping = _mapping(rotation=(0.5, 0.3, 0.0)) if case == "tilted_mapping" else _mapping()
        _link(texture.inputs["Vector"], mapping, from_socket_name="Vector")
        extra = [mapping]
        if case == "linked_mapping_value":
            _link(mapping.inputs["Scale"], _Node("VALUE", "Value", []), from_socket_name="Value")
        if case != "unfed_mapping":
            extra.append(_tex_coord_into(mapping.inputs["Vector"]))
        return extra

    result = _convert_bindings([_Binding(_uv_textured_material(tmp_path, chain), ("/World/Geom/Can",))])

    assert result.status is MaterialSceneConversionStatus.ERROR
    assert "image coordinate graph is not translated: Image Texture" in str(result.diagnostics)


def _library_identity(*names):
    """Stock-export shape: prims named by the base material name, one binding each."""
    paths = tuple(f"/World/_materials/{name}" for name in names)
    return {
        "available": True,
        "reason": "",
        "material_paths": paths,
        "bindings": tuple({"material_path": path, "binding_target": f"/World/Geom/{path.rsplit('/', 1)[-1]}"}
                          for path in paths),
    }


def _linked_material(name, library_tag, uid):
    material = _principled_material(name=name)
    material._source_usd_path = ""  # stock export: no ovrtx:sourceUsdPath
    material.name_full = f"{name} [{library_tag}]" if library_tag else name
    material.session_uid = uid
    return material


def test_library_linked_material_resolves_by_scene_generation_identity() -> None:
    lamp = _linked_material("Lamp", "lib", 7)
    identity = _library_identity("Lamp")

    unresolved = materialx_openpbr_conversion._resolve_binding(lamp, identity)
    resolved = materialx_openpbr_conversion._resolve_binding(lamp, identity, {7: "/World/_materials/Lamp"})

    assert unresolved["status"] == "missing_source_usd_path"
    assert resolved["status"] == "resolved"
    assert resolved["match_source"] == "scene_generation_identity"
    assert resolved["binding_targets"] == ["/World/Geom/Lamp"]


def test_local_and_linked_materials_with_one_base_name_bind_to_their_own_prims() -> None:
    local = _linked_material("Lamp", "", 1)
    linked = _linked_material("Lamp", "lib", 2)
    identity = _library_identity("Lamp", "Lamp_001")
    known = {1: "/World/_materials/Lamp", 2: "/World/_materials/Lamp_001"}

    assert materialx_openpbr_conversion._resolve_binding(local, identity, known)["material_path"] == "/World/_materials/Lamp"
    assert materialx_openpbr_conversion._resolve_binding(linked, identity, known)["material_path"] == "/World/_materials/Lamp_001"
    # Without the generation identity the linked material must not borrow the local prim.
    assert materialx_openpbr_conversion._resolve_binding(linked, identity)["material_path"] == ""


def test_known_material_path_missing_from_the_stage_is_ignored() -> None:
    lamp = _linked_material("Lamp", "lib", 7)
    resolved = materialx_openpbr_conversion._resolve_binding(
        lamp, _library_identity("Other"), {7: "/World/_materials/Lamp"})
    assert resolved["status"] == "missing_source_usd_path"


def test_scene_layer_converts_library_linked_material_with_known_paths() -> None:
    lamp = _linked_material("Lamp", "lib", 7)
    materialx_openpbr_conversion._MATERIALX_BINDING_IDENTITY_CACHE.clear()
    materialx_openpbr_conversion._MATERIALX_BINDING_IDENTITY_CACHE["test.usda"] = _library_identity("Lamp")

    result = scene_layer_from_materials([lamp], "test.usda", known_material_paths={7: "/World/_materials/Lamp"})

    assert result.status is MaterialSceneConversionStatus.OK
    assert result.diagnostics["materials"][0]["status"] == "generated"
    assert "rel material:binding = </OVRTX_Materials/" in result.value.layer_body


def _diffuse_material(name, color_source=None, normal_source=None, extra=()):
    diffuse = _Node("BSDF_DIFFUSE", "Diffuse BSDF", [_Socket("Color", (0.8, 0.8, 0.8, 1.0)),
                                                     _Socket("Roughness", 0.0), _Socket("Normal", (0.0, 0.0, 0.0))])
    if color_source is not None:
        _link(diffuse.inputs["Color"], color_source, from_socket_name="Color")
    if normal_source is not None:
        _link(diffuse.inputs["Normal"], normal_source, from_socket_name="Normal")
    output = _Node("OUTPUT_MATERIAL", "Material Output", [_Socket("Surface")])
    _link(output.inputs["Surface"], diffuse)
    nodes = [diffuse, output, *[n for n in (color_source, normal_source) if n is not None], *extra]
    return _Material(name, nodes, source_usd_path="/World/Looks/" + name)


def test_textured_diffuse_bsdf_converts_its_texture(tmp_path: Path) -> None:
    material = _diffuse_material("Map", _texture_node("Image Texture", tmp_path / "map.png"))

    result = _convert_bindings([_Binding(material, ("/World/Geom/Map",))])

    assert result.status is MaterialSceneConversionStatus.OK
    record = result.diagnostics["materials"][0]
    assert record["status"] == "generated"
    assert "base_color" in record["openpbr_values"]["textures"]
    assert "color3f inputs:base_color.connect = </OVRTX_Materials/Map/ND_image_color3_base_color.outputs:out>" in result.value.layer_body


def test_diffuse_bsdf_static_linked_color_is_used(tmp_path: Path) -> None:
    rgb = _Node("RGB", "RGB", [], outputs=[_Socket("Color", (0.1, 0.2, 0.3, 1.0))])
    material = _diffuse_material("Tint", rgb)

    result = _convert_bindings([_Binding(material, ("/World/Geom/Tint",))])

    values = result.diagnostics["materials"][0]["openpbr_values"]
    assert tuple(round(c, 6) for c in values["base_color"]) == (0.1, 0.2, 0.3)


@pytest.mark.parametrize("case", ["color", "normal"])
def test_diffuse_bsdf_untranslatable_links_reject_instead_of_flattening(tmp_path: Path, case: str) -> None:
    noise = _Node("TEX_NOISE", "Noise Texture", [])
    material = (_diffuse_material("Noisy", color_source=noise) if case == "color"
                else _diffuse_material("Bumpy", normal_source=_Node("VECT_MATH", "Vector Math", [])))

    result = _convert_bindings([_Binding(material, ("/World/Geom/X",))])

    assert result.status is MaterialSceneConversionStatus.ERROR
    assert f"linked_supported_input:{'Color' if case == 'color' else 'Normal'}" in str(result.diagnostics)


@pytest.mark.parametrize("vector_type,location,scale,expected", [
    ("POINT", (0.25, -0.5, 9.0), (2.0, 3.0, 7.0), ((2.0, 3.0), (0.25, -0.5))),
    ("TEXTURE", (0.25, -0.5, 9.0), (2.0, 4.0, 7.0), ((0.5, 0.25), (-0.125, 0.125))),
    ("VECTOR", (0.25, -0.5, 9.0), (2.0, 3.0, 7.0), ((2.0, 3.0), (0.0, 0.0))),
])
def test_mapping_becomes_a_uv_scale_and_offset(vector_type, location, scale, expected) -> None:
    mapping = _mapping(vector_type, location=location, scale=scale)
    _tex_coord_into(mapping.inputs["Vector"])
    link = SimpleNamespace(from_node=mapping, from_socket=SimpleNamespace(name="Vector"))

    transform = materialx_openpbr_conversion._uv_coordinate_transform(link)

    assert transform == (pytest.approx(expected[0]), pytest.approx(expected[1]))


def test_chained_mappings_compose() -> None:
    outer = _mapping("POINT", location=(1.0, 0.0, 0.0), scale=(2.0, 2.0, 1.0))
    inner = _mapping("POINT", location=(0.5, 0.5, 0.0), scale=(3.0, 1.0, 1.0))
    _link(outer.inputs["Vector"], inner, from_socket_name="Vector")
    _tex_coord_into(inner.inputs["Vector"])
    link = SimpleNamespace(from_node=outer, from_socket=SimpleNamespace(name="Vector"))

    # outer(inner(uv)) = 2 * (s_i * uv + l_i) + l_o
    assert materialx_openpbr_conversion._uv_coordinate_transform(link) == (
        pytest.approx((6.0, 2.0)), pytest.approx((2.0, 1.0)))


def test_scaled_mapping_authors_texcoord_scale_and_offset(tmp_path: Path) -> None:
    def chain(texture):
        mapping = _mapping("TEXTURE", scale=(0.1, 2.0, 1.0))
        _link(texture.inputs["Vector"], mapping, from_socket_name="Vector")
        return [mapping, _tex_coord_into(mapping.inputs["Vector"])]

    result = _convert_bindings([_Binding(_uv_textured_material(tmp_path, chain), ("/World/Geom/Frame",))])

    assert result.status is MaterialSceneConversionStatus.OK
    body = result.value.layer_body
    assert 'info:id = "ND_texcoord_vector2"' in body
    assert "float2 inputs:in2 = (10, 0.5)" in body
    assert "float2 inputs:texcoord.connect = </OVRTX_Materials/Paint/ND_image_color3_base_color_uv_offset.outputs:out>" in body


def test_identity_mapping_authors_no_texcoord_nodes(tmp_path: Path) -> None:
    def chain(texture):
        mapping = _mapping()
        _link(texture.inputs["Vector"], mapping, from_socket_name="Vector")
        return [mapping, _tex_coord_into(mapping.inputs["Vector"])]

    result = _convert_bindings([_Binding(_uv_textured_material(tmp_path, chain), ("/World/Geom/Can",))])

    assert "ND_texcoord_vector2" not in result.value.layer_body


@pytest.mark.parametrize("op,reported", [("CURVE_RGB", ["base_color:CURVE_RGB"]), ("HUE_SAT", ["base_color:HUE_SAT"])])
def test_diffuse_texture_chain_reports_what_it_approximates(tmp_path: Path, op: str, reported) -> None:
    """An untranslatable chain falls back to its first image; omitted operations are reported, never silent.

    These fake nodes lack the sockets the translator needs (curve mapping, hue/saturation
    inputs), so they exercise the fallback; real nodes are translated as networks.
    """
    texture = _texture_node("Image Texture", tmp_path / "wood.png")
    node_type = op
    inputs = []
    middle = _Node(node_type, node_type.title(), [_Socket("Color", (1.0, 1.0, 1.0, 1.0), socket_type="RGBA"), *inputs])
    _link(middle.inputs["Color"], texture, from_socket_name="Color")
    material = _diffuse_material("Wood", middle, extra=(texture,))

    result = _convert_bindings([_Binding(material, ("/World/Geom/Wood",))])

    record = result.diagnostics["materials"][0]
    assert record["status"] == "generated"
    assert record["approximations"] == reported


def test_direct_and_applied_texture_ops_are_not_reported(tmp_path: Path) -> None:
    texture = _texture_node("Image Texture", tmp_path / "wood.png")
    gamma = _Node("GAMMA", "Gamma", [_Socket("Color", (1.0, 1.0, 1.0, 1.0)), _Socket("Gamma", 2.2)])
    _link(gamma.inputs["Color"], texture, from_socket_name="Color")
    direct = _diffuse_material("Direct", _texture_node("Image Texture", tmp_path / "map.png"))
    with_gamma = _diffuse_material("Gamma", gamma, extra=(texture,))
    identity_texture = _texture_node("Image Texture", tmp_path / "ident.png")
    identity_gamma = _Node("GAMMA", "Gamma", [_Socket("Color", (1.0, 1.0, 1.0, 1.0)), _Socket("Gamma", 1.0)])
    _link(identity_gamma.inputs["Color"], identity_texture, from_socket_name="Color")
    with_identity = _diffuse_material("Identity", identity_gamma, extra=(identity_texture,))

    for material in (direct, with_gamma, with_identity):
        record = _convert_bindings([_Binding(material, ("/World/Geom/X",))]).diagnostics["materials"][0]
        assert record["status"] == "generated"
        assert "approximations" not in record, material.name

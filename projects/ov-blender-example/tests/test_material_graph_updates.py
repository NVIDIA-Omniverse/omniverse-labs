# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace as N
import sys
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'addon'))
sys.path.insert(0, str(ROOT))
from ovrtx_blender_example import material_graph_updates as graph
from runtime.inline_material_population import InlineMaterialPopulation
from test_materialx_openpbr_conversion import _principled_material, _texture_node, _link, _Node, _Socket


def test_graph_compiles_at_exact_existing_root_with_internal_texture_links(tmp_path):
    material = _principled_material(alpha=1, metallic=0)
    image = _texture_node('Checker', tmp_path / 'checker.png')
    material.node_tree.nodes.append(image)
    _link(material.node_tree.nodes[0].inputs['Base Color'], image, from_socket_name='Color')
    root = '/World/_materials/Sand'
    text, errors = graph.material_layer(material, root)
    assert not errors
    assert '/OVRTX_Materials/' not in text
    assert '<' + root + '/ND_image_color3_base_color_g' in text
    assert '<' + root + '/ND_open_pbr_surface_surfaceshader_g' in text
    assert 'info:id = "ND_image_color3"' in text
    assert 'info:id = "ND_open_pbr_surface_surfaceshader"' in text
    assert 'outputs:surface.connect' in text and 'outputs:mtlx:surface.connect' in text
    assert graph.validate_material_layer(root, text)


def test_unsupported_graph_returns_actual_converter_reason():
    material = _principled_material(extra_principled_inputs=[_Socket('Unknown Fancy Input', 1)])
    text, errors = graph.material_layer(material, '/World/Materials/A')
    assert text is None and errors


def test_used_image_vector_graph_is_rejected_instead_of_silently_defaulted(tmp_path):
    material = _principled_material(alpha=1, metallic=0)
    image = _texture_node('Mapped Checker', tmp_path / 'mapped.png')
    mapping = _Node('MAPPING', 'Authored Mapping', [])
    _link(image.inputs['Vector'], mapping, from_socket_name='Vector')
    _link(material.node_tree.nodes[0].inputs['Base Color'], image, from_socket_name='Color')
    material.node_tree.nodes.extend([image, mapping])
    text, errors = graph.material_layer(material, '/World/Materials/A')
    assert text is None and errors == ('image coordinate graph is not translated: Mapped Checker',)


def test_unused_image_vector_graph_does_not_reject_valid_surface(tmp_path):
    material = _principled_material(alpha=1, metallic=0)
    image = _texture_node('Unused', tmp_path / 'unused.png')
    mapping = _Node('MAPPING', 'Unused Mapping', [])
    _link(image.inputs['Vector'], mapping)
    material.node_tree.nodes.extend([image, mapping])
    text, errors = graph.material_layer(material, '/World/Materials/A')
    assert text and not errors


def test_graph_payload_rejects_camera_or_external_connection():
    with pytest.raises(ValueError, match='escapes'):
        graph.validate_material_layer('/M', '#usda 1.0\ndef Material "M" {}\ndef Camera "Camera" {}\n')
    with pytest.raises(ValueError, match='connection escapes'):
        graph.validate_material_layer('/M', '#usda 1.0\ndef Material "M" {\n token outputs:surface.connect = </Other/Shader.outputs:out>\n}\n')


def test_inline_graph_population_replaces_only_changed_material_reference():
    events = []; counter = iter(range(1, 10))
    api = N(add_usd_reference_from_string=lambda *args: (events.append(('add', args[1])), next(counter))[1],
            remove_usd=lambda stage, handle: events.append(('remove', handle)),
            apply_usd_changes=lambda stage, ordinal: events.append(('apply', ordinal)))
    owner = N(stage=object(), population=api, failed=False, ordinal=0)
    population = InlineMaterialPopulation(owner)
    a = '#usda 1.0\ndef Material "A" {}\n'
    b = '#usda 1.0\ndef Material "B" {}\n'
    population.apply(population.prepare([dict(material_path='/A', layer_body=a)]), ordinal=1)
    population.apply(population.prepare([dict(material_path='/B', layer_body=b)]), ordinal=2)
    events.clear()
    new_a = '#usda 1.0\ndef Material "A" {\n string comment = "changed"\n}\n'
    population.apply(population.prepare([dict(material_path='/A', layer_body=new_a)]), ordinal=3)
    assert events == [('add', new_a), ('remove', 1), ('apply', 3)]
    assert population.handles['/B'] == 2 and population.graphs['/B'] == b


def test_graph_parse_failure_keeps_previous_sources():
    events = []
    def fail(*args):
        raise ValueError('USD parser rejected graph')
    api = N(add_usd_reference_from_string=fail, remove_usd=lambda *args: events.append('remove'))
    owner = N(stage=object(), population=api, failed=False, ordinal=0)
    population = InlineMaterialPopulation(owner)
    population.handles['/A'] = 1
    with pytest.raises(ValueError, match='parser'):
        population.apply(population.prepare([dict(material_path='/A', layer_body='#usda 1.0\ninvalid')]), ordinal=1)
    assert not events and not owner.failed and population.handles['/A'] == 1


def test_graph_compile_plan_acknowledges_only_changed_original_material():
    from dataclasses import replace
    from ovrtx_blender_example.scene_generation import BlenderId
    from ovrtx_blender_example.material_direct_updates import MaterialContract, MaterialState
    material = _principled_material(alpha=1, metallic=0)
    material.session_uid = 41
    key = BlenderId('MATERIAL', 41)
    old = MaterialState('old_graph', (), {})
    contract = MaterialContract((1, 'accepted'), {key: old}, {}, {key: '/World/Materials/A'},
                                material_handles={key: material})
    result = graph.plan(None, N(number=1, digest='accepted'), {key}, contract)
    assert not result.fallback and len(result.updates) == 1
    assert result.updates[0]['material_path'] == '/World/Materials/A'
    assert result.next_contract.materials[key].graph != old.graph
    assert result.next_contract.materials[key].targets['inputs:base_color'].startswith('/World/Materials/A/ND_open_pbr_surface_surfaceshader_g')
    assert contract.materials[key] == old


def test_changed_graph_content_addresses_nodes_but_preserves_binding_root(tmp_path):
    material = _principled_material(alpha=1, metallic=0)
    image = _texture_node('Image', tmp_path / 'first.png')
    material.node_tree.nodes.append(image)
    _link(material.node_tree.nodes[0].inputs['Base Color'], image, from_socket_name='Color')
    root = '/World/Materials/Sand'
    first, errors = graph.material_layer(material, root)
    assert not errors
    assert graph.material_layer(material, root)[0] == first
    image.image.filepath = image.image.filepath_raw = str(tmp_path / 'second.png')
    (tmp_path / 'second.png').write_bytes(b'fake')
    second, errors = graph.material_layer(material, root)
    assert not errors and first != second
    from pxr import Usd
    def shader_names(text):
        stage = Usd.Stage.CreateInMemory();stage.GetRootLayer().ImportFromString(text)
        assert stage.GetPrimAtPath(root).GetTypeName() == 'Material'
        return {str(p.GetPath()) for p in stage.Traverse() if p.GetTypeName() == 'Shader'}
    assert not shader_names(first).intersection(shader_names(second))

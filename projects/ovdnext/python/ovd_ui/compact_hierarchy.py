# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compact scene presentation without changing the native ownership graph."""
from dataclasses import replace


ACTORS = frozenset(('PxRigidDynamic', 'PxRigidStatic', 'PxArticulationLink'))


def compact_scene_hierarchy(source):
    """Return presentation nodes and hidden wrapper keys; never mutate source.

    Original nodes remain addressable for native selection and explicit class
    inspection. Only the visible child paths change. Unknown/malformed geometry
    layouts retain their original shape row so no recorded data is lost.
    """
    nodes, hidden = dict(source), set()

    def geometry(shape):
        wrappers = [source[k] for k in shape.children
                    if source[k].kind == 'group' and source[k].name.casefold() in ('geom', 'geometry')]
        if len(wrappers) != 1 or len(wrappers[0].children) != 1:
            return None
        wrapper = wrappers[0]
        geom = source[wrapper.children[0]]
        if not geom.class_name.endswith('Geometry'):
            return None
        return wrapper, geom

    for actor in source.values():
        if actor.kind != 'object' or actor.class_name not in ACTORS:
            continue
        for group_key in actor.children:
            group = source[group_key]
            if group.kind != 'group' or group.name.casefold() != 'shapes' or not group.children:
                continue
            replacements = []
            for shape_key in group.children:
                shape = source[shape_key]
                original = source.get(shape.target_key) if shape.kind == 'reference' else shape
                layout = geometry(original) if original and original.class_name == 'PxShape' else None
                if layout is None:
                    replacements.append(shape_key)
                    continue
                wrapper, geom = layout
                if shape.kind == 'reference':
                    # Sharing remains an explicit jump to the original shape.
                    nodes[shape_key] = replace(shape, name=f'↗ {geom.class_name} (shape #{shape.instance_id})')
                    replacements.append(shape_key)
                elif geom.kind == 'object':
                    extras = [k for k in shape.children if k != wrapper.key]
                    nodes[geom.key] = replace(geom, children=[*geom.children, *extras])
                    for key in extras:
                        nodes[key] = replace(nodes[key], parent_key=geom.key)
                    replacements.append(geom.key)
                    hidden.update((shape_key, wrapper.key))
                else:
                    replacements.append(shape_key)
            parent = actor.key if len(replacements) == 1 else group_key
            for key in replacements:
                nodes[key] = replace(nodes[key], parent_key=parent)
            if len(replacements) == 1:
                nodes[actor.key] = replace(nodes[actor.key], children=[
                    replacements[0] if key == group_key else key for key in nodes[actor.key].children])
                hidden.add(group_key)
            else:
                nodes[group_key] = replace(group, name='Shapes', children=replacements)
    return nodes, hidden


def geometry_shape_ids(adapter, identity):
    """Native shape context for a compacted geometry's inspector."""
    if not adapter.class_name(identity).endswith('Geometry'):
        return ()
    nodes, _ = adapter.hierarchy()
    node = nodes.get(f'object:{identity}')
    wrapper = nodes.get(node.parent_key) if node else None
    shape = nodes.get(wrapper.parent_key) if wrapper else None
    if shape and shape.class_name == 'PxShape':
        return (shape.instance_id,)
    return ()

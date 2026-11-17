# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recorded fixture roles resolved in the actual session, never platform ID tables."""
from types import SimpleNamespace


def unique_instance_id(session, class_name):
    classes = {item.id: item.name for item in session.classes()}
    matches = [item.id for item in session.instances() if classes[item.class_id] == class_name]
    assert len(matches) == 1, (class_name, matches)
    return matches[0]


def test_scene_roles(session):
    return SimpleNamespace(**{role: unique_instance_id(session, name) for role, name in {
        'actor': 'PxRigidDynamic', 'shape': 'PxShape', 'geometry': 'PxSphereGeometry',
        'material': 'PxMaterial', 'physics': 'PxPhysics', 'scene': 'PxScene',
        'gpu': 'PxGpuDynamicsMemoryConfig', 'metadata': 'PxOmniPvdMetaData'}.items()})


test_scene_roles.__test__ = False


def role_key(template, roles):
    return template.format(**vars(roles))


def populated_scene(session, frame=1):
    matches = []
    for scene in session.scenes():
        view = session.presentation(scene.id, frame)
        owned = next(item for item in view.scenes() if item.id == scene.id)
        if any(owner == owned.instance_id for owner in view.geometry_snapshot().scene_ids):
            matches.append(owned)
    assert len(matches) == 1, [(scene.id, scene.instance_id) for scene in matches]
    return matches[0]


def dynamic_geometry_owner(session, kind, frame=2):
    """Select a unique recorded dynamic body by its actual geometry family."""
    classes = {item.id: item.name for item in session.classes()}
    dynamic = {item.id for item in session.instances() if classes[item.class_id] == 'PxRigidDynamic'}
    view = session.presentation(populated_scene(session, frame).id, frame)
    geometry = view.geometry_snapshot()
    matches = set(map(int, geometry.object_ids[geometry.kinds == kind])) & dynamic
    assert len(matches) == 1, (kind, matches)
    return matches.pop()


def multi_geom_roles(session, frame=1):
    """Resolve the fixture's exact actor families and four-link articulation chain."""
    from ovd_ui.adapters import PvdDataAdapter
    view=session.presentation(frame=frame) if hasattr(session,'presentation') else session
    adapter=PvdDataAdapter(view)
    def one(values, label):
        assert len(values)==1,(label,values)
        return next(iter(values))
    def refs(identity,name):
        prop=adapter.property(identity,name)
        return [] if prop is None else [value for value in view.references(identity,prop.id) if value>0]
    def actor_by_geometry(kind, actor_class='PxRigidDynamic'):
        matches=[]
        for instance in view.instances():
            if adapter.class_name(instance.id)!=actor_class:continue
            for shape in refs(instance.id,'shapes'):
                geometries=refs(shape,'geom')+refs(shape,'geometry')
                if any(adapter.class_name(geometry)==kind for geometry in geometries):
                    matches.append(instance.id)
        return one(set(matches),kind)
    articulation=unique_instance_id(view,'PxArticulationReducedCoordinate')
    links=set(refs(articulation,'links'))
    assert len(links)==4
    parents={}
    joints={}
    for link in links:
        inbound=refs(link,'inboundJoint')
        if inbound:
            joints[link]=one(inbound,'inbound joint')
            parent=refs(joints[link],'parentLink')
            if parent:parents[link]=one(parent,'parent link')
    root=one(links-set(parents),'root link')
    chain=[root]
    while len(chain)<4:
        chain.append(one({link for link,parent in parents.items() if parent==chain[-1]},'next link'))
    box=actor_by_geometry('PxBoxGeometry')
    sphere=actor_by_geometry('PxSphereGeometry')
    families={}
    for link in links:
        shape=one(refs(link,'shapes'),'link shape')
        geometry=one(set(refs(shape,'geom')+refs(shape,'geometry')),'link geometry')
        families[link]={link,shape,geometry}|set(refs(link,'inboundJoint'))
    assert len(families[chain[2]])==len(families[chain[3]])==4
    assert len(families[chain[2]]|families[chain[3]])==8
    return SimpleNamespace(physics=unique_instance_id(view,'PxPhysics'),
        scene=unique_instance_id(view,'PxScene'),gpu=unique_instance_id(view,'PxGpuDynamicsMemoryConfig'),
        articulation=articulation,root=chain[0],middle1=chain[1],middle2=chain[2],leaf=chain[3],
        leaf_joint=joints[chain[3]],box=box,sphere=sphere,plane=actor_by_geometry('PxPlaneGeometry','PxRigidStatic'),box_shape=one(refs(box,'shapes'),'box shape'),
        families=families,middle2_joint=joints[chain[2]],middle2_shape=one(refs(chain[2],'shapes'),'middle link shape'))

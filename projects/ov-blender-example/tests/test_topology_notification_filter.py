from types import SimpleNamespace
from ovrtx_blender_example.scene_generation import BlenderId, _changed_topology_notifications

def test_mixed_unchanged_objects_do_not_widen_mesh_export():
    a,b=BlenderId("OBJECT",1),BlenderId("OBJECT",2)
    shader,new=BlenderId("MATERIAL",3),BlenderId("OBJECT",4)
    g=SimpleNamespace(topology_fingerprints={a:"same",b:"before"},blender_prim_paths={})
    assert _changed_topology_notifications(g,(a,b,shader,new),{a:"same",b:"after",new:"new"})==(b,shader,new)

def test_shared_data_notifications_keep_changed_owner_and_unknown_data():
    a,b=BlenderId("OBJECT",1),BlenderId("OBJECT",2)
    data,unknown=BlenderId("MESH",3),BlenderId("MESH",4)
    m=SimpleNamespace(blender_id_type="MESH",data_session_uid=3)
    g=SimpleNamespace(topology_fingerprints={a:"same",b:"same"},blender_prim_paths={a:m,b:m})
    assert _changed_topology_notifications(g,(data,unknown),{a:"same",b:"same"})==(unknown,)
    assert _changed_topology_notifications(g,(data,),{a:"same",b:"changed"})==(data,)

def test_removed_and_unfingerprinted_objects_remain_notifications():
    a,b=BlenderId("OBJECT",1),BlenderId("OBJECT",2)
    g=SimpleNamespace(topology_fingerprints={a:"old"},blender_prim_paths={})
    assert _changed_topology_notifications(g,(a,b),{})==(a,b)

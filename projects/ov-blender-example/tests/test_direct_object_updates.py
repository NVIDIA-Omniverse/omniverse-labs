from types import SimpleNamespace as NS
from ovrtx_blender_example import direct_object_updates as updates


class LookupOnly(dict):
    def __iter__(self): raise AssertionError('whole-index scan')
    def items(self): raise AssertionError('whole-index scan')
    def values(self): raise AssertionError('whole-index scan')


def fixture():
    def obj(uid):
        return NS(parent=None,data=NS(session_uid=uid),type='MESH',hide_render=False,
                  hide_viewport=False,visible_shadow=True,instance_type='NONE')
    a,b=obj(10),obj(10)
    objects=LookupOnly(a=a,b=b)
    controls=LookupOnly(a=updates.controls(a),b=updates.controls(b))
    users=LookupOnly({10:{'a','b'}})
    return a,b,objects,controls,users


def test_shared_mesh_swap_repairs_only_touched_buckets_after_ack():
    a,b,objects,controls,users=fixture();a.data=NS(session_uid=20)
    change=updates.plan(objects,controls,users,{}, {'a'})
    assert change.examined==1 and change.swapped=={'a'}
    assert change.users=={10:{'b'},20:{'a'}}
    assert users[10]=={'a','b'} and 20 not in users
    updates.acknowledge(objects,controls,users,change.controls)
    assert users=={10:{'b'},20:{'a'}} and controls['a'][1]==20
    reverse=updates.plan(objects,controls,users,{}, {'b'})
    assert reverse.examined==1 and not reverse.users and not reverse.swapped


def test_pending_swap_reversal_keeps_acknowledged_index_until_accepted():
    a,b,objects,controls,users=fixture();a.data=NS(session_uid=20)
    first=updates.plan(objects,controls,users,{}, {'a'})
    a.data=NS(session_uid=10)
    reverse=updates.plan(objects,controls,users,first.controls, set())
    assert reverse.swapped=={'a'} and reverse.controls['a'][1]==10
    assert not reverse.users and users=={10:{'a','b'}}
    updates.acknowledge(objects,controls,users,reverse.controls)
    assert users=={10:{'a','b'}}


def test_many_shared_owners_swap_in_one_batch_and_empty_bucket_retires():
    a,b,objects,controls,users=fixture();a.data=b.data=NS(session_uid=20)
    change=updates.plan(objects,controls,users,{}, {'a','b'})
    assert change.users=={10:set(),20:{'a','b'}}
    updates.acknowledge(objects,controls,users,change.controls)
    assert users=={20:{'a','b'}}

# SPDX-License-Identifier: Apache-2.0
"""Changed-owner lookup repair. No scene iteration, RNA element reads or RPC."""
from dataclasses import dataclass, field


def controls(obj):
    from .object_matte import opinions
    return (int(obj.parent.session_uid) if obj.parent else 0,
            int(obj.data.session_uid) if obj.data else 0, str(obj.type),
            bool(obj.hide_render), bool(obj.hide_viewport),
            bool(getattr(obj, 'visible_shadow', True)),
            str(getattr(obj, 'instance_type', 'NONE')), tuple(sorted(opinions(obj).items())))


@dataclass
class OwnerChanges:
    controls: dict = field(default_factory=dict)
    users: dict = field(default_factory=dict)
    swapped: set = field(default_factory=set)
    examined: int = 0


def plan(objects, accepted_controls, data_users, pending_controls, nominated):
    """Overlay just touched user buckets; accepted indices advance on ACK.

    Include pending owners so A→B→A before acknowledgment emits a reversal.
    Unknown data without a nominated owner remains unresolved, not ignored.
    """
    result = OwnerChanges()
    for key in set(nominated) | set(pending_controls):
        obj = objects.get(key)
        before = accepted_controls.get(key)
        if obj is None or before is None:
            continue
        after = controls(obj)
        result.examined += 1
        if after != before or key in pending_controls:
            result.controls[key] = after
        if before[1] != after[1]:
            if before[1]:
                result.users.setdefault(before[1], set(data_users.get(before[1], ()))).discard(key)
            if after[1]:
                result.users.setdefault(after[1], set(data_users.get(after[1], ()))).add(key)
        prior = pending_controls.get(key, before)
        if (before[1] != after[1] or prior[1] != after[1]) and before[2] == after[2] == 'MESH':
            result.swapped.add(key)
    return result


def acknowledge(objects, accepted_controls, data_users, changes, structure=None):
    for key, after in changes.items():
        before = accepted_controls[key]
        if before[1] != after[1]:
            old = data_users.get(before[1])
            if old is not None:
                old.discard(key)
                if not old:
                    data_users.pop(before[1], None)
            if after[1]:
                data_users.setdefault(after[1], set()).add(key)
        accepted_controls[key] = after
        if structure is not None and key in structure.controls:
            # Shape identity is now proven/acknowledged, not merely observed.
            old = structure.controls[key]
            structure.controls[key] = (*old[:2], after[1] or None, *old[3:])

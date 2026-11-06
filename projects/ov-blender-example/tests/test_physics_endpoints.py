from types import SimpleNamespace as N
import pytest
from ovrtx_blender_example.physics_endpoints import (
    AXIS_KEYS, EndpointError, physics_namespace, require_writable, validate_axes,
)


def test_namespace_reservation_rejects_descendant_and_invalid_root():
    scene = N(ov=N(physics_namespace="/World/OVPhysics"))
    assert physics_namespace(scene, ["/World/Visual"]) == "/World/OVPhysics"
    for paths in (["/World/OVPhysics"], ["/World/OVPhysics/a"]):
        with pytest.raises(EndpointError):
            physics_namespace(scene, paths)
    scene.ov.physics_namespace = "/World/not valid"
    with pytest.raises(EndpointError):
        physics_namespace(scene)


def test_axes_reject_reorder_duplicate_and_missing():
    validate_axes(N(axes=[N(name=k) for k in AXIS_KEYS]))
    for keys in (AXIS_KEYS[::-1], AXIS_KEYS[:-1], (*AXIS_KEYS[:-1], "rotY")):
        with pytest.raises(EndpointError):
            validate_axes(N(axes=[N(name=k) for k in keys]))


@pytest.mark.parametrize("field", ["library", "override_library"])
def test_linked_and_override_fail_closed(field):
    owner = N(**{field: object()})
    with pytest.raises(EndpointError, match="local data"):
        require_writable(owner)

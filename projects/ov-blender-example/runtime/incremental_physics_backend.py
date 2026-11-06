# SPDX-License-Identifier: Apache-2.0
"""CPU collider update implementation, included in the physics adapter identity."""
from public_sdk_backend import PhysicsBackend


class IncrementalPhysicsBackend(PhysicsBackend):
    def __init__(self, output_root):
        super().__init__(output_root)
        self._reset_geometry_projection()
        self._geometry_device = None

    def _reset_geometry_projection(self):
        self._geometry_handle = None
        self._geometry_values = {}
        self._geometry_text = ''
        self._geometry_failed = False

    def dispatch(self, op, params):
        if op == 'write_geometry':
            if self._geometry_device != 'cpu':
                raise ValueError('Live collider geometry is measured on CPU only')
            from inline_physics_geometry import apply
            return apply(self, params)
        result = super().dispatch(op, params)
        if op == 'load':
            self._geometry_device = params.get('device', 'gpu')
        return result

    def unload(self):
        super().unload()
        self._reset_geometry_projection()
        self._geometry_device = None

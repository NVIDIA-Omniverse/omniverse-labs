# SPDX-License-Identifier: Apache-2.0
"""Native camera auto exposure, preserving saved scene and camera controls."""
from dataclasses import dataclass
import math

ENABLED_ATTRIBUTE = 'omni:rtx:autoExposure:enabled'
TARGET_ATTRIBUTE = 'omni:rtx:autoExposure:whitePointScale'


@dataclass(frozen=True)
class NativeExposure:
    enabled: bool = False
    white_point_scale: float = 10.0
    source: str = 'request_default'

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('Native exposure enabled must be boolean')
        if not math.isfinite(self.white_point_scale) or self.white_point_scale <= 0:
            raise ValueError('Native white point scale must be finite and positive')

    def usd_lines(self):
        return [f"bool {ENABLED_ATTRIBUTE} = {'true' if self.enabled else 'false'}",
                f'float {TARGET_ATTRIBUTE} = {self.white_point_scale:.12g}']

    def identity(self):
        return {'enabled': self.enabled, 'white_point_scale': self.white_point_scale}

    def diagnostics(self):
        return dict(self.identity(), source=self.source, route='native_camera_and_render_product', status='authored')


def _saved_value(settings, name, default):
    if settings is None:
        return default
    try:
        return getattr(settings, name)
    except AttributeError:
        get = getattr(settings, 'get', None)
        return get(name, default) if callable(get) else default


def _enabled(value):
    if type(value) is bool:
        return value
    if value in (0, 1, '0', '1'):
        return value in (1, '1')
    raise ValueError('Native exposure enabled must be boolean')


def from_scene(scene):
    defaults = getattr(scene, 'ovrtx_example', None)
    camera = getattr(scene, 'camera', None)
    override = getattr(getattr(camera, 'data', None), 'ovrtx_exposure', None)
    if override is not None and not _enabled(_saved_value(override, 'use_scene_defaults', True)):
        settings, source = override, 'camera'
    else:
        settings, source = defaults, 'scene'
    return NativeExposure(_enabled(_saved_value(settings, 'native_auto_exposure', False)),
        float(_saved_value(settings, 'native_exposure_white_point_scale', 10.0)), source)

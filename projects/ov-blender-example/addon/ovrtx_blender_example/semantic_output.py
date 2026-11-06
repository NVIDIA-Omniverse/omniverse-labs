# SPDX-License-Identifier: Apache-2.0
"""Owned lossless semantic products, separate from display color transport."""
from dataclasses import dataclass
import base64
import json
import math
import re
from pathlib import Path
import struct
MAX_ID_MAP_BYTES = 4 * 1024 * 1024

def decode_id_map(data):
    data = bytes(data)
    if not 4 <= len(data) <= MAX_ID_MAP_BYTES:
        raise ValueError('Invalid semantic ID-map size')
    count = struct.unpack_from('<I', data, len(data) - 4)[0]
    end = count * 24
    if end > len(data) - 4:
        raise ValueError('Semantic active count exceeds buffer')
    labels = {}
    for index in range(count):
        identity, _, _, _, length, offset = struct.unpack_from('<6I', data, index * 24)
        if offset < end or offset + length > len(data) - 4:
            raise ValueError('Semantic label offset outside string buffer')
        label = data[offset:offset + length].decode('utf-8').rstrip('\x00').strip()
        if identity in labels and labels[identity] != label:
            raise ValueError('Conflicting semantic IDs')
        labels[identity] = label
    return labels

@dataclass(frozen=True)
class SemanticCapture:
    width: int
    height: int
    ids: bytes | memoryview
    id_map: bytes
    sensor_path: str
    epoch: str
    ordinal: int
    simulation_time_ns: int
    frame_start_time: float
    frame_end_time: float
    origin: str = 'top_left'
    camera_prim_path: str = ''

    def __post_init__(self):
        if type(self.width) is not int or type(self.height) is not int or (not 0 < self.width <= 16384) or (not 0 < self.height <= 16384) or (len(self.ids) != self.width * self.height * 4):
            raise ValueError('Invalid semantic uint32 dimensions or byte count')
        if self.origin != 'top_left' or not self.sensor_path.startswith('/') or (not self.epoch):
            raise ValueError('Invalid semantic product identity or origin')
        if type(self.ordinal) is not int or self.ordinal < 0 or type(self.simulation_time_ns) is not int or (self.simulation_time_ns < 0) or (not math.isfinite(self.frame_start_time)) or (not math.isfinite(self.frame_end_time)) or (self.frame_end_time < self.frame_start_time):
            raise ValueError('Invalid semantic frame identity')
        decode_id_map(self.id_map)
        if not isinstance(self.ids, (bytes, memoryview)) or (isinstance(self.ids, memoryview) and (not self.ids.readonly)):
            object.__setattr__(self, 'ids', bytes(self.ids))
        object.__setattr__(self, 'id_map', bytes(self.id_map))

def capture_from_reply(reply, ids, *, width, height, sensor_path):
    record = reply.get('semantic')
    if not isinstance(record, dict):
        raise ValueError('Missing native semantic products')
    for key, expected in {'width': width, 'height': height, 'sensor_path': sensor_path, 'epoch': reply.get('epoch'), 'ordinal': reply.get('ordinal'), 'simulation_time_ns': reply.get('simulation_time_ns'), 'frame_start_time': reply.get('frame_start_time'), 'frame_end_time': reply.get('frame_end_time'), 'origin': 'top_left', 'camera_prim_path': reply.get('camera_prim_path', ''), 'id_dtype': 'uint32', 'id_map_dtype': 'uint8'}.items():
        if record.get(key) != expected or type(record.get(key)) is not type(expected):
            raise ValueError('Semantic/color frame identity differs: ' + key)
    encoded = record.get('id_map_base64')
    if not isinstance(encoded, str) or len(encoded) > (MAX_ID_MAP_BYTES + 2) // 3 * 4:
        raise ValueError('Invalid semantic ID-map encoding size')
    mapping = base64.b64decode(encoded, validate=True)
    return SemanticCapture(width, height, ids, mapping, sensor_path, record['epoch'], record['ordinal'], record['simulation_time_ns'], record['frame_start_time'], record['frame_end_time'], camera_prim_path=record.get('camera_prim_path', ''))

def save_capture(stem, result, labels, *, camera, frame):
    """Write native IDs/class inventory; caller binds its exact final RGB later."""
    import numpy as np
    capture = result.semantic_capture
    if capture is None:
        raise ValueError('Renderer returned no requested semantic capture')
    if (capture.width, capture.height, capture.simulation_time_ns) != (result.width, result.height, result.simulation_time_ns):
        raise ValueError('Semantic capture belongs to a different color result')
    if capture.camera_prim_path and capture.camera_prim_path != camera:
        raise ValueError('Semantic camera differs from requested capture')
    root = Path(stem)
    root.parent.mkdir(parents=True, exist_ok=True)
    ids = np.frombuffer(capture.ids, dtype='<u4').reshape(capture.height, capture.width)
    mapping = decode_id_map(capture.id_map)
    known = {str(label['id']) for label in labels}
    recognized = set()
    for native_id in np.unique(ids):
        description = mapping.get(int(native_id), '')
        match = re.search('(?:^|;)\\s*class:\\s*([^;]+)', description)
        if not match:
            # Native SDK reserves 0 for background and 1 for unlabeled geometry.
            # Its packed map contains labeled classes only. Keep raw IDs intact.
            if int(native_id) not in (0, 1) and int(native_id) not in mapping:
                raise ValueError('Semantic image contains an unmapped native ID')
            continue
        class_id = match.group(1).strip()
        if class_id not in known:
            raise ValueError('Renderer returned an unknown semantic class ' + class_id)
        recognized.add(class_id)
    if not recognized:
        raise ValueError('Native segmentation contains no recognized labeled regions')
    np.save(str(root) + '.ids.npy', ids, allow_pickle=False)
    metadata = dict(schema_version=1, width=capture.width, height=capture.height, frame=frame, camera=camera, simulation_time_ns=capture.simulation_time_ns, frame_start_time=capture.frame_start_time, frame_end_time=capture.frame_end_time, sensor_path=capture.sensor_path, epoch=capture.epoch, ordinal=capture.ordinal, origin=capture.origin, dtype='uint32', native_product='SemanticSegmentation', native_id_map=decode_id_map(capture.id_map), labels=list(labels))
    metadata['reserved_native_ids'] = {'0': 'BACKGROUND', '1': 'UNLABELLED'}
    Path(str(root) + '.capture.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return dict(verified=True, metadata=metadata, ids_path=str(root) + '.ids.npy', capture_path=str(root) + '.capture.json')

# SPDX-License-Identifier: Apache-2.0
"""Bind lossless native class IDs to an exact final RGB image and stable palette."""
from hashlib import sha256
from pathlib import Path
import json
import re
import struct
import zlib
import numpy as np
from .semantic_output import decode_id_map

def write_rgb_png(path, pixels):
    h, w, c = pixels.shape
    if c != 3 or pixels.dtype != np.uint8:
        raise ValueError('Palette PNG must be uint8 RGB')

    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 4294967295)
    rows = b''.join((b'\x00' + pixels[y].tobytes() for y in range(h)))
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>2I5B', w, h, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))

def finalize_sidecar(rgb_path, capture_path):
    """Bind captured semantic pixels to the final display-transformed RGB bytes."""
    rgb = Path(rgb_path)
    root = Path(capture_path)
    metadata = json.loads(Path(str(root) + '.capture.json').read_text())
    ids = np.load(str(root) + '.ids.npy', allow_pickle=False)
    if ids.shape != (metadata['height'], metadata['width']) or ids.dtype != np.uint32:
        raise ValueError('Invalid raw semantic image')
    labels = {x['id']: dict(x) for x in metadata['labels']}
    rgb_mask = np.zeros((*ids.shape, 3), dtype=np.uint8)
    used = []
    colors = set()
    native = metadata['native_id_map']
    for native_id in np.unique(ids):
        description = native.get(str(int(native_id)), '')
        match = re.search('(?:^|;)\\s*class:\\s*([^;]+)', description)
        if not match:
            continue
        label_id = match.group(1).strip()
        if label_id not in labels:
            raise ValueError('Renderer returned unknown semantic class ' + label_id)
        label = labels[label_id]
        if 'color' not in label:
            salt = 0
            while True:
                digest = sha256((label_id + ':' + str(salt)).encode()).digest()
                color = tuple((32 + int(c) * 191 // 255 for c in digest[:3]))
                if color not in colors:
                    break
                salt += 1
            colors.add(color)
            label['color'] = list(color)
            label['native_ids'] = []
            used.append(label)
        label['native_ids'].append(int(native_id))
        rgb_mask[ids == native_id] = label['color']
    if not used:
        raise ValueError('Native segmentation contains no recognized labeled regions')
    stem = rgb.with_suffix('')
    write_rgb_png(str(stem) + '.semantic.png', rgb_mask)
    metadata.update(rgb_sha256=sha256(rgb.read_bytes()).hexdigest(), labels=sorted(used, key=lambda x: x['id']), rgb_filename=rgb.name, background_color=[0, 0, 0], raw_ids=str(root) + '.ids.npy')
    Path(str(stem) + '.semantic.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata

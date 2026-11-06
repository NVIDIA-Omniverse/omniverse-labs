# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write and read versioned asset bytes using ovstorage's actual FileBackend."""
import asyncio
import hashlib
from pathlib import Path
import tempfile
import ovstorage
from ovstorage.file import FileBackend


async def main():
    # Temporary files isolate the example from visitor revisions.
    with tempfile.TemporaryDirectory(prefix='ovstorage-example-') as folder:
        root = Path(folder)
        request = ovstorage.ConnectionRequest('file')
        request.add_config('root', ovstorage.ConfigValue.string(str(root)))
        stack = await ovstorage.Stack(root='files').backend(FileBackend('files')).connection('files', request).build()
        for revision, levels in enumerate((2, 4), 1):
            data = f'#usda 1.0\ndef Xform "Rack" {{\n custom int levels = {levels}\n}}\n'.encode()
            address = (root/f'rack-{revision:04}.usda').as_uri()
            await stack.write(address, data)
            restored, _ = await stack.read_bytes(address, max_bytes=1024*1024)
            assert restored == data
            print('Verified revision', revision, hashlib.sha256(restored).hexdigest())
        page = await stack.list(root.as_uri()+'/', max_results=100)
        assert len(page.items) == 2
        print('Listed both revisions through ovstorage.')


if __name__ == '__main__':
    asyncio.run(main())

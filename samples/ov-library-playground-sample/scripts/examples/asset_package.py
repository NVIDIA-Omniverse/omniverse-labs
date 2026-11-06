# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run with <runtime>/package-env/Scripts/python.exe; local files, no GPU."""
import asyncio
from pathlib import Path
import tempfile
import ovpackage

async def main():
    with tempfile.TemporaryDirectory(prefix='ovpackage-example-') as folder:
        root=Path(folder)
        source=root/'source';source.mkdir()
        (source/'robot.usda').write_text('#usda 1.0\ndef Xform "Robot" {}\n')
        async with ovpackage.ContextManager() as context:
            await ovpackage.create('robot','1.0.0',str(source),catalog=False,
                                   repo=str(root/'repo'),scheduler=context)
            await ovpackage.install('robot','1.0.0',str(root/'installed'),
                                    repo=str(root/'repo'),scheduler=context)
            baseline=await ovpackage.catalog(str(source),local_hash=True,scheduler=context)
            installed=await ovpackage.catalog(str(root/'installed'),local_hash=True,scheduler=context)
            # ovpackage creates its own install receipt outside the asset payload.
            payload=installed.model_copy(update={'items':[item for item in installed.items
                                                           if item.relative_path!='.robot.wrapp']})
            missing,extra,changed=await ovpackage.diff_catalogs(baseline,payload,ignore_tags=True)
            assert not (missing or extra or changed)
            print('Published, installed and hash-verified robot.usda through ovpackage.')

if __name__=='__main__':
    asyncio.run(main())

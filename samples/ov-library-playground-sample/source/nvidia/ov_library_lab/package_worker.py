# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ovpackage lifecycle and async API live in their own process."""
import argparse
import asyncio
import contextlib
import json
import os
import sys
import traceback

async def main(root):
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', encoding='utf-8', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    with contextlib.redirect_stdout(sys.stderr):
        import ovpackage
        from package_engine import AssetHandoff
        engine = AssetHandoff(root)
    protocol.write(json.dumps({'ready': True, 'pid': os.getpid()})+'\n')
    protocol.flush()
    with contextlib.redirect_stdout(sys.stderr):
        context = ovpackage.ContextManager()
        scheduler = await context.__aenter__()
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request.get('close'):
                break
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    result = await engine.execute(request, scheduler)
                reply = {'id': request['id'], **result}
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                reply = {'id': request['id'], 'error': str(exc)}
            protocol.write(json.dumps(reply)+'\n')
            protocol.flush()
    finally:
        with contextlib.redirect_stdout(sys.stderr):
            await context.__aexit__(None, None, None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-root', required=True)
    asyncio.run(main(parser.parse_args().runtime_root))

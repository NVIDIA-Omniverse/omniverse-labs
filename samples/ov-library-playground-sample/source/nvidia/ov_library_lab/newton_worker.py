# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Line-delimited worker protocol; SDK diagnostics never enter stdout."""
import contextlib
import json
import os
import sys
import traceback

PROTOCOL = None

def send(value):
    PROTOCOL.write(json.dumps(value, allow_nan=False)+'\n')
    PROTOCOL.flush()

def main():
    global PROTOCOL
    # Population's C++ logging writes to fd 1, bypassing redirect_stdout.
    PROTOCOL = os.fdopen(os.dup(sys.stdout.fileno()), 'w', encoding='utf-8', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    with contextlib.redirect_stdout(sys.stderr):
        from newton_engine import GantrySimulation
        engine = GantrySimulation()
        engine.run(warm=True)
    send({'ready': True, 'pid': os.getpid()})
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('close'):
            break
        try:
            with contextlib.redirect_stdout(sys.stderr):
                result = engine.run(request['mode'])
            send({'id': request['id'], **result})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            send({'id': request['id'], 'error': str(exc)})

if __name__ == '__main__':
    main()

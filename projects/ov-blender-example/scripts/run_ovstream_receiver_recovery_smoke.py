#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real SHM attach failure/reconnect and slow-consumer hold on one live producer."""
import argparse
import hashlib
import json
import queue
from pathlib import Path
import subprocess
import sys
import threading
import time


CONSUMER = '''
import sys,json,time,hashlib
sys.path.insert(0,sys.argv[1])
from ovrtx_blender_example import ovstream_shm as shm
for line in sys.stdin:
    request=json.loads(line); held=shm.HeldFrame(**request['shm'])
    kwargs=dict(stream_name=held.stream_name,epoch=held.epoch,producer_pid=held.producer_pid)
    if request['action']=='fail_map':
        if sys.platform == 'win32':
            from ovrtx_blender_example import win32_shm
            owner, attribute = win32_shm, 'ReadOnlyMapping'
        else:
            owner, attribute = shm.mmap, 'mmap'
        original=getattr(owner, attribute)
        def fail(*a,**kw): raise OSError('injected receiver mapping failure')
        setattr(owner, attribute, fail)
        try:
            try: shm.OvstreamShmReader(**kwargs)
            except OSError as exc:
                assert 'injected' in str(exc)
            else: raise AssertionError('Mapping unexpectedly succeeded')
        finally: setattr(owner, attribute, original)
        result={'mapping_failure_observed':True}
    else:
        with shm.OvstreamShmReader(**kwargs) as reader:
            print(json.dumps({'attached':True}),flush=True)
            time.sleep(request.get('delay',0))
            frame=reader.copy_frame(held)
            result={'sha256':hashlib.sha256(frame.rgba8).hexdigest(),'sequence':frame.sequence}
    assert not {'ovrtx','ovstage','ovstream','ovphysx','warp'} & sys.modules.keys()
    print(json.dumps(result),flush=True)
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--consumer-python',default=sys.executable)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'runtime'))
    sys.path.insert(0,str(root/'scripts'))
    from run_ovstream_raw_frame_smoke import resource_state
    from runtime_identity import verified_identity
    runtime_identity=verified_identity()
    import numpy as np
    import warp as wp
    from ovstream_frames import ShmPublisher
    wp.init()
    process=subprocess.Popen([args.consumer_python,'-I','-u','-c',CONSUMER,str(root/'addon')],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    lines=queue.Queue()
    def read_lines():
        for line in process.stdout:
            lines.put(line)
        lines.put(None)
    threading.Thread(target=read_lines,daemon=True).start()
    paths=set(); frames=[]; blocked=0
    def send(request):
        process.stdin.write(json.dumps(request)+'\n'); process.stdin.flush()
    def receive():
        line=lines.get(timeout=30)
        if not line:
            raise RuntimeError(process.stderr.read())
        return json.loads(line)
    try:
        with ShmPublisher(device=args.device) as publisher:
            identity=None
            for index in range(4):
                pixels=np.resize(np.arange(256,dtype=np.uint8),320*241*4).reshape(241,320,4)
                pixels=(pixels+index).astype(np.uint8)
                source=wp.array(pixels,dtype=wp.uint8,device=args.device)
                held=publisher.publish_rgba_cuda(source,320,241)
                current=(held['producer_pid'],held['epoch'],held['stream_name'])
                if identity is None:
                    identity=current
                assert current==identity, 'Producer/stream changed during receiver recovery'
                paths.add(held['stream_name'])
                if index==0:
                    send(dict(action='fail_map',shm=held))
                    assert receive()['mapping_failure_observed']
                    publisher.release(held['copy_token'])
                    continue
                send(dict(action='copy',shm=held,delay=0.5 if index==2 else 0))
                assert receive()['attached']
                if index==2:
                    start=time.monotonic()
                    while time.monotonic()-start<0.25:
                        try:
                            publisher.publish_rgba_cuda(source,320,241)
                        except RuntimeError as error:
                            assert 'still held' in str(error)
                            blocked+=1
                        else:
                            raise AssertionError('Slow consumer hold was overwritten')
                        time.sleep(0.01)
                received=receive()
                assert received['sha256']==hashlib.sha256(pixels.tobytes()).hexdigest()
                publisher.release(held['copy_token'])
                frames.append(received)
        process.stdin.close()
        assert process.wait(timeout=10)==0,process.stderr.read()
    finally:
        if process.poll() is None:
            process.kill(); process.wait(timeout=5)
        process.stdout.close(); process.stderr.close()
    resources={name:resource_state(name) for name in paths}
    assert blocked>=10 and not any(value for state in resources.values() for value in state.values())
    report=dict(status='passed',mapping_failure_then_next_frame_reconnected=True,
        same_producer_pid_epoch_stream=True,producer_pid=identity[0],slow_consumer_delay_seconds=0.5,
        rejected_publish_attempts=blocked,exact_frames=frames,vendor_modules_in_receiver=False,
        cleanup_passed=True,resources=resources,runtime_identity=runtime_identity)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()

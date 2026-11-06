#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Native exact typed transfer, borrowed-source overwrite and bounded leases.

Run in the locked SDK interpreter; --consumer-python can be plain Python.
The separate consumer uses only stdlib and add-on modules.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import struct
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
VENDORS = {'ovrtx', 'ovphysx', 'ovstage', 'ovstream', 'warp', 'numpy'}


def consume():
    sys.path.insert(0, str(ROOT / 'addon'))
    from ovrtx_blender_example.ovstream_shm import HeldFrame, OvstreamShmReader
    from ovrtx_blender_example.ovstream_raw_frame import RawFrameAssembler
    reply = json.loads(sys.stdin.readline())
    held = HeldFrame(**reply['shm'])
    with OvstreamShmReader(stream_name=held.stream_name, epoch=held.epoch,
                          producer_pid=held.producer_pid) as reader:
        region = reader._read_region()
        print(json.dumps({'attached': True, 'ring_bytes': len(reader._mapping),
                          'pitch': region.max_pitch}), flush=True)
        time.sleep(reply.get('delay', 0))
        assembler = RawFrameAssembler(held, reply['raw_frame'], reply['chunk'])
        while not assembler.append(reader, held, reply['raw_frame'], reply['chunk']):
            print(json.dumps({'need_more': True, 'copy_token': held.copy_token}), flush=True)
            reply = json.loads(sys.stdin.readline())
            held = HeldFrame(**reply['shm'])
        raw = assembler.finish()
        assert all(p.data.readonly for p in raw.planes)
        hashes = {p.plane.name: hashlib.sha256(p.data).hexdigest() for p in raw.planes}
        presentation_checked = 0
        if 'presentation_rgba32f' in hashes:
            planes = {p.plane.name: p.data for p in raw.planes}
            count = min(65536, len(planes['linear_rgba16f']) // 2)
            halves = struct.iter_unpack('<e', planes['linear_rgba16f'][:count*2])
            floats = struct.iter_unpack('<f', planes['presentation_rgba32f'][:count*4])
            for index, ((half,), (single,)) in enumerate(zip(halves, floats)):
                expected = 1.0 if index % 4 == 3 else half
                assert (math.isnan(single) if math.isnan(expected) else single == expected)
                if expected == 0:
                    assert math.copysign(1, single) == math.copysign(1, expected)
            presentation_checked = count
    del assembler
    assert all(hashlib.sha256(p.data).hexdigest() == hashes[p.plane.name] for p in raw.planes)
    assert not VENDORS & sys.modules.keys()
    print(json.dumps({'hashes': hashes, 'sdk_modules': [], 'owned_after_detach': True,
                      'presentation_values_checked': presentation_checked}), flush=True)


class Consumer:
    def __init__(self, python, stderr_path):
        self.stderr = stderr_path.open('w')
        self.process = subprocess.Popen([python, '-I', '-u', str(Path(__file__).resolve()), '--consumer'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, text=True)
        self.lines = queue.Queue()
        def read():
            for line in self.process.stdout:
                self.lines.put(line)
            self.lines.put(None)
        self.thread = threading.Thread(target=read, daemon=True)
        self.thread.start()

    def send(self, value):
        self.process.stdin.write(json.dumps(value) + '\n')
        self.process.stdin.flush()

    def receive(self):
        line = self.lines.get(timeout=60)
        if line is None:
            raise RuntimeError('Consumer exited; inspect its retained stderr log')
        return json.loads(line)

    def finish(self):
        self.process.stdin.close()
        assert self.process.wait(timeout=30) == 0

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=10)
        for pipe in (self.process.stdin, self.process.stdout):
            if not pipe.closed:
                pipe.close()
        self.stderr.close()


def resource_state(stream):
    if sys.platform == 'win32':
        sys.path.insert(0, str(ROOT / 'addon'))
        from ovrtx_blender_example.win32_shm import existing_resources
        return existing_resources(stream)
    return {str(path): path.exists() for path in (
        Path('/dev/shm') / ('ovstream-' + stream),
        Path('/dev/shm') / ('sem.ovstream-' + stream + '-wake'),
        Path(os.environ.get('TMPDIR', '/tmp')) / ('ovstream-' + stream + '.sock'))}


def run(args):
    sys.path.insert(0, str(ROOT / 'runtime'))
    from runtime_identity import verified_identity
    runtime_identity = verified_identity()
    import numpy as np
    import warp as wp
    from ovstream_frames import ShmPublisher
    wp.init()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    case_specs = [(7, 5, False, False, None), (65, 9, True, False, None),
                  (257, 64, True, False, 4096)]
    for width, height in ((1280, 720), (1920, 1080), (3840, 2160)):
        case_specs.extend((width, height, hdr, staging, None)
                          for hdr, staging in ((False, False), (True, False), (True, True)))
    report = dict(status='running', cases=[], device=args.device,
                  producer_python=sys.executable, consumer_python=args.consumer_python,
                  command=sys.argv, platform=sys.platform, runtime_identity=runtime_identity,
                  lock_sha256=hashlib.sha256((ROOT / 'runtime/public-sdk-lock.json').read_bytes()).hexdigest(),
                  source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (Path(__file__), ROOT / 'runtime/ovstream_frames.py',
                          ROOT / 'addon/ovrtx_blender_example/ovstream_raw_frame.py',
                          ROOT / 'addon/ovrtx_blender_example/ovstream_shm.py',
                          ROOT / 'addon/ovrtx_blender_example/win32_shm.py')})
    streams = set()
    try:
        for number, (width, height, hdr, staging, chunk_bytes) in enumerate(case_specs):
            bits = np.resize(np.arange(65536, dtype=np.uint16), height*width*4).reshape(height, width, 4)
            pixels = (bits % 256).astype(np.uint8)
            expected = bits if hdr else pixels
            padding = 3 if width in (7, 65, 257) else 0
            backing = np.full((height, width+padding, 4), 0xdead if hdr else 0xaa,
                              dtype=np.uint16 if hdr else np.uint8)
            backing[:, :width] = expected
            dtype = wp.float16 if hdr else wp.uint8
            gpu = wp.array(backing.view(np.float16) if hdr else backing, dtype=dtype, device=args.device)
            source = wp.array(ptr=gpu.ptr, shape=(height, width, 4), strides=gpu.strides,
                              dtype=dtype, device=args.device, copy=False)
            name, fmt = ('linear_rgba16f', 'rgba16f') if hdr else ('rgba8', 'rgba8')
            planes = [(name, fmt, source)]
            registry = args.output.parent / ('registry-' + str(number) + '.json')
            with ShmPublisher(device=args.device, raw_chunk_bytes=chunk_bytes, registry_path=registry) as publisher:
                reply = publisher.publish_raw_planes_cuda(planes, presentation_rgba32f=staging)
                initial = reply
                streams.add(reply['shm']['stream_name'])
                partial_cancelled = False
                if reply['chunk']['count'] > 1:
                    try:
                        publisher.release(reply['shm']['copy_token'])
                    except RuntimeError as error:
                        assert 'Incomplete' in str(error)
                    else:
                        raise AssertionError('Premature final ACK accepted')
                    publisher.cancel_raw_transfer(reply['shm']['copy_token'])
                    reply = publisher.publish_raw_planes_cuda(planes, presentation_rgba32f=staging)
                    initial = reply
                    partial_cancelled = True
                # The publisher owns every chunk before renderer unmap/overwrite.
                source.zero_()
                wp.synchronize_device(args.device)
                consumer = Consumer(args.consumer_python, args.output.parent / ('consumer-' + str(number) + '.log'))
                rejected = 0
                try:
                    consumer.send(dict(reply, delay=.3))
                    attached = consumer.receive()
                    assert attached['attached']
                    assert attached['ring_bytes'] <= 256*1024*1024
                    deadline = time.monotonic() + .15
                    while time.monotonic() < deadline:
                        try:
                            publisher.publish_raw_planes_cuda(planes)
                        except RuntimeError as error:
                            assert 'still held' in str(error)
                            rejected += 1
                        else:
                            raise AssertionError('Consumer lease overwritten')
                        time.sleep(.01)
                    while True:
                        received = consumer.receive()
                        if not received.get('need_more'):
                            break
                        token = received['copy_token']
                        assert token == reply['shm']['copy_token']
                        reply = publisher.next_raw_chunk(token)
                        try:
                            publisher.next_raw_chunk(token)
                        except RuntimeError:
                            pass
                        else:
                            raise AssertionError('Duplicate chunk ACK accepted')
                        consumer.send(reply)
                    consumer.finish()
                finally:
                    consumer.close()
                assert rejected >= 1
                assert received['hashes'][name] == hashlib.sha256(expected.tobytes()).hexdigest()
                assert ('presentation_rgba32f' in received['hashes']) == staging
                token = reply['shm']['copy_token']
                publisher.release(token)
                try:
                    publisher.release(token)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError('Double final ACK accepted')
                assert publisher._held_token is None and publisher._raw_transfer is None
                cache_bytes = publisher._raw_payload_cache.size
                assert cache_bytes == initial['raw_frame']['byte_count'] <= 256*1024*1024
                report['cases'].append(dict(width=width, height=height, hdr=hdr, staging_requested=staging,
                    staging_delivered='presentation_rgba32f' in received['hashes'],
                    covered_half_bit_patterns=min(bits.size, 65536) if hdr else 0,
                    padded_source_pixels=padding, raw_chunk_bytes=publisher._raw_chunk_bytes,
                    carrier_padding_bytes=attached['pitch']-initial['shm']['width']*4,
                    chunks=initial['chunk']['count'], ring_bytes=attached['ring_bytes'],
                    cache_bytes=cache_bytes, blocked_overwrites=rejected,
                    partial_cancelled=partial_cancelled, source_overwritten=True,
                    duplicate_ack_rejected=True, descriptor=initial['raw_frame'], **received))
                report['cases'][-1]['resources_after_ack'] = publisher.diagnostics()
            assert json.loads(registry.read_text())['streams'] == []
            assert not any(resource_state(initial['shm']['stream_name']).values())
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({'case': number, 'width': width, 'height': height, 'hdr': hdr, 'passed': True}), flush=True)
        assert any(c['covered_half_bit_patterns'] == 65536 for c in report['cases'])
        assert any(c['carrier_padding_bytes'] for c in report['cases'])
        assert any(c['chunks'] > 1 for c in report['cases'])
        report.update(status='passed', cleanup={s: resource_state(s) for s in streams})
        assert not any(value for state in report['cleanup'].values() for value in state.values())
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--consumer', action='store_true')
    parser.add_argument('--consumer-python', default=sys.executable)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.consumer:
        consume()
    else:
        if args.output is None:
            parser.error('--output is required')
        run(args)


if __name__ == '__main__':
    main()

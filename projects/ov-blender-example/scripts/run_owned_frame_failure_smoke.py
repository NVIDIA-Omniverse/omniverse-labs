#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real renderer/SHM failure boundaries with SDK-free consumer fault injection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'addon'))
from run_public_sdk_first_frame_smoke import scene_spec, VENDORS, local_caches
from run_ovstream_raw_frame_smoke import resource_state
FAULTS = ('map', 'copy', 'disconnect', 'death_before_map', 'death_during_copy',
          'death_between_chunks', 'death_before_ack')


def run(args):
    from ovrtx_blender_example.direct_sdk_client import PublicSdkRenderClient
    from ovrtx_blender_example import win32_shm
    assert sys.platform == 'win32', 'This execution receipt is Windows-only'
    args.output.mkdir(parents=True, exist_ok=False)
    local_caches(args.output.parent / 'native-cache')
    os.environ.update(OV_BLENDER_PUBLIC_SDK_PYTHON=str(Path(args.python).absolute()),
                      OV_BLENDER_PUBLIC_SDK_OUTPUT=str(args.output.resolve() / 'workers'),
                      OV_BLENDER_PUBLIC_SDK_GPU=str(args.gpu), OVRTX_HDR_GPU_FLOAT32='0')
    # Exercise genuine multi-chunk renderer output without a large image. The
    # wrapper changes only publisher carrier capacity after production runtime
    # validation; worker authentication, native rendering and RPC remain intact.
    wrapper = args.output.resolve() / 'small-carrier-worker.py'
    wrapper.write_text('import sys\n' + 'sys.path.insert(0, ' + repr(str(ROOT / 'runtime')) + ')\n' + '''
import public_sdk_backend
original_publish = public_sdk_backend.RendererBackend._publish_products
def publish(self, *args, **kwargs):
    import ovstream_frames
    original = ovstream_frames.ShmPublisher
    def small_carrier(**options):
        return original(raw_chunk_bytes=4096, **options)
    ovstream_frames.ShmPublisher = small_carrier
    try:
        return original_publish(self, *args, **kwargs)
    finally:
        ovstream_frames.ShmPublisher = original
public_sdk_backend.RendererBackend._publish_products = publish
from public_sdk_worker import main
raise SystemExit(main())
''')
    report = dict(status='running', command=sys.argv, cases=[],
                  worker_wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest())
    client = PublicSdkRenderClient()
    client.runtime.worker_script = str(wrapper)
    # This harness deliberately kills four workers in one run. Its explicit
    # bounded budget covers the requested faults; the production default stays
    # two and the runtime unit suite separately proves budget exhaustion.
    client.runtime.max_restarts = sum(fault.startswith('death') for fault in args.faults)
    report['configured_restart_budget'] = client.runtime.max_restarts
    streams = set()
    original_copy = client._copy_frame
    def observe(reply, *values):
        streams.add(reply['shm']['stream_name'])
        return original_copy(reply, *values)
    client._copy_frame = observe
    try:
        identity = client.start_session(scene_spec(65, 49))
        report['runtime_identity'] = client.runtime.request('health')['runtime_identity']
        def render():
            frame = client.render_result(identity, selected_sensor_paths=('/Render/Camera',),
                                         render_var='HdrColor', additional_samples=2)
            assert len(frame.linear_rgba16f) == 65*49*8
            assert frame.native_timings['raw_chunk_count'] > 1
            return frame
        render()
        for fault in args.faults:
            old_identity = client.runtime.identity
            before = client.runtime.diagnostics()
            mapping_type = win32_shm.ReadOnlyMapping
            copy_into = mapping_type.copy_into
            call = client._call
            armed = [True]
            def die():
                assert client.runtime.identity == old_identity
                client.runtime._owner.stop()
                assert client.runtime.returncode is not None
            def fail_map(*a, **kw):
                if armed[0]:
                    armed[0] = False
                    raise OSError('injected mapping failure after native attach')
                return mapping_type(*a, **kw)
            def fail_copy(mapping, destination, *, destination_offset, source_offset, count):
                if armed[0]:
                    armed[0] = False
                    # Real bytes were copied before the failure, so partial
                    # owned storage must be discarded and never published.
                    copy_into(mapping, destination, destination_offset=destination_offset,
                              source_offset=source_offset, count=min(4, count))
                    if fault == 'death_during_copy':
                        die()
                    elif fault == 'disconnect':
                        client._reader._control.close()
                        raise OSError('injected consumer disconnect after partial copy')
                    else:
                        raise OSError('injected partial copy failure')
                return copy_into(mapping, destination, destination_offset=destination_offset,
                                 source_offset=source_offset, count=count)
            def fail_before_map(reply, *values):
                streams.add(reply['shm']['stream_name'])
                if armed[0]:
                    armed[0] = False
                    die()
                return original_copy(reply, *values)
            def fail_rpc(op, params):
                target = 'next_raw_chunk' if fault == 'death_between_chunks' else 'release_frame'
                if op == target and armed[0]:
                    armed[0] = False
                    die()
                return call(op, params)
            if client._reader is not None:
                client._reader.close()
                client._reader = None
                client._reader_key = None
            if fault == 'map':
                win32_shm.ReadOnlyMapping = fail_map
            elif fault in ('copy', 'disconnect', 'death_during_copy'):
                mapping_type.copy_into = fail_copy
            elif fault == 'death_before_map':
                client._copy_frame = fail_before_map
            else:
                client._call = fail_rpc
            error = None
            try:
                render()
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            finally:
                win32_shm.ReadOnlyMapping = mapping_type
                mapping_type.copy_into = copy_into
                client._copy_frame = observe
                client._call = call
            assert error is not None and not armed[0], f'Fault {fault} did not reject the frame'
            assert client.runtime.epoch == old_identity.epoch, 'Interrupted transfer was replayed'
            assert client.runtime.diagnostics()['restarts'] == before['restarts']
            if fault.startswith('death'):
                # ACK failure occurs after copy; the client must still retire
                # its mapping rather than keep a dead producer's SHM alive.
                assert client._reader is None, 'Failed ACK retained the SHM mapping'
                assert not any(v for s in streams for v in resource_state(s).values())
                assert client.states[identity].failure is not None
                assert client.recover_value_updates(identity)
                assert client.runtime.epoch == old_identity.epoch + 1
            else:
                assert client.states[identity].failure is None
                assert client._reader is None
                assert client.runtime.identity == old_identity
            frame = render()
            report['cases'].append(dict(fault=fault, observed_error=error,
                old_identity=vars(old_identity), recovered_identity=vars(client.runtime.identity),
                interrupted_transfer_replayed=False, fresh_frame_bytes=len(frame.linear_rgba16f),
                fresh_frame_sha256=hashlib.sha256(frame.linear_rgba16f).hexdigest(),
                diagnostics=client.runtime.diagnostics()))
            (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({'fault': fault, 'passed': True}), flush=True)
        assert not VENDORS & sys.modules.keys()
        report.update(status='passed', sdk_modules=[])
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            client.shutdown()
            report['runtime'] = client.runtime.diagnostics()
            report['resources'] = {s: resource_state(s) for s in streams}
            assert report['runtime']['closed'] and not report['runtime']['cleanup_errors']
            assert report['runtime']['returncode'] == 0
            assert not any(v for state in report['resources'].values() for v in state.values())
        except BaseException as error:
            report.update(status='failed', teardown_error=f'{type(error).__name__}: {error}')
            raise
        finally:
            (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--faults', nargs='+', choices=FAULTS, default=list(FAULTS))
    run(parser.parse_args())

"""Install the actual extension ZIP and local SDK package in an isolated profile.

GUI timers exercise the real preferences operator. --render then checks an
actual presented viewport frame. A second Blender launch with the same profile
and --reuse-installed MODULE --previous-result PATH proves receipt reuse.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import struct
import sys
import time
import tomllib
import zipfile
import zlib

import bpy

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--zip', type=Path)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--sidecar-source', type=Path, help='Sidecar ZIP or extracted package folder')
parser.add_argument('--reuse-installed', metavar='MODULE', help='Second launch: enabled extension module from the saved profile')
parser.add_argument('--previous-result', type=Path, help='Successful first-launch result.json for exact receipt/interpreter comparison')
parser.add_argument('--render', action='store_true', help='Present and save a real rendered viewport frame after setup')
parser.add_argument('--render-timeout', type=float, default=600)
parser.add_argument('--keep-open', action='store_true')
parser.add_argument('--timeout', type=float, default=2400)
parser.add_argument('--allow-existing-cache', action='store_true')
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
report = {'status': 'running', 'blender': bpy.app.version_string,
          'zip': str(args.zip.resolve()) if args.zip else None,
          'setup_states': [], 'subprocesses': [], 'restart_check': bool(args.reuse_installed)}


def save():
    (output / 'result.json').write_text(json.dumps(report, indent=2, default=str) + '\n')


def forbidden_modules():
    return sorted(name for name in sys.modules if name.split('.')[0] in
                  {'ovrtx', 'ovphysx', 'ovstage', 'ovstream', 'warp', 'srtx', 'ovphysx_bridge_client'})


def record_process(event, values):
    if event == 'subprocess.Popen':
        argv = values[1]
        report['subprocesses'].append(list(map(str, argv)) if isinstance(argv, (tuple, list)) else [str(argv)])


def assert_no_workers():
    assert not any(any('public_sdk_worker.py' in argument for argument in argv)
                   for argv in report['subprocesses']), 'Runtime worker started before engine selection'
    # Also catch children started during extension registration before this
    # fixture script ran, including on the saved-profile restart.
    if Path('/proc').is_dir():
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                if int(stat[1]) == os.getpid():
                    command = (entry / 'cmdline').read_bytes()
                    assert b'public_sdk_worker.py' not in command, command
            except (OSError, ValueError, IndexError):
                continue


def png_rgba(path, width, height, payload):
    """Save exact top-left RGBA display bytes without Blender color conversion."""
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
    stride = width * 4
    rows = b''.join(b'\x00' + payload[y * stride:(y + 1) * stride] for y in range(height))
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 6, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))


def start_viewport():
    from mathutils import Vector
    assert not bpy.app.background, '--render requires a real Blender window'
    assert_no_workers()
    report['worker_started_before_engine'] = False
    report['subprocess_count_before_engine'] = len(report['subprocesses'])
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 1))
    cube = bpy.context.object
    material = bpy.data.materials.new('Installed SDK Orange')
    material.use_nodes = True
    material.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value = (.8, .13, .025, 1)
    cube.data.materials.append(material)
    bpy.ops.mesh.primitive_plane_add(size=200)
    bpy.ops.object.light_add(type='AREA', location=(2, -3, 7))
    light = bpy.context.object
    light.data.energy, light.data.size = 1700, 5
    light.rotation_euler = (Vector((0, 0, 0)) - light.location).to_track_quat('-Z', 'Y').to_euler()
    bpy.ops.object.camera_add(location=(7, -11, 6))
    camera = bpy.context.object
    camera.rotation_euler = (Vector((0, 0, 1)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.resolution_x, scene.render.resolution_y = 640, 480
    scene.render.resolution_percentage = 100
    scene.ovrtx_example.max_samples = 8
    scene.world.color = (.05, .05, .05)
    scene.render.engine = 'OVRTX_EXAMPLE'
    areas = [area for area in bpy.context.screen.areas if area.type == 'VIEW_3D']
    assert areas, 'A visible 3D viewport is required'
    for area in areas:
        space = area.spaces.active
        space.region_3d.view_perspective = 'CAMERA'
        space.overlay.show_overlays = False
        space.shading.type = 'RENDERED'
        area.tag_redraw()
    report['viewport_started'] = True
    save()


def capture_viewport():
    engine = importlib.import_module(implementation.__name__ + '.engine')
    errors = []
    for instance in engine._live_viewport_engines():
        error = getattr(instance, '_viewport_reported_error', '')
        if error:
            errors.append(error)
        frame = getattr(instance, '_presented_frame', None)
        if frame is None or frame.render_result is None:
            continue
        result = frame.render_result
        if result.completed_samples < 2:
            continue
        width, height = result.width, result.height
        payload = result.rgba8_bytes(origin='top_left')
        assert width >= 64 and height >= 64 and len(payload) == width * height * 4
        rgb = [payload[index::4] for index in range(3)]
        assert max(max(channel) - min(channel) for channel in rgb) > 32, 'Presented frame has no useful contrast'
        orange = sum(r > g * 1.3 and r > b * 1.5 and r > 60 for r, g, b in zip(*rgb))
        assert orange > width * height * .001, 'Expected orange cube is absent from presented pixels'
        identity = {key: result.native_timings.get(key) for key in
                    ('runtime_pid', 'runtime_epoch', 'simulation_id', 'worker_load_requests', 'native_stage_id')}
        assert identity['runtime_pid'] and identity['runtime_pid'] != os.getpid(), identity
        assert identity['runtime_epoch'] and identity['native_stage_id'], identity
        png_rgba(output / 'viewport.png', width, height, payload)
        report['frame'] = dict(width=width, height=height, samples=result.completed_samples,
            publication=frame.publication_index, sha256=hashlib.sha256(payload).hexdigest(),
            render_var=result.render_var, session=identity, orange_pixels=orange,
            image=str(output / 'viewport.png'))
        return True
    report['viewport_errors'] = errors
    return False


try:
    config = Path(os.environ.get('BLENDER_USER_CONFIG', '')).expanduser()
    extensions = Path(os.environ.get('BLENDER_USER_EXTENSIONS', '')).expanduser()
    assert os.environ.get('BLENDER_USER_CONFIG') and config.is_dir(), 'Create isolated BLENDER_USER_CONFIG before Blender starts'
    assert os.environ.get('BLENDER_USER_EXTENSIONS') and extensions.is_dir(), 'Create isolated BLENDER_USER_EXTENSIONS before Blender starts'
    actual_config = Path(bpy.utils.user_resource('CONFIG')).resolve()
    assert actual_config == config.resolve(), f'Blender is using {actual_config}, not isolated config {config}'
    report['config_directory'] = str(actual_config)
    if os.environ.get('OV_BLENDER_PUBLIC_SDK_PYTHON', '').strip():
        raise AssertionError('Managed-install proof must not select an external SDK interpreter')
    assert not forbidden_modules(), forbidden_modules()
    sys.addaudithook(record_process)
    assert_no_workers()
    if args.reuse_installed:
        assert args.previous_result, '--previous-result is required for the restart proof'
        previous_report = json.loads(args.previous_result.read_text())
        assert previous_report['status'] == 'passed', previous_report
        module_name = args.reuse_installed
        assert module_name == previous_report['module']
        assert bpy.context.preferences.addons.get(module_name), 'Saved preferences did not enable the extension'
        wrapper = importlib.import_module(module_name)
        installed_root = Path(wrapper.__file__).resolve().parent
        package_mode = json.loads((installed_root / 'public-sdk.json').read_text()).get('install_mode', 'automatic')
    else:
        assert args.zip, '--zip is required for the fresh install'
        with zipfile.ZipFile(args.zip) as archive:
            package_mode = json.loads(archive.read('public-sdk.json')).get('install_mode', 'automatic')
            manifest = tomllib.loads(archive.read('blender_manifest.toml').decode())
            assert 'runtime/public_sdk_worker.py' in archive.namelist()
        repo_dir = output / 'extension-repository'
        assert not repo_dir.exists() or not any(repo_dir.iterdir()), 'Install proof requires an empty extension repository'
        repo_dir.mkdir(exist_ok=True)
        bpy.context.preferences.system.use_online_access = package_mode != 'local_package'
        repo = bpy.context.preferences.extensions.repos.new(name='Public SDK Install Proof',
            module='public_sdk_install_proof', custom_directory=str(repo_dir), remote_url='', source='USER')
        assert not repo.use_remote_url
        installed = bpy.ops.extensions.package_install_files('EXEC_DEFAULT', filepath=str(args.zip.resolve()),
            repo=repo.module, enable_on_install=True)
        assert installed == {'FINISHED'}, installed
        module_name = f'bl_ext.{repo.module}.{manifest["id"]}'
        wrapper = importlib.import_module(module_name)
        installed_root = Path(wrapper.__file__).resolve().parent
        assert installed_root.is_relative_to(repo_dir)
    implementation = wrapper._impl
    assert implementation.ADDON_PREFERENCES_ID == module_name, implementation.ADDON_PREFERENCES_ID
    assert bpy.context.preferences.addons.get(module_name) is not None
    assert implementation.get_addon_preferences(bpy.context) is not None
    installer = importlib.import_module(implementation.__name__ + '.public_sdk_install')
    client = importlib.import_module(implementation.__name__ + '.direct_sdk_client')
    status = installer.status()
    report.update(module=module_name, installed_root=str(installed_root), initial_setup=dict(status),
                  selected_public_mode=client.enabled())
    assert client.enabled(), 'Installed extension did not select public SDK mode'
    assert installer._manager._config is not None, 'Register did not configure managed setup'
    assert client.worker_script() == installed_root / 'runtime' / 'public_sdk_worker.py'
    if args.reuse_installed:
        assert status['state'] == 'ready', status
        assert installer.ready_python() == previous_report['ready_python'], 'Restart selected a different SDK environment'
        assert not report['subprocesses'], 'Restart unexpectedly started setup work'
    elif not args.allow_existing_cache:
        assert status['state'] != 'ready', 'Fresh proof unexpectedly reused ready SDK evidence'
    if package_mode == 'local_package' and status['state'] != 'ready':
        assert status['state'] != 'installing', 'Local sidecar setup started before selection'
        assert args.sidecar_source and args.sidecar_source.exists(), '--sidecar-source is required'
        report['sdk_setup_started_before_selection'] = False
        implementation.get_addon_preferences(bpy.context).runtime_source = str(args.sidecar_source.resolve())
        result = bpy.ops.ovrtx_example.install_runtime('EXEC_DEFAULT')
        assert result == {'FINISHED'}, result
        report['install_operator_result'] = sorted(result)
    deadline = time.monotonic() + args.timeout
    previous = None
    phase = 'setup'
    save()

    def finish():
        save()
        if not args.keep_open and not bpy.app.background:
            bpy.app.timers.register(lambda: bpy.ops.wm.quit_blender() and None, first_interval=.1)

    def poll():
        global previous, phase, deadline
        try:
            assert not forbidden_modules(), forbidden_modules()
            if phase == 'render':
                if capture_viewport():
                    report.update(status='passed', native_modules_in_blender=forbidden_modules())
                    finish()
                    print('PUBLIC_SDK_EXTENSION_INSTALL_RENDER_PASS', flush=True)
                    return None
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'No valid installed-extension viewport frame: {report.get("viewport_errors")}')
                save()
                return .2
            status = installer.status()
            state = status['state']
            if state != previous:
                report['setup_states'].append(dict(status))
                print(json.dumps({'managed_sdk_setup': state, 'message': status.get('message', '')}), flush=True)
                previous = state
                save()
            assert_no_workers()
            if state == 'ready':
                python = Path(installer.ready_python())
                assert python.is_file() and not python.is_relative_to(installed_root), python
                receipt = installer._manager._config[0] / 'ready.json'
                assert receipt.is_file(), receipt
                receipt_hash = hashlib.sha256(receipt.read_bytes()).hexdigest()
                if args.reuse_installed:
                    assert receipt_hash == previous_report['ready_receipt_sha256'], 'Restart rewrote the ready receipt'
                report.update(ready_python=str(python), final_setup=dict(status), ready_receipt=str(receipt),
                    ready_receipt_sha256=receipt_hash, native_modules_in_blender=forbidden_modules(),
                    worker_started_before_engine=False)
                # Verify again immediately before the only preference write.
                assert Path(bpy.utils.user_resource('CONFIG')).resolve() == config.resolve()
                bpy.ops.wm.save_userpref()
                assert (config / 'userpref.blend').is_file(), 'Preferences were not saved in the isolated profile'
                if args.render:
                    phase = 'render'
                    deadline = time.monotonic() + args.render_timeout
                    start_viewport()
                    return .2
                report['status'] = 'passed'
                finish()
                print('PUBLIC_SDK_EXTENSION_INSTALL_PASS', flush=True)
                return None
            if state in {'failed', 'error', 'cancelled', 'offline', 'offline_blocked'}:
                raise AssertionError(f'Managed SDK installation did not complete: {status}')
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Managed SDK installation timed out: {status}')
            return .1
        except BaseException as error:
            report.update(status='failed', error=f'{type(error).__name__}: {error}')
            finish()
            print('PUBLIC_SDK_EXTENSION_INSTALL_FAIL', report['error'], flush=True)
            return None

    if bpy.app.background:
        while poll() is not None:
            time.sleep(.1)
        if report['status'] != 'passed':
            raise AssertionError(report['error'])
    else:
        bpy.app.timers.register(poll, first_interval=.1)
except BaseException as error:
    report.update(status='failed', error=f'{type(error).__name__}: {error}')
    save()
    if not args.keep_open and not bpy.app.background:
        bpy.app.timers.register(lambda: bpy.ops.wm.quit_blender() and None, first_interval=.1)
    raise

# SPDX-License-Identifier: Apache-2.0
"""Background, stdlib-only provisioning of the packaged public SDK sidecar.

Blender supplies paths and its online-access permission on the main thread.
No SDK is imported here; a disposable external interpreter verifies readiness.
"""
from __future__ import annotations

import hashlib
import errno
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

_MIN_FREE_BYTES = 8 * 1024**3
_WINDOWS = os.name == 'nt'
_MODULES = {'ovrtx': 'ovrtx', 'ovstage': 'ovstage', 'ovphysx': 'ovphysx',
            'ovstream': 'ovstream', 'numpy': 'numpy', 'warp-lang': 'warp', 'packaging': 'packaging'}
# Schema data only: importing it registers with a stock USD runtime the SDK
# interpreter does not have, so verification checks its version, not an import.
_DATA_ONLY = frozenset(('physx-usd-schemas',))
_GATE = """import sys, runpy
if sys.stdin.buffer.read(1) != b'G': raise SystemExit(125)
sys.stdin.close()
args = sys.argv[1:]
if args[0] == '-m':
    sys.argv = args[1:]
    runpy.run_module(args[1], run_name='__main__', alter_sys=True)
elif args[0] == '-c':
    sys.argv = ['-c', *args[2:]]
    exec(compile(args[1], '<sidecar-setup>', 'exec'), {'__name__':'__main__'})
else: raise SystemExit(126)
"""


class _Cancelled(Exception):
    pass


class _WindowsJob:
    """Own nested venv/ensurepip processes before releasing their stdin gate."""
    def __init__(self):
        import ctypes as c
        from ctypes import wintypes as w
        self.c = c
        self.members = []
        self.api = c.WinDLL('kernel32', use_last_error=True)
        class Basic(c.Structure):
            _fields_ = [('process_time', c.c_int64), ('job_time', c.c_int64),
                ('flags', w.DWORD), ('min_working', c.c_size_t), ('max_working', c.c_size_t),
                ('active', w.DWORD), ('affinity', c.c_size_t), ('priority', w.DWORD), ('schedule', w.DWORD)]
        class Extended(c.Structure):
            _fields_ = [('basic', Basic), ('io', c.c_uint64 * 6),
                ('process_memory', c.c_size_t), ('job_memory', c.c_size_t),
                ('peak_process', c.c_size_t), ('peak_job', c.c_size_t)]
        for name, args, result in (
            ('CreateJobObjectW', [c.c_void_p, w.LPCWSTR], w.HANDLE),
            ('SetInformationJobObject', [w.HANDLE, c.c_int, c.c_void_p, w.DWORD], w.BOOL),
            ('AssignProcessToJobObject', [w.HANDLE, w.HANDLE], w.BOOL),
            ('TerminateJobObject', [w.HANDLE, w.UINT], w.BOOL),
            ('QueryInformationJobObject', [w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.c_void_p], w.BOOL),
            ('OpenProcess', [w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            ('WaitForSingleObject', [w.HANDLE, w.DWORD], w.DWORD),
            ('CloseHandle', [w.HANDLE], w.BOOL)):
            function = getattr(self.api, name); function.argtypes = args; function.restype = result
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise c.WinError(c.get_last_error())
        limits = Extended(); limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, c.byref(limits), c.sizeof(limits)):
            error = c.WinError(c.get_last_error()); self.close(); raise error

    def assign(self, process):
        if not self.api.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise self.c.WinError(self.c.get_last_error())

    def terminate(self):
        if not self.api.TerminateJobObject(self.handle, 125):
            raise self.c.WinError(self.c.get_last_error())

    def active(self):
        data = self.c.create_string_buffer(48)
        if not self.api.QueryInformationJobObject(self.handle, 1, data, len(data), None):
            raise self.c.WinError(self.c.get_last_error())
        return int.from_bytes(data.raw[40:44], 'little')

    def close(self):
        errors = []
        if self.handle:
            if not self.api.CloseHandle(self.handle):
                errors.append(repr(self.c.WinError(self.c.get_last_error())))
            else:
                self.handle = None
        for handle in tuple(self.members):
            if self.api.CloseHandle(handle):
                self.members.remove(handle)
            else:
                errors.append(repr(self.c.WinError(self.c.get_last_error())))
        if errors:
            raise RuntimeError('Setup handle close failed: '+ '; '.join(errors))

    def pin_members(self):
        capacity = 64
        while True:
            width = self.c.sizeof(self.c.c_size_t)
            data = self.c.create_string_buffer(8+capacity*width)
            if self.api.QueryInformationJobObject(self.handle, 3, data, len(data), None):
                break
            if self.c.get_last_error() != 234 or capacity >= 65536:
                raise self.c.WinError(self.c.get_last_error())
            capacity *= 2
        count = int.from_bytes(data.raw[4:8], 'little')
        for index in range(count):
            pid = int.from_bytes(data.raw[8+index*width:8+(index+1)*width], 'little')
            handle = self.api.OpenProcess(0x100000, False, pid)
            if handle:
                self.members.append(handle)
            elif self.c.get_last_error() != 87:
                raise self.c.WinError(self.c.get_last_error())

    def wait_members(self, deadline):
        for handle in self.members:
            if self.api.WaitForSingleObject(handle, max(0, int((deadline-time.monotonic())*1000))) != 0:
                raise RuntimeError('Setup descendant process death not confirmed')


def _pins(path):
    from .public_sdk_lock import versions, load_lock, lock_identity, LOCK_NAME
    result = versions(path)
    identity = lock_identity(load_lock(path.with_name(LOCK_NAME)))['lock_sha256']
    return result, hashlib.sha256(path.read_bytes() + identity.encode('ascii')).hexdigest()


def _input_stamp(requirements, environment=None):
    """Size and mtime of every input an installed runtime was verified against.

    Startup compares this metadata instead of hashing. Any change means the
    add-on or runtime changed, which is the only time hashing runs again.
    """
    from .public_sdk_lock import LOCK_NAME
    requirements = Path(requirements)
    paths = dict(requirements=requirements, lock=requirements.with_name(LOCK_NAME),
                 addon_pin=requirements.parent.parent / 'public-sdk.json')
    if environment is not None:
        paths['sidecar'] = Path(environment) / 'sidecar.json'
    stamp = {}
    for key, path in paths.items():
        try:
            info = path.stat()
        except FileNotFoundError:
            stamp[key] = None
        else:
            stamp[key] = [info.st_size, info.st_mtime_ns]
    return stamp


def _environment_python(directory):
    return directory / ('Scripts/python.exe' if _WINDOWS else 'bin/python')


class Installer:
    def __init__(self):
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._thread = None
        self._config = None
        self._store = None
        self._cleanup_safe = True
        self._state = dict(state='unconfigured', current_python=None, message='', progress=0.0, error='')

    def configure(self, storage_root, python_exe, requirements):
        store = Path(storage_root).expanduser().resolve()
        # Native code needs an ASCII folder: a non-ASCII settings store keeps its
        # runtime under the mapped root recorded in it (created by start()).
        from .native_paths import native_storage_root
        root = native_storage_root(store)
        python = Path(python_exe).expanduser().absolute()  # Never resolve a venv interpreter symlink.
        requirements = Path(requirements).expanduser().resolve()
        from .public_sdk_lock import versions
        # The requirements digest is computed only when installing or when the
        # recorded input stamp shows the add-on or runtime changed.
        config = (root, str(python), requirements, None, versions(requirements))
        with self._lock:
            if self._config == config and self._thread is not None and self._thread.is_alive():
                return
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError('Cannot reconfigure an active sidecar installation')
            self._config = config
            self._store = store
            current = self._read_receipt(config)
            incomplete = '' if current else self._incomplete_root(root, store)
            self._state = dict(state='ready' if current else 'unconfigured', current_python=current,
                message='Public SDK runtime is ready' if current else (
                    'An unfinished runtime folder remains; Remove Runtime deletes it' if incomplete
                    else 'Public SDK runtime needs setup'),
                progress=1.0 if current else 0.0, error='', incomplete_root=incomplete)

    @staticmethod
    def _incomplete_root(root, store):
        """A mapped root this settings folder owns that holds no committed runtime, or ''."""
        if store is None or root == store or os.path.lexists(root / 'ready.json'):
            return ''
        from .native_paths import recorded_root
        return str(root) if recorded_root(store) == root else ''

    @staticmethod
    def _read_receipt(config):
        root, _, requirements, _, pins = config
        try:
            from .public_sdk_sidecar import no_reparse_owned, installed_python
            receipt_path = no_reparse_owned(root / 'ready.json', root)
            if receipt_path.stat().st_size > 16384:
                return None
            receipt = json.loads(receipt_path.read_text())
            directory = Path(receipt['environment'])
            relative_python = None
            if receipt.get('schema') == 2:
                from .public_sdk_sidecar import package_path
                directory = root.joinpath(*package_path(receipt['environment_relative']).parts)
                relative_python = root.joinpath(*package_path(receipt['python_relative']).parts)
            if receipt.get('schema') not in (1, 2) or receipt.get('versions') != pins or directory.is_symlink():
                return None
            local_package = receipt.get('source_kind', 'automatic') == 'local_package'
            stamp = _input_stamp(requirements, directory if local_package else None)
            unchanged = relative_python is not None and receipt.get('input_stamp') == stamp
            # The stored path is immutable after venv creation. Avoid accepting
            # an arbitrary executable injected via a stale/malformed receipt.
            if not unchanged and receipt.get('requirements_sha256') != _pins(requirements)[1]:
                return None
            if local_package:
                if directory.parent != root / 'sidecars' or not re.fullmatch(r'[0-9a-f]{32}', directory.name):
                    return None
                if unchanged:
                    # Installation validated the manifest pin and payload, and
                    # nothing it checked has changed since: no hashing here.
                    python = no_reparse_owned(relative_python, root)
                else:
                    python = installed_python(directory, root, requirements, verify_files=False)
                if (relative_python or Path(receipt['python'])) != python:
                    return None
            else:
                if directory.parent != root / 'environments' or not re.fullmatch(r'env-[0-9a-f]{32}', directory.name):
                    return None
                python = _environment_python(directory)
                if relative_python is not None and relative_python != python:
                    return None
                if _WINDOWS:
                    no_reparse_owned(python, root)
            if (relative_python is None and receipt.get('python') != str(python)) or not python.is_file():
                return None
            if not unchanged and relative_python is not None:
                # Inputs changed and were re-verified above; record them so the
                # next start compares metadata only.
                receipt['input_stamp'] = stamp
                temporary = root / ('ready-' + uuid.uuid4().hex + '.tmp')
                temporary.write_text(json.dumps(receipt, indent=2) + '\n')
                os.replace(temporary, receipt_path)
            return str(python)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def status(self):
        with self._lock:
            return dict(self._state)

    def ready_python(self):
        with self._lock:
            return self._state['current_python']

    def ovrtx_cache_root(self):
        """Persistent OVRTX cache beside the sidecars, one directory per version."""
        with self._lock:
            if self._config is None:
                return None
            root, versions = self._config[0], self._config[4]
        return root / 'cache' / ('ovrtx-' + versions['ovrtx'])

    def _publish(self, **values):
        with self._lock:
            self._state.update(values)

    def start(self, online_access=True, source=''):
        with self._lock:
            if self._config is None:
                raise RuntimeError('Configure the public SDK installer first')
            if self._state['current_python'] and not source:
                return False
            if self._thread is not None and self._thread.is_alive():
                return False
            if not self._cleanup_safe:
                self._state.update(state='failed', error='Previous setup process teardown was not confirmed; restart Blender before retrying')
                return False
            if not online_access and not source:
                self._state.update(state='offline', message='Allow online access in Blender to download the runtime', error='')
                return False
            from .native_paths import native_storage_root, recorded_root
            try:
                mapped_before = recorded_root(self._store) is not None
                root = native_storage_root(self._store, create=True)
            except ValueError as error:
                self._state.update(state='failed', message='Unsupported Windows runtime location', error=str(error))
                return False
            if root != self._config[0]:
                self._config = (root, *self._config[1:])
            # A mapped root made for this install goes again if the install fails.
            created = root != self._store and not mapped_before
            self._cancel = threading.Event()
            self._state.update(state='installing', message='Preparing public SDK runtime', progress=0.0, error='',
                               incomplete_root='')
            self._thread = threading.Thread(target=self._install, args=(self._config, str(source)),
                                            kwargs=dict(store=self._store, created=created),
                                            name='ov-public-sdk-install', daemon=True)
            self._thread.start()
            return True

    def cancel(self):
        with self._lock:
            self._cancel.set()

    def remove(self):
        """Remove the selected packaged candidate after the UI closes its owners.

        External interpreters and automatically provisioned environments are not
        package-owned payloads. This operation never follows their selection.
        """
        with self._lock:
            if os.environ.get('OV_BLENDER_PUBLIC_SDK_PYTHON', '').strip():
                raise RuntimeError('External interpreter files are not managed by this add-on')
            if self._config is None:
                raise RuntimeError('Configure the public SDK installer first')
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError('Runtime setup or removal is still active')
            if not self._cleanup_safe:
                raise RuntimeError('Previous process teardown was not confirmed; restart Blender')
            previous = self._state['current_python']
            self._state.update(state='removing', current_python=None,
                message='Verifying runtime before removal', progress=0.0, error='')
            self._thread = threading.Thread(target=self._remove,
                args=(self._config, previous, self._store), name='ov-public-sdk-remove', daemon=True)
            try:
                self._thread.start()
            except Exception:
                self._state.update(state='failed', current_python=previous)
                raise
            return True

    def _remove(self, config, previous, store=None):
        from . import public_sdk_sidecar as sidecar
        root, _, requirements, _, _ = config
        ready = root/'ready.json'
        candidate = quarantine = saved_ready = None
        moved = False
        validated = False
        try:
            if store is not None and root != store and not os.path.lexists(ready):
                # An install that did not finish left a mapped root with no runtime
                # selected: remove it (this settings folder must own it; no links).
                from .native_paths import remove_mapped_root
                remove_mapped_root(store, root)
                self._publish(state='unconfigured', current_python=None, progress=1.0, incomplete_root='',
                    message='Unfinished runtime folder removed. Choose a package to install again.',
                    retained_path='', error='')
                return
            sidecar.no_reparse_owned(ready, root)
            if ready.stat().st_size > 16384:
                raise ValueError('Runtime receipt is too large')
            receipt_bytes = ready.read_bytes()
            receipt = json.loads(receipt_bytes)
            if receipt.get('source_kind') != 'local_package':
                raise ValueError('Only an installed local runtime package can be removed')
            # Require the addon-owned pin, not a candidate's self-declared hash.
            marker = requirements.parent.parent/'public-sdk.json'
            pin = json.loads(marker.read_text()).get('sidecar_manifest_sha256')
            if not isinstance(pin, str) or not re.fullmatch('[0-9a-f]{64}', pin):
                raise ValueError('Runtime removal requires the paired add-on package pin')
            python = self._read_receipt(config)
            if python is None:
                raise ValueError('Installed runtime failed integrity checks; nothing was removed')
            candidate = root.joinpath(*sidecar.package_path(receipt['environment_relative']).parts)
            manifest_bytes = sidecar._filesystem_path(candidate/'sidecar.json').read_bytes()
            if hashlib.sha256(manifest_bytes).hexdigest() != pin:
                raise ValueError('Runtime manifest changed during removal')
            # Receipt resolution is intentionally metadata-only at startup.
            # Removal must verify the complete owned payload before either
            # candidate or selector is moved at the deactivation boundary.
            if sidecar.installed_python(candidate, root, requirements, verify_files=True) != Path(python):
                raise ValueError('Installed runtime interpreter changed during removal')
            sidecar.validate_pair(manifest_bytes, requirements)
            manifest = sidecar.validate_manifest(sidecar._manifest_json(manifest_bytes), requirements)
            cache = sidecar.installed_cache_inventory(candidate, root, manifest)
            owned = dict(manifest['files'])
            owned.update({name: row['sha256'] for name, row in cache['files'].items()})
            owned['sidecar.json'] = hashlib.sha256(manifest_bytes).hexdigest()
            # Freeze the selector before the reversible deactivation boundary.
            if ready.read_bytes() != receipt_bytes:
                raise ValueError('Runtime selection changed during removal')
            validated = True
            token = uuid.uuid4().hex
            quarantine = root/'sidecars'/('removed-'+token)
            saved_ready = root/('removed-'+token+'.json')
            sidecar.no_reparse_owned(quarantine, root)
            os.rename(candidate, quarantine)
            moved = True
            try:
                os.rename(ready, saved_ready)
            except Exception:
                os.rename(quarantine, candidate)
                moved = False
                raise
            self._publish(message='Removing verified runtime files', progress=.5)
            # Never recursively delete: newly added names, empty user folders,
            # sibling candidates, logs, and external environments stay intact.
            directories = set()
            for name, expected in owned.items():
                path = quarantine.joinpath(*sidecar.package_path(name).parts)
                sidecar.no_reparse_owned(path, root)
                with sidecar._filesystem_path(path).open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                        raise ValueError('Runtime file changed during removal: '+name)
                sidecar._filesystem_path(path).unlink()
                directories.update(parent for parent in path.parents if parent != quarantine and quarantine in parent.parents)
            for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
                sidecar.no_reparse_owned(directory, root)
                try:
                    sidecar._filesystem_path(directory).rmdir()
                except OSError as error:
                    if error.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                        raise
                    # Preserve newly created/unknown entries, never rmtree.
            sidecar.no_reparse_owned(quarantine, root)
            retained = ''
            try:
                quarantine.rmdir()
            except OSError as error:
                if error.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                    raise
                retained = str(quarantine)
            sidecar.no_reparse_owned(saved_ready, root)
            saved_ready.unlink()
            if store is not None and root != store:
                # A mapped root exists only for this install: its caches, logs and
                # any earlier candidates go with the runtime, then its record.
                from .native_paths import remove_mapped_root
                remove_mapped_root(store, root)
                retained = ''
            self._publish(state='unconfigured', current_python=None, progress=1.0,
                message=('Runtime payload removed; additional entries were preserved at '+retained
                    if retained else 'Runtime removed. Choose a package to install again.'),
                retained_path=retained, error='')
        except Exception as error:
            self._publish(state='failed', current_python=None if moved or not validated else previous,
                message='Runtime is disabled; removal did not finish' if moved else 'Runtime removal refused',
                error=str(error)+(f'; remaining files: {quarantine}' if moved else ''))

    def close(self):
        # Add-on unregister must not block Blender's main thread on pip/native
        # teardown. The worker observes this event every 100ms and reaps its tree.
        self.cancel()

    def _check_cancel(self):
        if self._cancel.is_set():
            raise _Cancelled()

    def _run(self, command, *, log, timeout_s):
        self._check_cancel()
        job = None
        process = None
        env = dict(os.environ)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        for name in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV'):
            env.pop(name, None)
        if self._config:
            temporary = self._config[0] / 'tmp'
            temporary.mkdir(parents=True, exist_ok=True)
            env.update(TEMP=str(temporary), TMP=str(temporary), TMPDIR=str(temporary))
        try:
            job = _WindowsJob() if _WINDOWS else None
            options = dict(stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                           close_fds=True, env=env)
            if _WINDOWS:
                options.update(stdin=subprocess.PIPE, bufsize=0, creationflags=subprocess.CREATE_NO_WINDOW | 4)
                launch = [command[0], '-c', _GATE, *command[1:]]
            else:
                options['start_new_session'] = True
                launch = command
            process = subprocess.Popen(launch, **options)
            if job is not None:
                job.assign(process)
                resume = job.c.WinDLL('ntdll').NtResumeProcess
                resume.argtypes = [job.c.c_void_p]; resume.restype = job.c.c_long
                if resume(int(process._handle)) < 0:
                    raise OSError('Could not resume owned setup launcher')
                process.stdin.write(b'G'); process.stdin.close()
            deadline = time.monotonic() + timeout_s
            while True:
                self._check_cancel()
                code = process.poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f'Runtime setup command failed (exit {code}); see setup log')
                    if job is not None:
                        terminal_deadline = min(deadline, time.monotonic() + 15)
                        while job.active():
                            if time.monotonic() >= terminal_deadline:
                                raise RuntimeError('Setup descendants did not exit naturally')
                            self._cancel.wait(.05)
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError('Runtime setup command exceeded its time limit')
                self._cancel.wait(.1)
        finally:
            if job is not None:
                self._finish_windows_setup(job, process, sys.exception())
                # The Windows helper owns every cleanup operation and always
                # closes the guard. Do not enter the POSIX cleanup below.
                process = None
            try:
                if process is not None and process.poll() is None:
                    if job is not None:
                        # Closing the job also catches descendants if assigning
                        # or signalling the child failed before its gate opened.
                        job.close()
                        process.kill()
                    else:
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        if job is None:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        else:
                            process.kill()
                        process.wait(timeout=2)
                # A venv/ensurepip child may outlive its reaped parent. Kill the
                # remaining owned group even when the direct child has exited.
                if process is not None and job is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            except Exception:
                self._cleanup_safe = False
                raise
            finally:
                if process is not None and process.stdin is not None:
                    process.stdin.close()
                if job is not None:
                    try:
                        job.close()
                    except Exception:
                        self._cleanup_safe = False
                        raise

    def _finish_windows_setup(self, job, process, primary):
        errors = []
        try:
            if process is not None:
                try:
                    job.pin_members()
                except Exception as exc:
                    errors.append(repr(exc))
                needs_termination = True
                try:
                    needs_termination = process.poll() is None or job.active() != 0
                except Exception as exc:
                    errors.append(repr(exc))
                if needs_termination:
                    try:
                        job.terminate()
                    except Exception as exc:
                        errors.append(repr(exc))
                    try:
                        if process.poll() is None:
                            process.kill()  # Also handles failed Job assignment.
                    except Exception as exc:
                        errors.append(repr(exc))
                try:
                    process.wait(timeout=3)
                    deadline = time.monotonic()+3
                    while job.active():
                        if time.monotonic() >= deadline:
                            raise RuntimeError('Setup Job teardown not confirmed')
                        time.sleep(.01)
                    job.wait_members(deadline)
                except Exception as exc:
                    errors.append(repr(exc))
        finally:
            try:
                job.close()
            except Exception as exc:
                errors.append(repr(exc))
            if process is not None and process.stdin is not None:
                try:
                    process.stdin.close()
                except Exception as exc:
                    errors.append(repr(exc))
        if errors:
            self._cleanup_safe = False
            if primary is not None:
                primary.add_note('Setup cleanup errors: '+ '; '.join(errors))
                raise primary
            raise RuntimeError('Setup cleanup errors: '+ '; '.join(errors))

    def _install(self, config, source='', *, store=None, created=False):
        root, base_python, requirements, _, pins = config
        directory = root / 'environments' / ('env-' + uuid.uuid4().hex)
        log_path = root / 'logs' / (directory.name + '.log')
        committed = False
        temporary = None
        try:
            self._check_cancel()
            root.mkdir(parents=True, exist_ok=True)
            if not source and shutil.disk_usage(root).free < _MIN_FREE_BYTES:
                raise RuntimeError('Public SDK setup needs at least 8 GiB of free disk space')
            if not source and not Path(base_python).is_file():
                raise RuntimeError('Blender bundled Python was not found')
            # Installation is when pins are hashed; hash again before pip
            # consumes the file to reject changing package pins in flight.
            current_pins, digest = _pins(requirements)
            if current_pins != pins:
                raise RuntimeError('Public SDK requirements changed during setup')
            directory.parent.mkdir(parents=True, exist_ok=True)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open('ab', buffering=0) as log:
                self._publish(message='Preparing isolated runtime', progress=.05, log_path=str(log_path))
                if source:
                    from .public_sdk_sidecar import materialize_local
                    def progress(message, completed, total):
                        self._check_cancel()
                        self._publish(message=message, progress=.05 + .8 * completed / max(1, total))
                    local = materialize_local(source, root, requirements,
                                              cancelled=self._cancel.is_set, progress=progress)
                    directory = Path(local['root'])
                    python = str(local['python'])
                else:
                    self._run([base_python, '-m', 'venv', str(directory)], log=log, timeout_s=300)
                    python = str(_environment_python(directory))
                    self._publish(message='Downloading and installing public SDKs', progress=.15)
                    entry = requirements.with_name('materialize_sdk.py')
                    install_code = ('import runpy,sys; '
                        f'sys.argv=[{str(entry)!r},"install","--cache",{str(root / "cache")!r}]; '
                        f'runpy.run_path({str(entry)!r},run_name="__main__")')
                    self._run([python, '-c', install_code],
                        log=log, timeout_s=1800)
                self._publish(message='Checking the installed runtime', progress=.9)
                result_path = root / 'logs' / (directory.name + '-verified.json')
                policy_path = Path(__file__).with_name('public_sdk_lock.py')
                code = ('import importlib,importlib.metadata,json,sys,runpy; from pathlib import Path; '
                    f'policy=runpy.run_path({str(policy_path)!r}); '
                    f'identity=policy["inspect_installed"](policy["load_lock"]({str(requirements.with_name("public-sdk-lock.json"))!r}), verify_files=True); '
                    f'pins={pins!r}; modules={_MODULES!r}; '
                    'versions={name:importlib.metadata.version(name) for name in pins}; '
                    'assert versions==pins, (versions,pins); '
                    'assert sys.version_info[:2]==(3,13), sys.version; '
                    '[importlib.import_module(modules[name]) for name in pins if name in modules]; '
                    f'Path({str(result_path)!r}).write_text(json.dumps(versions))')
                self._run([python, '-c', code], log=log, timeout_s=180)
                self._check_cancel()
                if json.loads(result_path.read_text()) != pins or _pins(requirements) != (pins, digest):
                    raise RuntimeError('Public SDK verification did not match packaged requirements')
                receipt = dict(schema=2, requirements_sha256=digest, environment=str(directory),
                               environment_relative=directory.relative_to(root).as_posix(),
                               python_relative=Path(python).relative_to(root).as_posix(),
                               python=python, versions=pins, source_kind='local_package' if source else 'automatic',
                               input_stamp=_input_stamp(requirements, directory if source else None))
                temporary = root / ('ready-' + uuid.uuid4().hex + '.tmp')
                temporary.write_text(json.dumps(receipt, indent=2) + '\n')
                # Cancellation and commit form one ordered decision. A cancel
                # after this commit preserves a successfully verified runtime.
                with self._lock:
                    self._check_cancel()
                    os.replace(temporary, root / 'ready.json')
                    committed = True
                    self._state.update(state='ready', current_python=python,
                        message='Public SDK runtime is ready', progress=1.0, error='')
        except _Cancelled:
            self._publish(state='cancelled', message='Runtime setup cancelled', error='')
        except Exception as error:
            if not self._cleanup_safe:
                self._publish(state='failed', message='Runtime setup could not confirm process cleanup',
                              error=f'{error}; restart Blender before retrying')
            elif self._cancel.is_set():
                self._publish(state='cancelled', message='Runtime setup cancelled', error='')
            else:
                self._publish(state='failed', message='Public SDK runtime setup failed', error=str(error))
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as error:
                    self._publish(error=self.status()['error'] + f' (receipt cleanup: {error})')
            if not committed and self._cleanup_safe and directory.exists():
                try:
                    shutil.rmtree(directory)
                except OSError as error:
                    self._publish(error=self.status()['error'] + f' (incomplete runtime cleanup: {error})')
            if not committed and self._cleanup_safe and created:
                # As install.bat does: a failed install removes the mapped root it made.
                try:
                    from .native_paths import remove_mapped_root
                    remove_mapped_root(store, root)
                    self._publish(log_path='')
                except (OSError, ValueError) as error:
                    self._publish(error=self.status()['error'] + f' (runtime folder cleanup: {error})')
            if not committed:
                self._publish(incomplete_root=self._incomplete_root(root, store))


_manager = Installer()
configure = _manager.configure
status = _manager.status
start = _manager.start
cancel = _manager.cancel
remove = _manager.remove
ready_python = _manager.ready_python
ovrtx_cache_root = _manager.ovrtx_cache_root
close = _manager.close

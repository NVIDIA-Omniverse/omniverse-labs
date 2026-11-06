import ctypes
import ctypes.util
import errno
import os
import types
import unittest
from unittest.mock import patch
import ast
from pathlib import Path
import uuid

# Execute these actual stdlib-only platform functions without importing vendor
# packages or launching a native producer on the Windows contract-test host.
source_path = Path(__file__).resolve().parents[1] / 'runtime/ovstream_frames.py'
tree = ast.parse(source_path.read_bytes())
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {'_posix_shm_api', '_posix_available_memory', '_check_allocation'}]
assert len(functions) == 3
subject = types.ModuleType('publisher_platform_contracts')
subject.__dict__.update(ctypes=ctypes, os=os, uuid=uuid,
    sys=types.SimpleNamespace(platform='linux'), _RING_SLOTS=2)
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source_path), 'exec'), subject.__dict__)


class CapacityTests(unittest.TestCase):
    def run_probe(self, *, opened=42, unlink_results=(0,), stats=None,
                  stat_error=None, close_error=None):
        calls = []
        results = iter(unlink_results)
        def create(name, flags, mode):
            calls.append(('open', name, flags, mode))
            return opened
        def unlink(name):
            calls.append(('unlink', name))
            return next(results)
        def stat(fd):
            calls.append(('stat', fd))
            if stat_error: raise stat_error
            return stats or types.SimpleNamespace(f_bavail=17, f_frsize=4096)
        def close(fd):
            calls.append(('close', fd))
            if close_error: raise close_error
        self.calls = calls
        with patch.object(subject, '_posix_shm_api', return_value=(create, unlink)), \
             patch.object(subject.os, 'fstatvfs', stat, create=True), \
             patch.object(subject.os, 'close', close), \
             patch.object(subject.os, 'statvfs', side_effect=AssertionError('No mount-path lookup'), create=True):
            return subject._posix_available_memory()

    def test_namespace_capacity_and_immediate_unlink(self):
        self.assertEqual(self.run_probe(), 69632)
        self.assertEqual([row[0] for row in self.calls], ['open', 'unlink', 'stat', 'close'])
        name, flags, mode = self.calls[0][1:]
        self.assertRegex(name.decode(), r'^/blender-ovstream-capacity-[0-9a-f]{32}$')
        self.assertEqual((flags, mode), (os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600))
        self.assertEqual(self.calls[1][1], name)

    def test_collision_never_unlinks_existing_name(self):
        ctypes.set_errno(errno.EEXIST)
        with self.assertRaises(OSError) as error: self.run_probe(opened=-1)
        self.assertEqual(error.exception.errno, errno.EEXIST)
        self.assertEqual(len(self.calls), 1)

    def test_unlink_failure_refuses_even_after_cleanup_retry(self):
        ctypes.set_errno(errno.EACCES)
        with self.assertRaises(PermissionError): self.run_probe(unlink_results=(-1, 0))
        self.assertEqual([row[0] for row in self.calls], ['open', 'unlink', 'unlink', 'close'])

    def test_stat_error_preserved_and_descriptor_closed(self):
        error = OSError('capacity unavailable')
        with self.assertRaises(OSError) as observed: self.run_probe(stat_error=error)
        self.assertIs(observed.exception, error)
        self.assertEqual(self.calls[-1], ('close', 42))

    def test_cleanup_errors_preserve_primary_and_both_actions(self):
        ctypes.set_errno(errno.EACCES)
        with self.assertRaises(PermissionError) as observed:
            self.run_probe(unlink_results=(-1, -1), close_error=OSError('close failed'))
        self.assertEqual(len(observed.exception.__notes__), 2)
        self.assertEqual(self.calls[-1], ('close', 42))

    def test_close_failure_cannot_report_capacity_success(self):
        with self.assertRaises(ExceptionGroup): self.run_probe(close_error=OSError('close failed'))

    def test_invalid_capacity_refuses(self):
        for available, size in [(-1, 4096), (1, 0), (1, -1)]:
            with self.subTest(available=available, size=size), self.assertRaises(RuntimeError):
                self.run_probe(stats=types.SimpleNamespace(f_bavail=available, f_frsize=size))
            self.assertEqual(self.calls[-1], ('close', 42))

    def test_abi_symbols_and_librt_fallback(self):
        class Function:
            pass
        api = types.SimpleNamespace(shm_open=Function(), shm_unlink=Function())
        with patch.object(subject.ctypes, 'CDLL', side_effect=[object(), api]) as load, \
             patch.object(subject.ctypes.util, 'find_library', return_value='resolved-rt'):
            opened, unlinked = subject._posix_shm_api()
        self.assertEqual(load.call_args_list[1].args, ('resolved-rt',))
        self.assertEqual(opened.argtypes, [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint])
        self.assertEqual(unlinked.argtypes, [ctypes.c_char_p])
        self.assertIs(opened.restype, ctypes.c_int)

    def test_missing_api_refuses_without_mount_guess(self):
        with patch.object(subject.ctypes, 'CDLL', return_value=object()), \
             patch.object(subject.ctypes.util, 'find_library', return_value=None), \
             self.assertRaises(RuntimeError):
            subject._posix_shm_api()



    def test_actual_windows_branch_never_queries_posix(self):
        subject.sys.platform = 'win32'
        try:
            with patch.object(subject, '_windows_shm', return_value=types.SimpleNamespace(available_memory=lambda: 1024), create=True), \
                 patch.object(subject, '_posix_available_memory', side_effect=AssertionError('POSIX must not run')):
                self.assertEqual(subject._check_allocation(7, 5, max_region_bytes=1024, min_free_bytes=0), 896)
                with self.assertRaisesRegex(RuntimeError, 'Windows'):
                    subject._check_allocation(7, 5, max_region_bytes=1024, min_free_bytes=129)
        finally: subject.sys.platform = 'linux'

    def test_actual_posix_budget_and_invalid_preallocation(self):
        with patch.object(subject, '_posix_available_memory', return_value=912) as probe:
            self.assertEqual(subject._check_allocation(7,5,max_region_bytes=896,min_free_bytes=16),896)
            with self.assertRaises(RuntimeError): subject._check_allocation(7,5,max_region_bytes=896,min_free_bytes=17)
            self.assertEqual(probe.call_count,2)
            with self.assertRaises(ValueError): subject._check_allocation(0,5,max_region_bytes=896,min_free_bytes=0)
            self.assertEqual(probe.call_count,2)


if __name__ == '__main__': unittest.main()

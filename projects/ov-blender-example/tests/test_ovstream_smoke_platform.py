import importlib
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import importlib.util
path=Path(__file__).resolve().parents[1]/'scripts/run_ovstream_shm_smoke.py'
spec=importlib.util.spec_from_file_location('portable_smoke_contracts',path)
smoke=importlib.util.module_from_spec(spec);spec.loader.exec_module(smoke)

READER=Path(__file__).resolve().parents[1]/'addon/ovrtx_blender_example/ovstream_shm.py'


class SmokeTests(unittest.TestCase):
    def test_real_reader_has_sibling_package_context(self):
        reader=smoke.load_reader(READER)
        helper=importlib.import_module(reader.__package__+'.win32_shm')
        self.assertEqual(Path(helper.__file__).resolve(),READER.parent/'win32_shm.py')
        self.assertIs(importlib.import_module(reader.__name__),reader)

    def test_windows_resources_use_kernel_probe(self):
        reader=smoke.load_reader(READER)
        helper=importlib.import_module(reader.__package__+'.win32_shm')
        with patch.object(smoke.sys,'platform','win32'), \
             patch.object(helper,'existing_resources',return_value={'mapping':False,'event':False,'pipe':False}) as probe:
            state=smoke.resource_state(reader,'blender-test')
        probe.assert_called_once_with('blender-test')
        smoke.assert_closed({'blender-test':state})

    def test_missing_or_live_kernel_resource_refuses_cleanup_pass(self):
        for state in ({},{'name':{}},{'name':{'mapping':False,'pipe':True}}):
            with self.subTest(state=state),self.assertRaises(AssertionError):smoke.assert_closed(state)

    def test_posix_location_is_explicit_and_wrong_mount_is_not_live(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(smoke.sys,'platform','linux'):
            with self.assertRaises(ValueError):smoke.resource_state(None,'owned')
            state=smoke.resource_state(None,'owned',shm_directory=directory,control_directory=directory)
            self.assertFalse(any(state.values()))
            root=Path(directory)
            for filename in ('ovstream-owned','sem.ovstream-owned-wake','ovstream-owned.sock'):
                (root/filename).write_bytes(b'')
            observed=smoke.resource_state(None,'owned',shm_directory=root,control_directory=root)
            self.assertTrue(all(observed.values()))
            with self.assertRaises(AssertionError):smoke.assert_closed({'owned':observed})

    def test_posix_live_producer_needs_region_and_socket_not_legacy_semaphore(self):
        # OVStream 0.5 on Linux creates no sem.ovstream-*-wake; 0.4 did.
        with tempfile.TemporaryDirectory() as directory,patch.object(smoke.sys,'platform','linux'):
            root=Path(directory)
            (root/'ovstream-owned').write_bytes(b'')
            (root/'ovstream-owned.sock').write_bytes(b'')
            observed=smoke.resource_state(None,'owned',shm_directory=root,control_directory=root)
            self.assertFalse(observed[str(root/'sem.ovstream-owned-wake')])
            self.assertTrue(smoke.producer_resources_present(observed))
            with self.assertRaises(AssertionError):smoke.assert_closed({'owned':observed})
            (root/'ovstream-owned.sock').unlink()
            self.assertFalse(smoke.producer_resources_present(
                smoke.resource_state(None,'owned',shm_directory=root,control_directory=root)))
            (root/'ovstream-owned').unlink()
            closed=smoke.resource_state(None,'owned',shm_directory=root,control_directory=root)
            self.assertFalse(smoke.producer_resources_present(closed))
            smoke.assert_closed({'owned':closed})

    def test_producer_presence_requires_every_windows_kernel_object(self):
        live={'Local\\ovstream-owned':True,'Local\\ovstream-owned-wake':True,'\\\\.\\pipe\\ovstream-owned':True}
        self.assertTrue(smoke.producer_resources_present(live))
        for key in live:
            with self.subTest(missing=key):
                self.assertFalse(smoke.producer_resources_present({**live,key:False}))
        self.assertFalse(smoke.producer_resources_present({}))

    def test_posix_probe_permission_failure_propagates(self):
        with patch.object(smoke.sys,'platform','linux'), \
             patch.object(Path,'lstat',side_effect=PermissionError('denied')), \
             self.assertRaises(PermissionError):
            smoke.resource_state(None,'owned',shm_directory='x',control_directory='y')


if __name__=='__main__':unittest.main()

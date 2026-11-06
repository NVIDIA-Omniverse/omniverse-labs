# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline prepared-asset installation and recovery checks; standard library only."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import setup_simready
import scene_assets


class PreparedInstallTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        (self.root/'config').mkdir()
        self.runtime = self.root/'runtime'
        self.output = self.root/'data/models/simready'
        self.payload = {'carton.usdc': b'geometry', 'textures/color.png': b'texture',
                        'index.json': b'{"assets":{}}', 'ATTRIBUTION.md': b'fixture attribution'}
        records = [{'path': n, 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
                   for n, b in self.payload.items()]
        manifest = json.dumps({'files': records}).encode()
        self.archive = self.root/'exhibits.zip'
        with zipfile.ZipFile(self.archive, 'w') as archive:
            for name, data in {**self.payload, 'release-manifest.json': manifest}.items():
                archive.writestr('simready/'+name, data)
        self.pin = {'filename': self.archive.name, 'bytes': self.archive.stat().st_size,
                    'sha256': scene_assets.sha256(self.archive), 'prefix': 'simready/',
                    'entry': 'index.json', 'manifest': 'release-manifest.json',
                    'manifest_sha256': hashlib.sha256(manifest).hexdigest(), 'url': None}
        (self.root/'config/prepared-exhibit-release.json').write_text(json.dumps({'bundle': self.pin}))
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(setup_simready, 'ROOT', self.root).start()
        mock.patch.object(scene_assets, 'download', side_effect=AssertionError('unexpected download')).start()
        mock.patch.object(setup_simready, 'download', side_effect=AssertionError('unexpected download')).start()

    def install(self, repair=False):
        setup_simready.setup(self.runtime, repair=repair, archive=self.archive)

    def test_fresh_install_repeat_and_repair_without_authoring_environment(self):
        self.install()
        for name, data in self.payload.items():
            self.assertEqual((self.output/name).read_bytes(), data)
        # Verified installed files work even when the original archive is absent.
        self.archive.unlink()
        setup_simready.setup(self.runtime)
        setup_simready.setup(self.runtime, repair=True)
        self.assertFalse((self.runtime/'asset-tools').exists())
        self.assertFalse((self.runtime/'simready-sources').exists())
        self.assertNotIn('pymeshlab', sys.modules)
        self.assertNotIn('pxr', sys.modules)

    def test_bundled_archive_is_found_without_explicit_argument(self):
        bundled = self.root/'data/scene-archives'/self.archive.name
        bundled.parent.mkdir(parents=True)
        self.archive.rename(bundled)
        setup_simready.setup(self.runtime)
        self.assertEqual((self.output/'carton.usdc').read_bytes(), b'geometry')

    def test_same_size_corruption_is_repaired_and_old_art_preserved(self):
        self.install()
        (self.output/'carton.usdc').write_bytes(b'corrupt!')
        self.install(repair=True)
        self.assertEqual((self.output/'carton.usdc').read_bytes(), b'geometry')
        old = list((self.output.parent/'.simready-backups').glob('*/carton.usdc'))
        self.assertEqual(len(old), 1)
        self.assertEqual(old[0].read_bytes(), b'corrupt!')

    def test_missing_files_are_repaired_on_normal_launch(self):
        self.install()
        (self.output/'textures/color.png').unlink()
        self.install()
        self.assertEqual((self.output/'textures/color.png').read_bytes(), b'texture')

    def test_legacy_cache_and_custom_files_remain_recoverable(self):
        self.output.mkdir(parents=True)
        (self.output/'index.json').write_text('incomplete legacy index')
        (self.output/'my-edits.txt').write_text('authored changes')
        self.install()
        old = list((self.output.parent/'.simready-backups').glob('*/my-edits.txt'))
        self.assertEqual(old[0].read_text(), 'authored changes')

    def test_bad_replacement_archive_never_moves_existing_art(self):
        self.install()
        (self.output/'carton.usdc').write_bytes(b'bad')
        self.archive.write_bytes(b'bad zip')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.install(repair=True)
        self.assertEqual((self.output/'carton.usdc').read_bytes(), b'bad')
        self.assertFalse((self.output.parent/'.simready-backups').exists())

    def test_failed_replacement_restores_previous_directory(self):
        self.install()
        (self.output/'carton.usdc').write_bytes(b'previous')
        rename = Path.rename
        def fail_staged(path, target):
            if path.parent.name.startswith('.simready-install-'):
                raise OSError('synthetic publication failure')
            return rename(path, target)
        with mock.patch.object(Path, 'rename', fail_staged):
            with self.assertRaisesRegex(OSError, 'synthetic'):
                self.install(repair=True)
        self.assertEqual((self.output/'carton.usdc').read_bytes(), b'previous')

    def test_failed_fresh_extraction_leaves_no_final_installation(self):
        with mock.patch.object(scene_assets, 'verify_install', side_effect=ValueError('bad payload')):
            with self.assertRaisesRegex(ValueError, 'bad payload'):
                self.install()
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)

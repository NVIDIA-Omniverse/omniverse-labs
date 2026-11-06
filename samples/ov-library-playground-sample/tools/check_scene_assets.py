# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for integrity, safe installation, credentials and interrupted downloads."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.request
import zipfile

from scene_assets import (ReleaseRedirect, download, ensure_asset, install_archive,
                          safe_member, sha256)


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status, self.headers = status, headers or {}


class SceneAssets(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = b'#usda 1.0\n'
        manifest = json.dumps({'files': [{'path': 'scene.usda', 'bytes': len(self.data),
                              'sha256': hashlib.sha256(self.data).hexdigest()}]}).encode()
        self.archive = self.root/'art.zip'
        self.pin = {'filename': 'art.zip', 'url': 'https://example.org/art.zip', 'prefix': 'art/', 'manifest': 'manifest.json',
                    'manifest_sha256': hashlib.sha256(manifest).hexdigest(), 'entry': 'scene.usda'}
        self.members = {'art/scene.usda': self.data, 'art/manifest.json': manifest}
        self.make_archive()
        self.release = {'repository': 'test/repo', 'tag': 'v1'}

    def make_archive(self, extra=None):
        with zipfile.ZipFile(self.archive, 'w') as zipped:
            for name, data in (self.members | (extra or {})).items():
                zipped.writestr(name, data)
        self.pin.update(bytes=self.archive.stat().st_size, sha256=sha256(self.archive))

    def test_verified_install_reuses_stamp_without_network_and_preserves_edits(self):
        target, cache = self.root/'installed', self.root/'cache'
        ensure_asset(self.pin, self.release, target, cache, self.archive)
        with patch('scene_assets.download', side_effect=AssertionError('unexpected network')):
            ensure_asset(self.pin, self.release, target, cache)
        (target/'scene.usda').write_bytes(b'x'*len(self.data))
        with self.assertRaisesRegex(ValueError, 'preserved'):
            ensure_asset(self.pin, self.release, target, cache)
        self.assertEqual((target/'scene.usda').read_bytes(), b'x'*len(self.data))

    def test_missing_and_corrupt_members_never_publish_partial_install(self):
        for kind in ('missing', 'corrupt'):
            with self.subTest(kind=kind):
                old = self.members.pop('art/scene.usda', None)
                if kind == 'corrupt':
                    self.members['art/scene.usda'] = b'x'*len(self.data)
                self.make_archive()
                target = self.root/kind
                with self.assertRaises((FileNotFoundError, ValueError)):
                    install_archive(self.archive, target, self.pin)
                self.assertFalse(target.exists())
                self.assertEqual(list(self.root.glob('.install-*')), [])
                self.members['art/scene.usda'] = old or self.data

    def test_archive_hash_checked_before_extraction(self):
        self.archive.write_bytes(b'corrupt')
        target = self.root/'installed'
        with self.assertRaisesRegex(ValueError, 'checksum'):
            install_archive(self.archive, target, self.pin)
        self.assertFalse(target.exists())

    def test_transient_windows_lock_retries_verified_install(self):
        target = self.root/'installed'
        rename = Path.rename
        attempts = []
        def locked_then_ready(source, destination):
            attempts.append(source)
            if len(attempts) < 3:
                error = PermissionError('Temporary sharing lock')
                error.winerror = 5
                raise error
            return rename(source, destination)
        with patch.object(Path, 'rename', locked_then_ready), patch('scene_assets.time.sleep') as sleep:
            install_archive(self.archive, target, self.pin)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual((target/'scene.usda').read_bytes(), self.data)

    def test_persistent_windows_lock_fails_without_partial_install(self):
        target = self.root/'installed'
        error = PermissionError('Persistent access failure')
        error.winerror = 5
        with patch.object(Path, 'rename', side_effect=error) as rename, patch('scene_assets.time.sleep'):
            with self.assertRaises(PermissionError):
                install_archive(self.archive, target, self.pin)
        self.assertEqual(rename.call_count, 9)
        self.assertFalse(target.exists())
        self.assertEqual(list(self.root.glob('.install-*')), [])

    def test_authored_revision_size_field_and_modified_file_protection(self):
        record={'path':'scene.usda','size':len(self.data),
                'sha256':hashlib.sha256(self.data).hexdigest()}
        manifest=json.dumps({'schema':1,'revision':'r48','files':[record]}).encode()
        self.members['art/manifest.json']=manifest
        self.pin['manifest_sha256']=hashlib.sha256(manifest).hexdigest()
        self.make_archive()
        target=self.root/'revision'
        install_archive(self.archive,target,self.pin)
        (target/'scene.usda').write_bytes(b'x'*len(self.data))
        with self.assertRaisesRegex(ValueError,'preserved'):
            ensure_asset(self.pin,self.release,target,self.root/'cache')

    def test_traversal_drive_stream_reserved_and_duplicate_names_are_rejected(self):
        for name in ('../escape', 'C:/escape', '/escape', 'x:stream', 'a/../../x', 'CON', 'trailing.', 'a//b'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_member(self.root, name)
        for extra in ({'art/../escape': b'x'}, {'art/SCENE.USDA': self.data}):
            self.make_archive(extra)
            with self.assertRaises(ValueError):
                install_archive(self.archive, self.root/'installed', self.pin)
            self.assertFalse((self.root/'installed').exists())

    def test_symlink_rejected(self):
        info = zipfile.ZipInfo('art/link')
        info.create_system = 3
        info.external_attr = 0o120777 << 16
        with zipfile.ZipFile(self.archive, 'a') as zipped:
            zipped.writestr(info, '../outside')
        self.pin.update(bytes=self.archive.stat().st_size, sha256=sha256(self.archive))
        with self.assertRaisesRegex(ValueError, 'symlink'):
            install_archive(self.archive, self.root/'installed', self.pin)

    def test_redirect_removes_token_and_rejects_http(self):
        original = urllib.request.Request('https://api.github.com/asset',
                                           headers={'Authorization': 'Bearer test-only'})
        redirect = ReleaseRedirect()
        next_request = redirect.redirect_request(original, None, 302, '', {}, 'https://release-assets.githubusercontent.com/file')
        self.assertIsNone(next_request.get_header('Authorization'))
        same = redirect.redirect_request(original, None, 302, '', {}, 'https://api.github.com/other')
        self.assertEqual(same.get_header('Authorization'), 'Bearer test-only')
        with self.assertRaises(ValueError):
            redirect.redirect_request(original, None, 302, '', {}, 'http://api.github.com/asset')

    def fake_download(self, resume, ignore_range=False, wrong_range=False, corrupt=False):
        cache = self.root/'cache'
        cache.mkdir(exist_ok=True)
        data = self.archive.read_bytes()
        partial = cache/'art.zip.partial'
        if resume:
            partial.write_bytes(data[:30])
        requests = []
        outer = self

        class Opener:
            def open(self, request, timeout):
                requests.append(request)
                outer.assertEqual(request.full_url, 'https://example.org/art.zip')
                outer.assertIsNone(request.get_header('Authorization'))
                if resume:
                    outer.assertEqual(request.get_header('Range'), 'bytes=30-')
                if resume and not ignore_range:
                    return Response(data[30:], 206, {'Content-Range': f'bytes {0 if wrong_range else 30}-{len(data)-1}/{len(data)}'})
                return Response(b'x'*len(data) if corrupt else data)

        return cache, requests, Opener()

    def test_resumes_partial_and_accepts_ignored_range_without_duplicate_bytes(self):
        for ignored in (False, True):
            with self.subTest(ignored=ignored):
                cache, requests, opener = self.fake_download(True, ignore_range=ignored)
                result = download(self.pin, self.release, cache, opener=opener)
                self.assertEqual(result.read_bytes(), self.archive.read_bytes())
                self.assertFalse((cache/'art.zip.partial').exists())
                self.assertEqual(len(requests), 1)
                result.unlink()

    def test_rejects_wrong_range_and_corrupt_response(self):
        cache, _, opener = self.fake_download(True, wrong_range=True)
        with self.assertRaisesRegex(ValueError, 'range'):
            download(self.pin, self.release, cache, opener=opener)
        (cache/'art.zip.partial').unlink()
        cache, _, opener = self.fake_download(False, corrupt=True)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            download(self.pin, self.release, cache, opener=opener)
        self.assertFalse((cache/'art.zip').exists())
        self.assertFalse((cache/'art.zip.partial').exists())

    def test_unpublished_asset_fails_without_network(self):
        self.pin['url']=None
        with patch('urllib.request.build_opener',side_effect=AssertionError('unexpected network')):
            with self.assertRaisesRegex(RuntimeError,'awaiting publication'):
                download(self.pin,self.release,self.root/'cache')

    def test_download_refuses_insecure_or_authenticated_urls(self):
        for url in ('http://example.org/art.zip','https://user:password@example.org/art.zip'):
            self.pin['url']=url
            with self.assertRaisesRegex(ValueError,'anonymous HTTPS'):
                download(self.pin,self.release,self.root/'cache')


if __name__ == '__main__':
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verified release downloads and atomic art installation (standard library only)."""
import hashlib
import http.client
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile


def progress(message):
    print('LAB_PROGRESS: '+message, flush=True)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class ReleaseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, url):
        if urllib.parse.urlsplit(url).scheme != 'https':
            raise ValueError('Refusing a release redirect without HTTPS.')
        redirected = super().redirect_request(request, fp, code, msg, headers, url)
        if redirected and urllib.parse.urlsplit(url).netloc != urllib.parse.urlsplit(request.full_url).netloc:
            redirected.remove_header('Authorization')
        return redirected


def verify_archive(path, pin):
    if path.stat().st_size != pin['bytes'] or sha256(path) != pin['sha256']:
        raise ValueError('Archive checksum does not match the pinned release: '+path.name)


def download(pin, release, cache, *, opener=None):
    """Resume interrupted transfers; promote to the cache only after verification."""
    cache.mkdir(parents=True, exist_ok=True)
    final = cache/pin['filename']
    if final.is_file():
        verify_archive(final, pin)
        return final
    endpoint = pin.get('url')
    if not endpoint:
        raise RuntimeError('Scene downloads are awaiting publication. Place '+pin['filename']+
                           ' in data/scene-archives/ or supply the matching offline ZIP. See docs/SCENE_ASSETS.md.')
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Scene downloads require an anonymous HTTPS URL.')
    opener = opener or urllib.request.build_opener(ReleaseRedirect())
    headers = {'User-Agent': 'ov-library-playground-sample', 'Accept': 'application/octet-stream'}
    partial = final.with_suffix(final.suffix+'.partial')
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > pin['bytes']:
            raise ValueError('Cached partial download is too large; remove '+str(partial)+' and Retry.')
        if offset == pin['bytes']:
            break
        request_headers = dict(headers)
        if offset:
            request_headers['Range'] = f'bytes={offset}-'
        try:
            with opener.open(urllib.request.Request(endpoint, headers=request_headers), timeout=60) as response:
                if response.status == 206:
                    expected = f'bytes {offset}-{pin["bytes"]-1}/{pin["bytes"]}'
                    if response.headers.get('Content-Range') != expected:
                        raise ValueError('Unexpected range in release download.')
                elif response.status == 200:
                    offset = 0  # A server may ignore Range; restart, never append a full response.
                else:
                    raise ValueError('Unexpected release download response.')
                previous = -1
                with partial.open('ab' if offset else 'wb') as stream:
                    while chunk := response.read(1024*1024):
                        offset += len(chunk)
                        if offset > pin['bytes']:
                            raise ValueError('Release download exceeded its pinned size.')
                        stream.write(chunk)
                        percent = offset*100//pin['bytes']
                        if percent//5 != previous:
                            previous = percent//5
                            progress(f'Downloading {pin["filename"]}: {percent}%')
            if offset == pin['bytes']:
                break
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f'Scene download failed (HTTP {error.code}). Check the published scene URL and Retry.') from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
            pass
        progress('Scene download interrupted; retrying from the cached partial file')
        time.sleep(attempt+1)
    if not partial.is_file() or partial.stat().st_size != pin['bytes']:
        raise RuntimeError('Scene download is incomplete. Retry will resume from the cached partial file.')
    progress('Verifying '+pin['filename'])
    try:
        verify_archive(partial, pin)
    except ValueError:
        # A complete but corrupt transfer cannot be resumed. Keep an incomplete one.
        if partial.stat().st_size == pin['bytes']:
            partial.unlink()
        raise
    partial.replace(final)
    return final


def safe_member(root, name):
    name = name.replace('\\', '/')
    parts = name.split('/')
    # Reject Windows drive paths, alternate streams and ambiguous/reserved names,
    # even when these functions are exercised on another operating system.
    if (not name or name.startswith('/') or PureWindowsPath(name).drive or
            any(part in ('', '.', '..') or ':' in part or part.endswith((' ', '.')) or
                PureWindowsPath(part).is_reserved() for part in parts)):
        raise ValueError('Unsafe asset member: '+name)
    target = (root/name).resolve()
    if not target.is_relative_to(root.resolve()) or target == root.resolve():
        raise ValueError('Unsafe asset member: '+name)
    return target


def verify_install(destination, pin, *, full=True):
    manifest = safe_member(destination, pin['manifest'])
    if sha256(manifest) != pin['manifest_sha256']:
        raise ValueError('Asset manifest differs from the release: '+str(manifest))
    records = json.loads(manifest.read_text())['files']
    names = set()
    for record in records:
        name = record.get('path', record.get('file'))
        path = safe_member(destination, name)
        if name.casefold() in names:
            raise ValueError('Duplicate asset manifest member: '+name)
        names.add(name.casefold())
        size = record.get('bytes', record.get('size'))
        if (not isinstance(size, int) or size < 0 or
                ('bytes' in record and 'size' in record and record['bytes'] != record['size'])):
            raise ValueError('Invalid asset manifest size: '+name)
        if path.stat().st_size != size or (full and sha256(path) != record['sha256']):
            raise ValueError('Asset was modified or is incomplete: '+str(path)+
                             '. Existing art is preserved; move it aside before reinstalling.')
    if pin['entry'].casefold() not in names:
        raise ValueError('The asset manifest does not include the entry scene.')
    return [manifest]+[safe_member(destination, r.get('path', r.get('file'))) for r in records]


def publish_install(staging, destination):
    # Windows scanners can briefly retain a handle on freshly extracted art.
    # Retry only sharing/access errors; never replace another installation.
    for attempt in range(9):
        try:
            staging.rename(destination)
            return
        except OSError as error:
            if (getattr(error, 'winerror', None) not in (5, 32, 33)
                    or destination.exists() or attempt == 8):
                raise
            if attempt == 0:
                progress('Finishing scene installation; waiting for a Windows file lock')
            time.sleep(0.5)


def install_archive(archive, destination, pin):
    verify_archive(archive, pin)
    if destination.exists():
        return verify_install(destination, pin)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Interrupted extraction leaves the final location untouched. The private
    # staging directory and every member are contained under destination.parent.
    with tempfile.TemporaryDirectory(prefix='.install-', dir=destination.parent) as temporary:
        staging = Path(temporary).resolve()
        assert staging.is_relative_to(destination.parent.resolve())
        seen = set()
        with zipfile.ZipFile(archive) as zipped:
            for entry in zipped.infolist():
                if entry.is_dir():
                    continue
                name = entry.filename.replace('\\', '/')
                if pin['prefix'] and name.startswith(pin['prefix']):
                    name = name[len(pin['prefix']):]
                target = safe_member(staging, name)
                if name.casefold() in seen or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Duplicate or symlink archive member: '+entry.filename)
                seen.add(name.casefold())
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(entry) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output, 1024*1024)
        verify_install(staging, pin)
        publish_install(staging, destination)
    return verify_install(destination, pin, full=False)


def file_state(paths):
    return {str(p.resolve()): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}


def ensure_asset(pin, release, destination, cache, archive=None, repair=False):
    stamp = cache/(pin['filename']+'.installed.json')
    if destination.is_dir():
        paths = verify_install(destination, pin, full=False)
        state = {'sha256': pin['sha256'], 'files': file_state(paths)}
        try:
            if not repair and json.loads(stamp.read_text()) == state:
                return
        except (OSError, ValueError):
            pass
        progress('Checking installed '+pin['entry'])
        verify_install(destination, pin)
    else:
        archive = archive or download(pin, release, cache)
        progress('Installing '+pin['entry'])
        paths = install_archive(archive, destination, pin)
        state = {'sha256': pin['sha256'], 'files': file_state(paths)}
    cache.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps(state, indent=2))

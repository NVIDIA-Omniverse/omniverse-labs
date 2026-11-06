# SPDX-License-Identifier: Apache-2.0
import importlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import public_sdk_environment as environment


@pytest.fixture
def clean_lock(monkeypatch):
    monkeypatch.setattr(environment.public_sdk_lock, 'load_lock', lambda path: {})
    monkeypatch.setattr(environment.public_sdk_lock, 'inspect_installed', lambda lock, verify_files: {'lock': 1})
    monkeypatch.setattr(environment, 'requirements', lambda: {})
    monkeypatch.setattr(environment.sys, 'version_info', (3, 13, 13))
    monkeypatch.setattr(environment.platform, 'machine', lambda: 'AMD64')
    monkeypatch.setattr(environment.sys, 'platform', 'win32')


def fail_import(failing, exc):
    def load(name):
        if name == failing:
            raise exc
    return load


def test_driver_linked_import_is_deferred_only_without_a_cuda_driver(clean_lock, monkeypatch):
    monkeypatch.setattr(importlib, 'import_module', fail_import('ovstream', FileNotFoundError('ovstream.dll')))
    monkeypatch.setattr(environment, 'cuda_driver_available', lambda: False)
    report = environment.inspect_environment()
    assert report['ok'] and report['errors'] == []
    assert report['deferred_imports'] == ['ovstream import: FileNotFoundError: ovstream.dll']


def test_driver_linked_import_fails_when_the_driver_is_present(clean_lock, monkeypatch):
    monkeypatch.setattr(importlib, 'import_module', fail_import('ovstream', OSError('missing dependency')))
    monkeypatch.setattr(environment, 'cuda_driver_available', lambda: True)
    report = environment.inspect_environment()
    assert not report['ok'] and report['deferred_imports'] == []


@pytest.mark.parametrize('name,exc', [('ovrtx', OSError('x')), ('ovstream', ImportError('x'))])
def test_other_import_failures_are_never_deferred(clean_lock, monkeypatch, name, exc):
    monkeypatch.setattr(importlib, 'import_module', fail_import(name, exc))
    monkeypatch.setattr(environment, 'cuda_driver_available', lambda: False)
    report = environment.inspect_environment()
    assert not report['ok'] and report['deferred_imports'] == []

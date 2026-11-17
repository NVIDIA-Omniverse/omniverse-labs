# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every main-bound PR must carry the next, matching application version."""
import importlib.util
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_checker(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "ci"))
    path = ROOT / "ci/check_ovui_version.py"
    assert path.is_file(), "Pre-merge version check is missing"
    spec = importlib.util.spec_from_file_location("check_ovui_version", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "source tree"
    (root / "python/ovd_ui").mkdir(parents=True)
    (root / "CHANGELOG.md").write_text("# Changelog\n\n## v0.2.0 (2026-09-15)\n\n- Previous release.\n")
    (root / "python/ovd_ui/__init__.py").write_text('__version__ = "0.2.0"\n')
    def git(*arguments):
        return subprocess.run(["git", "-C", str(root), "-c", "user.name=Version test",
            "-c", "user.email=version-test@example.invalid", *arguments],
            check=True, capture_output=True, text=True).stdout.strip()
    git("init")
    git("add", ".")
    git("commit", "-m", "Base release")
    return root, git("rev-parse", "HEAD")


def write_version(root, version, application=None):
    (root / "CHANGELOG.md").write_text(f"# Changelog\n\n## v{version} (2026-09-17)\n\n- PR changes.\n\n"
        "## v0.2.0 (2026-09-15)\n\n- Previous release.\n")
    (root / "python/ovd_ui/__init__.py").write_text(f'__version__ = "{application or version}"\n')


@pytest.mark.parametrize("version", ["0.2.1", "0.3.0"])
def test_accepts_next_patch_or_minor_matching_application(monkeypatch, repository, version):
    checker = load_checker(monkeypatch)
    root, base = repository
    write_version(root, version)
    assert checker.check_version(root, base) == version


@pytest.mark.parametrize("version", ["0.2.0", "0.2.2", "1.0.0"])
def test_rejects_unchanged_or_skipped_version_before_merge(monkeypatch, repository, version):
    checker = load_checker(monkeypatch)
    root, base = repository
    write_version(root, version)
    with pytest.raises(ValueError):
        checker.check_version(root, base)


def test_rejects_application_and_changelog_mismatch(monkeypatch, repository):
    checker = load_checker(monkeypatch)
    root, base = repository
    write_version(root, "0.2.1", "0.2.0")
    with pytest.raises(ValueError, match="match"):
        checker.check_version(root, base)


def test_rejects_non_commit_base_before_git_lookup(monkeypatch, repository):
    checker = load_checker(monkeypatch)
    with pytest.raises(ValueError, match="commit"):
        checker.check_version(repository[0], "--help")


def test_rejects_fenced_heading_that_disagrees_with_release_version(monkeypatch, repository):
    checker = load_checker(monkeypatch)
    root, base = repository
    write_version(root, "0.2.1", "0.2.0")
    changelog = root / "CHANGELOG.md"
    changelog.write_text("~~~md\n## v0.2.0 (2026-09-15)\n~~~\n" + changelog.read_text())
    with pytest.raises(ValueError, match="match"):
        checker.check_version(root, base)


def test_cli_checks_explicit_base_and_prints_release_version(repository, monkeypatch, capsys):
    checker = load_checker(monkeypatch)
    root, base = repository
    write_version(root, "0.2.1")
    monkeypatch.setattr(checker, "ROOT", root)
    assert checker.main(["--base", base]) == 0
    assert "0.2.1" in capsys.readouterr().out

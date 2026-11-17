# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Release policy and package verification, without publishing to GitHub."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("release_ovui", ROOT / "ci/release_ovui.py")
release = importlib.util.module_from_spec(SPEC)
if Path(SPEC.origin).exists():
    SPEC.loader.exec_module(release)

SHA = "a" * 40
CHANGELOG = "# Changelog\n\n## v0.1.25 (2026-09-13)\n\n- Portable previews.\n\n## v0.1.24 (2026-09-12)\n\n- Gizmos.\n"


def context(**overrides):
    values = dict(event_name="push", private=True, ref="refs/heads/main", branch="main",
                  protected=True, enabled=True, build_result="success", package_result="success")
    return values | overrides


@pytest.mark.parametrize("overrides", [
    {"event_name": "pull_request"}, {"event_name": "workflow_dispatch"},
    {"private": False}, {"ref": "refs/heads/feature"}, {"protected": False},
    {"enabled": False}, {"build_result": "failure"}, {"build_result": "skipped"},
    {"build_result": "cancelled"}, {"branch": ""},
])
def test_untrusted_or_unverified_context_cannot_publish(overrides):
    assert hasattr(release, "release_allowed"), "Release gate is not implemented"
    assert not release.release_allowed(**context(**overrides))


def test_only_verified_private_release_branch_can_publish():
    assert hasattr(release, "release_allowed"), "Release gate is not implemented"
    assert release.release_allowed(**context())


@pytest.mark.parametrize("result", [None, "", "failure", "cancelled", "skipped"])
def test_package_test_result_must_explicitly_pass(result):
    assert not release.release_allowed(**context(package_result=result))


def test_package_test_success_is_required_in_addition_to_build_success():
    assert release.release_allowed(**context(package_result="success"))
    assert not release.release_allowed(**context(build_result="failure", package_result="success"))


@pytest.mark.parametrize("branch", ["DF/OVUI", "feature/ovdnext-ovui-windows", "staging"])
def test_integration_branch_cannot_publish_even_if_configured_as_release_branch(branch):
    assert not release.release_allowed(**context(branch=branch, ref="refs/heads/" + branch))


@pytest.mark.parametrize("expected", ["0.1.25", "0.2.0"])
def test_release_notes_use_only_newest_section_and_require_new_version(expected):
    assert hasattr(release, "release_notes"), "Release version check is not implemented"
    changelog = CHANGELOG.replace("v0.1.25", "v" + expected)
    version, notes = release.release_notes(changelog, "# Old\n\n## v0.1.24 (2026-09-12)\n")
    assert version == expected
    assert notes == f"## v{expected} (2026-09-13)\n\n- Portable previews.\n"
    with pytest.raises(ValueError, match="increase"):
        release.release_notes(changelog, changelog)


@pytest.mark.parametrize("heading", ["## v00.1.25 (2026-09-13)", "## v0.1.25-dev (2026-09-13)",
                                    "## Upcoming", "## v0.1.25+sha (2026-09-13)"])
def test_invalid_newest_version_does_not_fall_back(heading):
    assert hasattr(release, "release_notes"), "Release version check is not implemented"
    with pytest.raises(ValueError):
        release.release_notes(CHANGELOG.replace("## v0.1.25 (2026-09-13)", heading), "")


def test_fenced_headings_do_not_split_release_notes():
    assert hasattr(release, "release_notes"), "Release version check is not implemented"
    text = CHANGELOG.replace("- Portable previews.", "- Portable previews.\n```md\n## example\n```\n  ## indented example")
    version, notes = release.release_notes(text, "")
    assert version == "0.1.25" and "## example" in notes and "## indented example" in notes
    assert "## v0.1.24" not in notes


@pytest.mark.parametrize("version", ["0.1.99", "0.2.1", "4.8.2"])
def test_release_version_must_be_next_patch_or_next_minor(version):
    previous = "# Changelog\n\n## v0.1.24 (2026-09-12)\n\n- Previous.\n"
    with pytest.raises(ValueError, match="next patch|next minor"):
        release.release_notes(CHANGELOG.replace("v0.1.25", "v" + version), previous)


def packages(tmp_path):
    for platform, suffix in (("windows", ".zip"), ("linux", ".zip")):
        directory = tmp_path / platform
        directory.mkdir()
        archive = directory / f"OvdNext-{platform}-x86_64-0.1.25{suffix}"
        payload = f"verified {platform} archive".encode()
        archive.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        archive.with_name(archive.name + ".sha256").write_text(f"{digest}  {archive.name}\n")
        (directory / "package-report.json").write_text(json.dumps(dict(
            version="0.1.25", platform=platform, archive=archive.name,
            archive_bytes=len(payload), archive_sha256=digest, expanded_bytes=1000,
            source_commit=SHA, source_dirty=False)))
        (directory / (archive.name + ".package-report.json")).write_bytes(
            (directory / "package-report.json").read_bytes())
    return tmp_path


def test_release_requires_both_matching_platforms_and_hashes(tmp_path):
    assert hasattr(release, "verified_assets"), "Package verification is not implemented"
    result = release.verified_assets(packages(tmp_path), "0.1.25", SHA)
    assert [path.name for path in result] == [
        "OvdNext-windows-x86_64-0.1.25.zip", "OvdNext-windows-x86_64-0.1.25.zip.sha256",
        "OvdNext-windows-x86_64-0.1.25.zip.package-report.json",
        "OvdNext-linux-x86_64-0.1.25.zip", "OvdNext-linux-x86_64-0.1.25.zip.sha256",
        "OvdNext-linux-x86_64-0.1.25.zip.package-report.json"]


def test_single_platform_verifier_reuses_the_release_hash_checks(tmp_path):
    root = packages(tmp_path)
    (root / "linux/package-report.json").unlink()
    assets = release.verified_assets(root, "0.1.25", SHA, platforms=("windows",))
    assert len(assets) == 3 and all(path.parent.name == "windows" for path in assets)
    assets[0].write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        release.verified_assets(root, "0.1.25", SHA, platforms=("windows",))


def test_release_rejects_missing_or_conflicting_published_report(tmp_path):
    root = packages(tmp_path)
    report = root / "linux/OvdNext-linux-x86_64-0.1.25.zip.package-report.json"
    report.write_text('{}')
    with pytest.raises(ValueError, match="report"):
        release.verified_assets(root, "0.1.25", SHA)
    report.unlink()
    with pytest.raises(ValueError):
        release.verified_assets(root, "0.1.25", SHA)


@pytest.mark.parametrize("digest", [None, "0" * 64])
def test_release_rejects_report_digest_that_does_not_match_archive(tmp_path, digest):
    root = packages(tmp_path)
    report = root / "linux/package-report.json"
    payload = json.loads(report.read_text())
    payload["archive_sha256"] = digest
    for path in (report, root / "linux/OvdNext-linux-x86_64-0.1.25.zip.package-report.json"):
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checksum"):
        release.verified_assets(root, "0.1.25", SHA)


@pytest.mark.parametrize("change", [
    {"version": "0.1.24"}, {"platform": "windows"}, {"source_commit": "b" * 40},
    {"source_dirty": True}, {"source_dirty": "false"}, {"archive": "../outside.tar.gz"},
    {"archive_bytes": 0},
])
def test_mismatched_or_unsafe_package_report_blocks_release(tmp_path, change):
    assert hasattr(release, "verified_assets"), "Package verification is not implemented"
    root = packages(tmp_path)
    report = root / "linux/package-report.json"
    report.write_text(json.dumps(json.loads(report.read_text()) | change))
    with pytest.raises(ValueError):
        release.verified_assets(root, "0.1.25", SHA)


def test_modified_archive_and_missing_platform_block_release(tmp_path):
    assert hasattr(release, "verified_assets"), "Package verification is not implemented"
    root = packages(tmp_path)
    archive = root / "windows/OvdNext-windows-x86_64-0.1.25.zip"
    archive.write_bytes(b"modified archive")
    with pytest.raises(ValueError):
        release.verified_assets(root, "0.1.25", SHA)
    (root / "linux/package-report.json").unlink()
    with pytest.raises(ValueError):
        release.verified_assets(root, "0.1.25", SHA)


def test_merge_lookup_skips_unmerged_or_other_branch():
    assert hasattr(release, "merged_pr"), "Merged-PR gate is not implemented"
    valid = dict(number=6, merged_at="2026-09-13T12:00:00Z", merge_commit_sha=SHA,
                 base={"ref": "main"})
    assert release.merged_pr([valid], "main", SHA) == 6
    for candidate in (valid | {"merged_at": None}, valid | {"merge_commit_sha": "b" * 40},
                      valid | {"base": {"ref": "DF/OVUI"}}):
        assert release.merged_pr([candidate], "main", SHA) is None


def test_ambiguous_merge_is_rejected():
    valid = dict(number=6, merged_at="2026-09-14T12:00:00Z", merge_commit_sha=SHA,
                 base={"ref": "main"})
    with pytest.raises(ValueError, match="exactly one"):
        release.merged_pr([valid, valid | {"number": 7}], "main", SHA)


@pytest.mark.parametrize("old_branch_setting", ["main", "DF/OVUI"])
def test_direct_main_push_skips_version_packages_and_publication(monkeypatch, tmp_path, capsys, old_branch_setting):
    event = tmp_path / "event.json"
    event.write_text(json.dumps(dict(repository=dict(private=True, default_branch=old_branch_setting,
        full_name="NVIDIA-dev/example"), before="b" * 40, after=SHA)))
    for key, value in dict(GITHUB_EVENT_PATH=str(event), GITHUB_EVENT_NAME="push",
        GITHUB_REF="refs/heads/main", GITHUB_REF_PROTECTED="true",
        GITHUB_REPOSITORY="NVIDIA-dev/example", GITHUB_SHA=SHA,
        OVDNEXT_RELEASE_PUBLISH_ENABLED="true", OVDNEXT_BUILD_RESULT="success",
        OVDNEXT_PACKAGE_RESULT="success").items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("OVDNEXT_RELEASE_BRANCH", old_branch_setting)
    monkeypatch.setattr("sys.argv", ["release_ovui.py", "--packages", str(tmp_path / "absent"), "--publish"])
    def github(*arguments, **kwargs):
        assert arguments == ("api", f"repos/NVIDIA-dev/example/commits/{SHA}/pulls")
        return []
    monkeypatch.setattr(release, "github", github)
    def unexpected(*arguments, **kwargs):
        raise AssertionError("Non-MR pushes must not read version/package inputs or publish")
    monkeypatch.setattr(release.subprocess, "run", unexpected)
    assert release.main() == 0
    assert "not a merged PR" in capsys.readouterr().out


@pytest.mark.parametrize("result", [None, "failure", "cancelled", "skipped"])
def test_cli_cannot_publish_without_successful_package_jobs(monkeypatch, tmp_path, capsys, result):
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"repository": {"private": True, "full_name": "NVIDIA-dev/example"},
                                "before": "b" * 40, "after": SHA}))
    for key, value in dict(GITHUB_EVENT_PATH=str(event), GITHUB_EVENT_NAME="push",
                          GITHUB_REF="refs/heads/main", GITHUB_REF_PROTECTED="true",
                          GITHUB_REPOSITORY="NVIDIA-dev/example", GITHUB_SHA=SHA,
                          OVDNEXT_RELEASE_PUBLISH_ENABLED="true", OVDNEXT_BUILD_RESULT="success").items():
        monkeypatch.setenv(key, value)
    if result is None:
        monkeypatch.delenv("OVDNEXT_PACKAGE_RESULT", raising=False)
    else:
        monkeypatch.setenv("OVDNEXT_PACKAGE_RESULT", result)
    monkeypatch.setattr("sys.argv", ["release_ovui.py", "--packages", str(tmp_path), "--publish"])
    def unexpected(*arguments, **kwargs):
        raise AssertionError("Unverified packages must not reach GitHub")
    monkeypatch.setattr(release, "github", unexpected)
    assert release.main() == 0
    assert "no publication performed" in capsys.readouterr().out


def publication_assets(tmp_path):
    result = []
    for name, content in (("windows.zip", b"windows archive"), ("linux.zip", b"linux archive")):
        path = tmp_path / name
        path.write_bytes(content)
        result.append(path)
    return result


class ReleaseBoundary:
    """Stateful GitHub API boundary with the documented release/asset fields."""

    def __init__(self, assets, *, tag=True, release_exists=False, draft=True):
        self.expected_assets = {path.name: path for path in assets}
        self.tag = ({"ref": "refs/tags/v0.1.25", "object": {"type": "commit", "sha": SHA}}
                    if tag else None)
        self.release = self.release_data(draft=draft) if release_exists else None
        self.writes = []
        self.fail_upload_once = False
        self.move_tag_on_publish = False

    @staticmethod
    def release_data(*, draft):
        return dict(id=123, tag_name="v0.1.25", target_commitish=SHA,
                    name="OvdNext 0.1.25", body="notes", draft=draft,
                    prerelease=False, immutable=False, assets=[])

    @staticmethod
    def asset_data(path):
        return dict(id=200, name=path.name, label="", state="uploaded",
                    content_type="application/octet-stream", size=path.stat().st_size,
                    digest="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                    download_count=0)

    def __call__(self, *arguments, **kwargs):
        if arguments == ("api", "repos/NVIDIA-dev/example/git/ref/tags/v0.1.25"):
            return copy.deepcopy(self.tag)
        if arguments == ("release", "view", "v0.1.25", "--repo", "NVIDIA-dev/example",
                         "--json", "databaseId"):
            return None if self.release is None else '{"databaseId": 123}'
        if arguments == ("api", "repos/NVIDIA-dev/example/releases/123"):
            return copy.deepcopy(self.release)
        if arguments[:3] == ("release", "create", "v0.1.25"):
            self.writes.append(arguments)
            assert self.release is None
            assert "--draft" in arguments and "--target" in arguments
            assert not any(str(path) in arguments for path in self.expected_assets.values())
            self.release = self.release_data(draft=True)
            return ""
        if arguments[:3] == ("release", "upload", "v0.1.25"):
            self.writes.append(arguments)
            assert "--clobber" not in arguments
            repo_index = arguments.index("--repo")
            for value in arguments[3:repo_index]:
                path = Path(value)
                self.release["assets"].append(self.asset_data(path))
                if self.fail_upload_once:
                    self.fail_upload_once = False
                    raise RuntimeError("upload connection lost")
            return ""
        if arguments == ("api", "--method", "PATCH", "repos/NVIDIA-dev/example/releases/123",
                         "-F", "draft=false", "-f", "make_latest=legacy"):
            self.writes.append(arguments)
            assert {item["name"] for item in self.release["assets"]} == set(self.expected_assets)
            self.release["draft"] = False
            if self.move_tag_on_publish:
                self.tag = {"ref": "refs/tags/v0.1.25", "object": {"type": "commit", "sha": "b" * 40}}
            elif self.tag is None:
                self.tag = {"ref": "refs/tags/v0.1.25", "object": {"type": "commit", "sha": SHA}}
            return copy.deepcopy(self.release)
        raise AssertionError(f"Unexpected GitHub operation: {arguments!r}")


def test_publisher_creates_draft_then_uploads_complete_asset_set_before_publish(monkeypatch, tmp_path):
    assert hasattr(release, "publish"), "Release publisher is not implemented"
    boundary = ReleaseBoundary(publication_assets(tmp_path), tag=False)
    monkeypatch.setattr(release, "github", boundary)

    release.publish("NVIDIA-dev/example", SHA, "0.1.25", "notes", list(boundary.expected_assets.values()))

    assert boundary.release["draft"] is False
    assert {item["name"] for item in boundary.release["assets"]} == {"windows.zip", "linux.zip"}
    assert [operation[:2] for operation in boundary.writes] == [
        ("release", "create"), ("release", "upload"), ("api", "--method")]


def test_publisher_resumes_partial_draft_without_reuploading_existing_asset(monkeypatch, tmp_path):
    assets = publication_assets(tmp_path)
    boundary = ReleaseBoundary(assets, release_exists=True)
    boundary.release["assets"] = [boundary.asset_data(assets[0])]
    monkeypatch.setattr(release, "github", boundary)

    release.publish("NVIDIA-dev/example", SHA, "0.1.25", "notes", assets)

    upload = next(operation for operation in boundary.writes if operation[:2] == ("release", "upload"))
    assert str(assets[0]) not in upload and str(assets[1]) in upload
    assert boundary.release["draft"] is False


def test_interrupted_upload_can_be_retried_without_overwriting_asset(monkeypatch, tmp_path):
    assets = publication_assets(tmp_path)
    boundary = ReleaseBoundary(assets, release_exists=True)
    boundary.fail_upload_once = True
    monkeypatch.setattr(release, "github", boundary)

    with pytest.raises(RuntimeError, match="connection lost"):
        release.publish("NVIDIA-dev/example", SHA, "0.1.25", "notes", assets)
    assert boundary.release["draft"] is True
    assert [item["name"] for item in boundary.release["assets"]] == ["windows.zip"]

    release.publish("NVIDIA-dev/example", SHA, "0.1.25", "notes", assets)
    assert boundary.release["draft"] is False
    assert [item["name"] for item in boundary.release["assets"]] == ["windows.zip", "linux.zip"]


def test_matching_published_release_is_verified_without_any_mutation(monkeypatch, tmp_path):
    assets = publication_assets(tmp_path)
    boundary = ReleaseBoundary(assets, release_exists=True, draft=False)
    boundary.release["assets"] = [boundary.asset_data(path) for path in assets]
    monkeypatch.setattr(release, "github", boundary)

    release.publish("NVIDIA-dev/example", SHA, "0.1.25", "notes", assets)

    assert boundary.writes == []


def test_incomplete_published_release_fails_closed_without_mutation(monkeypatch, tmp_path):
    assets = publication_assets(tmp_path)
    boundary = ReleaseBoundary(assets, release_exists=True, draft=False)
    boundary.release["assets"] = [boundary.asset_data(assets[0])]
    monkeypatch.setattr(release, "github", boundary)

    with pytest.raises(ValueError, match="complete asset set"):
        release.publish("NVIDIA-dev/example", SHA, "0.1.25", "notes", assets)
    assert boundary.writes == []


def test_malformed_existing_release_fails_closed(monkeypatch, tmp_path):
    assets = publication_assets(tmp_path)
    boundary = ReleaseBoundary(assets, release_exists=True)
    boundary.release = []
    monkeypatch.setattr(release, "github", boundary)

    with pytest.raises(ValueError, match="does not match"):
        release.publish("NVIDIA-dev/example", SHA, "0.1.25", "notes", assets)
    assert boundary.writes == []


@pytest.mark.parametrize("field,value", [
    ("tag_name", "v0.1.24"), ("target_commitish", "main"),
    ("name", "Wrong title"), ("body", "different notes"), ("prerelease", True),
])
def test_existing_release_metadata_mismatch_fails_closed(monkeypatch, tmp_path, field, value):
    assets = publication_assets(tmp_path)
    boundary = ReleaseBoundary(assets, release_exists=True)
    boundary.release[field] = value
    monkeypatch.setattr(release, "github", boundary)

    with pytest.raises(ValueError, match="does not match"):
        release.publish("NVIDIA-dev/example", SHA, "0.1.25", "notes", assets)
    assert boundary.writes == []


@pytest.mark.parametrize("field,value", [
    ("name", "unknown.zip"), ("size", 1), ("digest", "sha256:" + "0" * 64),
    ("digest", None), ("state", "starter"),
])
def test_existing_asset_mismatch_or_unknown_name_fails_closed(monkeypatch, tmp_path, field, value):
    assets = publication_assets(tmp_path)
    boundary = ReleaseBoundary(assets, release_exists=True)
    item = boundary.asset_data(assets[0])
    item[field] = value
    boundary.release["assets"] = [item]
    monkeypatch.setattr(release, "github", boundary)

    with pytest.raises(ValueError, match="asset"):
        release.publish("NVIDIA-dev/example", SHA, "0.1.25", "notes", assets)
    assert boundary.writes == []


def test_tag_must_resolve_to_exact_release_commit(monkeypatch, tmp_path):
    assets = publication_assets(tmp_path)
    boundary = ReleaseBoundary(assets, release_exists=True)
    boundary.tag["object"]["sha"] = "b" * 40
    monkeypatch.setattr(release, "github", boundary)

    with pytest.raises(ValueError, match="commit"):
        release.publish("NVIDIA-dev/example", SHA, "0.1.25", "notes", assets)
    assert boundary.writes == []


def test_publisher_reverifies_tag_after_publication(monkeypatch, tmp_path):
    assets = publication_assets(tmp_path)
    boundary = ReleaseBoundary(assets, release_exists=True)
    boundary.release["assets"] = [boundary.asset_data(path) for path in assets]
    boundary.move_tag_on_publish = True
    monkeypatch.setattr(release, "github", boundary)

    with pytest.raises(ValueError, match="commit"):
        release.publish("NVIDIA-dev/example", SHA, "0.1.25", "notes", assets)


def test_annotated_tag_is_dereferenced_and_verified(monkeypatch, tmp_path):
    assets = publication_assets(tmp_path)
    boundary = ReleaseBoundary(assets, release_exists=True)
    boundary.tag["object"] = {"type": "tag", "sha": "c" * 40}
    original = boundary.__call__
    def github(*arguments, **kwargs):
        if arguments == ("api", "repos/NVIDIA-dev/example/git/tags/" + "c" * 40):
            return {"object": {"type": "commit", "sha": SHA}}
        return original(*arguments, **kwargs)
    monkeypatch.setattr(release, "github", github)

    release.publish("NVIDIA-dev/example", SHA, "0.1.25", "notes", assets)
    assert boundary.release["draft"] is False


def test_malformed_annotated_tag_target_fails_before_dereference(monkeypatch, tmp_path):
    assets = publication_assets(tmp_path)
    boundary = ReleaseBoundary(assets, release_exists=True)
    boundary.tag["object"] = {"type": "tag", "sha": "../untrusted"}
    monkeypatch.setattr(release, "github", boundary)

    with pytest.raises(ValueError, match="cannot be verified"):
        release.publish("NVIDIA-dev/example", SHA, "0.1.25", "notes", assets)
    assert boundary.writes == []


def test_matching_tagless_draft_is_resumed_and_gets_verified_published_tag(monkeypatch, tmp_path):
    assets = publication_assets(tmp_path)
    boundary = ReleaseBoundary(assets, tag=False, release_exists=True)
    monkeypatch.setattr(release, "github", boundary)

    release.publish("NVIDIA-dev/example", SHA, "0.1.25", "notes", assets)

    assert boundary.release["draft"] is False
    assert boundary.tag["object"] == {"type": "commit", "sha": SHA}


def test_published_release_without_verifiable_tag_fails_closed(monkeypatch, tmp_path):
    assets = publication_assets(tmp_path)
    boundary = ReleaseBoundary(assets, tag=False, release_exists=True, draft=False)
    boundary.release["assets"] = [boundary.asset_data(path) for path in assets]
    monkeypatch.setattr(release, "github", boundary)

    with pytest.raises(ValueError, match="tag"):
        release.publish("NVIDIA-dev/example", SHA, "0.1.25", "notes", assets)
    assert boundary.writes == []


@pytest.mark.parametrize("error,missing", [("release not found\n", True), ("HTTP 403: Forbidden\n", False)])
def test_draft_lookup_distinguishes_absence_from_access_failure(monkeypatch, error, missing):
    monkeypatch.setattr(release.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stdout="", stderr=error))
    arguments = ("release", "view", "v0.1.25", "--repo", "NVIDIA-dev/example", "--json", "databaseId")
    if missing:
        assert release.github(*arguments, missing_ok=True) is None
    else:
        with pytest.raises(RuntimeError):
            release.github(*arguments, missing_ok=True)

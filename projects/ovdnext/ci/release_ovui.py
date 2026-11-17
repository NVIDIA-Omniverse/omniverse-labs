# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify internal OvUI release inputs; publishing is an explicit, gated action."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"


def release_allowed(*, event_name, private, ref, branch, protected, enabled, build_result, package_result=None):
    return (event_name == "push" and private is True and branch == "main"
            and ref == "refs/heads/" + branch and protected is True
            and enabled is True and build_result == "success" and package_result == "success")


def newest_section(changelog):
    lines = changelog.splitlines(keepends=True)
    start, fence = None, None
    for index, line in enumerate(lines):
        stripped = line.rstrip("\r\n")
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", stripped)
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
            continue
        if marker:
            fence = marker[1]
        elif re.match(r"^##(?:\s|$)", stripped):
            if start is not None:
                return "".join(lines[start:index]).rstrip() + "\n"
            start = index
    if start is None:
        raise ValueError("Changelog has no release section")
    return "".join(lines[start:]).rstrip() + "\n"


def release_notes(changelog, previous):
    notes = newest_section(changelog)
    match = re.fullmatch(rf"## v({VERSION}) \(\d{{4}}-\d{{2}}-\d{{2}}\)", notes.splitlines()[0])
    if not match:
        raise ValueError("Newest changelog heading must contain a canonical release version")
    version = match[1]
    if previous:
        older, _ = release_notes(previous, "")
        if tuple(map(int, version.split("."))) <= tuple(map(int, older.split("."))):
            raise ValueError("Release version must increase beyond the pre-merge version")
        major, minor, patch = map(int, older.split("."))
        if tuple(map(int, version.split("."))) not in {(major, minor, patch + 1), (major, minor + 1, 0)}:
            raise ValueError("Release version must be the next patch or next minor with patch zero")
    return version, notes


def verified_assets(root, version, source_commit, *, platforms=("windows", "linux")):
    root = Path(root).resolve()
    assets = []
    for platform in platforms:
        directory = root / platform
        report_path = directory / "package-report.json"
        try:
            if not report_path.resolve().is_relative_to(root):
                raise ValueError("Package report escapes the downloaded artifact directory")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            name = f"OvdNext-{platform}-x86_64-{version}.zip"
            expected = dict(version=version, platform=platform, archive=name, source_commit=source_commit)
            if any(report.get(key) != value for key, value in expected.items()) or report.get("source_dirty") is not False:
                raise ValueError(f"Unverified {platform} package provenance")
            archive = directory / name
            checksum = directory / (name + ".sha256")
            published_report = directory / (name + ".package-report.json")
            if any(not path.resolve().is_relative_to(root) for path in (archive, checksum, published_report)):
                raise ValueError("Package asset escapes the downloaded artifact directory")
            if published_report.read_bytes() != report_path.read_bytes():
                raise ValueError(f"Conflicting {platform} package report")
            if type(report.get("archive_bytes")) is not int or report["archive_bytes"] != archive.stat().st_size:
                raise ValueError(f"Incorrect {platform} package size")
            with archive.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if (report.get("archive_sha256") != digest or
                    checksum.read_text(encoding="utf-8").strip() != f"{digest}  {name}"):
                raise ValueError(f"Incorrect {platform} archive checksum")
            assets.extend((archive, checksum, published_report))
        except (OSError, json.JSONDecodeError, TypeError, AttributeError) as error:
            raise ValueError(f"Missing or malformed {platform} package") from error
    return assets


def merged_pr(pulls, branch, source_commit):
    matches = [pull["number"] for pull in pulls if pull.get("merged_at")
               and pull.get("merge_commit_sha") == source_commit
               and pull.get("base", {}).get("ref") == branch]
    if len(matches) > 1:
        raise ValueError("Release commit must identify exactly one merged PR on the release branch")
    return matches[0] if matches else None


def github(*arguments, missing_ok=False):
    result = subprocess.run(["gh", *arguments], text=True, capture_output=True, check=False)
    if result.returncode:
        if missing_ok and ("(HTTP 404)" in result.stderr or
                arguments[:2] == ("release", "view") and result.stderr.strip() == "release not found"):
            return None
        raise RuntimeError(f"GitHub command failed: {result.stderr.strip()}")
    return json.loads(result.stdout) if arguments[0] == "api" else result.stdout


def _tag_commit(repository, tag, reference):
    if reference is None:
        return None
    try:
        if reference["ref"] != "refs/tags/" + tag:
            raise ValueError
        target = reference["object"]
        if target["type"] == "tag":
            if not re.fullmatch(r"[0-9a-f]{40}", target["sha"]):
                raise ValueError
            target = github("api", f"repos/{repository}/git/tags/{target['sha']}")["object"]
        if target["type"] != "commit" or not re.fullmatch(r"[0-9a-f]{40}", target["sha"]):
            raise ValueError
        return target["sha"]
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"Tag {tag} cannot be verified to a commit") from None


def _expected_assets(assets):
    expected = {}
    for path in map(Path, assets):
        if path.name in expected:
            raise ValueError(f"Duplicate release asset name: {path.name}")
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        expected[path.name] = (path, path.stat().st_size, "sha256:" + digest)
    return expected


def _release(repository, tag, *, missing_ok=False):
    found = github("release", "view", tag, "--repo", repository, "--json", "databaseId",
                   missing_ok=missing_ok)
    if found is None:
        return None
    try:
        release_id = json.loads(found)["databaseId"]
        if type(release_id) is not int:
            raise ValueError
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        raise ValueError(f"Release {tag} has an invalid identity") from None
    return github("api", f"repos/{repository}/releases/{release_id}")


def _verify_release(found, *, tag, source_commit, title, notes, expected, complete):
    fields = dict(tag_name=tag, target_commitish=source_commit, name=title, body=notes,
                  prerelease=False)
    if (not isinstance(found, dict) or type(found.get("id")) is not int
            or type(found.get("draft")) is not bool
            or any(found.get(key) != value for key, value in fields.items())):
        raise ValueError(f"Existing release {tag} does not match the requested release")
    if type(found.get("prerelease")) is not bool:
        raise ValueError(f"Existing release {tag} does not match the requested release")
    remote = {}
    if not isinstance(found.get("assets"), list):
        raise ValueError(f"Existing release {tag} has malformed assets")
    for asset in found["assets"]:
        name = asset.get("name") if isinstance(asset, dict) else None
        if name not in expected or name in remote:
            raise ValueError(f"Existing release {tag} has an unknown or duplicate asset")
        _, size, digest = expected[name]
        if (asset.get("state") != "uploaded" or type(asset.get("size")) is not int
                or asset["size"] != size or asset.get("digest") != digest):
            raise ValueError(f"Existing release asset {name} does not match the local file")
        remote[name] = asset
    if complete and set(remote) != set(expected):
        raise ValueError(f"Existing release {tag} does not have the complete asset set")
    return set(remote)


def publish(repository, source_commit, version, notes, assets):
    tag = "v" + version
    title = "OvdNext " + version
    expected = _expected_assets(assets)
    reference = github("api", f"repos/{repository}/git/ref/tags/{tag}", missing_ok=True)
    release = _release(repository, tag, missing_ok=True)
    resolved = _tag_commit(repository, tag, reference)
    if resolved is not None and resolved != source_commit:
        raise ValueError(f"Tag {tag} does not resolve to the release commit")

    if release is None:
        with tempfile.TemporaryDirectory(prefix="ovdnext-release-") as temporary:
            notes_path = Path(temporary) / "notes.md"
            notes_path.write_text(notes, encoding="utf-8")
            github("release", "create", tag, "--repo", repository,
                   "--target", source_commit, "--title", title,
                   "--notes-file", str(notes_path), "--draft")
        reference = github("api", f"repos/{repository}/git/ref/tags/{tag}", missing_ok=True)
        resolved = _tag_commit(repository, tag, reference)
        if resolved is not None and resolved != source_commit:
            raise ValueError(f"Tag {tag} does not resolve to the release commit")
        release = _release(repository, tag)

    complete = isinstance(release, dict) and release.get("draft") is False
    present = _verify_release(release, tag=tag, source_commit=source_commit, title=title,
                              notes=notes, expected=expected, complete=complete)
    if not release["draft"]:
        if resolved is None:
            raise ValueError(f"Release {tag} has no verifiable tag")
        return
    missing = [str(expected[name][0]) for name in expected if name not in present]
    if missing:
        github("release", "upload", tag, *missing, "--repo", repository)

    reference = github("api", f"repos/{repository}/git/ref/tags/{tag}", missing_ok=True)
    resolved = _tag_commit(repository, tag, reference)
    if resolved is not None and resolved != source_commit:
        raise ValueError(f"Tag {tag} does not resolve to the release commit")
    release = _release(repository, tag)
    _verify_release(release, tag=tag, source_commit=source_commit, title=title,
                    notes=notes, expected=expected, complete=True)
    if not release["draft"]:
        if resolved is None:
            raise ValueError(f"Release {tag} has no verifiable tag")
        return
    github("api", "--method", "PATCH", f"repos/{repository}/releases/{release['id']}",
           "-F", "draft=false", "-f", "make_latest=legacy")
    reference = github("api", f"repos/{repository}/git/ref/tags/{tag}")
    if _tag_commit(repository, tag, reference) != source_commit:
        raise ValueError(f"Tag {tag} does not resolve to the release commit")
    published = _release(repository, tag)
    _verify_release(published, tag=tag, source_commit=source_commit, title=title,
                    notes=notes, expected=expected, complete=True)
    if published["draft"]:
        raise RuntimeError(f"Release {tag} was not published")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", required=True, type=Path)
    parser.add_argument("--publish", action="store_true", help="Publish only after every release guard passes")
    args = parser.parse_args()
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    branch = "main"
    allowed = release_allowed(event_name=os.environ["GITHUB_EVENT_NAME"], private=event["repository"]["private"],
        ref=os.environ["GITHUB_REF"], branch=branch, protected=os.environ.get("GITHUB_REF_PROTECTED") == "true",
        enabled=os.environ.get("OVDNEXT_RELEASE_PUBLISH_ENABLED") == "true",
        build_result=os.environ.get("OVDNEXT_BUILD_RESULT"),
        package_result=os.environ.get("OVDNEXT_PACKAGE_RESULT"))
    if not allowed:
        print("Release not enabled for this event; no publication performed.")
        return 0
    repository, source_commit = os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_SHA"]
    if (repository != event["repository"]["full_name"] or not re.fullmatch(r"[\w.-]+/[\w.-]+", repository)
            or not re.fullmatch(r"[0-9a-f]{40}", source_commit) or event.get("after") != source_commit
            or not re.fullmatch(r"[0-9a-f]{40}", event.get("before", ""))):
        raise ValueError("Release event identity is invalid")
    number = merged_pr(github("api", f"repos/{repository}/commits/{source_commit}/pulls"), branch, source_commit)
    if number is None:
        print("Main push is not a merged PR; no publication performed.")
        return 0
    previous = subprocess.run(["git", "show", event["before"] + ":CHANGELOG.md"], cwd=ROOT,
                              text=True, capture_output=True, check=True).stdout
    version, notes = release_notes((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), previous)
    assets = verified_assets(args.packages, version, source_commit)
    print(json.dumps(dict(version=version, commit=source_commit, merged_pr=number,
                          assets=[path.name for path in assets], publish=args.publish)))
    if args.publish:
        publish(repository, source_commit, version, notes, assets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Check or export the public omniverse-labs tree against the release manifest.

scripts/public-release-manifest.json defines which paths may ship in a public
release export (`include`), which stay in the internal repo (`internal_only`),
and which must never appear in the public tree (`exclude`: contributor
templates, caches, environments, generated outputs, local machine files).

Usage (from repo root):
    python scripts/check-public-release.py                     # check git-tracked files
    python scripts/check-public-release.py --strict --path DIR # check an exported tree / public clone
    python scripts/check-public-release.py --export DIR        # write a clean public tree to DIR

Exit code is 1 when any violation is found.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "scripts" / "public-release-manifest.json"


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    parts = pattern.split("/")
    out = []
    for index, part in enumerate(parts):
        last = index == len(parts) - 1
        if part == "**":
            out.append(".*" if last else "(?:[^/]+/)*")
            continue
        segment = ""
        for char in part:
            if char == "*":
                segment += "[^/]*"
            elif char == "?":
                segment += "[^/]"
            else:
                segment += re.escape(char)
        out.append(segment if last else segment + "/")
    return re.compile("".join(out) + r"\Z")


class Manifest:
    def __init__(self, data: dict) -> None:
        self.include = [glob_to_regex(p) for p in data["include"]]
        self.internal_only = [glob_to_regex(p) for p in data.get("internal_only", [])]
        self.exclude: list[tuple[str, str, re.Pattern[str]]] = []
        for category, patterns in data.get("exclude", {}).items():
            for pattern in patterns:
                self.exclude.append((category, pattern, glob_to_regex(pattern)))

    def is_internal_only(self, path: str) -> bool:
        return any(rx.match(path) for rx in self.internal_only)

    def is_included(self, path: str) -> bool:
        return any(rx.match(path) for rx in self.include)

    def excluded_by(self, path: str) -> tuple[str, str] | None:
        for category, pattern, rx in self.exclude:
            if rx.match(path):
                return category, pattern
        return None

    def is_exportable(self, path: str) -> bool:
        return (
            self.is_included(path)
            and not self.is_internal_only(path)
            and self.excluded_by(path) is None
        )


def load_manifest() -> Manifest:
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        return Manifest(json.load(f))


def git_tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return [p for p in result.stdout.decode("utf-8").split("\0") if p]


def tree_files(root: Path) -> list[str]:
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for name in filenames:
            files.append((Path(dirpath) / name).relative_to(root).as_posix())
    return sorted(files)


def find_violations(files: list[str], manifest: Manifest, strict: bool) -> list[tuple[str, str]]:
    violations = []
    for path in files:
        if manifest.is_internal_only(path):
            if strict:
                violations.append((path, "internal-only path in public tree"))
            continue
        if not manifest.is_included(path):
            violations.append((path, "not in the public release allowlist (`include`)"))
            continue
        hit = manifest.excluded_by(path)
        if hit:
            category, pattern = hit
            violations.append((path, f"excluded ({category}: `{pattern}`)"))
    return violations


def report(violations: list[tuple[str, str]], checked: int, source: str) -> int:
    if not violations:
        print(f"Public release hygiene: OK ({checked} files checked in {source}).")
        return 0
    annotate = os.environ.get("GITHUB_ACTIONS") == "true"
    print(
        f"Public release hygiene: {len(violations)} violation(s) in {source}:",
        file=sys.stderr,
    )
    for path, reason in violations:
        print(f"  {path}: {reason}", file=sys.stderr)
        if annotate:
            print(f"::error file={path}::Public release hygiene: {reason}")
    print(
        "\nRemove these files (git rm --cached for caches/outputs), move contributor "
        "templates under _unpublished/templates/, or update "
        "scripts/public-release-manifest.json if a new public path is intended.",
        file=sys.stderr,
    )
    return 1


def export_tree(dest: Path, manifest: Manifest) -> int:
    if dest.exists() and any(dest.iterdir()):
        print(f"Export destination must be empty or missing: {dest}", file=sys.stderr)
        return 1
    files = git_tracked_files()
    exported = 0
    skipped = []
    for path in files:
        src = REPO_ROOT / path
        if not manifest.is_exportable(path):
            skipped.append(path)
            continue
        if not src.is_file():
            skipped.append(path)
            continue
        target = dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        exported += 1
    print(f"Exported {exported} files to {dest} ({len(skipped)} tracked files left out).")
    return report(find_violations(tree_files(dest), manifest, strict=True), exported, str(dest))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--path", type=Path, help="check every file under DIR instead of git-tracked files")
    group.add_argument("--export", type=Path, metavar="DIR", help="write a clean public tree to DIR")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat internal-only paths (e.g. _unpublished/) as violations; use for exported trees and the public repo",
    )
    args = parser.parse_args()

    if not MANIFEST_PATH.is_file():
        print(f"Missing manifest: {MANIFEST_PATH}", file=sys.stderr)
        return 1
    manifest = load_manifest()

    if args.export:
        return export_tree(args.export.resolve(), manifest)
    if args.path:
        root = args.path.resolve()
        files = tree_files(root)
        return report(find_violations(files, manifest, args.strict), len(files), str(root))
    files = git_tracked_files()
    return report(find_violations(files, manifest, args.strict), len(files), "git-tracked files")


if __name__ == "__main__":
    raise SystemExit(main())

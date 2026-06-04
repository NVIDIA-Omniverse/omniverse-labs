#!/usr/bin/env python3
"""Sync articles/ and samples/ assets into docs/ for GitHub Pages.

GitHub Pages only publishes the docs/ folder (branch deploy) or a docs/
artifact (Actions deploy). Source content lives at the repo root in
articles/ and samples/ — this script copies required assets into docs/
and validates that each published slug has a grid entry and project page.

Usage (from repo root):
    python scripts/sync-docs-site.py
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
MANIFEST_PATH = DOCS / "data" / "content-manifest.json"
PROJECTS_PATH = DOCS / "data" / "projects.json"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def copy_file(src: Path, dest: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"Missing source file: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"  copied {src.relative_to(REPO_ROOT)} -> {dest.relative_to(REPO_ROOT)}")


def sync_article_assets(entry: dict) -> None:
    source_dir = REPO_ROOT / entry["source_dir"]
    slug = entry["slug"]
    images_dir = entry.get("images_dir")
    if images_dir:
        src_images = source_dir / images_dir
        dest_images = DOCS / "assets" / "images" / "articles" / slug
        if src_images.is_dir():
            dest_images.mkdir(parents=True, exist_ok=True)
            for image in src_images.iterdir():
                if image.is_file():
                    shutil.copy2(image, dest_images / image.name)
                    print(
                        f"  copied {image.relative_to(REPO_ROOT)} -> "
                        f"{(dest_images / image.name).relative_to(REPO_ROOT)}"
                    )
    card_src = source_dir / entry["card_image"]
    card_dest = DOCS / entry["card_image_dest"]
    copy_file(card_src, card_dest)


def sync_sample_assets(entry: dict) -> None:
    source_dir = REPO_ROOT / entry["source_dir"]
    card_src = source_dir / entry["card_image"]
    card_dest = DOCS / entry["card_image_dest"]
    copy_file(card_src, card_dest)


def validate_slug(slug: str, projects: dict) -> None:
    project_slugs = {p["slug"] for p in projects.get("projects", [])}
    if slug not in project_slugs:
        raise SystemExit(
            f"Slug '{slug}' is in content-manifest.json but missing from docs/data/projects.json"
        )
    page = DOCS / "projects" / slug / "index.html"
    if not page.is_file():
        raise SystemExit(
            f"Slug '{slug}' is in content-manifest.json but missing docs/projects/{slug}/index.html"
        )


def main() -> int:
    if not MANIFEST_PATH.is_file():
        print(f"Missing manifest: {MANIFEST_PATH}", file=sys.stderr)
        return 1

    manifest = load_json(MANIFEST_PATH)
    projects = load_json(PROJECTS_PATH)

    print("Syncing article assets...")
    for entry in manifest.get("articles", []):
        validate_slug(entry["slug"], projects)
        sync_article_assets(entry)

    print("Syncing sample assets...")
    for entry in manifest.get("samples", []):
        validate_slug(entry["slug"], projects)
        sync_sample_assets(entry)

    data_version = date.today().strftime("%Y%m%d")
    projects.setdefault("site", {})["dataVersion"] = data_version
    save_json(PROJECTS_PATH, projects)
    print(f"Updated site.dataVersion in projects.json -> {data_version}")

    print("Sync complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

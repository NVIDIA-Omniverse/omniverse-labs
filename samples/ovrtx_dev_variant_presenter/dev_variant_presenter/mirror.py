# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mirror an http(s)-hosted USD and its dependency closure into the local data tree.

pxr (usd-core) has no http resolver, and the whole pipeline (scan, composer,
classification) operates on local pxr stages — so a remote stage is DOWNLOADED once
into a local mirror that preserves the host's path layout, then opened like any other
local file. The composite-sublayer design never writes to the source, so read-only /
remote content needs nothing further.

Fixpoint: download the root, ask UsdUtils.ComputeAllDependencies for unresolved asset
paths — they come back anchored-ABSOLUTE under the mirror root (verified against
usd-core; relative refs are anchored to their referencing layer before resolution) and
include sublayers, references, payloads, and asset-valued attributes (textures, MDLs).
Map each back to its URL by path-prefix, download, repeat until the closure is closed.

A `<root>.mirror_complete` marker skips the (stage-composing, seconds-long) dependency
check on later opens; delete it to force a re-check. Files that 404 upstream are left
unresolved — the mirror then matches the source's own brokenness.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as _ET
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from dev_variant_presenter.usd_guard import USD_LOCK

_MAX_ROUNDS = 64   # closure depth bound (each round resolves one nesting level of new deps)
_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def is_url(path: str) -> bool:
    return path.lower().startswith(("http://", "https://"))


def _host_dirname(netloc: str) -> str:
    """Mirror-directory name for a URL authority — ':' (host:port) is illegal on Windows."""
    return netloc.replace(":", "_")


def _mirror_base(data_root: Path, host: str) -> Path:
    """The mirror is fully self-contained under data/_mirror/<host>/ — it NEVER reads or
    writes user-curated data folders, even when they hold copies of the same host tree.
    (Costs re-downloading shared content once; buys a hard tool-cache/user-data boundary.)"""
    del host
    return data_root / "_mirror"


_JUNCTION_ROOT = Path(r"C:\ovml")   # windows-only: short junctions defeat MAX_PATH (260)


def _shorten_windows(host_root: Path, local_root: Path) -> Path | None:
    """Collected-asset trees nest the host-named folder TWICE, blowing file paths past
    Windows MAX_PATH — Python (long-path aware) sees the files, but NATIVE code (pxr's
    Ar resolver, ovrtx's MDL entity resolver) cannot open them, so materials silently
    fail to compile. A short junction (no admin needed, unlike symlinks) keeps every
    anchored path under the limit. Returns the junction equivalent of host_root, or
    None when shortening is unneeded/impossible."""
    if os.name != "nt" or len(str(local_root.resolve())) < 180:   # ABSOLUTE length decides
        return None
    import hashlib
    import subprocess
    target = host_root.resolve()
    tag = hashlib.sha1(str(target).lower().encode()).hexdigest()[:8]
    link = _JUNCTION_ROOT / tag
    try:
        if not link.exists():
            _JUNCTION_ROOT.mkdir(exist_ok=True)
            subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                           check=True, capture_output=True)
        if link.resolve() == target:
            return link
    except Exception:  # noqa: BLE001
        pass
    return None


def _download(client, url: str, dest: Path) -> None:
    tmp = Path(str(dest) + ".part")
    with client.stream("GET", url) as r:
        r.raise_for_status()                      # before mkdir: a 404 must not litter empty dirs
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    os.replace(tmp, dest)


def parse_s3_listing(text: str) -> tuple[list[str], str | None]:
    """Keys + continuation token from an S3 ListObjectsV2 response. Raises on non-listing XML."""
    root = _ET.fromstring(text)
    if not root.tag.endswith("ListBucketResult"):
        raise ValueError("not an S3 listing")
    keys = [k.text for k in root.iter(f"{_S3_NS}Key") if k.text]
    tok = root.find(f"{_S3_NS}NextContinuationToken")
    truncated = (root.findtext(f"{_S3_NS}IsTruncated") or "").lower() == "true"
    return keys, (tok.text if truncated and tok is not None else None)


def _iter_s3_keys(client, scheme: str, netloc: str, prefix: str):
    """All keys under an S3 prefix via anonymous ListObjectsV2 (raises if listing is refused)."""
    token = None
    while True:
        url = f"{scheme}://{netloc}/?list-type=2&prefix={quote(prefix)}&max-keys=1000"
        if token:
            url += f"&continuation-token={quote(token)}"
        r = client.get(url)
        r.raise_for_status()
        keys, token = parse_s3_listing(r.text)
        yield from keys
        if not token:
            return


def _heal_collected_nesting(host_root: Path, rel: str, host: str) -> int:
    """Collected-asset stages nest a copy of the host tree inside the stage folder
    (`<stage_dir>/<host>/...`) and reference into it; the published nested copy can be
    INCOMPLETE upstream while its MDLs use package-relative imports (`.::Module`) that
    require siblings to exist there (Composer survives via MDL search-path fallback;
    ovrtx does not). Hard-link anything present in the top-level tree but missing at the
    same relative path in the nested copy — same bytes, no extra disk. Idempotent."""
    import shutil
    stage_dir = (host_root / rel).parent
    nested = stage_dir / host
    if not nested.is_dir():
        return 0
    prefix = Path(rel).parent                    # e.g. Samples/.../ConceptCar
    src_root = host_root / prefix
    healed = 0
    for src in src_root.rglob("*"):
        if not src.is_file() or nested in src.parents:   # don't recurse the nested copy itself
            continue
        if src.name.endswith((".part", ".mirror_complete")):
            continue
        dst = nested / prefix / src.relative_to(src_root)
        if dst.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        healed += 1
    return healed


def ensure_local(url: str, data_root: str | os.PathLike = "data", progress=None) -> str:
    """Return the local mirror path for `url`, downloading the USD + its dependency
    closure on first use. `progress(downloaded_count, current_filename)` per file."""
    import httpx
    from pxr import UsdUtils

    parts = urlsplit(url)
    host = _host_dirname(parts.netloc)
    rel = unquote(parts.path.lstrip("/"))
    host_root = _mirror_base(Path(data_root), host) / host
    local_root = host_root / rel
    marker = Path(str(local_root) + ".mirror_complete")
    if local_root.is_file() and marker.is_file():
        short = _shorten_windows(host_root, local_root)
        return str(short / rel) if short else str(local_root)

    done = 0
    failed: set[str] = set()
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        if not local_root.is_file():
            if progress:
                progress(done, local_root.name)
            _download(client, url, local_root)
            done += 1
        # BULK FIRST: S3-style hosts allow anonymous prefix listing — mirror EVERYTHING
        # under the root's folder in one cheap pass. This also covers what pxr cannot
        # see (imports INSIDE .mdl files — missing modules break material compile with
        # SdrShaderNode errors). The fixpoint below then only verifies + catches refs
        # outside the prefix; non-S3 hosts (listing refused) rely on it entirely.
        prefix = rel.rsplit("/", 1)[0] + "/" if "/" in rel else ""
        listed_ok = False
        try:
            for key in _iter_s3_keys(client, parts.scheme, parts.netloc, prefix):
                dest = host_root / unquote(key)
                if dest.is_file():
                    continue
                if progress:
                    progress(done, Path(key).name)
                try:
                    _download(client, f"{parts.scheme}://{parts.netloc}/{quote(key)}", dest)
                    done += 1
                except (httpx.HTTPStatusError, httpx.TransportError):
                    pass
            listed_ok = True
        except Exception:  # noqa: BLE001 — listing unsupported/refused: fixpoint result stands
            pass
        # A complete prefix listing SUPERSEDES dependency discovery for everything inside the
        # prefix — and ComputeAllDependencies composes the whole stage (minutes on big scenes).
        # Only crawl when listing wasn't available (non-S3 hosts).
        hr = host_root.resolve()
        for _ in range(0 if listed_ok else _MAX_ROUNDS):
            with USD_LOCK:   # ComputeAllDependencies composes stages — pxr is process-global
                _layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(str(local_root))
            todo = []
            for u in unresolved:
                p = Path(u)
                try:
                    relpath = p.resolve().relative_to(hr)
                except ValueError:
                    continue   # anchored outside the mirror (foreign absolute ref) — can't fetch
                dep_url = f"{parts.scheme}://{parts.netloc}/{relpath.as_posix()}"
                if dep_url not in failed:
                    todo.append((dep_url, p))
            if not todo:
                break
            for dep_url, dest in todo:
                if progress:
                    progress(done, dest.name)
                try:
                    _download(client, dep_url, dest)
                except (httpx.HTTPStatusError, httpx.TransportError):
                    failed.add(dep_url)   # missing/unreachable upstream too — skip from now on
                done += 1
    _heal_collected_nesting(host_root, rel, host)
    marker.write_text("ok", encoding="utf-8")
    short = _shorten_windows(host_root, local_root)
    return str(short / rel) if short else str(local_root)

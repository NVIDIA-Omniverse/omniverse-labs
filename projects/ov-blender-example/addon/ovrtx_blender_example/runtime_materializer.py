# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize a pinned runtime bundle into extension user storage."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import posixpath
import shutil
import stat
import subprocess
import tarfile
import time
from typing import Callable
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen
import zipfile

from . import bundled_runtime
from .runtime_manifest import (
    RUNTIME_MANIFEST_NAME,
    RuntimeComponent,
    RuntimeManifest,
    RuntimeManifestError,
    RuntimeTarget,
    parse_manifest_bytes,
)
from .runtime_store import INSTALL_RECORD_NAME, _filesystem_path, paths


_GITHUB_TOKEN_ENV = ("OVRTX_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")


class RuntimeMaterializerError(RuntimeError):
    """Raised when runtime materialization cannot complete safely."""


class RuntimeMaterializerCancelled(RuntimeMaterializerError):
    """Raised when runtime materialization is cancelled by the user."""


RuntimeProgressCallback = Callable[[str, int, int], None]
RuntimeCancelCallback = Callable[[], bool]


def _runtime_source_mismatch(observation: str) -> RuntimeMaterializerError:
    return RuntimeMaterializerError(
        f"{observation}\n"
        "The selected runtime does not match this add-on.\n"
        "Use the Release URL that supplied this add-on ZIP.\n"
        "Or select a folder containing all assets from that Release."
    )


def materialize_runtime(
    expected_manifest_sha256: str,
    storage_root: Path,
    *,
    source: str,
    progress: RuntimeProgressCallback | None = None,
    cancelled: RuntimeCancelCallback | None = None,
) -> Path:
    manifest, artifacts = _runtime_source_bundle(expected_manifest_sha256, source)
    source_label = source.strip()
    store_paths = paths(storage_root, manifest.platform)
    total_bytes = sum(component.size_bytes for component in manifest.components)
    completed_bytes = 0
    _check_cancelled(cancelled)
    _report_progress(progress, "Preparing runtime installation", 0, total_bytes)
    # Stage downloads and extraction as siblings under platform_root (Blender's own
    # convention) so every move is a same-filesystem rename, never cross-device. The
    # filesystem-path wrappers below keep deep extracted trees clear of Windows MAX_PATH.
    extract_root = store_paths.platform_root / ".extract"
    shutil.rmtree(_filesystem_path(store_paths.staging_root), ignore_errors=True)
    shutil.rmtree(_filesystem_path(store_paths.download_root), ignore_errors=True)
    shutil.rmtree(_filesystem_path(extract_root), ignore_errors=True)
    store_paths.staging_root.mkdir(parents=True, exist_ok=True)
    store_paths.download_root.mkdir(parents=True, exist_ok=True)
    extract_root.mkdir(parents=True, exist_ok=True)

    try:
        for index, component in enumerate(manifest.components):
            try:
                _check_cancelled(cancelled)
                archive = _download_component(
                    component,
                    store_paths.download_root,
                    index,
                    artifact=artifacts[component.id],
                    progress=progress,
                    completed_bytes=completed_bytes,
                    total_bytes=total_bytes,
                    cancelled=cancelled,
                )
                completed_bytes += component.size_bytes
                _check_cancelled(cancelled)
                component_root = extract_root / component.id
                component_root.mkdir()
                _report_progress(
                    progress,
                    f"Extracting {component.id}",
                    completed_bytes,
                    total_bytes,
                )
                _extract_archive(archive, component_root)
                _check_cancelled(cancelled)
                _report_progress(
                    progress,
                    f"Installing {component.id}",
                    completed_bytes,
                    total_bytes,
                )
                _copy_targets(component_root, store_paths.staging_root, component.targets)
                _check_cancelled(cancelled)
            except RuntimeMaterializerCancelled:
                raise
            except (RuntimeMaterializerError, OSError) as exc:
                raise RuntimeMaterializerError(
                    f"{component.id} from {source_label}: {exc}"
                ) from exc
        _report_progress(
            progress,
            "Finalizing runtime installation",
            total_bytes,
            total_bytes,
        )
        _relink_native_client_libraries(store_paths.staging_root, manifest)
        _chmod_executables(store_paths.staging_root, manifest)
        _write_install_record(store_paths.staging_root, manifest)
        _check_cancelled(cancelled)
        _promote_staging(store_paths.staging_root, store_paths.current_root)
    except Exception:
        shutil.rmtree(_filesystem_path(store_paths.staging_root), ignore_errors=True)
        raise
    finally:
        shutil.rmtree(_filesystem_path(store_paths.download_root), ignore_errors=True)
        shutil.rmtree(_filesystem_path(extract_root), ignore_errors=True)

    _report_progress(
        progress,
        "Runtime installation complete",
        total_bytes,
        total_bytes,
    )
    return store_paths.current_root


def _download_component(
    component: RuntimeComponent,
    download_root: Path,
    index: int,
    *,
    artifact: str | Path | None = None,
    progress: RuntimeProgressCallback | None = None,
    completed_bytes: int = 0,
    total_bytes: int = 0,
    cancelled: RuntimeCancelCallback | None = None,
) -> Path:
    if artifact is None:
        raise RuntimeMaterializerError(f"{component.id} artifact is missing")
    archive = download_root / f"component-{index}.archive"
    tmp = archive.with_suffix(archive.suffix + ".tmp")
    downloaded_bytes = 0
    _report_progress(
        progress,
        f"Downloading {component.id}",
        completed_bytes,
        total_bytes,
    )
    try:
        with _open_component_download(component, artifact) as response, tmp.open("wb") as output:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                _check_cancelled(cancelled)
                output.write(block)
                downloaded_bytes += len(block)
                _report_progress(
                    progress,
                    f"Downloading {component.id}",
                    completed_bytes + downloaded_bytes,
                    total_bytes,
                )
    except RuntimeMaterializerError:
        tmp.unlink(missing_ok=True)
        raise
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeMaterializerError(f"{component.id} download failed: {exc}") from exc
    _report_progress(
        progress,
        f"Verifying {component.id}",
        completed_bytes + downloaded_bytes,
        total_bytes,
    )
    if tmp.stat().st_size != component.size_bytes:
        actual_size = tmp.stat().st_size
        tmp.unlink(missing_ok=True)
        raise RuntimeMaterializerError(
            f"{component.id} size mismatch: expected {component.size_bytes}, got {actual_size}"
        )
    actual = _sha256(tmp)
    if actual.lower() != component.sha256.lower():
        tmp.unlink(missing_ok=True)
        raise RuntimeMaterializerError(f"{component.id} sha256 mismatch: expected {component.sha256}, got {actual}")
    tmp.replace(archive)
    return archive


def _report_progress(
    callback: RuntimeProgressCallback | None,
    message: str,
    completed: int,
    total: int,
) -> None:
    if callback is not None:
        callback(message, completed, total)


def _check_cancelled(callback: RuntimeCancelCallback | None) -> None:
    if callback is not None and callback():
        raise RuntimeMaterializerCancelled("Runtime installation cancelled")


def _open_component_download(component: RuntimeComponent, artifact: str | Path):
    if isinstance(artifact, Path):
        return artifact.open("rb")
    try:
        return urlopen(artifact)
    except HTTPError as exc:
        if exc.code not in (401, 403, 404) or _github_release_asset(artifact) is None:
            raise
        token = _github_token()
        if not token:
            raise RuntimeMaterializerError(
                f"{component.id} download failed: HTTP {exc.code}. "
                "If this is a private GitHub Release asset, authenticate with gh or set "
                + ", ".join(_GITHUB_TOKEN_ENV)
                + "."
            ) from exc
        return _open_github_release_asset(artifact, token)


def runtime_source_uses_network(source: str) -> bool:
    """Distinguish URL-like input without treating Windows drives as schemes."""

    value = source.strip()
    if PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute():
        return False
    return bool(urlparse(value).scheme)


def _runtime_source_bundle(
    expected_manifest_sha256: str,
    source: str,
) -> tuple[RuntimeManifest, dict[str, str | Path]]:
    value = source.strip()
    if not value:
        raise RuntimeMaterializerError("Install Runtime From is empty")
    if runtime_source_uses_network(value):
        owner, repo, tag = _github_release_page(value)
        root = (
            f"https://github.com/{quote(owner, safe='')}/{quote(repo, safe='')}/"
            f"releases/download/{quote(tag, safe='')}/"
        )
        manifest_bytes = _read_download(root + quote(RUNTIME_MANIFEST_NAME, safe=""), "runtime manifest")
        artifact_root: str | Path = root
    else:
        folder = _local_runtime_folder(value)
        manifest_path = folder / RUNTIME_MANIFEST_NAME
        try:
            manifest_bytes = manifest_path.read_bytes()
        except OSError as exc:
            raise _runtime_source_mismatch(
                f"The selected folder does not contain {RUNTIME_MANIFEST_NAME}."
            ) from exc
        artifact_root = folder
    actual = hashlib.sha256(manifest_bytes).hexdigest()
    if actual != expected_manifest_sha256:
        raise _runtime_source_mismatch(
            "The selected manifest SHA-256 differs from the add-on pin."
        )
    try:
        manifest = parse_manifest_bytes(manifest_bytes)
    except RuntimeManifestError as exc:
        raise RuntimeMaterializerError(str(exc)) from exc
    host_platform = bundled_runtime.current_platform_id()
    if manifest.platform != host_platform:
        raise _runtime_source_mismatch(
            f"The selected runtime is for {manifest.platform}, but this host is "
            f"{host_platform or 'unsupported'}."
        )
    if isinstance(artifact_root, str):
        artifacts = {
            component.id: (
                artifact_root + quote(component.filename, safe="")
            )
            for component in manifest.components
        }
    else:
        artifacts = {component.id: artifact_root / component.filename for component in manifest.components}
        missing = [component.filename for component in manifest.components if not artifacts[component.id].is_file()]
        if missing:
            raise RuntimeMaterializerError(
                f"local runtime folder {artifact_root} is missing: {', '.join(missing)}"
            )
    return manifest, artifacts


def _github_release_page(source: str) -> tuple[str, str, str]:
    parsed = urlparse(source)
    parts = parsed.path.split("/")
    release = parts[4:] if len(parts) >= 5 else []
    if len(release) == 2 and release[0] == "tag":
        tag = release[1]
    elif len(release) == 1 or (len(release) == 2 and not release[1]):
        tag = release[0]
    else:
        tag = ""
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != "github.com"
        or parsed.params
        or parsed.query
        or parsed.fragment
        or len(parts) not in (5, 6)
        or parts[0]
        or parts[3] != "releases"
        or not parts[1]
        or not parts[2]
        or not tag
    ):
        raise RuntimeMaterializerError(
            "Install Runtime From must identify one GitHub Release URL, for example "
            "https://github.com/{owner}/{repository}/releases/{release}/: "
            f"{source}"
        )
    return unquote(parts[1]), unquote(parts[2]), unquote(tag)


def _local_runtime_folder(source: str) -> Path:
    if not Path(source).is_absolute():
        raise RuntimeMaterializerError(
            f"Install Runtime From must be an absolute folder or GitHub Release URL: {source}"
        )
    folder = Path(source)
    if not folder.exists():
        raise RuntimeMaterializerError(f"local runtime folder does not exist: {source}")
    if not folder.is_dir():
        raise RuntimeMaterializerError(f"local runtime location is not a folder: {source}")
    return folder


def _read_download(url: str, label: str) -> bytes:
    try:
        with urlopen(url) as response:
            return response.read()
    except HTTPError as exc:
        if exc.code not in (401, 403, 404) or _github_release_asset(url) is None:
            raise RuntimeMaterializerError(f"{label} download failed: HTTP {exc.code}") from exc
        token = _github_token()
        if not token:
            raise RuntimeMaterializerError(
                f"{label} download failed: HTTP {exc.code}. If this is a private GitHub "
                "Release asset, authenticate with gh or set " + ", ".join(_GITHUB_TOKEN_ENV) + "."
            ) from exc
        with _open_github_release_asset(url, token) as response:
            return response.read()


def _open_github_release_asset(url: str, token: str):
    owner, repo, tag, name = _github_release_asset(url) or ("", "", "", "")
    if not owner:
        raise RuntimeMaterializerError(f"not a GitHub Release asset URL: {url}")
    release_url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{quote(tag, safe='')}"
    request = Request(release_url, headers=_github_headers(token, "application/vnd.github+json"))
    try:
        with urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeMaterializerError(f"GitHub Release lookup failed for {owner}/{repo}@{tag}: {exc}") from exc
    assets = payload.get("assets") if isinstance(payload, dict) else None
    if not isinstance(assets, list):
        raise RuntimeMaterializerError(f"GitHub Release lookup for {owner}/{repo}@{tag} did not return assets")
    matches = [
        asset
        for asset in assets
        if isinstance(asset, dict) and asset.get("name") == name and isinstance(asset.get("url"), str)
    ]
    if len(matches) != 1:
        if name == RUNTIME_MANIFEST_NAME and not matches:
            raise _runtime_source_mismatch(
                f"The selected GitHub Release does not contain {RUNTIME_MANIFEST_NAME}."
            )
        raise RuntimeMaterializerError(
            f"GitHub Release {owner}/{repo}@{tag} must contain exactly one asset named {name}, got {len(matches)}"
        )
    asset_request = Request(matches[0]["url"], headers=_github_headers(token, "application/octet-stream"))
    try:
        return urlopen(asset_request)
    except OSError as exc:
        raise RuntimeMaterializerError(f"authenticated GitHub asset download failed for {name}: {exc}") from exc


def _github_headers(token: str, accept: str) -> dict[str, str]:
    return {
        "Accept": accept,
        "Authorization": f"Bearer {token}",
        "User-Agent": "ov-blender-example-runtime-materializer",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _github_token() -> str:
    for name in _GITHUB_TOKEN_ENV:
        token = os.environ.get(name)
        if token and token.strip():
            return token.strip()
    gh = shutil.which("gh")
    if not gh:
        return _gh_hosts_file_token()
    try:
        completed = subprocess.run(
            [gh, "auth", "token"],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return _gh_hosts_file_token()
    if completed.returncode != 0:
        return _gh_hosts_file_token()
    stdout = completed.stdout.strip()
    return stdout.splitlines()[0] if stdout else _gh_hosts_file_token()


def _gh_hosts_file_token(host: str = "github.com") -> str:
    """Fall back to the token stored in gh's hosts.yml.

    The `gh auth token` subprocess only works when the gh binary is on the
    process PATH. When the extension runs inside Blender that PATH is often
    the launcher's, not the shell's, so a valid `gh auth login` session is
    only reachable through the config file. This reads the `oauth_token` for
    the requested host directly, without a YAML dependency.
    """
    hosts_path = _gh_hosts_path()
    if hosts_path is None:
        return ""
    try:
        lines = hosts_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    current_host: str | None = None
    fallback = ""
    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw[:1] not in (" ", "\t"):
            current_host = raw.strip().rstrip(":").strip()
            continue
        stripped = raw.strip()
        if stripped.startswith("oauth_token:"):
            token = stripped.split(":", 1)[1].strip().strip("\"'")
            if not token:
                continue
            if current_host == host:
                return token
            if not fallback:
                fallback = token
    return fallback


def _gh_hosts_path() -> Path | None:
    override = os.environ.get("GH_CONFIG_DIR")
    if override and override.strip():
        return Path(override.strip()).expanduser() / "hosts.yml"
    if os.name == "nt":
        appdata = os.environ.get("AppData") or os.environ.get("APPDATA")
        if appdata and appdata.strip():
            return Path(appdata.strip()) / "GitHub CLI" / "hosts.yml"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg.strip()).expanduser() if xdg and xdg.strip() else Path.home() / ".config"
    return base / "gh" / "hosts.yml"


def _github_release_asset(url: str) -> tuple[str, str, str, str] | None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc.lower() != "github.com":
        return None
    parts = parsed.path.lstrip("/").split("/")
    if len(parts) != 6 or parts[2] != "releases" or parts[3] != "download":
        return None
    owner, repo, _releases, _download, tag, name = (unquote(part) for part in parts)
    if not owner or not repo or not tag or not name:
        return None
    return owner, repo, tag, name


def _extract_archive(archive: Path, destination: Path) -> None:
    if zipfile.is_zipfile(archive):
        _extract_zip(archive, destination)
        return
    if tarfile.is_tarfile(archive):
        _extract_tar(archive, destination)
        return
    raise RuntimeMaterializerError(f"unsupported archive type: {archive}")


def _extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            mode = (member.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise RuntimeMaterializerError(f"zip symlink entries are not supported: {member.filename}")
            target = _safe_target(destination, member.filename)
            if member.is_dir():
                os.makedirs(_filesystem_path(target), exist_ok=True)
                continue
            os.makedirs(_filesystem_path(target.parent), exist_ok=True)
            with package.open(member) as source, open(_filesystem_path(target), "wb") as output:
                shutil.copyfileobj(source, output)
            execute_bits = (member.external_attr >> 16) & 0o111
            if execute_bits:
                target_path = _filesystem_path(target)
                os.chmod(target_path, os.stat(target_path).st_mode | execute_bits)


def _extract_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive) as package:
        members = package.getmembers()
        directory_modes: list[tuple[Path, int]] = []
        # Validation prepass: reject the whole archive before writing anything.
        for member in members:
            _safe_target(destination, member.name)
            if member.issym():
                _safe_link_target(destination, member.name, member.linkname)
            elif member.islnk():
                raise RuntimeMaterializerError(f"unsupported archive hardlink: {member.name}")
            elif not (member.isfile() or member.isdir()):
                raise RuntimeMaterializerError(f"unsupported archive member type: {member.name}")
        # Extract member-by-member with _filesystem_path: tarfile.extractall cannot take a
        # \\?\ destination because tar member names use "/", which \\?\ treats literally.
        for member in members:
            target = _safe_target(destination, member.name)
            if member.isdir():
                os.makedirs(_filesystem_path(target), exist_ok=True)
                directory_modes.append((target, member.mode))
            elif member.issym():
                os.makedirs(_filesystem_path(target.parent), exist_ok=True)
                if os.path.lexists(_filesystem_path(target)):
                    os.unlink(_filesystem_path(target))
                os.symlink(member.linkname, _filesystem_path(target))
            else:
                os.makedirs(_filesystem_path(target.parent), exist_ok=True)
                source = package.extractfile(member)
                if source is None:
                    raise RuntimeMaterializerError(f"could not extract archive member: {member.name}")
                with source, open(_filesystem_path(target), "wb") as output:
                    shutil.copyfileobj(source, output)
                os.chmod(_filesystem_path(target), member.mode)
        for target, mode in reversed(directory_modes):
            os.chmod(_filesystem_path(target), mode)


# Build-tree debris that upstream component zips sometimes ship inside a target
# (e.g. an out-of-source CMake/MSBuild dir). None of it is needed at runtime, and
# its path depth overflows Windows MAX_PATH once rooted in extension user storage.
_BUILD_DEBRIS_PATTERNS = ("cmake-*-build", "cmake-*-src", "CMakeFiles", "*.tlog")


def _ignore_build_debris(_directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for pattern in _BUILD_DEBRIS_PATTERNS:
        ignored.update(fnmatch.filter(names, pattern))
    return ignored


def _copy_targets(source_root: Path, staging_root: Path, targets: tuple[RuntimeTarget, ...]) -> None:
    for target in targets:
        source_name = target.source
        target_name = target.target
        source = _safe_target(source_root, source_name.rstrip("/"))
        destination = _safe_target(staging_root, target_name.rstrip("/"))
        if not source.exists():
            raise RuntimeMaterializerError(f"runtime component source missing: {source_name}")
        if source.is_dir():
            shutil.copytree(
                _filesystem_path(source), _filesystem_path(destination), dirs_exist_ok=True, symlinks=True, ignore=_ignore_build_debris
            )
        elif source.is_symlink():
            os.makedirs(_filesystem_path(destination.parent), exist_ok=True)
            destination.symlink_to(os.readlink(source))
        else:
            os.makedirs(_filesystem_path(destination.parent), exist_ok=True)
            shutil.copy2(_filesystem_path(source), _filesystem_path(destination))


# The bundle's native binaries -- the in-process OVRTX / OVPhysX client
# extensions, grpcio's cygrpc, and the ovrtx-bridge-server executable --
# all bake a defunct packman scratch dir into their DT_RPATH, so their gRPC / USD /
# ovrtx dependencies are unresolvable unless Blender happens to be launched with
# LD_LIBRARY_PATH already covering the bundle. That works for the release launcher
# but not a plain Blender start, and for the in-process extensions no env tweak can
# help at all: glibc captures LD_LIBRARY_PATH at process start, and grpc/upb are
# circularly dependent with eager relocations, so an incremental ctypes preload
# cannot resolve them either. Instead we repoint each binary's DT_RPATH at the
# directories that already ship the libraries it needs, relative via $ORIGIN so it
# survives the staging->current rename. glibc searches DT_RPATH transitively, so
# covering a binary also covers the dependency chain it pulls in.
#
# Directories the bundle ships runtime libraries in, checked for each binary's
# DT_NEEDED. The worker executable also links libovrtx-dynamic.so from its own
# bin/ dir, which is covered by the plain $ORIGIN entry.
_BUNDLED_LIBRARY_DIRS = (
    ("bin",),
    ("runtime", "ovphysx-bridge-server", "lib"),
    ("runtime", "ovrtx-bridge-server", "lib"),
    ("runtime", "ovrtx-bridge-server", "plugins"),
)

_DT_NEEDED = 1
_DT_RPATH = 15
_DT_RUNPATH = 29


def _relink_native_client_libraries(root: Path, manifest: RuntimeManifest) -> None:
    if manifest.platform != "linux-x64":
        return
    provided: dict[str, Path] = {}
    for parts in _BUNDLED_LIBRARY_DIRS:
        library_dir = root.joinpath(*parts)
        if not library_dir.is_dir():
            continue
        for entry in library_dir.iterdir():
            if entry.is_file() and _is_shared_object(entry.name):
                provided.setdefault(entry.name, library_dir)
    if not provided:
        return

    targets: list[Path] = []
    native_dir = root / "native"
    if native_dir.is_dir():
        targets.extend(native_dir.rglob("*.so"))
    worker = root / "bin" / "ovrtx-bridge-server"
    if worker.is_file():
        targets.append(worker)

    for candidate in sorted(set(targets)):
        if not candidate.is_file():
            continue
        info = _read_elf_dynamic(candidate)
        if info is None:
            continue
        # Preserve DT_NEEDED order so the search path is deterministic, keeping the
        # first directory that provides each dependency.
        needed_dirs: list[Path] = []
        for soname in info.needed:
            directory = provided.get(soname)
            if directory is not None and directory not in needed_dirs:
                needed_dirs.append(directory)
        if not needed_dirs:
            continue
        _set_elf_rpath(candidate, info, _origin_relative_rpath(candidate.parent, needed_dirs))


def _origin_relative_rpath(binary_dir: Path, library_dirs: list[Path]) -> str:
    entries: list[str] = []
    for library_dir in library_dirs:
        rel = os.path.relpath(library_dir, binary_dir)
        entries.append("$ORIGIN" if rel == "." else "$ORIGIN/" + PurePosixPath(rel.replace(os.sep, "/")).as_posix())
    return ":".join(entries)


def _is_shared_object(name: str) -> bool:
    return name.endswith(".so") or ".so." in name


class _ElfDynamicInfo:
    __slots__ = ("needed", "rpath_entry_offset", "rpath_tag", "rpath_str_offset", "rpath_old_length")

    def __init__(
        self,
        needed: list[str],
        rpath_entry_offset: int | None,
        rpath_tag: int | None,
        rpath_str_offset: int | None,
        rpath_old_length: int | None,
    ) -> None:
        self.needed = needed
        self.rpath_entry_offset = rpath_entry_offset
        self.rpath_tag = rpath_tag
        self.rpath_str_offset = rpath_str_offset
        self.rpath_old_length = rpath_old_length


def _read_elf_dynamic(path: Path) -> _ElfDynamicInfo | None:
    """Parse a 64-bit little-endian ELF's DT_NEEDED / DT_RPATH, best effort."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    # Only the x86_64 Linux extensions this bundle ships are relevant.
    if data[:4] != b"\x7fELF" or data[4:6] != b"\x02\x01":
        return None

    def u16(off: int) -> int:
        return int.from_bytes(data[off : off + 2], "little")

    def u32(off: int) -> int:
        return int.from_bytes(data[off : off + 4], "little")

    def u64(off: int) -> int:
        return int.from_bytes(data[off : off + 8], "little")

    try:
        sh_off = u64(0x28)
        sh_entsize = u16(0x3A)
        sh_num = u16(0x3C)
        sections = [sh_off + i * sh_entsize for i in range(sh_num)]
        dynamic = next((s for s in sections if u32(s + 4) == 6), None)  # SHT_DYNAMIC
        if dynamic is None:
            return None
        dyn_off, dyn_size, str_index, dyn_entsize = (
            u64(dynamic + 24),
            u64(dynamic + 32),
            u32(dynamic + 40),
            u64(dynamic + 56) or 16,
        )
        str_header = sections[str_index]
        str_off, str_size = u64(str_header + 24), u64(str_header + 32)

        def read_string(value: int) -> str:
            start = str_off + value
            end = data.index(b"\x00", start, str_off + str_size)
            return data[start:end].decode("ascii", "replace")

        needed: list[str] = []
        rpath_entry_offset = rpath_tag = rpath_value = None
        for pos in range(dyn_off, dyn_off + dyn_size, dyn_entsize):
            tag = u64(pos)
            if tag == 0:  # DT_NULL terminates the array
                break
            value = u64(pos + 8)
            if tag == _DT_NEEDED:
                needed.append(read_string(value))
            elif tag in (_DT_RPATH, _DT_RUNPATH) and rpath_entry_offset is None:
                rpath_entry_offset, rpath_tag, rpath_value = pos, tag, value
        if rpath_entry_offset is None:
            return _ElfDynamicInfo(needed, None, None, None, None)
        old_length = data.index(b"\x00", str_off + rpath_value) - (str_off + rpath_value)
        return _ElfDynamicInfo(needed, rpath_entry_offset, rpath_tag, str_off + rpath_value, old_length)
    except (IndexError, ValueError):
        return None


def _set_elf_rpath(path: Path, info: _ElfDynamicInfo, new_rpath: str) -> None:
    encoded = new_rpath.encode("ascii")
    # We can only rewrite the search path in place, so the replacement must fit in
    # the defunct scratch path it displaces. That path is hundreds of characters,
    # so a $ORIGIN-relative rpath comfortably fits; bail loudly if it ever won't.
    if info.rpath_str_offset is None or info.rpath_old_length is None:
        raise RuntimeMaterializerError(f"{path.name} has no DT_RPATH/DT_RUNPATH to repoint")
    if len(encoded) > info.rpath_old_length:
        raise RuntimeMaterializerError(
            f"{path.name} rpath replacement ({len(encoded)}) exceeds available space ({info.rpath_old_length})"
        )
    try:
        with open(_filesystem_path(path), "r+b") as handle:
            # Prefer RPATH: glibc searches it transitively, so the client module's
            # rpath also resolves the gRPC libraries it pulls in. DT_RUNPATH would
            # only cover the module's own direct NEEDED, missing that chain.
            if info.rpath_tag == _DT_RUNPATH and info.rpath_entry_offset is not None:
                handle.seek(info.rpath_entry_offset)
                handle.write(_DT_RPATH.to_bytes(8, "little"))
            handle.seek(info.rpath_str_offset)
            handle.write(encoded + b"\x00" * (info.rpath_old_length - len(encoded) + 1))
    except OSError as exc:
        raise RuntimeMaterializerError(f"could not repoint {path.name} rpath: {exc}") from exc


def _chmod_executables(root: Path, manifest: RuntimeManifest) -> None:
    for component in manifest.components:
        for executable in component.executables:
            target = _safe_target(root, executable)
            if not target.is_file():
                raise RuntimeMaterializerError(f"declared executable missing: {executable}")
            target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _write_install_record(root: Path, manifest: RuntimeManifest) -> None:
    payload = {
        "schema_version": 1,
        "platform": manifest.platform,
        "manifest_sha256": manifest.sha256,
        "components": [
            {
                "id": component.id,
                "filename": component.filename,
                "sha256": component.sha256,
                "size_bytes": component.size_bytes,
                "targets": [
                    {"source": target.source, "target": target.target, "mode": target.mode}
                    for target in component.targets
                ],
                "executables": list(component.executables),
            }
            for component in manifest.components
        ],
    }
    (root / INSTALL_RECORD_NAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _promote_staging(staging_root: Path, current_root: Path) -> None:
    previous_root = current_root.parent / ".current.previous"
    shutil.rmtree(_filesystem_path(previous_root), ignore_errors=True)
    if current_root.exists():
        _rename_with_retry(current_root, previous_root)
    try:
        _rename_with_retry(staging_root, current_root)
    except OSError:
        if previous_root.exists() and not current_root.exists():
            previous_root.rename(current_root)
        raise
    shutil.rmtree(_filesystem_path(previous_root), ignore_errors=True)


def _rename_with_retry(source: Path, target: Path, *, attempts: int = 12, delay: float = 0.5) -> None:
    """Rename ``source`` to ``target``, retrying transient sharing violations.

    On Windows the promotion renames a directory that was just populated with the
    runtime's executables and shared libraries. Security software frequently keeps
    a freshly written binary open long enough to make the directory rename fail
    with a sharing violation (WinError 5). Those handles clear on their own, so a
    bounded backoff turns a spurious failure into a short wait.
    """
    for attempt in range(attempts):
        try:
            source.rename(target)
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay * (attempt + 1))


def _safe_target(root: Path, relative_name: str) -> Path:
    name = _safe_archive_name(relative_name)
    candidate = (root / name).resolve()
    root_resolved = root.resolve()
    if os.path.commonpath([str(root_resolved), str(candidate)]) != str(root_resolved):
        raise RuntimeMaterializerError(f"archive member escapes root: {relative_name}")
    return candidate


def _safe_link_target(root: Path, member_name: str, link_name: str) -> None:
    if os.path.isabs(link_name):
        raise RuntimeMaterializerError(f"absolute archive link target rejected: {member_name} -> {link_name}")
    target = PurePosixPath(posixpath.normpath(str(PurePosixPath(_safe_archive_name(member_name)).parent / link_name)))
    if target.is_absolute() or ".." in target.parts:
        raise RuntimeMaterializerError(f"archive link target escapes root: {member_name} -> {link_name}")
    _safe_target(root, target.as_posix())


def _safe_archive_name(value: str) -> str:
    normalised = posixpath.normpath(value.replace("\\", "/"))
    while normalised.startswith("./"):
        normalised = normalised[2:]
    if normalised in {"", "."}:
        raise RuntimeMaterializerError(f"unsafe archive path: {value}")
    path = PurePosixPath(normalised)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeMaterializerError(f"unsafe archive path: {value}")
    return normalised


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "RuntimeMaterializerCancelled",
    "RuntimeMaterializerError",
    "materialize_runtime",
    "runtime_source_uses_network",
]

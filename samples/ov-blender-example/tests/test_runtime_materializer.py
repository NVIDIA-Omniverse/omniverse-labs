# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import struct
import sys
import tarfile
from types import SimpleNamespace
import zipfile
from urllib.error import HTTPError
from urllib.request import Request

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import bundled_runtime, runtime_materializer, runtime_store  # noqa: E402
from ovrtx_blender_example.runtime_manifest import (  # noqa: E402
    RUNTIME_MANIFEST_NAME,
    RUNTIME_MANIFEST_PIN_NAME,
    RuntimeManifestError,
    load_manifest_pin,
    parse_manifest_bytes as _parse_manifest_bytes,
)
from ovrtx_blender_example.runtime_materializer import (  # noqa: E402
    RuntimeMaterializerCancelled,
    RuntimeMaterializerError,
    materialize_runtime as _materialize_runtime,
    runtime_source_uses_network,
)
from ovrtx_blender_example.runtime_store import read_install_record, remove_runtime, status as _status, verify  # noqa: E402


_MANIFEST_BYTES: dict[str, bytes] = {}
_LAST_MANIFEST_SOURCE: Path | None = None


@pytest.fixture(autouse=True)
def _linux_runtime_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bundled_runtime, "current_platform_id", lambda: "linux-x64")


def parse_manifest_bytes(data: bytes):
    manifest = _parse_manifest_bytes(data)
    _MANIFEST_BYTES[manifest.sha256] = data
    if _LAST_MANIFEST_SOURCE is not None:
        _MANIFEST_SOURCES[manifest.sha256] = _LAST_MANIFEST_SOURCE
    return manifest


def materialize_runtime(manifest, storage_root: Path, *, source: str | None = None, **kwargs):
    data = _MANIFEST_BYTES[manifest.sha256]
    if source is None:
        source = str(_MANIFEST_SOURCES[manifest.sha256])
    if not runtime_source_uses_network(source):
        (Path(source) / RUNTIME_MANIFEST_NAME).write_bytes(data)
    return _materialize_runtime(manifest.sha256, storage_root, source=source, **kwargs)


def status(storage_root: Path, manifest):
    return _status(storage_root, manifest.platform, manifest.sha256)


_MANIFEST_SOURCES: dict[str, Path] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)


def _tar_xz_archive(source: Path, archive: Path) -> str:
    with tarfile.open(archive, "w:xz") as package:
        for path in sorted(source.rglob("*")):
            package.add(path, arcname=path.relative_to(source).as_posix(), recursive=False)
    return _sha256(archive)


def _bad_tar_xz_archive(archive: Path) -> str:
    with tarfile.open(archive, "w:xz") as package:
        info = tarfile.TarInfo("../escape.txt")
        data = b"bad\n"
        info.size = len(data)
        package.addfile(info, io.BytesIO(data))
    return _sha256(archive)


def _zip_archive(archive: Path, members: dict[str, bytes]) -> str:
    with zipfile.ZipFile(archive, "w") as package:
        for name, data in members.items():
            package.writestr(name, data)
    return _sha256(archive)


def _zip_archive_with_modes(archive: Path, members: dict[str, tuple[bytes, int]]) -> str:
    with zipfile.ZipFile(archive, "w") as package:
        for name, (data, mode) in members.items():
            info = zipfile.ZipInfo(name)
            info.external_attr = (stat.S_IFREG | mode) << 16
            package.writestr(info, data)
    return _sha256(archive)


def _manifest_bytes(archive: Path, archive_sha: str) -> bytes:
    global _LAST_MANIFEST_SOURCE
    payload = {
        "schema_version": 1,
        "kind": "ov-blender-example-runtime-bundle",
        "platform": "linux-x64",
        "components": [
            {
                "id": component_id,
                "filename": f"{component_id}-{archive.name}",
                "sha256": archive_sha,
                "size_bytes": archive.stat().st_size,
                "targets": (
                    [{"source": "runtime/", "target": "runtime/"}]
                    if component_id == "ovrtx-bridge-server"
                    else [{"source": "bin/", "target": "bin/"}]
                    if component_id == "ovphysx-bridge-server"
                    else [{"source": "native/", "target": "native/"}]
                ),
                "executables": (
                    ["bin/ovrtx-bridge-server"]
                    if component_id == "ovrtx-bridge-server"
                    else ["bin/ovphysx-bridge-server"]
                    if component_id == "ovphysx-bridge-server"
                    else []
                ),
            }
            for component_id in (
                "ovrtx-bridge-server",
                "ovphysx-bridge-server",
                "ovrtx-bridge-client",
                "ovphysx-bridge-client",
            )
        ],
    }
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    for component in payload["components"]:
        (archive.parent / component["filename"]).write_bytes(archive.read_bytes())
    digest = hashlib.sha256(data).hexdigest()
    (archive.parent / RUNTIME_MANIFEST_NAME).write_bytes(data)
    _MANIFEST_BYTES[digest] = data
    _MANIFEST_SOURCES[digest] = archive.parent
    _LAST_MANIFEST_SOURCE = archive.parent
    return data


class _Response:
    def __init__(self, data: bytes):
        self._data = io.BytesIO(data)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self, size: int = -1) -> bytes:
        return self._data.read(size)


def test_materialize_runtime_cancellation_leaves_no_partial_install(tmp_path: Path) -> None:
    archive = tmp_path / "runtime.zip"
    archive_sha = _zip_archive(archive, {"runtime/file": b"data"})
    manifest = parse_manifest_bytes(_manifest_bytes(archive, archive_sha))
    storage_root = tmp_path / "storage"
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 3

    with pytest.raises(RuntimeMaterializerCancelled):
        materialize_runtime(manifest, storage_root, cancelled=cancelled)

    platform_root = storage_root / "runtimes" / "linux-x64"
    assert not (platform_root / "current").exists()
    assert not (platform_root / ".current.staging").exists()
    assert not (platform_root / ".downloads").exists()


def test_materialize_runtime_installs_verified_bundle(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write(source / "bin" / "ovrtx-bridge-server", "#!/bin/sh\nexit 0\n")
    _write(source / "bin" / "ovphysx-bridge-server", "#!/bin/sh\nexit 0\n")
    _write(source / "native" / "ovsensors_worker_client.py", "VALUE = 1\n")
    _write(source / "runtime" / "ovrtx-bridge-server" / "package.marker", "render\n")
    _write(source / "runtime" / "ovrtx-bridge-server" / "helper", "#!/bin/sh\nexit 0\n", 0o755)
    _write(source / "runtime" / "ovphysx-bridge-server" / "package.marker", "grpc\n")
    _write(source / "runtime" / "ovphysx" / "package.marker", "physx\n")
    _write(source / "runtime" / "ovruntime" / "package.marker", "runtime\n")
    if os.name != "nt":
        os.symlink("package.marker", source / "runtime" / "ovrtx-bridge-server" / "package.link")
    archive = tmp_path / "runtime.tar.xz"
    archive_sha = _tar_xz_archive(source, archive)
    manifest = parse_manifest_bytes(_manifest_bytes(archive, archive_sha))
    storage_root = tmp_path / "storage"
    progress: list[tuple[str, int, int]] = []
    assert manifest.components[0].targets[0].source == "runtime/"
    assert manifest.components[0].targets[0].target == "runtime/"

    current_root = materialize_runtime(
        manifest,
        storage_root,
        progress=lambda message, completed, total: progress.append(
            (message, completed, total)
        ),
    )

    install_record = read_install_record(current_root)
    assert install_record is not None
    assert install_record["manifest_sha256"] == manifest.sha256
    assert "_".join(("source", "commit")) not in install_record
    if os.name != "nt":
        assert (current_root / "runtime" / "ovrtx-bridge-server" / "package.link").is_symlink()
        assert os.access(current_root / "runtime" / "ovrtx-bridge-server" / "helper", os.X_OK)
        assert os.access(current_root / "bin" / "ovrtx-bridge-server", os.X_OK)
        assert os.access(current_root / "bin" / "ovphysx-bridge-server", os.X_OK)
    defaults = bundled_runtime.defaults(root=current_root)
    assert defaults.worker_command
    assert defaults.native_client_path == str(current_root / "native")
    assert defaults.ovphysx_worker_command
    runtime_status = status(storage_root, manifest)
    assert runtime_status.state == "ready"
    assert runtime_status.current_root == current_root
    runtime_verify = verify(storage_root, manifest)
    assert runtime_verify.state == "ready"
    assert runtime_verify.message == "Runtime is verified."
    assert not (storage_root / "runtimes" / "linux-x64" / ".current.staging").exists()
    assert not (storage_root / "runtimes" / "linux-x64" / ".downloads").exists()
    total_bytes = sum(component.size_bytes for component in manifest.components)
    assert progress[0] == ("Preparing runtime installation", 0, total_bytes)
    assert progress[-1] == ("Runtime installation complete", total_bytes, total_bytes)
    assert any(
        message.startswith("Downloading ") and 0 < completed <= total_bytes
        for message, completed, _total in progress
    )
    assert [completed for _message, completed, _total in progress] == sorted(
        completed for _message, completed, _total in progress
    )


def test_materialize_runtime_preserves_zip_execute_bits(tmp_path: Path) -> None:
    if os.name == "nt":
        return
    archive = tmp_path / "runtime.zip"
    archive_sha = _zip_archive_with_modes(
        archive,
        {
            "bin/ovrtx-bridge-server": (b"worker\n", 0o755),
            "bin/ovphysx-bridge-server": (b"server\n", 0o755),
            "native/client.py": (b"client\n", 0o644),
            "runtime/ovrtx-bridge-server/helper": (b"helper\n", 0o755),
            "runtime/ovrtx-bridge-server/data.txt": (b"data\n", 0o644),
        },
    )
    manifest = parse_manifest_bytes(_manifest_bytes(archive, archive_sha))
    current_root = materialize_runtime(manifest, tmp_path / "storage")

    assert os.access(current_root / "runtime" / "ovrtx-bridge-server" / "helper", os.X_OK)
    assert not os.access(current_root / "runtime" / "ovrtx-bridge-server" / "data.txt", os.X_OK)


def test_materialize_runtime_downloads_github_release_asset_with_token(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    _write(source / "bin" / "ovrtx-bridge-server", "#!/bin/sh\nexit 0\n")
    _write(source / "bin" / "ovphysx-bridge-server", "#!/bin/sh\nexit 0\n")
    _write(source / "native" / "client.py", "VALUE = 1\n")
    _write(source / "runtime" / "package.marker", "runtime\n")
    archive = tmp_path / "runtime.tar.xz"
    archive_sha = _tar_xz_archive(source, archive)
    manifest_bytes = _manifest_bytes(archive, archive_sha)
    manifest = parse_manifest_bytes(manifest_bytes)
    release_api = "https://api.github.com/repos/example/runtime/releases/tags/linux-x64"
    asset_api = "https://api.github.com/repos/example/runtime/releases/assets/123"
    requests = []

    def fake_urlopen(request):
        requests.append(request)
        if isinstance(request, str):
            raise HTTPError(request, 404, "Not Found", None, None)
        assert isinstance(request, Request)
        assert request.get_header("Authorization") == "Bearer token-for-test"
        if request.full_url == release_api:
            assert request.get_header("Accept") == "application/vnd.github+json"
            return _Response(json.dumps({"assets": [
                {"name": RUNTIME_MANIFEST_NAME, "url": asset_api + "/manifest"},
                *[
                    {"name": component.filename, "url": asset_api}
                    for component in manifest.components
                ],
            ]}).encode("utf-8"))
        if request.full_url == asset_api + "/manifest":
            return _Response(manifest_bytes)
        if request.full_url == asset_api:
            assert request.get_header("Accept") == "application/octet-stream"
            return _Response(archive.read_bytes())
        raise AssertionError(f"unexpected URL: {request.full_url}")

    monkeypatch.setenv("OVRTX_GITHUB_TOKEN", "token-for-test")
    monkeypatch.setattr(runtime_materializer, "urlopen", fake_urlopen)

    current_root = _materialize_runtime(
        manifest.sha256,
        tmp_path / "storage",
        source="https://github.com/example/runtime/releases/linux-x64/",
    )

    assert (current_root / "runtime" / "package.marker").read_text(encoding="utf-8") == "runtime\n"
    assert [
        getattr(request, "full_url", request)
        for request in requests
        if not isinstance(request, str) or not request.startswith("file:")
    ][0].startswith("https://github.com/example/runtime/releases/download/linux-x64/")


def test_missing_github_manifest_has_panel_ready_guidance(monkeypatch) -> None:
    def fake_urlopen(request):
        if isinstance(request, str):
            raise HTTPError(request, 404, "Not Found", None, None)
        return _Response(json.dumps({"assets": []}).encode())

    monkeypatch.setenv("OVRTX_GITHUB_TOKEN", "token-for-test")
    monkeypatch.setattr(runtime_materializer, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeMaterializerError) as error:
        runtime_materializer._read_download(
            "https://github.com/example/runtime/releases/download/linux-x64/"
            + RUNTIME_MANIFEST_NAME,
            "runtime manifest",
        )

    assert len(str(error.value).splitlines()) >= 3


def test_runtime_source_mismatch_errors_share_one_explanation_and_action() -> None:
    first = str(runtime_materializer._runtime_source_mismatch("first observation")).splitlines()
    second = str(runtime_materializer._runtime_source_mismatch("second observation")).splitlines()

    assert first[0] != second[0]
    assert first[1:] == second[1:]


def test_runtime_source_relocates_unchanged_manifest_by_filename(tmp_path: Path, monkeypatch) -> None:
    archive = tmp_path / "runtime.zip"
    archive_sha = _zip_archive(archive, {"runtime/payload": b"runtime\n"})
    manifest_bytes = _manifest_bytes(archive, archive_sha)
    requested = []
    monkeypatch.setattr(
        runtime_materializer,
        "urlopen",
        lambda url: requested.append(url) or _Response(
            manifest_bytes if str(url).endswith(RUNTIME_MANIFEST_NAME) else archive.read_bytes()
        ),
    )

    _manifest, artifacts = runtime_materializer._runtime_source_bundle(
        hashlib.sha256(manifest_bytes).hexdigest(),
        "https://github.com/example/new-home/releases/tag/test-release",
    )

    assert all("example/new-home" in str(url) for url in requested)
    assert all("example/new-home" in str(url) for url in artifacts.values())


@pytest.mark.parametrize(
    "source",
    [
        "https://github.com/example/runtime/releases/linux-x64/",
        "https://github.com/example/runtime/releases/linux-x64",
        "https://github.com/example/runtime/releases/tag/linux-x64",
    ],
)
def test_runtime_source_accepts_human_release_url_forms(
    tmp_path: Path,
    source: str,
) -> None:
    assert runtime_materializer._github_release_page(source) == ("example", "runtime", "linux-x64")


@pytest.mark.parametrize(
    ("source", "uses_network"),
    [
        ("https://github.com/example/runtime/releases/linux-x64/", True),
        ("https://example.invalid/runtime", True),
        ("/opt/example/runtime", False),
        (r"C:\runtime", False),
        (r"\\server\share\runtime", False),
    ],
)
def test_runtime_source_classification_handles_urls_and_cross_platform_paths(
    source: str,
    uses_network: bool,
) -> None:
    assert runtime_source_uses_network(source) is uses_network


def test_private_github_asset_preserves_slash_in_release_tag() -> None:
    assert runtime_materializer._github_release_asset(
        "https://github.com/example/runtime/releases/download/runtime%2F2026/component.zip"
    ) == ("example", "runtime", "runtime/2026", "component.zip")


@pytest.mark.parametrize(
    "source",
    [
        "http://github.com/example/runtime/releases/tag/linux-x64",
        "https://example.com/example/runtime/releases/tag/linux-x64",
        "https://github.com/example/runtime/releases/download/linux-x64",
        "https://github.com/example/runtime/releases/tag/linux-x64/",
        "https://github.com/example/runtime/releases/tag/linux-x64?asset=1",
    ],
)
def test_runtime_source_rejects_non_release_urls(tmp_path: Path, source: str) -> None:
    with pytest.raises(RuntimeMaterializerError, match="GitHub Release URL"):
        runtime_materializer._github_release_page(source)


def test_materialize_runtime_installs_from_complete_local_artifact_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "archive-source"
    _write(source / "bin" / "ovrtx-bridge-server", "#!/bin/sh\n")
    _write(source / "bin" / "ovphysx-bridge-server", "#!/bin/sh\n")
    _write(source / "native" / "client.py", "client\n")
    _write(source / "runtime" / "payload", "runtime\n")
    archive = tmp_path / "runtime.zip"
    archive_sha = _zip_archive(
        archive,
        {
            path.relative_to(source).as_posix(): path.read_bytes()
            for path in source.rglob("*")
            if path.is_file()
        },
    )
    payload = json.loads(_manifest_bytes(archive, archive_sha).decode("utf-8"))
    artifact_root = tmp_path / "artifact-set"
    artifact_root.mkdir(parents=True)
    for component in payload["components"]:
        name = f"{component['id']}-linux-x64.zip"
        component["filename"] = name
        (artifact_root / name).write_bytes(archive.read_bytes())
    manifest = parse_manifest_bytes(json.dumps(payload).encode("utf-8"))
    monkeypatch.setattr(
        runtime_materializer,
        "urlopen",
        lambda _url: (_ for _ in ()).throw(AssertionError("local install used network")),
    )

    current = materialize_runtime(
        manifest,
        tmp_path / "storage",
        source=str(tmp_path / "artifact-set"),
    )

    assert (current / "runtime" / "payload").read_text(encoding="utf-8") == "runtime\n"


def test_local_runtime_source_requires_manifest_in_exact_folder(tmp_path: Path) -> None:
    archive = tmp_path / "runtime.zip"
    archive_sha = _zip_archive(archive, {"runtime/payload": b"runtime\n"})
    folder = tmp_path / "artifact-set"
    (folder / "artifacts").mkdir(parents=True)
    manifest_bytes = _manifest_bytes(archive, archive_sha)
    (folder / "artifacts" / RUNTIME_MANIFEST_NAME).write_bytes(manifest_bytes)

    with pytest.raises(RuntimeMaterializerError):
        _materialize_runtime(
            hashlib.sha256(manifest_bytes).hexdigest(), tmp_path / "storage", source=str(folder)
        )


def test_manifest_pin_mismatch_fails_before_manifest_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "artifact-set"
    folder.mkdir()
    (folder / RUNTIME_MANIFEST_NAME).write_bytes(b"not the pinned manifest")
    monkeypatch.setattr(
        runtime_materializer,
        "parse_manifest_bytes",
        lambda _data: (_ for _ in ()).throw(AssertionError("mismatched manifest was parsed")),
    )

    with pytest.raises(RuntimeMaterializerError):
        _materialize_runtime("0" * 64, tmp_path / "storage", source=str(folder))
    assert not (tmp_path / "storage").exists()


def test_wrong_platform_manifest_fails_before_component_lookup(tmp_path: Path) -> None:
    archive = tmp_path / "runtime.zip"
    archive_sha = _zip_archive(archive, {"runtime/payload": b"runtime\n"})
    payload = json.loads(_manifest_bytes(archive, archive_sha))
    payload["platform"] = "windows-x64"
    manifest_bytes = json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"
    folder = tmp_path / "artifact-set"
    folder.mkdir()
    (folder / RUNTIME_MANIFEST_NAME).write_bytes(manifest_bytes)

    with pytest.raises(RuntimeMaterializerError):
        _materialize_runtime(
            hashlib.sha256(manifest_bytes).hexdigest(), tmp_path / "storage", source=str(folder)
        )


def test_addon_manifest_pin_is_exact_lowercase_sha256(tmp_path: Path) -> None:
    (tmp_path / RUNTIME_MANIFEST_PIN_NAME).write_text("a" * 64 + "\n", encoding="ascii")
    assert load_manifest_pin(tmp_path) == "a" * 64
    (tmp_path / RUNTIME_MANIFEST_PIN_NAME).write_text("A" * 64, encoding="ascii")
    with pytest.raises(RuntimeManifestError, match="pin is invalid"):
        load_manifest_pin(tmp_path)


def test_local_runtime_failure_names_source_and_component_and_keeps_previous(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "runtime.zip"
    archive_sha = _zip_archive(archive, {"runtime/payload": b"runtime\n"})
    manifest = parse_manifest_bytes(_manifest_bytes(archive, archive_sha))
    folder = tmp_path / "artifact-set"
    folder.mkdir()
    for component in manifest.components:
        (folder / component.filename).write_bytes(
            b"truncated" if component.id == "ovrtx-bridge-server" else archive.read_bytes()
        )
    storage = tmp_path / "storage"
    current = storage / "runtimes" / "linux-x64" / "current"
    _write(current / "previous.txt", "previous\n")

    with pytest.raises(RuntimeMaterializerError) as failure:
        materialize_runtime(manifest, storage, source=str(folder))

    assert f"ovrtx-bridge-server from {folder}" in str(failure.value)
    assert "size mismatch" in str(failure.value)
    assert (current / "previous.txt").read_text(encoding="utf-8") == "previous\n"


def test_materialize_runtime_failure_keeps_previous_current(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    current = storage_root / "runtimes" / "linux-x64" / "current"
    _write(current / "previous.txt", "previous\n")
    archive = tmp_path / "bad.tar.xz"
    archive_sha = _bad_tar_xz_archive(archive)
    manifest = parse_manifest_bytes(_manifest_bytes(archive, archive_sha))

    try:
        materialize_runtime(manifest, storage_root)
    except RuntimeMaterializerError as exc:
        assert "unsafe archive path" in str(exc)
    else:
        raise AssertionError("expected unsafe archive failure")

    assert (current / "previous.txt").read_text(encoding="utf-8") == "previous\n"
    assert not (storage_root / "escape.txt").exists()
    assert not (storage_root / "runtimes" / "linux-x64" / ".current.staging").exists()
    assert not (storage_root / "runtimes" / "linux-x64" / ".downloads").exists()


def test_materialize_runtime_rejects_size_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write(source / "bin" / "ovrtx-bridge-server", "#!/bin/sh\n")
    archive = tmp_path / "runtime.tar.xz"
    archive_sha = _tar_xz_archive(source, archive)
    payload = json.loads(_manifest_bytes(archive, archive_sha).decode("utf-8"))
    payload["components"][0]["size_bytes"] = archive.stat().st_size + 1
    manifest = parse_manifest_bytes(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n")

    try:
        materialize_runtime(manifest, tmp_path / "storage")
    except RuntimeMaterializerError as exc:
        assert "size mismatch" in str(exc)
    else:
        raise AssertionError("expected size mismatch failure")

    assert not (tmp_path / "storage" / "runtimes" / "linux-x64" / "current").exists()


def test_download_uses_bounded_internal_name_for_long_filename(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write(source / "runtime" / "payload", "runtime\n")
    archive = tmp_path / "runtime.zip"
    archive_sha = _zip_archive(archive, {"runtime/payload": b"runtime\n"})
    payload = json.loads(_manifest_bytes(archive, archive_sha).decode("utf-8"))
    payload["components"][0]["filename"] = "digest." * 30 + "runtime.tar.xz"
    manifest = parse_manifest_bytes(
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    download_root = tmp_path / "downloads"
    download_root.mkdir()

    downloaded = runtime_materializer._download_component(
        manifest.components[0], download_root, 7, artifact=archive
    )

    assert downloaded.name == "component-7.archive"
    assert len(downloaded.name) < 32


def test_runtime_manifest_rejects_source_provenance_and_wrong_components(tmp_path: Path) -> None:
    archive = tmp_path / "runtime.zip"
    archive_sha = _zip_archive(archive, {"runtime/payload": b"runtime\n"})
    payload = json.loads(_manifest_bytes(archive, archive_sha).decode("utf-8"))
    provenance_key = "_".join(("source", "commit"))
    payload[provenance_key] = "private-source-commit"

    try:
        parse_manifest_bytes(json.dumps(payload).encode("utf-8"))
    except RuntimeManifestError as exc:
        assert "fields must be" in str(exc)
    else:
        raise AssertionError("expected source provenance rejection")

    del payload[provenance_key]
    extra = dict(payload["components"][0])
    extra["id"] = "ovrtx-runtime-shader-cache"
    extra["filename"] = "ovrtx-runtime-shader-cache.zip"
    payload["components"].append(extra)
    assert parse_manifest_bytes(json.dumps(payload).encode("utf-8")).platform == "linux-x64"

    payload["components"].pop()
    payload["components"].pop()
    try:
        parse_manifest_bytes(json.dumps(payload).encode("utf-8"))
    except RuntimeManifestError as exc:
        assert "missing required components" in str(exc)
    else:
        raise AssertionError("expected incomplete component rejection")


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has problems with long paths")
def test_materialize_runtime_extracts_deep_zip_member_and_cleans_work_roots(
    tmp_path: Path,
) -> None:
    deep_member = "runtime/" + "/".join(["deep-segment"] * 24) + "/payload.txt"
    archive = tmp_path / "runtime.zip"
    archive_sha = _zip_archive(
        archive,
        {
            "bin/ovrtx-bridge-server": b"worker",
            "bin/ovphysx-bridge-server": b"server",
            "native/client.py": b"client",
            deep_member: b"runtime\n",
        },
    )
    manifest = parse_manifest_bytes(_manifest_bytes(archive, archive_sha))
    storage_root = tmp_path / "Blender Foundation" / "5.1" / "extensions" / "user_default"

    current = materialize_runtime(manifest, storage_root)

    assert (current / deep_member).read_text(encoding="utf-8") == "runtime\n"
    platform_root = storage_root / "runtimes" / "linux-x64"
    assert not (platform_root / ".downloads").exists()
    assert not (platform_root / ".extract").exists()
    assert not (platform_root / ".current.staging").exists()


def test_materialize_runtime_rejects_zip_traversal_and_symlink(tmp_path: Path) -> None:
    for name, configure in (
        ("traversal", lambda package: package.writestr("../escape.txt", b"bad")),
        (
            "symlink",
            lambda package: package.writestr(
                _zip_symlink("runtime/link"), b"../../escape.txt"
            ),
        ),
    ):
        archive = tmp_path / f"{name}.zip"
        with zipfile.ZipFile(archive, "w") as package:
            configure(package)
        manifest = parse_manifest_bytes(_manifest_bytes(archive, _sha256(archive)))
        storage_root = tmp_path / name

        try:
            materialize_runtime(manifest, storage_root)
        except RuntimeMaterializerError:
            pass
        else:
            raise AssertionError(f"expected {name} archive rejection")

        platform_root = storage_root / "runtimes" / "linux-x64"
        assert not (platform_root / ".downloads").exists()
        assert not (platform_root / ".extract").exists()
        assert not (platform_root / ".current.staging").exists()
        assert not (storage_root / "escape.txt").exists()


def _zip_symlink(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = (0o120777 << 16)
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def test_runtime_status_and_remove_runtime(tmp_path: Path) -> None:
    archive = tmp_path / "empty.tar.xz"
    source = tmp_path / "source"
    _write(source / "bin" / "ovrtx-bridge-server", "#!/bin/sh\n")
    archive_sha = _tar_xz_archive(source, archive)
    manifest = parse_manifest_bytes(_manifest_bytes(archive, archive_sha))
    storage_root = tmp_path / "storage"
    current = storage_root / "runtimes" / "linux-x64" / "current"
    _write(current / "installed-runtime.json", json.dumps({"manifest_sha256": "old"}) + "\n")
    _write(storage_root / "runtimes" / "linux-x64" / ".current.staging" / "leftover", "x\n")
    _write(storage_root / "runtimes" / "linux-x64" / ".downloads" / "leftover", "x\n")

    runtime_status = status(storage_root, manifest)

    assert runtime_status.state == "mismatch"
    assert runtime_status.installed_manifest_sha256 == "old"

    remove_runtime(storage_root, "linux-x64")

    assert not current.exists()
    assert not (storage_root / "runtimes" / "linux-x64" / ".current.staging").exists()
    assert not (storage_root / "runtimes" / "linux-x64" / ".downloads").exists()


def _reembedded_manifest_bytes(original: bytes) -> bytes:
    payload = json.loads(original.decode("utf-8"))
    return json.dumps(payload, indent=4, sort_keys=True).encode("utf-8") + b"\n"


def _write_runtime_layout(current: Path) -> None:
    _write(current / "runtime" / "marker", "runtime\n")
    _write(current / "bin" / "ovphysx-bridge-server", "#!/bin/sh\n", 0o755)
    _write(current / "native" / "marker", "native\n")
    _write(current / "bin" / "ovrtx-bridge-server", "#!/bin/sh\n", 0o755)


def _install_record(manifest: object, manifest_sha256: str) -> str:
    return json.dumps(
        {
            "manifest_sha256": manifest_sha256,
            "components": [
                {
                    "id": component.id,
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
    ) + "\n"


def test_runtime_status_rejects_different_manifest_bytes_with_same_components(tmp_path: Path) -> None:
    archive = tmp_path / "empty.tar.xz"
    source = tmp_path / "source"
    _write(source / "runtime" / "marker", "runtime\n")
    archive_sha = _tar_xz_archive(source, archive)
    old_manifest = parse_manifest_bytes(_manifest_bytes(archive, archive_sha))
    new_manifest = parse_manifest_bytes(_reembedded_manifest_bytes(_manifest_bytes(archive, archive_sha)))
    assert old_manifest.sha256 != new_manifest.sha256

    storage_root = tmp_path / "storage"
    current = storage_root / "runtimes" / "linux-x64" / "current"
    _write_runtime_layout(current)
    _write(current / "installed-runtime.json", _install_record(old_manifest, old_manifest.sha256))

    runtime_status = status(storage_root, new_manifest)

    assert runtime_status.state == "mismatch"
    assert runtime_status.installed_manifest_sha256 == old_manifest.sha256
    assert verify(storage_root, new_manifest).state == "mismatch"


def test_materialize_runtime_retries_after_verification_failure(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write(source / "bin" / "ovrtx-bridge-server", "#!/bin/sh\n")
    _write(source / "bin" / "ovphysx-bridge-server", "#!/bin/sh\n")
    _write(source / "native" / "client.py", "client\n")
    _write(source / "runtime" / "payload", "runtime\n")
    archive = tmp_path / "runtime.tar.xz"
    archive_sha = _tar_xz_archive(source, archive)
    invalid_payload = json.loads(_manifest_bytes(archive, archive_sha).decode("utf-8"))
    invalid_payload["components"][0]["sha256"] = "0" * 64
    invalid = parse_manifest_bytes(
        json.dumps(invalid_payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    valid = parse_manifest_bytes(_manifest_bytes(archive, archive_sha))
    storage_root = tmp_path / "storage"

    try:
        materialize_runtime(invalid, storage_root)
    except RuntimeMaterializerError as exc:
        assert "sha256 mismatch" in str(exc)
    else:
        raise AssertionError("expected digest verification failure")

    current = materialize_runtime(valid, storage_root)

    assert (current / "bin/ovrtx-bridge-server").is_file()
    platform_root = storage_root / "runtimes" / "linux-x64"
    assert not (platform_root / ".downloads").exists()
    assert not (platform_root / ".extract").exists()
    assert not (platform_root / ".current.staging").exists()


def test_promotion_failure_restores_previous_runtime_and_cleans_staging(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    _write(source / "bin" / "ovrtx-bridge-server", "#!/bin/sh\n")
    _write(source / "bin" / "ovphysx-bridge-server", "#!/bin/sh\n")
    _write(source / "native" / "client.py", "client\n")
    _write(source / "runtime" / "payload", "runtime\n")
    archive = tmp_path / "runtime.tar.xz"
    manifest = parse_manifest_bytes(_manifest_bytes(archive, _tar_xz_archive(source, archive)))
    storage_root = tmp_path / "storage"
    current = storage_root / "runtimes" / "linux-x64" / "current"
    _write(current / "previous.txt", "previous\n")
    original_rename = Path.rename

    def fail_staging_promotion(path: Path, target: Path):
        if path.name == ".current.staging":
            raise OSError("injected promotion failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_staging_promotion)

    try:
        materialize_runtime(manifest, storage_root)
    except OSError as exc:
        assert "injected promotion failure" in str(exc)
    else:
        raise AssertionError("expected promotion failure")

    assert (current / "previous.txt").read_text(encoding="utf-8") == "previous\n"
    platform_root = current.parent
    assert not (platform_root / ".current.previous").exists()
    assert not (platform_root / ".current.staging").exists()
    assert not (platform_root / ".downloads").exists()
    assert not (platform_root / ".extract").exists()


def test_remove_runtime_removes_deep_installed_members(tmp_path: Path) -> None:
    storage_root = tmp_path / "Blender Foundation" / "5.1" / "extensions" / "user_default"
    current = storage_root / "runtimes" / "windows-x64" / "current"
    deep_file = current.joinpath(*(["deep-segment"] * 24), "payload.txt")
    _write(deep_file, "runtime\n")

    remove_runtime(storage_root, "windows-x64")

    assert not current.exists()


def test_remove_runtime_propagates_filesystem_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = tmp_path / "runtimes/linux-x64/current"
    _write(current / "loaded-module.pyd", "loaded")
    monkeypatch.setattr(
        runtime_store.shutil,
        "rmtree",
        lambda _path: (_ for _ in ()).throw(PermissionError()),
    )

    with pytest.raises(PermissionError):
        remove_runtime(tmp_path, "linux-x64")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path semantics")
def test_windows_local_runtime_folder_rejects_drive_relative_root(tmp_path: Path) -> None:
    assert runtime_materializer._local_runtime_folder(str(tmp_path)) == tmp_path
    with pytest.raises(RuntimeMaterializerError):
        runtime_materializer._local_runtime_folder("/artifact-set")


def test_runtime_verify_reports_broken_install(tmp_path: Path) -> None:
    archive = tmp_path / "empty.tar.xz"
    source = tmp_path / "source"
    _write(source / "bin" / "ovrtx-bridge-server", "#!/bin/sh\n")
    archive_sha = _tar_xz_archive(source, archive)
    manifest = parse_manifest_bytes(_manifest_bytes(archive, archive_sha))
    storage_root = tmp_path / "storage"
    current = storage_root / "runtimes" / "linux-x64" / "current"
    _write(current / "installed-runtime.json", json.dumps({"manifest_sha256": manifest.sha256}) + "\n")

    runtime_status = verify(storage_root, manifest)

    assert runtime_status.state == "broken"
    assert "target is missing" in runtime_status.message


def test_github_token_reads_hosts_file_when_gh_absent(tmp_path: Path, monkeypatch) -> None:
    for name in runtime_materializer._GITHUB_TOKEN_ENV:
        monkeypatch.delenv(name, raising=False)
    config_dir = tmp_path / "gh"
    _write(
        config_dir / "hosts.yml",
        "github.com:\n"
        "    oauth_token: gho_hostsfiletoken\n"
        "    user: octocat\n",
    )
    monkeypatch.setenv("GH_CONFIG_DIR", str(config_dir))
    # Blender's process often can't resolve gh on PATH.
    monkeypatch.setattr(runtime_materializer.shutil, "which", lambda _name: None)

    assert runtime_materializer._github_token() == "gho_hostsfiletoken"


def test_github_token_prefers_env_over_hosts_file(tmp_path: Path, monkeypatch) -> None:
    for name in runtime_materializer._GITHUB_TOKEN_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OVRTX_GITHUB_TOKEN", "env-token")
    config_dir = tmp_path / "gh"
    _write(config_dir / "hosts.yml", "github.com:\n    oauth_token: hosts-token\n")
    monkeypatch.setenv("GH_CONFIG_DIR", str(config_dir))

    assert runtime_materializer._github_token() == "env-token"


def _fake_elf_so(rpath: str, needed: list[str], *, tag: int = runtime_materializer._DT_RPATH) -> bytes:
    """A minimal 64-bit LE ELF exposing just the .dynamic/.dynstr the parser reads."""
    dynstr = bytearray(b"\x00")
    rpath_off = len(dynstr)
    dynstr += rpath.encode("ascii") + b"\x00"
    needed_offs = []
    for name in needed:
        needed_offs.append(len(dynstr))
        dynstr += name.encode("ascii") + b"\x00"

    entries = [(runtime_materializer._DT_NEEDED, off) for off in needed_offs]
    entries.append((tag, rpath_off))
    entries.append((0, 0))  # DT_NULL
    dyn = b"".join(struct.pack("<QQ", t, v) for t, v in entries)

    ehdr_size, shdr_size = 64, 64
    dynstr_off = ehdr_size
    dyn_off = dynstr_off + len(dynstr)
    dyn_off += (-dyn_off) % 8
    sh_off = dyn_off + len(dyn)
    sh_off += (-sh_off) % 8

    def shdr(sh_type: int, offset: int, size: int, link: int, entsize: int) -> bytes:
        header = bytearray(shdr_size)
        struct.pack_into("<I", header, 4, sh_type)
        struct.pack_into("<Q", header, 24, offset)
        struct.pack_into("<Q", header, 32, size)
        struct.pack_into("<I", header, 40, link)
        struct.pack_into("<Q", header, 56, entsize)
        return bytes(header)

    sections = b"".join(
        [
            shdr(0, 0, 0, 0, 0),  # SHT_NULL
            shdr(6, dyn_off, len(dyn), 2, 16),  # SHT_DYNAMIC, sh_link -> dynstr (index 2)
            shdr(3, dynstr_off, len(dynstr), 0, 0),  # SHT_STRTAB
        ]
    )

    blob = bytearray(sh_off + len(sections))
    blob[0:4] = b"\x7fELF"
    blob[4:6] = b"\x02\x01"  # 64-bit, little-endian
    struct.pack_into("<Q", blob, 0x28, sh_off)  # e_shoff
    struct.pack_into("<H", blob, 0x3A, shdr_size)  # e_shentsize
    struct.pack_into("<H", blob, 0x3C, 3)  # e_shnum
    struct.pack_into("<H", blob, 0x3E, 2)  # e_shstrndx
    blob[dynstr_off : dynstr_off + len(dynstr)] = dynstr
    blob[dyn_off : dyn_off + len(dyn)] = dyn
    blob[sh_off : sh_off + len(sections)] = sections
    return bytes(blob)


def _rpath_of(path: Path) -> str:
    info = runtime_materializer._read_elf_dynamic(path)
    assert info is not None and info.rpath_str_offset is not None
    data = path.read_bytes()
    end = data.index(b"\x00", info.rpath_str_offset)
    return data[info.rpath_str_offset : end].decode("ascii")


def _stage_native(tmp_path: Path, so_bytes: bytes, *, name: str = "ovsensors_worker_client.so") -> Path:
    root = tmp_path / "current"
    (root / "runtime" / "ovphysx-bridge-server" / "lib" / "libgrpc.so.41").parent.mkdir(parents=True, exist_ok=True)
    (root / "runtime" / "ovphysx-bridge-server" / "lib" / "libgrpc.so.41").write_bytes(b"stub")
    so_path = root / "native" / name
    so_path.parent.mkdir(parents=True, exist_ok=True)
    so_path.write_bytes(so_bytes)
    return root


def test_relink_repoints_native_client_rpath_to_bundled_runtime(tmp_path: Path) -> None:
    scratch = "/tmp/ov-blender-release/defunct/target-deps/grpc/release/lib"
    root = _stage_native(tmp_path, _fake_elf_so(scratch, ["libgrpc.so.41"]))
    so_path = root / "native" / "ovsensors_worker_client.so"

    runtime_materializer._relink_native_client_libraries(root, SimpleNamespace(platform="linux-x64"))

    assert _rpath_of(so_path) == "$ORIGIN/../runtime/ovphysx-bridge-server/lib"


def test_relink_repoints_worker_across_bundled_library_dirs(tmp_path: Path) -> None:
    root = tmp_path / "current"
    (root / "runtime" / "ovphysx-bridge-server" / "lib").mkdir(parents=True)
    (root / "runtime" / "ovphysx-bridge-server" / "lib" / "libgrpc.so.41").write_bytes(b"x")
    (root / "runtime" / "ovrtx-bridge-server" / "plugins").mkdir(parents=True)
    (root / "runtime" / "ovrtx-bridge-server" / "plugins" / "libov_25.11usd_ms.so").write_bytes(b"x")
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "libovrtx-dynamic.so").write_bytes(b"x")  # ships beside the worker
    worker = root / "bin" / "ovrtx-bridge-server"
    scratch = (
        "/tmp/ov-blender-release/defunct/ovrtx-bridge-server/release/bin:"
        "/tmp/ov-blender-release/defunct/ovphysx-bridge-server/release/lib"
    )
    worker.write_bytes(_fake_elf_so(scratch, ["libovrtx-dynamic.so", "libgrpc.so.41", "libov_25.11usd_ms.so"]))

    runtime_materializer._relink_native_client_libraries(root, SimpleNamespace(platform="linux-x64"))

    assert _rpath_of(worker) == (
        "$ORIGIN:$ORIGIN/../runtime/ovphysx-bridge-server/lib:"
        "$ORIGIN/../runtime/ovrtx-bridge-server/plugins"
    )


def test_relink_keeps_server_runtime_lib_searchable(tmp_path: Path) -> None:
    root = tmp_path / "current"
    library = root / "runtime" / "ovrtx-bridge-server" / "lib" / "libovrtx-dynamic.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"x")
    server = root / "bin" / "ovrtx-bridge-server"
    server.parent.mkdir(parents=True)
    server.write_bytes(
        _fake_elf_so("/tmp/ov-blender-release/defunct/ovrtx-bridge-server/lib", [library.name])
    )

    runtime_materializer._relink_native_client_libraries(root, SimpleNamespace(platform="linux-x64"))

    assert _rpath_of(server) == "$ORIGIN/../runtime/ovrtx-bridge-server/lib"


def test_relink_flips_runpath_to_rpath_for_transitive_search(tmp_path: Path) -> None:
    scratch = "/tmp/ov-blender-release/defunct/target-deps/grpc/release/lib"
    root = _stage_native(tmp_path, _fake_elf_so(scratch, ["libgrpc.so.41"], tag=runtime_materializer._DT_RUNPATH))
    so_path = root / "native" / "ovsensors_worker_client.so"

    runtime_materializer._relink_native_client_libraries(root, SimpleNamespace(platform="linux-x64"))

    info = runtime_materializer._read_elf_dynamic(so_path)
    assert info is not None and info.rpath_tag == runtime_materializer._DT_RPATH
    assert _rpath_of(so_path) == "$ORIGIN/../runtime/ovphysx-bridge-server/lib"


def test_relink_leaves_self_contained_extension_untouched(tmp_path: Path) -> None:
    scratch = "/tmp/ov-blender-release/defunct/lib"
    # NEEDED does not intersect the bundled runtime, so this extension is skipped.
    root = _stage_native(tmp_path, _fake_elf_so(scratch, ["libc.so.6"]))
    so_path = root / "native" / "ovsensors_worker_client.so"

    runtime_materializer._relink_native_client_libraries(root, SimpleNamespace(platform="linux-x64"))

    assert _rpath_of(so_path) == scratch


def test_relink_rejects_when_replacement_does_not_fit(tmp_path: Path) -> None:
    root = _stage_native(tmp_path, _fake_elf_so("/lib", ["libgrpc.so.41"]))

    try:
        runtime_materializer._relink_native_client_libraries(root, SimpleNamespace(platform="linux-x64"))
    except RuntimeMaterializerError as exc:
        assert "exceeds available space" in str(exc)
    else:  # pragma: no cover - the short rpath must not fit the $ORIGIN replacement
        raise AssertionError("expected RuntimeMaterializerError for an rpath that cannot fit")


def test_relink_is_a_no_op_on_non_linux_platforms(tmp_path: Path) -> None:
    scratch = "/tmp/ov-blender-release/defunct/target-deps/grpc/release/lib"
    root = _stage_native(tmp_path, _fake_elf_so(scratch, ["libgrpc.so.41"]))
    so_path = root / "native" / "ovsensors_worker_client.so"

    runtime_materializer._relink_native_client_libraries(root, SimpleNamespace(platform="windows-x64"))

    assert _rpath_of(so_path) == scratch

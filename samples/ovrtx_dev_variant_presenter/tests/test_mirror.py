# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remote-stage mirror: an http(s) USD is downloaded with its dependency closure into the
local data tree (preserving the host's path layout), then opened like any local file."""
import http.server
import threading
from functools import partial
from pathlib import Path

from dev_variant_presenter import mirror

ROOT_USDA = """#usda 1.0
(
    subLayers = [@./SubUSDs/lighting.usda@]
)
def Xform "World" {
    def Mesh "Body" {
        asset inputs:tex = @./textures/paint.png@
    }
}
"""
SUB_USDA = """#usda 1.0
def Xform "Lights" {}
"""


def _serve(tree: Path):
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(tree))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _make_remote(tmp_path: Path) -> Path:
    remote = tmp_path / "remote" / "Samples" / "Car"
    (remote / "SubUSDs").mkdir(parents=True)
    (remote / "textures").mkdir()
    (remote / "Car.usd").write_text(ROOT_USDA, encoding="utf-8")
    (remote / "SubUSDs" / "lighting.usda").write_text(SUB_USDA, encoding="utf-8")
    (remote / "textures" / "paint.png").write_bytes(b"\x89PNG fake")
    return tmp_path / "remote"


def test_is_url():
    assert mirror.is_url("https://host/x.usd")
    assert mirror.is_url("HTTP://host/x.usd")
    assert not mirror.is_url(r"C:\data\x.usd")
    assert not mirror.is_url("./rel/x.usd")


def test_mirror_downloads_full_dependency_closure(tmp_path):
    srv, base = _serve(_make_remote(tmp_path))
    data_root = tmp_path / "data"
    try:
        seen = []
        local = mirror.ensure_local(f"{base}/Samples/Car/Car.usd", data_root,
                                    progress=lambda n, f: seen.append(f))
        local = Path(local)
        assert local.is_file() and local.name == "Car.usd"
        host_root = local.parents[2]                       # <...>/<host>/Samples/Car/Car.usd
        assert (host_root / "Samples/Car/SubUSDs/lighting.usda").is_file()   # sublayer
        assert (host_root / "Samples/Car/textures/paint.png").is_file()      # asset attr
        assert Path(str(local) + ".mirror_complete").is_file()
        assert len(seen) >= 3                              # root + 2 deps reported
    finally:
        srv.shutdown()


def test_mirrored_stage_is_cached_and_opens_offline(tmp_path):
    srv, base = _serve(_make_remote(tmp_path))
    data_root = tmp_path / "data"
    url = f"{base}/Samples/Car/Car.usd"
    try:
        first = mirror.ensure_local(url, data_root)
    finally:
        srv.shutdown()                                     # network gone
    again = mirror.ensure_local(url, data_root)            # marker short-circuits: no requests
    assert again == first
    from pxr import Usd
    stage = Usd.Stage.Open(again)                          # composes fully from the mirror
    assert stage and stage.GetPrimAtPath("/World/Body").IsValid()


def test_mirror_is_isolated_from_user_data_folders(tmp_path):
    """Even when a user folder holds a copy of the same host tree, the mirror must not
    read from or write into it — tool cache lives ONLY under data/_mirror/."""
    srv, base = _serve(_make_remote(tmp_path))
    data_root = tmp_path / "data"
    host = base.split("//")[1].replace(":", "_")
    pre = data_root / "UserCurated" / host / "Samples" / "Car" / "SubUSDs"
    pre.mkdir(parents=True)
    (pre / "lighting.usda").write_text(SUB_USDA + "# user copy\n", encoding="utf-8")
    before = sorted(p for p in (data_root / "UserCurated").rglob("*") if p.is_file())
    try:
        local = mirror.ensure_local(f"{base}/Samples/Car/Car.usd", data_root)
    finally:
        srv.shutdown()
    assert str(data_root / "_mirror") in local                        # self-contained
    after = sorted(p for p in (data_root / "UserCurated").rglob("*") if p.is_file())
    assert after == before                                            # user folder untouched
    assert (data_root / "_mirror" / host / "Samples/Car/SubUSDs/lighting.usda").is_file()


def test_parse_s3_listing_keys_and_continuation():
    xml = ('<?xml version="1.0" encoding="UTF-8"?>'
           '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
           '<Name>b</Name><IsTruncated>true</IsTruncated>'
           '<NextContinuationToken>TOK</NextContinuationToken>'
           '<Contents><Key>a/x.usd</Key></Contents><Contents><Key>a/y.mdl</Key></Contents>'
           '</ListBucketResult>')
    keys, tok = mirror.parse_s3_listing(xml)
    assert keys == ["a/x.usd", "a/y.mdl"] and tok == "TOK"
    done = xml.replace("<IsTruncated>true</IsTruncated>", "<IsTruncated>false</IsTruncated>")
    assert mirror.parse_s3_listing(done)[1] is None       # last page -> no token
    import pytest
    with pytest.raises(Exception):
        mirror.parse_s3_listing("<html>nope</html>")       # non-listing -> fall back to fixpoint

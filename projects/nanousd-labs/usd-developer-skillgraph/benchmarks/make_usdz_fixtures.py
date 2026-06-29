#!/usr/bin/env python3
"""Create deterministic USDZ fixtures with 64-byte-aligned stored entries."""

from __future__ import annotations

import binascii
import struct
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "benchmarks" / "fixtures"
USDZ = FIXTURES / "usdz"


def deflate_raw(data: bytes) -> bytes:
    obj = zlib.compressobj(level=9, wbits=-15)
    return obj.compress(data) + obj.flush()


def local_padding(offset: int, name: str, align: bool) -> bytes:
    if not align:
        return b""
    base = offset + 30 + len(name.encode("utf-8"))
    pad = (-base) % 64
    return b"\0" * pad


def make_zip(entries: list[dict], align: bool = True, comment: bytes = b"") -> bytes:
    out = bytearray()
    central = bytearray()
    central_records: list[tuple[dict, int, int, int, bytes, bytes]] = []
    for entry in entries:
        name = entry["name"]
        data = entry.get("data", b"")
        method = int(entry.get("method", 0))
        flags = int(entry.get("flags", 0))
        payload = deflate_raw(data) if method == 8 else data
        crc = binascii.crc32(data) & 0xFFFFFFFF
        name_b = name.encode("utf-8")
        extra = local_padding(len(out), name, bool(entry.get("align", align)))
        local_offset = len(out)
        out += struct.pack(
            "<IHHHHHIIIHH",
            0x04034B50,
            20,
            flags,
            method,
            0,
            0,
            crc,
            len(payload),
            len(data),
            len(name_b),
            len(extra),
        )
        out += name_b
        out += extra
        out += payload
        central_records.append((entry, local_offset, crc, len(payload), name_b, data))

    cd_offset = len(out)
    for entry, local_offset, crc, payload_size, name_b, data in central_records:
        method = int(entry.get("method", 0))
        flags = int(entry.get("flags", 0))
        central += struct.pack(
            "<IHHHHHHIIIHHHHHII",
            0x02014B50,
            20,
            20,
            flags,
            method,
            0,
            0,
            crc,
            payload_size,
            len(data),
            len(name_b),
            0,
            0,
            0,
            0,
            0,
            local_offset,
        )
        central += name_b
    out += central
    out += struct.pack(
        "<IHHHHIIH",
        0x06054B50,
        0,
        0,
        len(central_records),
        len(central_records),
        len(central),
        cd_offset,
        len(comment),
    )
    out += comment
    return bytes(out)


def write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def read_text_lf(path: Path) -> bytes:
    """Read a text fixture with newlines normalized to LF.

    USDZ payloads are embedded verbatim, so the generator must be insensitive
    to how the source text fixtures were checked out. git autocrlf can present
    .usda sources as CRLF on Windows; normalizing here keeps the generated
    packages byte-identical across platforms (the deterministic guarantee).
    """
    return path.read_bytes().replace(b"\r\n", b"\n")


def main() -> int:
    one_root_usda = read_text_lf(FIXTURES / "one_root_prim.usda")
    one_root_usdc = (FIXTURES / "one_root_prim.usdc").read_bytes()
    minimal_usda = read_text_lf(FIXTURES / "minimal_empty.usda")

    write(USDZ / "default_root_usda.usdz", make_zip([{"name": "root.usda", "data": one_root_usda}]))
    write(USDZ / "default_root_usdc.usdz", make_zip([{"name": "root.usdc", "data": one_root_usdc}]))
    write(
        USDZ / "extra_assets.usdz",
        make_zip([
            {"name": "root.usda", "data": one_root_usda},
            {"name": "assets/notes.txt", "data": b"ignored asset\n"},
        ]),
    )
    write(
        USDZ / "explicit_root_usda.usdz",
        make_zip([
            {"name": "assets/notes.txt", "data": b"ignored first file\n"},
            {"name": "scenes/root.usda", "data": one_root_usda},
        ]),
    )
    write(
        USDZ / "explicit_root_usdc.usdz",
        make_zip([
            {"name": "assets/notes.txt", "data": b"ignored first file\n"},
            {"name": "scenes/root.usdc", "data": one_root_usdc},
        ]),
    )

    write(USDZ / "invalid_zip.usdz", b"not a zip archive")
    write(USDZ / "compressed_entry.usdz", make_zip([{"name": "root.usda", "data": one_root_usda, "method": 8}]))
    write(USDZ / "encrypted_entry.usdz", make_zip([{"name": "root.usda", "data": one_root_usda, "flags": 1}]))
    write(USDZ / "misaligned_entry.usdz", make_zip([{"name": "root.usda", "data": one_root_usda}], align=False))
    write(USDZ / "empty.usdz", make_zip([]))
    write(USDZ / "unsupported_usd.usdz", make_zip([{"name": "root.usd", "data": minimal_usda}]))
    write(USDZ / "nested_usdz.usdz", make_zip([{"name": "nested.usdz", "data": make_zip([])}]))
    write(
        USDZ / "duplicate_names.usdz",
        make_zip([
            {"name": "root.usda", "data": one_root_usda},
            {"name": "root.usda", "data": one_root_usda},
        ]),
    )
    write(USDZ / "unsafe_path.usdz", make_zip([{"name": "../root.usda", "data": one_root_usda}]))
    # Out-of-spec but readable: a non-empty EOCD comment. AOUSD §16.4.1.4 requires
    # comment length zero, but §16.4.2 permits readers to accept such files. See
    # docs/usdz-spec-errata.md#2.
    write(
        USDZ / "commented_eocd.usdz",
        make_zip([{"name": "root.usda", "data": one_root_usda}], comment=b"generated by make_usdz_fixtures"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

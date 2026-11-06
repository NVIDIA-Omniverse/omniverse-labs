# SPDX-License-Identifier: Apache-2.0
"""Independent wire fixtures exercise Blender-side owned copies without SDKs."""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import socket
import struct
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example.ovstream_shm import HeldFrame, OvstreamShmReader, ShmFrameError

pytestmark = pytest.mark.skipif(not sys.platform.startswith('linux'), reason='POSIX wire fixture; Windows kernel fixture is separate')


@pytest.fixture
def wire(tmp_path):
    name = "wire-fixture"
    path = tmp_path / ("ovstream-" + name)
    data = bytearray(512)
    struct.pack_into("<8I2Q3I", data, 0, 0x4D53564F, 2, 1, 2, 2, 2, 64, 192,
                     128, 1, 0, 1, os.getpid())
    struct.pack_into("<QQ4I", data, 128, 1, 12345, 2, 2, 64, 0)
    data[192:200] = bytes([1, 2, 3, 4, 5, 6, 7, 8])
    data[256:264] = bytes([9, 10, 11, 12, 13, 14, 15, 16])
    path.write_bytes(data)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(tmp_path / ("ovstream-" + name + ".sock")))
    listener.listen()
    listener.settimeout(2)
    messages = []
    def serve():
        try:
            conn, _ = listener.accept()
            with conn:
                conn.settimeout(2)
                messages.append(conn.recv(256))
                conn.sendall(b"ATTACH_OK protocolVersion=2 slotCount=2\n")
                messages.append(conn.recv(256))
        except (TimeoutError, OSError):
            pass
    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    ticket = HeldFrame(name, "epoch-a", 1, os.getpid(), 2, 2, 12345, "held-copy-1")
    def connect():
        return OvstreamShmReader(stream_name=name, epoch=ticket.epoch, producer_pid=os.getpid(),
                                 shm_directory=tmp_path, control_directory=tmp_path)
    def write(offset, format, *values):
        with path.open("r+b") as handle:
            handle.seek(offset)
            handle.write(struct.pack(format, *values))
    yield ticket, connect, write, path, messages
    listener.close()
    thread.join(3)


def test_pitched_bgra_preserves_alpha_and_orientation_and_detaches(wire):
    ticket, connect, _, path, messages = wire
    with connect() as reader:
        frame = reader.copy_frame(ticket)
        assert frame.rgba8 == bytes([3,2,1,4,7,6,5,8,11,10,9,12,15,14,13,16])
        assert frame.source_pitch_bytes == 64
        assert frame.sequence == 1
        with pytest.raises(TypeError):
            reader._mapping[192] = 0
    assert path.exists()  # A consumer never unlinks producer resources.
    assert messages[0] == b"ATTACH protocolVersion=2\n"
    with pytest.raises(ShmFrameError, match="closed"):
        reader.copy_frame(replace(ticket, sequence=2))
    reader.close()  # Idempotent close.


def test_copy_is_immutable_after_producer_reuses_ring(wire):
    ticket, connect, write, _, _ = wire
    with connect() as reader:
        frame = reader.copy_frame(ticket)
        write(192, "8B", *([255] * 8))
        assert frame.rgba8[:4] == bytes([3,2,1,4])
        with pytest.raises(ShmFrameError, match="duplicate"):
            reader.copy_frame(ticket)


def test_direct_owned_copy_pitched_partial_and_no_mapping_alias(wire):
    ticket,connect,write,_,_=wire
    owned=bytearray(b'x'*15)
    with connect() as reader:
        reader.copy_bytes_into(ticket,owned,destination_offset=2,count=11)
        expected=b'xx'+bytes(range(1,12))+b'xx'
        assert owned==expected
        write(192,'8B',*([255]*8))
        assert owned==expected


@pytest.mark.parametrize("change", [dict(epoch="epoch-b"), dict(stream_name="foreign"),
    dict(producer_pid=999999), dict(copy_token=""), dict(sequence=2), dict(width=1),
    dict(capture_timestamp_ns=12346)])
def test_reject_unheld_or_wrong_identity_frame(wire, change):
    ticket, connect, _, _, _ = wire
    with connect() as reader:
        with pytest.raises(ShmFrameError):
            reader.copy_frame(replace(ticket, **change))
        # Rejection does not advance the reader's accepted sequence.
        assert reader.copy_frame(ticket).sequence == 1


@pytest.mark.parametrize("offset,format,values", [
    (0,"I",(0,)), (4,"I",(1,)), (8,"I",(2,)), (12,"I",(9,)),
    (16,"I",(0,)), (24,"I",(7,)), (28,"I",(64,)), (32,"Q",(4096,)),
    (48,"I",(2,)), (52,"I",(0,)), (56,"I",(999999,)),
])
def test_reject_corrupt_region_before_mapping_use(wire, offset, format, values):
    _, connect, write, _, _ = wire
    write(offset, "<" + format, *values)
    with pytest.raises(ShmFrameError):
        connect()


@pytest.mark.parametrize("offset,format,value", [(128,"Q",0), (144,"I",0),
    (148,"I",3), (152,"I",4), (152,"I",128), (156,"I",1)])
def test_reject_bad_slot(wire, offset, format, value):
    ticket, connect, write, _, _ = wire
    with connect() as reader:
        write(offset, "<" + format, value)
        with pytest.raises(ShmFrameError):
            reader.copy_frame(ticket)


def test_producer_stop_during_copy_is_rejected(wire, monkeypatch):
    ticket, connect, write, _, _ = wire
    with connect() as reader:
        original = reader._read_region
        calls = 0
        def stop_after_first_read():
            nonlocal calls
            calls += 1
            if calls == 2:
                write(52, "<I", 0)
            return original()
        monkeypatch.setattr(reader, "_read_region", stop_after_first_read)
        with pytest.raises(ShmFrameError, match="stopped"):
            reader.copy_frame(ticket)


@pytest.mark.parametrize("name", ["../foreign", "x/y", "", "a" * 64])
def test_reject_unsafe_stream_name(name):
    with pytest.raises(ShmFrameError, match="stream name"):
        OvstreamShmReader(stream_name=name, epoch="a", producer_pid=os.getpid())

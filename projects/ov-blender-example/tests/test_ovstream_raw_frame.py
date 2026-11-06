# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example.frame_protocol import payload_header
from ovrtx_blender_example.ovstream_raw_frame import RawFrameLayout, RawFrameAssembler, copy_raw_frame
from ovrtx_blender_example.ovstream_shm import HeldFrame, ShmFrameError


def layout():
    return dict(**payload_header("/Render/Product", "HdrColor"),schema=1,encoding="raw_bytes_over_bgra8",byte_order="little",storage_width=3,storage_height=6,byte_count=72,
        planes=[dict(name="linear_rgba16f",format="rgba16f",width=3,height=2,offset_bytes=0,row_bytes=24,byte_count=48),
                dict(name="rgba8",format="rgba8",width=3,height=2,offset_bytes=48,row_bytes=12,byte_count=24)])


def test_half_float_bits_and_ldr_sidecar_survive_without_numerical_conversion():
    # Signed zero, infinities, distinct NaN payloads, subnormals, HDR >1.
    patterns = [0,0x8000,0x7c00,0xfc00,0x7e01,0x7fff,1,0x8001,0x4000,0x7bff,0x3c00,0xbc00]*2
    hdr = struct.pack("<24H", *patterns)
    ldr = bytes(range(24))
    class Reader:
        def copy_bytes(self, held):
            return SimpleNamespace(data=hdr+ldr,epoch=held.epoch,sequence=held.sequence)
    held = HeldFrame("blender-test","a",1,123,3,6,1,"hold")
    copied = copy_raw_frame(Reader(), held, layout())
    assert copied.payload("linear_rgba16f",pixel_format="rgba16f",width=3,height=2) == hdr
    assert copied.payload("rgba8",pixel_format="rgba8",width=3,height=2) == ldr
    assert struct.unpack("<24H", copied.planes[0].data) == tuple(patterns)
    with pytest.raises(ShmFrameError, match="requested format"):
        copied.payload("linear_rgba16f",pixel_format="rgba8",width=3,height=2)


@pytest.mark.parametrize("field,value", [("schema",2),("schema",True),("byte_order","big"),("encoding","bgra8"),("storage_width",0),
    ("storage_width",True),("storage_height",8193),("byte_count",73),("byte_count",0)])
def test_reject_bad_storage_descriptor(field,value):
    source=layout()
    source[field]=value
    with pytest.raises(ShmFrameError):
        RawFrameLayout.from_mapping(source)


@pytest.mark.parametrize("field,value", [("offset_bytes",1),("format","rgba16"),("row_bytes",12),
    ("byte_count",24),("width",0),("name","../outside"),("name","rgba8")])
def test_reject_plane_overlap_inconsistent_formats_and_bounds(field,value):
    source=layout()
    source["planes"][0][field]=value
    with pytest.raises(ShmFrameError):
        RawFrameLayout.from_mapping(source)


def test_bad_layout_rejected_before_any_shared_read():
    class Reader:
        def copy_bytes(self, _):
            raise AssertionError("Invalid descriptor must not read memory")
    held=HeldFrame("test","a",1,123,6,3,1,"hold")
    with pytest.raises(ShmFrameError,match="dimensions disagree"):
        copy_raw_frame(Reader(),held,layout())


def chunk(index):
    return dict(transfer_id="a"*32,index=index,count=3,offset_bytes=index*24,byte_count=24,total_bytes=72)


def held_chunk(index):
    return HeldFrame("test","a",index+1,123,3,2,1,f"hold{index}")


class ChunkReader:
    def copy_bytes(self, held):
        offset=(held.sequence-1)*24
        return SimpleNamespace(data=bytes(range(offset,offset+24)))


def test_complete_chunked_frame_retains_exact_plane_offsets_and_owned_bytes():
    assembler=RawFrameAssembler(held_chunk(0),layout(),chunk(0))
    for index in range(3):
        assert assembler.append(ChunkReader(),held_chunk(index),layout(),chunk(index)) == (index==2)
    result=assembler.finish()
    assert result.payload("linear_rgba16f",pixel_format="rgba16f",width=3,height=2)==bytes(range(48))
    assert result.payload("rgba8",pixel_format="rgba8",width=3,height=2)==bytes(range(48,72))
    with pytest.raises(ShmFrameError,match="already"):
        assembler.finish()


@pytest.mark.parametrize("field,value", [("transfer_id","b"*32),("index",0),("count",4),
    ("offset_bytes",23),("byte_count",23),("total_bytes",71)])
def test_reject_reordered_duplicate_foreign_or_truncated_chunks(field,value):
    assembler=RawFrameAssembler(held_chunk(0),layout(),chunk(0))
    assembler.append(ChunkReader(),held_chunk(0),layout(),chunk(0))
    changed=chunk(1)
    changed[field]=value
    with pytest.raises(ShmFrameError):
        assembler.append(ChunkReader(),held_chunk(1),layout(),changed)
    with pytest.raises(ShmFrameError,match="incomplete"):
        assembler.finish()
    assert not assembler.append(ChunkReader(),held_chunk(1),layout(),chunk(1))


def test_reject_stale_worker_epoch_before_reading_memory():
    from dataclasses import replace
    assembler=RawFrameAssembler(held_chunk(0),layout(),chunk(0))
    with pytest.raises(ShmFrameError,match="identity"):
        assembler.append(ChunkReader(),replace(held_chunk(0),epoch="old"),layout(),chunk(0))


def test_direct_copy_assembly_retains_owned_readonly_planes_without_legacy_copy():
    class DirectReader:
        def copy_bytes(self,*args):pytest.fail('Direct path must not allocate intermediate chunk bytes')
        def copy_bytes_into(self,held,destination,*,destination_offset,count):
            start=(held.sequence-1)*24
            destination[destination_offset:destination_offset+count]=bytes(range(start,start+count))
    assembler=RawFrameAssembler(held_chunk(0),layout(),chunk(0))
    for index in range(3):assembler.append(DirectReader(),held_chunk(index),layout(),chunk(index))
    result=assembler.finish()
    assert bytes(result.planes[0].data)==bytes(range(48))
    assert result.planes[0].data.readonly
    assert not assembler._buffer


def test_partial_direct_copy_failure_cannot_finish_or_retry_owned_assembly():
    class BrokenReader:
        def copy_bytes_into(self,held,destination,**kwargs):
            destination[:4]=b'bad!'
            raise ShmFrameError('Producer changed during copy')
    assembler=RawFrameAssembler(held_chunk(0),layout(),chunk(0))
    with pytest.raises(ShmFrameError,match='changed'):assembler.append(BrokenReader(),held_chunk(0),layout(),chunk(0))
    assert not assembler._buffer
    with pytest.raises(ShmFrameError,match='incomplete'):assembler.finish()
    with pytest.raises(ShmFrameError,match='failed'):assembler.append(ChunkReader(),held_chunk(0),layout(),chunk(0))


def test_house_hdr_can_use_one_large_carrier_and_still_reject_oversize():
    source=dict(**payload_header("/Render/Product", "HdrColor"),schema=1,encoding="raw_bytes_over_bgra8",byte_order="little",storage_width=2544,storage_height=2596,byte_count=2530*1305*8,
        planes=[dict(name="linear_rgba16f",format="rgba16f",width=2530,height=1305,offset_bytes=0,row_bytes=2530*8,byte_count=2530*1305*8)])
    capacity=2544*2596*4
    held=HeldFrame("test","epoch",1,123,2544,2596,1,"hold")
    chunk=dict(transfer_id="a"*32,index=0,count=1,offset_bytes=0,byte_count=source['byte_count'],total_bytes=source['byte_count'])
    assembler=RawFrameAssembler(held,source,chunk)
    class DirectReader:
        def copy_bytes_into(self, descriptor, destination, *, destination_offset, count):
            destination[0:4]=b"HDR!"
            destination[-4:]=b"last"
    assert assembler.append(DirectReader(),held,source,chunk)
    frame=assembler.finish()
    assert frame.planes[0].data[:4]==b"HDR!" and frame.planes[0].data[-4:]==b"last"
    from dataclasses import replace
    with pytest.raises(ShmFrameError,match="carrier budget"):
        RawFrameAssembler(replace(held,width=8192,height=2049),source,chunk)

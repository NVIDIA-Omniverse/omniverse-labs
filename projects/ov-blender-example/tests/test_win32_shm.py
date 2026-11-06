# SPDX-License-Identifier: Apache-2.0
"""Win32 transport contracts mocked on Linux; no SDK/GPU imports required."""
import ctypes
import json
import mmap
import os
from pathlib import Path
import struct
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'addon'))
from ovrtx_blender_example import win32_shm as win
from ovrtx_blender_example import ovstream_shm as reader
from ovrtx_blender_example import ovstream_shm_cleanup as cleanup


def wire():
    data = bytearray(512)
    struct.pack_into('<8I2Q3I',data,0,0x4D53564F,2,1,2,2,2,64,192,128,1,0,1,os.getpid())
    struct.pack_into('<QQ4I',data,128,1,123,2,2,64,0)
    data[192:200] = b'\x01\x02\x03\x04'*2
    data[256:264] = b'\x05\x06\x07\x08'*2
    return data


class FakeKernel:
    def __init__(self, data=None):
        self.error=0;self.closed=[];self.mapped=[];self.unmapped=[];self.opens=[]
        self.buffer=ctypes.create_string_buffer(bytes(data or wire()))
        self.reads=[b'ATTACH_OK protocolVersion=2 slotCount=2\n']
        self.writes=[];self.state=258;self.mode=None
    def OpenFileMappingW(self,access,inherit,name):
        self.opens.append((access,inherit,name));return 101
    def MapViewOfFile(self,handle,access,high,low,size):
        self.mapped.append((access,size));return ctypes.addressof(self.buffer)
    def UnmapViewOfFile(self,address):self.unmapped.append(address);return 1
    def CloseHandle(self,handle):self.closed.append(handle);return 1
    def OpenProcess(self,*args):return 102
    def WaitForSingleObject(self,*args):return self.state
    def CreateFileW(self,*args):self.pipe_name=args[0];return 103
    def SetNamedPipeHandleState(self,handle,mode,*args):self.mode=mode._obj.value;return 1
    def ReadFile(self,handle,buffer,size,count,unused):
        if not self.reads:self.error=232;return 0
        value=self.reads[0][:size];self.reads[0]=self.reads[0][size:]
        if not self.reads[0]:self.reads.pop(0)
        ctypes.memmove(buffer,value,len(value));count._obj.value=len(value);return 1
    def WriteFile(self,handle,buffer,size,count,unused):
        count._obj.value=min(3,size);self.writes.append(ctypes.string_at(buffer,count._obj.value));return 1
    def WaitNamedPipeW(self,*args):return 1
    def OpenEventW(self,*args):return 104


@pytest.fixture
def kernel(monkeypatch):
    api=FakeKernel();monkeypatch.setattr(win,'_API',api)
    monkeypatch.setattr(win.C,'get_last_error',lambda:api.error,raising=False)
    return api


def test_existing_mapping_readonly_bounded_open_and_close(kernel):
    mapping=win.ReadOnlyMapping('fixture')
    assert kernel.opens==[(4,False,'Local\\ovstream-fixture')]
    assert kernel.mapped==[(4,128),(4,512)]
    assert mapping[192:200]==b'\x01\x02\x03\x04'*2
    with pytest.raises(TypeError):mapping[192]=0
    mapping.close();mapping.close()
    assert kernel.closed==[101] and len(kernel.unmapped)==2


def test_mapping_never_creates_missing_region(kernel):
    kernel.OpenFileMappingW=lambda *args:0;kernel.error=2
    with pytest.raises(OSError,match='OpenFileMappingW'):win.ReadOnlyMapping('missing')
    assert kernel.mapped==[]


def test_mapping_copy_into_owned_destination_checks_both_extents(kernel):
    mapping=win.ReadOnlyMapping('fixture')
    owned=bytearray(b'x'*12)
    mapping.copy_into(owned,destination_offset=2,source_offset=192,count=8)
    assert owned==b'xx'+b'\x01\x02\x03\x04'*2+b'xx'
    for source,offset,count in [(508,0,8),(192,8,8),(-1,0,1),(192,-1,1),(192,0,True)]:
        with pytest.raises(ValueError):mapping.copy_into(owned,destination_offset=offset,source_offset=source,count=count)
    for wrong in [bytes(12),memoryview(owned)]:
        with pytest.raises(TypeError):mapping.copy_into(wrong,destination_offset=0,source_offset=192,count=8)
    mapping.close()
    with pytest.raises(ValueError,match='closed'):mapping.copy_into(owned,destination_offset=0,source_offset=192,count=8)


@pytest.mark.parametrize('count',[5,16])
def test_windows_direct_owned_copy_pitched_rows_and_partial_tail(kernel,monkeypatch,count):
    monkeypatch.setattr(reader.sys,'platform','win32')
    ticket=reader.HeldFrame('fixture','epoch',1,os.getpid(),2,2,123,'hold')
    owned=bytearray(b'x'*(count+4))
    with reader.OvstreamShmReader(stream_name='fixture',epoch='epoch',producer_pid=os.getpid()) as consumer:
        consumer.copy_bytes_into(ticket,owned,destination_offset=2,count=count)
        expected=(b'\x01\x02\x03\x04'*2+b'\x05\x06\x07\x08'*2)[:count]
        assert owned==b'xx'+expected+b'xx'
        ctypes.memset(ctypes.addressof(kernel.buffer)+192,255,8)
        assert owned==b'xx'+expected+b'xx'
        with pytest.raises(reader.ShmFrameError,match='duplicate'):consumer.copy_bytes_into(ticket,owned,count=count)


def test_windows_direct_owned_copy_failure_keeps_reader_cursor(kernel,monkeypatch):
    monkeypatch.setattr(reader.sys,'platform','win32')
    ticket=reader.HeldFrame('fixture','epoch',1,os.getpid(),2,2,123,'hold')
    with reader.OvstreamShmReader(stream_name='fixture',epoch='epoch',producer_pid=os.getpid()) as consumer:
        for dest,count in [(bytes(16),16),(bytearray(15),16),(bytearray(17),17)]:
            with pytest.raises(reader.ShmFrameError):consumer.copy_bytes_into(ticket,dest,count=count)
        mapping=consumer._mapping;original=mapping.copy_into
        def corrupt(*args,**kwargs):
            original(*args,**kwargs)
            struct.pack_into('<Q',kernel.buffer,128,2)
        mapping.copy_into=corrupt
        with pytest.raises(reader.ShmFrameError,match='modified'):consumer.copy_bytes_into(ticket,bytearray(16))
        assert consumer._last_sequence==0
        struct.pack_into('<Q',kernel.buffer,128,1)
        mapping.copy_into=original
        consumer.copy_bytes_into(ticket,bytearray(16))
        assert consumer._last_sequence==1


@pytest.mark.parametrize('mutation',['epoch_sequence','producer','stopped','reserved'])
def test_direct_owned_copy_rejects_late_identity_or_slot_mutation(kernel,monkeypatch,mutation):
    monkeypatch.setattr(reader.sys,'platform','win32')
    ticket=reader.HeldFrame('fixture','epoch',1,os.getpid(),2,2,123,'hold')
    with reader.OvstreamShmReader(stream_name='fixture',epoch='epoch',producer_pid=os.getpid()) as consumer:
        mapping=consumer._mapping;original=mapping.copy_into
        def corrupt(*args,**kwargs):
            original(*args,**kwargs)
            if mutation=='epoch_sequence':struct.pack_into('<Q',kernel.buffer,40,2)
            elif mutation=='producer':struct.pack_into('<I',kernel.buffer,56,os.getpid()+1)
            elif mutation=='stopped':struct.pack_into('<I',kernel.buffer,52,0)
            else:struct.pack_into('<I',kernel.buffer,156,1)
        mapping.copy_into=corrupt
        with pytest.raises(reader.ShmFrameError):consumer.copy_bytes_into(ticket,bytearray(16))
        assert consumer._last_sequence==0


def test_direct_owned_full_tight_chunk_copies_without_intermediate_slice(kernel,monkeypatch):
    monkeypatch.setattr(reader.sys,'platform','win32')
    struct.pack_into('<QQ4I',kernel.buffer,128,1,123,2,2,8,0)
    expected=bytes(range(16))
    ctypes.memmove(ctypes.addressof(kernel.buffer)+192,expected,16)
    ticket=reader.HeldFrame('fixture','epoch',1,os.getpid(),2,2,123,'hold')
    with reader.OvstreamShmReader(stream_name='fixture',epoch='epoch',producer_pid=os.getpid()) as consumer:
        original=win.ReadOnlyMapping.__getitem__
        def headers_only(mapping,key):
            if key.start is not None and key.start>=192:pytest.fail('Pixel bytes allocated an intermediate slice')
            return original(mapping,key)
        monkeypatch.setattr(win.ReadOnlyMapping,'__getitem__',headers_only)
        destination=bytearray(16)
        consumer.copy_bytes_into(ticket,destination)
        assert destination==expected


def test_direct_owned_copy_mmap_fallback_releases_all_scoped_views(kernel,monkeypatch,tmp_path):
    monkeypatch.setattr(reader.sys,'platform','win32')
    ticket=reader.HeldFrame('fixture','epoch',1,os.getpid(),2,2,123,'hold')
    path=tmp_path/'readonly-ring';path.write_bytes(wire())
    with reader.OvstreamShmReader(stream_name='fixture',epoch='epoch',producer_pid=os.getpid()) as consumer:
        consumer._mapping.close()
        with path.open('rb') as file:
            consumer._mapping=mmap.mmap(file.fileno(),512,access=mmap.ACCESS_READ)
        destination=bytearray(11)
        consumer.copy_bytes_into(ticket,destination,count=11)
        assert destination==b'\x01\x02\x03\x04'*2+b'\x05\x06\x07'
    # Closing the mmap would raise BufferError if any borrowed view survived.


def test_reject_large_header_before_full_mapping(kernel):
    struct.pack_into('<Q',kernel.buffer,32,2**40)
    with pytest.raises(ValueError,match='bounds'):win.ReadOnlyMapping('fixture')
    assert kernel.mapped==[(4,128)] and kernel.closed==[101]


def test_mapping_close_failure_retains_handle_for_retry(kernel):
    mapping=win.ReadOnlyMapping('fixture')
    original=kernel.UnmapViewOfFile
    kernel.UnmapViewOfFile=lambda *_:0
    with pytest.raises(OSError):mapping.close()
    assert mapping._handle==101 and mapping._address and not kernel.closed
    kernel.UnmapViewOfFile=original;mapping.close()
    assert kernel.closed==[101]


def test_liveness_is_readonly_and_access_denied_is_not_death(kernel):
    assert win.process_exists(123)
    kernel.state=0;assert not win.process_exists(123)
    kernel.OpenProcess=lambda *args:0;kernel.error=87;assert not win.process_exists(123)
    kernel.error=5;assert win.process_exists(123)
    kernel.error=6
    with pytest.raises(OSError):win.process_exists(123)


def test_pipe_byte_mode_handles_partial_writes_and_attach(kernel):
    pipe=win.ControlPipe('fixture',.5)
    assert kernel.pipe_name=='\\\\.\\pipe\\ovstream-fixture' and kernel.mode==1
    pipe.sendall(b'ATTACH protocolVersion=2\n')
    reply=b''
    while not reply.endswith(b'\n'):reply+=pipe.recv(1)
    assert reply==b'ATTACH_OK protocolVersion=2 slotCount=2\n'
    assert b''.join(kernel.writes)==b'ATTACH protocolVersion=2\n'
    pipe.close();pipe.close();assert kernel.closed==[103]


def test_empty_pipe_and_absent_endpoint_have_bounded_timeout(kernel):
    pipe=win.ControlPipe('fixture',.005);kernel.reads=[]
    with pytest.raises(TimeoutError):pipe.recv(1)
    pipe.close()
    kernel.CreateFileW=lambda *args:win._INVALID_HANDLE;kernel.error=2
    with pytest.raises(TimeoutError):win.ControlPipe('missing',.005)


def test_windows_reader_same_wire_copy_epoch_and_cleanup_without_kill(kernel,monkeypatch,tmp_path):
    monkeypatch.setattr(reader.sys,'platform','win32')
    monkeypatch.setattr(reader.os,'kill',lambda *_:pytest.fail('Never signal a Windows PID'))
    ticket=reader.HeldFrame('fixture','epoch',1,os.getpid(),2,2,123,'hold')
    with reader.OvstreamShmReader(stream_name='fixture',epoch='epoch',producer_pid=os.getpid()) as consumer:
        assert consumer.copy_frame(ticket).rgba8==b'\x03\x02\x01\x04'*2+b'\x07\x06\x05\x08'*2
    assert 101 in kernel.closed and 103 in kernel.closed
    kernel.state=0
    path=tmp_path/'registry.json';path.write_text(json.dumps(dict(schema=1,pid=123,epoch='e',streams=['blender-'+'a'*32])))
    assert cleanup.cleanup_registry(path,expected_pid=123)==[]
    assert path.exists()


def test_windows_cleanup_refuses_alive_producer(kernel,monkeypatch,tmp_path):
    monkeypatch.setattr(cleanup.sys,'platform','win32')
    monkeypatch.setattr(cleanup.os,'kill',lambda *_:pytest.fail('Never signal a Windows PID'))
    with pytest.raises(RuntimeError,match='still exists'):
        cleanup.cleanup_registry(tmp_path/'missing',expected_pid=123)


@pytest.mark.parametrize('reply',[b'ATTACH_OK protocolVersion=1 slotCount=2\n',b'x'*256,b''])
def test_rejected_windows_handshake_closes_pipe_without_mapping(kernel,monkeypatch,reply):
    monkeypatch.setattr(reader.sys,'platform','win32')
    kernel.reads=[reply]
    with pytest.raises((reader.ShmFrameError,TimeoutError)):
        reader.OvstreamShmReader(stream_name='fixture',epoch='e',producer_pid=os.getpid(),timeout_seconds=.02)
    assert kernel.closed==[103] and not kernel.opens


def test_windows_registry_rejects_foreign_identity_before_kernel_cleanup(kernel,monkeypatch,tmp_path):
    monkeypatch.setattr(cleanup.sys,'platform','win32');kernel.state=0
    path=tmp_path/'registry.json';path.write_text(json.dumps(dict(schema=1,pid=999,epoch='e',streams=[])))
    with pytest.raises(ValueError,match='expected worker'):cleanup.cleanup_registry(path,expected_pid=123)
    assert path.exists() and not kernel.opens


def test_resource_probe_opens_existing_only_and_closes(kernel):
    assert all(win.existing_resources('fixture').values())
    assert kernel.closed==[101,104]
    kernel.OpenFileMappingW=kernel.OpenEventW=kernel.WaitNamedPipeW=lambda *args:0
    kernel.error=2;assert not any(win.existing_resources('fixture').values())


def test_capacity_is_minimum_of_physical_commit_and_address_space(kernel):
    def status(pointer):
        pointer._obj.avail_phys=1000;pointer._obj.avail_page=500;pointer._obj.avail_virtual=2000;return 1
    kernel.GlobalMemoryStatusEx=status
    assert win.available_memory()==500

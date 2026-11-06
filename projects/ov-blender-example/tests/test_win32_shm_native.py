# SPDX-License-Identifier: Apache-2.0
"""Real Windows kernel interoperability fixture; no NVIDIA SDK/GPU required.

This independently creates documented wire bytes and a duplex byte pipe.
It proves ctypes mapping/pipe behavior, not official OVStream CUDA publication.
"""
import ctypes as C
import os
from pathlib import Path
import struct
import sys
import threading
import uuid

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'addon'))
from ovrtx_blender_example import win32_shm as win
from ovrtx_blender_example.ovstream_shm import OvstreamShmReader,HeldFrame

pytestmark=pytest.mark.skipif(sys.platform!='win32',reason='Requires real Windows kernel')


def test_actual_windows_existing_mapping_pipe_copy_and_kernel_cleanup():
    api=win._api();name='blender-'+uuid.uuid4().hex;base='ovstream-'+name
    for symbol,args,result in [
        ('CreateFileMappingW',[C.c_void_p,C.c_void_p,C.c_uint32,C.c_uint32,C.c_uint32,C.c_wchar_p],C.c_void_p),
        ('CreateNamedPipeW',[C.c_wchar_p,C.c_uint32,C.c_uint32,C.c_uint32,C.c_uint32,C.c_uint32,C.c_uint32,C.c_void_p],C.c_void_p),
        ('CreateEventW',[C.c_void_p,C.c_int32,C.c_int32,C.c_wchar_p],C.c_void_p),
        ('ConnectNamedPipe',[C.c_void_p,C.c_void_p],C.c_int32),
        ('DisconnectNamedPipe',[C.c_void_p],C.c_int32)]:
        function=getattr(api,symbol);function.argtypes=args;function.restype=result
    region=api.CreateFileMappingW(C.c_void_p(-1),None,4,0,512,'Local\\'+base)
    assert region
    address=api.MapViewOfFile(region,2,0,0,512)
    assert address
    data=bytearray(512)
    struct.pack_into('<8I2Q3I',data,0,0x4D53564F,2,1,2,2,2,64,192,128,1,0,1,os.getpid())
    struct.pack_into('<QQ4I',data,128,1,12345,2,2,64,0)
    data[192:200]=bytes([1,2,3,4,5,6,7,8]);data[256:264]=bytes([9,10,11,12,13,14,15,16])
    C.memmove(address,bytes(data),512)
    wake=api.CreateEventW(None,True,False,'Local\\'+base+'-wake')
    pipe=api.CreateNamedPipeW('\\\\.\\pipe\\'+base,3|0x00080000,0,1,1024,1024,1000,None)
    assert wake and pipe and pipe!=win._INVALID_HANDLE
    messages=[];errors=[]
    def serve():
        try:
            connected=api.ConnectNamedPipe(pipe,None)
            assert connected or C.get_last_error()==535
            buffer=C.create_string_buffer(256);count=C.c_uint32()
            assert api.ReadFile(pipe,buffer,256,C.byref(count),None)
            messages.append(buffer.raw[:count.value])
            response=b'ATTACH_OK protocolVersion=2 slotCount=2\n'
            assert api.WriteFile(pipe,C.create_string_buffer(response),len(response),C.byref(count),None)
            # EOF is also a documented detach; do not require a late DETACH verb.
            api.ReadFile(pipe,buffer,256,C.byref(count),None)
        except BaseException as error:errors.append(error)
    thread=threading.Thread(target=serve,daemon=True);thread.start()
    try:
        ticket=HeldFrame(name,'win-kernel',1,os.getpid(),2,2,12345,'held')
        with OvstreamShmReader(stream_name=name,epoch='win-kernel',producer_pid=os.getpid(),timeout_seconds=2) as reader:
            copied=reader.copy_frame(ticket)
            assert copied.rgba8==bytes([3,2,1,4,7,6,5,8,11,10,9,12,15,14,13,16])
            assert win.process_exists(os.getpid())
        # Consumer must not destroy the producer's open objects.
        assert win.existing_resources(name)['Local\\'+base]
        thread.join(3);assert not thread.is_alive();assert not errors
        assert messages==[b'ATTACH protocolVersion=2\n']
    finally:
        api.DisconnectNamedPipe(pipe);api.CloseHandle(pipe)
        api.CloseHandle(wake);api.UnmapViewOfFile(address);api.CloseHandle(region)
    assert not any(win.existing_resources(name).values())

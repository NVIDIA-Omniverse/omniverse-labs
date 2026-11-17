/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Real original comm-stream producer: deallocation-only memory has no location
// metadata but remains a valid recorded measurement. No Python capture parsing.
#define main ingestionFixtureMain
#include "native_ingestion_fixture.cpp"
#undef main
#include "PxProfileMemory.h"
#include "PxProfileEventBufferClient.h"
#include "PxPvdCommStreamTypes.h"
#include <vector>
struct Bytes:physx::profile::PxProfileEventBufferClient {
    std::vector<uint8_t> data;
    void handleBufferFlush(const uint8_t* bytes,uint32_t size) override {data.insert(data.end(),bytes,bytes+size);}
    void handleClientRemoved() override {}
};
int main(int argc,char** argv) {
    if(argc!=3)return 2;
    struct Error:PxErrorCallback {void reportError(PxErrorCode::Enum,const char*,const char*,int) override {}} error;
    pvdapp::CheckingAllocator allocator;
    auto* foundation=PxCreateFoundation(PX_FOUNDATION_VERSION,allocator,error);
    if(!foundation)return 3;
    {
        std::ofstream out(argv[1],std::ios::binary);
        if(!out)return 4;
        write(out,StreamInitialization::getStreamId());write(out,StreamInitialization::getStreamVersion());
        write(out,PxU64(1));write(out,PxU64(1));write(out,PxU32(sizeof(void*)==8));
        const auto type=getPvdNamespacedNameForType<uint8_t>();
        const char* names[]={"physx3","PxScene","frame","_debugger_","PxProfileMemoryEventBuffer","events",
            PvdCommStreamEmbeddedTypes::getMemoryEventStreamSemantic(),type.mNamespace,type.mName,"UserCustomedEvent"};
        for(uint32_t i=0;i<10;++i)event(out,StringHandleEvent(names[i],i+1));
        const StreamNamespacedName scene(StringHandle(1),StringHandle(2)),buffer(StringHandle(4),StringHandle(5)),
            bytes(StringHandle(8),StringHandle(9)),log(StringHandle(4),StringHandle(10));
        event(out,CreateClass(scene));event(out,CreateInstance(scene,100));
        event(out,CreateClass(buffer));
        event(out,CreateProperty(buffer,StringHandle(6),StringHandle(7),bytes,PropertyType::Array,{}));
        event(out,CreateInstance(buffer,200));
        // The core provides the error-log class/Events schema. Presence must not
        // depend on a nonempty Events array.
        event(out,CreateInstance(log,300));
        event(out,BeginSection(100,StringHandle(3),100));event(out,EndSection(100,StringHandle(3),150));
        event(out,BeginSection(100,StringHandle(3),200));
        Bytes collected;
        auto& memory=physx::profile::PxProfileMemoryEventBuffer::createMemoryEventBuffer(allocator);
        memory.addClient(collected);
        if(std::string(argv[2])!="empty")memory.onDeallocation(reinterpret_cast<void*>(0x1234));
        memory.flushProfileEvents();memory.removeClient(collected);memory.release();
        event(out,SetPropertyValue(200,StringHandle(6),DataRef<const uint8_t>(collected.data.data(),collected.data.size()),bytes,collected.data.size()));
        event(out,EndSection(100,StringHandle(3),250));
        event(out,BeginSection(100,StringHandle(3),300));event(out,EndSection(100,StringHandle(3),350));
        event(out,StreamEndEvent());
    }
    foundation->release();return 0;
}

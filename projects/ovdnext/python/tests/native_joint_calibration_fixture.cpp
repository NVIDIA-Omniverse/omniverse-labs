/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Captures produced by the pinned writer, consumed by the real owned API.
#include "OmniPvdWriter.h"
#include "OmniPvdFileWriteStream.h"
#include "OmniPvdLibraryFunctions.h"
#include <string>
#include <cstdint>
int main(int argc,char** argv) {
    if(argc!=3)return 2;
    const std::string mode=argv[2];
    auto* w=createOmniPvdWriter();auto* stream=createOmniPvdFileWriteStream();
    stream->setFileName(argv[1]);if(!stream->openStream())return 3;w->setWriteStream(*stream);
    const auto real=OmniPvdDataType::eFLOAT32,ref=OmniPvdDataType::eOBJECT_HANDLE;
    auto scene=w->registerClass("PxScene"),body=w->registerClass("PxRigidDynamic"),shape=w->registerClass("PxShape");
    auto geometry=w->registerClass(mode=="plane"?"PxPlaneGeometry":"PxBoxGeometry");
    auto surface=w->registerClass("PxDeformableSurface"),triGeom=w->registerClass("PxTriangleMeshGeometry"),mesh=w->registerClass("PxTriangleMesh");
    auto actors=w->registerUniqueListAttribute(scene,"actors",ref),shapes=w->registerUniqueListAttribute(body,"shapes",ref);
    auto surfaceShapes=w->registerUniqueListAttribute(surface,"shapes",ref),scale=w->registerAttribute(scene,"tolerancesScale",real,2);
    auto global=w->registerAttribute(body,"globalPose",real,7),local=w->registerAttribute(shape,"localPose",real,7);
    auto geom=w->registerAttribute(shape,"geometry",ref,1),half=w->registerAttribute(geometry,"halfExtents",real,3);
    auto triangle=w->registerAttribute(triGeom,"triangleMesh",ref,1),verts=w->registerAttribute(mesh,"verts",real,0);
    auto tris=w->registerAttribute(mesh,"tris",OmniPvdDataType::eUINT32,0);
    auto create=[&](OmniPvdClassHandle cls,uint64_t id){w->createObject(1,cls,id,"");};
    auto set=[&](uint64_t id,OmniPvdAttributeHandle attr,const void* data,uint32_t size){w->setAttribute(1,id,attr,reinterpret_cast<const uint8_t*>(data),size);};
    auto link=[&](uint64_t id,OmniPvdAttributeHandle attr,uint64_t target){set(id,attr,&target,8);};
    auto add=[&](uint64_t id,OmniPvdAttributeHandle attr,uint64_t target){w->addToUniqueListAttribute(1,id,attr,reinterpret_cast<const uint8_t*>(&target),8);};
    create(scene,10);
    const float identity[]={0,0,0,1,0,0,0};
    if(mode.compare(0,7,"surface")==0) {
        create(surface,20);add(10,actors,20);
        for(uint64_t index=0;index<2;++index) {
            create(shape,30+index);create(triGeom,40+index);create(mesh,50+index);
            add(20,surfaceShapes,30+index);link(30+index,geom,40+index);link(40+index,triangle,50+index);
            const float size=index?20.f:2.f;
            const float v[]={0,0,0,size,0,0,0,size,0,size,size,0};const uint32_t t[]={0,1,2,1,3,2};
            if(index || mode!="surface-empty-first"){set(50+index,verts,v,sizeof(v));set(50+index,tris,t,sizeof(t));}
        }
    } else if(mode!="empty") {
        create(body,20);create(shape,30);create(geometry,40);add(10,actors,20);add(20,shapes,30);link(30,geom,40);
        set(20,global,identity,sizeof(identity));set(30,local,identity,sizeof(identity));
    }
    for(uint64_t frame=1;frame<=3;++frame) {
        w->startFrame(1,frame);const float tolerance[]={float(frame==1?7:9),10};set(10,scale,tolerance,sizeof(tolerance));
        if(mode.compare(0,7,"surface")!=0 && mode!="empty") {
            const float h=mode=="zero"?0.f:mode=="overflow"?3e38f:frame==1?2.f:8.f;
            const float dimensions[]={h,h,h};set(40,half,dimensions,sizeof(dimensions));
            if(frame==3)w->destroyObject(1,20);
        }
        w->stopFrame(1,frame);
    }
    const bool closed=stream->closeStream();destroyOmniPvdWriter(*w);destroyOmniPvdFileWriteStream(*stream);return closed?0:4;
}

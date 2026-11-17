/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Real pinned writer supplies scene/physics ownership, scalar/compound schema,
// malformed payloads and animated values to the production loader.
#include "OmniPvdWriter.h"
#include "OmniPvdFileWriteStream.h"
#include "OmniPvdLibraryFunctions.h"
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
int main(int argc,char** argv) {
    if(argc!=3)return 2;
    const std::string mode=argv[2];
    auto* writer=createOmniPvdWriter();auto* stream=createOmniPvdFileWriteStream();
    if(!writer || !stream)return 3;
    stream->setFileName(argv[1]);if(!stream->openStream())return 4;writer->setWriteStream(*stream);
    auto& w=*writer;const auto real=OmniPvdDataType::eFLOAT32,ref=OmniPvdDataType::eOBJECT_HANDLE;
    const auto physics=w.registerClass("PxPhysics"),scene=w.registerClass("PxScene"),actor=w.registerClass("PxRigidDynamic"),shape=w.registerClass("PxShape"),sphere=w.registerClass("PxSphereGeometry");
    const auto scenes=w.registerUniqueListAttribute(physics,"scenes",ref),actors=w.registerUniqueListAttribute(scene,"actors",ref),shapes=w.registerUniqueListAttribute(actor,"shapes",ref);
    const auto physicsScale=w.registerAttribute(physics,"tolerancesScale",real,2);
    const auto sceneScale=w.registerAttribute(scene,"tolerancesScale",mode=="wrong-type"?OmniPvdDataType::eUINT32:real,
        mode=="variable"?0:2);
    const bool gravityCase=mode.compare(0,8,"gravity-")==0;
    const uint32_t gravityCount=mode=="gravity-variable"?0:mode=="gravity-pair"?2:mode=="gravity-four"?4:3;
    const auto gravity=gravityCase && mode!="gravity-missing" ? w.registerAttribute(scene,"gravity",
        mode=="gravity-wrong-type"?OmniPvdDataType::eUINT32:real,gravityCount) : 0;
    const auto pose=w.registerAttribute(actor,"globalPose",real,7),velocity=w.registerAttribute(actor,"linearVelocity",real,3),geom=w.registerAttribute(shape,"geometry",ref,1),radius=w.registerAttribute(sphere,"radius",real,1);
    auto create=[&](OmniPvdClassHandle cls,uint64_t id){w.createObject(1,cls,id,"");};
    auto add=[&](uint64_t id,OmniPvdAttributeHandle attr,uint64_t object){w.addToUniqueListAttribute(1,id,attr,reinterpret_cast<const uint8_t*>(&object),8);};
    auto set=[&](uint64_t id,OmniPvdAttributeHandle attr,const void* data,uint32_t size){w.setAttribute(1,id,attr,reinterpret_cast<const uint8_t*>(data),size);};
    const bool single=mode=="single" || mode=="dead";
    const bool withPhysics=mode=="scene" || mode=="owner" || mode=="ambiguous" || mode=="conflicting" || single;
    const bool linked=mode=="scene" || mode=="owner";
    if(withPhysics){create(physics,100);if(!single)create(physics,200);}
    for(uint64_t id:{10,20}) {
        create(scene,id);create(actor,id+1);create(shape,id+2);create(sphere,id+3);
        add(id,actors,id+1);add(id+1,shapes,id+2);uint64_t geometry=id+3;set(id+2,geom,&geometry,8);
        const float p[]={0,0,0,1,float(id),0,0},v[]={12,6,0},r=1;
        set(id+1,pose,p,sizeof(p));set(id+1,velocity,v,sizeof(v));set(id+3,radius,&r,4);
        if(linked)add(id==10?100:200,scenes,id);
        if(mode=="conflicting"){add(100,scenes,id);add(200,scenes,id);}
    }
    for(uint64_t frame=1;frame<=3;++frame) {
        w.startFrame(1,frame);
        if(gravity)for(uint64_t id:{10,20}) {
            float g[]={frame==1?0.f:-3.f,frame==1?-3.f:-4.f,frame==1?-4.f:0.f,0.f};
            if(id==20){g[0]=-4;g[1]=0;g[2]=-3;}
            if(mode=="gravity-z"){g[0]=0;g[1]=0;g[2]=-9.81f;}
            if(mode=="gravity-tiny"){g[0]=.0003f;g[1]=0;g[2]=-.0004f;}
            if(mode=="gravity-small"){g[0]=.0008f;g[1]=0;g[2]=-.0015f;}
            if(mode=="gravity-zero"){g[0]=g[1]=g[2]=0;}
            if(mode=="gravity-nan")g[0]=std::numeric_limits<float>::quiet_NaN();
            if(mode=="gravity-infinity")g[0]=std::numeric_limits<float>::infinity();
            const uint32_t bytes=mode=="gravity-short"?8:mode=="gravity-empty"?0:gravityCount?gravityCount*4:12;
            set(id,gravity,g,bytes);
        }
        if(withPhysics) {
            const float a[]={float(frame*50),float(frame*5)},b[]={float(frame*70),float(frame*7)};
            if(mode!="dead" || frame==1)set(100,physicsScale,a,8);
            if(!single)set(200,physicsScale,b,8);
            if(mode=="dead" && frame==2)w.destroyObject(1,100);
        }
        if(mode!="owner" && mode!="single" && mode!="ambiguous" && mode!="conflicting" && mode!="missing" && mode!="dead")for(uint64_t id:{10,20}) {
            float values[]={float(frame*(id==10?10:30)),float(frame*(id==10?2:3))};
            if(mode=="zero")values[1]=0;
            if(mode=="negative")values[1]=-1;
            if(mode=="nan")values[1]=std::numeric_limits<float>::quiet_NaN();
            if(mode=="infinity")values[1]=std::numeric_limits<float>::infinity();
            if(mode=="bad-length")values[0]=0;
            if(mode=="length-negative")values[0]=-1;
            if(mode=="length-nan")values[0]=std::numeric_limits<float>::quiet_NaN();
            if(mode=="length-infinity")values[0]=std::numeric_limits<float>::infinity();
            uint32_t size=mode=="short"?4:mode=="empty"?0:8;
            set(id,sceneScale,values,size);
        }
        if(frame==2)w.registerClass("PublicationMarker");
        w.stopFrame(1,frame);
    }
    const bool closed=stream->closeStream();destroyOmniPvdWriter(w);destroyOmniPvdFileWriteStream(*stream);return closed?0:5;
}

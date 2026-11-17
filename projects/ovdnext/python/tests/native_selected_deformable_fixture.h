/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Selected capture samples belong to actors; topology can be shared.
#pragma once
#include <limits>
#include <string>

static void writeSelectedDeformableSegment(OmniPvdWriter* w, const char* mode, bool second=false) {
    const auto object=OmniPvdDataType::eOBJECT_HANDLE, real=OmniPvdDataType::eFLOAT32;
    const auto metadata=w->registerClass("PxOmniPvdMetaData");
    const auto major=w->registerAttribute(metadata,"ovdIntegrationVersionMajor",OmniPvdDataType::eUINT32,1);
    const auto minor=w->registerAttribute(metadata,"ovdIntegrationVersionMinor",OmniPvdDataType::eUINT32,1);
    const auto scene=w->registerClass("PxScene"), actor=w->registerClass("PxActor");
    const auto actors=w->registerUniqueListAttribute(scene,"actors",object);
    const auto shapes=w->registerUniqueListAttribute(actor,"shapes",object);
    const auto surface=w->registerClass("PxDeformableSurface",actor);
    const auto surfacePos=w->registerAttribute(surface,"positions",real,0);
    const auto surfaceVel=w->registerAttribute(surface,"velocities",real,0);
    const auto volume=w->registerClass("PxDeformableVolume",actor);
    const auto collision=w->registerAttribute(volume,"collisionMesh",object,1);
    const auto simulation=w->registerAttribute(volume,"simulationMesh",object,1);
    const auto simShapes=w->registerUniqueListAttribute(volume,"simulationMeshShapes",object);
    const auto collisionPos=w->registerAttribute(volume,"collisionPositions",real,0);
    const auto simulationPos=w->registerAttribute(volume,"simulationPositions",real,0);
    const auto volumeVel=w->registerAttribute(volume,"velocities",real,0);
    const auto tet=w->registerClass("PxTetrahedronMesh"), tri=w->registerClass("PxTriangleMesh");
    const auto tetVerts=w->registerAttribute(tet,"verts",real,0), triVerts=w->registerAttribute(tri,"verts",real,0);
    const auto tetPos=w->registerAttribute(tet,"positions",real,0), triPos=w->registerAttribute(tri,"positions",real,0);
    const auto tetVel=w->registerAttribute(tet,"velocities",real,0), triVel=w->registerAttribute(tri,"velocities",real,0);
    const auto tets=w->registerAttribute(tet,"tets",OmniPvdDataType::eUINT32,0);
    const auto tris=w->registerAttribute(tri,"tris",OmniPvdDataType::eUINT32,0);
    const auto shape=w->registerClass("PxShape"), geom=w->registerClass("PxTriangleMeshGeometry");
    const auto geometry=w->registerAttribute(shape,"geometry",object,1), triangle=w->registerAttribute(geom,"triangleMesh",object,1);
    auto create=[&](OmniPvdClassHandle c,uint64_t id){w->createObject(1,c,id,"");};
    auto ref=[&](uint64_t id,OmniPvdAttributeHandle a,uint64_t value){w->setAttribute(1,id,a,reinterpret_cast<const uint8_t*>(&value),8);};
    auto add=[&](uint64_t id,OmniPvdAttributeHandle a,uint64_t value){w->addToUniqueListAttribute(1,id,a,reinterpret_cast<const uint8_t*>(&value),8);};
    auto floats=[&](uint64_t id,OmniPvdAttributeHandle a,const std::vector<float>& v){w->setAttribute(1,id,a,reinterpret_cast<const uint8_t*>(v.data()),uint32_t(v.size()*4));};
    create(metadata,1000);
    const uint32_t versionMajor=3,versionMinor=5;
    w->setAttribute(1,1000,major,reinterpret_cast<const uint8_t*>(&versionMajor),4);
    w->setAttribute(1,1000,minor,reinterpret_cast<const uint8_t*>(&versionMinor),4);
    create(scene,1);create(volume,10);create(surface,20);create(surface,21);create(tet,101);create(tri,103);
    for(uint64_t id:{201,202,204,205})create(shape,id);
    create(geom,203);ref(202,geometry,203);ref(204,geometry,203);ref(203,triangle,103);
    ref(10,collision,101);ref(10,simulation,101);add(10,shapes,201);add(10,simShapes,205);
    add(20,shapes,202);add(21,shapes,204);
    for(uint64_t id:{10,20,21})add(1,actors,id);
    const std::vector<float> tv=second?std::vector<float>{0,0,0, 1,0,0, 0,1,0, 0,0,1, 0,0,-1}:
        std::vector<float>{0,0,0, 1,0,0, 0,1,0, 0,0,1};
    const std::vector<float> sv=second?std::vector<float>{0,0,0, 2,0,0, 0,2,0, 2,2,0}:
        std::vector<float>{0,0,0, 1,0,0, 0,1,0};
    floats(101,tetVerts,tv);floats(103,triVerts,sv);
    std::vector<uint32_t> ti=second?std::vector<uint32_t>{0,1,2,3, 0,2,1,4}:std::vector<uint32_t>{0,1,2,3};
    std::vector<uint32_t> si=second?std::vector<uint32_t>{0,1,2, 1,3,2}:std::vector<uint32_t>{0,1,2};
    if(std::strcmp(mode,"triangle-index")==0)si.back()=99;
    if(std::strcmp(mode,"tet-index")==0)ti.back()=99;
    if(std::strcmp(mode,"triangle-stride")==0)si.pop_back();
    if(std::strcmp(mode,"tet-stride")==0)ti.pop_back();
    w->setAttribute(1,101,tets,reinterpret_cast<const uint8_t*>(ti.data()),uint32_t(ti.size()*4));
    w->setAttribute(1,103,tris,reinterpret_cast<const uint8_t*>(si.data()),uint32_t(si.size()*4));
    auto translated=[](const std::vector<float>& xyz,float x,float y){
        std::vector<float> result;
        for(size_t k=0;k<xyz.size();k+=3)result.insert(result.end(),{xyz[k]+x,xyz[k+1]+y,xyz[k+2],1});
        return result;
    };
    auto velocity=[](size_t count,float y){
        std::vector<float> values;
        for(size_t k=0;k<count;++k)values.insert(values.end(),{0,y,0,0});
        return values;
    };
    const std::string requested(mode);
    auto sampled=[&](uint64_t id,OmniPvdAttributeHandle attribute,const char* target,std::vector<float> values,uint64_t frame){
        const std::string prefix=std::string(target)+"-";
        if(frame==2 && requested.compare(0,prefix.size(),prefix)==0) {
            const auto fault=requested.substr(prefix.size());
            if(fault=="short")values.resize(values.size()-4);
            if(fault=="extra")values.insert(values.end(),{11,12,13,1});
            if(fault=="stride")values.pop_back();
            if(fault=="nan")values[0]=std::numeric_limits<float>::quiet_NaN();
            if(fault=="inf")values[0]=std::numeric_limits<float>::infinity();
            if(fault=="w-nan")for(size_t k=3;k<values.size();k+=4)values[k]=std::numeric_limits<float>::quiet_NaN();
            if(fault=="byte") {
                w->setAttribute(1,id,attribute,reinterpret_cast<const uint8_t*>(values.data()),uint32_t(values.size()*4-1));
                return;
            }
        }
        floats(id,attribute,values);
    };
    const bool legacy=std::strcmp(mode,"legacy")==0 || std::strcmp(mode,"empty")==0;
    for(uint64_t frame=1;frame<=3;++frame) {
        w->startFrame(1,frame);
        // Deliberately different legacy data catches accidental domain mixing.
        floats(101,tetPos,translated(tv,20+float(frame),0));
        floats(103,triPos,translated(sv,30+float(frame),0));
        floats(101,tetVel,{9,0,0,0, 9,0,0,0, 9,0,0,0, 9,0,0,0});
        floats(103,triVel,{9,0,0,0, 9,0,0,0, 9,0,0,0});
        if(!legacy) {
            const float dx=3*float(frame-1);
            const float offset=second?10.f:0.f;
            sampled(10,collisionPos,"collision",translated(tv,-6+dx+offset,-3),frame);
            sampled(10,simulationPos,"simulation",translated(tv,2+dx+offset,-3),frame);
            sampled(10,volumeVel,"volume-velocity",velocity(tv.size()/3,2),frame);
            auto positions=translated(sv,-6+dx+offset,2);
            if(std::strcmp(mode,"malformed")==0)positions.pop_back();
            if(std::strcmp(mode,"nonfinite")==0)positions[0]=std::numeric_limits<float>::infinity();
            sampled(20,surfacePos,"surface",positions,frame);
            floats(21,surfacePos,translated(sv,2-dx+offset,2));
            if(std::strcmp(mode,"missing-velocity")!=0)
                sampled(20,surfaceVel,"surface-velocity",std::strcmp(mode,"mismatch")==0?std::vector<float>{0,4,0,0}:velocity(sv.size()/3,4),frame);
            floats(21,surfaceVel,velocity(sv.size()/3,-3));
        } else if(std::strcmp(mode,"empty")==0) {
            floats(10,collisionPos,{});floats(10,simulationPos,{});floats(10,volumeVel,{});
            for(uint64_t id:{20,21}){floats(id,surfacePos,{});floats(id,surfaceVel,{});}
        }
        w->stopFrame(1,frame);
    }
}

static bool writeSelectedDeformableFixture(const char* path, const char* mode) {
    auto* w=createOmniPvdWriter(); auto* stream=createOmniPvdFileWriteStream();
    stream->setFileName(path);
    if(!stream->openStream()) return false;
    w->setWriteStream(*stream);
    writeSelectedDeformableSegment(w,mode);
    if(std::strcmp(mode,"segments")==0)writeSelectedDeformableSegment(w,mode,true);
    const bool ok=stream->closeStream();destroyOmniPvdWriter(*w);destroyOmniPvdFileWriteStream(*stream);
    return ok;
}

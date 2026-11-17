/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Integration fixture produced with the pinned Physics writer, never a second
// Python implementation of the OVD wire format. Shared tet faces and animated
// vertices expose topology/ownership bugs that the committed rigid captures do not.
#include "OmniPvdWriter.h"
#include "OmniPvdFileWriteStream.h"
#include "OmniPvdLibraryFunctions.h"
#include "PxPvdUserRenderer.h"
#include "PvdSimpleAllocator.h"
#include "PxFoundation.h"
#include "PxFoundationVersion.h"
#include "PxErrorCallback.h"
#include "PVDClient.h"
#include "PvdCommStreamMetaData.h"
#include "PvdObjectModelEventStream.h"
#include "PxPvdObjectModel.h"
#include "PvdCheckedOutput.h"
#include "PxTransform.h"
#include "PxBounds3.h"
#include <cstring>
#include <cstdint>
#include <vector>
#include "native_selected_deformable_fixture.h"

template <typename T>
static void setJointInspectorAttribute(OmniPvdWriter& writer, OmniPvdObjectHandle object,
                                       OmniPvdAttributeHandle attribute, const T& value) {
    writer.setAttribute(1, object, attribute,
        reinterpret_cast<const uint8_t*>(&value), sizeof(value));
}

static void setJointInspectorName(OmniPvdWriter& writer, OmniPvdObjectHandle object,
                                  OmniPvdAttributeHandle attribute, const char* value) {
    writer.setAttribute(1, object, attribute,
        reinterpret_cast<const uint8_t*>(value), static_cast<uint32_t>(std::strlen(value) + 1));
}

static void addJointInspectorReference(OmniPvdWriter& writer, OmniPvdObjectHandle object,
                                       OmniPvdAttributeHandle attribute, OmniPvdObjectHandle target) {
    writer.addToUniqueListAttribute(1, object, attribute,
        reinterpret_cast<const uint8_t*>(&target), sizeof(target));
}

static void writeJointInspectorSegment(OmniPvdWriter& writer, bool secondSegment,
                                       bool finiteLifetime, bool recreateScene,
                                       uint32_t integrationMajor = 3, bool membershipFirst = false,
                                       bool missingEndpoints = false, bool coreOnly = false,
                                       bool unknownEndpoint = false) {
    const bool core = integrationMajor >= 4;
    const OmniPvdObjectHandle metadata = 0xA000, physics = 0xA001, scene = 0xA002;
    const OmniPvdObjectHandle actor = 0xA003, constraint = 0xA006, joint = 0xA007, world = 0;
    const auto metadataClass = writer.registerClass("PxOmniPvdMetaData");
    const auto major = writer.registerAttribute(metadataClass, "ovdIntegrationVersionMajor", OmniPvdDataType::eUINT32, 1);
    const auto minor = writer.registerAttribute(metadataClass, "ovdIntegrationVersionMinor", OmniPvdDataType::eUINT32, 1);
    const auto physicsClass = writer.registerClass("PxPhysics");
    const auto scenes = writer.registerUniqueListAttribute(physicsClass, "scenes", OmniPvdDataType::eOBJECT_HANDLE);
    const auto sceneClass = writer.registerClass("PxScene");
    const auto sceneName = writer.registerAttribute(sceneClass, "name", OmniPvdDataType::eSTRING, 0);
    const auto actors = writer.registerUniqueListAttribute(sceneClass, "actors", OmniPvdDataType::eOBJECT_HANDLE);
    const auto constraints = writer.registerUniqueListAttribute(sceneClass, "constraints", OmniPvdDataType::eOBJECT_HANDLE);
    const auto actorClass = writer.registerClass("PxRigidDynamic");
    const auto actorName = writer.registerAttribute(actorClass, "name", OmniPvdDataType::eSTRING, 0);
    const auto constraintClass = writer.registerClass("PxConstraint");
    const auto jointClass = writer.registerClass("PxJoint");
    const auto jointName = writer.registerAttribute(jointClass, "name", OmniPvdDataType::eSTRING, 0);
    const auto jointConstraint = writer.registerAttribute(jointClass, "constraint", OmniPvdDataType::eOBJECT_HANDLE, 1);
    const auto commonClass = core ? constraintClass : jointClass;
    const auto actor0 = writer.registerAttribute(commonClass, "actor0", OmniPvdDataType::eOBJECT_HANDLE, 1);
    const auto actor1 = writer.registerAttribute(commonClass, "actor1", OmniPvdDataType::eOBJECT_HANDLE, 1);
    const auto breakForce = writer.registerAttribute(commonClass, "breakForce", OmniPvdDataType::eFLOAT32, 1);
    const auto breakTorque = writer.registerAttribute(commonClass, "breakTorque", OmniPvdDataType::eFLOAT32, 1);
    const auto flagClass = writer.registerClass("PxConstraintFlag");
    writer.registerEnumValue(flagClass, "eBROKEN", 1);
    const auto flags = writer.registerFlagsAttribute(commonClass, "constraintFlags", flagClass);
    const auto pose = writer.registerAttribute(actorClass, "globalPose", OmniPvdDataType::eFLOAT32, 7);
    const auto local0 = writer.registerAttribute(jointClass, "actor0LocalPose", OmniPvdDataType::eFLOAT32, 7);
    const auto local1 = writer.registerAttribute(jointClass, "actor1LocalPose", OmniPvdDataType::eFLOAT32, 7);
    const auto d6Class = writer.registerClass("PxD6Joint", jointClass);
    const auto motion = writer.registerAttribute(d6Class, "motions", OmniPvdDataType::eUINT32, 6);

    writer.createObject(1, metadataClass, metadata, "");
    setJointInspectorAttribute(writer, metadata, major, integrationMajor);
    setJointInspectorAttribute(writer, metadata, minor, uint32_t(core ? 0 : 2));
    writer.createObject(1, physicsClass, physics, "");
    if (recreateScene) {
        writer.createObject(1, sceneClass, scene, "");
        setJointInspectorName(writer, scene, sceneName, "Setup scene");
        addJointInspectorReference(writer, physics, scenes, scene);
        writer.startFrame(scene, 1); writer.stopFrame(scene, 1);
        writer.removeFromUniqueListAttribute(1, physics, scenes,
            reinterpret_cast<const uint8_t*>(&scene), sizeof(scene));
        writer.destroyObject(1, scene);
    }
    writer.createObject(1, sceneClass, scene, "");
    setJointInspectorName(writer, scene, sceneName, secondSegment ? "Second scene" : "First scene");
    addJointInspectorReference(writer, physics, scenes, scene);
    writer.createObject(1, actorClass, actor, "");
    setJointInspectorName(writer, actor, actorName, secondSegment ? "Second box" : "First box");
    addJointInspectorReference(writer, scene, actors, actor);
    const float actorPose[] = {0, 0, 0, 1, secondSegment ? 40.f : 10.f, 20, 30};
    setJointInspectorAttribute(writer, actor, pose, actorPose);
    if (finiteLifetime) {
        for (uint64_t frame = 1; frame <= 2; ++frame) {
            writer.startFrame(scene, frame); writer.stopFrame(scene, frame);
        }
        writer.startFrame(scene, 3);
    }
    if (!membershipFirst) writer.createObject(1, constraintClass, constraint, "");
    addJointInspectorReference(writer, scene, constraints, constraint);
    if (membershipFirst) writer.createObject(1, constraintClass, constraint, "");
    writer.createObject(1, d6Class, joint, "");
    setJointInspectorName(writer, joint, jointName, secondSegment ? "Second world joint" : "First world joint");
    setJointInspectorAttribute(writer, joint, jointConstraint, constraint);
    const auto common = core ? constraint : joint;
    if (!missingEndpoints) {
        setJointInspectorAttribute(writer, common, actor0, world);
        setJointInspectorAttribute(writer, common, actor1, unknownEndpoint ? uint64_t(0xDEAD) : actor);
    }
    setJointInspectorAttribute(writer, common, breakForce, secondSegment ? 250.0f : 125.0f);
    setJointInspectorAttribute(writer, common, breakTorque, 75.f);
    setJointInspectorAttribute(writer, common, flags, uint16_t(1));
    const float parentPose[] = {0, 0, 0, 1, 1, 2, 3}, childPose[] = {0, 0, 0, 1, 4, 5, 6};
    setJointInspectorAttribute(writer, joint, local0, parentPose);
    setJointInspectorAttribute(writer, joint, local1, childPose);
    const uint32_t motions[] = {0, 1, 2, 0, 1, 2};
    setJointInspectorAttribute(writer, joint, motion, motions);
    if (coreOnly) writer.destroyObject(1, joint);
    if (finiteLifetime) {
        writer.stopFrame(scene, 3);
        writer.startFrame(scene, 4); writer.stopFrame(scene, 4);
        writer.startFrame(scene, 5);
        writer.removeFromUniqueListAttribute(1, scene, constraints,
            reinterpret_cast<const uint8_t*>(&constraint), sizeof(constraint));
        writer.destroyObject(1, joint); writer.destroyObject(1, constraint);
        writer.stopFrame(scene, 5);
        writer.startFrame(scene, 6); writer.stopFrame(scene, 6);
    } else {
        writer.startFrame(scene, 1); writer.stopFrame(scene, 1);
        writer.startFrame(scene, 2); writer.stopFrame(scene, 2);
    }
}

static bool writeJointInspectorFixture(const char* path, const char* mode) {
    auto* writer = createOmniPvdWriter();
    auto* stream = createOmniPvdFileWriteStream();
    if (!writer || !stream) return false;
    stream->setFileName(path);
    if (!stream->openStream()) return false;
    writer->setWriteStream(*stream);
    const bool segments = std::strstr(mode, "-segments") != nullptr;
    const uint32_t integrationMajor = std::strstr(mode, "-v5") ? 5 : std::strstr(mode, "-v4") ? 4 : 3;
    writeJointInspectorSegment(*writer, false,
        std::strstr(mode, "-lifetime") != nullptr,
        std::strstr(mode, "-recreated") != nullptr, integrationMajor,
        std::strstr(mode, "-membership-first") != nullptr,
        std::strstr(mode, "-missing-endpoints") != nullptr,
        std::strstr(mode, "-core-only") != nullptr,
        std::strstr(mode, "-unknown-endpoint") != nullptr);
    if (segments) writeJointInspectorSegment(*writer, true, false, false, integrationMajor);
    const bool closed = stream->closeStream();
    destroyOmniPvdWriter(*writer);
    destroyOmniPvdFileWriteStream(*stream);
    return closed;
}

static std::vector<uint8_t> jointEvents(uint32_t frame) {
    using namespace physx;using namespace physx::pvdsdk;
    struct Bytes:RendererEventClient {
        std::vector<uint8_t> data;
        void handleBufferFlush(const uint8_t* bytes,uint32_t size) override {data.insert(data.end(),bytes,bytes+size);}
    } bytes;
    auto* renderer=PvdUserRenderer::create();renderer->setClient(&bytes);renderer->setInstanceId(reinterpret_cast<void*>(200));
    const PxTransform parent(PxVec3(float(frame),4,5)),child(PxVec3(float(frame)+1,6,7));
    renderer->visualizeJointFrames(parent,child);
    renderer->visualizeLinearLimit(PxTransform(PxIdentity),PxTransform(PxIdentity),2,true);
    renderer->visualizeAngularLimit(PxTransform(PxIdentity),-.25f,.75f,false);
    renderer->visualizeLimitCone(PxTransform(PxIdentity),.5f,.25f,true);
    renderer->visualizeDoubleCone(PxTransform(PxIdentity),.5f,false);
    const PvdDebugLine line(PxVec3(7,8,9),PxVec3(8,9,10),0xff123456);
    renderer->drawLines(&line,1);renderer->flushRenderEvents();renderer->release();return bytes.data;
}
static bool writeEvidenceFixture(const char* path,const std::string& mode) {
    auto* writer=createOmniPvdWriter();auto* stream=createOmniPvdFileWriteStream();
    if(!writer || !stream)return false;
    stream->setFileName(path);if(!stream->openStream())return false;writer->setWriteStream(*stream);
    auto& w=*writer;
    const uint64_t context=9007199254740993ULL,actor=9007199254741003ULL,physics=100,metadata=200;
    const auto metaClass=w.registerClass("PxOmniPvdMetaData");
    const auto major=w.registerAttribute(metaClass,"ovdIntegrationVersionMajor",OmniPvdDataType::eUINT32,1);
    const auto minor=w.registerAttribute(metaClass,"ovdIntegrationVersionMinor",OmniPvdDataType::eUINT32,1);
    const auto physicsClass=w.registerClass("PxPhysics"),sceneClass=w.registerClass("PxScene"),actorClass=w.registerClass("PxRigidDynamic");
    const auto scenes=w.registerUniqueListAttribute(physicsClass,"scenes",OmniPvdDataType::eOBJECT_HANDLE);
    const auto actors=w.registerUniqueListAttribute(sceneClass,"actors",OmniPvdDataType::eOBJECT_HANDLE);
    const auto velocity=w.registerAttribute(actorClass,"linearVelocity",OmniPvdDataType::eFLOAT32,3);
    const auto angular=w.registerAttribute(actorClass,"angularVelocity",OmniPvdDataType::eFLOAT32,3);
    const bool energy=mode=="energy";
    const auto mass=energy?w.registerAttribute(actorClass,"mass",OmniPvdDataType::eFLOAT32,1):0;
    const auto inertia=energy?w.registerAttribute(actorClass,"massSpaceInertiaTensor",OmniPvdDataType::eFLOAT32,3):0;
    const auto com=energy?w.registerAttribute(actorClass,"cMassLocalPose",OmniPvdDataType::eFLOAT32,7):0;
    const auto pose=energy?w.registerAttribute(actorClass,"globalPose",OmniPvdDataType::eFLOAT32,7):0;
    const auto gravity=energy?w.registerAttribute(sceneClass,"gravity",OmniPvdDataType::eFLOAT32,3):0;
    w.createObject(1,metaClass,metadata,"");
    setJointInspectorAttribute(w,metadata,major,uint32_t(3));setJointInspectorAttribute(w,metadata,minor,uint32_t(1));
    w.createObject(1,physicsClass,physics,"");
    auto createScene=[&] {
        w.createObject(1,sceneClass,context,"");addJointInspectorReference(w,physics,scenes,context);
        w.createObject(1,actorClass,actor,"");addJointInspectorReference(w,context,actors,actor);
    };
    createScene();
    if(energy) {
        const float tensor[]={1,2,3},local[]={0,0,0,1,0,0,1},g[]={0,0,-10};
        setJointInspectorAttribute(w,actor,mass,2.f);setJointInspectorAttribute(w,actor,inertia,tensor);
        setJointInspectorAttribute(w,actor,com,local);setJointInspectorAttribute(w,context,gravity,g);
    }
    if(mode=="budget") {
        // Repeated residency creates real metadata without exceeding the pinned
        // reader's 2047-byte string limit or changing any production limits.
        for(uint64_t id=1000;id<1512;++id) {
            w.createObject(1,actorClass,id,"");addJointInspectorReference(w,context,actors,id);
        }
        for(uint64_t frame=1;frame<=1050;++frame) {
            w.startFrame(context,2*frame-1);
            for(uint64_t id=1000;id<1512;++id)
                w.removeFromUniqueListAttribute(1,context,actors,reinterpret_cast<const uint8_t*>(&id),sizeof(id));
            w.stopFrame(context,2*frame-1);w.startFrame(context,2*frame);
            for(uint64_t id=1000;id<1512;++id)addJointInspectorReference(w,context,actors,id);
            w.stopFrame(context,2*frame);
        }
        const bool closed=stream->closeStream();destroyOmniPvdWriter(w);destroyOmniPvdFileWriteStream(*stream);return closed;
    }
    const uint64_t frameContext=mode=="unmapped"?1:context;
    const std::vector<uint64_t> frames=mode=="gap"?std::vector<uint64_t>{1,3}:
        mode=="reset"?std::vector<uint64_t>{1,2,1}:mode=="min"?std::vector<uint64_t>{3,4}:std::vector<uint64_t>{1,2};
    if(mode=="overflow") {
        for(uint64_t index=0;index<4100;++index){w.startFrame(context+index,1);w.stopFrame(context+index,1);}
    } else for(size_t index=0;index<frames.size();++index) {
        const auto frame=frames[index];w.startFrame(frameContext,frame);
        if(energy) {
            const float v[]={2,0,0},omega[]={0,0,1},world[]={0,0,0,1,1,2,3};
            setJointInspectorAttribute(w,actor,velocity,v);setJointInspectorAttribute(w,actor,angular,omega);setJointInspectorAttribute(w,actor,pose,world);
        } else if(index==0){const float value[]={1,2,3};w.setAttribute(1,actor,velocity,reinterpret_cast<const uint8_t*>(value),sizeof(value));}
        w.stopFrame(frameContext,frame);
        if(index==0 && mode=="recreated") {
            w.removeFromUniqueListAttribute(1,physics,scenes,reinterpret_cast<const uint8_t*>(&context),sizeof(context));
            w.destroyObject(1,actor);w.destroyObject(1,context);createScene();
        }
    }
    const bool closed=stream->closeStream();destroyOmniPvdWriter(w);destroyOmniPvdFileWriteStream(*stream);return closed;
}

int main(int argc,char** argv) {
    if(argc!=2 && argc!=3) return 2;
    struct Error:physx::PxErrorCallback { void reportError(physx::PxErrorCode::Enum,const char*,const char*,int) override {} } error;
    pvdapp::CheckingAllocator allocator;
    auto* foundation=PxCreateFoundation(PX_FOUNDATION_VERSION,allocator,error);
    if(!foundation) return 6;
    if(argc==3 && std::strncmp(argv[2],"--selected-deformable-",22)==0) {
        const bool written=writeSelectedDeformableFixture(argv[1],argv[2]+22);
        foundation->release();return written?0:5;
    }
    if(argc==3 && std::strncmp(argv[2],"--evidence-",11)==0) {
        const bool written=writeEvidenceFixture(argv[1],std::string(argv[2]+11));
        foundation->release();return written?0:5;
    }
    if(argc==3 && std::strncmp(argv[2],"--joint-inspector",17)==0) {
        const bool written=writeJointInspectorFixture(argv[1],argv[2]);
        foundation->release();return written?0:5;
    }
    if(argc==3 && (std::strcmp(argv[2],"--legacy-shift")==0 || std::strcmp(argv[2],"--legacy-joints")==0)) {
        const bool joints=std::strcmp(argv[2],"--legacy-joints")==0;
        using namespace pvdapp;using namespace physx;using namespace physx::pvdsdk;
        const std::string backingPath=std::string(argv[1])+".backing";
        auto stream=PvdRandomAccessIOStream::open(allocator,backingPath.c_str(),true,true).getValue();stream->addRef();
        auto& mutator=PvdClientMutator::create(allocator,*stream,false);
        auto& client=mutator.getClient();auto& metadata=client.getMetaData();
        auto& model=mutator.getObjectModelMutator();auto& frames=mutator.getCommStreamMetaDataMutator();
        const auto sceneClass=metadata.getOrCreateClass(NamespacedName("physx3","PxScene"));
        const auto actorClass=metadata.getOrCreateClass(NamespacedName("physx3","PxRigidDynamic"));
        const auto dynamics=metadata.createProperty(sceneClass.mClassId,"RigidDynamics","children",getPvdTypeForType<ObjectRef>(),PropertyType::Array).getValue();
        metadata.createProperty(actorClass.mClassId,"GlobalPose","",getPvdTypeForType<PxTransform>(),PropertyType::Scalar);
        metadata.createProperty(actorClass.mClassId,"WorldBounds","",getPvdTypeForType<PxBounds3>(),PropertyType::Scalar);
        const auto scene=model.createInstance(sceneClass.mClassId,1),actor=model.createInstance(actorClass.mClassId,2);
        frames.setInstanceId(100,1);frames.setInstanceId(200,2);
        const int32_t ref=2;model.setPropertyValue(scene.mInstPtr,dynamics.mPropertyId,reinterpret_cast<const uint8_t*>(&ref),4,getPvdTypeForType<ObjectRef>());
        model.setPropertyValue(actor.mInstPtr,"GlobalPose",PxTransform(PxVec3(100,200,300)));
        model.setPropertyValue(actor.mInstPtr,"WorldBounds",PxBounds3(PxVec3(99,199,299),PxVec3(101,201,301)));
        InstanceDescription rendererInstance{};int32_t eventsProperty=-1;
        if(joints) {
            const auto rendererClass=metadata.getOrCreateClass(NamespacedName("physx3","PvdUserRenderer"));
            eventsProperty=metadata.createProperty(rendererClass.mClassId,"events","",getPvdTypeForType<PxU8>(),PropertyType::Array).getValue().mPropertyId;
            metadata.createProperty(rendererClass.mClassId,"Scene","",getPvdTypeForType<ObjectRef>(),PropertyType::Scalar);
            rendererInstance=model.createInstance(rendererClass.mClassId,3);frames.setInstanceId(300,3);
            const int32_t sceneId=1;
            const auto sceneProperty=metadata.findProperty(rendererClass.mClassId,"Scene").getValue().mPropertyId;
            model.setPropertyValue(rendererInstance.mInstPtr,sceneProperty,reinterpret_cast<const uint8_t*>(&sceneId),4,getPvdTypeForType<ObjectRef>());
        }
        const auto marker=metadata.getStringTable().strToHandle("frame");
        for(uint32_t frame=1;frame<=3;++frame) {
            frames.addSectionEvent(100,marker,true,frame*100);
            if(joints && frame<3) {
                const auto bytes=jointEvents(frame);
                model.setPropertyValue(rendererInstance.mInstPtr,eventsProperty,bytes.data(),uint32_t(bytes.size()),getPvdTypeForType<PxU8>());
            } else if(!joints && frame>1)model.originShift(scene.mInstPtr,PxVec3(10,20,30));
            frames.addSectionEvent(100,marker,false,frame*100+50);
        }
        mutator.getEventStreamFactory()->flush();
        {ovdnext::OutputFile output(argv[1],false);client.save(output);output.commit();}
        mutator.release();stream->release();foundation->release();return 0;
    }
    auto w=createOmniPvdWriter(); auto stream=createOmniPvdFileWriteStream();
    if(!w || !stream) return 3;
    stream->setFileName(argv[1]); if(!stream->openStream()) return 4; w->setWriteStream(*stream);
    const auto object=OmniPvdDataType::eOBJECT_HANDLE, real=OmniPvdDataType::eFLOAT32, integer=OmniPvdDataType::eUINT32;
    const bool messages=argc==3 && std::strcmp(argv[2],"--messages")==0;
    if(argc==3 && std::strcmp(argv[2],"--joint-live-tail")!=0 && !messages) {
        const auto cls=w->registerClass("PxScene"), sample=w->registerClass("HierarchySample");
        const auto list=w->registerUniqueListAttribute(cls,"actors",object);
        const auto name=w->registerAttribute(sample,"name",OmniPvdDataType::eSTRING,0);
        const auto ref=w->registerAttribute(sample,"reference",object,1), scalar=w->registerAttribute(sample,"mass",real,1);
        w->createObject(1,cls,1,"");w->createObject(1,sample,2,"");w->createObject(1,sample,3,"");w->createObject(1,sample,4,"");
        for(uint64_t id:{2,3,4})w->addToUniqueListAttribute(1,1,list,reinterpret_cast<uint8_t*>(&id),8);
        for(uint64_t frame=1;frame<=4;++frame) {
            w->startFrame(1,frame);
            const char* label=frame<3?"Alpha":"Beta";
            w->setAttribute(1,2,name,reinterpret_cast<const uint8_t*>(label),frame<3?6:5);
            const uint64_t target=frame<4?3:4;
            if(frame==1 || frame==4)w->setAttribute(1,2,ref,reinterpret_cast<const uint8_t*>(&target),8);
            const float mass=float(frame);w->setAttribute(1,2,scalar,reinterpret_cast<const uint8_t*>(&mass),4);
            w->stopFrame(1,frame);
        }
        stream->closeStream();destroyOmniPvdWriter(*w);destroyOmniPvdFileWriteStream(*stream);foundation->release();return 0;
    }
    auto scene=w->registerClass("PxScene"); auto actors=w->registerUniqueListAttribute(scene,"actors",object);
    auto contactPoints=w->registerAttribute(scene,"pairsContactPoints",real,0), contactNormals=w->registerAttribute(scene,"pairsContactNormals",real,0);
    auto contactActors=w->registerAttribute(scene,"pairsActors",object,0), contactCounts=w->registerAttribute(scene,"pairsContactCounts",integer,0);
    auto debugClass=w->registerClass("PvdImmediateRenderer");auto debugEvents=w->registerAttribute(debugClass,"events",OmniPvdDataType::eUINT8,0);
    auto actor=w->registerClass("PxActor"); auto shapes=w->registerUniqueListAttribute(actor,"shapes",object);
    auto volume=w->registerClass("PxDeformableVolume",actor);
    auto collision=w->registerAttribute(volume,"collisionMesh",object,1), simulation=w->registerAttribute(volume,"simulationMesh",object,1);
    auto simHandles=w->registerUniqueListAttribute(volume,"simulationMeshShapes",object);
    auto surface=w->registerClass("PxDeformableSurface",actor);
    auto particles=w->registerClass("PxPBDParticleSystem",actor);
    auto buffers=w->registerUniqueListAttribute(particles,"particleBuffers",object), radius=w->registerAttribute(particles,"fluidRestOffset",real,1);
    auto particleBuffer=w->registerClass("PxParticleBuffer"); auto particlePos=w->registerAttribute(particleBuffer,"positionInvMasses",real,0);
    auto tet=w->registerClass("PxTetrahedronMesh");
    auto tetVerts=w->registerAttribute(tet,"verts",real,0), tetPos=w->registerAttribute(tet,"positions",real,0), tetVelocity=w->registerAttribute(tet,"velocities",real,0), tets=w->registerAttribute(tet,"tets",integer,0);
    auto triangle=w->registerClass("PxTriangleMesh");
    auto triVerts=w->registerAttribute(triangle,"verts",real,0), triPos=w->registerAttribute(triangle,"positions",real,0), tris=w->registerAttribute(triangle,"tris",integer,0);
    auto analytics=w->registerClass("AnalysisSample");
    auto scalar=w->registerAttribute(analytics,"scalar",real,1), variable=w->registerAttribute(analytics,"variable",real,0);
    auto pair=w->registerAttribute(analytics,"pair",integer,2);
    auto shape=w->registerClass("PxShape"); auto geometry=w->registerAttribute(shape,"geometry",object,1);
    auto triGeom=w->registerClass("PxTriangleMeshGeometry"); auto triRef=w->registerAttribute(triGeom,"triangleMesh",object,1);
    auto create=[&](OmniPvdClassHandle c,uint64_t id){w->createObject(1,c,id,"");};
    auto setRef=[&](uint64_t id,OmniPvdAttributeHandle p,uint64_t ref){w->setAttribute(1,id,p,reinterpret_cast<uint8_t*>(&ref),8);};
    auto add=[&](uint64_t id,OmniPvdAttributeHandle p,uint64_t ref){w->addToUniqueListAttribute(1,id,p,reinterpret_cast<uint8_t*>(&ref),8);};
    auto floats=[&](uint64_t id,OmniPvdAttributeHandle p,const std::vector<float>& v){w->setAttribute(1,id,p,reinterpret_cast<const uint8_t*>(v.data()),v.size()*4);};
    auto ints=[&](uint64_t id,OmniPvdAttributeHandle p,const std::vector<uint32_t>& v){w->setAttribute(1,id,p,reinterpret_cast<const uint8_t*>(v.data()),v.size()*4);};
    create(scene,1);create(volume,10);create(surface,20);create(particles,30);create(analytics,110);create(debugClass,120);
    create(tet,101);create(tet,102);create(triangle,103);create(particleBuffer,104);
    create(shape,201);create(shape,202);create(triGeom,203);
    setRef(10,collision,101);setRef(10,simulation,102);add(10,shapes,201);add(10,simHandles,102);
    add(20,shapes,202);setRef(202,geometry,203);setRef(203,triRef,103);
    add(30,buffers,104);floats(30,radius,{0.2f});
    for(uint64_t id:{10,20,30,110,120}) add(1,actors,id);
    struct DebugBytes:physx::pvdsdk::RendererEventClient {
        std::vector<uint8_t> bytes;
        void handleBufferFlush(const uint8_t* data,uint32_t size) override {bytes.insert(bytes.end(),data,data+size);}
    } debug;
    auto* renderer=physx::pvdsdk::PvdUserRenderer::create();renderer->setClient(&debug);renderer->setInstanceId(reinterpret_cast<void*>(110));
    physx::pvdsdk::PvdDebugPoint point(physx::PxVec3(1,2,3),0xffff0000);
    physx::pvdsdk::PvdDebugLine line(physx::PxVec3(1,0,0),physx::PxVec3(2,0,0),0xff00ff00);
    physx::pvdsdk::PvdDebugTriangle tri(physx::PxVec3(0),physx::PxVec3(1,0,0),physx::PxVec3(0,1,0),0xff0000ff);
    renderer->drawPoints(&point,1);renderer->drawLines(&line,1);renderer->drawTriangles(&tri,1);
    renderer->visualizeJointFrames(physx::PxTransform(physx::PxVec3(2,0,0)),physx::PxTransform(physx::PxVec3(3,0,0)));
    renderer->visualizeLinearLimit(physx::PxTransform(physx::PxIdentity),physx::PxTransform(physx::PxIdentity),2,true);
    renderer->visualizeAngularLimit(physx::PxTransform(physx::PxIdentity),-physx::PxPi/4,physx::PxPi/4,false);
    renderer->visualizeLimitCone(physx::PxTransform(physx::PxIdentity),.5f,.25f,true);
    renderer->visualizeDoubleCone(physx::PxTransform(physx::PxIdentity),.5f,false);
    renderer->flushRenderEvents();renderer->release();
    const std::vector<float> verts={0,0,0, 1,0,0, 0,1,0, 0,0,1, 0,0,-1};
    for(uint64_t id:{101,102}) {floats(id,tetVerts,verts);ints(id,tets,{0,1,2,3,0,2,1,4});}
    floats(103,triVerts,{0,0,0,1,0,0,0,1,0});ints(103,tris,{0,1,2});
    OmniPvdClassHandle errorCodes=OMNI_PVD_INVALID_HANDLE;
    if(messages) {
        errorCodes=w->registerClass("PxErrorCode");
        w->registerEnumValue(errorCodes,"eDEBUG_INFO",1);
        w->registerEnumValue(errorCodes,"eDEBUG_WARNING",2);
        w->registerEnumValue(errorCodes,"eINVALID_PARAMETER",4);
        w->recordMessage(1,"Capture started before the first frame","startup.cpp",1,1,errorCodes);
    }
    for(uint64_t frame=1;frame<=(messages?12u:3u);++frame) {
        w->startFrame(1,frame);
        if(messages) {
            if(frame==2) {
                w->recordMessage(1,"Invalid contact distance","A.cpp",10,4,errorCodes);
                w->recordMessage(1,"Invalid shape radius","A.cpp",2,4,errorCodes);
            }
            if(frame==3)w->recordMessage(1,"Warning: contact cache exhausted","B.cpp",7,2,errorCodes);
            if(frame==10)w->recordMessage(1,"Frame ten completed","Info.cpp",12,1,errorCodes);
            if(frame==11)w->recordMessage(1,"Unnamed recorded type","Unknown.cpp",0,99);
        }
        floats(110,scalar,{float(frame)*10});
        floats(110,variable,frame==1?std::vector<float>{}:frame==2?std::vector<float>{3}:std::vector<float>{4,5});
        ints(110,pair,{2,3});
        w->setAttribute(1,120,debugEvents,debug.bytes.data(),debug.bytes.size());
        if(frame==2) {
            floats(1,contactPoints,{5,6,7});floats(1,contactNormals,{0,2,0});ints(1,contactCounts,{1});
            const uint64_t pairActors[]={10,20};w->setAttribute(1,1,contactActors,reinterpret_cast<const uint8_t*>(pairActors),sizeof(pairActors));
        }
        for(uint64_t id:{101,102}) {
            std::vector<float> values;
            for(std::size_t j=0;j<verts.size();j+=3) values.insert(values.end(),{verts[j]+float(frame-1),verts[j+1],verts[j+2],1});
            floats(id,tetPos,values);
            floats(id,tetVelocity,{1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0});
        }
        floats(103,triPos,{float(frame-1),0,0,1, float(frame),0,0,1, float(frame-1),1,0,1});
        floats(104,particlePos,{float(frame),2,3,1, 4,5,6,0});
        w->stopFrame(1,frame);
        // Real unpublished topology between completed frames, for live owned
        // overlay-request retention tests. Ordinary value updates do not mark
        // the adapter's structural index as unpublished.
        if(argc==3 && std::strcmp(argv[2],"--joint-live-tail")==0 && frame==2)create(analytics,111);
    }
    const bool closed=stream->closeStream();
    destroyOmniPvdWriter(*w);destroyOmniPvdFileWriteStream(*stream);
    foundation->release();
    return closed?0:5;
}

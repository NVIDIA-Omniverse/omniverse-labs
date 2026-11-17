/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Asset-free CPU diagnostic control for the parameters discussed in public PhysX issue584.
// Independently authored; this does not reproduce or fix the GPU contact-reduction report.
#include "PxPhysicsAPI.h"
#include "omnipvd/PxOmniPvd.h"
#include "OmniPvdWriter.h"
#include "OmniPvdWriteStream.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace physx;
static_assert(PX_PHYSICS_VERSION_MAJOR==5&&PX_PHYSICS_VERSION_MINOR==12&&PX_PHYSICS_VERSION_BUGFIX==0,"The qualified control requires SDK5.12.0 headers");

static void require(bool value,const char* reason) { if(!value)throw std::runtime_error(reason); }
static std::string quoted(const std::string& value) {
    std::ostringstream out;out<<'"';
    for(unsigned char c:value) {
        if(c=='"'||c=='\\')out<<'\\'<<c;
        else if(c<32)out<<"\\u00"<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(c)<<std::dec;
        else out<<c;
    }
    return out.str()+'"';
}
struct FileStream final:OmniPvdWriteStream {
    FILE* file=nullptr;bool failed=false;
    explicit FileStream(const char* path) { file=std::fopen(path,"wbx");require(file,"Capture destination must be a new writable file"); }
    ~FileStream() override { if(file)closeStream(); }
    uint64_t writeBytes(const uint8_t* bytes,uint64_t count) override {
        if(!file||failed)return 0;
        auto written=std::fwrite(bytes,1,size_t(count),file);failed=written!=count;return written;
    }
    bool flush() override { if(!file)return false;failed=std::fflush(file)!=0||failed;return !failed; }
    bool openStream() override { return file&&!failed; }
    bool closeStream() override {
        if(!file)return !failed;
        flush();failed=std::fclose(file)!=0||failed;file=nullptr;return !failed;
    }
};
struct World {
    PxDefaultAllocator allocator;PxDefaultErrorCallback errors;
    PxFoundation* foundation=nullptr;PxOmniPvd* omni=nullptr;PxPhysics* physics=nullptr;
    PxDefaultCpuDispatcher* dispatcher=nullptr;PxScene* scene=nullptr;
    ~World() {
        if(omni&&omni->isSampling())omni->stopSampling();
        if(scene)scene->release();if(physics)physics->release();
        if(dispatcher)dispatcher->release();if(omni)omni->release();if(foundation)foundation->release();
    }
};
struct Mesh { std::vector<PxVec3> vertices;std::vector<PxU32> triangles; };
static Mesh surfaceGrid() {
    const std::array<double,3> lengths{.1,.0157,.1177};
    std::array<int,3> cells{};for(int a=0;a<3;++a)cells[a]=std::max(1,int(std::lround(lengths[a]/.005)));
    Mesh result;std::map<std::array<int,3>,PxU32> vertices;
    auto vertex=[&](std::array<int,3> grid) {
        auto found=vertices.find(grid);if(found!=vertices.end())return found->second;
        PxU32 index=PxU32(result.vertices.size());vertices.emplace(grid,index);
        result.vertices.emplace_back(float(lengths[0]*(double(grid[0])/cells[0]-.5)),
            float(lengths[1]*(double(grid[1])/cells[1]-.5)),float(lengths[2]*(double(grid[2])/cells[2]-.5)));
        return index;
    };
    for(int normal=0;normal<3;++normal)for(int high=0;high<2;++high) {
        int u=(normal+1)%3,v=(normal+2)%3;
        for(int i=0;i<cells[u];++i)for(int j=0;j<cells[v];++j) {
            std::array<PxU32,4> corners{};
            for(int k=0;k<4;++k) {
                std::array<int,3> point{};point[normal]=high*cells[normal];
                point[u]=i+(k==1||k==2);point[v]=j+(k>=2);corners[k]=vertex(point);
            }
            for(auto t:std::array<std::array<int,3>,2>{{{0,1,2},{0,2,3}}}) {
                if(!high)std::swap(t[1],t[2]);for(int k:t)result.triangles.push_back(corners[k]);
            }
        }
    }
    return result;
}
static void vector(std::ostream& out,const PxVec3& v) { out<<'['<<v.x<<','<<v.y<<','<<v.z<<']'; }
static void pose(std::ostream& out,const PxTransform& p) { out<<'['<<p.q.x<<','<<p.q.y<<','<<p.q.z<<','<<p.q.w<<','<<p.p.x<<','<<p.p.y<<','<<p.p.z<<']'; }
static void sample(std::ostream& out,int step,const char* phase,PxRigidDynamic& body,PxScene& scene) {
    auto p=body.getGlobalPose();auto up=p.q.rotate(PxVec3(0,0,1));
    out<<"{\"step\":"<<step<<",\"phase\":"<<quoted(phase)<<",\"pose\":";pose(out,p);
    out<<",\"linear\":";vector(out,body.getLinearVelocity());out<<",\"angular\":";vector(out,body.getAngularVelocity());
    out<<",\"mass\":"<<body.getMass()<<",\"inertia\":";vector(out,body.getMassSpaceInertiaTensor());
    out<<",\"comPose\":";pose(out,body.getCMassLocalPose());out<<",\"gravity\":";vector(out,scene.getGravity());
    out<<",\"tilt\":"<<std::asin(std::clamp(-double(up.y),-1.,1.))*180/std::acos(-1.)<<'}';
}
int main(int argc,char** argv) {
    FILE* log=nullptr;std::string ownedLog;
    try {
        std::map<std::string,std::string> args;
        require(argc==11,"Use --capture NEW_OVD --log NEW_JSON --steps N --hz 240|480 --tilt 1.5|3");
        for(int i=1;i<argc;i+=2)require(args.emplace(argv[i],argv[i+1]).second,"Duplicate option");
        require(args.size()==5&&args.count("--capture")&&args.count("--log")&&args.count("--steps")&&args.count("--hz")&&args.count("--tilt"),"Unknown option");
        size_t used=0;int steps=std::stoi(args.at("--steps"),&used);require(used==args.at("--steps").size()&&steps>=1&&steps<=4800,"Steps must be 1..4800");
        int hz=std::stoi(args.at("--hz"),&used);require(used==args.at("--hz").size()&&(hz==240||hz==480),"Rate must be 240 or 480");
        double tilt=std::stod(args.at("--tilt"),&used);require(used==args.at("--tilt").size()&&(tilt==1.5||tilt==3.),"Tilt must be 1.5 or 3 degrees");
        log=std::fopen(args.at("--log").c_str(),"wbx");require(log,"Log destination must be a new writable file");
        ownedLog=args.at("--log");
        FileStream stream(args.at("--capture").c_str());World world;
        world.foundation=PxCreateFoundation(PX_PHYSICS_VERSION,world.allocator,world.errors);require(world.foundation,"Foundation creation failed");
        world.omni=PxCreateOmniPvd(*world.foundation);require(world.omni,"OmniPVD creation failed");
        auto writer=world.omni->getWriter();require(writer,"Writer unavailable");writer->setWriteStream(stream);
        world.physics=PxCreatePhysics(PX_PHYSICS_VERSION,*world.foundation,PxTolerancesScale(),true,nullptr,world.omni);require(world.physics,"Physics creation failed");
        world.dispatcher=PxDefaultCpuDispatcherCreate(1);require(world.dispatcher,"Dispatcher creation failed");
        PxSceneDesc desc(world.physics->getTolerancesScale());desc.gravity=PxVec3(0,0,-9.81f);desc.cpuDispatcher=world.dispatcher;
        desc.filterShader=PxDefaultSimulationFilterShader;desc.solverType=PxSolverType::eTGS;
        world.scene=world.physics->createScene(desc);require(world.scene,"Scene creation failed");
        auto material=world.physics->createMaterial(.5f,.4f,0);require(material,"Material creation failed");
        auto ground=PxCreatePlane(*world.physics,PxPlane(0,0,1,0),*material);require(ground,"Plane creation failed");world.scene->addActor(*ground);
        auto grid=surfaceGrid();PxTriangleMeshDesc meshDesc;
        meshDesc.points.count=PxU32(grid.vertices.size());meshDesc.points.stride=sizeof(PxVec3);meshDesc.points.data=grid.vertices.data();
        meshDesc.triangles.count=PxU32(grid.triangles.size()/3);meshDesc.triangles.stride=3*sizeof(PxU32);meshDesc.triangles.data=grid.triangles.data();
        PxSDFDesc sdf;sdf.spacing=.001f;meshDesc.sdfDesc=&sdf;
        PxCookingParams cooking(world.physics->getTolerancesScale());cooking.buildGPUData=false;
        auto mesh=PxCreateTriangleMesh(cooking,meshDesc,world.physics->getPhysicsInsertionCallback());require(mesh,"CPU SDF cooking failed");
        PxQuat rotation(float(tilt*std::acos(-1.)/180),PxVec3(1,0,0));PxVec3 pivot(0,-.0157f/2,0);
        auto body=world.physics->createRigidDynamic(PxTransform(pivot+rotation.rotate(PxVec3(0,0,.1177f/2)-pivot)+PxVec3(0,0,.0001f),rotation));require(body,"Body creation failed");
        auto shape=world.physics->createShape(PxTriangleMeshGeometry(mesh),*material,true);require(shape,"Shape creation failed");require(body->attachShape(*shape),"Shape attachment failed");shape->release();mesh->release();material->release();
        const double mass=1300*.1*.0157*.1177;body->setMass(float(mass));
        body->setMassSpaceInertiaTensor(PxVec3(float(mass*(.0157*.0157+.1177*.1177)/12),float(mass*(.1*.1+.1177*.1177)/12),float(mass*(.1*.1+.0157*.0157)/12)));
        body->setLinearDamping(0);body->setAngularDamping(0);body->setSleepThreshold(0);world.scene->addActor(*body);
        require(world.omni->startSampling()&&world.omni->isSampling(),"Sampling did not start at idle boundary");require(writer->getStatus()==0,"Writer failed during snapshot");
        float dt=1.f/float(hz);std::ostringstream output;output<<std::setprecision(17);
        output<<"{\"format\":\"ovdnext.cpu-rocking-control\",\"version\":1,\"complete\":true,\"sdkVersion\":\""<<PX_PHYSICS_VERSION_MAJOR<<'.'<<PX_PHYSICS_VERSION_MINOR<<'.'<<PX_PHYSICS_VERSION_BUGFIX<<"\",\"sdkRevision\":"<<quoted(SHOWCASE_SDK_REVISION)
            <<",\"sdkLibrarySha256\":"<<quoted(SHOWCASE_SDK_SHA256)<<",\"clientSha256\":"<<quoted(SHOWCASE_CLIENT_SHA256)<<",\"compiler\":"<<quoted(SHOWCASE_COMPILER)
            <<",\"hz\":"<<hz<<",\"stepSeconds\":"<<double(dt)<<",\"initialTiltDegrees\":"<<tilt<<",\"steps\":"<<steps
            <<",\"units\":\"SI\",\"workPolicy\":\"diagnostic-no-work-accounting\",\"solver\":\"CPU-TGS\",\"buildGPUData\":false,\"actorHandle\":\""<<uintptr_t(body)<<"\",\"sceneHandle\":\""<<uintptr_t(world.scene)<<"\",\"argv\":[";
        for(int i=0;i<argc;++i){if(i)output<<',';output<<quoted(argv[i]);}output<<"],\"initial\":";sample(output,0,"initial",*body,*world.scene);output<<",\"samples\":[";
        for(int step=1;step<=steps;++step) {
            if(step>1)output<<',';sample(output,step,"pre",*body,*world.scene);
            world.scene->simulate(dt);require(world.scene->fetchResults(true),"Simulation fetch failed");
            output<<',';sample(output,step,"post",*body,*world.scene);require(writer->getStatus()==0,"Writer failed while stepping");
        }
        require(world.omni->stopSampling()&&!world.omni->isSampling(),"Sampling did not stop");
        require(writer->getStatus()==0&&stream.closeStream(),"Capture failed to finalize");output<<"]}\n";
        auto text=output.str();bool written=std::fwrite(text.data(),1,text.size(),log)==text.size();
        bool closed=std::fclose(log)==0;log=nullptr;require(written&&closed,"Log failed to finalize; capture is unpaired");
        std::puts("Finalized qualified CPU capture and paired live-observation log");return 0;
    } catch(const std::exception& error) {
        if(log){auto text=std::string("{\"format\":\"ovdnext.cpu-rocking-control\",\"version\":1,\"complete\":false,\"error\":")+quoted(error.what())+"}\n";std::fwrite(text.data(),1,text.size(),log);std::fclose(log);}
        // A close/write failure can leave syntactically complete bytes: retain a
        // separate failure marker so the generator rejects the entire pair.
        if(!ownedLog.empty())if(FILE* failed=std::fopen((ownedLog+".failed").c_str(),"wbx")){std::fputs("Recorder failed; do not use this capture/log pair\n",failed);std::fclose(failed);}
        std::fprintf(stderr,"Control failed: %s\n",error.what());return 1;
    }
}

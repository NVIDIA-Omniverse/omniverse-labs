/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "Geometry.h"
#include "PvdEngine.h"
#include "fakeinstanceheader.h"
#include "PxBounds3.h"
#include "PxMat44.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace ovdnext {
namespace {
using namespace pvdapp;
using namespace physx;
constexpr size_t maxGeometryBytes = 256 * 1024 * 1024;
uint64_t hashBytes(const uint8_t* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i=0; i<n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
template<class T> uint64_t hashVector(const std::vector<T>& v) {
    return hashBytes(reinterpret_cast<const uint8_t*>(v.data()), v.size()*sizeof(T));
}
struct CacheEntry {
    uint64_t verticesHash=0, indicesHash=0;
    std::shared_ptr<GeometryMesh> mesh;
};
}
struct GeometryExtractor::Impl {
    PvdEngine* engine = nullptr;
    std::vector<InstanceDescription> instances;
    std::unordered_map<int32_t, const InstanceDescription*> byId;
    std::unordered_map<int32_t, std::string> classes;
    std::map<std::pair<int32_t,std::string>, int32_t> propertyIds;
    // Topology identity is not sampled-vertex identity: selected actors may
    // share a mesh, including both channels of one deformable volume.
    using MeshKey=std::tuple<int32_t,uint8_t,int32_t,uint8_t>;
    std::map<MeshKey, CacheEntry> cache;
    struct SourceCache { uint64_t vertices=0, topology=0; std::shared_ptr<GeometryMesh> mesh; };
    std::map<MeshKey, SourceCache> sourceCache;
    std::map<int32_t,std::pair<uint64_t,std::shared_ptr<const std::vector<uint32_t>>>> clothTopology;
    std::map<MeshKey, std::shared_ptr<GeometryMesh>> frameMeshes;
    uint64_t nextId = 1, epoch = std::numeric_limits<uint64_t>::max();
    std::shared_ptr<GeometrySnapshot> output;
    std::unordered_set<uint64_t> emittedMeshes;

    void initialize(PvdEngine& source) {
    engine=&source; output=std::make_shared<GeometrySnapshot>();
    output->segmentEpoch=source.getOvdSegmentEpoch();
    if(epoch!=output->segmentEpoch) { cache.clear(); sourceCache.clear(); clothTopology.clear(); propertyIds.clear(); epoch=output->segmentEpoch; }
    instances.resize(source.getNbInstances());
    instances.resize(source.getInstances(instances.data(),instances.size()));
    byId.clear(); classes.clear(); frameMeshes.clear(); emittedMeshes.clear();
    std::vector<ClassDescription> nativeClasses(source.getNbClasses()); nativeClasses.resize(source.getClasses(nativeClasses.data(),nativeClasses.size()));
    for(const auto& c:nativeClasses) classes[c.mClassId]=c.mName.mName?c.mName.mName:"";
    for(const auto& i:instances) if(i.mAlive && i.mInstPtr) byId.emplace(i.mId,&i);
    }
    const InstanceDescription* get(int32_t id) {
        auto it=byId.find(id);
        return it==byId.end() ? nullptr : it->second;
    }
    int32_t prop(const InstanceDescription& i, const char* name) {
        auto key=std::make_pair(i.mClassId,std::string(name));
        auto it=propertyIds.find(key);
        if(it!=propertyIds.end()) return it->second;
        auto p=engine->findPropertyFlex(i.mClassId,name);
        int32_t id=p.hasValue()?p.getValue().mPropertyId:-1;
        propertyIds.emplace(std::move(key),id);
        return id;
    }
    template<class T> T scalar(const InstanceDescription& i,const char* name,T fallback) {
        int32_t p=prop(i,name);
        T value=fallback;
        if(p>=0) engine->getPropertyValue(i.mInstPtr,p,&value,sizeof(T));
        return value;
    }
    bool hasData(const InstanceDescription& i,const char* name) {
        const int32_t p=prop(i,name);
        return p>=0 && engine->getPropertyByteSize(i.mInstPtr,p)>0;
    }
    DataRef<uint8_t> raw(const InstanceDescription& i,const char* name) {
        int32_t p=prop(i,name);
        if(p<0) return {};
        auto v=engine->getRawPropertyValue(i.mInstPtr,p);
        if(v.size()>maxGeometryBytes) {
            diagnostic(i,"geometry array exceeds 256 MiB transfer limit");
            return {};
        }
        return v;
    }
    template<class T> std::vector<T> array(const InstanceDescription& i,const char* name) {
        const int32_t p=prop(i,name);
        if(p<0) return {};
        const size_t bytes=engine->getPropertyByteSize(i.mInstPtr,p);
        if(bytes>maxGeometryBytes || bytes%sizeof(T)) { diagnostic(i,std::string("invalid array: ")+name); return {}; }
        std::vector<T> result(bytes/sizeof(T));
        if(bytes) {
            auto written=engine->getPropertyValue(i.mInstPtr,p,result.data(),bytes);
            result.resize(written/sizeof(T));
        }
        return result;
    }
    std::vector<int32_t> refs(const InstanceDescription& i,const char* name) {
        // getPropertyValue maps OVD 64-bit writer handles to stable engine IDs.
        return array<int32_t>(i,name);
    }
    int32_t first(const InstanceDescription& i,const char* name) {
        auto v=refs(i,name); return v.empty()?0:v.front();
    }
    void diagnostic(const InstanceDescription& i,const std::string& what) {
        output->diagnostics.push_back(std::to_string(i.mId)+" ("+classes[i.mClassId]+"): "+what);
    }
    uint64_t fingerprint(const InstanceDescription& i, std::initializer_list<const char*> names) {
        uint64_t hash=0;
        for(const char* name:names) {
            auto bytes=raw(i,name);
            hash ^= hashBytes(bytes.begin(),bytes.size())+0x9e3779b97f4a7c15ull+(hash<<6)+(hash>>2);
        }
        return hash;
    }
    std::vector<float> vec3s(const InstanceDescription& i,const char* name,size_t stride=3) {
        auto values=array<float>(i,name);
        if(stride==3) {
            if(values.size()%3) { diagnostic(i,std::string("malformed xyz array: ")+name); return {}; }
            return values;
        }
        if(values.size()%stride) { diagnostic(i,std::string("malformed vector array: ")+name); return {}; }
        std::vector<float> xyz; xyz.reserve(values.size()/stride*3);
        for(size_t j=0;j<values.size();j+=stride) xyz.insert(xyz.end(),values.begin()+j,values.begin()+j+3);
        return xyz;
    }
    std::shared_ptr<GeometryMesh> storeMesh(const InstanceDescription& source, GeometryKind kind,
                                         std::vector<float> vertices,std::vector<uint32_t> indices,int32_t owner=0,uint8_t channel=0) {
        return storeMesh(source,kind,std::move(vertices),std::make_shared<const std::vector<uint32_t>>(std::move(indices)),owner,channel);
    }
    std::shared_ptr<GeometryMesh> storeMesh(const InstanceDescription& source, GeometryKind kind,
                                         std::vector<float> vertices,std::shared_ptr<const std::vector<uint32_t>> indexBuffer,int32_t owner=0,uint8_t channel=0) {
        const auto& indices=*indexBuffer;
        if(vertices.empty()) return nullptr;
        for(float v:vertices) if(!std::isfinite(v)) { diagnostic(source,"non-finite mesh vertices"); return nullptr; }
        const size_t count=vertices.size()/3;
        const size_t width=kind==GeometryKind::Lines?2:3;
        if(kind!=GeometryKind::Points && (indices.empty() || indices.size()%width)) {
            diagnostic(source,"missing or malformed topology"); return nullptr;
        }
        if(std::any_of(indices.begin(),indices.end(),[count](uint32_t i){return i>=count;})) {
            diagnostic(source,"mesh index outside vertex buffer"); return nullptr;
        }
        const MeshKey key{source.mId,uint8_t(kind),owner,channel};
        auto& entry=cache[key];
        const auto vh=hashVector(vertices), ih=hashVector(indices);
        if(entry.mesh && entry.verticesHash==vh && entry.indicesHash==ih) return entry.mesh;
        auto mesh=std::make_shared<GeometryMesh>(); mesh->id=nextId++; mesh->sourceId=source.mId;
        mesh->vertices=std::move(vertices);
        if(entry.mesh && entry.indicesHash==ih) {
            mesh->indices=entry.mesh->indices;
            mesh->topologyId=entry.mesh->topologyId;
        } else {
            mesh->indices=std::move(indexBuffer);
            mesh->topologyId=nextId++;
        }
        PxBounds3 b=PxBounds3::empty();
        for(size_t j=0;j<mesh->vertices.size();j+=3) b.include(PxVec3(mesh->vertices[j],mesh->vertices[j+1],mesh->vertices[j+2]));
        mesh->bounds={b.minimum.x,b.minimum.y,b.minimum.z,b.maximum.x,b.maximum.y,b.maximum.z};
        entry.mesh=mesh; entry.verticesHash=vh; entry.indicesHash=ih;
        return mesh;
    }
    std::shared_ptr<GeometryMesh> mesh(const InstanceDescription& i,bool deformable=false,
                                     const InstanceDescription* actor=nullptr,const char* positions=nullptr,uint8_t channel=0) {
        const bool selected=actor && positions && hasData(*actor,positions);
        const int32_t owner=selected?actor->mId:0;
        if(!selected) channel=0;
        const MeshKey key{i.mId,uint8_t(GeometryKind::Mesh),owner,channel};
        auto existing=frameMeshes.find(key); if(existing!=frameMeshes.end()) return existing->second;
        std::vector<float> vertices;
        std::vector<uint32_t> indices;
        const auto& cls=classes[i.mClassId];
        const uint64_t verticesHash=selected?fingerprint(*actor,{positions}):engine->isOvdFile()?fingerprint(i,deformable?
            std::initializer_list<const char*>{"positions","verts"}:std::initializer_list<const char*>{"verts"}):
            fingerprint(i,{"Points","Samples","NbRows","NbColumns"});
        const uint64_t topologyHash=engine->isOvdFile()?fingerprint(i,{"tris","tets"}):
            fingerprint(i,{"Triangles","HullPolygons","PolygonIndexes","NbRows","NbColumns"});
        auto& prior=sourceCache[key];
        if(prior.mesh && prior.vertices==verticesHash && prior.topology==topologyHash) {
            frameMeshes[key]=prior.mesh; return prior.mesh;
        }
        const bool reuseTopology=prior.mesh && prior.topology==topologyHash;
        if(engine->isOvdFile()) {
            if(selected) vertices=vec3s(*actor,positions,4);
            else {
                if(deformable) vertices=vec3s(i,"positions",4);
                if(vertices.empty()) vertices=vec3s(i,"verts");
            }
            if(!reuseTopology && cls=="PxTetrahedronMesh") {
                auto tets=array<uint32_t>(i,"tets");
                if(tets.size()%4) { diagnostic(i,"malformed tetrahedron topology"); return nullptr; }
                std::map<std::array<uint32_t,3>,std::pair<size_t,std::array<uint32_t,3>>> faces;
                for(size_t j=0;j<tets.size();j+=4) {
                    const auto* q=tets.data()+j;
                    const std::array<uint32_t,3> fs[4]={{q[0],q[2],q[1]},{q[0],q[1],q[3]},{q[0],q[3],q[2]},{q[1],q[2],q[3]}};
                    for(auto f:fs) { auto sorted=f; std::sort(sorted.begin(),sorted.end()); auto& e=faces[sorted]; ++e.first; e.second=f; }
                }
                for(const auto& f:faces) if(f.second.first==1) indices.insert(indices.end(),f.second.second.begin(),f.second.second.end());
            } else if(!reuseTopology) indices=array<uint32_t>(i,"tris");
        } else if(cls=="PxHeightField") {
            const uint32_t rows=scalar<uint32_t>(i,"NbRows",0), cols=scalar<uint32_t>(i,"NbColumns",0);
            auto samples=array<uint32_t>(i,"Samples");
            if(rows<2 || cols<2 || uint64_t(rows)*cols>samples.size()) { diagnostic(i,"missing heightfield samples"); return nullptr; }
            vertices.reserve(size_t(rows)*cols*3);
            for(uint32_t r=0;r<rows;++r) for(uint32_t c=0;c<cols;++c)
                vertices.insert(vertices.end(),{float(r),float(int16_t(samples[r*cols+c]&0xffffu)),float(c)});
            // Preserve the reference renderer's fixed row/column triangulation.
            if(!reuseTopology) for(uint32_t r=0;r+1<rows;++r) for(uint32_t c=0;c+1<cols;++c) {
                const uint32_t k=r*cols+c;
                indices.insert(indices.end(),{k,k+1,k+cols,k+1,k+cols+1,k+cols});
            }
        } else {
            vertices=vec3s(i,"Points");
            if(!reuseTopology && cls=="PxConvexMesh") {
                struct Hull { uint16_t count, base; };
                auto hulls=array<Hull>(i,"HullPolygons"); auto polygon=array<uint8_t>(i,"PolygonIndexes");
                for(auto h:hulls) {
                    if(size_t(h.base)+h.count>polygon.size()) { diagnostic(i,"convex hull index outside polygon buffer"); return nullptr; }
                    for(uint32_t j=2;j<h.count;++j) indices.insert(indices.end(),{polygon[h.base],polygon[h.base+j-1],polygon[h.base+j]});
                }
            } else if(!reuseTopology) indices=array<uint32_t>(i,"Triangles");
        }
        auto result=reuseTopology?storeMesh(i,GeometryKind::Mesh,std::move(vertices),prior.mesh->indices,owner,channel):
            storeMesh(i,GeometryKind::Mesh,std::move(vertices),std::move(indices),owner,channel);
        prior={verticesHash,topologyHash,result};
        frameMeshes[key]=result; return result;
    }
    void emit(const InstanceDescription& actor,const InstanceDescription* shape,const InstanceDescription* geom,
              GeometryKind kind,const PxMat44& transform,const PxVec3& dims,std::shared_ptr<GeometryMesh> mesh={}) {
        for(uint32_t col=0;col<4;++col) for(uint32_t row=0;row<4;++row)
            if(!std::isfinite(transform(row,col))) { diagnostic(actor,"non-finite transform"); return; }
        const int32_t scene=engine->isOvdFile()?engine->getOwningSceneId(actor.mId):scalar<int32_t>(actor,"Scene",-1);
        output->objectIds.push_back(actor.mId); output->shapeIds.push_back(shape?shape->mId:actor.mId);
        output->geometryIds.push_back(geom?geom->mId:0); output->sceneIds.push_back(scene);
        output->actorClasses.push_back(actor.mClassId); output->kinds.push_back(uint8_t(kind));
        output->sleeping.push_back(scalar<uint8_t>(actor,"IsSleeping",0));
        output->doubleSided.push_back(geom && classes[geom->mClassId]=="PxHeightFieldGeometry");
        output->particleRadiusModes.push_back(kind==GeometryKind::Points ?
            (classes[actor.mClassId]=="PxPBDParticleSystem" ? 2 : 1) : 0);
        output->actorFlags.push_back(scalar<uint32_t>(actor,"RigidDynamicFlags",0));
        output->shapeFlags.push_back(shape?scalar<uint32_t>(*shape,"Flags",0):0);
        output->resourceIds.push_back(mesh?mesh->id:0);
        for(uint32_t row=0;row<4;++row) for(uint32_t col=0;col<4;++col) output->transforms.push_back(transform(row,col));
        output->dimensions.insert(output->dimensions.end(),{dims.x,dims.y,dims.z});
        PxBounds3 local;
        if(mesh) {
            local=PxBounds3(PxVec3(mesh->bounds[0],mesh->bounds[1],mesh->bounds[2]),PxVec3(mesh->bounds[3],mesh->bounds[4],mesh->bounds[5]));
            if(emittedMeshes.insert(mesh->id).second) output->meshes.push_back(mesh);
        } else {
            PxVec3 extent=dims;
            if(kind==GeometryKind::Sphere) extent=PxVec3(dims.x);
            if(kind==GeometryKind::Capsule) extent=PxVec3(dims.x+dims.y,dims.x,dims.x);
            if(kind==GeometryKind::Plane) extent=PxVec3(0,1,1); // finite frame marker; mathematical plane is unbounded
            local=PxBounds3(-extent,extent);
        }
        auto world=PxBounds3::empty();
        for(int c=0;c<8;++c) world.include(transform.transform(PxVec3((c&1)?local.maximum.x:local.minimum.x,
            (c&2)?local.maximum.y:local.minimum.y,(c&4)?local.maximum.z:local.minimum.z)));
        output->bounds.insert(output->bounds.end(),{world.minimum.x,world.minimum.y,world.minimum.z,world.maximum.x,world.maximum.y,world.maximum.z});
    }
    void geometry(const InstanceDescription& actor,const InstanceDescription* shape,const InstanceDescription* geom,PxMat44 transform) {
        PxVec3 dims(1);
    const auto& cls=classes[geom->mClassId]; GeometryKind kind;
    std::shared_ptr<GeometryMesh> geometry;
    if(cls=="PxSphereGeometry") { kind=GeometryKind::Sphere; dims=PxVec3(scalar<float>(*geom,"Radius",1)); }
    else if(cls=="PxBoxGeometry") { kind=GeometryKind::Box; dims=scalar<PxVec3>(*geom,"HalfExtents",PxVec3(1)); }
    else if(cls=="PxCapsuleGeometry") { kind=GeometryKind::Capsule; dims=PxVec3(scalar<float>(*geom,"Radius",1),scalar<float>(*geom,"HalfHeight",0),0); }
    else if(cls=="PxPlaneGeometry") { kind=GeometryKind::Plane; dims=PxVec3(1); }
    else {
        const char* ref=cls=="PxConvexMeshGeometry"?"ConvexMesh":cls=="PxTriangleMeshGeometry"?"TriangleMesh":cls=="PxHeightFieldGeometry"?"HeightField":nullptr;
        if(!ref) { diagnostic(*geom,"unsupported geometry class"); return; }
        auto source=get(scalar<int32_t>(*geom,ref,0)); if(!source) { diagnostic(*geom,"mesh reference is absent or not alive"); return; }
        geometry=mesh(*source); if(!geometry) return;
        kind=GeometryKind::Mesh;
        PxVec3 scale(1); PxQuat rotation(PxIdentity);
        if(engine->isOvdFile()) scale=scalar<PxVec3>(*geom,"scale",PxVec3(1));
        else if(cls=="PxHeightFieldGeometry") scale=PxVec3(scalar<float>(*geom,"RowScale",1),scalar<float>(*geom,"HeightScale",1),scalar<float>(*geom,"ColumnScale",1));
        else { scale=scalar<PxVec3>(*geom,"Scale.Scale",PxVec3(1)); rotation=scalar<PxQuat>(*geom,"Scale.Rotation",PxQuat(PxIdentity)); }
        PxMat44 scaling(PxIdentity); scaling.scale(PxVec4(scale,1));
        transform=transform*scaling*PxMat44(rotation); // existing RenderTranslator/PVDRenderer convention
    }
        emit(actor,shape,geom,kind,transform,dims,geometry);
    }
    void rigid(const InstanceDescription& actor) {
        auto global=scalar<PxTransform>(actor,"GlobalPose",PxTransform(PxIdentity));
        for(int32_t sid:refs(actor,"Shapes")) {
            auto shape=get(sid); if(!shape) continue;
            auto geom=get(scalar<int32_t>(*shape,"Geometry",0)); if(!geom) continue;
            auto local=scalar<PxTransform>(*shape,"LocalPose",PxTransform(PxIdentity));
            geometry(actor,shape,geom,PxMat44(global.transform(local)));
        }
    }
    void particles(const InstanceDescription& actor,bool ovd) {
        float radius=0.05f;
        if(ovd) {
            radius=scalar<float>(actor,"fluidRestOffset",0);
            if(!(radius>0)) radius=scalar<float>(actor,"restOffset",0.05f);
            if(!(radius>0)) radius=0.05f;
            const auto buffers=refs(actor,"particleBuffers");
            if(buffers.empty()) diagnostic(actor,"capture has no particle buffers at this frame");
            for(int32_t id:buffers) {
                auto buffer=get(id); if(!buffer) continue;
                auto data=storeMesh(*buffer,GeometryKind::Points,vec3s(*buffer,"positionInvMasses",4),std::vector<uint32_t>{});
                if(data) emit(actor,buffer,nullptr,GeometryKind::Points,PxMat44(PxIdentity),PxVec3(radius),data);
            }
        } else {
            auto data=storeMesh(actor,GeometryKind::Points,vec3s(actor,"Positions"),std::vector<uint32_t>{});
            if(data) emit(actor,nullptr,nullptr,GeometryKind::Points,PxMat44(PxIdentity),PxVec3(radius),data);
        }
    }
    void cloth(const InstanceDescription& actor) {
        const int32_t pid=prop(actor,"ParticleBuffer"); if(pid<0) return;
        auto desc=engine->findPropertyFlex(actor.mClassId,"ParticleBuffer");
        auto dtype=engine->getClass(desc.getValue().mDatatype); if(!dtype.hasValue()) return;
        const auto stride=dtype.getValue().get32BitSize();
        auto bytes=raw(actor,"ParticleBuffer"); if(stride<12 || bytes.size()%stride) { diagnostic(actor,"malformed cloth particles"); return; }
        // The original cloth translator resolves Pos through the native struct
        // schema. Preserve this, including schemas whose Pos isn't byte zero.
        std::vector<float> vertices;
        for(size_t j=0;j<bytes.size();j+=stride) {
            FakeInstanceHeader header(1,desc.getValue().mDatatype,desc.getValue().mDatatypeName,stride);
            header.mDataSectionStart=reinterpret_cast<uint64_t>(bytes.begin()+j);
            auto p=engine->getPropertyValue<PxVec3>(&header,"Pos");
            if(!p.hasValue()) { diagnostic(actor,"missing cloth Pos field"); return; }
            vertices.insert(vertices.end(),{p.getValue().x,p.getValue().y,p.getValue().z});
        }
        auto fabric=get(scalar<int32_t>(actor,"Fabric",0)); if(!fabric) { diagnostic(actor,"cloth fabric reference absent"); return; }
        const auto topologyHash=fingerprint(*fabric,{"ParticleIndices"});
        auto& topology=clothTopology[fabric->mId];
        if(!topology.second || topology.first!=topologyHash)
            topology={topologyHash,std::make_shared<const std::vector<uint32_t>>(array<uint32_t>(*fabric,"ParticleIndices"))};
        auto data=storeMesh(actor,GeometryKind::Lines,std::move(vertices),topology.second);
        if(data) {
            const auto pose=PxMat44(scalar<PxTransform>(actor,"GlobalPose",PxTransform(PxIdentity)));
            emit(actor,nullptr,nullptr,GeometryKind::Lines,pose,PxVec3(1),data);
            // RenderTranslator::prepareCloth submits both fabric edges and the
            // cloth particles. Share local vertices; the point draw ignores indices.
            emit(actor,nullptr,nullptr,GeometryKind::Points,pose,PxVec3(0.05f),data);
        }
    }
    void deformable(const InstanceDescription& actor,bool volume) {
        if(volume) {
            int32_t collision=scalar<int32_t>(actor,"collisionMesh",0), simulation=scalar<int32_t>(actor,"simulationMesh",0);
            if(collision<=0) collision=simulation;
            const bool selected=hasData(actor,"collisionPositions") || hasData(actor,"simulationPositions");
            for(int index=0;index<2;++index) {
                const int32_t mid=index==0?collision:simulation;
                if(mid<=0 || (index==1 && mid==collision && !selected)) continue;
                auto source=get(mid); if(!source) continue;
                auto data=mesh(*source,true,&actor,index==0?"collisionPositions":"simulationPositions",uint8_t(index+1)); if(!data) continue;
                auto handle=get(first(actor,index==0?"shapes":"simulationMeshShapes"));
                emit(actor,handle,source,GeometryKind::Mesh,PxMat44(PxIdentity),PxVec3(1),data);
            }
        } else {
            for(int32_t sid:refs(actor,"shapes")) {
                auto shape=get(sid); if(!shape) continue;
                auto geom=get(scalar<int32_t>(*shape,"Geometry",0)); if(!geom) continue;
                auto source=get(scalar<int32_t>(*geom,"TriangleMesh",0)); if(!source) continue;
                auto data=mesh(*source,true,&actor,"positions",1);
                if(data) emit(actor,shape,geom,GeometryKind::Mesh,PxMat44(PxIdentity),PxVec3(1),data);
            }
        }
    }
};
GeometryExtractor::GeometryExtractor():m(new Impl) {}
GeometryExtractor::~GeometryExtractor()=default;
std::shared_ptr<GeometrySnapshot> GeometryExtractor::extract(PvdEngine& engine) {
    m->initialize(engine);
    for(const auto& i:m->instances) {
        if(!i.mAlive || !i.mInstPtr) continue;
        const auto& cls=m->classes[i.mClassId];
        if(cls=="PxRigidStatic" || cls=="PxRigidDynamic" || cls=="PxArticulationLink") m->rigid(i);
        else if(cls=="PxPBDParticleSystem") m->particles(i,true);
        else if(cls=="PxParticleFluid" || cls=="PxParticleSystem") m->particles(i,false);
        else if(cls=="PxCloth") m->cloth(i);
        else if(cls=="PxDeformableVolume") m->deformable(i,true);
        else if(cls=="PxDeformableSurface") m->deformable(i,false);
    }
    return m->output;
}
std::shared_ptr<GeometrySnapshot> GeometryExtractor::extractReferences(PvdEngine& engine,const std::vector<GeometryRequest>& requests) {
    m->initialize(engine);
    for(const auto& request:requests) {
        const auto* geom=m->get(request.geometryId);if(!geom) continue;
        PxMat44 transform(PxIdentity);
        for(int row=0;row<4;++row) for(int col=0;col<4;++col) transform(row,col)=request.transform[row*4+col];
        const size_t count=m->output->objectIds.size();
        m->geometry(*geom,nullptr,geom,transform);
        if(m->output->objectIds.size()>count) {
            m->output->objectIds.back()=request.objectId;m->output->shapeIds.back()=request.objectId;
            m->output->sceneIds.back()=request.sceneId;
            m->output->overrideColors.insert(m->output->overrideColors.end(),request.color.begin(),request.color.end());
        }
    }
    return m->output;
}

}

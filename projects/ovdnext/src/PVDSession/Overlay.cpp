/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "Overlay.h"
#include "PvdEngine.h"
#include "PxBounds3.h"
#include "PxVec4.h"
#include "PxMat44.h"
#include "fakeinstanceheader.h"
#include "PvdObjectModelEventStream.h"
#include "PxPvdUserRenderImpl.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>

namespace ovdnext {
namespace {
using namespace pvdapp;
using namespace physx;
using namespace physx::pvdsdk;
constexpr size_t MaxArray=64*1024*1024,MaxPrimitives=1000000;
PxVec4 rgba(uint32_t color) { return PxVec4(float((color>>16)&255),float((color>>8)&255),float(color&255),float((color>>24)&255))/255.0f; }
struct Reader {
    PvdEngine& e;
    std::shared_ptr<OverlaySnapshot> out=std::make_shared<OverlaySnapshot>();
    float scale,speed,sdkJointScale,ovdJointScale;
    bool metadataOnly=false;
    uint32_t queryCounters[3]={0,0,0};
    std::vector<InstanceDescription> instances;
    std::unordered_map<int32_t,const InstanceDescription*> byId;
    std::unordered_map<int32_t,std::string> classes;
    std::map<std::pair<int32_t,std::string>,int32_t> properties;
    Reader(PvdEngine& engine,uint32_t mask,float gizmoScale,float toleranceSpeed,const JointScaleRequest& jointScaling={},const std::map<uint32_t,float>& familyScales={},float masterScale=1.0f):e(engine),scale(gizmoScale),speed(toleranceSpeed),
        sdkJointScale((jointScaling.independent?jointScaling.scale:gizmoScale)*masterScale),
        ovdJointScale((jointScaling.independent?jointScaling.baseLength*std::max(jointScaling.scale,.5f):gizmoScale)*masterScale) {
        out->jointScaling=jointScaling;
        out->familyScales=familyScales;
        out->masterScale=masterScale;
        out->mask=mask;out->segmentEpoch=e.getOvdSegmentEpoch();
        std::vector<SectionListInfo> scenes(e.getNbScenes());e.getScenes(scenes.data(),scenes.size());
        if(!scenes.empty()) out->currentFrame=e.getCurrentFrame(scenes.front().mStreamId);
        instances.resize(e.getNbInstances());instances.resize(e.getInstances(instances.data(),instances.size()));
        std::vector<ClassDescription> cls(e.getNbClasses());cls.resize(e.getClasses(cls.data(),cls.size()));
        for(auto& c:cls) classes[c.mClassId]=c.mName.mName?c.mName.mName:"";
        for(auto& i:instances) if(i.mAlive && i.mInstPtr) byId[i.mId]=&i;
        out->ready=true;
    }
    float familyScale(uint32_t category) const {
        const auto found=out->familyScales.find(category);
        return found==out->familyScales.end()?scale:found->second;
    }
    int32_t prop(const InstanceDescription& i,const char* name) {
        auto key=std::make_pair(i.mClassId,std::string(name));auto cached=properties.find(key);
        if(cached!=properties.end()) return cached->second;
        auto p=e.findPropertyFlex(i.mClassId,name);int32_t id=p.hasValue()?p.getValue().mPropertyId:-1;
        properties[key]=id;return id;
    }
    template<class T> bool read(const InstanceDescription& i,const char* name,T& value) {
        const auto p=prop(i,name);return p>=0 && e.getPropertyValue(i.mInstPtr,p,&value,sizeof(T))==sizeof(T);
    }
    template<class T> T value(const InstanceDescription& i,const char* name,T fallback) { read(i,name,fallback);return fallback; }
    const InstanceDescription* get(int32_t id) {auto found=byId.find(id);return found==byId.end()?nullptr:found->second;}
    int32_t scene(const InstanceDescription& i) {
        if(classes[i.mClassId]=="PxScene") return i.mId;
        return e.isOvdFile()?e.getOwningSceneId(i.mId):value<int32_t>(i,"Scene",-1);
    }
    template<class T> std::vector<T> array(const InstanceDescription& i,const char* name,bool recorded=false,uint64_t frame=0) {
        auto p=prop(i,name);if(p<0) return {};
        if(recorded) {
            const auto raw=e.getRawPropertyValueRecordedAtFrame(i.mInstPtr,p,frame);
            if(raw.size()>MaxArray || raw.size()%sizeof(T)) throw std::length_error("Malformed or oversized overlay array");
            std::vector<T> result(raw.size()/sizeof(T));
            if(raw.size()) std::memcpy(result.data(),raw.begin(),raw.size());return result;
        }
        const auto bytes=e.getPropertyByteSize(i.mInstPtr,p);
        if(bytes>MaxArray || bytes%sizeof(T)) throw std::length_error("Malformed or oversized overlay array");
        std::vector<T> result(bytes/sizeof(T));
        if(bytes) result.resize(e.getPropertyValue(i.mInstPtr,p,result.data(),bytes)/sizeof(T));return result;
    }
    void emit(OverlayKind kind,uint32_t category,int32_t object,int32_t sceneId,const PxVec3& a,const PxVec3& b,const PxVec3& c,
              const PxVec4& ca,const PxVec4& cb,const PxVec4& cc,int32_t r0=0,int32_t r1=0,OverlayColorRole role=OverlayColorRole::Captured) {
        if(metadataOnly || !(out->mask&category)) return;
        if(!a.isFinite() || !b.isFinite() || !c.isFinite() || !ca.isFinite() || !cb.isFinite() || !cc.isFinite()) return;
        if(out->kinds.size()>=MaxPrimitives) throw std::length_error("Overlay snapshot exceeds one million primitives; narrow the category mask");
        out->positions.insert(out->positions.end(),{a.x,a.y,a.z,b.x,b.y,b.z,c.x,c.y,c.z});
        out->colors.insert(out->colors.end(),{ca.x,ca.y,ca.z,ca.w,cb.x,cb.y,cb.z,cb.w,cc.x,cc.y,cc.z,cc.w});
        out->kinds.push_back(uint8_t(kind));out->categories.push_back(category);out->objectIds.push_back(object);
        out->colorRoles.push_back(uint8_t(role));
        out->sceneIds.push_back(sceneId);out->relatedIds.insert(out->relatedIds.end(),{r0,r1});
    }
    void line(uint32_t category,int32_t object,int32_t sid,const PxVec3& a,const PxVec3& b,const PxVec4& color,int32_t r0=0,int32_t r1=0,bool arrow=false,OverlayColorRole role=OverlayColorRole::Captured) {
        if(role==OverlayColorRole::Captured) {
            if(category==OverlayBounds) role=OverlayColorRole::Bounds;
            else if(category==OverlayCenterOfMass) role=OverlayColorRole::CenterOfMass;
            else if(category==OverlayVelocity || category==OverlayAngularVelocity) role=OverlayColorRole::Velocity;
        }
        emit(arrow?OverlayKind::Arrow:OverlayKind::Line,category,object,sid,a,b,PxVec3(0),color,color,color,r0,r1,role);
    }
    void point(uint32_t category,int32_t object,int32_t sid,const PxVec3& p,const PxVec4& color,int32_t r0=0,int32_t r1=0,OverlayColorRole role=OverlayColorRole::Captured) {
        emit(OverlayKind::Point,category,object,sid,p,PxVec3(0),PxVec3(0),color,color,color,r0,r1,role);
    }
    void axes(uint32_t category,int32_t object,int32_t sid,const PxTransform& pose,float extent,float brightness=1,int32_t r0=0,int32_t r1=0,OverlayColorRole role=OverlayColorRole::Captured) {
        for(int axis=0;axis<3;++axis) {PxVec3 direction(0);direction[axis]=extent;PxVec4 color(0,0,0,1);color[axis]=brightness;
            const auto axisRole=role==OverlayColorRole::Captured?OverlayColorRole(uint8_t(OverlayColorRole::AxisX)+axis):role;
            line(category,object,sid,pose.p,pose.transform(direction),color,r0,r1,true,axisRole);}
    }
    void diagnostic(const InstanceDescription& i,const std::string& message) { out->diagnostics.push_back(std::to_string(i.mId)+" ("+classes[i.mClassId]+"): "+message); }
    void actors(const InstanceDescription& actor) {
        const int32_t sid=scene(actor);
        PxTransform global(PxIdentity);if(!read(actor,"GlobalPose",global)) return;
        if(!global.isValid()) return;
        PxTransform mass(PxIdentity);const bool hasMass=read(actor,"CMassLocalPose",mass) && mass.isValid();
        if(out->mask&(OverlayAxes|OverlayVelocity|OverlayAngularVelocity)) {
            for(int32_t shapeId:array<int32_t>(actor,"Shapes")) {
                const auto* shape=get(shapeId);if(!shape) continue;
                PxTransform local(PxIdentity);read(*shape,"LocalPose",local);if(!local.isValid()) continue;
                const PxTransform pose=global*local*(hasMass?mass:PxTransform(PxIdentity));
                if(out->mask&OverlayAxes) axes(OverlayAxes,actor.mId,sid,pose,familyScale(OverlayAxes)*0.3f*out->masterScale,1,shapeId);
                if(out->mask&OverlayVelocity) {
                    PxVec3 v(0);if(read(actor,"LinearVelocity",v)) line(OverlayVelocity,actor.mId,sid,pose.p,pose.p+v*familyScale(OverlayVelocity)/speed*out->masterScale,PxVec4(1),shapeId,0,true);
                }
                if(out->mask&OverlayAngularVelocity) {
                    PxVec3 v(0);if(read(actor,"AngularVelocity",v)) line(OverlayAngularVelocity,actor.mId,sid,pose.p,pose.p+v*familyScale(OverlayAngularVelocity)/speed*out->masterScale,PxVec4(0.9f,0.3f,1,1),shapeId,0,true);
                }
            }
        }
        if(out->mask&OverlayBounds) {
            PxBounds3 bb; if(read(actor,"WorldBounds",bb) && bb.minimum.isFinite() && bb.maximum.isFinite()) {
                const auto extent=bb.maximum-bb.minimum;
                if(extent.x>=0 && extent.y>=0 && extent.z>=0 && extent.x+extent.y+extent.z>0) {
                    std::array<PxVec3,8> p;
                    for(int k=0;k<8;++k) p[k]=PxVec3(k&1?bb.maximum.x:bb.minimum.x,k&2?bb.maximum.y:bb.minimum.y,k&4?bb.maximum.z:bb.minimum.z);
                    for(int k=0;k<8;++k) for(int axis=0;axis<3;++axis) if(!(k&(1<<axis))) line(OverlayBounds,actor.mId,sid,p[k],p[k|(1<<axis)],PxVec4(1));
                }
            }
        }
        if((out->mask&OverlayCenterOfMass) && hasMass) {
            const auto p=(global*mass).p;
            for(int axis=0;axis<3;++axis) {PxVec3 d(0);d[axis]=familyScale(OverlayCenterOfMass)*.05f*out->masterScale;line(OverlayCenterOfMass,actor.mId,sid,p-d,p+d,PxVec4(1,.7f,.1f,1));}
        }
    }
    void deformableVelocity(const InstanceDescription& actor) {
        const InstanceDescription* source=nullptr;int32_t handle=0;
        const bool volume=classes[actor.mClassId]=="PxDeformableVolume";
        if(volume) {
            handle=value<int32_t>(actor,"simulationMesh",0);source=get(handle);
        } else {
            for(int32_t sid:array<int32_t>(actor,"shapes")) {
                const auto* shape=get(sid);if(!shape) continue;
                const auto* geom=get(value<int32_t>(*shape,"Geometry",0));if(!geom) continue;
                source=get(value<int32_t>(*geom,"TriangleMesh",0));if(source) {handle=sid;break;}
            }
        }
        if(!source) return;
        // Volume velocities are simulation-vertex data, never collision data.
        std::vector<PxVec4> positions,velocities;
        try {
            positions=array<PxVec4>(actor,volume?"simulationPositions":"positions");
            velocities=array<PxVec4>(actor,"velocities");
        } catch(const std::length_error& error) {
            diagnostic(actor,error.what());return;
        }
        if(!positions.empty() || !velocities.empty()) {
            if(positions.size()!=velocities.size()) {
                diagnostic(actor,"missing or mismatched deformable positions/velocities");return;
            }
            // Reject the selected actor's pair as a unit; a partially drawn
            // velocity field would hide invalid samples. The fourth float is
            // inverse mass/reserved data, not a rendered coordinate.
            auto finite=[](const PxVec4& v){return PxVec3(v.x,v.y,v.z).isFinite();};
            if(!std::all_of(positions.begin(),positions.end(),finite) ||
               !std::all_of(velocities.begin(),velocities.end(),finite)) {
                diagnostic(actor,"non-finite deformable positions/velocities");return;
            }
        } else {
            positions=array<PxVec4>(*source,"positions");velocities=array<PxVec4>(*source,"velocities");
            if(positions.empty()) for(const auto& p:array<PxVec3>(*source,"verts")) positions.push_back(PxVec4(p,1));
        }
        const size_t count=std::min(positions.size(),velocities.size());
        for(size_t k=0;k<count;++k) {
            const PxVec3 p(positions[k].x,positions[k].y,positions[k].z),v(velocities[k].x,velocities[k].y,velocities[k].z);
            line(OverlayVelocity,actor.mId,scene(actor),p,p+v*familyScale(OverlayVelocity)/speed*out->masterScale,PxVec4(1),handle,0,true);
        }
    }
    void joints(const InstanceDescription& joint) {
        const auto& cls=classes[joint.mClassId];const bool articulation=cls=="PxArticulationJointReducedCoordinate";
        if(!articulation && cls!="PxFixedJoint" && cls!="PxPrismaticJoint" && cls!="PxRevoluteJoint" && cls!="PxSphericalJoint" && cls!="PxDistanceJoint" && cls!="PxD6Joint" && cls!="PxGearJoint" && cls!="PxRackAndPinionJoint") return;
        const InstanceDescription* endpoints=&joint;
        if(!articulation && prop(joint,"actor0")<0) {
            // Integration 4 moved shared joint data to the core constraint.
            endpoints=get(value<int32_t>(joint,"constraint",0));
            if(!endpoints || classes[endpoints->mClassId]!="PxConstraint") return;
        }
        int32_t pId=0,cId=0;
        const auto endpoint=[&](const char* name,int32_t& id) {
            if(!read(*endpoints,name,id)) return false;
            if(id>0) return true;
            // Both null and unresolved handles map to -1 in the inspector.
            // Only an explicitly recorded null endpoint denotes the world.
            const auto raw=e.getRawPropertyValue(endpoints->mInstPtr,prop(*endpoints,name));
            uint64_t handle=0;
            if(raw.size()!=sizeof(handle)) return false;
            std::memcpy(&handle,raw.begin(),sizeof(handle));
            return handle==0;
        };
        if(!endpoint(articulation?"parentLink":"actor0",pId) ||
           !endpoint(articulation?"childLink":"actor1",cId)) return;
        const auto parent=get(pId),child=get(cId);
        // A null joint endpoint denotes the world; non-null dead endpoints do
        // not create surrogate frames at the origin.
        if((pId>0 && !parent)||(cId>0 && !child)) return;
        const auto pActor=parent?value<PxTransform>(*parent,"GlobalPose",PxTransform(PxIdentity)):PxTransform(PxIdentity);
        const auto cActor=child?value<PxTransform>(*child,"GlobalPose",PxTransform(PxIdentity)):PxTransform(PxIdentity);
        PxTransform pLocal(PxIdentity),cLocal(PxIdentity);
        if(articulation) {
            pLocal=PxTransform(value<PxVec3>(joint,"parentTranslation",PxVec3(0)),value<PxQuat>(joint,"parentRotation",PxQuat(PxIdentity)));
            cLocal=PxTransform(value<PxVec3>(joint,"childTranslation",PxVec3(0)),value<PxQuat>(joint,"childRotation",PxQuat(PxIdentity)));
        } else { read(joint,"actor0LocalPose",pLocal);read(joint,"actor1LocalPose",cLocal); }
        if(!pActor.isValid() || !cActor.isValid() || !pLocal.isValid() || !cLocal.isValid()) return;
        int32_t sid=scene(joint);if(sid<0) sid=parent?scene(*parent):child?scene(*child):-1;
        axes(OverlayJoints,joint.mId,sid,pActor*pLocal,ovdJointScale*1.5f,136/255.0f,pId,cId,OverlayColorRole::JointParent);
        axes(OverlayJoints,joint.mId,sid,cActor*cLocal,ovdJointScale,1,pId,cId,OverlayColorRole::JointChild);
    }
    void ovdContacts(const InstanceDescription& scene) {
        if(out->currentFrame%2==0) return;
        const uint64_t frame=uint64_t(out->currentFrame)+1;
        auto points=array<PxVec3>(scene,"pairsContactPoints",true,frame), normals=array<PxVec3>(scene,"pairsContactNormals",true,frame);
        auto actors=array<uint64_t>(scene,"pairsActors",true,frame);auto counts=array<uint32_t>(scene,"pairsContactCounts",true,frame);
        std::vector<std::array<int32_t,2>> pairs(points.size(),{{0,0}});
        size_t offset=0;
        for(size_t pair=0;pair<counts.size() && pair*2+1<actors.size();++pair) {
            if(counts[pair]>points.size()-offset) {diagnostic(scene,"contact pair counts exceed point count");break;}
            const std::array<int32_t,2> ids={{e.resolveInstanceId(actors[2*pair],scene.mInstPtr),e.resolveInstanceId(actors[2*pair+1],scene.mInstPtr)}};
            std::fill(pairs.begin()+offset,pairs.begin()+offset+counts[pair],ids);offset+=counts[pair];
        }
        const float normalLength=std::max(familyScale(OverlayContacts)*.005f,.001f)*out->masterScale, half=normalLength*.33f;
        for(size_t index=0;index<points.size();++index) {
            const auto p=points[index];if(!p.isFinite()) continue;
            PxVec3 n(0,1,0);bool validNormal=false;
            if(index<normals.size() && normals[index].isFinite() && normals[index].magnitudeSquared()>1e-8f) {n=normals[index].getNormalized();validNormal=true;}
            PxVec3 tangent=n.cross(PxVec3(1,0,0));if(tangent.magnitudeSquared()<1e-6f) tangent=n.cross(PxVec3(0,1,0));tangent.normalize();
            const auto second=n.cross(tangent).getNormalized();const auto ids=pairs[index];
            line(OverlayContacts,scene.mId,scene.mId,p-tangent*half,p+tangent*half,PxVec4(1,.6f,0,1),ids[0],ids[1],false,OverlayColorRole::ContactPoint);
            line(OverlayContacts,scene.mId,scene.mId,p-second*half,p+second*half,PxVec4(1,.6f,0,1),ids[0],ids[1],false,OverlayColorRole::ContactPoint);
            if(validNormal) line(OverlayContacts,scene.mId,scene.mId,p,p+n*normalLength,PxVec4(1,1,0,1),ids[0],ids[1],false,OverlayColorRole::ContactNormal);
        }
    }
    template<class Callback> void structs(const InstanceDescription& instance,const char* name,Callback callback) {
        auto p=e.findPropertyFlex(instance.mClassId,name);if(!p.hasValue()) return;
        auto cls=e.getClass(p.getValue().mDatatype);if(!cls.hasValue()) return;
        const size_t stride=cls.getValue().get32BitSize();if(!stride) return;
        auto raw=e.getRawPropertyValue(instance.mInstPtr,p.getValue().mPropertyId);
        if(raw.size()>MaxArray || raw.size()%stride) throw std::length_error("Malformed legacy overlay structure array");
        for(size_t j=0;j<raw.size();j+=stride) {
            FakeInstanceHeader h(1,p.getValue().mDatatype,p.getValue().mDatatypeName,stride);h.mDataSectionStart=reinterpret_cast<uint64_t>(raw.begin()+j);callback(h,j/stride);
        }
    }
    template<class T> T field(FakeInstanceHeader& h,const char* name,T fallback) {auto v=e.getPropertyValue<T>(&h,name);return v.hasValue()?v.getValue():fallback;}
    void legacyContacts(const InstanceDescription& scene) {
        structs(scene,"Contacts",[&](FakeInstanceHeader& h,size_t) {
            const auto p=field<PxVec3>(h,"Point",PxVec3(0)),n=field<PxVec3>(h,"Axis",PxVec3(0));
            const float penetration=-field<float>(h,"Separation",0);
            point(OverlayContacts,scene.mId,scene.mId,p,PxVec4(1,.6f,0,1),0,0,OverlayColorRole::ContactPoint);
            if(penetration>1e-6f) line(OverlayContacts,scene.mId,scene.mId,p,p+n*penetration*familyScale(OverlayContacts)*out->masterScale,PxVec4(.1f,.1f,1,1),0,0,true,OverlayColorRole::ContactNormal);
        });
    }
    void queryGeometry(int32_t id,int32_t sceneId,int32_t geometryId,const PxTransform& pose,bool hit=false) {
        if(metadataOnly) return;
        GeometryRequest request;request.geometryId=geometryId;request.objectId=id;request.sceneId=sceneId;request.color=hit?std::array<float,4>{{1,0,0,1}}:std::array<float,4>{{0,1,0,1}};
        PxMat44 matrix(pose);for(int row=0;row<4;++row) for(int col=0;col<4;++col) request.transform[row*4+col]=matrix(row,col);
        out->queryRequests.push_back(request);
    }
    void record(const InstanceDescription& scene,const std::string& name,const FakeInstanceHeader& h,size_t index,int32_t id) {
        if(out->queryRecords.size()>=MaxPrimitives) throw std::length_error("Query record metadata exceeds one million records");
        out->queryRecords.push_back({id,scene.mId,prop(scene,name.c_str()),h.mClassId,uint32_t(index),h.mDataSize});
    }
    void queries(const InstanceDescription& scene) {
        for(int batched=0;batched<2;++batched) {
            const std::string prefix=batched?"BatchedQueries.":"SceneQueries.";
            structs(scene,(prefix+"Raycasts").c_str(),[&](FakeInstanceHeader& h,size_t index) {
                const auto p=field<PxVec3>(h,"origin",PxVec3(0)),v=field<PxVec3>(h,"unitDir",PxVec3(0));
                const float distance=field<float>(h,"distance",0);const bool hit=field<uint32_t>(h,"hits_count",0)>0;
                const int32_t id=-int32_t((queryCounters[1]++<<4)+(batched?3:6));
                record(scene,prefix+"Raycasts",h,index,id);
                line(OverlayQueries,id,scene.mId,p,p+v*distance,hit?PxVec4(1,0,0,1):PxVec4(0,1,0,1),0,0,true);
            });
            structs(scene,(prefix+"Hits").c_str(),[&](FakeInstanceHeader& h,size_t index) {
                const auto p=field<PxVec3>(h,"Impact",PxVec3(0)),n=field<PxVec3>(h,"Normal",PxVec3(0));const int32_t id=-int32_t((queryCounters[0]++<<4)+(batched?2:5));
                record(scene,prefix+"Hits",h,index,id);
                line(OverlayQueries,id,scene.mId,p,p+n*scale*out->masterScale,PxVec4(0,0,1,1),0,0,true);point(OverlayQueries,id,scene.mId,p,PxVec4(1,1,0,1));
            });
            const auto poses=array<PxTransform>(scene,(prefix+"PoseList").c_str());
            const auto geometries=array<int32_t>(scene,(prefix+"GeometryList").c_str());
            structs(scene,(prefix+"Sweeps").c_str(),[&](FakeInstanceHeader& h,size_t index) {
                const auto base=field<uint32_t>(h,"pose_baseIndex",0),count=field<uint32_t>(h,"pose_count",0);
                if(count!=1 || base>=poses.size()) return;
                const auto p=poses[base].p,v=field<PxVec3>(h,"unitDir",PxVec3(0));const float distance=field<float>(h,"distance",0);
                const int32_t id=-int32_t((queryCounters[2]++<<4)+(batched?4:7));
                record(scene,prefix+"Sweeps",h,index,id);
                line(OverlayQueries,id,scene.mId,p,p+v*distance,PxVec4(1,.5f,0,1),0,0,true);
                const auto geomBase=field<uint32_t>(h,"geom_baseIndex",0),geomCount=field<uint32_t>(h,"geom_count",0);
                if(geomCount==1 && geomBase<geometries.size()) {
                    const bool hit=field<uint32_t>(h,"hits_count",0)>0;
                    queryGeometry(id,scene.mId,geometries[geomBase],poses[base],hit);
                    auto endPose=poses[base];endPose.p+=v*distance;queryGeometry(id,scene.mId,geometries[geomBase],endPose,hit);
                } else diagnostic(scene,"sweep geometry reference absent or malformed");
            });
            structs(scene,(prefix+"Overlaps").c_str(),[&](FakeInstanceHeader& h,size_t){
                const auto base=field<uint32_t>(h,"geom_baseIndex",0),count=field<uint32_t>(h,"geom_count",0);
                const auto pose=field<PxTransform>(h,"pose",PxTransform(PxIdentity));
                for(uint32_t k=0;k<count && uint64_t(base)+k<geometries.size();++k) queryGeometry(geometries[base+k],scene.mId,geometries[base+k],pose);
            });
        }
    }
    void custom(const InstanceDescription& instance);
};
struct DebugHandler: PvdUserRenderHandler,PvdObjectModelEventStreamPropertyHandler {
    Reader& r;const InstanceDescription& source;int32_t object,sid;std::vector<uint8_t> data;
    DebugHandler(Reader& reader,const InstanceDescription& instance):r(reader),source(instance),object(instance.mId),sid(reader.scene(instance)){}
    void handleProperty(PxU64,PxI32,PxI32,const PxU8* bytes,PxU32 length) override {
        if(data.size()+length>MaxArray) throw std::length_error("Custom debug event batch exceeds 64 MiB");
        if(length) data.insert(data.end(),bytes,bytes+length);
    }
    void handleRenderEvent(const SetInstanceIdRenderEvent& event) override {
        if(r.e.isOvdFile()) object=r.e.resolveInstanceId(event.mInstanceId,source.mInstPtr);
        else {
            object=0;
            for(const auto& i:r.instances) if(r.e.getStreamId(i.mId)==event.mInstanceId) {object=i.mId;break;}
        }
        const auto* i=r.get(object);if(i) sid=r.scene(*i);
    }
    void draw(const PvdDebugLine& line) {r.line(OverlayCustom,object,sid,line.pos0,line.pos1,rgba(line.color0));}
    void draw(const PvdDebugTriangle& t) {
        r.line(OverlayCustom,object,sid,t.pos0,t.pos1,rgba(t.color0));
        r.line(OverlayCustom,object,sid,t.pos1,t.pos2,rgba(t.color1));
        r.line(OverlayCustom,object,sid,t.pos2,t.pos0,rgba(t.color2));
    }
    void handleRenderEvent(const LinesRenderEvent& event) override {for(uint32_t i=0;i<event.mData.size();++i) draw(event.mData[i]);}
    void handleRenderEvent(const TrianglesRenderEvent& event) override {for(uint32_t i=0;i<event.mData.size();++i) draw(event.mData[i]);}
    void handleRenderEvent(const PointsRenderEvent& event) override {for(uint32_t i=0;i<event.mData.size();++i) r.point(OverlayCustom,object,sid,event.mData[i].pos,rgba(event.mData[i].color));}
    void handleRenderEvent(const DebugRenderEvent& event) override {
        for(uint32_t i=0;i<event.mLineData.size();++i) draw(event.mLineData[i]);
        for(uint32_t i=0;i<event.mTriangleData.size();++i) draw(event.mTriangleData[i]);
        for(uint32_t i=0;i<event.mPointData.size();++i) r.point(OverlayCustom,object,sid,event.mPointData[i].pos,rgba(event.mPointData[i].color));
    }
    void handleRenderEvent(const JointFramesRenderEvent& event) override {
        r.axes(OverlayJoints,object,sid,event.parent,r.sdkJointScale*1.5f,136/255.0f,0,0,OverlayColorRole::JointParent);
        r.axes(OverlayJoints,object,sid,event.child,r.sdkJointScale,1,0,0,OverlayColorRole::JointChild);
    }
    void limitLine(const PxTransform& pose,const PxVec3& a,const PxVec3& b,bool active) {
        r.line(OverlayJoints,object,sid,pose.transform(a),pose.transform(b),rgba(active?0xffff0000:0xff808080));
    }
    void handleRenderEvent(const LinearLimitRenderEvent& event) override {
        if(!(r.out->mask&OverlayJoints)) return;
        const PxTransform pose(event.t0.p+event.value*event.t0.q.getBasisVector0(),event.t0.q*PxQuat(PxHalfPi,PxVec3(0,1,0)));
        const float radius=r.sdkJointScale*.3f;PxVec3 previous(0,radius,0);
        for(int i=1;i<=20;++i) {const float angle=PxTwoPi*i/20;PxVec3 next(radius*PxSin(angle),radius*PxCos(angle),0);limitLine(pose,previous,next,event.active);previous=next;}
    }
    void handleRenderEvent(const AngularLimitRenderEvent& event) override {
        if(!(r.out->mask&OverlayJoints)) return;
        PxVec3 previous(0);
        for(int i=0;i<=20;++i) {const float angle=event.lower+(event.upper-event.lower)*i/20;const PxVec3 next=PxVec3(0,PxCos(angle),PxSin(angle))*r.sdkJointScale;
            limitLine(event.t0,previous,next,event.active);previous=next;}
        limitLine(event.t0,previous,PxVec3(0),event.active);
    }
    void handleRenderEvent(const LimitConeRenderEvent& event) override {
        if(!(r.out->mask&OverlayJoints)) return;
        PxVec3 previous(0);
        for(int i=0;i<=32;++i) {const float angle=PxTwoPi*i/32;const PxVec3 rv(0,-event.zSwing*PxSin(angle),event.ySwing*PxCos(angle));const float squared=rv.magnitudeSquared();
            const PxQuat q=PxQuat(0,2*rv.y,2*rv.z,1-squared)*(1/(1+squared));const auto next=q.rotate(PxVec3(1,0,0))*r.sdkJointScale;
            limitLine(event.t,previous,next,event.active);limitLine(event.t,PxVec3(0),next,event.active);previous=next;}
    }
    void handleRenderEvent(const DoubleConeRenderEvent& event) override {
        if(!(r.out->mask&OverlayJoints)) return;
        const float height=PxTan(event.angle);
        for(int side:{-1,1}) {
            PxVec3 previous=PxVec3(side*height,1,0)*r.sdkJointScale;
            for(int i=1;i<=32;++i) {const float angle=PxTwoPi*i/32;const auto next=PxVec3(side*height,PxCos(angle),PxSin(angle))*r.sdkJointScale;
                limitLine(event.t,previous,next,event.active);previous=next;}
            for(int i=0;i<32;++i) {const float angle=PxTwoPi*i/32;limitLine(event.t,PxVec3(0),PxVec3(side*height,PxCos(angle),PxSin(angle))*r.sdkJointScale,event.active);}
        }
    }
    void handleRenderEvent(const TextRenderEvent&) override {r.diagnostic(source,"captured debug text overlay is not yet exposed");}
};
void Reader::custom(const InstanceDescription& instance) {
    const int32_t property=prop(instance,"events");if(property<0) return;
    DebugHandler handler(*this,instance);
    auto* stream=e.getEventStream();
    if(stream) {
        std::vector<SectionListInfo> scenes(e.getNbScenes());e.getScenes(scenes.data(),scenes.size());
        if(scenes.empty()) return;
        uint64_t start=0;const auto end=stream->getCurrentEvent();const auto sceneId=scenes.front().mStreamId;
        uint32_t frame=e.getCurrentFrame(sceneId);if(frame) --frame;
        auto info=e.getFrameInfo(frame,sceneId);if(info.hasValue() && info.getValue().mEnd.isEventValid()) start=info.getValue().mEnd.mEventIndex;
        stream->getPropertyHistory(InstancePropertyId(instance.mId,property),start,end,handler);
    } else handler.data=array<uint8_t>(instance,"events");
    auto& parser=PvdUserRenderParser::create(false);
    try {parser.parseData(DataRef<const uint8_t>(handler.data.data(),handler.data.size()),handler);}
    catch(...) {parser.release();throw;}
    parser.release();
}
}
std::vector<QueryRecord> extractQueryRecords(PvdEngine& engine) {
    if(engine.isOvdFile()) return {};
    Reader reader(engine,OverlayQueries,1,10);reader.metadataOnly=true;
    for(const auto& i:reader.instances) if(i.mAlive && i.mInstPtr && reader.classes[i.mClassId]=="PxScene") reader.queries(i);
    return std::move(reader.out->queryRecords);
}
void validateJointScaleRequest(const JointScaleRequest& request,bool structuredOvd) {
    if(!std::isfinite(request.scale) || request.scale<0 || request.scale>2 ||
       !std::isfinite(request.baseLength) || request.baseLength<=0)
        throw std::invalid_argument("Joint scale must be finite in [0,2] and joint base length finite and positive");
    if(request.independent && structuredOvd &&
       !std::isfinite(request.baseLength*std::max(request.scale,.5f)*1.5f))
        throw std::invalid_argument("Structured OVD joint arm exceeds the finite float range");
}
void validateFamilyScales(const std::map<uint32_t,float>& scales) {
    for(const auto& entry:scales) {
        if(entry.first!=OverlayAxes && entry.first!=OverlayVelocity && entry.first!=OverlayAngularVelocity &&
           entry.first!=OverlayContacts && entry.first!=OverlayCenterOfMass)
            throw std::invalid_argument("Unsupported overlay family scale category");
        if(!std::isfinite(entry.second) || entry.second<=0)
            throw std::invalid_argument("Overlay family scales must be finite and positive");
    }
}
void validateMasterScale(float masterScale,float gizmoScale,const JointScaleRequest& jointScaling,bool structuredOvd,const std::map<uint32_t,float>& familyScales) {
    if(!std::isfinite(masterScale) || masterScale<=0)
        throw std::invalid_argument("Overlay master scale must be finite and positive");
    auto fits=[masterScale](float value){return std::isfinite(value*masterScale);};
    const float joint=jointScaling.independent?jointScaling.scale:gizmoScale;
    const float ovdJoint=jointScaling.independent?jointScaling.baseLength*std::max(jointScaling.scale,.5f):gizmoScale;
    if(!fits(gizmoScale) || !fits(joint*1.5f) || (structuredOvd && !fits(ovdJoint*1.5f)))
        throw std::invalid_argument("Master-scaled overlay size exceeds the finite float range");
    for(const auto& entry:familyScales) if(!fits(entry.second))
        throw std::invalid_argument("Master-scaled overlay family size exceeds the finite float range");
}
std::shared_ptr<OverlaySnapshot> extractOverlays(PvdEngine& engine,uint32_t mask,float gizmoScale,float toleranceSpeed,const JointScaleRequest& jointScaling,const std::map<uint32_t,float>& familyScales,float masterScale) {
    validateJointScaleRequest(jointScaling,engine.isOvdFile());
    validateFamilyScales(familyScales);
    validateMasterScale(masterScale,gizmoScale,jointScaling,engine.isOvdFile(),familyScales);
    if(mask&~uint32_t(OverlayAll)) throw std::invalid_argument("Unknown overlay category mask");
    if(!std::isfinite(gizmoScale) || gizmoScale<=0 || !std::isfinite(toleranceSpeed) || toleranceSpeed<=0)
        throw std::invalid_argument("Overlay scales must be finite and positive");
    Reader reader(engine,mask,gizmoScale,toleranceSpeed,jointScaling,familyScales,masterScale);
    if(!mask) return reader.out;
    for(const auto& i:reader.instances) {
        if(!i.mAlive || !i.mInstPtr) continue;
        const auto& cls=reader.classes[i.mClassId];
        if((mask&(OverlayBounds|OverlayCenterOfMass|OverlayAxes|OverlayVelocity|OverlayAngularVelocity)) &&
           (cls=="PxRigidDynamic" || cls=="PxRigidStatic" || cls=="PxArticulationLink")) reader.actors(i);
        if((mask&OverlayJoints) && engine.isOvdFile()) reader.joints(i);
        if((mask&OverlayVelocity) && engine.isOvdFile() && (cls=="PxDeformableVolume" || cls=="PxDeformableSurface")) reader.deformableVelocity(i);
        if(cls=="PxScene") {
            if(mask&OverlayContacts) {if(engine.isOvdFile()) reader.ovdContacts(i);else reader.legacyContacts(i);}
            if((mask&OverlayQueries) && !engine.isOvdFile()) reader.queries(i);
        }
        if((mask&(OverlayCustom|OverlayJoints)) && (cls=="PvdUserRenderer" || cls=="PvdImmediateRenderer")) reader.custom(i);
    }
    return reader.out;
}
}

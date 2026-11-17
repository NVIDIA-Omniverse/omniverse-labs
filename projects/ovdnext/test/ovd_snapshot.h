/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef OVD_SNAPSHOT_H
#define OVD_SNAPSHOT_H

// Structural snapshot of what the PVD3 renderer would draw at a given frame.
// Purpose: a deterministic, GPU-free fingerprint of a rendered frame that
// tests can diff against a golden value or compare across frames.
//
// Records contain the same data the renderer consumes:
//   - actor class, id, global pose, shape count
//   - per-shape geometry class + referenced mesh id (convex/triangle/heightfield)
//   - mesh resources that were uploaded (class, id, verts, tris)
//
// Records are sorted by instance id so output is fully deterministic for a
// given engine state.  Pose floats are emitted with fixed precision to keep
// the text representation stable across platforms.

#ifdef PVD_OVD_SUPPORT

#include <PvdEngine.h>
#include <PxPvdObjectModelBaseTypes.h>
#include <PxTransform.h>

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace ovdtest
{

struct ShapeRecord
{
    physx::pvdsdk::PxI32 mShapeId        = -1;
    physx::PxTransform   mLocalPose      = physx::PxTransform(physx::PxIdentity);
    bool                 mHasLocalPose   = false;
    std::string          mGeometryClass;
    physx::pvdsdk::PxI32 mGeometryId     = -1;
    physx::pvdsdk::PxI32 mMeshId         = -1;  // convex/triangle/heightfield ref, -1 if none
    physx::PxVec3        mGeomScale      = physx::PxVec3(1.0f);
    bool                 mHasGeomScale   = false;  // true only for mesh geometries
};

struct ActorRecord
{
    physx::pvdsdk::PxI32 mInstId = -1;
    std::string          mClass;
    physx::PxTransform   mGlobalPose;
    std::vector<ShapeRecord> mShapes;
};

struct MeshRecord
{
    physx::pvdsdk::PxI32 mInstId = -1;
    std::string          mClass;
    physx::pvdsdk::PxU32 mNbVerts = 0;
    physx::pvdsdk::PxU32 mNbTriIndices = 0;
};

struct FrameSnapshot
{
    physx::pvdsdk::PxU64    mFrame = 0;
    std::vector<ActorRecord> mActors;
    std::vector<MeshRecord>  mMeshes;
};

namespace detail
{
    inline bool isActorClass(const char* n)
    {
        return std::strcmp(n, "PxRigidDynamic") == 0
            || std::strcmp(n, "PxRigidStatic") == 0
            || std::strcmp(n, "PxArticulationLink") == 0;
    }

    inline bool isMeshClass(const char* n)
    {
        return std::strcmp(n, "PxConvexMesh") == 0
            || std::strcmp(n, "PxTriangleMesh") == 0
            || std::strcmp(n, "PxHeightField") == 0;
    }

    inline bool isMeshGeometryClass(const char* n)
    {
        return std::strcmp(n, "PxConvexMeshGeometry") == 0
            || std::strcmp(n, "PxTriangleMeshGeometry") == 0
            || std::strcmp(n, "PxHeightFieldGeometry") == 0;
    }

    inline const char* meshRefAttr(const char* geomCls)
    {
        if (std::strcmp(geomCls, "PxConvexMeshGeometry") == 0)      return "convexMesh";
        if (std::strcmp(geomCls, "PxTriangleMeshGeometry") == 0)    return "triangleMesh";
        if (std::strcmp(geomCls, "PxHeightFieldGeometry") == 0)     return "heightField";
        return nullptr;
    }
}

// Capture the structural snapshot at the engine's current frame.
inline FrameSnapshot captureFrameSnapshot(pvdapp::PvdEngine& engine, physx::pvdsdk::PxU64 frame)
{
    using namespace physx;
    using namespace physx::pvdsdk;

    FrameSnapshot snap;
    snap.mFrame = frame;

    const PxU32 nInst = engine.getNbInstances();
    std::vector<InstanceDescription> insts(nInst);
    engine.getInstances(insts.data(), nInst);

    // Build id → instance lookup for shape/mesh ref resolution.
    auto findInst = [&](PxI32 id) -> InstanceDescription* {
        for (PxU32 i = 0; i < nInst; ++i)
            if (insts[i].mId == id) return &insts[i];
        return nullptr;
    };

    for (PxU32 i = 0; i < nInst; ++i)
    {
        InstanceDescription& inst = insts[i];
        Option<ClassDescription> cls = engine.getClass(inst.mClassId);
        if (!cls.hasValue()) continue;
        const char* clsName = cls.getValue().mName.mName;

        if (detail::isMeshClass(clsName))
        {
            MeshRecord m;
            m.mInstId = inst.mId;
            m.mClass  = clsName;
            Option<PropertyDescription> vp = engine.findPropertyFlex(inst.mClassId, "verts");
            Option<PropertyDescription> tp = engine.findPropertyFlex(inst.mClassId, "tris");
            if (vp.hasValue())
                m.mNbVerts = engine.getPropertyByteSize(inst.mInstPtr, vp.getValue().mPropertyId) / sizeof(PxVec3);
            if (tp.hasValue())
                m.mNbTriIndices = engine.getPropertyByteSize(inst.mInstPtr, tp.getValue().mPropertyId) / sizeof(PxU32);
            snap.mMeshes.push_back(m);
            continue;
        }

        if (!detail::isActorClass(clsName)) continue;

        // Read global pose (OVD attr: globalPose as 7 floats)
        ActorRecord a;
        a.mInstId = inst.mId;
        a.mClass  = clsName;
        a.mGlobalPose = PxTransform(PxIdentity);
        Option<PropertyDescription> poseProp = engine.findPropertyFlex(inst.mClassId, "globalPose");
        if (poseProp.hasValue())
        {
            // PxTransform layout: PxQuat q (4 floats) followed by PxVec3 p
            // (3 floats).  OVD serializes the raw memcpy, so the wire order
            // is qx, qy, qz, qw, px, py, pz.
            float buf[7] = {0};
            engine.getPropertyValue(inst.mInstPtr, poseProp.getValue().mPropertyId, buf, sizeof(buf));
            a.mGlobalPose.q = PxQuat(buf[0], buf[1], buf[2], buf[3]);
            a.mGlobalPose.p = PxVec3(buf[4], buf[5], buf[6]);
        }

        Option<PropertyDescription> shapesProp = engine.findPropertyFlex(inst.mClassId, "shapes");
        if (shapesProp.hasValue())
        {
            const PxU32 nShapes = engine.getNbArrayElements(inst.mInstPtr, shapesProp.getValue().mPropertyId);
            if (nShapes > 0)
            {
                std::vector<PxI32> shapeIds(nShapes);
                engine.getPropertyValue(inst.mInstPtr, shapesProp.getValue().mPropertyId, shapeIds.data(), nShapes * sizeof(PxI32));
                for (PxU32 s = 0; s < nShapes; ++s)
                {
                    ShapeRecord sr;
                    sr.mShapeId = shapeIds[s];
                    InstanceDescription* shp = findInst(shapeIds[s]);
                    if (!shp) { a.mShapes.push_back(sr); continue; }

                    // Shape localPose (7 floats: translation + quat)
                    Option<PropertyDescription> lpProp = engine.findPropertyFlex(shp->mClassId, "localPose");
                    if (lpProp.hasValue())
                    {
                        float buf[7] = {0};
                        PxU32 r = engine.getPropertyValue(shp->mInstPtr, lpProp.getValue().mPropertyId, buf, sizeof(buf));
                        if (r >= sizeof(buf))
                        {
                            // Wire order matches PxTransform: quat then pos.
                            sr.mLocalPose.q = PxQuat(buf[0], buf[1], buf[2], buf[3]);
                            sr.mLocalPose.p = PxVec3(buf[4], buf[5], buf[6]);
                            sr.mHasLocalPose = true;
                        }
                    }

                    // Shape → geom
                    Option<PropertyDescription> geomProp = engine.findPropertyFlex(shp->mClassId, "geom");
                    if (geomProp.hasValue())
                    {
                        PxI32 geomId = -1;
                        engine.getPropertyValue(shp->mInstPtr, geomProp.getValue().mPropertyId, &geomId, sizeof(PxI32));
                        sr.mGeometryId = geomId;
                        InstanceDescription* gm = findInst(geomId);
                        if (gm)
                        {
                            Option<ClassDescription> gcls = engine.getClass(gm->mClassId);
                            if (gcls.hasValue())
                            {
                                sr.mGeometryClass = gcls.getValue().mName.mName;
                                if (detail::isMeshGeometryClass(sr.mGeometryClass.c_str()))
                                {
                                    const char* attr = detail::meshRefAttr(sr.mGeometryClass.c_str());
                                    Option<PropertyDescription> mrProp = engine.findPropertyFlex(gm->mClassId, attr);
                                    if (mrProp.hasValue())
                                    {
                                        PxI32 meshId = -1;
                                        engine.getPropertyValue(gm->mInstPtr, mrProp.getValue().mPropertyId, &meshId, sizeof(PxI32));
                                        sr.mMeshId = meshId;
                                    }
                                    // Geometry scale (OVD: single PxVec3 `scale` attribute,
                                    // unlike PXD2 which splits into Scale.Scale + Scale.Rotation).
                                    Option<PropertyDescription> scProp = engine.findPropertyFlex(gm->mClassId, "scale");
                                    if (scProp.hasValue())
                                    {
                                        float sbuf[3] = {1.0f, 1.0f, 1.0f};
                                        PxU32 r = engine.getPropertyValue(gm->mInstPtr, scProp.getValue().mPropertyId, sbuf, sizeof(sbuf));
                                        if (r >= sizeof(sbuf))
                                        {
                                            sr.mGeomScale = PxVec3(sbuf[0], sbuf[1], sbuf[2]);
                                            sr.mHasGeomScale = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    a.mShapes.push_back(sr);
                }
            }
        }
        snap.mActors.push_back(a);
    }

    auto byId = [](auto& a, auto& b) { return a.mInstId < b.mInstId; };
    std::sort(snap.mActors.begin(), snap.mActors.end(), byId);
    std::sort(snap.mMeshes.begin(), snap.mMeshes.end(), byId);
    return snap;
}

// Emit a stable, diff-friendly text form of the snapshot.
// Fixed 6-decimal precision; trailing zeros preserved; sorted by id.
inline std::string formatFrameSnapshot(const FrameSnapshot& s)
{
    std::ostringstream os;
    os << std::fixed << std::setprecision(6);
    os << "Frame " << s.mFrame << "\n";
    for (const ActorRecord& a : s.mActors)
    {
        os << "  Actor[" << a.mInstId << "] " << a.mClass
           << " pos=(" << a.mGlobalPose.p.x << ", " << a.mGlobalPose.p.y << ", " << a.mGlobalPose.p.z << ")"
           << " quat=(" << a.mGlobalPose.q.x << ", " << a.mGlobalPose.q.y << ", " << a.mGlobalPose.q.z << ", " << a.mGlobalPose.q.w << ")"
           << " shapes=" << a.mShapes.size() << "\n";
        for (const ShapeRecord& sr : a.mShapes)
        {
            os << "    Shape[" << sr.mShapeId << "]";
            if (sr.mHasLocalPose)
            {
                os << " localPos=(" << sr.mLocalPose.p.x << ", " << sr.mLocalPose.p.y << ", " << sr.mLocalPose.p.z << ")"
                   << " localQuat=(" << sr.mLocalPose.q.x << ", " << sr.mLocalPose.q.y << ", " << sr.mLocalPose.q.z << ", " << sr.mLocalPose.q.w << ")";
            }
            os << " geom="
               << (sr.mGeometryClass.empty() ? "<none>" : sr.mGeometryClass.c_str())
               << "(" << sr.mGeometryId << ")";
            if (sr.mHasGeomScale)
                os << " scale=(" << sr.mGeomScale.x << ", " << sr.mGeomScale.y << ", " << sr.mGeomScale.z << ")";
            if (sr.mMeshId >= 0)
                os << " mesh=" << sr.mMeshId;
            os << "\n";
        }
    }
    for (const MeshRecord& m : s.mMeshes)
    {
        os << "  Mesh[" << m.mInstId << "] " << m.mClass
           << " verts=" << m.mNbVerts << " triIdx=" << m.mNbTriIndices << "\n";
    }
    return os.str();
}

} // namespace ovdtest

#endif // PVD_OVD_SUPPORT
#endif // OVD_SNAPSHOT_H

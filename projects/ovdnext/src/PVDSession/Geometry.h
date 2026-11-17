/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pvdapp { class PvdEngine; }
namespace ovdnext {
// The PhysX capsule axis and plane normal are local +X. Matrices are row-major,
// acting on column vectors. Primitive dimensions are NOT baked into matrices;
// mesh scale (including legacy scale rotation) IS baked into matrices.
enum class GeometryKind : uint8_t { Sphere=1, Box=2, Capsule=3, Plane=4, Mesh=5, Lines=6, Points=7 };
struct GeometryMesh {
    uint64_t id = 0;               // Session-local immutable vertex revision.
    uint64_t topologyId = 0;       // Remains stable during vertex-only animation.
    int32_t sourceId = 0;
    std::vector<float> vertices;   // tightly packed xyz
    std::shared_ptr<const std::vector<uint32_t>> indices;
    std::array<float, 6> bounds;
};
struct GeometrySnapshot {
    uint64_t segmentEpoch = 0;
    std::vector<int32_t> objectIds, shapeIds, geometryIds, sceneIds, actorClasses;
    std::vector<uint8_t> kinds, sleeping, doubleSided, particleRadiusModes;
    // Radius mode: 0 non-particle, 1 legacy slider-sized billboard,
    // 2 OVD recorded physical radius (normalized against the legacy slider).
    std::vector<uint32_t> shapeFlags, actorFlags;
    std::vector<uint64_t> resourceIds;
    std::vector<float> transforms, dimensions, bounds, overrideColors;
    std::vector<std::shared_ptr<GeometryMesh>> meshes;
    // Explicit omissions, never surrogate render geometry.
    std::vector<std::string> diagnostics;
};
struct GeometryRequest {
    int32_t geometryId=0,objectId=0,sceneId=0;
    std::array<float,16> transform;
    std::array<float,4> color={{0,1,0,1}};
};
class GeometryExtractor {
public:
    GeometryExtractor();
    ~GeometryExtractor();
    std::shared_ptr<GeometrySnapshot> extract(pvdapp::PvdEngine& engine);
    std::shared_ptr<GeometrySnapshot> extractReferences(pvdapp::PvdEngine& engine,const std::vector<GeometryRequest>& requests);
private:
    struct Impl;
    std::unique_ptr<Impl> m;
};
}

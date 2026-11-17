/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Real GPU state, initialized through public SDK APIs; no authored stream bytes.
#pragma once
#include "cudamanager/PxCudaContextManager.h"
#include "extensions/PxCudaHelpersExt.h"
#include "extensions/PxParticleExt.h"
#include "extensions/PxDeformableSurfaceExt.h"
#include "extensions/PxDeformableVolumeExt.h"

struct GpuObjects
{
    PxCudaContextManager* cuda = nullptr;
    PxPBDParticleSystem* system = nullptr;
    PxParticleBuffer* buffer = nullptr;
    PxParticleAndDiffuseBuffer* diffuse = nullptr;
    PxDeformableSurface* surface = nullptr;
    PxDeformableVolume* volume = nullptr;
    std::vector<PxBase*> resources;

    ~GpuObjects() { if (cuda) cuda->release(); }
    void releaseObjects()
    {
        if (system) { system->release(); system = nullptr; }
        if (buffer) { buffer->release(); buffer = nullptr; }
        if (diffuse) { diffuse->release(); diffuse = nullptr; }
        if (surface) { surface->release(); surface = nullptr; }
        if (volume) { volume->release(); volume = nullptr; }
        for (auto it = resources.rbegin(); it != resources.rend(); ++it) (*it)->release();
        resources.clear();
    }
    void configure(PxFoundation& foundation, PxSceneDesc& desc)
    {
        cuda = PxCreateCudaContextManager(foundation, PxCudaContextManagerDesc(), PxGetProfilerCallback());
        require(cuda && cuda->contextIsValid(), "No usable GPU CUDA context");
        desc.cudaContextManager = cuda;
        desc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS | PxSceneFlag::eENABLE_PCM;
        desc.broadPhaseType = PxBroadPhaseType::eGPU;
        desc.solverType = PxSolverType::eTGS;
    }
    void particles(PxPhysics& physics, PxScene& scene)
    {
        auto* material = physics.createPBDMaterial(.05f, .05f, 0, .001f, .5f, .005f, .01f, 0, 0);
        require(material != nullptr, "PBD material allocation failed"); resources.push_back(material);
        system = physics.createPBDParticleSystem(*cuda, 96);
        require(system != nullptr, "PBD system allocation failed");
        system->setName("particles");
        system->setRestOffset(.1667f); system->setContactOffset(.18f);
        system->setParticleContactOffset(.1667f); system->setSolidRestOffset(.1667f); system->setFluidRestOffset(.1f);
        system->setMaxLinearVelocity(30.f);
        scene.addActor(*system);
        const PxU32 phase = system->createPhase(material, PxParticlePhaseFlags(
            PxParticlePhaseFlag::eParticlePhaseFluid | PxParticlePhaseFlag::eParticlePhaseSelfCollide));
        const PxU32 count = 64;
        auto* positions = PX_EXT_PINNED_MEMORY_ALLOC(PxVec4, *cuda, count);
        auto* velocities = PX_EXT_PINNED_MEMORY_ALLOC(PxVec4, *cuda, count);
        auto* phases = PX_EXT_PINNED_MEMORY_ALLOC(PxU32, *cuda, count);
        require(positions && velocities && phases, "Particle host allocation failed");
        for (PxU32 i = 0; i < count; ++i)
        {
            positions[i] = PxVec4(.2f * (i % 4), 4.f + .2f * ((i / 4) % 4), .2f * (i / 16), 1.f);
            velocities[i] = PxVec4(0, 10.f, 0, 0); phases[i] = phase;
        }
        ExtGpu::PxParticleAndDiffuseBufferDesc desc;
        desc.maxParticles = desc.numActiveParticles = count;
        desc.positions = positions; desc.velocities = velocities; desc.phases = phases;
        desc.maxDiffuseParticles = desc.maxActiveDiffuseParticles = 2048;
        desc.diffuseParams.threshold = .1f; desc.diffuseParams.kineticEnergyWeight = 100.f;
        desc.diffuseParams.pressureWeight = desc.diffuseParams.divergenceWeight = 0.f;
        desc.diffuseParams.lifetime = 10.f;
        diffuse = ExtGpu::PxCreateAndPopulateParticleAndDiffuseBuffer(desc, cuda);
        require(diffuse != nullptr, "Diffuse buffer allocation failed");
        // This pinned helper populates base arrays, not the diffuse parameters.
        diffuse->setDiffuseParticleParams(desc.diffuseParams); system->addParticleBuffer(diffuse);
        for (PxU32 i = 0; i < count; ++i) positions[i].x += 4.f;
        buffer = ExtGpu::PxCreateAndPopulateParticleBuffer(desc, cuda);
        require(buffer != nullptr, "Particle buffer allocation failed"); system->addParticleBuffer(buffer);
        PX_EXT_PINNED_MEMORY_FREE(*cuda, positions);
        PX_EXT_PINNED_MEMORY_FREE(*cuda, velocities);
        PX_EXT_PINNED_MEMORY_FREE(*cuda, phases);
    }
    void deformables(PxPhysics& physics, PxScene& scene)
    {
        PxCookingParams params(physics.getTolerancesScale()); params.buildGPUData = true;
        PxVec3 vertices[] = {PxVec3(-4, 5, 0), PxVec3(-3, 5, 0), PxVec3(-4, 5, 1)};
        PxU32 triangle[] = {0, 2, 1};
        PxTriangleMeshDesc meshDesc;
        meshDesc.points.count = 3; meshDesc.points.stride = sizeof(PxVec3); meshDesc.points.data = vertices;
        meshDesc.triangles.count = 1; meshDesc.triangles.stride = 3 * sizeof(PxU32); meshDesc.triangles.data = triangle;
        auto* mesh = PxCreateTriangleMesh(params, meshDesc, physics.getPhysicsInsertionCallback());
        require(mesh != nullptr, "Surface mesh cooking failed"); resources.push_back(mesh);
        auto* surfaceMaterial = physics.createDeformableSurfaceMaterial(1.e6f, .3f, .5f, .01f, .00001f);
        require(surfaceMaterial != nullptr, "Surface material allocation failed"); resources.push_back(surfaceMaterial);
        auto* shape = physics.createShape(PxTriangleMeshGeometry(mesh), &surfaceMaterial, 1, true);
        require(shape != nullptr, "Surface shape allocation failed"); resources.push_back(shape);
        surface = physics.createDeformableSurface(*cuda);
        require(surface && surface->attachShape(*shape), "Surface shape attachment failed");
        surface->setName("surface"); scene.addActor(*surface);
        PxVec3 velocities[] = {PxVec3(.1f, 0, 0), PxVec3(.1f, 0, 0), PxVec3(.1f, 0, 0)};
        PxVec4 *positions, *velocity, *rest;
        PxDeformableSurfaceExt::allocateAndInitializeHostMirror(*surface, vertices, velocities, vertices, .5f,
            PxTransform(PxIdentity), cuda, positions, velocity, rest);
        PxDeformableSurfaceExt::copyToDevice(*surface, PxDeformableSurfaceDataFlag::eALL, 3, positions, velocity, rest);
        PX_EXT_PINNED_MEMORY_FREE(*cuda, positions); PX_EXT_PINNED_MEMORY_FREE(*cuda, velocity); PX_EXT_PINNED_MEMORY_FREE(*cuda, rest);

        PxVec3 tetra[] = {PxVec3(-8, 5, 0), PxVec3(-7, 5, 0), PxVec3(-8, 6, 0), PxVec3(-8, 5, 1)};
        PxU32 faces[] = {0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3};
        PxSimpleTriangleMesh hull;
        hull.points.count = 4; hull.points.stride = sizeof(PxVec3); hull.points.data = tetra;
        hull.triangles.count = 4; hull.triangles.stride = 3 * sizeof(PxU32); hull.triangles.data = faces;
        auto* volumeMesh = PxDeformableVolumeExt::createDeformableVolumeMeshNoVoxels(params, hull, physics.getPhysicsInsertionCallback());
        require(volumeMesh != nullptr, "Volume mesh cooking failed"); resources.push_back(volumeMesh);
        auto* volumeMaterial = physics.createDeformableVolumeMaterial(2.e5f, .3f, .1f, .02f);
        require(volumeMaterial != nullptr, "Volume material allocation failed"); resources.push_back(volumeMaterial);
        shape = physics.createShape(PxTetrahedronMeshGeometry(volumeMesh->getCollisionMesh()), &volumeMaterial, 1, true);
        require(shape != nullptr, "Volume shape allocation failed"); resources.push_back(shape);
        volume = physics.createDeformableVolume(*cuda);
        require(volume && volume->attachShape(*shape) && volume->attachSimulationMesh(*volumeMesh->getSimulationMesh(),
            *volumeMesh->getDeformableVolumeAuxData()), "Volume attachment failed");
        volume->setName("volume"); scene.addActor(*volume);
        PxVec4* collision;
        PxDeformableVolumeExt::allocateAndInitializeHostMirror(*volume, cuda, positions, velocity, collision, rest);
        PxDeformableVolumeExt::updateMass(*volume, 100.f, 50.f, positions);
        PxDeformableVolumeExt::copyToDevice(*volume, PxDeformableVolumeDataFlag::eALL, positions, velocity, collision, rest);
        PX_EXT_PINNED_MEMORY_FREE(*cuda, positions); PX_EXT_PINNED_MEMORY_FREE(*cuda, velocity);
        PX_EXT_PINNED_MEMORY_FREE(*cuda, collision); PX_EXT_PINNED_MEMORY_FREE(*cuda, rest);
    }
    void initialize(PxPhysics& physics, PxScene& scene)
    {
        particles(physics, scene); deformables(physics, scene);
        for (int i = 0; i < 60 && !diffuse->getNbActiveDiffuseParticles(); ++i)
        {
            scene.simulate(1.f / 60.f); scene.fetchResults(true); scene.fetchResultsParticleSystem();
        }
        require(diffuse->getNbActiveDiffuseParticles() > 0, "Fixture did not generate real diffuse particles");
    }
};

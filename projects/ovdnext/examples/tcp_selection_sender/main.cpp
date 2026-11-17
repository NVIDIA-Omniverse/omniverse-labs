/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Genuine SDK integration fixture; all producer mutations run at idle points on this thread.
#include "PxConfig.h"
#include "PxPhysicsAPI.h"
#include "omnipvd/PxOmniPvd.h"
#include "OmniPvdLibraryFunctions.h"
#include "OmniPvdSocketWriteStream.h"
#include "OmniPvdFileReadStream.h"
#include "OmniPvdReader.h"
#include "OmniPvdWriter.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace physx;

static void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

#if OVDNEXT_SENDER_GPU
#include "gpu_scene.h"
#endif

template<typename T> static void array(std::ostream& out, const std::vector<T>& values)
{
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) out << (i ? "," : "") << values[i];
    out << ']';
}

struct CountedErrors : PxErrorCallback
{
    std::atomic<unsigned> diagnostics{0};
    PxDefaultErrorCallback sink;
    void reportError(PxErrorCode::Enum code, const char* message, const char* file, int line) override
    {
        if (code != PxErrorCode::eNO_ERROR && code != PxErrorCode::eDEBUG_INFO) ++diagnostics;
        sink.reportError(code, message, file, line);
    }
};

struct Runtime
{
    PxDefaultAllocator allocator;
    CountedErrors errors;
    PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
    PxOmniPvd* ovd = foundation ? PxCreateOmniPvd(*foundation) : nullptr;
    ~Runtime() { if (ovd) ovd->release(); if (foundation) foundation->release(); }
};

static void describe()
{
    Runtime runtime;
    require(runtime.ovd != nullptr, "SDK was not built OVD capable");
    std::cout << "{\"ovd_capable\":true,\"wire\":[" << OMNI_PVD_VERSION_MAJOR << ','
              << OMNI_PVD_VERSION_MINOR << ',' << OMNI_PVD_VERSION_PATCH << "],\"integration\":["
              << PX_PHYSICS_OVD_INTEGRATION_VERSION_MAJOR << ',' << PX_PHYSICS_OVD_INTEGRATION_VERSION_MINOR
              << "],\"gpu_capable\":" << (OVDNEXT_SENDER_GPU ? "true" : "false") << "}\n";
}

static void quoted(std::ostream& out, const std::string& value)
{
    const char* hex = "0123456789abcdef";
    out << '"';
    for (unsigned char c : value)
    {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u00" << hex[c >> 4] << hex[c & 15];
        else out << c;
    }
    out << '"';
}

struct ObservedObject
{
    std::string type;
    std::map<std::string, std::vector<std::string>> attributes;
};

static void inspect(const char* path)
{
    auto* stream = createOmniPvdFileReadStream();
    auto* reader = createOmniPvdReader();
    require(stream && reader, "Reader allocation failed");
    stream->setFileName(path);
    require(stream->openStream(), "Cannot open captured OVD");
    reader->setReadStream(*stream);
    std::set<OmniPvdClassHandle> classes;
    std::set<OmniPvdAttributeHandle> attributes;
    std::vector<OmniPvdClassHandle> metadata, catalogs;
    std::vector<OmniPvdObjectHandle> instances;
    std::map<OmniPvdClassHandle, std::string> classNames;
    std::map<OmniPvdAttributeHandle, std::string> attributeNames;
    std::vector<std::map<OmniPvdObjectHandle, ObservedObject>> segments;
    size_t duplicateClasses = 0, duplicateAttributes = 0, frames = 0;
    OmniPvdCommand::Enum command;
    while ((command = reader->getNextCommand()) != OmniPvdCommand::eINVALID)
    {
        if (command == OmniPvdCommand::eREGISTER_CLASS)
        {
            const auto handle = reader->getClassHandle();
            duplicateClasses += !classes.insert(handle).second;
            const std::string name(reader->getClassName());
            classNames[handle] = name;
            if (name == "PxOmniPvdMetaData") metadata.push_back(handle);
            if (name == "PxOmniPvdCaptureEntry") catalogs.push_back(handle);
        }
        if (command == OmniPvdCommand::eREGISTER_ATTRIBUTE || command == OmniPvdCommand::eREGISTER_CLASS_ATTRIBUTE ||
            command == OmniPvdCommand::eREGISTER_UNIQUE_LIST_ATTRIBUTE)
        {
            duplicateAttributes += !attributes.insert(reader->getAttributeHandle()).second;
            attributeNames[reader->getAttributeHandle()] = classNames.at(reader->getClassHandle()) + "." + reader->getAttributeName();
        }
        if (command == OmniPvdCommand::eCREATE_OBJECT &&
            std::find(metadata.begin(), metadata.end(), reader->getClassHandle()) != metadata.end())
        {
            instances.push_back(reader->getObjectHandle());
            segments.emplace_back();
        }
        // Decode actual SDK payloads independently of the viewer DOM/selection model.
        if (!segments.empty() && reader->getContextHandle() == 1)
        {
            if (command == OmniPvdCommand::eCREATE_OBJECT)
                segments.back()[reader->getObjectHandle()].type = classNames.at(reader->getClassHandle());
            if (command == OmniPvdCommand::eSET_ATTRIBUTE || command == OmniPvdCommand::eADD_TO_UNIQUE_LIST_ATTRIBUTE ||
                command == OmniPvdCommand::eREMOVE_FROM_UNIQUE_LIST_ATTRIBUTE)
            {
                const char* hex = "0123456789abcdef";
                std::string value;
                const auto* data = reader->getAttributeDataPointer();
                for (uint32_t i = 0; i < reader->getAttributeDataLength(); ++i)
                { value += hex[data[i] >> 4]; value += hex[data[i] & 15]; }
                const std::string prefix = command == OmniPvdCommand::eSET_ATTRIBUTE ? "" :
                    command == OmniPvdCommand::eADD_TO_UNIQUE_LIST_ATTRIBUTE ? "add:" : "remove:";
                segments.back()[reader->getObjectHandle()].attributes[prefix + attributeNames.at(reader->getAttributeHandle())].push_back(value);
            }
        }
        frames += command == OmniPvdCommand::eSTOP_FRAME;
    }
    const auto status = getOmniPvdReaderStatus(*reader);
    std::cout << "{\"reader_eof\":" << (status.code == OmniPvdReaderStatus::eEND_OF_STREAM ? "true" : "false")
              << ",\"classes\":" << classes.size() << ",\"attributes\":" << attributes.size()
              << ",\"duplicate_classes\":" << duplicateClasses << ",\"duplicate_attributes\":" << duplicateAttributes
              << ",\"stop_frames\":" << frames << ",\"metadata_class_handles\":";
    array(std::cout, metadata);
    std::cout << ",\"catalog_class_handles\":"; array(std::cout, catalogs);
    std::cout << ",\"metadata_instances\":"; array(std::cout, instances);
    std::cout << ",\"segments\":[";
    for (size_t i = 0; i < segments.size(); ++i)
    {
        std::cout << (i ? ",{": "{");
        bool first = true;
        for (const auto& object : segments[i])
        {
            if (!first) std::cout << ','; first = false;
            quoted(std::cout, std::to_string(object.first)); std::cout << ":{\"type\":";
            quoted(std::cout, object.second.type); std::cout << ",\"attributes\":{";
            bool firstAttribute = true;
            for (const auto& attribute : object.second.attributes)
            {
                if (!firstAttribute) std::cout << ','; firstAttribute = false;
                quoted(std::cout, attribute.first); std::cout << ":[";
                for (size_t j = 0; j < attribute.second.size(); ++j)
                { if (j) std::cout << ','; quoted(std::cout, attribute.second[j]); }
                std::cout << ']';
            }
            std::cout << "}}";
        }
        std::cout << '}';
    }
    std::cout << ']';
    std::cout << "}\n";
    destroyOmniPvdReader(*reader);
    destroyOmniPvdFileReadStream(*stream);
    require(status.code == OmniPvdReaderStatus::eEND_OF_STREAM, "Captured stream did not decode to EOF");
}

struct CaptureFilter : PxOmniPvdCaptureFilter
{
    std::string mode = "allow";
    const PxBase* target = nullptr;
    bool filterCaptureRequest(PxOmniPvdFilterDesc& desc) override
    {
        if (mode == "deny") return false;
        if (mode == "narrow" || mode == "escape")
        {
            desc.scenes = nullptr; desc.nbScenes = 0;
            desc.objects = &target; desc.nbObjects = 1;
            desc.recordSecondaryData = false;
        }
        return true;
    }
};

struct Scene
{
    Runtime runtime;
#if OVDNEXT_SENDER_GPU
    GpuObjects gpu;
#endif
    OmniPvdSocketWriteStream* stream = nullptr;
    PxPhysics* physics = nullptr;
    PxDefaultCpuDispatcher* dispatcher = nullptr;
    PxScene* scene = nullptr;
    PxScene* otherScene = nullptr;
    PxMaterial* material = nullptr;
    PxRigidDynamic* box = nullptr;
    PxArticulationReducedCoordinate* arm = nullptr;
    CaptureFilter filter;
    unsigned frames = 0, starts = 0;

    ~Scene()
    {
        if (physics) runtime.ovd->stopSampling();
#if OVDNEXT_SENDER_GPU
        gpu.releaseObjects();
#endif
        if (arm) arm->release();
        if (scene) scene->release();
        if (otherScene) otherScene->release();
        if (material) material->release();
        if (physics) physics->release();
        if (dispatcher) dispatcher->release();
        if (runtime.ovd) { runtime.ovd->release(); runtime.ovd = nullptr; }
        if (stream) destroyOmniPvdSocketWriteStream(*stream);
    }

    void connect(const char* host, unsigned port, const std::string& scenario)
    {
        require(runtime.ovd != nullptr, "SDK is not OVD capable");
        physics = PxCreatePhysics(PX_PHYSICS_VERSION, *runtime.foundation, PxTolerancesScale(), true, nullptr, runtime.ovd);
        require(physics != nullptr, "Cannot create PhysX");
        dispatcher = PxDefaultCpuDispatcherCreate(1);
        PxSceneDesc desc(physics->getTolerancesScale());
        desc.gravity = PxVec3(0, -9.81f, 0);
        desc.cpuDispatcher = dispatcher;
        desc.filterShader = PxDefaultSimulationFilterShader;
#if OVDNEXT_SENDER_GPU
        if (scenario == "gpu-families") gpu.configure(*runtime.foundation, desc);
#endif
        scene = physics->createScene(desc);
        require(scene != nullptr, "Cannot create scene");
        scene->setName("selection fixture");
        material = physics->createMaterial(0.5f, 0.5f, 0.1f);
        auto* ground = PxCreatePlane(*physics, PxPlane(0, 1, 0, 0), *material);
        ground->setName("ground"); scene->addActor(*ground);
        box = PxCreateDynamic(*physics, PxTransform(PxVec3(0, scenario == "schema-lifetimes" ? 2.0f : 0.5f, 0)),
                              PxBoxGeometry(0.5f, 0.5f, 0.5f), *material, 1.0f);
        box->setName("box"); scene->addActor(*box);
        filter.target = ground;
        if (scenario != "schema-lifetimes" && scenario != "gpu-families")
        {
            otherScene = physics->createScene(desc);
            require(otherScene != nullptr, "Cannot create second scene");
            otherScene->setName("other scene");
            auto* other = PxCreateDynamic(*physics, PxTransform(PxVec3(0, 4, 0)), PxSphereGeometry(.5f), *material, 1.0f);
            other->setName("other box"); otherScene->addActor(*other);
            arm = physics->createArticulationReducedCoordinate();
            arm->setName("arm"); arm->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
            auto* root = arm->createLink(nullptr, PxTransform(PxVec3(10, 3, 0)));
            auto* link = arm->createLink(root, PxTransform(PxVec3(10, 4, 0)));
            root->setName("arm root"); link->setName("arm link");
            PxRigidActorExt::createExclusiveShape(*root, PxBoxGeometry(.2f, .2f, .2f), *material);
            PxRigidActorExt::createExclusiveShape(*link, PxBoxGeometry(.2f, .2f, .2f), *material);
            PxRigidBodyExt::updateMassAndInertia(*root, 1.0f);
            PxRigidBodyExt::updateMassAndInertia(*link, 1.0f);
            auto* joint = link->getInboundJoint();
            joint->setJointType(PxArticulationJointType::eREVOLUTE);
            joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
            joint->setParentPose(PxTransform(PxVec3(0, .5f, 0)));
            joint->setChildPose(PxTransform(PxVec3(0, -.5f, 0)));
            scene->addArticulation(*arm);
            joint->setJointVelocity(PxArticulationAxis::eTWIST, .25f);
        }
#if OVDNEXT_SENDER_GPU
        if (scenario == "gpu-families") gpu.initialize(*physics, *scene);
#endif
        stream = createOmniPvdSocketWriteStream(host, static_cast<uint16_t>(port), 3000);
        require(stream && stream->openStream(), "Cannot connect genuine SDK socket writer");
        runtime.ovd->getWriter()->setWriteStream(*stream);
        PxOmniPvdFilterDesc scope;
        const PxBase* scoped = box; scope.objects = &scoped; scope.nbObjects = 1;
        require(runtime.ovd->startCaptureDiscovery(scenario == "scoped" ? &scope : nullptr), "Cannot start capture discovery");
        require(stream->flush(), "Cannot flush discovery");
    }

    void command(const std::string& line)
    {
        if (line == "start" || line == "restart")
        {
            if (line == "restart") require(runtime.ovd->stopSampling(), "Cannot stop recording");
            require(runtime.ovd->startSampling(), "Cannot start sampling on bound stream");
            ++starts;
        }
        else if (line == "step")
        {
            scene->simulate(1.0f / 60.0f); scene->fetchResults(true); ++frames;
#if OVDNEXT_SENDER_GPU
            if (gpu.cuda) scene->fetchResultsParticleSystem();
#endif
            if (otherScene) { otherScene->simulate(1.0f / 60.0f); otherScene->fetchResults(true); }
        }
        else if (line == "drop-and-restart")
        {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
            while (!runtime.ovd->hasPendingCaptureRequest())
            {
                require(std::chrono::steady_clock::now() < deadline, "No pending request to drop");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            require(runtime.ovd->stopSampling(), "Cannot discard pending request");
            require(runtime.ovd->startSampling(), "Cannot locally restart after discarded request");
            ++starts;
        }
        else if (line == "poll") require(runtime.ovd->processCaptureRequests(&filter), "Capture processing failed");
        else if (line == "busy")
        {
            scene->simulate(1.0f / 60.0f);
            const bool processed = runtime.ovd->processCaptureRequests(&filter);
            scene->fetchResults(true); ++frames;
            require(processed, "Busy acknowledgement failed");
        }
        else if (line == "force") { require(box != nullptr, "Box retired"); box->addForce(PxVec3(3, 0, 0)); }
        else if (line == "retire") { require(box != nullptr, "Box already retired"); box->release(); box = nullptr; }
        else if (line == "allow" || line == "deny" || line == "narrow" || line == "escape") filter.mode = line;
        else throw std::runtime_error("Unknown stdin command");
        require(stream->flush(), "Cannot flush sender");
    }
};

int main(int argc, char** argv)
{
    try
    {
        if (argc == 2 && std::string(argv[1]) == "--describe") { describe(); return 0; }
        if (argc == 3 && std::string(argv[1]) == "--inspect") { inspect(argv[2]); return 0; }
        std::map<std::string, std::string> args;
        for (int i = 1; i + 1 < argc; i += 2) args.emplace(argv[i], argv[i + 1]);
        const auto scenario = args["--scenario"];
        require(args.count("--host") && args.count("--port") && args.count("--report") &&
                (scenario == "schema-lifetimes" || scenario == "cpu-selection" || scenario == "scoped" ||
                 (OVDNEXT_SENDER_GPU && scenario == "gpu-families")),
                "Expected host, port, report and a supported scenario");
        const auto port = std::stoul(args["--port"]);
        require(port > 0 && port <= 65535, "Invalid TCP port");
        Scene fixture;
        fixture.connect(args["--host"].c_str(), static_cast<unsigned>(port), scenario);
        std::string line;
        while (std::getline(std::cin, line) && line != "quit") fixture.command(line);
        require(fixture.runtime.ovd->stopSampling(), "Cannot stop fixture");
        require(fixture.stream->flush(), "Cannot flush final bytes");
        fixture.stream->closeStream();
        std::ofstream report(args["--report"]);
        report << "{\"scenario\":\"" << scenario << "\",\"frames\":" << fixture.frames
               << ",\"starts\":" << fixture.starts << ",\"sdk_diagnostics\":" << fixture.runtime.errors.diagnostics.load()
               << ",\"sender_revision\":\"ad5c86f498025752261284367f0f1908d392a6b1\"";
#if OVDNEXT_SENDER_GPU
        if (fixture.gpu.cuda)
        {
            report << ",\"gpu_device\":"; quoted(report, fixture.gpu.cuda->getDeviceName());
            report << ",\"cuda_driver_version\":" << fixture.gpu.cuda->getDriverVersion();
        }
#endif
        report << "}\n";
        require(bool(report), "Cannot write fixture report");
        return 0;
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

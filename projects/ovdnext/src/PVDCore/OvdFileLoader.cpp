/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifdef PVD_OVD_SUPPORT

#include "OvdFileLoader.h"
#include "OvdDomAdapter.h"
#include "OvdInputValidation.h"
#include "OvdCaptureCatalog.h"
#include "OvdMessageCollector.h"
#include "PvdDomParser.h"
#include "OvdLog.h"
#include "OmniPvdReadStream.h"
#include "OmniPvdReader.h"

#include <cstring>
#include <array>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#ifdef _WIN32
    #include <string.h>
    #define PVD_STRICMP _stricmp
#else
    #include <strings.h>
    #define PVD_STRICMP strcasecmp
#endif

extern "C" {
    OmniPvdReader* OMNI_PVD_CALL createOmniPvdReader();
    void OMNI_PVD_CALL destroyOmniPvdReader(OmniPvdReader& reader);
}

namespace {
// The pinned runtime's file stream uses narrow fopen. Decode our UTF-8 path
// here while retaining its public byte-stream interface and native validator.
class NativeFileReadStream : public OmniPvdReadStream {
public:
    explicit NativeFileReadStream(const char* path) : mPath(std::filesystem::u8path(path)) {}
    bool OMNI_PVD_CALL openStream() override {
        if(mFile.is_open()) return true;
        mFile.clear();
        mFile.open(mPath, std::ios::binary);
        return mFile.is_open();
    }
    bool OMNI_PVD_CALL closeStream() override {
        if(!mFile.is_open()) return true;
        mFile.clear();
        mFile.close();
        return !mFile.fail();
    }
    uint64_t OMNI_PVD_CALL readBytes(uint8_t* bytes, uint64_t count) override {
        if(!mFile.is_open() || !bytes || !count) return 0;
        const auto bounded = static_cast<std::streamsize>(std::min<uint64_t>(
            count, static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())));
        mFile.read(reinterpret_cast<char*>(bytes), bounded);
        return static_cast<uint64_t>(mFile.gcount());
    }
    uint64_t OMNI_PVD_CALL skipBytes(uint64_t count) override {
        std::array<uint8_t, 64 * 1024> scratch;
        uint64_t skipped = 0;
        while(skipped < count) {
            const auto chunk = std::min<uint64_t>(count - skipped, scratch.size());
            const auto actual = readBytes(scratch.data(), chunk);
            skipped += actual;
            if(actual != chunk) break;
        }
        return skipped;
    }
private:
    std::filesystem::path mPath;
    std::ifstream mFile;
};
}

bool pvdapp::hasOvdExtension(const char* filename)
{
    if (!filename) return false;
    size_t len = strlen(filename);
    if (len < 4) return false;
    const char* ext = filename + len - 4;
    return (PVD_STRICMP(ext, ".ovd") == 0);
}

std::unique_ptr<pvdapp::OvdDomAdapter> pvdapp::loadOvdFile(const char* ovdFilePath)
{
    return loadOvdFile(ovdFilePath, OvdResourceLimits::defaults(), nullptr);
}

std::unique_ptr<pvdapp::OvdDomAdapter> pvdapp::loadOvdFile(
    const char* ovdFilePath, const OvdResourceLimits& limits, OvdInputErrorCode* errorCode, LoadControl* control)
{
    if (errorCode) *errorCode = OvdInputErrorCode::None;
    if (control && control->cancelled()) return nullptr;
    uint64_t totalBytes = 0;
    if (control) {
        std::ifstream measured(std::filesystem::u8path(ovdFilePath), std::ios::binary | std::ios::ate);
        const auto size = measured.tellg();
        totalBytes = size > 0 ? static_cast<uint64_t>(size) : 0;
        control->update("Reading OVD commands", 0, totalBytes);
    }
    clearOvdParserMessages();
    std::vector<OmniPvdMessage> messages;
    ovdLog("[OVD] Loading: %s\n", ovdFilePath);

    auto domState = std::make_unique<OmniPvdDOMState>();
    NativeFileReadStream stream(ovdFilePath);
    OmniPvdReader* reader = createOmniPvdReader();
    if (!reader)
    {
        if (errorCode) *errorCode = OvdInputErrorCode::ReaderRejectedCommand;
        return nullptr;
    }

    if (!stream.openStream())
    {
        destroyOmniPvdReader(*reader);
        if (errorCode) *errorCode = OvdInputErrorCode::ReaderRejectedCommand;
        return nullptr;
    }

    OvdCaptureCatalog catalog;
    OvdValidatedReadStream validated(stream, limits, CatalogValidationMode::SeparateNonRetained, &catalog);
    reader->setReadStream(validated);
    OmniPvdVersionType major = 0, minor = 0, patch = 0;
    bool ok = reader->startReading(major, minor, patch);
    OvdInputErrorCode failure = OvdInputErrorCode::None;
    if (!ok)
        failure = validated.failed() ? validated.errorCode() : OvdInputErrorCode::ReaderRejectedCommand;

    uint64_t commands = 0;
    OvdEvidenceFrameScopes evidence;evidence.complete=true;
    std::map<std::pair<uint64_t,uint64_t>,size_t> scopeIndex;
    while (ok)
    {
        if (control && control->cancelled()) break;
        const OmniPvdCommand::Enum command = reader->getNextCommand();
        if (command == OmniPvdCommand::eINVALID)
        {
            if (validated.failed())
                failure = validated.errorCode();
            else if (!validated.cleanEof())
                failure = OvdInputErrorCode::ReaderRejectedCommand;
            break;
        }
        if (!validated.validateCommand(command, *reader))
        {
            failure = validated.errorCode();
            break;
        }
        if (validated.commandDestination() == OvdCommandDestination::Dom)
            applyOvdDomCommandAndCollect(*reader, *domState, command, messages);
        if(command==OmniPvdCommand::eSTART_FRAME) {
            const auto context=reader->getContextHandle(),frame=reader->getFrameTimeStart();
            const auto key=std::make_pair(domState->mCurrentRecordingSegmentId,context);
            auto found=scopeIndex.find(key);
            if(found==scopeIndex.end() && evidence.scopes.size()>=4096)evidence.complete=false;
            else {
                if(found==scopeIndex.end()) {
                    scopeIndex.emplace(key,evidence.scopes.size());
                    OvdEvidenceFrameScope scope;scope.segment=key.first;scope.context=context;
                    evidence.scopes.push_back(scope);found=scopeIndex.find(key);
                }
                auto& scope=evidence.scopes[found->second];
                // Legacy START broadcasts to every scene; a coincidentally equal
                // handle does not establish a context-aware frame relation.
                uint64_t sceneUid=0;
                const bool contextAware=domState->mStreamOvdIntegVersionMajor>1 ||
                    (domState->mStreamOvdIntegVersionMajor==1 && domState->mStreamOvdIntegVersionMinor>=4);
                const auto handle=domState->mExternalToInternalHandleMap.find(context);
                if(contextAware && handle!=domState->mExternalToInternalHandleMap.end()) {
                    const auto object=domState->mObjectHandleToObjectMap.find(handle->second);
                    if(object!=domState->mObjectHandleToObjectMap.end() && object->second &&
                        object->second->mOmniPvdClass && object->second->mOmniPvdClass->mClassName=="PxScene")
                        sceneUid=object->second->mUID;
                }
                if(!scope.count){scope.first=frame;scope.sceneUid=sceneUid;scope.sceneMapped=sceneUid!=0;}
                else {
                    if(frame<=scope.last)scope.monotonic=false;
                    if(frame<=scope.last || frame-scope.last!=1)scope.consecutive=false;
                    if(sceneUid!=scope.sceneUid)scope.sceneRecreated=true;
                    scope.sceneMapped=scope.sceneMapped && sceneUid!=0;
                }
                scope.last=frame;++scope.count;
            }
        }
        if (control && (++commands % 256 == 0))
            control->update("Reading OVD commands", validated.streamBytes(), totalBytes);
        if (domState->mOvdIntegVersionWasChecked && !domState->mOvdIntegVersionPassed)
        {
            failure = OvdInputErrorCode::UnsupportedIntegrationVersion;
            break;
        }
    }

    const bool cancelled = control && control->cancelled();
    if (!cancelled && failure == OvdInputErrorCode::None && domState->mExternalToInternalHandleMap.empty())
        failure = OvdInputErrorCode::EmptyDom;

    if (control) control->update("Validating OVD topology", validated.streamBytes(), totalBytes, false);
    if (!cancelled && failure == OvdInputErrorCode::None && !validateOvdDomTopology(*domState))
        failure = OvdInputErrorCode::InvalidDomTopology;

    stream.closeStream();
    destroyOmniPvdReader(*reader);

    if (control && control->cancelled()) {
        clearOvdParserMessages();
        return nullptr;
    }

    if (failure != OvdInputErrorCode::None)
    {
        ovdLog("[OVD] validation failed: %s\n", ovdInputErrorMessage(failure));
        clearOvdParserMessages();
        if (errorCode) *errorCode = failure;
        return nullptr;
    }

    ovdLog("[OVD] DOM parsed: %zu classes, %zu objects, frames %llu-%llu, %zu messages\n",
        domState->mClassHandleToClassMap.size(),
        domState->mObjectHandleToObjectMap.size(),
        (unsigned long long)domState->mMinFrame,
        (unsigned long long)domState->mMaxFrame,
        messages.size());

    if (control) control->update("Indexing OVD capture", 0, 0, false);
    auto adapter = std::make_unique<OvdDomAdapter>(std::move(domState), std::move(messages),std::move(evidence));
    if (control && control->cancelled()) return nullptr;
    if (!adapter->isValid())
    {
        if (errorCode) *errorCode = adapter->errorCode();
        return nullptr;
    }
    ovdLog("[OVD] Adapter created: %u classes, %u instances, %zu messages\n",
        adapter->getNbClasses(), adapter->getNbInstances(), adapter->getMessages().size());

    if (control) control->update("Capture parsed", totalBytes, totalBytes);
    return adapter;
}

#endif // PVD_OVD_SUPPORT

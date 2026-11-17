/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "PvdEngine.h"
#include "PvdInputValidation.h"
#include "PsArray.h"
#include "PvdRepXExport.h"
#include "PxFoundation.h"
#include <stdexcept>
#include <memory>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <cctype>

#include "profiler.h"

#ifdef PVD_OVD_SUPPORT
#include "OvdFileLoader.h"
#include "OvdDomAdapter.h"
#include "OvdLiveSource.h"
#include "OvdLog.h"
#include <mutex>
#endif

using namespace pvdapp;

PvdEngine::PvdEngine() {}

PvdEngine::~PvdEngine()
{
#ifdef PVD_OVD_SUPPORT
	// Stop the live ingest worker before anything else: it writes into the
	// adapter's DOM under the adapter's mutex, so it must be joined while both
	// are still alive. (m_live is also declared after m_ovdAdapter, so the
	// implicit member teardown order already does this; this is explicit and
	// idempotent.)
	if (m_live)
		m_live->stop();
	m_live.reset();
#endif
	if(m_loader)
	{
		// The stream loader owns both the source stream and the client/backing
		// stream. Releasing the client directly would leave the loader holding a
		// dangling reference and would leak the loader's source stream.
		m_loader->release();
		m_loader = nullptr;
		m_client = nullptr;
	}
	else if(m_client)
		m_client->release();
}

// ---------------------------------------------------------------------------
// Frame navigation
// ---------------------------------------------------------------------------

PxU32 PvdEngine::getFrameCount(PxU64 sceneId)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->getFrameCount();
#endif
	return m_client->getFrameCount(sceneId);
}

PxU32 PvdEngine::getCurrentFrame(PxU64 sceneId)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->getCurrentFrame();
#endif
	return m_client->getCurrentFrame(sceneId);
}

void PvdEngine::setCurrentFrame(PxU32 frame, PxU64 sceneId)
{
#ifdef PROFILING
	profiler::ProfilerScope prof_scope("PvdEngine::setCurrentFrame()", 0);
#endif
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) { m_ovdAdapter->setCurrentFrame(frame); return; }
#endif
	m_client->setCurrentFrame(frame, sceneId);
}

PxU64 PvdEngine::getMaxFrame() const
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->getMaxFrame();
#endif
	return 0;
}

// ---------------------------------------------------------------------------
// Event stream (PXD2 only)
// ---------------------------------------------------------------------------

PxU64 PvdEngine::getEventCount()
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd) return 0;
#endif
	return m_client->getEventCount();
}

PxI64 PvdEngine::getCurrentEvent()
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd) return 0;
#endif
	return m_client->getCurrentEvent();
}

PxI64 PvdEngine::getInnerFrameTime()
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd) return 0;
#endif
	PxU64 event = m_client->getCurrentEvent();
	return m_streamMeta->getInnerFrameTime(event);
}

PxI64 PvdEngine::getTotalFrameTime()
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd) return 0;
#endif
	PxU64 event = m_client->getCurrentEvent();
	return m_streamMeta->getTotalFrameTime(event);
}

PvdObjectModelEventStream* PvdEngine::getEventStream()
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd) return nullptr;
#endif
	return m_client->getEventStream();
}

PxI64 PvdEngine::findNextFrameEvent(PxI64 inStartEvent, PxU64 sceneId)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd) return 0;
#endif
	return m_streamMeta->findNextFrameEvent(inStartEvent, sceneId);
}

PxI64 PvdEngine::findPreviousFrameEvent(PxI64 inStartEvent, PxU64 sceneId)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd) return 0;
#endif
	return m_streamMeta->findPreviousFrameEvent(inStartEvent, sceneId);
}

void PvdEngine::getTimestampNanosecondConversion(PxU64& numerator, PxU64& denominator)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd) { numerator = 1; denominator = 1; return; }
#endif
	return m_streamMeta->getTimestampNanosecondConversion(numerator, denominator);
}

// ---------------------------------------------------------------------------
// Profile / Memory events (PXD2 only)
// ---------------------------------------------------------------------------

void PvdEngine::parseProfileEvents(PxI64 startEvent, PxI64 endEvent, int profileZone, profile::PxProfileBulkEventHandler& handler)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd) return;
#endif
	m_client->parseProfileEvents(startEvent, endEvent, profileZone, handler);
}

void PvdEngine::parseMemoryEvents(PxI64 startEvent, PxI64 endEvent, profile::PxProfileBulkMemoryEventHandler& handler)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd) return;
#endif
	m_client->parseMemoryEvents(startEvent, endEvent, handler);
}

PxI32 PvdEngine::getMemoryEventBufferInstance() const
{
	if (m_isOvd || !m_streamMeta) return -1;
	return m_streamMeta->getMemoryEventBufferInstance();
}

PxU32 PvdEngine::getNbMemoryEventEntries()
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd) return 0;
#endif
	return m_streamMeta->getNbMemoryEventEntries();
}

PxU32 PvdEngine::getMemoryEventEntries(PvdMemoryEventEntry* buffer, PxU32 bufSize, PxU32 idx)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd) return 0;
#endif
	return m_streamMeta->getMemoryEventEntries(buffer, bufSize, idx);
}

PxU32 PvdEngine::getNbMemoryTotals()
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd) return 0;
#endif
	return m_client->getNbMemoryTotals();
}

PxU32 PvdEngine::getMemotyEventTotals(PxI32* buffer, PxU32 bufSize)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd) return 0;
#endif
	return m_client->getMemoryTotals(buffer, bufSize);
}

PxU32 PvdEngine::getNbProfileZones()
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd) return 0;
#endif
	return m_streamMeta->getNbProfileZones();
}

PxU32 PvdEngine::getProfileZones(PvdProfileZone* buffer, PxU32 bufSize, PxU32 idx)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd) return 0;
#endif
	return m_streamMeta->getProfileZones(buffer, bufSize, idx);
}

// ---------------------------------------------------------------------------
// Class metadata
// ---------------------------------------------------------------------------

PxU32 PvdEngine::getNbClasses() const
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->getNbClasses();
#endif
	return m_meta->getNbClasses();
}

PxU32 PvdEngine::getClasses(ClassDescription* outClasses, PxU32 requestCount, PxU32 startIndex)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->getClasses(outClasses, requestCount, startIndex);
#endif
	return m_meta->getClasses(outClasses, requestCount, startIndex);
}

PxU32 PvdEngine::getNbProperties(const PxI32 classId) const
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->getNbProperties(classId);
#endif
	return m_meta->getNbProperties(classId);
}

PxU32 PvdEngine::getProperties(const PxI32 classId, PropertyDescription* outBuffer, PxU32 bufCount, PxU32 startIdx) const
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->getProperties(classId, outBuffer, bufCount, startIdx);
#endif
	return m_meta->getProperties(classId, outBuffer, bufCount, startIdx);
}

// ---------------------------------------------------------------------------
// Instance queries
// ---------------------------------------------------------------------------

PxU32 PvdEngine::getNbInstances() const
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->getNbInstances();
#endif
	return m_reader->getNbInstances();
}

PxU32 PvdEngine::getInstances(InstanceDescription* outBuffer, PxU32 count, PxU32 startIndex) const
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->getInstances(outBuffer, count, startIndex);
#endif
	return m_reader->getInstances(outBuffer, count, startIndex);
}

PxU32 PvdEngine::getNbScenes() const
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->getNbScenes();
#endif
	return m_streamMeta->getNbSectionLists();
}

PxU32 PvdEngine::getScenes(SectionListInfo* outBuffer, PxU32 count, PxU32 startIndex) const
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->getScenes(outBuffer, count, startIndex);
#endif
	return m_streamMeta->getSectionLists(outBuffer, count, startIndex);
}

bool PvdEngine::isTopLevel(PxI32 instanceId)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->isTopLevel(instanceId);
#endif
	return m_streamMeta->isTopLevel(instanceId);
}

PxI32 PvdEngine::getOwningSceneId(PxI32 instanceId)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->getOwningSceneId(instanceId);
#endif
	// PXD2 has no equivalent DOM-tree scene ownership; -1 = "no owning scene"
	// so the visibility check degrades to own-id-only (legacy behavior).
	return -1;
}

PxI32 PvdEngine::getOwningPhysicsId(PxI32 instanceId)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->getOwningPhysicsId(instanceId);
#endif
	return -1;
}

bool PvdEngine::isOvdSceneJoint(PxI32 instanceId)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->isSceneJoint(instanceId);
#endif
	return false;
}

ClassDescription PvdEngine::getClassOf(void* instPtr)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->getClassOf(instPtr);
#endif
	PX_ASSERT(instPtr != nullptr);
	return m_reader->getClassOf(instPtr);
}

Option<ClassDescription> PvdEngine::getClass(PxI32 classId)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->getClass(classId);
#endif
	return m_meta->getClass(classId);
}

Option<PropertyDescription> PvdEngine::findProperty(PxI32 classId, const char* propName)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->findProperty(classId, propName);
#endif
	return m_meta->findProperty(classId, propName);
}

Option<PropertyDescription> PvdEngine::findPropertyFlex(PxI32 classId, const char* propName)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->findPropertyFlex(classId, propName);
#endif
	// PXD2 path: try exact match first
	Option<PropertyDescription> result = m_meta->findProperty(classId, propName);
	if (result.hasValue())
		return result;
	return result;
}

Option<Section> PvdEngine::getFrameInfo(PxU32 frame, PxU64 sceneId)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd) return Option<Section>();
#endif
	return m_streamMeta->getFrameInfo(frame, sceneId);
}

const char* PvdEngine::handleToString(const uint32_t handle) const
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->handleToString(handle);
#endif
	return m_meta->getStringTable().handleToStr(handle);
}

// ---------------------------------------------------------------------------
// Property values
// ---------------------------------------------------------------------------

PxU32 PvdEngine::getPropertyByteSize(void* instPtr, const PxU32 propId)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->getPropertyByteSize(instPtr, propId);
#endif
	PX_ASSERT(instPtr != nullptr);
	return m_reader->getPropertyByteSize(instPtr, propId);
}

bool PvdEngine::hasPropertySample(void* instPtr, int32_t propId)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->hasPropertySample(instPtr, propId);
#endif
	return true;
}

OvdEvidenceDomain PvdEngine::ovdEvidenceDomain() const {
#ifdef PVD_OVD_SUPPORT
    if(m_isOvd && m_ovdAdapter)return m_ovdAdapter->evidenceDomain();
#endif
    return {};
}
OvdEvidenceIdentity PvdEngine::ovdEvidenceIdentity(int32_t instanceId) const {
#ifdef PVD_OVD_SUPPORT
    if(m_isOvd && m_ovdAdapter)return m_ovdAdapter->evidenceIdentity(instanceId);
#endif
    return {};
}
OvdEvidenceSampleStamp PvdEngine::ovdEvidenceSampleStamp(void* instance,int32_t property,uint64_t frame) const {
#ifdef PVD_OVD_SUPPORT
    if(m_isOvd && m_ovdAdapter)return m_ovdAdapter->evidenceSampleStamp(instance,property,frame);
#endif
    return {};
}

PxU64 PvdEngine::getStreamId(PxI32 instance)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->getStreamId(instance);
#endif
	return m_streamMeta->getStreamId(instance);
}

InstanceDescription PvdEngine::createInstance() { return InstanceDescription(); }

PxU32 PvdEngine::getNbArrayElements(void* instPtr, const char* propName)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->getNbArrayElements(instPtr, propName);
#endif
	PX_ASSERT(instPtr != nullptr);
	return m_reader->getNbArrayElements(instPtr, propName);
}

PxU32 PvdEngine::getNbArrayElements(void* instPtr, int propId)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->getNbArrayElements(instPtr, propId);
#endif
	return m_reader->getNbArrayElements(instPtr, propId);
}

PxU32 PvdEngine::getPropertyValue(void* instPtr, PxI32 propId, void* outData, PxU32 outDataLen)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->getPropertyValue(instPtr, propId, outData, outDataLen);
#endif
	PX_ASSERT(instPtr != nullptr);
	return m_reader->getPropertyValue(instPtr, propId, static_cast<PxU8*>(outData), outDataLen);
}

DataRef<uint8_t> PvdEngine::getRawPropertyValue(void* instPtr, int32_t propId)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->getRawPropertyValue(instPtr, propId);
	if (m_isOvd) return DataRef<uint8_t>();
#endif
	PX_ASSERT(instPtr != nullptr);
	return m_reader->getRawPropertyValue(instPtr, propId);
}

DataRef<uint8_t> PvdEngine::getRawPropertyValueAtFrame(void* instPtr, int32_t propId, uint64_t frame)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->getRawPropertyValueAtFrame(instPtr, propId, frame);
#endif
	return DataRef<uint8_t>();
}

DataRef<uint8_t> PvdEngine::getRawPropertyValueRecordedAtFrame(void* instPtr, int32_t propId, uint64_t frame)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->getRawPropertyValueRecordedAtFrame(instPtr, propId, frame);
#endif
	return DataRef<uint8_t>();
}

PxI32 PvdEngine::resolveInstanceId(PxU64 externalHandle, void* referrerInstPtr) const
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->resolveInstanceId(externalHandle, referrerInstPtr);
#endif
	return -1;
}

#ifdef PVD_OVD_SUPPORT
const std::vector<OmniPvdMessage>& PvdEngine::getMessages() const
{
	static const std::vector<OmniPvdMessage> sEmpty;
	if (m_isOvd && m_ovdAdapter) return m_ovdAdapter->getMessages();
	return sEmpty;
}

int64_t PvdEngine::getOvdPropertyFieldCount(PxI32 propId) const
{
	return (m_isOvd && m_ovdAdapter) ? m_ovdAdapter->getPropertyFieldCount(propId) : -1;
}

std::string PvdEngine::getOvdNativeTypeLabel(PxI32 propId) const
{
	if (m_isOvd && m_ovdAdapter)
		return m_ovdAdapter->getNativeTypeLabel(propId);
	return std::string();
}
#endif

DataRef<NamedValue> PvdEngine::getNamedPropertyValues(PxI32 propId) const
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd && m_ovdAdapter)
		return m_ovdAdapter->getNamedPropertyValues(propId);
#endif
	return m_meta->getNamedPropertyValues(propId);
}

// ---------------------------------------------------------------------------
// Save / Export (PXD2 only)
// ---------------------------------------------------------------------------

void PvdEngine::save(PvdRandomAccessIOStream* fileStream)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd) return;
#endif
	m_client->save(*fileStream);
}

bool PvdEngine::canExportRepX()
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd) return false;
#endif
	return repx::PvdRepxExporter::canExport(*m_client);
}

void PvdEngine::exportRepX(physx::pvdsdk::PxAllocatorCallback& alloc, PvdRandomAccessIOStream* fileStream, const std::vector<uint64_t>* objs)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd) return;
#endif
	physx::shdfnd::Array<uint64_t> _objs;
	if(objs)
	{
		for(uint32_t i = 0; i < objs->size(); i++)
		{
			_objs.pushBack((*objs)[i]);
		}
	}
	repx::PvdRepxExporter::exportToRepX(_objs, *m_client, *fileStream, alloc);
}

PvdClient* PvdEngine::crop(PxI64 startFrame, PxI64 endFrame, const char* tempDir)
{
#ifdef PVD_OVD_SUPPORT
	if (m_isOvd) return nullptr;
#endif
    const auto marker = m_streamMeta->getDefaultFrameMarker();
    if(!marker.hasValue()) throw std::invalid_argument("Capture has no frame timeline to crop");
    return crop(marker.getValue().mStreamId, startFrame, endFrame, tempDir);
}

PvdClient* PvdEngine::crop(PxU64 sceneId, PxI64 startFrame, PxI64 endFrame, const char* tempDir)
{
    if(m_isOvd || !m_client || !m_client->getEventStream())
        throw std::invalid_argument("Native crop requires a PXD2 event stream");
    if(startFrame < 0 || endFrame < startFrame || uint64_t(endFrame) >= m_client->getFrameCount(sceneId))
        throw std::out_of_range("Invalid inclusive crop frame range");
    const auto startEvent = m_streamMeta->frameToEvent(PxU32(startFrame), sceneId);
    const auto endEvent = m_streamMeta->frameToEvent(PxU32(endFrame), sceneId);
    if(endEvent < startEvent || endEvent >= m_client->getEventCount())
        throw std::out_of_range("Crop boundary has no completed frame event");
    auto releaseClient = [](PvdClient* client) { if(client) client->release(); };
    auto releaseStream = [](PvdRandomAccessIOStream* stream) { if(stream) stream->release(); };
    // Branch owns separate mutable model/index state; cropping never seeks the
    // displayed client or invalidates its cached instance pointers.
    std::unique_ptr<PvdClient, decltype(releaseClient)> branch(m_client->branch(), releaseClient);
    if(!branch) throw std::runtime_error("Unable to branch native capture");
    branch->setCurrentEvent(startEvent);
    std::unique_ptr<PvdRandomAccessIOStream, decltype(releaseStream)> backing(nullptr, releaseStream);
    auto& allocator = mAllocator ? *mAllocator : PxGetFoundation().getAllocatorCallback();
    const std::string root = tempDir ? tempDir : ".";
    for(int attempt = 0; attempt < 16 && !backing; ++attempt) {
        const std::string path = pvdapp::input::temporaryBackingFileCandidate(root);
        const auto opened = PvdRandomAccessIOStream::open(allocator, path.c_str(), true, true);
        if(opened.hasValue()) { opened.getValue()->addRef(); backing.reset(opened.getValue()); }
    }
    if(!backing) throw std::runtime_error("Unable to create temporary native clip backing");
    // transferEvents snapshots the current state at event zero and transfers
    // events (startEvent,endEvent], hence the inclusive event-count increment.
    return branch->transferEvents(endEvent - startEvent + 1, *backing);
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

#ifdef PVD_OVD_SUPPORT
PvdEngine* PvdEngine::load(PxAllocatorCallback& alloc, const char* tempDir, const char* filename,
                           OvdInputErrorCode* ovdErrorCode, LoadControl* control)
#else
PvdEngine* PvdEngine::load(PxAllocatorCallback& alloc, const char* tempDir, const char* filename, LoadControl* control)
#endif
{
#ifdef PVD_OVD_SUPPORT
	if (ovdErrorCode)
		*ovdErrorCode = OvdInputErrorCode::None;
	// OVD files: parse DOM directly, no PVD3 object model
	if (hasOvdExtension(filename))
	{
		ovdLog("[PvdEngine::load] OVD: %s\n", filename);
		PvdEngine* engine = new PvdEngine();
		engine->mAllocator = &alloc;
		engine->m_isOvd = true;

		engine->m_ovdAdapter = loadOvdFile(
			filename, OvdResourceLimits::defaults(), ovdErrorCode, control);
		if (!engine->m_ovdAdapter)
		{
			ovdLog("[PvdEngine::load] OVD load FAILED\n");
			delete engine;
			return nullptr;
		}
		// m_client, m_meta, m_reader, m_streamMeta all stay nullptr
		ovdLog("[PvdEngine::load] OVD load complete\n");
		return engine;
	}
#endif

	if (control && control->cancelled()) return nullptr;
	// PXD2/PXD3 files: existing path
	PxU32 firstWord = 0;
	std::ifstream fin;
	fin.open(std::filesystem::u8path(filename), std::ios::binary | std::ios::in);
	if(!fin)
		return nullptr;
	fin.seekg(0, std::ios::end);
	const auto measuredSize = fin.tellg();
	const uint64_t totalBytes = measuredSize > 0 ? static_cast<uint64_t>(measuredSize) : 0;
	fin.seekg(0);
	fin.read(reinterpret_cast<char*>(&firstWord), sizeof(firstWord));
	fin.close();
	if (totalBytes < sizeof(firstWord)) return nullptr;

	PvdEngine* engine = new PvdEngine();
	engine->mAllocator = &alloc;

	if(firstWord == STREAM_MAGIC_WORD)
	{
		ErrorCodeOption<PvdRandomAccessIOStream*> stream = PvdRandomAccessIOStream::open(alloc, filename, false, false);
		if(stream.hasValue() == false)
		{
			delete engine;
			return nullptr;
		}
		PvdRandomAccessIOStream* backingStream = nullptr;
		const std::string tempRoot = tempDir ? tempDir : ".";
		for(PxU32 attempt = 0; attempt < 16 && !backingStream; ++attempt)
		{
			const std::string candidate = pvdapp::input::temporaryBackingFileCandidate(tempRoot);
			ErrorCodeOption<PvdRandomAccessIOStream*> opened =
				PvdRandomAccessIOStream::open(alloc, candidate.c_str(), true, true);
			if(opened.hasValue())
				backingStream = opened.getValue();
		}
		if(!backingStream)
		{
			stream.getValue()->release();
			delete engine;
			return nullptr;
		}
		PvdFileLoaderFactory& pvdFactory = PvdFileLoaderFactory::createFileLoaderFactory(alloc);
		PvdFileStreamLoader* loader = pvdFactory.createFileStreamLoader(*stream.getValue(), *backingStream);
		pvdFactory.release();
		if(!loader)
		{
			stream.getValue()->release();
			backingStream->release();
			delete engine;
			return nullptr;
		}
		uint64_t processed = 0;
		if (control) control->update("Reading legacy stream", 0, totalBytes);
		for(;;)
		{
			if (control && control->cancelled())
			{
				loader->release();
				delete engine;
				return nullptr;
			}
			const PvdFileStreamLoadResult result = loader->loadNextChunkResult(0x80000);
			processed += result.mBytesProcessed;
			if (control) control->update("Reading legacy stream", std::min(processed, totalBytes), totalBytes);
			if(result.mStatus == PvdFileStreamLoadResult::Failed)
			{
				loader->release();
				delete engine;
				return nullptr;
			}
			if(result.mStatus == PvdFileStreamLoadResult::Completed)
				break;
		}
		engine->m_loader = loader;
		engine->m_client = &(engine->m_loader->getClient().getClient());
		engine->m_meta = &engine->m_client->getMetaData();
		engine->m_reader = &engine->m_client->getReader();
		engine->m_streamMeta = &engine->m_client->getStreamMetaData();
	}
	else if(firstWord == FILE_MAGIC_WORD)
	{
		ErrorCodeOption<PvdRandomAccessIOStream*> stream = PvdRandomAccessIOStream::open(alloc, filename, false, false);
		if(stream.hasValue() == false)
		{
			delete engine;
			return nullptr;
		}
		if (control && control->cancelled())
		{
			stream.getValue()->release();
			delete engine;
			return nullptr;
		}
		if (control) control->update("Reading indexed legacy capture", 0, 0, false);
		engine->m_client = PvdClient::load(alloc, *stream.getValue());
		if(engine->m_client == nullptr)
		{
			stream.getValue()->release();
			delete engine;
			return nullptr;
		}
		engine->m_meta = &engine->m_client->getMetaData();
		engine->m_reader = &engine->m_client->getReader();
		engine->m_streamMeta = &engine->m_client->getStreamMetaData();
	}
	else
	{
		delete engine;
		return nullptr;
	}

	if (control && control->cancelled()) { delete engine; return nullptr; }
	if (control) control->update("Capture parsed", totalBytes, totalBytes);
	engine->m_meta = &engine->m_client->getMetaData();
	engine->m_reader = &engine->m_client->getReader();
	engine->m_streamMeta = &engine->m_client->getStreamMetaData();
	return engine;
}

PvdEngine* PvdEngine::create(PvdClient* client)
{
	if(!client)
		return nullptr;
	PvdEngine* engine = new PvdEngine();
	// A live PvdServer keeps its own client reference. The engine must own a
	// separate reference because either the listener or the UI can be torn down
	// first when replacing a connection or closing the window.
	client->addRef();
	engine->m_client = client;
	engine->m_meta = &engine->m_client->getMetaData();
	engine->m_reader = &engine->m_client->getReader();
	engine->m_streamMeta = &engine->m_client->getStreamMetaData();
	return engine;
}

// ---------------------------------------------------------------------------
// Live OVD connection
// ---------------------------------------------------------------------------

PvdEngine* PvdEngine::connectLive(PxAllocatorCallback& alloc, uint16_t port,
                                  const char* recordPath)
{
#ifdef PVD_OVD_SUPPORT
	ovdLog("[PvdEngine::connectLive] listening on port %u\n", (unsigned)port);

	// Start from an empty (self-initialized) DOM; the adapter owns it and the
	// ingest worker grows it in place under the adapter's recursive mutex.
	auto domState = std::make_unique<OmniPvdDOMState>();
	auto adapter  = std::make_unique<OvdDomAdapter>(std::move(domState), std::vector<OmniPvdMessage>());

	auto live = std::make_unique<OvdLiveSource>(
		adapter->dom(), &adapter->liveMutex(), OvdResourceLimits::defaults(),
		&adapter->liveMessageSink());
	if (!live->start(port, recordPath))
	{
		ovdLog("[PvdEngine::connectLive] could not start listening\n");
		return nullptr;
	}

	PvdEngine* engine    = new PvdEngine();
	engine->mAllocator   = &alloc;
	engine->m_isOvd      = true;
	engine->m_ovdAdapter = std::move(adapter);
	engine->m_live       = std::move(live);
	// m_client, m_meta, m_reader, m_streamMeta all stay nullptr (OVD path)
	ovdLog("[PvdEngine::connectLive] live attached\n");
	return engine;
#else
	(void)alloc; (void)port; (void)recordPath;
	return nullptr;
#endif
}

bool PvdEngine::isLive() const
{
#ifdef PVD_OVD_SUPPORT
	return m_live != nullptr;
#else
	return false;
#endif
}

bool PvdEngine::isLiveConnected() const
{
#ifdef PVD_OVD_SUPPORT
	return m_live && m_live->isConnected();
#else
	return false;
#endif
}

bool PvdEngine::isLiveEnded() const
{
#ifdef PVD_OVD_SUPPORT
	return m_live && m_live->hasEnded();
#else
	return false;
#endif
}

#ifdef PVD_OVD_SUPPORT
OvdInputErrorCode PvdEngine::getLiveErrorCode() const
{
	return m_live ? m_live->errorCode() : OvdInputErrorCode::None;
}
#endif

PxU64 PvdEngine::getLiveMaxFrame() const
{
#ifdef PVD_OVD_SUPPORT
	return m_live ? (PxU64)m_live->maxFrame() : 0;
#else
	return 0;
#endif
}

PxU64 PvdEngine::getLiveCurrentFrame() const
{
#ifdef PVD_OVD_SUPPORT
	return m_live ? (PxU64)m_live->currentFrame() : 0;
#else
	return 0;
#endif
}

PxU64 PvdEngine::getLiveCompletedFrameSerial() const
{
#ifdef PVD_OVD_SUPPORT
	return m_live ? (PxU64)m_live->completedFrameSerial() : 0;
#else
	return 0;
#endif
}

bool PvdEngine::hasLiveUnpublishedIndexChanges() const
{
#ifdef PVD_OVD_SUPPORT
	return m_live && m_live->hasUnpublishedIndexChanges();
#else
	return false;
#endif
}

PxU64 PvdEngine::getOvdSegmentEpoch() const
{
#ifdef PVD_OVD_SUPPORT
	if(!m_ovdAdapter)
		return 0;
	return m_live ? m_live->completedSegmentEpoch() : m_ovdAdapter->getSegmentEpoch();
#else
	return 0;
#endif
}

PxU64 PvdEngine::getOvdRecordingSegment(PxI32 instId) const
{
#ifdef PVD_OVD_SUPPORT
	return m_ovdAdapter ? m_ovdAdapter->getRecordingSegment(instId) : 0;
#else
	(void)instId;
	return 0;
#endif
}

#ifdef PVD_OVD_SUPPORT
CaptureCatalogSnapshot PvdEngine::captureCatalogSnapshot() const
{
	return m_live ? m_live->catalogSnapshot() : CaptureCatalogSnapshot{};
}
CaptureBudgetSnapshot PvdEngine::captureBudget() const
{
	return m_live ? m_live->captureBudget() : CaptureBudgetSnapshot{};
}
CaptureMetadataSnapshot PvdEngine::captureMetadata() const { return m_live ? m_live->metadataSnapshot() : CaptureMetadataSnapshot{}; }
CaptureTransportSnapshot PvdEngine::captureTransport() const { return m_live ? m_live->controlTransportState() : CaptureTransportSnapshot{}; }
uint16_t PvdEngine::captureCapabilities() const { return m_live ? m_live->controlCapabilities() : 0; }
void PvdEngine::queueCaptureRequest(const CaptureRequest& request, uint64_t generation)
{
	if (!m_live) throw std::runtime_error("File sessions cannot control a producer");
	m_live->queueAuthorizedCaptureRequest(request, generation);
}
CaptureCatalogPage PvdEngine::captureCatalogPage(uint64_t generation, uint64_t offset, uint32_t limit,
                                               uint64_t parent, std::string_view prefix) const
{
	return m_live ? m_live->catalogPage(generation, offset, limit, parent, prefix) : CaptureCatalogPage{CaptureCatalogPageState::Busy};
}
uint64_t PvdEngine::getLiveMetadataBoundarySerial() const
{
	return m_live ? m_live->completedMetadataBoundarySerial() : 0;
}
#endif

void PvdEngine::refreshLive()
{
#ifdef PVD_OVD_SUPPORT
	if (!m_live || !m_ovdAdapter)
		return;
	std::lock_guard<std::recursive_mutex> lk(m_ovdAdapter->liveMutex());
	if (m_live->hasUnpublishedIndexChanges())
		return;
	const PxU64 completedFrameSerial = m_live->completedFrameSerial();
	if(completedFrameSerial == 0 || completedFrameSerial == m_lastRefreshedCompletedFrameSerial)
		return;
	if (!m_ovdAdapter->rebuildIndex())
		return;
	m_lastRefreshedCompletedFrameSerial = completedFrameSerial;
	++mLiveIndexRebuildCount;
#endif
}

void PvdEngine::stopLive()
{
#ifdef PVD_OVD_SUPPORT
	if (m_live)
		m_live->stop(); // joins the worker + finalizes the recording; DOM/adapter stay alive
#endif
}

std::string PvdEngine::getLiveRecordingPath() const
{
#ifdef PVD_OVD_SUPPORT
	return m_live ? m_live->recordedFilePath() : std::string();
#else
	return std::string();
#endif
}

void PvdEngine::flushLiveRecording()
{
#ifdef PVD_OVD_SUPPORT
	if (m_live)
		m_live->flushRecording();
#endif
}

uint64_t PvdEngine::getLiveRecordingSize() const
{
#ifdef PVD_OVD_SUPPORT
	return m_live ? m_live->recordingSize() : 0;
#else
	return 0;
#endif
}

bool PvdEngine::writeLiveRecording(OmniPvdWriteStream& destination) const
{
#ifdef PVD_OVD_SUPPORT
	return m_live && m_live->writeRecording(destination);
#else
	(void)destination;
	return false;
#endif
}

PvdEngine::LiveReadGuard::LiveReadGuard(PvdEngine& engine) : m_mtx(nullptr)
{
#ifdef PVD_OVD_SUPPORT
	if (engine.m_live && engine.m_ovdAdapter)
	{
		std::recursive_mutex* m = &engine.m_ovdAdapter->liveMutex();
		m->lock();
		m_mtx = m;
	}
#else
	(void)engine;
#endif
}

PvdEngine::LiveReadGuard::~LiveReadGuard()
{
#ifdef PVD_OVD_SUPPORT
	if (m_mtx)
		static_cast<std::recursive_mutex*>(m_mtx)->unlock();
#endif
}

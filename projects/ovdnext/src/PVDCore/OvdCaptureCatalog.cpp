/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "OvdCaptureCatalog.h"
#include "PvdDom.h"
#include <algorithm>
#include <atomic>
#include <memory_resource>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace pvdapp
{
namespace
{
class CatalogMemory final : public std::pmr::memory_resource
{
public:
    explicit CatalogMemory(uint64_t maximum) : maximum(maximum) {}
    std::atomic<uint64_t> used{0}, peak{0};
private:
    uint64_t maximum;
    void* do_allocate(size_t bytes, size_t alignment) override
    {
        if (bytes > maximum - used.load()) throw std::bad_alloc();
        void* result = std::pmr::new_delete_resource()->allocate(bytes, alignment);
        const auto current = used.fetch_add(bytes) + bytes;
        auto previous = peak.load();
        while (previous < current && !peak.compare_exchange_weak(previous, current)) {}
        return result;
    }
    void do_deallocate(void* memory, size_t bytes, size_t alignment) override
    {
        std::pmr::new_delete_resource()->deallocate(memory, bytes, alignment);
        used.fetch_sub(bytes);
    }
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }
};
struct Row
{
    Row(std::pmr::memory_resource* resource, uint32_t schema, uint64_t serial)
        : schema(schema), serial(serial), typeName(resource), name(resource) {}
    uint32_t schema;
    uint64_t serial, target = 0, owner = 0, scene = 0;
    std::pmr::string typeName, name;
    uint8_t fields = 0;
    bool selectable = false;
    uint64_t parent() const { return owner ? owner : (scene == target ? 0 : scene); }
};
}

struct OvdCaptureCatalog::Impl
{
    explicit Impl(CaptureCatalogLimits limits) : limits(limits), memory(limits.allocations),
        rows(&memory), byName(&memory), byParent(&memory), byTarget(&memory) {}
    CaptureCatalogLimits limits;
    CatalogMemory memory; // outlives every PMR container
    std::pmr::unordered_map<uint64_t, Row> rows;
    std::pmr::vector<uint64_t> byName, byParent, byTarget;
    mutable std::mutex publication;
    bool ready = false, building = true;
    uint64_t generation = 0, revision = 0, serial = 1, oldRemaining = 0, commands = 0, mutations = 0;
    std::atomic<uint64_t> rowCount{0}, stringBytes{0};
    std::atomic<CaptureCatalogError> failure{CaptureCatalogError::None};
    CaptureMetadataSnapshot metadata;
    uint64_t metadataRow = 0, pendingAckId = 0;
    uint32_t metadataSchema = 0, pendingAckStatus = 0;
    uint8_t ackStage = 0, policyFields = 0;
    bool sawMetadata = false;

    bool fail(CaptureCatalogError error)
    {
        std::lock_guard<std::mutex> lock(publication);
        ready = false; ++revision; failure.store(error);
        return false;
    }
    void begin()
    {
        if (building) return;
        {
            std::lock_guard<std::mutex> lock(publication);
            ready = false; ++revision;
        }
        building = true; ++serial; oldRemaining = rows.size(); commands = mutations = 0;
        byName.clear(); byParent.clear(); byTarget.clear();
    }
    bool commit(uint64_t next)
    {
        if (!next || next <= generation || oldRemaining) return fail(CaptureCatalogError::Invalid);
        byName.clear(); byParent.clear(); byTarget.clear();
        byName.reserve(rows.size()); byParent.reserve(rows.size()); byTarget.reserve(rows.size());
        for (const auto& item : rows)
        {
            if (item.second.fields != 63 || (item.second.selectable && !item.second.target))
                return fail(CaptureCatalogError::Invalid);
            byName.push_back(item.first); byParent.push_back(item.first); byTarget.push_back(item.first);
        }
        const auto nameLess = [&](uint64_t a, uint64_t b) {
            const auto& left = rows.at(a); const auto& right = rows.at(b);
            return left.name == right.name ? left.target < right.target : left.name < right.name;
        };
        std::sort(byName.begin(), byName.end(), nameLess);
        std::sort(byParent.begin(), byParent.end(), [&](uint64_t a, uint64_t b) {
            const auto left = rows.at(a).parent(), right = rows.at(b).parent();
            return left == right ? nameLess(a, b) : left < right;
        });
        std::sort(byTarget.begin(), byTarget.end(), [&](uint64_t a, uint64_t b) { return rows.at(a).target < rows.at(b).target; });
        for (size_t i = 1; i < byTarget.size(); ++i)
            if (rows.at(byTarget[i - 1]).target == rows.at(byTarget[i]).target) return fail(CaptureCatalogError::Invalid);
        building = false;
        std::lock_guard<std::mutex> lock(publication);
        generation = next; ++revision; ready = true;
        return true;
    }
};

OvdCaptureCatalog::OvdCaptureCatalog(CaptureCatalogLimits limits) : m(new Impl(limits)) {}
OvdCaptureCatalog::~OvdCaptureCatalog() = default;
CaptureCatalogError OvdCaptureCatalog::error() const { return m->failure.load(); }
bool OvdCaptureCatalog::contains(uint64_t row) const { return m->rows.find(row) != m->rows.end(); }
uint32_t OvdCaptureCatalog::catalogRowSchema(uint64_t row) const
{
    const auto found = m->rows.find(row);
    return found == m->rows.end() ? 0 : found->second.schema;
}
OvdInputErrorCode OvdCaptureCatalog::observeValidatedCapture(const OvdValidatedCaptureCommand& command)
{
    if (command.kind >= OvdValidatedCaptureCommand::MetadataCreate)
    {
        // Filtering catalog CREATEs removes the pinned DOM's usual trigger for
        // its version check. Check an owned, typed major field even when it was
        // registered after CREATE and therefore cannot authorize control state.
        if (command.kind == OvdValidatedCaptureCommand::MetadataField && command.field == 3 &&
            command.value > PVDDOM_ACCEPTED_OVD_INTEG_VERSION_MAJOR)
            return OvdInputErrorCode::UnsupportedIntegrationVersion;
        // A generation commit may sort a large index, so leave the publication
        // mutex before handing that work to the ingest-only catalog store.
        std::unique_lock<std::mutex> lock(m->publication);
        if (command.kind == OvdValidatedCaptureCommand::MetadataCreate)
        {
            if (m->sawMetadata) ++m->metadata.boundarySerial;
            m->sawMetadata = true; m->metadataRow = command.row; m->metadataSchema = command.schema;
            m->ackStage = m->policyFields = 0;
            m->metadata.versionKnown = m->metadata.policyKnown = false;
            ++m->metadata.revision;
            return OvdInputErrorCode::None;
        }
        if (command.row != m->metadataRow || command.schema != m->metadataSchema) return OvdInputErrorCode::None;
        if (command.kind == OvdValidatedCaptureCommand::MetadataDestroy)
        {
            m->metadataRow = 0; m->ackStage = 0;
            return OvdInputErrorCode::None;
        }
        auto& state = m->metadata;
        ++state.revision;
        switch (command.field)
        {
        case 3:
            state.versionKnown = true; break;
        case 5:
        case 6:
            if (command.value > 1) return OvdInputErrorCode::InvalidAttributePayload;
            if (command.field == 5) state.secondary = command.value != 0;
            else state.all = command.value != 0;
            m->policyFields |= command.field == 5 ? 1 : 2;
            state.policyKnown = m->policyFields == 3; break;
        case 7:
            m->pendingAckId = command.value; m->ackStage = command.value ? 1 : 0;
            return OvdInputErrorCode::None;
        case 8:
            m->ackStage = m->ackStage == 1 && command.value >= 1 && command.value <= 4 ? 2 : 0;
            m->pendingAckStatus = static_cast<uint32_t>(command.value);
            return OvdInputErrorCode::None;
        case 9:
            state.captureEpoch = command.value;
            if (m->ackStage == 2 && state.versionKnown)
            {
                state.ackId = m->pendingAckId; state.ackStatus = m->pendingAckStatus;
                state.ackEpoch = command.value; ++state.ackRevision;
            }
            break;
        case 10:
            if (!state.versionKnown) return OvdInputErrorCode::InvalidCaptureCatalog;
            if (!command.value || (!m->building && command.value == m->generation)) break;
            m->ackStage = 0;
            lock.unlock();
            return observeValidatedCapture({OvdValidatedCaptureCommand::Commit, 0, 0, 0, command.value});
        default: break;
        }
        m->ackStage = 0;
        return OvdInputErrorCode::None;
    }
    if (observeValidated({static_cast<CaptureCatalogCommand::Kind>(command.kind), command.row, command.schema,
        static_cast<CaptureCatalogField>(command.field), command.value,
        std::string_view(command.text ? command.text : "", command.textSize)})) return OvdInputErrorCode::None;
    switch (error())
    {
    case CaptureCatalogError::Rows: return OvdInputErrorCode::CaptureCatalogRowLimit;
    case CaptureCatalogError::Strings: return OvdInputErrorCode::CaptureCatalogStringLimit;
    case CaptureCatalogError::Allocations: return OvdInputErrorCode::CaptureCatalogAllocationLimit;
    case CaptureCatalogError::Commands: return OvdInputErrorCode::CaptureCatalogCommandLimit;
    case CaptureCatalogError::Mutations: return OvdInputErrorCode::CaptureCatalogMutationLimit;
    default: return OvdInputErrorCode::InvalidCaptureCatalog;
    }
}

bool OvdCaptureCatalog::observeValidated(const CaptureCatalogCommand& command)
try
{
    if (error() != CaptureCatalogError::None) return false;
    m->begin();
    if (command.kind == CaptureCatalogCommand::Commit) return m->commit(command.value);
    if (m->commands >= m->limits.commands) return m->fail(CaptureCatalogError::Commands);
    ++m->commands;
    if (command.kind == CaptureCatalogCommand::Create)
    {
        if (!command.row || !command.schema || contains(command.row)) return m->fail(CaptureCatalogError::Invalid);
        if (m->rows.size() >= m->limits.rows) return m->fail(CaptureCatalogError::Rows);
        m->rows.try_emplace(command.row, &m->memory, command.schema, m->serial);
        m->rowCount.store(m->rows.size());
        return true;
    }
    const auto found = m->rows.find(command.row);
    if (found == m->rows.end()) return m->fail(CaptureCatalogError::Invalid);
    Row& row = found->second;
    if (command.kind == CaptureCatalogCommand::Destroy)
    {
        if (row.serial != m->serial) --m->oldRemaining;
        m->stringBytes.fetch_sub(row.name.size() + row.typeName.size());
        m->rows.erase(found); m->rowCount.store(m->rows.size());
        return true;
    }
    const auto field = static_cast<unsigned>(command.field);
    if (command.kind != CaptureCatalogCommand::Field || row.schema != command.schema ||
        field > 5 || (row.fields & (1u << field))) return m->fail(CaptureCatalogError::Invalid);
    if (m->mutations >= m->limits.mutations) return m->fail(CaptureCatalogError::Mutations);
    ++m->mutations;
    switch (command.field)
    {
    case CaptureCatalogField::Target: row.target = command.value; break;
    case CaptureCatalogField::Owner: row.owner = command.value; break;
    case CaptureCatalogField::Scene: row.scene = command.value; break;
    case CaptureCatalogField::Selectable:
        if (command.value > 1) return m->fail(CaptureCatalogError::Invalid);
        row.selectable = command.value != 0; break;
    case CaptureCatalogField::Name:
    case CaptureCatalogField::TypeName:
        if (command.text.size() > 2048 || command.text.size() > m->limits.stringBytes - m->stringBytes.load())
            return m->fail(CaptureCatalogError::Strings);
        (command.field == CaptureCatalogField::Name ? row.name : row.typeName).assign(command.text);
        m->stringBytes.fetch_add(command.text.size());
        break;
    }
    row.fields |= static_cast<uint8_t>(1u << field);
    return true;
}
catch (const std::bad_alloc&) { return m->fail(CaptureCatalogError::Allocations); }

CaptureCatalogSnapshot OvdCaptureCatalog::snapshot() const
{
    std::lock_guard<std::mutex> lock(m->publication);
    return {m->ready, m->generation, m->revision, m->rowCount.load(), m->stringBytes.load(),
            m->memory.used.load(), m->memory.peak.load(), error()};
}
CaptureMetadataSnapshot OvdCaptureCatalog::metadataSnapshot() const
{
    std::lock_guard<std::mutex> lock(m->publication);
    return m->metadata;
}

CaptureCatalogPage OvdCaptureCatalog::catalogPage(uint64_t generation, uint64_t offset, uint32_t limit,
                                                uint64_t parent, std::string_view prefix) const
{
    if (limit > 1000 || prefix.size() > 2048) throw std::invalid_argument("Catalog query exceeds page/prefix bounds");
    std::lock_guard<std::mutex> lock(m->publication);
    CaptureCatalogPage result{CaptureCatalogPageState::Busy, m->generation};
    if (!m->ready) return result;
    if (generation != m->generation) { result.state = CaptureCatalogPageState::Stale; return result; }
    result.state = CaptureCatalogPageState::Ready;
    const auto& index = parent ? m->byParent : m->byName;
    const auto compare = [&](uint64_t key) {
        const auto& row = m->rows.at(key);
        if (parent && row.parent() != parent) return row.parent() < parent ? -1 : 1;
        return prefix.empty() ? 0 : row.name.compare(0, prefix.size(), prefix.data(), prefix.size());
    };
    const auto first = std::lower_bound(index.begin(), index.end(), 0, [&](uint64_t key, int) { return compare(key) < 0; });
    const auto last = std::upper_bound(first, index.end(), 0, [&](int, uint64_t key) { return compare(key) > 0; });
    result.total = static_cast<uint64_t>(last - first);
    offset = std::min(offset, result.total);
    const auto count = std::min<uint64_t>(limit, result.total - offset);
    result.entries.reserve(static_cast<size_t>(count));
    result.copiedBytes = count * sizeof(CaptureCatalogEntry);
    for (auto it = first + static_cast<size_t>(offset); it != last && result.entries.size() < count; ++it)
    {
        const auto& row = m->rows.at(*it);
        // Includes the current MSVC string-capacity rounding, not just payload.
        const auto charge = row.name.size() + row.typeName.size() + 32;
        if (charge > (1ull << 20) - result.copiedBytes) break;
        result.entries.push_back({row.target, row.owner, row.scene, std::string(row.typeName), std::string(row.name), row.selectable});
        result.copiedBytes += charge;
    }
    return result;
}

bool OvdCaptureCatalog::canSelect(uint64_t generation, uint64_t target, bool scene) const
{
    std::lock_guard<std::mutex> lock(m->publication);
    if (!m->ready || generation != m->generation) return false;
    const auto found = std::lower_bound(m->byTarget.begin(), m->byTarget.end(), target,
        [&](uint64_t key, uint64_t value) { return m->rows.at(key).target < value; });
    if (found == m->byTarget.end()) return false;
    const auto& row = m->rows.at(*found);
    return row.target == target && row.selectable && (row.typeName == "PxScene") == scene;
}

void OvdCaptureCatalog::withSelection(uint64_t generation, const std::vector<uint64_t>& scenes,
    const std::vector<uint64_t>& objects, const std::function<void()>& enqueue) const
{
    std::lock_guard<std::mutex> lock(m->publication);
    if (!m->ready || generation != m->generation) throw std::runtime_error("Capture catalog is busy or stale");
    const auto check = [&](uint64_t target, bool scene) {
        const auto found = std::lower_bound(m->byTarget.begin(), m->byTarget.end(), target,
            [&](uint64_t key, uint64_t value) { return m->rows.at(key).target < value; });
        if (found == m->byTarget.end()) return false;
        const auto& row = m->rows.at(*found);
        return row.target == target && row.selectable && (row.typeName == "PxScene") == scene;
    };
    for (auto target : scenes) if (!check(target, true)) throw std::invalid_argument("Scene is not selectable in this catalog");
    for (auto target : objects) if (!check(target, false)) throw std::invalid_argument("Object is not selectable in this catalog");
    enqueue(); // pre-encoded immutable frame; no allocation, socket I/O or DOM work
}
}

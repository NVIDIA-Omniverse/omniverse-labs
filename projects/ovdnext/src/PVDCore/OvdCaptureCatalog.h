/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PVD_OVD_CAPTURE_CATALOG_H
#define PVD_OVD_CAPTURE_CATALOG_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include "OvdInputValidation.h"

namespace pvdapp
{
enum class CaptureCatalogField : uint8_t { Target, TypeName, Name, Owner, Scene, Selectable };
enum class CaptureCatalogError { None, Invalid, Rows, Strings, Allocations, Commands, Mutations };
enum class CaptureCatalogPageState { Ready, Busy, Stale };
struct CaptureCatalogLimits
{
    uint64_t rows, stringBytes, allocations, commands, mutations;
    static CaptureCatalogLimits defaults() { return {262144, 64ull << 20, 128ull << 20, 9 * 262144 + 16, 6 * 262144}; }
};
struct CaptureCatalogCommand
{
    enum Kind { Create, Destroy, Field, Commit } kind;
    uint64_t row = 0;
    uint32_t schema = 0;
    CaptureCatalogField field = CaptureCatalogField::Target;
    uint64_t value = 0;
    std::string_view text;
};
struct CaptureCatalogEntry
{
    uint64_t targetHandle, ownerHandle, sceneHandle;
    std::string typeName, name;
    bool canSelect;
};
struct CaptureCatalogSnapshot
{
    bool ready;
    uint64_t generation, revision, rows, stringBytes, allocations, peakAllocations;
    CaptureCatalogError error;
};
struct CaptureCatalogPage
{
    CaptureCatalogPageState state;
    uint64_t generation = 0, total = 0, copiedBytes = 0;
    std::vector<CaptureCatalogEntry> entries;
};
struct CaptureMetadataSnapshot
{
    uint64_t revision = 0, boundarySerial = 0, captureEpoch = 0, ackId = 0;
    uint64_t ackEpoch = 0, ackRevision = 0;
    uint32_t ackStatus = 0;
    bool versionKnown = false, policyKnown = false, secondary = false, all = false;
};

// One ingest-thread writer. Readers copy bounded pages under a publication
// mutex; mutations/index construction happen only while publication is invalid.
class OvdCaptureCatalog : public OvdCaptureValidationSink
{
public:
    explicit OvdCaptureCatalog(CaptureCatalogLimits limits = CaptureCatalogLimits::defaults());
    ~OvdCaptureCatalog();
    bool observeValidated(const CaptureCatalogCommand& command);
    CaptureCatalogSnapshot snapshot() const;
    CaptureMetadataSnapshot metadataSnapshot() const;
    CaptureCatalogError error() const;
    CaptureCatalogPage catalogPage(uint64_t generation, uint64_t offset = 0, uint32_t limit = 1000,
                                   uint64_t parent = 0, std::string_view namePrefix = {}) const;
    bool canSelect(uint64_t generation, uint64_t target, bool scene) const;
    void withSelection(uint64_t generation, const std::vector<uint64_t>& scenes,
                       const std::vector<uint64_t>& objects, const std::function<void()>& enqueue) const;
    bool contains(uint64_t row) const; // ingest-thread lifetime check
    uint32_t catalogRowSchema(uint64_t row) const override;
    OvdInputErrorCode observeValidatedCapture(const OvdValidatedCaptureCommand& command) override;
private:
    struct Impl;
    std::unique_ptr<Impl> m;
};
}
#endif

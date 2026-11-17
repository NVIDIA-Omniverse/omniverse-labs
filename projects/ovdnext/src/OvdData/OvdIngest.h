/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "OvdData.h"
#include "OvdRecordedModel.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ovdnext
{
namespace data
{

enum class CommandKind
{
    RegisterClass,
    RegisterAttribute,
    CreateObject,
    DestroyObject,
    SetProperty,
    AddToList,
    RemoveFromList,
    StartFrame,
    StopFrame
};

// String and payload fields are borrowed and remain valid only for consume().
struct Command
{
    CommandKind kind = CommandKind::RegisterClass;
    std::uint64_t classHandle = 0;
    std::uint64_t baseClassHandle = 0;
    std::uint64_t ownerClassHandle = 0;
    std::uint64_t attributeHandle = 0;
    std::uint64_t attributeClassHandle = 0;
    std::uint64_t flagsEnumClassHandle = 0;
    std::uint64_t objectHandle = 0;
    std::uint64_t context = 0;
    std::uint64_t frame = 0;
    RecordedAttributeKind attributeKind = RecordedAttributeKind::Value;
    RecordedDataType dataType = RecordedDataType::None;
    bool hasElementCount = false;
    std::uint32_t elementCount = 0;
    std::uint32_t enumValue = 0;
    const char* name = nullptr;
    const std::uint8_t* payload = nullptr;
    std::size_t payloadSize = 0;
};

enum class IngestError
{
    None,
    AlreadyFinished,
    UnknownBaseClass,
    InheritanceCycle,
    ConflictingClassRegistration,
    InvalidSourceHandle,
    UnknownAttributeOwner,
    DuplicateAttributeRegistration,
    UnknownRelatedClass,
    InvalidAttributeDeclaration,
    UnknownClass,
    UnknownProperty,
    UnknownObject,
    PropertyKindMismatch,
    NestedStart,
    StopWithoutStart,
    MismatchedStop,
    SegmentTransitionDuringFrame,
    ActiveFrameAtEof,
    StoreFailure
};

struct IngestStats
{
    std::uint64_t commandCount = 0;
    std::uint64_t publicationCount = 0;
    std::uint64_t retainedPayloadBytes = 0;
    std::uint64_t filteredPayloadBytes = 0;
};

class Capture
{
public:
    ~Capture();
    Capture(Capture&&) noexcept;
    Capture& operator=(Capture&&) noexcept;
    Capture(const Capture&) = delete;
    Capture& operator=(const Capture&) = delete;

    const std::vector<FrameStamp>& frames() const;
    const IngestStats& stats() const;
    std::shared_ptr<const RecordedClassModel> recordedClassModel() const;
    std::shared_ptr<const FrameView> view(std::size_t directoryIndex) const;
    std::shared_ptr<const FrameView> view(const FrameAddress& address) const;
    std::shared_ptr<const FrameView> view(const FrameViewRequest& request) const;

private:
    struct Impl;
    explicit Capture(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> mImpl;

    friend class CommandIngestor;
};

class CommandIngestor
{
public:
    explicit CommandIngestor(const LoadPlan& loadPlan);
    ~CommandIngestor();
    CommandIngestor(CommandIngestor&&) noexcept;
    CommandIngestor& operator=(CommandIngestor&&) noexcept;
    CommandIngestor(const CommandIngestor&) = delete;
    CommandIngestor& operator=(const CommandIngestor&) = delete;

    bool consume(const Command& command);
    bool finish();
    IngestError error() const;
    const IngestStats& stats() const;
    const std::vector<FrameStamp>& frames() const;
    std::unique_ptr<Capture> takeCapture();

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace data
} // namespace ovdnext

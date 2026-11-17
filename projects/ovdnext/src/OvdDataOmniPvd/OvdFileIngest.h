/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "OvdIngest.h"

#include <cstdint>
#include <memory>
#include <string>

namespace ovdnext
{
namespace omnipvd
{

// These are parser safety limits. maxRetainedValidationBytes bounds the
// validation accounting model; it does not cause this adapter to retain source
// payloads. LoadPlan determines retained OvdData payloads.
struct FileIngestLimits
{
    FileIngestLimits();

    std::uint64_t maxSingleAttributePayloadBytes;
    std::uint64_t maxCumulativeAttributePayloadBytes;
    std::uint64_t maxStreamBytes;
    std::uint64_t maxCommands;
    std::uint64_t maxAttributeMutations;
    std::uint64_t maxClasses;
    std::uint64_t maxAttributes;
    std::uint64_t maxAttributesPerClass;
    std::uint64_t maxObjectCreates;
    std::uint64_t maxMessages;
    std::uint64_t maxRetainedValidationBytes;

    static FileIngestLimits defaults();
};

enum class FileIngestError
{
    None,
    FileOpenFailed,
    TruncatedInput,
    ValidationFailed,
    ReaderRejected,
    UnsupportedIntegrationVersion,
    TranslationFailed,
    IncompleteInput
};

struct FileIngestResult
{
    std::unique_ptr<data::Capture> capture;
    FileIngestError error = FileIngestError::None;
    std::string diagnostic;

    FileIngestResult() = default;
    FileIngestResult(FileIngestResult&&) noexcept = default;
    FileIngestResult& operator=(FileIngestResult&&) noexcept = default;
    FileIngestResult(const FileIngestResult&) = delete;
    FileIngestResult& operator=(const FileIngestResult&) = delete;
};

FileIngestResult ingestOvdFile(const std::string& path, const data::LoadPlan& loadPlan,
                               const FileIngestLimits& limits = FileIngestLimits::defaults());

} // namespace omnipvd
} // namespace ovdnext

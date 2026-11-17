/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "OvdRecordedModel.h"

#include <cstdint>
#include <memory>
#include <string>

namespace ovdnext
{
namespace data
{
namespace detail
{

enum class ModelBuildError
{
    None,
    InvalidSourceHandle,
    UnknownBaseClass,
    ConflictingClassRegistration,
    DuplicateAttributeRegistration,
    InheritanceCycle,
    UnknownAttributeOwner,
    UnknownRelatedClass,
    InvalidAttributeDeclaration
};

class RecordedClassModelBuilder
{
public:
    RecordedClassModelBuilder();
    ~RecordedClassModelBuilder();
    RecordedClassModelBuilder(RecordedClassModelBuilder&&) noexcept;
    RecordedClassModelBuilder& operator=(RecordedClassModelBuilder&&) noexcept;
    RecordedClassModelBuilder(const RecordedClassModelBuilder&) = delete;
    RecordedClassModelBuilder& operator=(const RecordedClassModelBuilder&) = delete;
    bool registerClass(std::uint64_t sourceClassHandle,
                       std::uint64_t sourceBaseClassHandle,
                       const std::string& name);
    ClassId classIdForSource(std::uint64_t sourceClassHandle) const;
    const RecordedClass* findClassBySource(
        std::uint64_t sourceClassHandle) const;
    bool registerAttribute(std::uint64_t sourceAttributeHandle,
                           std::uint64_t sourceOwnerClassHandle,
                           RecordedAttributeKind kind,
                           RecordedDataType dataType,
                           bool hasElementCount,
                           std::uint32_t elementCount,
                           std::uint64_t sourceAttributeClassHandle,
                           std::uint64_t sourceFlagsEnumClassHandle,
                           std::uint32_t enumValue,
                           const std::string& name,
                           AttributeId& recordedAttribute,
                           PropertyId& valueProperty);
    bool registerObject(ObjectId object,
                        std::uint64_t sourceClassHandle,
                        const std::string& recordedName);
    AttributeId attributeIdForSource(std::uint64_t sourceAttributeHandle) const;
    PropertyId propertyIdForSource(std::uint64_t sourceAttributeHandle) const;
    const RecordedAttribute* findAttributeBySource(
        std::uint64_t sourceAttributeHandle) const;
    ModelBuildError error() const;
    std::shared_ptr<const RecordedClassModel> freeze();

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace detail
} // namespace data
} // namespace ovdnext

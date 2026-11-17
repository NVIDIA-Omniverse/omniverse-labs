/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "OvdRecordedModelBuilder.h"

#include <unordered_map>
#include <utility>
#include <vector>

namespace ovdnext
{
namespace data
{
namespace detail
{

struct RecordedClassModelBuilder::Impl
{
    struct SourceDeclaration
    {
        std::uint64_t baseHandle;
        std::string name;
        std::size_t classIndex;
    };

    struct SourceAttribute
    {
        std::size_t attributeIndex;
    };

    bool fail(ModelBuildError value)
    {
        if (buildError == ModelBuildError::None)
        {
            buildError = value;
        }
        return false;
    }

    std::vector<RecordedClass> classes;
    std::vector<RecordedAttribute> attributes;
    std::vector<RecordedObject> objects;
    std::unordered_map<std::uint64_t, SourceDeclaration> declarations;
    std::unordered_map<std::uint64_t, SourceAttribute> attributeDeclarations;
    PropertyId nextProperty = 1;
    ModelBuildError buildError = ModelBuildError::None;
};

namespace
{

bool isOrdinaryValueType(RecordedDataType dataType)
{
    switch (dataType)
    {
    case RecordedDataType::Int8:
    case RecordedDataType::Int16:
    case RecordedDataType::Int32:
    case RecordedDataType::Int64:
    case RecordedDataType::UInt8:
    case RecordedDataType::UInt16:
    case RecordedDataType::UInt32:
    case RecordedDataType::UInt64:
    case RecordedDataType::Float32:
    case RecordedDataType::Float64:
    case RecordedDataType::String:
    case RecordedDataType::ObjectHandle:
        return true;
    case RecordedDataType::None:
    case RecordedDataType::EnumValue:
    case RecordedDataType::FlagsWord:
    default:
        return false;
    }
}

} // namespace

RecordedClassModelBuilder::RecordedClassModelBuilder()
    : mImpl(new Impl)
{
}

RecordedClassModelBuilder::~RecordedClassModelBuilder() = default;
RecordedClassModelBuilder::RecordedClassModelBuilder(RecordedClassModelBuilder&&) noexcept = default;
RecordedClassModelBuilder& RecordedClassModelBuilder::operator=(RecordedClassModelBuilder&&) noexcept = default;

bool RecordedClassModelBuilder::registerClass(std::uint64_t sourceClassHandle,
    std::uint64_t sourceBaseClassHandle, const std::string& name)
{
    if (mImpl->buildError != ModelBuildError::None)
    {
        return false;
    }
    if (sourceClassHandle == 0)
    {
        return mImpl->fail(ModelBuildError::InvalidSourceHandle);
    }
    if (sourceClassHandle == sourceBaseClassHandle)
    {
        return mImpl->fail(ModelBuildError::InheritanceCycle);
    }

    const auto existing = mImpl->declarations.find(sourceClassHandle);
    if (existing != mImpl->declarations.end())
    {
        return existing->second.baseHandle == sourceBaseClassHandle &&
                existing->second.name == name
            ? true : mImpl->fail(ModelBuildError::ConflictingClassRegistration);
    }

    ClassId baseClass = 0;
    if (sourceBaseClassHandle != 0)
    {
        const auto base = mImpl->declarations.find(sourceBaseClassHandle);
        if (base == mImpl->declarations.end())
        {
            return mImpl->fail(ModelBuildError::UnknownBaseClass);
        }
        baseClass = mImpl->classes[base->second.classIndex].id;
    }

    const ClassId id = static_cast<ClassId>(mImpl->classes.size() + 1);
    const std::size_t index = mImpl->classes.size();
    mImpl->classes.push_back(RecordedClass{id, baseClass, name, {}});
    mImpl->declarations.emplace(sourceClassHandle,
        Impl::SourceDeclaration{sourceBaseClassHandle, name, index});
    return true;
}

ClassId RecordedClassModelBuilder::classIdForSource(std::uint64_t sourceClassHandle) const
{
    const auto found = mImpl->declarations.find(sourceClassHandle);
    return found == mImpl->declarations.end()
        ? ClassId{0} : mImpl->classes[found->second.classIndex].id;
}

const RecordedClass* RecordedClassModelBuilder::findClassBySource(
    std::uint64_t sourceClassHandle) const
{
    const auto found = mImpl->declarations.find(sourceClassHandle);
    return found == mImpl->declarations.end()
        ? nullptr : &mImpl->classes[found->second.classIndex];
}

bool RecordedClassModelBuilder::registerAttribute(std::uint64_t sourceAttributeHandle,
    std::uint64_t sourceOwnerClassHandle, RecordedAttributeKind kind,
    RecordedDataType dataType, bool hasElementCount, std::uint32_t elementCount,
    std::uint64_t sourceAttributeClassHandle,
    std::uint64_t sourceFlagsEnumClassHandle, std::uint32_t enumValue,
    const std::string& name, AttributeId& recordedAttribute, PropertyId& valueProperty)
{
    recordedAttribute = 0;
    valueProperty = 0;
    if (mImpl->buildError != ModelBuildError::None)
    {
        return false;
    }
    if (sourceAttributeHandle == 0)
    {
        return mImpl->fail(ModelBuildError::InvalidSourceHandle);
    }
    if (mImpl->attributeDeclarations.count(sourceAttributeHandle) != 0)
    {
        return mImpl->fail(ModelBuildError::DuplicateAttributeRegistration);
    }

    const auto owner = mImpl->declarations.find(sourceOwnerClassHandle);
    if (owner == mImpl->declarations.end())
    {
        return mImpl->fail(ModelBuildError::UnknownAttributeOwner);
    }

    ClassId attributeClass = 0;
    ClassId flagsEnumClass = 0;
    bool allocatesProperty = false;
    bool valid = elementCount == 0 || hasElementCount;
    switch (kind)
    {
    case RecordedAttributeKind::Value:
        if (dataType == RecordedDataType::FlagsWord)
        {
            valid = valid && !hasElementCount && elementCount == 0 &&
                sourceAttributeClassHandle == 0 && sourceFlagsEnumClassHandle != 0 &&
                enumValue == 0;
            if (valid)
            {
                const auto related = mImpl->declarations.find(sourceFlagsEnumClassHandle);
                if (related == mImpl->declarations.end())
                {
                    return mImpl->fail(ModelBuildError::UnknownRelatedClass);
                }
                flagsEnumClass = mImpl->classes[related->second.classIndex].id;
            }
        }
        else
        {
            valid = valid && isOrdinaryValueType(dataType) && hasElementCount &&
                sourceAttributeClassHandle == 0 && sourceFlagsEnumClassHandle == 0 &&
                enumValue == 0;
        }
        allocatesProperty = true;
        break;
    case RecordedAttributeKind::ClassAttribute:
        valid = valid && dataType == RecordedDataType::None && !hasElementCount &&
            elementCount == 0 && sourceAttributeClassHandle != 0 &&
            sourceFlagsEnumClassHandle == 0 && enumValue == 0;
        if (valid)
        {
            const auto related = mImpl->declarations.find(sourceAttributeClassHandle);
            if (related == mImpl->declarations.end())
            {
                return mImpl->fail(ModelBuildError::UnknownRelatedClass);
            }
            attributeClass = mImpl->classes[related->second.classIndex].id;
        }
        break;
    case RecordedAttributeKind::UniqueList:
        valid = valid && isOrdinaryValueType(dataType) && !hasElementCount &&
            elementCount == 0 && sourceAttributeClassHandle == 0 &&
            sourceFlagsEnumClassHandle == 0 && enumValue == 0;
        allocatesProperty = true;
        break;
    case RecordedAttributeKind::EnumConstant:
        valid = valid && dataType == RecordedDataType::EnumValue && !hasElementCount &&
            elementCount == 0 && sourceAttributeClassHandle == 0 &&
            sourceFlagsEnumClassHandle == 0;
        break;
    default:
        valid = false;
        break;
    }
    if (!valid)
    {
        return mImpl->fail(ModelBuildError::InvalidAttributeDeclaration);
    }

    recordedAttribute = static_cast<AttributeId>(mImpl->attributes.size() + 1);
    valueProperty = allocatesProperty ? mImpl->nextProperty++ : PropertyId{0};
    const ClassId ownerClass = mImpl->classes[owner->second.classIndex].id;
    mImpl->attributes.push_back(RecordedAttribute{recordedAttribute, valueProperty,
        ownerClass, attributeClass, flagsEnumClass, kind, dataType, hasElementCount,
        elementCount, enumValue, name});
    mImpl->classes[owner->second.classIndex].declaredAttributes.push_back(recordedAttribute);
    mImpl->attributeDeclarations.emplace(sourceAttributeHandle,
        Impl::SourceAttribute{mImpl->attributes.size() - 1});
    return true;
}

bool RecordedClassModelBuilder::registerObject(ObjectId object,
    std::uint64_t sourceClassHandle, const std::string& recordedName)
{
    if (mImpl->buildError != ModelBuildError::None)
    {
        return false;
    }
    if (object == 0 || sourceClassHandle == 0)
    {
        return mImpl->fail(ModelBuildError::InvalidSourceHandle);
    }
    const auto recordedClass = mImpl->declarations.find(sourceClassHandle);
    if (recordedClass == mImpl->declarations.end())
    {
        return false;
    }
    mImpl->objects.push_back(RecordedObject{object,
        mImpl->classes[recordedClass->second.classIndex].id, recordedName});
    return true;
}

AttributeId RecordedClassModelBuilder::attributeIdForSource(
    std::uint64_t sourceAttributeHandle) const
{
    const auto found = mImpl->attributeDeclarations.find(sourceAttributeHandle);
    return found == mImpl->attributeDeclarations.end() ? AttributeId{0}
        : mImpl->attributes[found->second.attributeIndex].id;
}

PropertyId RecordedClassModelBuilder::propertyIdForSource(
    std::uint64_t sourceAttributeHandle) const
{
    const RecordedAttribute* attribute = findAttributeBySource(sourceAttributeHandle);
    return attribute == nullptr ? PropertyId{0} : attribute->valueProperty;
}

const RecordedAttribute* RecordedClassModelBuilder::findAttributeBySource(
    std::uint64_t sourceAttributeHandle) const
{
    const auto found = mImpl->attributeDeclarations.find(sourceAttributeHandle);
    return found == mImpl->attributeDeclarations.end() ? nullptr
        : &mImpl->attributes[found->second.attributeIndex];
}

ModelBuildError RecordedClassModelBuilder::error() const
{
    return mImpl->buildError;
}

std::shared_ptr<const RecordedClassModel> RecordedClassModelBuilder::freeze()
{
    if (mImpl->buildError != ModelBuildError::None)
    {
        return nullptr;
    }
    return std::shared_ptr<const RecordedClassModel>(
        new RecordedClassModel(mImpl->classes, mImpl->attributes, mImpl->objects));
}

} // namespace detail
} // namespace data
} // namespace ovdnext

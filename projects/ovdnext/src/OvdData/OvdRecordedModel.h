/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "OvdData.h"

#include <memory>
#include <string>
#include <vector>

namespace ovdnext
{
namespace data
{

namespace detail
{
class RecordedClassModelBuilder;
}

enum class RecordedAttributeKind
{
    Value,
    ClassAttribute,
    UniqueList,
    EnumConstant
};

enum class RecordedDataType
{
    None,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float32, Float64,
    String, ObjectHandle, EnumValue, FlagsWord
};

struct RecordedAttribute
{
    AttributeId id = 0;
    PropertyId valueProperty = 0;
    ClassId ownerClass = 0;
    ClassId attributeClass = 0;
    ClassId flagsEnumClass = 0;
    RecordedAttributeKind kind = RecordedAttributeKind::Value;
    RecordedDataType dataType = RecordedDataType::None;
    bool hasElementCount = false;
    std::uint32_t elementCount = 0;
    std::uint32_t enumValue = 0;
    std::string name;
};

struct RecordedObject
{
    ObjectId id = 0;
    ClassId recordedClass = 0;
    std::string recordedName;
};

struct RecordedClass
{
    ClassId id = 0;
    ClassId baseClass = 0;
    std::string name;
    std::vector<AttributeId> declaredAttributes;
};

class RecordedClassModel
{
public:
    const std::vector<RecordedClass>& classes() const;
    const std::vector<RecordedAttribute>& attributes() const;
    const std::vector<RecordedObject>& objects() const;
    const RecordedClass* findClass(ClassId id) const;
    const RecordedAttribute* findAttribute(AttributeId id) const;
    const RecordedObject* findObject(ObjectId id) const;
    std::vector<ClassId> findClassesByName(const std::string& name) const;
    std::vector<AttributeId> findDeclaredAttributesByName(
        ClassId ownerClass, const std::string& name) const;
    std::vector<AttributeId> visibleAttributes(ClassId recordedClass) const;
    bool isSameOrDerived(ClassId candidate, ClassId base) const;

private:
    struct Data;
    RecordedClassModel(std::vector<RecordedClass> classes,
                       std::vector<RecordedAttribute> attributes,
                       std::vector<RecordedObject> objects);
    std::shared_ptr<const Data> mData;
    friend class detail::RecordedClassModelBuilder;
};

} // namespace data
} // namespace ovdnext

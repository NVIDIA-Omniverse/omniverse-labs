/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "OvdRecordedModel.h"

#include <algorithm>
#include <unordered_map>
#include <utility>

namespace ovdnext
{
namespace data
{

struct RecordedClassModel::Data
{
    Data(std::vector<RecordedClass> recordedClasses,
         std::vector<RecordedAttribute> recordedAttributes,
         std::vector<RecordedObject> recordedObjects)
        : classes(std::move(recordedClasses))
        , attributes(std::move(recordedAttributes))
        , objects(std::move(recordedObjects))
    {
        for (std::size_t index = 0; index < classes.size(); ++index)
        {
            classIndex.emplace(classes[index].id, index);
        }
        for (std::size_t index = 0; index < attributes.size(); ++index)
        {
            attributeIndex.emplace(attributes[index].id, index);
        }
        for (std::size_t index = 0; index < objects.size(); ++index)
        {
            objectIndex.emplace(objects[index].id, index);
        }
    }

    std::vector<RecordedClass> classes;
    std::vector<RecordedAttribute> attributes;
    std::vector<RecordedObject> objects;
    std::unordered_map<ClassId, std::size_t> classIndex;
    std::unordered_map<AttributeId, std::size_t> attributeIndex;
    std::unordered_map<ObjectId, std::size_t> objectIndex;
};

RecordedClassModel::RecordedClassModel(std::vector<RecordedClass> classes,
    std::vector<RecordedAttribute> attributes, std::vector<RecordedObject> objects)
    : mData(new Data(std::move(classes), std::move(attributes), std::move(objects)))
{
}

const std::vector<RecordedAttribute>& RecordedClassModel::attributes() const
{
    return mData->attributes;
}

const std::vector<RecordedObject>& RecordedClassModel::objects() const
{
    return mData->objects;
}

const std::vector<RecordedClass>& RecordedClassModel::classes() const
{
    return mData->classes;
}

const RecordedAttribute* RecordedClassModel::findAttribute(AttributeId id) const
{
    const auto found = mData->attributeIndex.find(id);
    return found == mData->attributeIndex.end() ? nullptr : &mData->attributes[found->second];
}

const RecordedObject* RecordedClassModel::findObject(ObjectId id) const
{
    const auto found = mData->objectIndex.find(id);
    return found == mData->objectIndex.end() ? nullptr : &mData->objects[found->second];
}

const RecordedClass* RecordedClassModel::findClass(ClassId id) const
{
    const auto found = mData->classIndex.find(id);
    return found == mData->classIndex.end() ? nullptr : &mData->classes[found->second];
}

std::vector<AttributeId> RecordedClassModel::findDeclaredAttributesByName(
    ClassId ownerClass, const std::string& name) const
{
    std::vector<AttributeId> matches;
    const RecordedClass* recordedClass = findClass(ownerClass);
    if (recordedClass == nullptr)
    {
        return matches;
    }
    for (AttributeId id : recordedClass->declaredAttributes)
    {
        const RecordedAttribute* attribute = findAttribute(id);
        if (attribute != nullptr && attribute->name == name)
        {
            matches.push_back(id);
        }
    }
    return matches;
}

std::vector<AttributeId> RecordedClassModel::visibleAttributes(ClassId recordedClass) const
{
    std::vector<const RecordedClass*> chain;
    const RecordedClass* current = findClass(recordedClass);
    while (current != nullptr)
    {
        chain.push_back(current);
        current = current->baseClass == 0 ? nullptr : findClass(current->baseClass);
    }
    std::reverse(chain.begin(), chain.end());

    std::vector<AttributeId> result;
    for (const RecordedClass* declarationClass : chain)
    {
        result.insert(result.end(), declarationClass->declaredAttributes.begin(),
            declarationClass->declaredAttributes.end());
    }
    return result;
}

std::vector<ClassId> RecordedClassModel::findClassesByName(const std::string& name) const
{
    std::vector<ClassId> matches;
    for (const RecordedClass& recordedClass : mData->classes)
    {
        if (recordedClass.name == name)
        {
            matches.push_back(recordedClass.id);
        }
    }
    return matches;
}

bool RecordedClassModel::isSameOrDerived(ClassId candidate, ClassId base) const
{
    if (candidate == 0 || base == 0)
    {
        return false;
    }
    const RecordedClass* current = findClass(candidate);
    while (current != nullptr)
    {
        if (current->id == base)
        {
            return true;
        }
        current = current->baseClass == 0 ? nullptr : findClass(current->baseClass);
    }
    return false;
}

} // namespace data
} // namespace ovdnext

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "OvdDataProbe.h"

#include "OvdFileIngest.h"
#include "OvdRecordedModel.h"

#include <cstddef>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace
{

std::string escapeQuoted(const std::string& value)
{
    std::string escaped;
    for (char character : value)
    {
        switch (character)
        {
        case '"': escaped += "\\\""; break;
        case '\\': escaped += "\\\\"; break;
        case '\n': escaped += "\\n"; break;
        case '\r': escaped += "\\r"; break;
        case '\t': escaped += "\\t"; break;
        default: escaped += character; break;
        }
    }
    return escaped;
}

} // namespace

namespace ovdnext
{
namespace omnipvd
{

void writeExactClassAliveSummary(const data::FrameView& view,
                                 std::ostream& output)
{
    std::map<data::ClassId, std::size_t> aliveByClass;
    for (data::ObjectId object : view.aliveObjects())
    {
        const data::RecordedObject* recordedObject = view.recordedObject(object);
        if (recordedObject != nullptr)
        {
            ++aliveByClass[recordedObject->recordedClass];
        }
    }

    const std::shared_ptr<const data::RecordedClassModel> model =
        view.recordedClassModel();
    output << " classes=[";
    bool first = true;
    for (const auto& entry : aliveByClass)
    {
        const data::RecordedClass* recordedClass =
            model ? model->findClass(entry.first) : nullptr;
        if (recordedClass == nullptr)
        {
            continue;
        }
        if (!first)
        {
            output << ',';
        }
        first = false;
        output << "{id=" << entry.first
               << " name=\"" << escapeQuoted(recordedClass->name)
               << "\" alive=" << entry.second << '}';
    }
    output << ']';
}

int runProbe(int argc, const char* const argv[], std::ostream& output, std::ostream& error)
{
    if (argc != 2)
    {
        error << "Usage: ovdnext_data_probe <path-to-ovd>\n";
        return 2;
    }

    FileIngestResult result = ingestOvdFile(argv[1], data::LoadPlan::allProperties());
    if (result.error != FileIngestError::None || !result.capture)
    {
        error << "error: " << result.diagnostic << '\n';
        return 1;
    }

    const std::vector<data::FrameStamp>& frames = result.capture->frames();
    for (std::size_t index = 0; index != frames.size(); ++index)
    {
        const data::FrameStamp& stamp = frames[index];
        const std::shared_ptr<const data::FrameView> view = result.capture->view(index);
        output << "publication=" << stamp.publication
               << " segment=" << stamp.address.segment
               << " context=" << stamp.address.context
               << " frame=" << stamp.address.frame
               << " alive=" << view->aliveObjects().size();
        writeExactClassAliveSummary(*view, output);
        output << '\n';
    }
    return 0;
}

} // namespace omnipvd
} // namespace ovdnext

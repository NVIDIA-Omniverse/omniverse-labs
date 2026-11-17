/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "OvdDataProbe.h"
#include "OvdIngest.h"

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace
{

void check(bool value, const char* expression, const char* file, int line)
{
    if (!value)
    {
        std::fprintf(stderr, "%s:%d: check failed: %s\n", file, line, expression);
        std::abort();
    }
}

#undef assert
#define assert(expression) check(static_cast<bool>(expression), #expression, __FILE__, __LINE__)

class TemporaryFile
{
public:
    TemporaryFile()
        : mPath(std::string("ovdnext_probe_truncated_")
                + std::to_string(static_cast<unsigned long>(std::time(nullptr))) + ".ovd")
    {
    }

    ~TemporaryFile()
    {
        std::remove(mPath.c_str());
    }

    const std::string& path() const { return mPath; }

private:
    std::string mPath;
};

void copyPrefix(const char* source, const std::string& target, std::size_t byteCount)
{
    std::ifstream input(source, std::ios::binary);
    std::ofstream output(target.c_str(), std::ios::binary | std::ios::trunc);
    assert(input && output);
    std::vector<char> bytes(byteCount);
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    output.write(bytes.data(), input.gcount());
    assert(output.good());
}

int invoke(const std::vector<std::string>& arguments, std::string& output, std::string& error)
{
    std::vector<const char*> argv;
    for (const std::string& argument : arguments)
        argv.push_back(argument.c_str());

    std::ostringstream capturedOutput;
    std::ostringstream capturedError;
    const int result = ovdnext::omnipvd::runProbe(
        static_cast<int>(argv.size()), argv.data(), capturedOutput, capturedError);
    output = capturedOutput.str();
    error = capturedError.str();
    return result;
}

std::shared_ptr<const ovdnext::data::FrameView> escapedClassNameView()
{
    using namespace ovdnext::data;

    CommandIngestor ingestor(LoadPlan::allProperties());

    const std::string className = "quote\"backslash\\newline\ncarriage\rreturn\ttab";
    Command registerClass{};
    registerClass.kind = CommandKind::RegisterClass;
    registerClass.classHandle = 10;
    registerClass.name = className.c_str();
    assert(ingestor.consume(registerClass));

    Command createObject{};
    createObject.kind = CommandKind::CreateObject;
    createObject.classHandle = 10;
    createObject.objectHandle = 20;
    assert(ingestor.consume(createObject));

    Command startFrame{};
    startFrame.kind = CommandKind::StartFrame;
    startFrame.frame = 1;
    assert(ingestor.consume(startFrame));

    Command stopFrame{};
    stopFrame.kind = CommandKind::StopFrame;
    stopFrame.frame = 1;
    assert(ingestor.consume(stopFrame));

    assert(ingestor.finish());
    std::unique_ptr<Capture> capture = ingestor.takeCapture();
    assert(capture);
    return capture->view(0);
}

// Catches accepting a malformed invocation or emitting usage on stdout.
void testMissingPathPrintsExactUsage()
{
    std::string output;
    std::string error;
    const int result = invoke({"ovdnext_data_probe"}, output, error);
    assert(result == 2);
    assert(output.empty());
    assert(error == "Usage: ovdnext_data_probe <path-to-ovd>\n");
}

// Catches skipped publications, reordered frames, wrong addresses, or a wrong alive view.
void testScenePrintsOneCanonicalRowPerPublication()
{
    std::string output;
    std::string error;
    const int result = invoke(
        {"ovdnext_data_probe", OVDNEXT_PROBE_TEST_SCENE}, output, error);
    assert(result == 0);
    assert(error.empty());
    assert(output ==
        "publication=1 segment=0 context=1342900227584 frame=1 alive=8 classes=[{id=46 name=\"PxOmniPvdMetaData\" alive=1},{id=47 name=\"PxPhysics\" alive=1},{id=48 name=\"PxGpuDynamicsMemoryConfig\" alive=1},{id=49 name=\"PxScene\" alive=1},{id=51 name=\"PxMaterial\" alive=1},{id=61 name=\"PxRigidDynamic\" alive=1},{id=70 name=\"PxShape\" alive=1},{id=72 name=\"PxSphereGeometry\" alive=1}]\n"
        "publication=2 segment=0 context=1342900227584 frame=2 alive=8 classes=[{id=46 name=\"PxOmniPvdMetaData\" alive=1},{id=47 name=\"PxPhysics\" alive=1},{id=48 name=\"PxGpuDynamicsMemoryConfig\" alive=1},{id=49 name=\"PxScene\" alive=1},{id=51 name=\"PxMaterial\" alive=1},{id=61 name=\"PxRigidDynamic\" alive=1},{id=70 name=\"PxShape\" alive=1},{id=72 name=\"PxSphereGeometry\" alive=1}]\n"
        "publication=3 segment=0 context=1342900227584 frame=3 alive=7 classes=[{id=46 name=\"PxOmniPvdMetaData\" alive=1},{id=47 name=\"PxPhysics\" alive=1},{id=48 name=\"PxGpuDynamicsMemoryConfig\" alive=1},{id=51 name=\"PxMaterial\" alive=1},{id=61 name=\"PxRigidDynamic\" alive=1},{id=70 name=\"PxShape\" alive=1},{id=72 name=\"PxSphereGeometry\" alive=1}]\n");
}

// Catches unescaped quoted class names or a formatter that does not use exact leaf classes.
void testExactClassSummaryEscapesRecordedClassNames()
{
    const std::shared_ptr<const ovdnext::data::FrameView> view = escapedClassNameView();
    assert(view);

    std::ostringstream output;
    ovdnext::omnipvd::writeExactClassAliveSummary(*view, output);
    assert(output.str() ==
        " classes=[{id=1 name=\"quote\\\"backslash\\\\newline\\ncarriage\\rreturn\\ttab\" alive=1}]");
}

// Catches collapsing adapter rejection into success or losing the stable diagnostic.
void testTruncatedInputPrintsExactError()
{
    const TemporaryFile file;
    copyPrefix(OVDNEXT_PROBE_TEST_SCENE, file.path(), 16);

    std::string output;
    std::string error;
    const int result = invoke({"ovdnext_data_probe", file.path()}, output, error);
    assert(result == 1);
    assert(output.empty());
    assert(error == "error: truncated OVD input\n");
}

} // namespace

int main()
{
    testMissingPathPrintsExactUsage();
    testScenePrintsOneCanonicalRowPerPublication();
    testExactClassSummaryEscapesRecordedClassNames();
    testTruncatedInputPrintsExactError();
    return 0;
}

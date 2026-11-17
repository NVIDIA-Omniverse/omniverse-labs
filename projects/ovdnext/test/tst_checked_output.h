/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "PvdCheckedOutput.h"
#include <gtest/gtest.h>
#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>

namespace {
struct OutputTestDirectory {
    std::filesystem::path path;
    OutputTestDirectory() {
        std::random_device random;
        for(int attempt = 0; attempt < 64; ++attempt) {
            const auto candidate = std::filesystem::temp_directory_path() /
                ("ovd-output-test-" + std::to_string(random()) + "-" + std::to_string(random()));
            if(std::filesystem::create_directory(candidate)) {
                path = candidate;
                return;
            }
        }
        throw std::runtime_error("Cannot create exclusive test directory");
    }
    ~OutputTestDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
    OutputTestDirectory(const OutputTestDirectory&) = delete;
    OutputTestDirectory& operator=(const OutputTestDirectory&) = delete;
};
std::string read(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if(!stream) throw std::runtime_error("Cannot read test file");
    return std::string(std::istreambuf_iterator<char>(stream), {});
}
void write(const std::filesystem::path& path, const std::string& bytes) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(bytes.data(), bytes.size());
    stream.close();
    if(!stream) throw std::runtime_error("Cannot write test file");
}
}

TEST(CheckedOutputTest, AlignedPageStillWritesAFullPaddingPage)
{
    OutputTestDirectory directory;
    ovdnext::OutputFile output((directory.path / "aligned.pxd2").u8string(), false);
    const std::array<uint8_t, 4096> bytes{};
    ASSERT_TRUE(output.write(bytes.data(), uint32_t(bytes.size())));
    output.zeroPadToPageSize();
    EXPECT_EQ(output.getLength(), 8192u);
}

TEST(CheckedOutputTest, CopyFailureRemainsStickyThroughScalarWritesAndCommit)
{
    OutputTestDirectory directory;
    const auto path = directory.path / "existing.pxd2";
    write(path, "keep");
    struct ShortInput : physx::pvdsdk::PvdInputStream {
        bool read(uint8_t*, uint32_t& count) override { count = 0; return false; }
    } input;
    {
        ovdnext::OutputFile output(path.u8string(), true);
        const uint8_t byte = 7;
        ASSERT_TRUE(output.write(&byte, 1));
        EXPECT_FALSE(output.directCopy(input, uint32_t(4)));
        EXPECT_FALSE(output.write(&byte, 1));
        EXPECT_FALSE(output.directCopy(input, uint64_t(4)));
        EXPECT_NO_THROW(output << uint32_t(42));
        output.zeroPadToPageSize();
        EXPECT_TRUE(output.flush().hasValue());
        EXPECT_TRUE(output.asyncFlush().hasValue());
        EXPECT_THROW(output.commit(), std::runtime_error);
    }
    EXPECT_EQ(read(path), "keep");
    EXPECT_EQ(std::distance(std::filesystem::directory_iterator(directory.path),
                            std::filesystem::directory_iterator()), 1);
}

TEST(CheckedOutputTest, PublicationRequiresCommitAndNeverClobbersWhenOverwriteIsFalse)
{
    OutputTestDirectory directory;
    const auto path = directory.path / "result.pxd2";
    const uint8_t bytes[] = {1, 2, 3};
    { ovdnext::OutputFile output(path.u8string(), false); ASSERT_TRUE(output.write(bytes, 3)); }
    EXPECT_FALSE(std::filesystem::exists(path));
    { ovdnext::OutputFile output(path.u8string(), false); ASSERT_TRUE(output.write(bytes, 3)); EXPECT_EQ(output.commit(), 3u); }
    EXPECT_THROW((ovdnext::OutputFile(path.u8string(), false)), std::runtime_error);
    { ovdnext::OutputFile output(path.u8string(), true); ASSERT_TRUE(output.write(bytes, 1)); }
    EXPECT_EQ(read(path), std::string(reinterpret_cast<const char*>(bytes), 3));
    const auto raced = directory.path / "raced.pxd2";
    {
        ovdnext::OutputFile output(raced.u8string(), false);
        ASSERT_TRUE(output.write(bytes, 3));
        write(raced, "winner");
        EXPECT_THROW(output.commit(), std::runtime_error);
    }
    EXPECT_EQ(read(raced), "winner");
    EXPECT_EQ(std::distance(std::filesystem::directory_iterator(directory.path),
                            std::filesystem::directory_iterator()), 2);
}


TEST(CheckedOutputTest, EmptyOverwriteKeepsDestination) {
    OutputTestDirectory directory;
    const auto path = directory.path / "existing.pxd2";
    write(path, "keep");
    { ovdnext::OutputFile output(path.u8string(), true);
      EXPECT_THROW(output.commit(), std::runtime_error); }
    EXPECT_EQ(read(path), "keep");
    EXPECT_EQ(std::distance(std::filesystem::directory_iterator(directory.path),
                            std::filesystem::directory_iterator()), 1);
}
TEST(CheckedOutputTest, Utf8DestinationPublishesExactBytes) {
    OutputTestDirectory directory;
    const auto path = directory.path / std::filesystem::u8path(u8"capture-\u00e9-\U0001f30d.pxd2");
    const uint8_t bytes[] = {0, 1, 127, 255};
    { ovdnext::OutputFile output(path.u8string(), false);
      ASSERT_TRUE(output.write(bytes, 4)); EXPECT_EQ(output.commit(), 4u); }
    EXPECT_EQ(read(path), std::string(reinterpret_cast<const char*>(bytes), 4));
}
TEST(CheckedOutputTest, DirectoryDestinationIsNeverReplaced) {
    OutputTestDirectory directory;
    const auto path = directory.path / "existing-directory";
    ASSERT_TRUE(std::filesystem::create_directory(path));
    EXPECT_THROW(( [&] { ovdnext::OutputFile output(path.u8string(), true);
                        const uint8_t byte=1; ASSERT_TRUE(output.write(&byte, 1));
                        output.commit(); }() ), std::runtime_error);
    EXPECT_TRUE(std::filesystem::is_directory(path));
}

TEST(CheckedOutputTest, EmbeddedNullNeverTouchesTheTruncatedDestination) {
    OutputTestDirectory directory;
    const auto existing = directory.path / "existing.pxd2";
    const auto missing = directory.path / "missing.pxd2";
    write(existing, "keep");
    for(bool overwrite : {false, true}) {
        for(const auto& path : {existing, missing}) {
            const std::string invalid = path.u8string() + std::string("\0suffix", 7);
            EXPECT_THROW((ovdnext::OutputFile(invalid, overwrite)), std::invalid_argument);
            EXPECT_EQ(read(existing), "keep");
            EXPECT_FALSE(std::filesystem::exists(missing));
            EXPECT_EQ(std::distance(std::filesystem::directory_iterator(directory.path),
                                    std::filesystem::directory_iterator()), 1);
        }
    }
}

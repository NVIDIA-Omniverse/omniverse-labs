// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    // GTest has written the test result by this point. Exit without running
    // late plugin/static destructors that can turn a passed suite into SIGSEGV.
    std::cout.flush();
    std::cerr.flush();
    std::clog.flush();
    std::fflush(nullptr);
    std::_Exit(result);
}

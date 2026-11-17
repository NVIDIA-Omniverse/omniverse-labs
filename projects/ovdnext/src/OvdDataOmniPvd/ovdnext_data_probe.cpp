/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "OvdDataProbe.h"

#include <iostream>

int main(int argc, char* argv[])
{
    return ovdnext::omnipvd::runProbe(argc, argv, std::cout, std::cerr);
}

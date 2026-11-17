/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifdef PVD_OVD_SUPPORT

#include "OvdMessageCollector.h"

#include <iterator>
#include <mutex>

namespace
{
std::mutex& parserMessageMutex()
{
    static std::mutex mutex;
    return mutex;
}
}

void pvdapp::applyOvdDomCommandAndCollect(
    OmniPvdReader& reader, OmniPvdDOMState& dom, OmniPvdCommand::Enum command,
    std::vector<OmniPvdMessage>& messages)
{
    std::lock_guard<std::mutex> lock(parserMessageMutex());
    gOmniPvdMessages.clear();
    applyPvdDomCommand(reader, dom, command);
    messages.insert(messages.end(),
                    std::make_move_iterator(gOmniPvdMessages.begin()),
                    std::make_move_iterator(gOmniPvdMessages.end()));
    gOmniPvdMessages.clear();
}

void pvdapp::clearOvdParserMessages()
{
    std::lock_guard<std::mutex> lock(parserMessageMutex());
    gOmniPvdMessages.clear();
}

#endif // PVD_OVD_SUPPORT

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PVD_OVD_MESSAGE_COLLECTOR_H
#define PVD_OVD_MESSAGE_COLLECTOR_H

#ifdef PVD_OVD_SUPPORT

#include "OmniPvdCommands.h"
#include "PvdDomParser.h"

#include <vector>

class OmniPvdReader;
class OmniPvdDOMState;

namespace pvdapp
{

// pvddom exposes diagnostic messages through one process-global vector.
// Isolate each parser command behind one mutex and drain that global into the
// engine-owned sink before another file or live source can enter the parser.
void applyOvdDomCommandAndCollect(OmniPvdReader& reader, OmniPvdDOMState& dom,
                                  OmniPvdCommand::Enum command,
                                  std::vector<OmniPvdMessage>& messages);

void clearOvdParserMessages();

} // namespace pvdapp

#endif // PVD_OVD_SUPPORT
#endif // PVD_OVD_MESSAGE_COLLECTOR_H

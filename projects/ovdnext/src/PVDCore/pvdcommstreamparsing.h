/*
 * SPDX-FileCopyrightText: Copyright (c) 2008-2014 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Copyright (c) 2008-2014 NVIDIA Corporation. All rights reserved.
#ifndef PVD_COMM_STREAM_PARSING_H
#define PVD_COMM_STREAM_PARSING_H

#include "PvdSimpleFoundation.h"
#include "PxPvdObjectModelInternalTypes.h"

using namespace physx;

namespace physx
{
namespace pvdsdk
{
class PvdObjectModelMutator;
class PvdObjectModelMetaData;
class PvdInputStream;
class PvdOutputStream;
} // namespace pvdsdk
} // namespace physx
namespace pvdapp
{
class PvdCommStreamMetaDataMutator;

using namespace physx::pvdsdk;

struct PvdCommStreamParseResult
{
	enum Enum
	{
		Success,
		Failed
	};
};

struct PvdCommStreamReadResult
{
	enum Enum
	{
		Success,
		Failed
	};
};

// The parser does interact with the database, so it really cannot be on
// another thread or its interaction needs to be protected.
class PvdCommStreamParser
{
  protected:
	virtual ~PvdCommStreamParser() {}

  public:
	virtual bool parseNextEventGroup(DataRef<const PxU8> eventData) = 0;
	virtual void release() = 0;

	// Appended after the legacy virtuals to preserve their slot order and source
	// overrides. Subclasses must still be recompiled for this extended interface.
	virtual PvdCommStreamParseResult::Enum parseNextEventGroupResult(DataRef<const PxU8> eventData)
	{
		return parseNextEventGroup(eventData)
			? PvdCommStreamParseResult::Success
			: PvdCommStreamParseResult::Failed;
	}

	// Called when the containing input reaches ordinary EOF. Legacy parsers do
	// not expose protocol state, so they retain their previous EOF behavior.
	virtual PvdCommStreamParseResult::Enum finalizeParsingResult()
	{
		return PvdCommStreamParseResult::Success;
	}
};

// The reader can sit and just read data over and over again.
// It doesn't interact with the database, so we can put it on another thread
// if desired.
class PvdCommStreamEventReader
{
  protected:
	virtual ~PvdCommStreamEventReader() {}

  public:
	static bool isPvdStream(const PxU32 firstWord);
	// Read the next event and place the data, unchanged or endian converted, into result.
	virtual bool readNextEventGroup(PvdInputStream& inStream, PvdOutputStream& result) = 0;
	virtual bool isSwappingBytes() = 0;
	virtual void release() = 0;
	virtual PvdCommStreamParser& createParser(PvdObjectModelMetaData& metaData, PvdObjectModelMutator& model, PvdCommStreamMetaDataMutator& commStreamMeta) = 0;

	// Explicit result API appended after the original virtual method sequence.
	virtual PvdCommStreamReadResult::Enum readNextEventGroupResult(PvdInputStream& inStream, PvdOutputStream& result)
	{
		return readNextEventGroup(inStream, result)
			? PvdCommStreamReadResult::Success
			: PvdCommStreamReadResult::Failed;
	}

	static PvdCommStreamEventReader* create(PxAllocatorCallback& alloc, PvdInputStream& inStream, const PxU32 inFirstWord);
};
} // namespace pvdapp

#endif

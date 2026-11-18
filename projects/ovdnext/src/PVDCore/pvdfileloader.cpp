/*
 * SPDX-FileCopyrightText: Copyright (c) 2008-2014 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Copyright (c) 2008-2014 NVIDIA Corporation. All rights reserved.
#include "pvdfileloader.h"

#include "PVDClient.h"
#include "pvdcommstreamparsing.h"
#include "PvdRandomAccessIOStream.h"
#include "PvdObjectModelEventStream.h"
#include "PxPvdInternalByteStreams.h"

using namespace pvdapp;

namespace
{

class FileStreamLoader : public PvdFileStreamLoader
{
	//    PxAllocatorCallback& mAllocator;
	IOStreamToInputStream mInStream;
	PvdClientMutator& mClient;
	MemPvdOutputStream mEventStream;

	PvdCommStreamEventReader* mReader;
	PvdCommStreamParser* mParser;
	bool mCompleted;
	bool mFailed;

	PvdFileStreamLoadResult completeAtEof(PxU32 bytesProcessed)
	{
		if(!mParser ||
		   mParser->finalizeParsingResult() == PvdCommStreamParseResult::Failed)
		{
			mFailed = true;
			return PvdFileStreamLoadResult(
				PvdFileStreamLoadResult::Failed, bytesProcessed);
		}
		mClient.getEventStreamFactory()->flush();
		mCompleted = true;
		return PvdFileStreamLoadResult(
			PvdFileStreamLoadResult::Completed, bytesProcessed);
	}

  public:
	FileStreamLoader(PxAllocatorCallback& alloc, PxU32 firstWord, PvdRandomAccessIOStream& inputStream, PvdRandomAccessIOStream& backingStream)
	//    : mAllocator(alloc)
	: mInStream(4, inputStream)
	, mClient(PvdClientMutator::create(alloc, backingStream, false))
	, mEventStream("FileStreamLoader::mEventStream")
	, mReader(PvdCommStreamEventReader::create(alloc, mInStream, firstWord))
	, mParser(NULL)
	, mCompleted(false)
	, mFailed(false)
	{
		if(mReader != NULL)
			mParser = &mClient.createParser(*mReader);
		mClient.addRef();
	}
	virtual ~FileStreamLoader()
	{
		mInStream.mStream.release();
		mClient.release();
		if(mReader)
			mReader->release();
		if(mParser)
			mParser->release();
	}

	virtual PxU32 loadNextChunk(PxU32 inRequestedAmountInBytes)
	{
		const PvdFileStreamLoadResult result = loadNextChunkResult(inRequestedAmountInBytes);
		return result.mStatus == PvdFileStreamLoadResult::Failed
			? 0
			: result.mBytesProcessed;
	}

	virtual PvdFileStreamLoadResult loadNextChunkResult(PxU32 inRequestedAmountInBytes)
	{
		if(mFailed)
			return PvdFileStreamLoadResult(PvdFileStreamLoadResult::Failed, 0);
		if(!mReader || !mParser)
		{
			mFailed = true;
			return PvdFileStreamLoadResult(PvdFileStreamLoadResult::Failed, 0);
		}
		if(mCompleted)
			return PvdFileStreamLoadResult(PvdFileStreamLoadResult::Completed, 0);
		if(mInStream.mOffset >= mInStream.mLen)
			return completeAtEof(0);

		PxU64 current = mInStream.mOffset;
		while((mInStream.mOffset == current ||
		       static_cast<PxU32>(mInStream.mOffset - current) < inRequestedAmountInBytes) &&
		      mInStream.mOffset < mInStream.mLen && mReader)
		{
			mEventStream.clear();
			if(mReader->readNextEventGroupResult(mInStream, mEventStream) !=
			   PvdCommStreamReadResult::Success)
			{
				mFailed = true;
				return PvdFileStreamLoadResult(
					PvdFileStreamLoadResult::Failed,
					static_cast<PxU32>(mInStream.mOffset - current));
			}
			const PvdCommStreamParseResult::Enum parseResult =
				mParser->parseNextEventGroupResult(mEventStream.toRef());
			if(parseResult == PvdCommStreamParseResult::Failed)
			{
				mFailed = true;
				return PvdFileStreamLoadResult(
					PvdFileStreamLoadResult::Failed,
					static_cast<PxU32>(mInStream.mOffset - current));
			}
			mEventStream.clear();
		}
		const PxU32 bytesProcessed = static_cast<PxU32>(mInStream.mOffset - current);
		if(mInStream.mOffset >= mInStream.mLen)
			return completeAtEof(bytesProcessed);
		return PvdFileStreamLoadResult(PvdFileStreamLoadResult::Loaded, bytesProcessed);
	}
	virtual PvdClientMutator& getClient() { return mClient; }
	virtual void release() { PVD_DELETE(this); }
};

class LoaderFactory : public PvdFileLoaderFactory
{
	PxAllocatorCallback& mAllocator;

  public:
	LoaderFactory(PxAllocatorCallback& alloc) : mAllocator(alloc) {}

	virtual PvdFileStreamLoader* createFileStreamLoader(PvdRandomAccessIOStream& stream, PvdRandomAccessIOStream& backingStream)
	{
		PxU32 firstWord;
		stream.read(reinterpret_cast<PxU8*>(&firstWord), 4, 0);
		PvdFileTypes::Enum ftype = getFileType(firstWord);
		if(ftype != PvdFileTypes::Stream)
		{
			PX_ASSERT(false);
			return nullptr;
		}
		return PVD_NEW(FileStreamLoader)(mAllocator, firstWord, stream, backingStream);
	}
	virtual PvdFileTypes::Enum getFileType(const PxU32 inFirstFileWord)
	{
		if(PvdCommStreamEventReader::isPvdStream(inFirstFileWord))
			return PvdFileTypes::Stream;
		if(inFirstFileWord == PvdClient::getFileId())
			return PvdFileTypes::Database;
		return PvdFileTypes::Unknown;
	}
	virtual void release() { PVD_DELETE(this); }
};
} // namespace

PvdFileLoaderFactory& PvdFileLoaderFactory::createFileLoaderFactory(PxAllocatorCallback& alloc) { return *PVD_NEW(LoaderFactory)(alloc); }

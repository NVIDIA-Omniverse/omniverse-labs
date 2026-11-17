/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "PvdRandomAccessIOStream.h"
#include "PsMutex.h"
#include "PsTime.h"
#include "PxPvdFoundation.h"
#include "PxPvdObjectModelInternalTypes.h"
#include <cerrno>
#include <filesystem>
#include <limits>
#include <stdio.h>
#include <string>
#include <vector>

// strsafe.h / windows.h are used by the Win32 implementation below. On other
// platforms (macOS, Linux) the Apple-ish POSIX implementation further down
// handles it. Gate on _WIN32 instead of !__APPLE__ which spuriously matched
// Linux too.
#ifdef _WIN32
#include <strsafe.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace pvdapp;
using namespace physx::shdfnd;
using namespace physx::pvdsdk;

// POSIX (macOS + Linux) implementation. Used to be gated on __APPLE__ only,
// which left Linux with no implementation at all — the Windows branch below
// was the sole fallback. Now any non-_WIN32 build uses this POSIX path.
#ifndef _WIN32
namespace
{
class RandomAccessIOStreamImpl : public PvdRandomAccessIOStream
{
  private:
	FILE* mFile;
	PxU64 mFileLen;
	PxU32 mRefCount;
	std::string mFilename;
	bool mIsTemporary;
	mutable Mutex mMutex;

  public:
	RandomAccessIOStreamImpl(bool openForWrite, FILE* inFile, const char* filename, bool isTemporary)
	: mFile(inFile)
	, mFileLen(0ULL)
	, mRefCount(0)
	, mFilename(filename ? filename : "")
	, mIsTemporary(isTemporary)
	, mMutex("RandomAccessIOStreamImpl::mMutex")
	{
		if(openForWrite == false)
		{
			fseek(mFile, 0, SEEK_SET);
			long start = ftell(mFile);
			fseek(mFile, 0, SEEK_END);
			long end = ftell(mFile);
			mFileLen = end - start;
			fseek(mFile, 0, SEEK_SET);
		}
		else
		{
			//                PX_ASSERT(0);
		}
	}

	virtual ~RandomAccessIOStreamImpl()
	{
		if(mFile)
			fclose(mFile);
		if(mIsTemporary && !mFilename.empty())
			remove(mFilename.c_str());
	}

	// PvdOutputStream interface
  public:
	virtual bool write(const uint8_t* buffer, uint32_t len)
	{
		fseek(mFile, 0, SEEK_END);
		size_t wb = fwrite(buffer, 1, len, mFile);
		PX_ASSERT(wb == len);
		mFileLen += len;
		return true;
	}
	virtual bool directCopy(PvdInputStream& inStream, uint32_t len)
	{
		PX_UNUSED(inStream);
		PX_UNUSED(len);
		PX_ASSERT(false);
		return true;
	}

	// PvdRandomAccessIOStream interface
  public:
	virtual void addRef()
	{
		Mutex::ScopedLock theLock(mMutex);
		++mRefCount;
	}
	virtual void release()
	{
		PxU32 theRefCount = 0;
		{
			Mutex::ScopedLock theLock(mMutex);
			if(mRefCount)
				--mRefCount;
			theRefCount = mRefCount;
		}
		if(!theRefCount)
			PVD_DELETE(this);
	}
	virtual Option<ErrorCode> read(PxU8* buffer, PxU32 inLen, PxU64 inOffset)
	{
		int res = fseek(mFile, inOffset, SEEK_SET);
		PX_ASSERT(res == 0);
		size_t count = fread(buffer, 1, inLen, mFile);
		PX_ASSERT(count == inLen);
		return None();
	}
	virtual void zeroPadToPageSize(PxU32 inPageSize)
	{
		PxU64 paddedLen = (mFileLen + inPageSize) - mFileLen % inPageSize;
		PxU32 amountToWrite = static_cast<PxU32>(paddedLen - mFileLen);
		std::vector<char> buff(amountToWrite);
		write(reinterpret_cast<const PxU8*>(&buff[0]), amountToWrite);
	}
	virtual PxU64 getLength() const { return mFileLen; }
	virtual Option<ErrorCode> flush()
	{
		fflush(mFile);
		return None();
	}
	virtual Option<ErrorCode> asyncFlush()
	{
		fflush(mFile);
		return None();
	}
	virtual double getTimeCost() { return 1.0; }

	// PvdRandomAccessIOStream interface
  public:
	virtual bool directCopy(PvdInputStream& inStream, PxU64 inLen)
	{
		std::vector<uint8_t> buff(inLen);
		bool res = inStream.read(buff.data(), inLen);
		PX_ASSERT(res == true);
		res = write(static_cast<const uint8_t*>(buff.data()), buff.size());
		PX_ASSERT(res == true);
		return true;
	}
};
} // namespace

ErrorCodeOption<PvdRandomAccessIOStream*> PvdRandomAccessIOStream::open(PxAllocatorCallback& alloc, const char* inFilename, bool openForWrite, bool inIsTemporary)
{
	FILE* file = NULL;
	if(openForWrite)
	{
		if(inIsTemporary)
		{
			const int fd = ::open(inFilename, O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR);
			if(fd >= 0)
			{
				file = fdopen(fd, "wb+");
				if(!file)
				{
					::close(fd);
					remove(inFilename);
				}
			}
		}
		else
			file = fopen(inFilename, "wb+");
	}
	else
	{
		file = fopen(inFilename, "rb");
	}
	if(!file)
		return ErrorCodeOption<PvdRandomAccessIOStream*>(ErrorCode(static_cast<PxU32>(errno)));

	RandomAccessIOStreamImpl* newStream = (RandomAccessIOStreamImpl*)alloc.allocate(sizeof(RandomAccessIOStreamImpl), "RandomAccessIOStreamImpl", __FILE__, __LINE__);
	if(!newStream)
	{
		fclose(file);
		if(inIsTemporary)
			remove(inFilename);
		return ErrorCodeOption<PvdRandomAccessIOStream*>(ErrorCode(static_cast<PxU32>(ENOMEM)));
	}
	new(newStream) RandomAccessIOStreamImpl(openForWrite, file, inFilename, inIsTemporary);
	return newStream;
}

#endif

// Windows OVERLAPPED-I/O implementation — the !__APPLE__ guard used to cover
// this on Linux too, but OVERLAPPED / WaitForMultipleObjects are Win32-only.
// The Apple branch above provides a POSIX implementation that also works on
// Linux, so everything non-Windows falls through to that.
#if !defined(__APPLE__) && defined(_WIN32)
namespace
{

struct FileBuffer : public OVERLAPPED
{
	ForwardingMemoryBuffer mBuffer;
	PxU32 mIOSetIndex;
	FileBuffer() : mBuffer("FileBuffer::mBuffer"), mIOSetIndex(PX_MAX_U32)
	{
		OVERLAPPED* base = static_cast<OVERLAPPED*>(this);
		memset(base, 0, sizeof(OVERLAPPED));
	}
	~FileBuffer() {}

	bool isInFlight() { return mIOSetIndex != PX_MAX_U32; }

	DWORD clear()
	{
		mBuffer.clear();
		return ERROR_SUCCESS;
	}
};

struct FileBufferIOSetSetter
{
	void operator()(FileBuffer& buffer, PxU32 idx) { buffer.mIOSetIndex = idx; }
};

struct FileBufferIOSetGetter
{
	PxU32 operator()(FileBuffer& buffer) { return buffer.mIOSetIndex; }
};

typedef InvasiveSet<FileBuffer, FileBufferIOSetGetter, FileBufferIOSetSetter> TFileBufferIOSet;

#define TNumBuffers 16

// 1 Gigabyte
#define FILE_MAP_THRESHOLD 1024 * 1024 * 1024

class RandomAccessIOStreamImpl : public PvdRandomAccessIOStream
{

	PxAllocatorCallback& mAllocator;
	HANDLE mFile;
	HANDLE mIOCompletionPort;

	HANDLE mMapFile;
	LPVOID lpMapAddress;
	PxU8* pMappedData;

	mutable Mutex mMutex;
	Array<FileBuffer*> mBuffers;
	Array<FileBuffer*> mAvailableBuffers;
	PxU64 mFileLen;
	FileBuffer* mWriteBuffer;
	const PxU32 mMaxBufferLen;
	PxU32 mRefCount;
	bool mIsTemporaryFile;
	bool mIsFileMapped;
	bool mIsReadOnly;
	std::wstring mFilename;
	TFileBufferIOSet mOutstandingIOOps;
	Time::Second mTimeCost;
	Time mProfileTimer;

  public:
	RandomAccessIOStreamImpl(PxAllocatorCallback& inCallback, PxU32 inMaxBufferLen, std::wstring&& inFilename, bool openForWrite, bool inIsTemporary, HANDLE inFile, HANDLE inIOCompletionPort)
	: mAllocator(inCallback)
	, mFileLen(0)
	, mWriteBuffer(0)
	, mMutex("RandomAccessIOStreamImpl::mMutex")
	, mBuffers("RandomAccessIOStreamImpl::mBuffers")
	, mAvailableBuffers("RandomAccessIOStreamImpl::mAvailableBuffers")
	, mMaxBufferLen(inMaxBufferLen)
	, mRefCount(0)
	, mIsTemporaryFile(inIsTemporary)
	, mOutstandingIOOps("RandomAccessIOStreamImpl::mOutstandingIOOps")
	, mFile(inFile)
	, mIOCompletionPort(inIOCompletionPort)
	, mTimeCost(0.0)
	, mIsFileMapped(false)
	, mIsReadOnly(!openForWrite)
	, mFilename(std::move(inFilename))
	{
		if(openForWrite == false)
		{
			// set the write offset based on the final file position.
			DWORD sizeHigh = 0;
			DWORD sizeLow = GetFileSize(mFile, &sizeHigh);
			mFileLen = sizeLow;
			mFileLen += ((PxU64)sizeHigh) << 32;

			// Use file mapping for files less than 1 Gigabyte
			if(mFileLen && mFileLen < FILE_MAP_THRESHOLD)
			{
				mMapFile = CreateFileMapping(mFile, NULL, PAGE_READONLY, 0, (DWORD)mFileLen, NULL);
				PX_ASSERT(mMapFile);
				lpMapAddress = MapViewOfFile(mMapFile, FILE_MAP_READ, 0, 0, (size_t)mFileLen);
				PX_ASSERT(lpMapAddress);
				pMappedData = (PxU8*)lpMapAddress;
				mIsFileMapped = true;
			}
		}

		for(PxU32 idx = 0; idx < TNumBuffers; ++idx)
		{
			mBuffers.pushBack(PVD_NEW(FileBuffer)());
			mAvailableBuffers.pushBack(mBuffers[idx]);
		}
	}

	virtual ~RandomAccessIOStreamImpl()
	{
		if(mIsFileMapped == true)
		{
			if(lpMapAddress != NULL)
			{
				UnmapViewOfFile(lpMapAddress);
			}
			if(mMapFile != NULL)
			{
				CloseHandle(mMapFile);
			}
		}
		if(mFile != INVALID_HANDLE_VALUE)
		{
			// Wait for any pending io.
			flush();
			CloseHandle(mFile);
			if(mIsTemporaryFile)
				_wremove(mFilename.c_str());
		}
		mFile = INVALID_HANDLE_VALUE;
		if(mIOCompletionPort != NULL)
		{
			CloseHandle(mIOCompletionPort);
			mIOCompletionPort = NULL;
		}
		mOutstandingIOOps.clear();
		for(PxU32 idx = 0; idx < TNumBuffers; ++idx)
			PVD_DELETE(mBuffers[idx]);
	}

	virtual double getTimeCost()
	{
		Mutex::ScopedLock theLock(mMutex);
		return (double)mTimeCost;
	}

	virtual void addRef()
	{
		Mutex::ScopedLock theLock(mMutex);
		++mRefCount;
	}

	virtual void release()
	{
		PxU32 theRefCount = 0;
		{
			Mutex::ScopedLock theLock(mMutex);
			if(mRefCount)
				--mRefCount;
			theRefCount = mRefCount;
		}
		if(!theRefCount)
			PVD_DELETE(this);
	}

	FileBuffer* getNextQueuedBuffer()
	{
		LPOVERLAPPED theOverlapped = NULL;
		DWORD transferBytes = 0;
		ULONG_PTR transferKey = NULL;
		do
		{
			GetQueuedCompletionStatus(mIOCompletionPort, &transferBytes, &transferKey, &theOverlapped, INFINITE);
			PX_ASSERT(transferKey == reinterpret_cast<ULONG_PTR>(this));
		} while(theOverlapped == NULL);
		return static_cast<FileBuffer*>(theOverlapped);
	}

	// Buffer is in mAvailableBuffers when found.
	FileBuffer* waitForNextCompletedIOOp(PxU32& outOpLen)
	{
		outOpLen = 0;
		PxU32 theCount = 0;
		{
			Mutex::ScopedLock theLock(mMutex);
			theCount = mOutstandingIOOps.size();
		}
		if(theCount != 0)
		{
			FileBuffer* theBuffer = getNextQueuedBuffer();
			{
				Mutex::ScopedLock theLock(mMutex);
				PX_ASSERT(theBuffer->isInFlight());
				mOutstandingIOOps.remove(*theBuffer);
				outOpLen = static_cast<PxU32>(theBuffer->InternalHigh);
				theBuffer->clear();
				mAvailableBuffers.pushBack(theBuffer);
				return theBuffer;
			}
		}
		return NULL;
	}

	FileBuffer* nextAvailableBuffer()
	{
		FileBuffer* retval = NULL;
		do
		{
			{
				// atomically grab the next buffer and add it to the outstanding
				// io operation record.
				Mutex::ScopedLock theLock(mMutex);
				if(mAvailableBuffers.size())
				{
					retval = mAvailableBuffers.back();
					PX_ASSERT(retval->isInFlight() == false);
					mAvailableBuffers.popBack();
				}
			}
			if(retval == NULL)
			{
				PxU32 ignored;
				waitForNextCompletedIOOp(ignored);
			}
		} while(retval == NULL);

		return retval;
	}

	void addOutstandingIOOperation(FileBuffer* inBuffer)
	{
		Mutex::ScopedLock theLock(mMutex);
		mOutstandingIOOps.insert(*inBuffer);
	}

	virtual Option<ErrorCode> flush()
	{
		// keep anyone else from messing with the file while we do this.
		Mutex::ScopedLock theLock(mMutex);
		sendOutstandingWriteBuffer();
		PxU32 ignored;

		for(FileBuffer* theBuffer = waitForNextCompletedIOOp(ignored); theBuffer; theBuffer = waitForNextCompletedIOOp(ignored))
		{
			// empty for loop on purpose.
		}
		return None();
	}

	virtual Option<ErrorCode> read(PxU8* inBytes, PxU32 inLength, PxU64 inOffset)
	{
		if(inLength == 0)
			return None();
		if(inBytes == NULL)
			return ErrorCode(PX_MAX_U32);

		const PxU64 desiredEnd = inLength + inOffset;
		if(desiredEnd > getLength())
			return ErrorCode(PX_MAX_U32 - 1);

		if(mIsFileMapped == true)
		{
			physx::intrinsics::memCopy(inBytes, pMappedData + inOffset, inLength);
			return None();
		}

		// We have to flush before a read else we get hammered because at this point I am too
		// lazy to write the proper logic to mess with the write buffer along with the outstanding
		// buffers that haven't been cleared yet.
		flush();
		// Get the next available buffer.
		FileBuffer* nextBuffer = nextAvailableBuffer();
		// confirmed means you can actually read from the file itself.
		// unconfirmed means you have to read from the overlapped structure in flight.
		while(inLength)
		{
			nextBuffer->clear();
			nextBuffer->Offset = static_cast<PxU32>(inOffset);
			nextBuffer->OffsetHigh = static_cast<PxU32>(inOffset >> 32);

			DWORD readLen = 0;
			BOOL success = ReadFile(mFile, inBytes, static_cast<DWORD>(inLength), &readLen, nextBuffer);
			DWORD error = GetLastError();
			PxU32 readResultLen = 0;
			if(success || error == ERROR_IO_PENDING)
			{
				addOutstandingIOOperation(nextBuffer);
				bool containsIOOp = false;
				do
				{
					PxU32 lastOpLen = 0;
					{
						Mutex::ScopedLock theLock(mMutex);
						containsIOOp = nextBuffer->isInFlight();
					}
					if(containsIOOp)
						waitForNextCompletedIOOp(lastOpLen);
					else
						readResultLen = lastOpLen;
				} while(containsIOOp);
			}
			else
				return ErrorCode(error);
			if(readResultLen == 0)
			{
				if(success)
					readResultLen = readLen;
				else
					readResultLen = inLength; // assume we read everything.  Pending ops usually do.
			}
			inBytes += readResultLen;
			inLength -= readResultLen;
			inOffset += readResultLen;
		}
		return None();
	}

	Option<ErrorCode> sendOutstandingWriteBuffer()
	{
		Mutex::ScopedLock theLock(mMutex);
		if(mWriteBuffer != NULL && mWriteBuffer->mBuffer.size())
		{
			BOOL success = WriteFile(mFile, mWriteBuffer->mBuffer.begin(), mWriteBuffer->mBuffer.size(), NULL, mWriteBuffer);
			DWORD error = GetLastError();
			if(success || error == ERROR_IO_PENDING)
			{
				addOutstandingIOOperation(mWriteBuffer);
				mWriteBuffer = NULL;
			}
			else
				return ErrorCode(error);
		}
		return None();
	}

	template <typename TWriteOperator>
	void doWrite(TWriteOperator inOperator)
	{
		if(inOperator.getLength())
		{
			Mutex::ScopedLock theLock(mMutex);

			mProfileTimer.getElapsedSeconds();

			while(inOperator.getLength())
			{
				if(mWriteBuffer == NULL)
				{
					mWriteBuffer = nextAvailableBuffer();
					mWriteBuffer->Offset = static_cast<PxU32>(mFileLen);
					mWriteBuffer->OffsetHigh = static_cast<PxU32>(mFileLen >> 32);
					mWriteBuffer->mBuffer.clear();
				}

				PxU32 maxWritePossible = static_cast<PxU32>(mMaxBufferLen - static_cast<PxU32>(mWriteBuffer->mBuffer.size()));

				PxU32 amountToWrite = static_cast<PxU32>(PxMin((PxU64)inOperator.getLength(), (PxU64)maxWritePossible));

				inOperator.writeData(mWriteBuffer->mBuffer, amountToWrite);

				if(static_cast<PxU32>(mWriteBuffer->mBuffer.size()) == mMaxBufferLen)
					sendOutstandingWriteBuffer();

				mFileLen += amountToWrite;
			}

			mTimeCost += mProfileTimer.getElapsedSeconds();
		}
	}

	struct BytesWriteOperator
	{
		const PxU8* mBytes;
		PxU32 mLength;
		BytesWriteOperator(const PxU8* inBytes, PxU32 inLen) : mBytes(inBytes), mLength(inLen) {}

		PxU32 getLength() const { return mLength; }

		void writeData(ForwardingMemoryBuffer& ioList, PxU32 inLen)
		{
			ioList.write(mBytes, inLen);
			mLength -= inLen;
			mBytes += inLen;
		}
	};

	virtual bool write(const PxU8* inBytes, PxU32 inLength)
	{
		doWrite(BytesWriteOperator(inBytes, inLength));
		return true;
	}

	struct ZeroPadWriteOperator
	{
		PxU32 mLength;
		explicit ZeroPadWriteOperator(PxU32 inLen) : mLength(inLen) {}
		PxU32 getLength() const { return mLength; }
		void writeData(ForwardingMemoryBuffer& ioList, PxU32 inLen)
		{
			ioList.writeZeros(inLen);
			mLength -= inLen;
		}
	};

	virtual void zeroPadToPageSize(PxU32 inPageSize)
	{
		Mutex::ScopedLock theLock(mMutex);
		PxU64 paddedLen = (mFileLen + inPageSize) - mFileLen % inPageSize;
		PxU32 amountToWrite = static_cast<PxU32>(paddedLen - mFileLen);
		doWrite(ZeroPadWriteOperator(amountToWrite));
	}

	struct SerializerInStreamWriter
	{
		PxU64 mLength;
		PvdInputStream& mStream;
		SerializerInStreamWriter(PvdInputStream& s, PxU64 l) : mLength(l), mStream(s) {}

		PxU64 getLength() const { return mLength; }

		void writeData(ForwardingMemoryBuffer& ioList, PxU32 inLen)
		{
			PxU32 listSize = ioList.size();
			ioList.growBuf(inLen);
			mStream.read(reinterpret_cast<char*>(ioList.begin() + listSize), inLen);
			mLength -= inLen;
		}
	};

	virtual bool directCopy(PvdInputStream& inStream, PxU32 inLen)
	{

		doWrite(SerializerInStreamWriter(inStream, inLen));
		return true;
	}

	virtual bool directCopy(PvdInputStream& inStream, PxU64 inLen)
	{
		doWrite(SerializerInStreamWriter(inStream, inLen));
		return true;
	}

	virtual PxU64 getLength() const
	{
		if(mIsReadOnly == true)
		{
			return mFileLen;
		}
		Mutex::ScopedLock theLock(mMutex);
		return mFileLen;
	}

	virtual Option<ErrorCode> asyncFlush() { return sendOutstandingWriteBuffer(); }
};
} // namespace

#define TBufSize 0x200000

ErrorCodeOption<PvdRandomAccessIOStream*> PvdRandomAccessIOStream::open(PxAllocatorCallback& alloc, const char* inFilename, bool openForWrite, bool inIsTemporary)
{
	std::wstring nativeFilename = std::filesystem::u8path(inFilename).wstring();
	DWORD mask = GENERIC_READ;
	if(openForWrite)
		mask |= GENERIC_WRITE;
	HANDLE file = INVALID_HANDLE_VALUE;
	if(openForWrite)
	{
		const DWORD disposition = inIsTemporary ? CREATE_NEW : TRUNCATE_EXISTING;
		file = CreateFileW(nativeFilename.c_str(), mask, FILE_SHARE_READ, NULL, disposition,
		                  FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN, 0);
	}
	else
		file = CreateFileW(nativeFilename.c_str(), mask, FILE_SHARE_READ, NULL, OPEN_EXISTING,
		                  FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN, 0);
	if(file == INVALID_HANDLE_VALUE && openForWrite && !inIsTemporary)
		file = CreateFileW(nativeFilename.c_str(), mask, FILE_SHARE_READ, NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN, 0);

	if(file == INVALID_HANDLE_VALUE)
		return ErrorCodeOption<PvdRandomAccessIOStream*>(ErrorCode(GetLastError()));

	RandomAccessIOStreamImpl* newStream = (RandomAccessIOStreamImpl*)alloc.allocate(sizeof(RandomAccessIOStreamImpl), "RandomAccessIOStreamImpl", __FILE__, __LINE__);
	if(!newStream)
	{
		CloseHandle(file);
		if(inIsTemporary)
			_wremove(nativeFilename.c_str());
		return ErrorCodeOption<PvdRandomAccessIOStream*>(ErrorCode(ERROR_OUTOFMEMORY));
	}
	// We use buffers for everything, attempt to get to zero-copy writes and reads.
	HANDLE port = CreateIoCompletionPort(file, NULL, reinterpret_cast<ULONG_PTR>(newStream), 10);
	if(port == NULL)
	{
		const DWORD error = GetLastError();
		CloseHandle(file);
		if(inIsTemporary)
			_wremove(nativeFilename.c_str());
		alloc.deallocate(newStream);
		return ErrorCodeOption<PvdRandomAccessIOStream*>(ErrorCode(error));
	}
	new(newStream) RandomAccessIOStreamImpl(alloc, TBufSize, std::move(nativeFilename), openForWrite, inIsTemporary, file, port);
	return newStream;
}

#endif

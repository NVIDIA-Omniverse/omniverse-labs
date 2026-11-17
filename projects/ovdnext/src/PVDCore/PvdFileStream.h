/*
 * SPDX-FileCopyrightText: Copyright (c) 2008-2016 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Copyright (c) 2008-2016 NVIDIA Corporation. All rights reserved.



#ifndef PVD_FILE_STREAMS_H
#define PVD_FILE_STREAMS_H
#include "PxPvdByteStreams.h"
#include <stdio.h>

namespace pvdapp {

PX_INLINE PxI32 fopen_s(FILE** file, const char* name, const char* mode)
{
	static const PxU32 MAX_LEN = 300; 
	char buf[MAX_LEN+1];

	PxU32 i;
	for(i = 0; i<MAX_LEN && name[i]; i++)
		buf[i] = name[i] == '/' ? '\\' : name[i];
	buf[i] = 0;

	return i == MAX_LEN ? -1 : ::fopen_s(file, buf, mode);
};


class PvdFileOutputStream : public physx::pvdsdk::PvdOutputStream
{
public:
	PvdFileOutputStream(const char* filename)
	{
		mFile = NULL;
		pvdapp::fopen_s(&mFile, filename, "wb");
	}

	virtual	~PvdFileOutputStream()
	{
		if(mFile)
			fclose(mFile);
	}

	virtual bool write( const char* buffer, PxU32 len )
	{
		if(mFile)
		{
			if(len == fwrite(buffer, 1, len, mFile))
				return true;
		}
		return false;
	}

	virtual bool write( const PxU8* buffer, PxU32 len )
	{
		if(mFile)
		{
			if(len == fwrite(buffer, 1, len, mFile))
				return true;
		}
		return false;
	}

	virtual bool directCopy( physx::pvdsdk::PvdInputStream& inStream, PxU32 len )
	{
		PX_ASSERT(0);
		return false;
	}

private:
	FILE* mFile;
};

}


#endif

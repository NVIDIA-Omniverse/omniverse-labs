/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "profiler.h"
#include <assert.h>
#include <chrono>
#include <cstring>
#include <stdint.h>
#include <stdio.h>
#include <string>

namespace profiler
{
const size_t ProfilerThreadMaxEvents = 32 * 1024; // max this # of events per thread

struct ProfilerEvent
{
	uint64_t m_Time;
	uint64_t m_Duration;
	uint32_t m_Phase;
	std::string m_Name;
	std::string m_Info;
};

struct ProfilerThread
{
	std::string m_ScratchStrings;
	ProfilerEvent* m_Events;
	int m_EventCount;
	bool m_IsBegin;
};

struct ProfilerState
{
	char* m_FileName;
	ProfilerThread* m_Threads;
	int m_ThreadCount;
};

static ProfilerState s_ProfilerState;

bool g_ProfilerEnabled;

void ProfilerInit(const char* fileName, int threadCount)
{
	assert(!g_ProfilerEnabled);
	assert(threadCount > 0);

	g_ProfilerEnabled = true;

	s_ProfilerState.m_ThreadCount = threadCount;

	size_t fileNameLen = strlen(fileName);
	s_ProfilerState.m_FileName = new char[fileNameLen + 1];
	memcpy(s_ProfilerState.m_FileName, fileName, fileNameLen + 1);

	s_ProfilerState.m_Threads = new ProfilerThread[s_ProfilerState.m_ThreadCount];

	for(int i = 0; i < s_ProfilerState.m_ThreadCount; ++i)
	{
		ProfilerThread& thread = s_ProfilerState.m_Threads[i];
		thread.m_Events = new ProfilerEvent[ProfilerThreadMaxEvents];
		thread.m_EventCount = 0;
		thread.m_ScratchStrings = "profilerStrings";
		thread.m_IsBegin = false;
	}
}

static void EscapeString(const char* src, char* dst, int dstSpace)
{
	while(*src && dstSpace > 2) // some input chars might expand into 2 in the output
	{
		char c = *src;
		switch(c)
		{
		case '"':
		case '\\':
		case '\b':
		case '\f':
		case '\n':
		case '\r':
		case '\t':
			*dst++ = '\\';
			*dst++ = c;
			dstSpace -= 2;
			break;
		default:
			if(c >= 32 && c < 126)
			{
				*dst++ = c;
				dstSpace -= 1;
			}
		}
		++src;
	}
	*dst = 0;
}

void ProfilerWriteOutput()
{
	FILE* f = fopen(s_ProfilerState.m_FileName, "w");
	if(!f)
	{
		return;
	}

	fputs("{\n", f);
	fputs("\"instructions_readme\": \"1) Open Chrome, 2) go to chrome://tracing, 3) click Load, 4) navigate to this file.\",\n", f);
	fputs("\"traceEvents\":[\n", f);
	fputs("{ \"cat\":\"\", \"pid\":1, \"tid\":0, \"ts\":0, \"ph\":\"M\", \"name\":\"process_name\", \"args\": { \"name\":\"PVD\" } }\n", f);
	for(int i = 0; i < s_ProfilerState.m_ThreadCount; ++i)
	{
		const ProfilerThread& thread = s_ProfilerState.m_Threads[i];
		for(int j = 0; j < thread.m_EventCount; ++j)
		{
			const ProfilerEvent& evt = thread.m_Events[j];
			double timeUs = evt.m_Time;
			//        double durUs = evt.m_Duration;

			char phase = 0;
			if(evt.m_Phase == 0)
				phase = 'B';
			else
				phase = 'E';
			char name[1024];
			char info[1024];
			EscapeString(evt.m_Name.c_str(), name, sizeof(name));
			EscapeString(evt.m_Info.c_str(), info, sizeof(info));
			fprintf(f, ",{ \"pid\":1, \"tid\":%d, \"ts\":%.0f, \"ph\":\"%c\", \"name\": \"%s\", \"args\": { \"detail\":\"%s\" }}\n", i, timeUs, phase, name, info);
		}
	}
	fputs("\n]\n", f);
	fputs("}\n", f);

	fclose(f);
}

void ProfilerDestroy()
{
	if(!g_ProfilerEnabled)
		return;

	//    for (int i = 0; i < s_ProfilerState.m_ThreadCount; ++i)
	//    {
	//      ProfilerThread& thread = s_ProfilerState.m_Threads[i];
	//      if (thread.m_IsBegin)
	//        ProfilerEndImpl(i);
	//    }

	ProfilerWriteOutput();

	for(int i = 0; i < s_ProfilerState.m_ThreadCount; ++i)
	{
		ProfilerThread& thread = s_ProfilerState.m_Threads[i];
		delete[] thread.m_Events;
	}
	delete[] s_ProfilerState.m_Threads;
	delete[] s_ProfilerState.m_FileName;
}

void ProfilerBeginImpl(const char* name, int threadIndex)
{
	assert(g_ProfilerEnabled);
	assert(threadIndex >= 0 && threadIndex < s_ProfilerState.m_ThreadCount);
	ProfilerThread& thread = s_ProfilerState.m_Threads[threadIndex];
	//    assert(!thread.m_IsBegin);
	thread.m_IsBegin = true;
	if(thread.m_EventCount >= ProfilerThreadMaxEvents)
	{
		return;
	}
	ProfilerEvent& evt = thread.m_Events[thread.m_EventCount++];
	auto ts = std::chrono::high_resolution_clock::now();
	auto tsMs = std::chrono::duration_cast<std::chrono::microseconds>(ts.time_since_epoch());
	evt.m_Time = tsMs.count();
	evt.m_Phase = 0;
	// split input name by first space
	const char* nextWord = strchr(name, ' ');
	if(nextWord == nullptr)
	{
		evt.m_Name = std::string(name);
		evt.m_Info = "";
	}
	else
	{
		evt.m_Name = std::string(name);
		evt.m_Info = std::string(nextWord + 1);
	}
}

void ProfilerEndImpl(std::string& name, int threadIndex)
{
	assert(g_ProfilerEnabled);
	assert(threadIndex >= 0 && threadIndex < s_ProfilerState.m_ThreadCount);
	ProfilerThread& thread = s_ProfilerState.m_Threads[threadIndex];
	//    assert(thread.m_IsBegin);
	assert(thread.m_EventCount > 0);
	thread.m_IsBegin = false;
	if(thread.m_EventCount > ProfilerThreadMaxEvents)
		return;
	ProfilerEvent& evt = thread.m_Events[thread.m_EventCount++];
	auto ts = std::chrono::high_resolution_clock::now();
	auto tsMs = std::chrono::duration_cast<std::chrono::microseconds>(ts.time_since_epoch());
	evt.m_Name = name;
	evt.m_Phase = 1;
	evt.m_Time = tsMs.count();
	evt.m_Duration = tsMs.count() - evt.m_Time;
}
} // namespace profiler

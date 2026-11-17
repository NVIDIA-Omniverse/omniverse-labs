/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PROFILER_H
#define PROFILER_H

#include <string>

namespace profiler
{

extern bool g_ProfilerEnabled;

void ProfilerInit(const char* fileName, int threadCount);
void ProfilerDestroy();
void ProfilerBeginImpl(const char* name, int threadIndex);
void ProfilerEndImpl(std::string& name, int threadIndex);

inline void ProfilerBegin(const char* name, int threadIndex)
{
	if(g_ProfilerEnabled)
		ProfilerBeginImpl(name, threadIndex);
}

inline void ProfilerEnd(std::string name, int threadIndex)
{
	if(g_ProfilerEnabled)
		ProfilerEndImpl(name, threadIndex);
}

struct ProfilerScope
{
	int m_ThreadIndex;
	std::string m_Name;
	ProfilerScope(const char* name, int threadIndex)
	{
		m_ThreadIndex = threadIndex;
		m_Name = name;
		ProfilerBegin(name, threadIndex);
	}

	~ProfilerScope() { ProfilerEnd(m_Name, m_ThreadIndex); }
};

} // namespace profiler

#endif // PROFILER_H

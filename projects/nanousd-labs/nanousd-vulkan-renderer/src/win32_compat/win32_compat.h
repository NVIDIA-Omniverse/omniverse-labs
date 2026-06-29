// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <BaseTsd.h>
#include <direct.h>
#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <windows.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#ifndef _SSIZE_T_DEFINED
typedef SSIZE_T ssize_t;
#define _SSIZE_T_DEFINED
#endif

#ifndef _OFF_T_DEFINED
typedef __int64 off_t;
#define _OFF_T_DEFINED
#endif

#ifndef _TIMESPEC_DEFINED
struct timespec {
    time_t tv_sec;
    long tv_nsec;
};
#define _TIMESPEC_DEFINED
#endif

#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

static inline int clock_gettime(int clock_id, struct timespec* ts)
{
    (void)clock_id;
    if (!ts) {
        errno = EINVAL;
        return -1;
    }

    LARGE_INTEGER freq;
    LARGE_INTEGER now;
    if (!QueryPerformanceFrequency(&freq) || !QueryPerformanceCounter(&now)) {
        errno = EINVAL;
        return -1;
    }

    ts->tv_sec = (time_t)(now.QuadPart / freq.QuadPart);
    ts->tv_nsec = (long)(((now.QuadPart % freq.QuadPart) * 1000000000LL) /
                         freq.QuadPart);
    return 0;
}

#ifndef S_ISDIR
#define S_ISDIR(mode) (((mode) & _S_IFMT) == _S_IFDIR)
#endif
#ifndef S_ISREG
#define S_ISREG(mode) (((mode) & _S_IFMT) == _S_IFREG)
#endif

#ifndef O_RDONLY
#define O_RDONLY _O_RDONLY
#endif

#define stat _stat64
#define fstat _fstat64
#define lstat(path, st) _stat64((path), (st))

#ifndef __cplusplus
#define open(path, flags) _open((path), (flags) | _O_BINARY)
#define close(fd) _close((fd))
#define read(fd, buf, count) _read((fd), (buf), (unsigned int)(count))
#define mkdir(path, mode) _mkdir((path))
#define strcasecmp(a, b) _stricmp((a), (b))
#define strncasecmp(a, b, n) _strnicmp((a), (b), (n))
#endif

static inline ssize_t pread(int fd, void* buf, size_t count, off_t offset)
{
    intptr_t handle_value = _get_osfhandle(fd);
    if (handle_value == -1) {
        errno = EBADF;
        return -1;
    }

    OVERLAPPED ov;
    memset(&ov, 0, sizeof(ov));
    ov.Offset = (DWORD)((uint64_t)offset & 0xffffffffu);
    ov.OffsetHigh = (DWORD)(((uint64_t)offset >> 32) & 0xffffffffu);

    if (count > 0xffffffffu)
        count = 0xffffffffu;

    DWORD got = 0;
    if (!ReadFile((HANDLE)handle_value, buf, (DWORD)count, &got, &ov)) {
        DWORD err = GetLastError();
        if (err == ERROR_HANDLE_EOF)
            return 0;
        errno = EIO;
        return -1;
    }
    return (ssize_t)got;
}

static inline char* realpath(const char* path, char* resolved)
{
    if (!path || !resolved) {
        errno = EINVAL;
        return NULL;
    }
    return _fullpath(resolved, path, PATH_MAX);
}

static inline int setenv(const char* name, const char* value, int overwrite)
{
    if (!name || !name[0] || strchr(name, '=')) {
        errno = EINVAL;
        return -1;
    }
    if (!overwrite && getenv(name))
        return 0;
    return _putenv_s(name, value ? value : "");
}

static inline int unsetenv(const char* name)
{
    if (!name || !name[0] || strchr(name, '=')) {
        errno = EINVAL;
        return -1;
    }
    return _putenv_s(name, "");
}

static inline char* dirname(char* path)
{
    if (!path || !path[0])
        return (char*)".";

    size_t len = strlen(path);
    while (len > 1 && (path[len - 1] == '/' || path[len - 1] == '\\'))
        path[--len] = '\0';

    char* slash = strrchr(path, '/');
    char* backslash = strrchr(path, '\\');
    char* sep = slash > backslash ? slash : backslash;
    if (!sep) {
        strcpy(path, ".");
        return path;
    }

    if (sep == path || (sep == path + 2 && path[1] == ':')) {
        sep[1] = '\0';
        return path;
    }

    *sep = '\0';
    return path;
}

#endif /* _WIN32 */

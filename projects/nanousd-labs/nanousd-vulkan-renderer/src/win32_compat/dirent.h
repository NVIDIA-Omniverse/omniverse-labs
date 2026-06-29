// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "win32_compat.h"

#if defined(_WIN32)

#include <io.h>

struct dirent {
    char d_name[PATH_MAX];
};

typedef struct DIR {
    intptr_t handle;
    int first;
    struct _finddata_t data;
    struct dirent entry;
} DIR;

static inline DIR* opendir(const char* path)
{
    if (!path || !path[0]) {
        errno = ENOENT;
        return NULL;
    }

    size_t len = strlen(path);
    int has_sep = (path[len - 1] == '/' || path[len - 1] == '\\');
    size_t pattern_len = len + (has_sep ? 1u : 2u) + 1u;
    char* pattern = (char*)malloc(pattern_len);
    if (!pattern)
        return NULL;

    memcpy(pattern, path, len);
    if (!has_sep)
        pattern[len++] = '\\';
    pattern[len++] = '*';
    pattern[len] = '\0';

    DIR* dir = (DIR*)calloc(1, sizeof(*dir));
    if (!dir) {
        free(pattern);
        return NULL;
    }

    dir->handle = _findfirst(pattern, &dir->data);
    free(pattern);
    if (dir->handle == -1) {
        free(dir);
        return NULL;
    }
    dir->first = 1;
    return dir;
}

static inline struct dirent* readdir(DIR* dir)
{
    if (!dir) {
        errno = EBADF;
        return NULL;
    }

    if (dir->first) {
        dir->first = 0;
    } else if (_findnext(dir->handle, &dir->data) != 0) {
        return NULL;
    }

    strncpy(dir->entry.d_name, dir->data.name, sizeof(dir->entry.d_name) - 1u);
    dir->entry.d_name[sizeof(dir->entry.d_name) - 1u] = '\0';
    return &dir->entry;
}

static inline int closedir(DIR* dir)
{
    if (!dir) {
        errno = EBADF;
        return -1;
    }
    int rc = _findclose(dir->handle);
    free(dir);
    return rc;
}

#endif /* _WIN32 */

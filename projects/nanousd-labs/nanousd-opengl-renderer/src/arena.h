// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NUSD_ARENA_H
#define NUSD_ARENA_H

/*
 * arena.h — Header-only block-based arena allocator.
 *
 * Non-relocating: blocks are never moved once allocated, so pointers
 * from previous allocations remain valid for the arena's lifetime.
 * This is critical for zero-copy + arena mixed workflows.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ArenaBlock_ {
    struct ArenaBlock_* next;
    size_t              used;
    size_t              capacity;
    /* data follows (flexible array) */
} ArenaBlock_;

typedef struct {
    ArenaBlock_* head;
    ArenaBlock_* current;
    size_t       default_block_size;
} Arena;

static inline ArenaBlock_* arena_new_block_(size_t capacity) {
    ArenaBlock_* b = (ArenaBlock_*)malloc(sizeof(ArenaBlock_) + capacity);
    if (!b) return NULL;
    b->next = NULL;
    b->used = 0;
    b->capacity = capacity;
    return b;
}

static inline Arena arena_create(size_t capacity) {
    Arena a;
    a.default_block_size = capacity;
    a.head = arena_new_block_(capacity);
    a.current = a.head;
    return a;
}

static inline void* arena_alloc(Arena* a, size_t size, size_t align) {
    if (!a->current) return NULL;

    size_t mask = align - 1;
    char* block_data = (char*)(a->current + 1); /* data follows header */
    size_t aligned = (a->current->used + mask) & ~mask;

    if (aligned + size <= a->current->capacity) {
        void* ptr = block_data + aligned;
        a->current->used = aligned + size;
        return ptr;
    }

    /* Need a new block — size it to fit this allocation or default, whichever is larger */
    size_t new_cap = a->default_block_size;
    if (new_cap < size + align) new_cap = size + align;

    ArenaBlock_* nb = arena_new_block_(new_cap);
    if (!nb) return NULL;

    a->current->next = nb;
    a->current = nb;

    size_t aligned2 = (nb->used + mask) & ~mask;
    void* ptr = (char*)(nb + 1) + aligned2;
    nb->used = aligned2 + size;
    return ptr;
}

static inline void arena_reset(Arena* a) {
    ArenaBlock_* b = a->head;
    while (b) {
        b->used = 0;
        b = b->next;
    }
    a->current = a->head;
}

static inline void arena_destroy(Arena* a) {
    ArenaBlock_* b = a->head;
    while (b) {
        ArenaBlock_* next = b->next;
        free(b);
        b = next;
    }
    a->head = NULL;
    a->current = NULL;
}

/* Convenience: allocate and zero */
static inline void* arena_calloc(Arena* a, size_t count, size_t elem_size) {
    void* p = arena_alloc(a, count * elem_size, 8);
    if (p) memset(p, 0, count * elem_size);
    return p;
}

#ifdef __cplusplus
}
#endif

#endif /* NUSD_ARENA_H */

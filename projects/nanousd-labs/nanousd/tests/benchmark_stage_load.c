// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * benchmark_stage_load — Benchmark stage open, traversal, and diagnostics.
 *
 * Pure C11, links only nanousdapi — works with any backend via NANOUSD_BACKEND.
 *
 * Usage:
 *   ./benchmark_stage_load <root.usd>
 *   NANOUSD_BACKEND=./libopenusd_backend.so ./benchmark_stage_load <root.usd>
 */

#include "nanousd/nanousdapi.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * Cross-platform high-resolution timer
 * ============================================================ */

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static double now_ms(void) {
    static double freq = 0.0;
    LARGE_INTEGER li;
    if (freq == 0.0) {
        QueryPerformanceFrequency(&li);
        freq = (double)li.QuadPart / 1000.0;
    }
    QueryPerformanceCounter(&li);
    return (double)li.QuadPart / freq;
}
#else
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}
#endif

/* ============================================================
 * Cross-platform peak RSS (resident set size) in bytes
 * ============================================================ */

#ifdef _WIN32
#include <psapi.h>

static size_t current_rss(void) {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return (size_t)pmc.WorkingSetSize;
    return 0;
}

static size_t peak_rss(void) {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return (size_t)pmc.PeakWorkingSetSize;
    return 0;
}
#else
#include <sys/resource.h>

static size_t current_rss(void) {
    FILE* f = fopen("/proc/self/statm", "r");
    if (!f) return 0;
    long ignored = 0;
    long pages = 0;
    /* statm fields: size resident shared text lib data dt */
    if (fscanf(f, "%ld %ld", &ignored, &pages) != 2) pages = 0;
    fclose(f);
    return (size_t)pages * 4096;
}

static size_t peak_rss(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0)
        return (size_t)ru.ru_maxrss * 1024; /* ru_maxrss is in KB on Linux */
    return 0;
}
#endif

/* ============================================================
 * Helpers
 * ============================================================ */

static const char* severity_str(int s) {
    switch (s) {
        case 0: return "Info";
        case 1: return "Warning";
        case 2: return "Error";
        default: return "Unknown";
    }
}

static const char* category_str(int c) {
    switch (c) {
        case 0: return "MissingSublayer";
        case 1: return "SublayerParseFail";
        case 2: return "MissingReference";
        case 3: return "ReferenceParseFail";
        case 4: return "MissingDefaultPrim";
        case 5: return "MissingPayload";
        case 6: return "PayloadParseFail";
        case 7: return "Other";
        default: return "Unknown";
    }
}

static const char* arc_type_str(int a) {
    switch (a) {
        case 0: return "sublayer";
        case 1: return "reference";
        case 2: return "payload";
        case 3: return "none";
        default: return "unknown";
    }
}

static void format_bytes(size_t bytes, char* buf, size_t bufsz) {
    if (bytes >= 1024ULL * 1024 * 1024)
        snprintf(buf, bufsz, "%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
    else if (bytes >= 1024ULL * 1024)
        snprintf(buf, bufsz, "%.2f MB", bytes / (1024.0 * 1024.0));
    else if (bytes >= 1024)
        snprintf(buf, bufsz, "%.2f KB", bytes / 1024.0);
    else
        snprintf(buf, bufsz, "%zu B", bytes);
}

/* Recursively count prims including children */
static int count_prims_recursive(NanousdPrim prim) {
    int count = 1;
    int nchildren = nanousd_nchildren(prim);
    for (int i = 0; i < nchildren; i++) {
        NanousdPrim child = nanousd_child(prim, i);
        count += count_prims_recursive(child);
        nanousd_freeprim(child);
    }
    return count;
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <stage.usd>\n", argv[0]);
        return 1;
    }
    const char* filepath = argv[1];

    /* Auto-enable per-prim equivalence validation on benchmark runs:
     * benchmarks are where we want the strongest check that lazy compose
     * would produce the same result as eager. Caller can still opt out
     * by setting NANOUSD_PRIMINDEX_EQUIV=0 explicitly. */
#ifdef _WIN32
    if (!getenv("NANOUSD_PRIMINDEX_EQUIV")) _putenv("NANOUSD_PRIMINDEX_EQUIV=1");
#else
    setenv("NANOUSD_PRIMINDEX_EQUIV", "1", 0);
#endif

    printf("=== nanousd Stage Load Benchmark ===\n");
    printf("File: %s\n\n", filepath);

    /* ---- Phase 1: Open stage (parse + compose) ---- */

    size_t rss_before = current_rss();
    double t0 = now_ms();

    NanousdStage stage = nanousd_open(filepath);

    double t_open = now_ms() - t0;
    size_t rss_after = current_rss();
    size_t rss_peak = peak_rss();

    if (!stage || !nanousd_isvalid(stage)) {
        const char* err = stage ? nanousd_error(stage) : "null stage";
        fprintf(stderr, "Failed to open stage: %s\n", err);
        if (stage) nanousd_close(stage);
        return 1;
    }

    /* ---- Phase 2: Traverse all prims ---- */

    double t1 = now_ms();

    int total_prims = nanousd_nprims(stage);
    double t_traverse_build = now_ms() - t1;

    int total_attribs = 0;
    double t_prim_handle = 0.0, t_nattribs = 0.0, t_free_handle = 0.0;
    double t_loop_start = now_ms();
    for (int i = 0; i < total_prims; i++) {
        double a = now_ms();
        NanousdPrim p = nanousd_prim(stage, i);
        double b = now_ms();
        total_attribs += nanousd_nattribs(p);
        double c = now_ms();
        nanousd_freeprim(p);
        double d = now_ms();
        t_prim_handle += b - a;
        t_nattribs   += c - b;
        t_free_handle += d - c;
    }
    double t_loop = now_ms() - t_loop_start;
    double t_traverse = now_ms() - t1;

    /* ---- Phase 3: Collect diagnostics ---- */

    double t2 = now_ms();

    int diag_count = 0;
    NanousdDiagnostic* diags = nanousd_diagnostics(stage, &diag_count);

    double t_diag = now_ms() - t2;

    /* ---- Report: Timing ---- */

    printf("--- Timing ---\n");
    printf("  Stage open (parse + compose): %10.1f ms\n", t_open);
    printf("  Traverse all prims + attribs: %10.1f ms\n", t_traverse);
    printf("    traverse (build prim list): %10.1f ms\n", t_traverse_build);
    printf("    loop (nprims iterations):   %10.1f ms   (prim_handle=%.1f nattribs=%.1f free=%.1f)\n",
           t_loop, t_prim_handle, t_nattribs, t_free_handle);
    printf("  Collect diagnostics:          %10.1f ms\n", t_diag);
    printf("  Total:                        %10.1f ms\n", t_open + t_traverse + t_diag);
    printf("\n");

    /* ---- Report: Memory ---- */

    char buf1[64], buf2[64], buf3[64];
    format_bytes(rss_before, buf1, sizeof(buf1));
    format_bytes(rss_after, buf2, sizeof(buf2));
    format_bytes(rss_peak, buf3, sizeof(buf3));
    size_t rss_delta = rss_after > rss_before ? rss_after - rss_before : 0;
    char buf4[64];
    format_bytes(rss_delta, buf4, sizeof(buf4));

    printf("--- Memory (RSS) ---\n");
    printf("  Before open: %s\n", buf1);
    printf("  After open:  %s\n", buf2);
    printf("  Delta:       %s\n", buf4);
    printf("  Peak:        %s\n", buf3);
    printf("\n");

    /* ---- Report: Stage stats ---- */

    printf("--- Stage ---\n");
    printf("  Total prims:      %d\n", total_prims);
    printf("  Total attributes: %d\n", total_attribs);
    printf("\n");

    /* ---- Report: Diagnostics ---- */

    printf("--- Diagnostics: %d total ---\n", diag_count);

    if (diag_count > 0) {
        /* Count by severity */
        int n_info = 0, n_warn = 0, n_error = 0;
        /* Count by category */
        int cat_counts[8] = {0};

        for (int i = 0; i < diag_count; i++) {
            switch (diags[i].severity) {
                case 0: n_info++; break;
                case 1: n_warn++; break;
                case 2: n_error++; break;
            }
            if (diags[i].category >= 0 && diags[i].category < 8)
                cat_counts[diags[i].category]++;
        }

        printf("\n  By severity:\n");
        if (n_info)  printf("    Info:    %d\n", n_info);
        if (n_warn)  printf("    Warning: %d\n", n_warn);
        if (n_error) printf("    Error:   %d\n", n_error);

        printf("\n  By category:\n");
        for (int c = 0; c < 8; c++) {
            if (cat_counts[c] > 0)
                printf("    %-22s %d\n", category_str(c), cat_counts[c]);
        }

        /* Print first N diagnostics in detail */
        int show = diag_count < 20 ? diag_count : 20;
        printf("\n  First %d diagnostic(s):\n", show);
        for (int i = 0; i < show; i++) {
            printf("    [%d] %s/%s (arc=%s)\n",
                   i, severity_str(diags[i].severity),
                   category_str(diags[i].category),
                   arc_type_str(diags[i].arc_type));
            if (diags[i].message && diags[i].message[0])
                printf("        msg:   %s\n", diags[i].message);
            if (diags[i].prim_path && diags[i].prim_path[0])
                printf("        prim:  %s\n", diags[i].prim_path);
            if (diags[i].layer_path && diags[i].layer_path[0])
                printf("        layer: %s\n", diags[i].layer_path);
            if (diags[i].asset_path && diags[i].asset_path[0])
                printf("        asset: %s\n", diags[i].asset_path);
        }
        if (diag_count > show)
            printf("    ... and %d more\n", diag_count - show);
    }

    /* ---- Cleanup ---- */

    nanousd_free_diagnostics(diags, diag_count);
    nanousd_close(stage);

    printf("\nDone.\n");
    return 0;
}

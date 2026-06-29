// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// benchmark_load_phases — RSS / timing probe at each stage-load phase.
//
// Whereas benchmark_stage_load measures only around the public C API's
// one-shot `nanousd_open`, this tool calls the lower-level entry points
// individually so we can isolate where memory growth comes from. The
// phase split mirrors the actual code path: parse the layer, compose
// the graph, build the stage, traverse all prims + attributes, then
// fault-in every default value (forcing lazy values to decode).
//
// Usage:
//   ./benchmark_load_phases <root.usd>
//
// The output shows current RSS at each phase boundary plus the delta
// since the previous one, so the dominant contributor is whichever
// phase shows the biggest jump.

#include "nanousd/compose.h"
#include "nanousd/spec.h"
#include "nanousd/stage.h"
#include "nanousd/usd_parser.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <ctime>
#endif

namespace {

double NowMs() {
#ifdef _WIN32
    static double freq = 0.0;
    LARGE_INTEGER li;
    if (freq == 0.0) {
        QueryPerformanceFrequency(&li);
        freq = static_cast<double>(li.QuadPart) / 1000.0;
    }
    QueryPerformanceCounter(&li);
    return static_cast<double>(li.QuadPart) / freq;
#else
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
#endif
}

size_t CurrentRss() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return static_cast<size_t>(pmc.WorkingSetSize);
    return 0;
#else
    FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0;
    long ignored = 0;
    long pages = 0;
    if (std::fscanf(f, "%ld %ld", &ignored, &pages) != 2) pages = 0;
    std::fclose(f);
    return static_cast<size_t>(pages) * 4096u;
#endif
}

// "Private bytes" — committed memory backed only by the pagefile, not
// shared with mapped files. Useful to separate heap allocations from
// mmap'd file pages on Windows. Linux equivalent: VmData from /proc.
size_t CurrentPrivate() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
                             sizeof(pmc)))
        return static_cast<size_t>(pmc.PrivateUsage);
    return 0;
#else
    FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    size_t bytes = 0;
    while (std::fgets(line, sizeof(line), f)) {
        long kb = 0;
        if (std::sscanf(line, "VmData: %ld kB", &kb) == 1) {
            bytes = static_cast<size_t>(kb) * 1024u;
            break;
        }
    }
    std::fclose(f);
    return bytes;
#endif
}

size_t PeakRss() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return static_cast<size_t>(pmc.PeakWorkingSetSize);
    return 0;
#else
    rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) == 0)
        return static_cast<size_t>(ru.ru_maxrss) * 1024u;
    return 0;
#endif
}

std::string FormatBytes(int64_t bytes) {
    char buf[64];
    double absb = static_cast<double>(bytes < 0 ? -bytes : bytes);
    const char* sign = bytes < 0 ? "-" : "";
    if (absb >= 1024.0 * 1024 * 1024)
        std::snprintf(buf, sizeof(buf), "%s%.2f GB",  sign, absb / (1024.0 * 1024.0 * 1024.0));
    else if (absb >= 1024.0 * 1024)
        std::snprintf(buf, sizeof(buf), "%s%.2f MB",  sign, absb / (1024.0 * 1024.0));
    else if (absb >= 1024.0)
        std::snprintf(buf, sizeof(buf), "%s%.2f KB",  sign, absb / 1024.0);
    else
        std::snprintf(buf, sizeof(buf), "%s%.0f B",   sign, absb);
    return std::string(buf);
}

struct Mark {
    size_t rss = 0;
    size_t priv = 0;
    double t   = 0.0;
};

Mark TakeMark() { return Mark{CurrentRss(), CurrentPrivate(), NowMs()}; }

void PrintPhase(const char* name, const Mark& before, const Mark& after) {
    int64_t deltaRss  = static_cast<int64_t>(after.rss)  - static_cast<int64_t>(before.rss);
    int64_t deltaPriv = static_cast<int64_t>(after.priv) - static_cast<int64_t>(before.priv);
    double  dt        = after.t - before.t;
    std::printf("  %-32s  rss +%-10s  priv +%-10s  (rss=%-10s priv=%-10s)   %.1f ms\n",
                name,
                FormatBytes(deltaRss).c_str(),
                FormatBytes(deltaPriv).c_str(),
                FormatBytes(static_cast<int64_t>(after.rss)).c_str(),
                FormatBytes(static_cast<int64_t>(after.priv)).c_str(),
                dt);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <stage.usd>\n", argv[0]);
        return 1;
    }
    const char* filepath = argv[1];

    // Auto-enable per-prim equivalence validation on benchmark runs.
    // Caller can opt out by setting NANOUSD_PRIMINDEX_EQUIV=0 explicitly.
#ifdef _WIN32
    if (!std::getenv("NANOUSD_PRIMINDEX_EQUIV")) _putenv("NANOUSD_PRIMINDEX_EQUIV=1");
#else
    setenv("NANOUSD_PRIMINDEX_EQUIV", "1", 0);
#endif

    std::printf("=== nanousd load-phase probe ===\n");
    std::printf("File: %s\n\n", filepath);

    Mark m0 = TakeMark();
    std::printf("--- Phase deltas (RSS delta, current RSS, time) ---\n");

    // ---- Phase A: parse the root file (no composition).
    nanousd::UsdParseResult parseResult = nanousd::ParseUsdFile(filepath);
    Mark m1 = TakeMark();
    PrintPhase("A. ParseUsdFile",        m0, m1);
    if (!parseResult.success) {
        std::fprintf(stderr, "Parse failed: %s\n", parseResult.error.c_str());
        return 1;
    }

    // ---- Phase B: compose the graph (sublayers, references, payloads,
    //               variants).
    nanousd::ComposeResult composeResult =
        nanousd::Compose(parseResult.layer, filepath, nullptr);
    Mark m2 = TakeMark();
    PrintPhase("B. Compose",             m1, m2);
    if (!composeResult.success) {
        std::fprintf(stderr, "Compose failed: %s\n", composeResult.error.c_str());
        return 1;
    }

    // ---- B-stats: introspect the composition graph BEFORE handing
    //               it to a Stage. We're trying to understand where
    //               Compose's RSS delta goes — primIndices, opinion
    //               entries, layer count, child-name index.
    {
        const auto& g = composeResult.graph;
        size_t numLayers = g.layers.size();
        size_t numPrimIdx = g.primIndices.size();
        size_t totalEntries = 0;
        size_t entriesCap = 0;
        std::array<size_t, 7> entryHist{}; // 0,1,2,3,4-7,8-15,16+
        size_t hasArcCount = 0;
        size_t nonIdentityMappings = 0;
        for (const auto& [path, idx] : g.primIndices) {
            (void)path;
            size_t n = idx.entries.size();
            totalEntries += n;
            entriesCap += idx.entries.capacity();
            if (idx.hasArcOpinions) hasArcCount++;
            for (const auto& e : idx.entries) {
                if (e.pathMapping && !e.pathMapping->isIdentity)
                    nonIdentityMappings++;
            }
            size_t b = (n == 0) ? 0 :
                       (n == 1) ? 1 :
                       (n == 2) ? 2 :
                       (n == 3) ? 3 :
                       (n <= 7) ? 4 :
                       (n <= 15) ? 5 : 6;
            entryHist[b]++;
        }
        std::printf("\n--- Compose graph stats ---\n");
        std::printf("  Layers in graph:           %zu\n", numLayers);
        std::printf("  PrimIndices:               %zu\n", numPrimIdx);
        std::printf("  Total opinion entries:     %zu\n", totalEntries);
        std::printf("  Vec capacity (entries):    %zu  (slack %zu)\n",
                    entriesCap, entriesCap > totalEntries
                        ? entriesCap - totalEntries : 0);
        std::printf("  hasArcOpinions=true:       %zu\n", hasArcCount);
        std::printf("  Non-identity PathMappings: %zu\n", nonIdentityMappings);
        std::printf("  Entries-per-prim hist (0/1/2/3/4-7/8-15/16+): "
                    "%zu/%zu/%zu/%zu/%zu/%zu/%zu\n",
                    entryHist[0], entryHist[1], entryHist[2], entryHist[3],
                    entryHist[4], entryHist[5], entryHist[6]);

        // sizeof(OpinionEntry) and sizeof(PrimIndex) — back of envelope
        std::printf("  sizeof(OpinionEntry):      %zu B\n",
                    sizeof(nanousd::OpinionEntry));
        std::printf("  sizeof(PrimIndex):         %zu B\n",
                    sizeof(nanousd::PrimIndex));
        std::printf("  sizeof(Path):              %zu B\n",
                    sizeof(nanousd::Path));
    }

    // ---- Phase C: hand the composed graph to a Stage.
    nanousd::Stage stage =
        nanousd::Stage::CreateFromComposedLayer(std::move(composeResult.graph));
    Mark m3 = TakeMark();
    PrintPhase("C. Stage construction",  m2, m3);
    if (!stage.IsValid()) {
        std::fprintf(stderr, "Stage construction failed.\n");
        return 1;
    }

    // ---- Phase D: traverse all prims + count attributes (no values
    //               read yet — values that are lazy stay lazy).
    auto prims = stage.Traverse();
    size_t totalAttribs = 0;
    for (const auto& p : prims) {
        totalAttribs += p.GetAttributeNames().size();
    }
    Mark m4 = TakeMark();
    PrintPhase("D. Traverse + attr count", m3, m4);

    // ---- Phase E: force every attribute's default to resolve. This
    //               is the step that triggers any lazy decoding still
    //               referenced from the file buffer; if eager-decode
    //               duplication is the dominant cost, RSS jumps here
    //               while phase B/C stays small.
    size_t resolved = 0;
    for (const auto& p : prims) {
        for (const auto& name : p.GetAttributeNames()) {
            auto attr = p.GetAttribute(name);
            if (!attr.IsValid()) continue;
            auto r = attr.Get(nanousd::UsdTimeCode::Default());
            if (r.found) ++resolved;
        }
    }
    Mark m5 = TakeMark();
    PrintPhase("E. Resolve all defaults",   m4, m5);

    // ---- Phase F: Field-distribution analysis. Walk every layer's
    //               specs, force materialization, tally SpecType counts,
    //               field-count histogram, and top field names. Answers
    //               two questions:
    //                 1. How many specs have 0/1/2/few/many fields?
    //                    (informs map-vs-vector storage choice)
    //                 2. How many specs author nothing at all?
    //                    (could those be elided?)
    auto specTypeName = [](nanousd::SpecType t) -> const char* {
        using nanousd::SpecType;
        switch (t) {
            case SpecType::Layer:        return "Layer";
            case SpecType::Prim:         return "Prim";
            case SpecType::Attribute:    return "Attribute";
            case SpecType::Relationship: return "Relationship";
            case SpecType::VariantSet:   return "VariantSet";
            case SpecType::Variant:      return "Variant";
        }
        return "?";
    };

    constexpr size_t kNumTypes = 6;
    constexpr size_t kNumBuckets = 7; // 0, 1, 2, 3, 4-7, 8-15, 16+
    std::array<size_t, kNumTypes> typeCount{};
    std::array<std::array<size_t, kNumBuckets>, kNumTypes> hist{};
    std::array<size_t, kNumTypes> totalFields{};
    std::unordered_map<std::string, size_t> nameTally;

    auto bucketIdx = [](size_t n) -> size_t {
        if (n == 0) return 0;
        if (n == 1) return 1;
        if (n == 2) return 2;
        if (n == 3) return 3;
        if (n <= 7) return 4;
        if (n <= 15) return 5;
        return 6;
    };

    auto& graph = stage.GetGraph();
    size_t numLayers = graph.GetNumLayers();
    size_t totalSpecs = 0;
    for (size_t i = 0; i < numLayers; ++i) {
        const nanousd::Layer& layer = graph.GetLayer(i);
        // Layer spec
        {
            const nanousd::Spec& s = layer.GetLayerSpec();
            size_t t = static_cast<size_t>(s.GetType());
            const auto& fs = s.GetFields(); // forces materialization
            size_t n = fs.size();
            typeCount[t]++;
            hist[t][bucketIdx(n)]++;
            totalFields[t] += n;
            for (const auto& [name, _] : fs) nameTally[name.GetString()]++;
            totalSpecs++;
        }
        // Other specs
        layer.ForEachSpec([&](const nanousd::Path&, const nanousd::Spec& s) {
            size_t t = static_cast<size_t>(s.GetType());
            const auto& fs = s.GetFields(); // forces materialization
            size_t n = fs.size();
            typeCount[t]++;
            hist[t][bucketIdx(n)]++;
            totalFields[t] += n;
            for (const auto& [name, _] : fs) nameTally[name.GetString()]++;
            totalSpecs++;
        });
    }
    Mark m6 = TakeMark();
    PrintPhase("F. Field-stats walk", m5, m6);

    std::printf("\n--- Field-count distribution per SpecType ---\n");
    std::printf("  %-13s  %10s  %10s  %s\n", "type", "count", "totalflds",
                "histogram (0/1/2/3/4-7/8-15/16+)");
    for (size_t t = 0; t < kNumTypes; ++t) {
        if (typeCount[t] == 0) continue;
        std::printf("  %-13s  %10zu  %10zu  %zu/%zu/%zu/%zu/%zu/%zu/%zu\n",
                    specTypeName(static_cast<nanousd::SpecType>(t)),
                    typeCount[t], totalFields[t],
                    hist[t][0], hist[t][1], hist[t][2], hist[t][3],
                    hist[t][4], hist[t][5], hist[t][6]);
    }
    std::printf("  %-13s  %10zu specs total\n", "(sum)", totalSpecs);

    std::printf("\n--- Top 20 most-frequent field names (across all specs) ---\n");
    std::vector<std::pair<std::string, size_t>> names(nameTally.begin(),
                                                       nameTally.end());
    std::sort(names.begin(), names.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    size_t shown = 0;
    for (const auto& [n, c] : names) {
        std::printf("  %-30s  %10zu\n", n.c_str(), c);
        if (++shown >= 20) break;
    }

    // Memory back-of-envelope: each unordered_map has ~120 B fixed
    // overhead (control block + initial bucket array) regardless of
    // size, plus ~50 B per stored entry (node alloc + hash + Value).
    constexpr size_t kEmptyMapBytes = 120;
    constexpr size_t kPerEntryBytes = 50;
    size_t totalEntries = 0;
    for (size_t t = 0; t < kNumTypes; ++t) totalEntries += totalFields[t];
    int64_t fixedOverhead =
        static_cast<int64_t>(totalSpecs * kEmptyMapBytes);
    int64_t entryOverhead =
        static_cast<int64_t>(totalEntries * kPerEntryBytes);
    std::printf("\n--- Estimated unordered_map overhead ---\n");
    std::printf("  Fixed   (per-spec ~%zuB):  %s  (%zu specs)\n",
                kEmptyMapBytes, FormatBytes(fixedOverhead).c_str(), totalSpecs);
    std::printf("  Entries (per-field ~%zuB): %s  (%zu entries)\n",
                kPerEntryBytes, FormatBytes(entryOverhead).c_str(), totalEntries);
    std::printf("  Total:                     %s\n",
                FormatBytes(fixedOverhead + entryOverhead).c_str());

    std::printf("\n--- Totals ---\n");
    std::printf("  Prims:               %zu\n", prims.size());
    std::printf("  Attributes:          %zu\n", totalAttribs);
    std::printf("  Resolved defaults:   %zu\n", resolved);
    std::printf("  Final RSS:           %s\n", FormatBytes(static_cast<int64_t>(m5.rss)).c_str());
    std::printf("  Peak RSS:            %s\n", FormatBytes(static_cast<int64_t>(PeakRss())).c_str());
    std::printf("  Per-prim final RSS:  %s\n",
                FormatBytes(prims.empty()
                            ? 0
                            : static_cast<int64_t>(m5.rss / prims.size())).c_str());
    std::printf("\nDone.\n");
    return 0;
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * USD Core Specification Compliance Test Suite
 *
 * Pure C test driver that exercises spec-observable behavior through
 * the nanousd C API.  No implementation internals are used — only
 * #include "nanousdapi.h".
 *
 * To run against any backend:
 *   NANOUSD_BACKEND=path/to/backend.dll  ./compliance_test [usda_dir]
 *
 * Exit code: 0 if all tests pass, 1 otherwise.
 */

#include "nanousd/nanousdapi.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * Test harness
 * ============================================================ */

static int g_tests_run    = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

static const char* g_usda_dir = "tests/compliance/usda";

/* Cap for strnlen() over arbitrary C strings inside this test suite. Picked
 * far above any realistic test input so behavior is unchanged, but bounded
 * so static analyzers don't flag the read as unbounded (cpp:S5813). */
#define TEST_STR_MAX (4u * 1024u * 1024u)  /* 4 MiB */

#define ASSERT_MSG(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("  FAIL: %s (line %d): %s\n", __func__, __LINE__, msg); \
            g_tests_failed++; \
            g_tests_run++; \
            return; \
        } \
    } while (0)

#define ASSERT(cond) ASSERT_MSG(cond, #cond)

#define ASSERT_STR_EQ(a, b) \
    do { \
        if (strcmp((a), (b)) != 0) { \
            printf("  FAIL: %s (line %d): \"%s\" != \"%s\"\n", __func__, __LINE__, (a), (b)); \
            g_tests_failed++; \
            g_tests_run++; \
            return; \
        } \
    } while (0)

#define ASSERT_FLOAT_EQ(a, b, eps) \
    ASSERT_MSG(fabs((double)(a) - (double)(b)) < (eps), #a " ~= " #b)

#define TEST_PASS() \
    do { g_tests_passed++; g_tests_run++; printf("  ok - %s\n", __func__); } while (0)

/* Helper: build path to a USDA fixture file */
static char g_path_buf[1024];

static const char* usda_path(const char* filename) {
    snprintf(g_path_buf, sizeof(g_path_buf), "%s/%s", g_usda_dir, filename);
    return g_path_buf;
}

/* Helper: build a cross-platform temp file path.
 * Uses TMPDIR (Unix) or TEMP/TMP (Windows), falls back to /tmp. */
static char g_tmp_buf[1024];

static const char* tmp_path(const char* filename) {
    const char* dir = getenv("TMPDIR");
    if (!dir) dir = getenv("TEMP");
    if (!dir) dir = getenv("TMP");
    if (!dir) dir = "/tmp";
    snprintf(g_tmp_buf, sizeof(g_tmp_buf), "%s/%s", dir, filename);
    return g_tmp_buf;
}

static void file_uri_from_path(const char* path, char* out, size_t out_size) {
#ifdef _WIN32
    if (isalpha((unsigned char)path[0]) && path[1] == ':') {
        snprintf(out, out_size, "file:///%s", path);
        return;
    }
#endif
    snprintf(out, out_size, "file://%s", path);
}

static int define_prim_ok(NanousdStage stage, const char* path, const char* type_name) {
    NanousdPrim prim = nanousd_define_prim(stage, path, type_name);
    if (!prim) return 0;
    nanousd_freeprim(prim);
    return 1;
}

static int primpath_exists(NanousdStage stage, const char* path) {
    NanousdPrim prim = nanousd_primpath(stage, path);
    if (!prim) return 0;
    nanousd_freeprim(prim);
    return 1;
}

static int file_starts_with(const char* path, const char* magic, size_t magic_len) {
    unsigned char buf[16];
    FILE* f = fopen(path, "rb");
    size_t n = 0;
    if (!f) return 0;
    if (magic_len > sizeof(buf)) {
        fclose(f);
        return 0;
    }
    n = fread(buf, 1, magic_len, f);
    fclose(f);
    return n == magic_len && memcmp(buf, magic, magic_len) == 0;
}

static void print_text_diff(const char* actual,
                            const char* expected,
                            const char* actual_name,
                            const char* expected_name) {
    size_t i = 0;
    size_t line = 1;
    size_t col = 1;
    while (actual[i] != '\0' && expected[i] != '\0' &&
           actual[i] == expected[i]) {
        if (actual[i] == '\n') {
            line++;
            col = 1;
        } else {
            col++;
        }
        i++;
    }

    printf("  first diff at byte %zu, line %zu, col %zu\n", i, line, col);
    printf("  %s near:\n%.*s\n", actual_name, 240, actual + i);
    printf("  %s near:\n%.*s\n", expected_name, 240, expected + i);
}

typedef struct TextBuf_s {
    char* data;
    size_t len;
    size_t cap;
} TextBuf;

static int tb_reserve(TextBuf* b, size_t extra) {
    size_t need = b->len + extra + 1;
    char* next = NULL;
    if (need <= b->cap) return 1;
    size_t cap = b->cap ? b->cap : 1024;
    while (cap < need) cap *= 2;
    next = (char*)realloc(b->data, cap);
    if (!next) return 0;
    b->data = next;
    b->cap = cap;
    return 1;
}

static int tb_append_len(TextBuf* b, const char* s, size_t n) {
    if (!tb_reserve(b, n)) return 0;
    memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = '\0';
    return 1;
}

static int tb_append(TextBuf* b, const char* s) {
    /* When the caller's buffer size is visible at compile time (string
     * literal or small stack array), use that as the strnlen bound so
     * GCC's -Wstringop-overread doesn't fire after inlining. Otherwise
     * fall back to TEST_STR_MAX so the read is still bounded. */
#if defined(__GNUC__) || defined(__clang__)
    size_t bos = __builtin_object_size(s, 1);
    size_t bound = (bos == (size_t)-1) ? TEST_STR_MAX : bos;
#else
    size_t bound = TEST_STR_MAX;
#endif
    return tb_append_len(b, s, strnlen(s, bound));
}

static int tb_printf(TextBuf* b, const char* fmt, ...) {
    va_list ap;
    va_list ap2;
    int n = 0;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) {
        va_end(ap2);
        return 0;
    }
    if (!tb_reserve(b, (size_t)n)) {
        va_end(ap2);
        return 0;
    }
    vsnprintf(b->data + b->len, (size_t)n + 1, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)n;
    return 1;
}

static int tb_append_json_string(TextBuf* b, const char* s) {
    const unsigned char* p = (const unsigned char*)(s ? s : "");
    if (!tb_append(b, "\"")) return 0;
    while (*p) {
        char esc[7];
        switch (*p) {
            case '"':  if (!tb_append(b, "\\\"")) return 0; break;
            case '\\': if (!tb_append(b, "\\\\")) return 0; break;
            case '\b': if (!tb_append(b, "\\b")) return 0; break;
            case '\f': if (!tb_append(b, "\\f")) return 0; break;
            case '\n': if (!tb_append(b, "\\n")) return 0; break;
            case '\r': if (!tb_append(b, "\\r")) return 0; break;
            case '\t': if (!tb_append(b, "\\t")) return 0; break;
            default:
                if (*p < 0x20) {
                    snprintf(esc, sizeof(esc), "\\u%04x", (unsigned)*p);
                    if (!tb_append(b, esc)) return 0;
                } else {
                    if (!tb_append_len(b, (const char*)p, 1)) return 0;
                }
                break;
        }
        ++p;
    }
    return tb_append(b, "\"");
}

static int tb_append_number(TextBuf* b, double v) {
    char tmp[64];
    int n = snprintf(tmp, sizeof(tmp), "%.17g", v);
    if (n < 0) return 0;
    if ((size_t)n >= sizeof(tmp)) n = (int)sizeof(tmp) - 1;
    return tb_append_len(b, tmp, (size_t)n);
}

static int read_text_file(const char* path, char** out_text) {
    FILE* f = fopen(path, "rb");
    long size = 0;
    char* data = NULL;
    if (!f) return 0;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return 0;
    }
    size = ftell(f);
    if (size < 0) {
        fclose(f);
        return 0;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return 0;
    }
    data = (char*)malloc((size_t)size + 1);
    if (!data) {
        fclose(f);
        return 0;
    }
    if (fread(data, 1, (size_t)size, f) != (size_t)size) {
        free(data);
        fclose(f);
        return 0;
    }
    fclose(f);
    data[size] = '\0';
    *out_text = data;
    return 1;
}

static int write_flattened_fixture_usda(const char* fixture_name,
                                        const char* const* mask_paths,
                                        int mask_path_count,
                                        const char** out_text) {
    char path[1024];
    NanousdStage stage = NULL;
    const char* text = NULL;

    snprintf(path, sizeof(path), "%s", usda_path(fixture_name));
    stage = mask_path_count > 0
        ? nanousd_open_masked(path, mask_paths, mask_path_count)
        : nanousd_open(path);
    if (stage == NULL) {
        printf("  failed to open fixture %s\n", path);
        return 0;
    }
    if (nanousd_isvalid(stage) != 1) {
        printf("  invalid fixture %s: %s\n", path, nanousd_error(stage));
        nanousd_close(stage);
        return 0;
    }

    text = nanousd_write_usda_string(stage);
    nanousd_close(stage);
    if (text == NULL) {
        printf("  failed to flatten fixture %s\n", path);
        return 0;
    }

    *out_text = text;
    return 1;
}

static int write_text_file(const char* path, const char* text) {
    FILE* f = fopen(path, "wb");
    if (!f) return 0;
    if (fputs(text, f) == EOF) {
        fclose(f);
        return 0;
    }
    fclose(f);
    return 1;
}

static int open_flattened_fixture_stage(const char* root_fixture,
                                        const char* const* mask_paths,
                                        int mask_path_count,
                                        NanousdStage* out_stage,
                                        char* tmp_file,
                                        size_t tmp_file_size) {
    const char* flat_text = NULL;
    NanousdStage stage = NULL;

    if (!write_flattened_fixture_usda(
            root_fixture, mask_paths, mask_path_count, &flat_text)) {
        return 0;
    }

    snprintf(tmp_file, tmp_file_size, "%s",
             tmp_path("nanousd_compliance_normalized_actual.usda"));
    remove(tmp_file);
    if (!write_text_file(tmp_file, flat_text)) {
        printf("  failed to write temporary flattened fixture %s\n", tmp_file);
        nanousd_free_string(flat_text);
        return 0;
    }
    nanousd_free_string(flat_text);

    stage = nanousd_open(tmp_file);
    if (stage == NULL) {
        printf("  failed to open temporary flattened fixture %s\n", tmp_file);
        return 0;
    }
    if (nanousd_isvalid(stage) != 1) {
        printf("  invalid temporary flattened fixture %s: %s\n",
               tmp_file, nanousd_error(stage));
        nanousd_close(stage);
        return 0;
    }

    *out_stage = stage;
    return 1;
}

static int append_json_value_for_attr(TextBuf* b,
                                      NanousdPrim prim,
                                      const char* name,
                                      const char* type_name,
                                      double time,
                                      int sampled,
                                      int* wrote_value) {
    int ok = 0;
    *wrote_value = 0;
    if (strcmp(type_name, "double") == 0) {
        double v = sampled
            ? nanousd_sampled(prim, name, time, &ok)
            : nanousd_attribd(prim, name, &ok);
        if (!ok) return 1;
        *wrote_value = 1;
        return tb_append_number(b, v);
    }
    if (strcmp(type_name, "float") == 0 || strcmp(type_name, "half") == 0) {
        float v = sampled
            ? nanousd_samplef(prim, name, time, &ok)
            : nanousd_attribf(prim, name, &ok);
        if (!ok) return 1;
        *wrote_value = 1;
        return tb_append_number(b, (double)v);
    }
    if (strcmp(type_name, "int") == 0) {
        int v = sampled ? 0 : nanousd_attribi(prim, name, &ok);
        if (sampled || !ok) return 1;
        *wrote_value = 1;
        return tb_printf(b, "%d", v);
    }
    if (strcmp(type_name, "bool") == 0) {
        int v = sampled ? 0 : nanousd_attribb(prim, name, &ok);
        if (sampled || !ok) return 1;
        *wrote_value = 1;
        return tb_append(b, v ? "true" : "false");
    }
    if (strcmp(type_name, "string") == 0) {
        const char* v = sampled ? NULL : nanousd_attribs(prim, name, &ok);
        if (sampled || !ok) return 1;
        *wrote_value = 1;
        return tb_append_json_string(b, v);
    }
    if (strcmp(type_name, "token") == 0) {
        const char* v = sampled ? NULL : nanousd_attrib_token(prim, name, &ok);
        if (sampled || !ok) return 1;
        *wrote_value = 1;
        return tb_append_json_string(b, v);
    }
    if (strcmp(type_name, "asset") == 0) {
        const char* v = sampled ? NULL : nanousd_attribasset(prim, name, &ok);
        if (sampled || !ok) return 1;
        *wrote_value = 1;
        return tb_append_json_string(b, v);
    }
    return 1;
}

static int append_normalized_stage_json(NanousdStage stage, char** out_json) {
    TextBuf b = {0};
    NanousdPrim default_prim = nanousd_defaultprim(stage);
    int nprims = nanousd_nprims(stage);

    if (!tb_append(&b, "{\n  \"defaultPrim\":")) goto fail;
    if (!tb_append_json_string(
            &b, default_prim ? nanousd_name(default_prim) : ""))
        goto fail;
    if (default_prim) {
        nanousd_freeprim(default_prim);
        default_prim = NULL;
    }
    if (!tb_append(&b, ",\n  \"prims\":[\n")) goto fail;

    for (int i = 0; i < nprims; ++i) {
        NanousdPrim prim = nanousd_prim(stage, i);
        int nchildren = 0;
        int nprops = 0;
        int wrote_prop = 0;
        if (!prim) goto fail;

        if (i > 0 && !tb_append(&b, ",\n")) goto fail_prim;
        if (!tb_append(&b, "    {\n      \"path\":")) goto fail_prim;
        if (!tb_append_json_string(&b, nanousd_path(prim))) goto fail_prim;
        if (!tb_append(&b, ",\n      \"type\":")) goto fail_prim;
        if (!tb_append_json_string(&b, nanousd_typename(prim))) goto fail_prim;
        if (!tb_printf(&b, ",\n      \"defined\":%s",
                       nanousd_isdefined(prim) ? "true" : "false"))
            goto fail_prim;
        if (!tb_printf(&b, ",\n      \"abstract\":%s",
                       nanousd_isabstract(prim) ? "true" : "false"))
            goto fail_prim;
        if (!tb_printf(&b, ",\n      \"instanceable\":%s",
                       nanousd_isinstanceable(prim) ? "true" : "false"))
            goto fail_prim;

        if (!tb_append(&b, ",\n      \"children\":[")) goto fail_prim;
        nchildren = nanousd_nchildren(prim);
        for (int c = 0; c < nchildren; ++c) {
            NanousdPrim child = nanousd_child(prim, c);
            if (!child) goto fail_prim;
            if (c > 0 && !tb_append(&b, ",")) {
                nanousd_freeprim(child);
                goto fail_prim;
            }
            if (!tb_append_json_string(&b, nanousd_name(child))) {
                nanousd_freeprim(child);
                goto fail_prim;
            }
            nanousd_freeprim(child);
        }
        if (!tb_append(&b, "],\n      \"properties\":[\n")) goto fail_prim;

        nprops = nanousd_nproperties(prim);
        for (int p = 0; p < nprops; ++p) {
            const char* prop_name = nanousd_propertyname(prim, p);
            if (!prop_name || prop_name[0] == '\0') continue;

            if (nanousd_property_is_attribute(prim, prop_name)) {
                const char* type_name = nanousd_attribtype(prim, prop_name);
                int nconn = nanousd_nconnections(prim, prop_name);
                int nsamples = nanousd_nsamplekeys(prim, prop_name);
                int has_default = 0;
                size_t default_insert_pos = 0;

                if (!nanousd_attrib_authored(prim, prop_name)) continue;
                if (wrote_prop && !tb_append(&b, ",\n")) goto fail_prim;
                wrote_prop = 1;
                if (!tb_append(&b, "        {\"kind\":\"attribute\",\"name\":"))
                    goto fail_prim;
                if (!tb_append_json_string(&b, prop_name)) goto fail_prim;
                if (!tb_append(&b, ",\"type\":")) goto fail_prim;
                if (!tb_append_json_string(&b, type_name)) goto fail_prim;

                default_insert_pos = b.len;
                if (!tb_append(&b, ",\"default\":")) goto fail_prim;
                if (!append_json_value_for_attr(
                        &b, prim, prop_name, type_name, 0.0, 0, &has_default))
                    goto fail_prim;
                if (!has_default) b.len = default_insert_pos;

                if (nconn > 0) {
                    if (!tb_append(&b, ",\"connections\":[")) goto fail_prim;
                    for (int k = 0; k < nconn; ++k) {
                        if (k > 0 && !tb_append(&b, ",")) goto fail_prim;
                        if (!tb_append_json_string(
                                &b, nanousd_connection(prim, prop_name, k)))
                            goto fail_prim;
                    }
                    if (!tb_append(&b, "]")) goto fail_prim;
                }

                if (nsamples > 0) {
                    if (!tb_append(&b, ",\"timeSamples\":[")) goto fail_prim;
                    for (int k = 0; k < nsamples; ++k) {
                        double t = nanousd_samplekey(prim, prop_name, k);
                        int wrote_sample = 0;
                        if (k > 0 && !tb_append(&b, ",")) goto fail_prim;
                        if (!tb_append(&b, "[")) goto fail_prim;
                        if (!tb_append_number(&b, t)) goto fail_prim;
                        if (!tb_append(&b, ",")) goto fail_prim;
                        if (!append_json_value_for_attr(
                                &b, prim, prop_name, type_name, t, 1,
                                &wrote_sample))
                            goto fail_prim;
                        if (!wrote_sample && !tb_append(&b, "null"))
                            goto fail_prim;
                        if (!tb_append(&b, "]")) goto fail_prim;
                    }
                    if (!tb_append(&b, "]")) goto fail_prim;
                }

                if (!tb_append(&b, "}")) goto fail_prim;
            } else if (nanousd_property_is_relationship(prim, prop_name) &&
                       nanousd_rel_authored(prim, prop_name)) {
                int ntargets = nanousd_nreltargets(prim, prop_name);
                if (wrote_prop && !tb_append(&b, ",\n")) goto fail_prim;
                wrote_prop = 1;
                if (!tb_append(&b, "        {\"kind\":\"relationship\",\"name\":"))
                    goto fail_prim;
                if (!tb_append_json_string(&b, prop_name)) goto fail_prim;
                if (!tb_append(&b, ",\"targets\":[")) goto fail_prim;
                for (int k = 0; k < ntargets; ++k) {
                    if (k > 0 && !tb_append(&b, ",")) goto fail_prim;
                    if (!tb_append_json_string(
                            &b, nanousd_reltarget(prim, prop_name, k)))
                        goto fail_prim;
                }
                if (!tb_append(&b, "]}")) goto fail_prim;
            }
        }

        if (!tb_append(&b, "\n      ]\n    }")) goto fail_prim;
        nanousd_freeprim(prim);
        continue;

fail_prim:
        nanousd_freeprim(prim);
        goto fail;
    }

    if (!tb_append(&b, "\n  ]\n}\n")) goto fail;
    *out_json = b.data;
    return 1;

fail:
    if (default_prim) nanousd_freeprim(default_prim);
    free(b.data);
    return 0;
}

static int compare_flattened_fixture_json(const char* root_fixture,
                                          const char* const* mask_paths,
                                          int mask_path_count,
                                          const char* expected_json_fixture) {
    NanousdStage actual_stage = NULL;
    char tmp_actual[1024];
    char expected_path[1024];
    char* actual_json = NULL;
    char* expected_json = NULL;
    int ok = 0;

    if (!open_flattened_fixture_stage(
            root_fixture, mask_paths, mask_path_count,
            &actual_stage, tmp_actual, sizeof(tmp_actual))) {
        return 0;
    }
    if (!append_normalized_stage_json(actual_stage, &actual_json)) {
        printf("  failed to build normalized JSON for %s\n", root_fixture);
        goto done;
    }

    if (getenv("NANOUSD_DUMP_FIXTURE_JSON")) {
        printf("----- normalized JSON for %s -----\n%s", root_fixture, actual_json);
    }

    snprintf(expected_path, sizeof(expected_path), "%s",
             usda_path(expected_json_fixture));
    if (!read_text_file(expected_path, &expected_json)) {
        printf("  failed to read expected JSON %s\n", expected_path);
        goto done;
    }

    ok = strcmp(actual_json, expected_json) == 0;
    if (!ok) {
        printf("  normalized fixture JSON mismatch:\n");
        printf("    root:     %s\n", root_fixture);
        printf("    expected: %s\n", expected_json_fixture);
        print_text_diff(actual_json, expected_json, "actual JSON", "expected JSON");
    }

done:
    free(actual_json);
    free(expected_json);
    if (actual_stage) nanousd_close(actual_stage);
    remove(tmp_actual);
    return ok;
}

static int compare_flattened_fixture_usdc_roundtrip_json(
        const char* root_fixture,
        const char* const* mask_paths,
        int mask_path_count) {
    NanousdStage flat_stage = NULL;
    NanousdStage usdc_stage = NULL;
    char tmp_flat[1024];
    char tmp_usdc[1024];
    char* flat_json = NULL;
    char* usdc_json = NULL;
    int ok = 0;

    if (!open_flattened_fixture_stage(
            root_fixture, mask_paths, mask_path_count,
            &flat_stage, tmp_flat, sizeof(tmp_flat))) {
        return 0;
    }
    if (!append_normalized_stage_json(flat_stage, &flat_json)) {
        printf("  failed to build normalized JSON before USDC roundtrip for %s\n",
               root_fixture);
        goto done;
    }

    snprintf(tmp_usdc, sizeof(tmp_usdc), "%s",
             tmp_path("nanousd_compliance_normalized_actual.usdc"));
    remove(tmp_usdc);
    if (nanousd_write_usdc(flat_stage, tmp_usdc) != 1) {
        printf("  failed to write flattened fixture to USDC: %s\n", root_fixture);
        goto done;
    }

    usdc_stage = nanousd_open(tmp_usdc);
    if (!usdc_stage || nanousd_isvalid(usdc_stage) != 1) {
        printf("  failed to reopen flattened fixture USDC: %s\n", tmp_usdc);
        goto done;
    }
    if (!append_normalized_stage_json(usdc_stage, &usdc_json)) {
        printf("  failed to build normalized JSON after USDC roundtrip for %s\n",
               root_fixture);
        goto done;
    }

    ok = strcmp(flat_json, usdc_json) == 0;
    if (!ok) {
        printf("  flattened USDA -> USDC normalized JSON mismatch:\n");
        printf("    root: %s\n", root_fixture);
        print_text_diff(usdc_json, flat_json, "USDC JSON", "flattened JSON");
    }

done:
    free(flat_json);
    free(usdc_json);
    if (usdc_stage) nanousd_close(usdc_stage);
    if (flat_stage) nanousd_close(flat_stage);
    remove(tmp_flat);
    remove(tmp_usdc);
    return ok;
}

/* Minimal stored-ZIP writer for USDZ compliance fixtures. USDZ requires
 * uncompressed, unencrypted ZIP entries; the compliance tests generate a
 * package on the fly so alternate backends read identical package bytes. */
typedef struct ZipTestEntry_s {
    const char* name;
    const unsigned char* data;
    uint32_t size;
} ZipTestEntry;

typedef struct ZipCentralEntry_s {
    const char* name;
    uint32_t crc;
    uint32_t size;
    uint32_t offset;
} ZipCentralEntry;

static void zip_write_u16(FILE* f, uint16_t v) {
    unsigned char b[2];
    b[0] = (unsigned char)(v & 0xffu);
    b[1] = (unsigned char)((v >> 8) & 0xffu);
    fwrite(b, 1, 2, f);
}

static void zip_write_u32(FILE* f, uint32_t v) {
    unsigned char b[4];
    b[0] = (unsigned char)(v & 0xffu);
    b[1] = (unsigned char)((v >> 8) & 0xffu);
    b[2] = (unsigned char)((v >> 16) & 0xffu);
    b[3] = (unsigned char)((v >> 24) & 0xffu);
    fwrite(b, 1, 4, f);
}

static uint32_t zip_crc32(const unsigned char* data, uint32_t size) {
    uint32_t crc = 0xffffffffu;
    uint32_t i;
    for (i = 0; i < size; ++i) {
        int bit;
        crc ^= data[i];
        for (bit = 0; bit < 8; ++bit) {
            uint32_t mask = 0u - (crc & 1u);
            crc = (crc >> 1) ^ (0xedb88320u & mask);
        }
    }
    return ~crc;
}

static int write_stored_zip(const char* path,
                            const ZipTestEntry* entries,
                            int nentries) {
    FILE* f = fopen(path, "wb");
    ZipCentralEntry* central = NULL;
    int i;
    long central_offset;
    long central_size;
    if (!f) return 0;

    central = (ZipCentralEntry*)calloc((size_t)nentries, sizeof(ZipCentralEntry));
    if (!central) {
        fclose(f);
        return 0;
    }

    for (i = 0; i < nentries; ++i) {
        size_t name_len = strnlen(entries[i].name, 0xffffu + 1u);
        long offset = ftell(f);
        if (offset < 0 || offset > 0x7fffffffL || name_len > 0xffffu) {
            free(central);
            fclose(f);
            return 0;
        }

        central[i].name = entries[i].name;
        central[i].crc = zip_crc32(entries[i].data, entries[i].size);
        central[i].size = entries[i].size;
        central[i].offset = (uint32_t)offset;

        zip_write_u32(f, 0x04034b50u);
        zip_write_u16(f, 20); /* version needed */
        zip_write_u16(f, 0);  /* flags */
        zip_write_u16(f, 0);  /* stored */
        zip_write_u16(f, 0);  /* mod time */
        zip_write_u16(f, 0);  /* mod date */
        zip_write_u32(f, central[i].crc);
        zip_write_u32(f, central[i].size);
        zip_write_u32(f, central[i].size);
        zip_write_u16(f, (uint16_t)name_len);
        zip_write_u16(f, 0);  /* extra length */
        fwrite(entries[i].name, 1, name_len, f);
        fwrite(entries[i].data, 1, entries[i].size, f);
    }

    central_offset = ftell(f);
    if (central_offset < 0 || central_offset > 0x7fffffffL) {
        free(central);
        fclose(f);
        return 0;
    }

    for (i = 0; i < nentries; ++i) {
        /* central[i].name was already validated against 0xffffu in the
         * local-header loop above; the same bound is safe to enforce here. */
        size_t name_len = strnlen(central[i].name, 0xffffu + 1u);
        zip_write_u32(f, 0x02014b50u);
        zip_write_u16(f, 20); /* version made by */
        zip_write_u16(f, 20); /* version needed */
        zip_write_u16(f, 0);  /* flags */
        zip_write_u16(f, 0);  /* stored */
        zip_write_u16(f, 0);  /* mod time */
        zip_write_u16(f, 0);  /* mod date */
        zip_write_u32(f, central[i].crc);
        zip_write_u32(f, central[i].size);
        zip_write_u32(f, central[i].size);
        zip_write_u16(f, (uint16_t)name_len);
        zip_write_u16(f, 0); /* extra length */
        zip_write_u16(f, 0); /* comment length */
        zip_write_u16(f, 0); /* disk start */
        zip_write_u16(f, 0); /* internal attrs */
        zip_write_u32(f, 0); /* external attrs */
        zip_write_u32(f, central[i].offset);
        fwrite(central[i].name, 1, name_len, f);
    }

    central_size = ftell(f) - central_offset;
    if (central_size < 0 || central_size > 0x7fffffffL || nentries > 0xffff) {
        free(central);
        fclose(f);
        return 0;
    }

    zip_write_u32(f, 0x06054b50u);
    zip_write_u16(f, 0);
    zip_write_u16(f, 0);
    zip_write_u16(f, (uint16_t)nentries);
    zip_write_u16(f, (uint16_t)nentries);
    zip_write_u32(f, (uint32_t)central_size);
    zip_write_u32(f, (uint32_t)central_offset);
    zip_write_u16(f, 0);

    free(central);
    if (fclose(f) != 0) return 0;
    return 1;
}

/* ============================================================
 * Stage lifecycle
 * ============================================================ */

static void test_stage_open_valid(void) {
    NanousdStage stage = nanousd_open(usda_path("stage_metadata.usda"));
    ASSERT(stage != NULL);
    ASSERT(nanousd_isvalid(stage) == 1);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_stage_open_null(void) {
    NanousdStage stage = nanousd_open(NULL);
    ASSERT(stage == NULL);
    TEST_PASS();
}

static void test_stage_open_missing(void) {
    NanousdStage stage = nanousd_open("nonexistent_file_xyz.usda");
    ASSERT(stage != NULL);
    ASSERT(nanousd_isvalid(stage) == 0);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_stage_open_masked(void) {
    const char* mask[] = {"/World/Keep/Leaf"};
    NanousdStage stage =
        nanousd_open_masked(usda_path("stage_population_mask.usda"), mask, 1);
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    ASSERT(nanousd_nprims(stage) == 3);
    ASSERT(primpath_exists(stage, "/World"));
    ASSERT(primpath_exists(stage, "/World/Keep"));
    ASSERT(primpath_exists(stage, "/World/Keep/Leaf"));
    ASSERT(!primpath_exists(stage, "/World/Keep/Leaf/Grandchild"));
    ASSERT(!primpath_exists(stage, "/World/Keep/Sibling"));
    ASSERT(!primpath_exists(stage, "/World/Drop"));
    ASSERT(!primpath_exists(stage, "/Other"));

    NanousdPrim world = nanousd_primpath(stage, "/World");
    ASSERT(world != NULL);
    ASSERT(nanousd_nchildren(world) == 1);
    NanousdPrim child = nanousd_child(world, 0);
    ASSERT(child != NULL);
    ASSERT_STR_EQ(nanousd_name(child), "Keep");
    nanousd_freeprim(child);
    nanousd_freeprim(world);

    NanousdPrim leaf = nanousd_primpath(stage, "/World/Keep/Leaf");
    ASSERT(leaf != NULL);
    ASSERT(nanousd_nchildren(leaf) == 0);
    nanousd_freeprim(leaf);

    NanousdPrim default_prim = nanousd_defaultprim(stage);
    ASSERT(default_prim != NULL);
    nanousd_freeprim(default_prim);
    nanousd_close(stage);

    stage = nanousd_open_masked(
        usda_path("stage_population_mask.usda"), NULL, 0);
    ASSERT(stage != NULL && nanousd_isvalid(stage));
    ASSERT(nanousd_nprims(stage) == 8);
    ASSERT(primpath_exists(stage, "/World"));
    ASSERT(primpath_exists(stage, "/World/Keep/Leaf/Grandchild"));
    ASSERT(primpath_exists(stage, "/Other"));
    default_prim = nanousd_defaultprim(stage);
    ASSERT(default_prim != NULL);
    nanousd_freeprim(default_prim);
    nanousd_close(stage);

    {
        const char* bad_mask[] = {"World"};
        stage = nanousd_open_masked(
            usda_path("stage_population_mask.usda"), bad_mask, 1);
        ASSERT(stage != NULL);
        ASSERT(nanousd_isvalid(stage) == 0);
        ASSERT(strcmp(nanousd_error(stage), "") != 0);
        nanousd_close(stage);
    }

    TEST_PASS();
}

static void test_stage_null_handle(void) {
    ASSERT(nanousd_isvalid(NULL) == 0);
    ASSERT(strcmp(nanousd_error(NULL), "") != 0); /* non-empty error */
    nanousd_close(NULL); /* must not crash */
    TEST_PASS();
}

/* ============================================================
 * Stage metadata
 * ============================================================ */

static void test_stage_metadata(void) {
    NanousdStage stage = nanousd_open(usda_path("stage_metadata.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    ASSERT_FLOAT_EQ(nanousd_frames_per_second(stage), 30.0, 0.001);
    ASSERT_FLOAT_EQ(nanousd_timecodes_per_second(stage), 48.0, 0.001);
    ASSERT_FLOAT_EQ(nanousd_start_time(stage), 1.0, 0.001);
    ASSERT_FLOAT_EQ(nanousd_end_time(stage), 100.0, 0.001);

    nanousd_close(stage);
    TEST_PASS();
}

static void test_stage_metadata_root_layer_only(void) {
    NanousdStage stage = nanousd_open(usda_path("stage_metadata_root_only.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    ASSERT_FLOAT_EQ(nanousd_frames_per_second(stage), 24.0, 0.001);
    ASSERT_FLOAT_EQ(nanousd_timecodes_per_second(stage), 24.0, 0.001);
    ASSERT_FLOAT_EQ(nanousd_start_time(stage), 0.0, 0.001);
    ASSERT_FLOAT_EQ(nanousd_end_time(stage), 0.0, 0.001);

    int ok = 0;
    (void)nanousd_metadatad(stage, "metersPerUnit", &ok);
    ASSERT(ok == 0);

    ok = 0;
    (void)nanousd_metadatas(stage, "upAxis", &ok);
    ASSERT(ok == 0);

    nanousd_close(stage);
    TEST_PASS();
}

static void test_stage_default_prim(void) {
    NanousdStage stage = nanousd_open(usda_path("stage_metadata.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim dp = nanousd_defaultprim(stage);
    ASSERT(dp != NULL);
    ASSERT_STR_EQ(nanousd_name(dp), "Root");
    nanousd_freeprim(dp);

    nanousd_close(stage);
    TEST_PASS();
}

static void test_stage_generic_metadata(void) {
    NanousdStage stage = nanousd_open(usda_path("stage_metadata.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    int ok = 0;
    double mpu = nanousd_metadatad(stage, "metersPerUnit", &ok);
    ASSERT(ok == 1);
    ASSERT_FLOAT_EQ(mpu, 0.01, 0.0001);

    ok = 0;
    const char* up = nanousd_metadatas(stage, "upAxis", &ok);
    ASSERT(ok == 1);
    ASSERT_STR_EQ(up, "Y");

    /* Non-existent key */
    ok = 0;
    nanousd_metadatad(stage, "noSuchKey", &ok);
    ASSERT(ok == 0);

    nanousd_close(stage);
    TEST_PASS();
}

static void test_stage_root_layer_path(void) {
    /* Opened from file: path should be non-empty and contain the filename */
    NanousdStage stage = nanousd_open(usda_path("stage_metadata.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));
    const char* path = nanousd_stage_get_root_layer_path(stage);
    ASSERT(path != NULL && path[0] != '\0');
    /* Should contain the filename */
    ASSERT(strstr(path, "stage_metadata.usda") != NULL);
    nanousd_close(stage);

    /* In-memory stage: path should be empty */
    NanousdStage mem = nanousd_create();
    ASSERT(mem != NULL && nanousd_isvalid(mem));
    const char* mempath = nanousd_stage_get_root_layer_path(mem);
    ASSERT(mempath != NULL && mempath[0] == '\0');
    nanousd_close(mem);

    TEST_PASS();
}

/* ============================================================
 * Prim traversal and hierarchy
 * ============================================================ */

static void test_prim_traversal(void) {
    NanousdStage stage = nanousd_open(usda_path("prim_hierarchy.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    int n = nanousd_nprims(stage);
    ASSERT(n > 0);

    /* Access first prim by index */
    NanousdPrim p0 = nanousd_prim(stage, 0);
    ASSERT(p0 != NULL);
    ASSERT(nanousd_prim_isvalid(p0) == 1);
    ASSERT(nanousd_path(p0)[0] != '\0');
    ASSERT(nanousd_name(p0)[0] != '\0');
    nanousd_freeprim(p0);

    /* Out-of-bounds */
    ASSERT(nanousd_prim(stage, -1) == NULL);
    ASSERT(nanousd_prim(stage, 9999) == NULL);

    /* By path */
    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);
    ASSERT_STR_EQ(nanousd_name(root), "Root");

    /* Non-existent path */
    ASSERT(nanousd_primpath(stage, "/DoesNotExist") == NULL);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_prim_hierarchy(void) {
    NanousdStage stage = nanousd_open(usda_path("prim_hierarchy.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    /* Root has children */
    int nc = nanousd_nchildren(root);
    ASSERT(nc >= 2); /* Child, Empty (Abstract may or may not be populated) */

    /* Child by name */
    NanousdPrim child = nanousd_childname(root, "Child");
    ASSERT(child != NULL);
    ASSERT_STR_EQ(nanousd_name(child), "Child");
    ASSERT_STR_EQ(nanousd_typename(child), "Mesh");
    nanousd_freeprim(child);

    /* Empty child */
    NanousdPrim empty = nanousd_childname(root, "Empty");
    ASSERT(empty != NULL);
    ASSERT(nanousd_nchildren(empty) == 0);
    ASSERT_STR_EQ(nanousd_typename(empty), "Scope");
    nanousd_freeprim(empty);

    /* Non-existent */
    ASSERT(nanousd_childname(root, "NoSuch") == NULL);
    ASSERT(nanousd_child(root, -1) == NULL);
    ASSERT(nanousd_child(root, 9999) == NULL);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_define_prim_child_index_unique(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    ASSERT(define_prim_ok(stage, "/World", "Xform"));
    ASSERT(define_prim_ok(stage, "/World/B", "Xform"));
    ASSERT(define_prim_ok(stage, "/World/A", "Xform"));
    ASSERT(define_prim_ok(stage, "/World/B", "Xform"));

    NanousdPrim root = nanousd_primpath(stage, "/World");
    ASSERT(root != NULL);
    ASSERT(nanousd_nchildren(root) == 2);

    NanousdPrim a = nanousd_child(root, 0);
    NanousdPrim b = nanousd_child(root, 1);
    ASSERT(a != NULL && b != NULL);
    ASSERT_STR_EQ(nanousd_name(a), "B");
    ASSERT_STR_EQ(nanousd_name(b), "A");
    nanousd_freeprim(a);
    nanousd_freeprim(b);

    NanousdPrim by_name = nanousd_childname(root, "B");
    ASSERT(by_name != NULL);
    ASSERT_STR_EQ(nanousd_path(by_name), "/World/B");
    nanousd_freeprim(by_name);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_define_prim_child_authored_order(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    /* This is a prim-child enumeration test. AOUSD path Element Ordering is
     * for properties; prim children preserve authored order unless primOrder
     * says otherwise. */
    ASSERT(define_prim_ok(stage, "/Root", "Xform"));
    ASSERT(define_prim_ok(stage, "/Root/foobar", "Xform"));
    ASSERT(define_prim_ok(stage, "/Root/Foobar", "Xform"));
    ASSERT(define_prim_ok(stage, "/Root/_foobar", "Xform"));
    ASSERT(define_prim_ok(stage, "/Root/foo_bar", "Xform"));
    ASSERT(define_prim_ok(stage, "/Root/foo001bar001abc", "Xform"));
    ASSERT(define_prim_ok(stage, "/Root/foo001bar002abc", "Xform"));
    ASSERT(define_prim_ok(stage, "/Root/foo0001bar0002xyz", "Xform"));
    ASSERT(define_prim_ok(stage, "/Root/foo00001bar", "Xform"));
    ASSERT(define_prim_ok(stage, "/Root/a0", "Xform"));
    ASSERT(define_prim_ok(stage, "/Root/a\xC3\xBC", "Xform"));
    ASSERT(define_prim_ok(stage, "/Root/ab", "Xform"));

    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);
    ASSERT(nanousd_nchildren(root) == 11);

    const char* expected[] = {
        "foobar",
        "Foobar",
        "_foobar",
        "foo_bar",
        "foo001bar001abc",
        "foo001bar002abc",
        "foo0001bar0002xyz",
        "foo00001bar",
        "a0",
        "a\xC3\xBC",
        "ab",
    };

    for (int i = 0; i < 11; ++i) {
        NanousdPrim child = nanousd_child(root, i);
        ASSERT(child != NULL);
        ASSERT_STR_EQ(nanousd_name(child), expected[i]);
        nanousd_freeprim(child);
    }

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_prim_queries(void) {
    NanousdStage stage = nanousd_open(usda_path("prim_hierarchy.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);
    ASSERT(nanousd_isactive(root) == 1);
    ASSERT(nanousd_isdefined(root) == 1);
    ASSERT(nanousd_isabstract(root) == 0);
    ASSERT_STR_EQ(nanousd_typename(root), "Xform");

    /* Child has kind */
    NanousdPrim child = nanousd_childname(root, "Child");
    ASSERT(child != NULL);
    ASSERT_STR_EQ(nanousd_kind(child), "component");

    /* Child's attribute */
    int ok = 0;
    int fc = nanousd_attribi(child, "faceCount", &ok);
    ASSERT(ok == 1);
    ASSERT(fc == 6);

    nanousd_freeprim(child);
    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Scalar attribute reads
 * ============================================================ */

static void test_scalar_attributes(void) {
    NanousdStage stage = nanousd_open(usda_path("scalar_attributes.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    int ok = 0;

    /* float */
    float f = nanousd_attribf(root, "scalarF", &ok);
    ASSERT(ok == 1);
    ASSERT_FLOAT_EQ(f, 3.14f, 0.001f);

    /* double */
    ok = 0;
    double d = nanousd_attribd(root, "scalarD", &ok);
    ASSERT(ok == 1);
    ASSERT_FLOAT_EQ(d, 2.718, 0.001);

    /* half */
    ok = 0;
    f = nanousd_attribf(root, "scalarH", &ok);
    ASSERT(ok == 1);
    ASSERT_FLOAT_EQ(f, 0.000000059604644775390625, 0.000000000001);

    ok = 0;
    d = nanousd_attribd(root, "scalarH", &ok);
    ASSERT(ok == 1);
    ASSERT_FLOAT_EQ(d, 0.000000059604644775390625, 0.000000000001);

    ok = 0;
    f = nanousd_attribf(root, "scalarHTie", &ok);
    ASSERT(ok == 1);
    ASSERT_FLOAT_EQ(f, 1.0f, 0.000001f);

    /* int */
    ok = 0;
    int i = nanousd_attribi(root, "scalarI", &ok);
    ASSERT(ok == 1);
    ASSERT(i == 42);

    /* string */
    ok = 0;
    const char* s = nanousd_attribs(root, "scalarS", &ok);
    ASSERT(ok == 1);
    ASSERT_STR_EQ(s, "hello");

    /* bool */
    ok = 0;
    int b = nanousd_attribb(root, "scalarB", &ok);
    ASSERT(ok == 1);
    ASSERT(b == 1);

    /* type queries */
    ASSERT_STR_EQ(nanousd_attribtype(root, "scalarF"), "float");
    ASSERT_STR_EQ(nanousd_attribtype(root, "scalarD"), "double");
    ASSERT_STR_EQ(nanousd_attribtype(root, "scalarH"), "half");
    ASSERT_STR_EQ(nanousd_attribtype(root, "scalarI"), "int");
    ASSERT_STR_EQ(nanousd_attribtype(root, "scalarS"), "string");

    /* non-existent */
    ok = 0;
    nanousd_attribf(root, "nonexistent", &ok);
    ASSERT(ok == 0);

    /* null ok pointer is safe */
    f = nanousd_attribf(root, "scalarF", NULL);
    ASSERT_FLOAT_EQ(f, 3.14f, 0.001f);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Vector attribute reads
 * ============================================================ */

static void test_vec_attributes(void) {
    NanousdStage stage = nanousd_open(usda_path("vec_attributes.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    /* float2 */
    { float v[2] = {0}; ASSERT(nanousd_attribv2f(root, "v2f", v) == 1);
      ASSERT(v[0] == 1.0f && v[1] == 2.0f); }

    /* float3 */
    { float v[3] = {0}; ASSERT(nanousd_attribv3f(root, "v3f", v) == 1);
      ASSERT(v[0] == 1.0f && v[1] == 2.0f && v[2] == 3.0f); }

    /* half3 read through float and double vector APIs */
    { float v[3] = {0}; ASSERT(nanousd_attribv3f(root, "v3h", v) == 1);
      ASSERT_FLOAT_EQ(v[0], 0.000000059604644775390625, 0.000000000001);
      ASSERT_FLOAT_EQ(v[1], 1.5f, 0.000001f);
      ASSERT_FLOAT_EQ(v[2], -2.0f, 0.000001f); }
    { double v[3] = {0}; ASSERT(nanousd_attribv3d(root, "v3h", v) == 1);
      ASSERT_FLOAT_EQ(v[0], 0.000000059604644775390625, 0.000000000001);
      ASSERT_FLOAT_EQ(v[1], 1.5, 0.000001);
      ASSERT_FLOAT_EQ(v[2], -2.0, 0.000001); }

    /* float4 */
    { float v[4] = {0}; ASSERT(nanousd_attribv4f(root, "v4f", v) == 1);
      ASSERT(v[0] == 1.0f && v[1] == 2.0f && v[2] == 3.0f && v[3] == 4.0f); }

    /* double2 */
    { double v[2] = {0}; ASSERT(nanousd_attribv2d(root, "v2d", v) == 1);
      ASSERT(v[0] == 10.0 && v[1] == 20.0); }

    /* double3 */
    { double v[3] = {0}; ASSERT(nanousd_attribv3d(root, "v3d", v) == 1);
      ASSERT(v[0] == 10.0 && v[1] == 20.0 && v[2] == 30.0); }

    /* double4 */
    { double v[4] = {0}; ASSERT(nanousd_attribv4d(root, "v4d", v) == 1);
      ASSERT(v[0] == 10.0 && v[1] == 20.0 && v[2] == 30.0 && v[3] == 40.0); }

    /* int2 */
    { int v[2] = {0}; ASSERT(nanousd_attribv2i(root, "v2i", v) == 1);
      ASSERT(v[0] == 1 && v[1] == 2); }

    /* int3 */
    { int v[3] = {0}; ASSERT(nanousd_attribv3i(root, "v3i", v) == 1);
      ASSERT(v[0] == 1 && v[1] == 2 && v[2] == 3); }

    /* int4 */
    { int v[4] = {0}; ASSERT(nanousd_attribv4i(root, "v4i", v) == 1);
      ASSERT(v[0] == 1 && v[1] == 2 && v[2] == 3 && v[3] == 4); }

    /* non-existent */
    { float v[3] = {0}; ASSERT(nanousd_attribv3f(root, "nonexistent", v) == 0); }

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_matrix_attributes(void) {
    NanousdStage stage = nanousd_open(usda_path("vec_attributes.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    double m[16] = {0};
    ASSERT(nanousd_attribm4d(root, "mat4", m) == 1);
    /* Identity matrix */
    ASSERT(m[0] == 1.0 && m[5] == 1.0 && m[10] == 1.0 && m[15] == 1.0);
    ASSERT(m[1] == 0.0 && m[2] == 0.0 && m[3] == 0.0);

    /* non-existent */
    ASSERT(nanousd_attribm4d(root, "nonexistent", m) == 0);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Array attribute reads
 * ============================================================ */

static void test_array_attributes(void) {
    NanousdStage stage = nanousd_open(usda_path("array_attributes.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    /* float array */
    {
        int len = nanousd_attribarraylen(root, "arrayF");
        ASSERT(len == 3);

        float buf[3] = {0};
        int written = nanousd_attribarrayf(root, "arrayF", buf, 3);
        ASSERT(written == 3);
        ASSERT(buf[0] == 1.0f && buf[1] == 2.0f && buf[2] == 3.0f);

        /* Partial read */
        float small[2] = {0};
        written = nanousd_attribarrayf(root, "arrayF", small, 2);
        ASSERT(written == 2);
        ASSERT(small[0] == 1.0f && small[1] == 2.0f);
    }

    /* int array */
    {
        int len = nanousd_attribarraylen(root, "arrayI");
        ASSERT(len == 3);

        int buf[3] = {0};
        int written = nanousd_attribarrayi(root, "arrayI", buf, 3);
        ASSERT(written == 3);
        ASSERT(buf[0] == 10 && buf[1] == 20 && buf[2] == 30);
    }

    /* half array read through float and double array APIs */
    {
        int len = nanousd_attribarraylen(root, "arrayH");
        ASSERT(len == 3);

        float fbuf[3] = {0};
        int written = nanousd_attribarrayf(root, "arrayH", fbuf, 3);
        ASSERT(written == 3);
        ASSERT_FLOAT_EQ(fbuf[0], 0.000000059604644775390625,
                        0.000000000001);
        ASSERT_FLOAT_EQ(fbuf[1], 1.5f, 0.000001f);
        ASSERT_FLOAT_EQ(fbuf[2], -2.0f, 0.000001f);

        double dbuf[3] = {0};
        written = nanousd_attribarrayd(root, "arrayH", dbuf, 3);
        ASSERT(written == 3);
        ASSERT_FLOAT_EQ(dbuf[0], 0.000000059604644775390625,
                        0.000000000001);
        ASSERT_FLOAT_EQ(dbuf[1], 1.5, 0.000001);
        ASSERT_FLOAT_EQ(dbuf[2], -2.0, 0.000001);
    }

    /* double array */
    {
        int len = nanousd_attribarraylen(root, "arrayD");
        ASSERT(len == 3);

        double buf[3] = {0};
        int written = nanousd_attribarrayd(root, "arrayD", buf, 3);
        ASSERT(written == 3);
        ASSERT_FLOAT_EQ(buf[0], 1.5, 0.001);
        ASSERT_FLOAT_EQ(buf[1], 2.5, 0.001);
        ASSERT_FLOAT_EQ(buf[2], 3.5, 0.001);
    }

    /* non-existent */
    ASSERT(nanousd_attribarraylen(root, "nonexistent") == -1);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_foundational_type_names(void) {
    NanousdStage stage = nanousd_open(usda_path("foundational_types.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim root = nanousd_primpath(stage, "/FoundationalTypes");
    ASSERT(root != NULL);

    ASSERT_STR_EQ(nanousd_attribtype(root, "ucharMax"), "uchar");
    ASSERT_STR_EQ(nanousd_attribtype(root, "timecodeVal"), "timecode");
    ASSERT_STR_EQ(nanousd_attribtype(root, "uchars"), "uchar[]");
    ASSERT_STR_EQ(nanousd_attribtype(root, "timecodes"), "timecode[]");
    ASSERT_STR_EQ(nanousd_attribtype(root, "h2s"), "half2[]");
    ASSERT_STR_EQ(nanousd_attribtype(root, "qaths"), "quath[]");
    ASSERT_STR_EQ(nanousd_attribtype(root, "matrices2"), "matrix2d[]");
    ASSERT_STR_EQ(nanousd_attribtype(root, "matrices3"), "matrix3d[]");
    ASSERT_STR_EQ(nanousd_attribtype(root, "points"), "point3f[]");
    ASSERT_STR_EQ(nanousd_attribtype(root, "c3h"), "color3h");
    ASSERT_STR_EQ(nanousd_attribtype(root, "c4h"), "color4h");
    ASSERT_STR_EQ(nanousd_attribtype(root, "frame"), "frame4d");

    ASSERT(nanousd_attribarraylen(root, "uchars") == 3);
    ASSERT(nanousd_attribarraylen(root, "timecodes") == 3);
    ASSERT(nanousd_attribarraylen(root, "h2s") == 1);
    ASSERT(nanousd_attribarraylen(root, "qaths") == 1);
    ASSERT(nanousd_attribarraylen(root, "matrices2") == 1);
    ASSERT(nanousd_attribarraylen(root, "matrices3") == 1);
    ASSERT(nanousd_attribarraylen(root, "points") == 2);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Time samples
 * ============================================================ */

static void test_timesamples(void) {
    NanousdStage stage = nanousd_open(usda_path("timesamples.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    /* Float time samples */
    ASSERT(nanousd_hassamples(root, "anim") == 1);
    ASSERT(nanousd_nsamplekeys(root, "anim") == 3);

    int ok = 0;
    float fval = nanousd_samplef(root, "anim", 1.0, &ok);
    ASSERT(ok == 1);
    ASSERT_FLOAT_EQ(fval, 0.0f, 0.001f);
    fval = nanousd_samplef(root, "anim", 12.0, &ok);
    ASSERT(ok == 1);
    ASSERT_FLOAT_EQ(fval, 0.5f, 0.001f);
    fval = nanousd_samplef(root, "anim", 24.0, &ok);
    ASSERT(ok == 1);
    ASSERT_FLOAT_EQ(fval, 1.0f, 0.001f);

    /* Double time samples */
    ok = 0;
    double dval = nanousd_sampled(root, "animD", 12.0, &ok);
    ASSERT(ok == 1);
    ASSERT_FLOAT_EQ(dval, 0.5, 0.001);

    /* float3 time samples */
    ASSERT(nanousd_hassamples(root, "dir") == 1);
    {
        float v[3] = {0};
        ASSERT(nanousd_samplev3f(root, "dir", 1.0, v) == 1);
        ASSERT(v[0] == 1.0f && v[1] == 0.0f && v[2] == 0.0f);
        ASSERT(nanousd_samplev3f(root, "dir", 24.0, v) == 1);
        ASSERT(v[0] == 0.0f && v[1] == 1.0f && v[2] == 0.0f);
    }

    /* double3 time samples */
    ASSERT(nanousd_hassamples(root, "pos") == 1);
    ASSERT(nanousd_nsamplekeys(root, "pos") == 3);
    {
        double v[3] = {0};
        ASSERT(nanousd_samplev3d(root, "pos", 1.0, v) == 1);
        ASSERT(v[0] == 0.0 && v[1] == 0.0 && v[2] == 0.0);
        ASSERT(nanousd_samplev3d(root, "pos", 12.0, v) == 1);
        ASSERT(v[0] == 5.0 && v[1] == 5.0 && v[2] == 5.0);
    }

    /* Non-time-sampled attribute */
    ASSERT(nanousd_hassamples(root, "scalarF") == 0);
    ASSERT(nanousd_nsamplekeys(root, "scalarF") == 0);

    /* Non-existent */
    ASSERT(nanousd_hassamples(root, "nonexistent") == 0);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Relationships
 * ============================================================ */

static void test_relationships(void) {
    NanousdStage stage = nanousd_open(usda_path("relationships.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    ASSERT(nanousd_hasrel(root, "myRel") == 1);
    ASSERT(nanousd_hasrel(root, "nonexistent") == 0);
    ASSERT(nanousd_hasrel(root, "scalarF") == 0);
    for (int i = 0; i < nanousd_nattribs(root); ++i) {
        const char* name = nanousd_attribname(root, i);
        ASSERT(name == NULL || strcmp(name, "myRel") != 0);
        ASSERT(name == NULL || strcmp(name, "multiRel") != 0);
    }

    /* Relationship targets */
    ASSERT(nanousd_nreltargets(root, "myRel") == 1);
    ASSERT_STR_EQ(nanousd_reltarget(root, "myRel", 0), "/Root/Child");

    /* Multi-target relationship */
    int nmt = nanousd_nreltargets(root, "multiRel");
    ASSERT(nmt == 2);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Collections
 * ============================================================ */

static void test_collections(void) {
    NanousdStage stage = nanousd_open(usda_path("collections.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim collections = nanousd_primpath(stage, "/World/Collections");
    ASSERT(collections != NULL);

    ASSERT(nanousd_collection_nmembers(collections, "all") == 5);
    ASSERT_STR_EQ(nanousd_collection_member(collections, "all", 0),
                  "/World/Set");
    ASSERT_STR_EQ(nanousd_collection_member(collections, "all", 1),
                  "/World/Set.keep");
    ASSERT_STR_EQ(nanousd_collection_member(collections, "all", 2),
                  "/World/Set/Child");
    ASSERT_STR_EQ(nanousd_collection_member(collections, "all", 3),
                  "/World/Set/Child.childAttr");
    ASSERT_STR_EQ(nanousd_collection_member(collections, "all", 4),
                  "/World/Other/Leaf");
    ASSERT_STR_EQ(nanousd_collection_member(collections, "all", 5), "");

    ASSERT(nanousd_collection_contains(collections, "all",
                                       "/World/Set.keep") == 1);
    ASSERT(nanousd_collection_contains(collections, "all",
                                       "/World/Set.drop") == 0);
    ASSERT(nanousd_collection_contains(collections, "all",
                                       "/World/Orphan") == 0);
    ASSERT(nanousd_collection_contains(collections, "all",
                                       "/World/Other/Leaf") == 1);

    ASSERT(nanousd_collection_nmembers(collections, "exact") == 1);
    ASSERT_STR_EQ(nanousd_collection_member(collections, "exact", 0),
                  "/World/Set.keep");

    ASSERT(nanousd_collection_nmembers(collections, "prims") == 1);
    ASSERT_STR_EQ(nanousd_collection_member(collections, "prims", 0),
                  "/World/Set");

    ASSERT(nanousd_collection_nmembers(collections, "rootOnly") == 1);
    ASSERT_STR_EQ(nanousd_collection_member(collections, "rootOnly", 0), "/");

    nanousd_freeprim(collections);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Schema queries (IsA / HasAPI)
 * ============================================================ */

static void test_schema_isa(void) {
    NanousdStage stage = nanousd_open(usda_path("schema_queries.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim mesh = nanousd_primpath(stage, "/MeshPrim");
    ASSERT(mesh != NULL);
    ASSERT(nanousd_isa(mesh, "Mesh") == 1);
    ASSERT(nanousd_isa(mesh, "Gprim") == 1);      /* Mesh inherits Gprim */
    ASSERT(nanousd_isa(mesh, "Boundable") == 1);
    ASSERT(nanousd_isa(mesh, "Xformable") == 1);
    ASSERT(nanousd_isa(mesh, "Imageable") == 1);
    ASSERT(nanousd_isa(mesh, "Xform") == 0);       /* Mesh is not Xform */
    nanousd_freeprim(mesh);

    NanousdPrim xf = nanousd_primpath(stage, "/XformPrim");
    ASSERT(xf != NULL);
    ASSERT(nanousd_isa(xf, "Xform") == 1);
    ASSERT(nanousd_isa(xf, "Xformable") == 1);
    ASSERT(nanousd_isa(xf, "Imageable") == 1);
    ASSERT(nanousd_isa(xf, "Mesh") == 0);
    nanousd_freeprim(xf);

    NanousdPrim sc = nanousd_primpath(stage, "/ScopePrim");
    ASSERT(sc != NULL);
    ASSERT(nanousd_isa(sc, "Scope") == 1);
    ASSERT(nanousd_isa(sc, "Imageable") == 1);
    ASSERT(nanousd_isa(sc, "Xformable") == 0);
    nanousd_freeprim(sc);

    NanousdPrim abstract_gprim = nanousd_primpath(stage, "/AbstractGprimPrim");
    ASSERT(abstract_gprim != NULL);
    ASSERT(nanousd_isa(abstract_gprim, "Gprim") == 0);
    ASSERT(nanousd_isa(abstract_gprim, "Imageable") == 0);
    ASSERT(nanousd_hasattrib(abstract_gprim, "doubleSided") == 0);
    nanousd_freeprim(abstract_gprim);

    NanousdPrim fallback_mesh = nanousd_primpath(stage, "/FallbackMeshPrim");
    ASSERT(fallback_mesh != NULL);
    ASSERT(nanousd_isa(fallback_mesh, "FutureMesh") == 0);
    ASSERT(nanousd_isa(fallback_mesh, "Mesh") == 1);
    ASSERT(nanousd_isa(fallback_mesh, "Gprim") == 1);
    ASSERT(nanousd_hasattrib(fallback_mesh, "subdivisionScheme") == 1);
    {
        int ok = 0;
        const char* subd = nanousd_attrib_token(
            fallback_mesh, "subdivisionScheme", &ok);
        ASSERT(ok == 1);
        ASSERT_STR_EQ(subd, "catmullClark");
    }
    nanousd_freeprim(fallback_mesh);

    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Path operations
 * ============================================================ */

static void test_path_parse(void) {
    /* Absolute prim path */
    NanousdPath p = nanousd_path_parse("/Root/Child");
    ASSERT(p != NULL);
    ASSERT_STR_EQ(nanousd_path_str(p), "/Root/Child");
    ASSERT(nanousd_path_is_absolute(p) == 1);
    ASSERT(nanousd_path_is_root(p) == 0);
    ASSERT(nanousd_path_is_property(p) == 0);
    ASSERT_STR_EQ(nanousd_path_name(p), "Child");
    nanousd_path_free(p);

    /* Root path */
    NanousdPath root = nanousd_path_parse("/");
    ASSERT(root != NULL);
    ASSERT(nanousd_path_is_root(root) == 1);
    ASSERT(nanousd_path_is_absolute(root) == 1);
    nanousd_path_free(root);

    /* Property path */
    NanousdPath prop = nanousd_path_parse("/Root.size");
    ASSERT(prop != NULL);
    ASSERT(nanousd_path_is_property(prop) == 1);
    ASSERT(nanousd_path_is_absolute(prop) == 1);
    nanousd_path_free(prop);

    /* NULL input */
    ASSERT(nanousd_path_parse(NULL) == NULL);

    TEST_PASS();
}

static void test_path_operations(void) {
    NanousdPath root = nanousd_path_parse("/Root");
    ASSERT(root != NULL);

    /* Append child */
    NanousdPath child = nanousd_path_append_child(root, "Child");
    ASSERT(child != NULL);
    ASSERT_STR_EQ(nanousd_path_str(child), "/Root/Child");

    /* Append property */
    NanousdPath prop = nanousd_path_append_property(root, "size");
    ASSERT(prop != NULL);
    ASSERT_STR_EQ(nanousd_path_str(prop), "/Root.size");
    ASSERT(nanousd_path_is_property(prop) == 1);

    /* Parent */
    NanousdPath parent = nanousd_path_parent(child);
    ASSERT(parent != NULL);
    ASSERT_STR_EQ(nanousd_path_str(parent), "/Root");

    /* Equality */
    ASSERT(nanousd_path_equal(root, parent) == 1);
    ASSERT(nanousd_path_equal(root, child) == 0);

    /* NULL equality */
    ASSERT(nanousd_path_equal(NULL, NULL) == 1);
    ASSERT(nanousd_path_equal(root, NULL) == 0);

    nanousd_path_free(prop);
    nanousd_path_free(parent);
    nanousd_path_free(child);
    nanousd_path_free(root);

    /* Free NULL is safe */
    nanousd_path_free(NULL);

    TEST_PASS();
}

static void test_unicode_identifiers(void) {
    const char* unicode_path =
        "/M\xC3\xBC" "nchen/\xE6\xA8\xA1\xE5\x9E\x8B.primvars:\xE5\x80\xBC";
    NanousdPath path = nanousd_path_parse(unicode_path);
    ASSERT(path != NULL);
    ASSERT_STR_EQ(nanousd_path_str(path), unicode_path);
    ASSERT(nanousd_path_is_property(path) == 1);
    ASSERT_STR_EQ(nanousd_path_name(path), "primvars:\xE5\x80\xBC");
    nanousd_path_free(path);

    path = nanousd_path_parse("/Cafe\xCC\x81");
    ASSERT(path != NULL);
    ASSERT_STR_EQ(nanousd_path_str(path), "/Cafe\xCC\x81");
    nanousd_path_free(path);

    ASSERT(nanousd_path_parse("/\xF0\x9F\x98\x80") == NULL);
    ASSERT(nanousd_path_parse("/\xEE\x80\x80") == NULL);
    ASSERT(nanousd_path_parse("/\xCC\x81" "bad") == NULL);
    ASSERT(nanousd_path_parse("/bad\xC3(") == NULL);

    {
        char stage_path[1024];
        FILE* f;
        const char usda[] =
            "#usda 1.0\n"
            "def Xform \"World\"\n"
            "{\n"
            "    double \xC3\xBC" "ber = 3.25\n"
            "    token inputs:\xE7\x8A\xB6\xE6\x85\x8B = \"ready\"\n"
            "}\n";
        snprintf(stage_path, sizeof(stage_path), "%s",
                 tmp_path("nanousd_unicode_identifiers.usda"));
        remove(stage_path);
        f = fopen(stage_path, "wb");
        ASSERT(f != NULL);
        ASSERT(fwrite(usda, 1, sizeof(usda) - 1, f) == sizeof(usda) - 1);
        ASSERT(fclose(f) == 0);

        NanousdStage stage = nanousd_open(stage_path);
        ASSERT(stage != NULL && nanousd_isvalid(stage));
        NanousdPrim prim = nanousd_primpath(stage, "/World");
        ASSERT(prim != NULL);
        {
            int ok = 0;
            double value = nanousd_attribd(prim, "\xC3\xBC" "ber", &ok);
            ASSERT(ok == 1);
            ASSERT_FLOAT_EQ(value, 3.25, 1e-9);
        }
        ASSERT(nanousd_hasattrib(prim, "inputs:\xE7\x8A\xB6\xE6\x85\x8B") == 1);
        nanousd_freeprim(prim);
        nanousd_close(stage);
        remove(stage_path);
    }

    TEST_PASS();
}

static void test_usda_grammar_strictness(void) {
    const char* valid_path = tmp_path("nanousd_usda_grammar_valid.usda");
    FILE* f = fopen(valid_path, "wb");
    ASSERT(f != NULL);
    fputs("#usda 1.0\n"
          "def Scope \"Escapes\"\n"
          "{\n"
          "    string value = \"\\a\\b\\f\\v\\x41\\101\"\n"
          "}\n",
          f);
    ASSERT(fclose(f) == 0);

    NanousdStage stage = nanousd_open(valid_path);
    ASSERT(stage != NULL && nanousd_isvalid(stage));
    NanousdPrim prim = nanousd_primpath(stage, "/Escapes");
    ASSERT(prim != NULL);
    {
        int ok = 0;
        const char* value = nanousd_attribs(prim, "value", &ok);
        const char expected[] = "\a\b\f\vAA";
        ASSERT(ok == 1);
        ASSERT_STR_EQ(value, expected);
    }
    nanousd_freeprim(prim);
    nanousd_close(stage);
    remove(valid_path);

    {
        const char* invalid_cases[] = {
            "#usda 1.0\n"
            "def Scope \"Bad\"\n"
            "{\n"
            "    float value = +1\n"
            "}\n",
            "#usda 1.0\n"
            "def Scope \"Bad\"\n"
            "{\n"
            "    float value = 1e\n"
            "}\n",
            "#usda 1.0\n"
            "def Scope \"Bad\"\n"
            "{\n"
            "    float value = +inf\n"
            "}\n",
            "#usda 1.0\n"
            "def Scope \"Bad\"\n"
            "{\n"
            "    string value = \"\\q\"\n"
            "}\n",
            "#usda 1.0\n"
            "def Scope \"Bad\"\n"
            "{\n"
            "    asset value = @foo\nbar@\n"
            "}\n",
        };
        for (int i = 0; i < 5; ++i) {
            char filename[128];
            char path[1024];
            snprintf(filename, sizeof(filename),
                     "nanousd_usda_grammar_invalid_%d.usda", i);
            snprintf(path, sizeof(path), "%s", tmp_path(filename));
            f = fopen(path, "wb");
            ASSERT(f != NULL);
            fputs(invalid_cases[i], f);
            ASSERT(fclose(f) == 0);
            stage = nanousd_open(path);
            ASSERT(stage == NULL || nanousd_isvalid(stage) == 0);
            if (stage) nanousd_close(stage);
            remove(path);
        }
    }

    TEST_PASS();
}

/* ============================================================
 * ListOp operations
 * ============================================================ */

static void test_listop_explicit(void) {
    const char* items[] = { "a", "b", "c" };
    NanousdListOp op = nanousd_listop_create_explicit(items, 3);
    ASSERT(op != NULL);
    ASSERT(nanousd_listop_is_explicit(op) == 1);
    ASSERT(nanousd_listop_nitems(op) == 3);
    ASSERT_STR_EQ(nanousd_listop_item(op, 0), "a");
    ASSERT_STR_EQ(nanousd_listop_item(op, 1), "b");
    ASSERT_STR_EQ(nanousd_listop_item(op, 2), "c");
    nanousd_listop_free(op);
    TEST_PASS();
}

static void test_listop_composable(void) {
    const char* prepend[] = { "x", "y" };
    const char* append[] = { "z" };
    const char* del[] = { "w" };
    NanousdListOp op = nanousd_listop_create(prepend, 2, append, 1, del, 1);
    ASSERT(op != NULL);
    ASSERT(nanousd_listop_is_explicit(op) == 0);
    ASSERT(nanousd_listop_nprepended(op) == 2);
    ASSERT_STR_EQ(nanousd_listop_prepended(op, 0), "x");
    ASSERT_STR_EQ(nanousd_listop_prepended(op, 1), "y");
    ASSERT(nanousd_listop_nappended(op) == 1);
    ASSERT_STR_EQ(nanousd_listop_appended(op, 0), "z");
    ASSERT(nanousd_listop_ndeleted(op) == 1);
    ASSERT_STR_EQ(nanousd_listop_deleted(op, 0), "w");

    /* GetItems: prepend items not in append, then append */
    ASSERT(nanousd_listop_nitems(op) == 3); /* x, y, z */
    ASSERT_STR_EQ(nanousd_listop_item(op, 0), "x");
    ASSERT_STR_EQ(nanousd_listop_item(op, 1), "y");
    ASSERT_STR_EQ(nanousd_listop_item(op, 2), "z");

    nanousd_listop_free(op);
    TEST_PASS();
}

static void test_listop_combine(void) {
    /* Stronger explicit overrides weaker */
    const char* strong[] = { "a", "b" };
    const char* weak[] = { "c", "d" };
    NanousdListOp s = nanousd_listop_create_explicit(strong, 2);
    NanousdListOp w = nanousd_listop_create_explicit(weak, 2);
    ASSERT(s != NULL && w != NULL);

    NanousdListOp combined = nanousd_listop_combine(s, w);
    ASSERT(combined != NULL);
    ASSERT(nanousd_listop_is_explicit(combined) == 1);
    ASSERT(nanousd_listop_nitems(combined) == 2);
    ASSERT_STR_EQ(nanousd_listop_item(combined, 0), "a");
    ASSERT_STR_EQ(nanousd_listop_item(combined, 1), "b");

    nanousd_listop_free(combined);
    nanousd_listop_free(s);
    nanousd_listop_free(w);

    /* Free NULL is safe */
    nanousd_listop_free(NULL);

    TEST_PASS();
}

/* ============================================================
 * Math utilities
 * ============================================================ */

static void test_math_vec3(void) {
    /* dot product */
    {
        float a[3] = {1.0f, 0.0f, 0.0f};
        float b[3] = {0.0f, 1.0f, 0.0f};
        ASSERT_FLOAT_EQ(nanousd_dot3f(a, b), 0.0f, 0.0001f);
        float c[3] = {1.0f, 2.0f, 3.0f};
        float d[3] = {4.0f, 5.0f, 6.0f};
        ASSERT_FLOAT_EQ(nanousd_dot3f(c, d), 32.0f, 0.0001f);
    }
    {
        double a[3] = {1.0, 2.0, 3.0};
        double b[3] = {4.0, 5.0, 6.0};
        ASSERT_FLOAT_EQ(nanousd_dot3d(a, b), 32.0, 0.0001);
    }

    /* length */
    {
        float v[3] = {3.0f, 4.0f, 0.0f};
        ASSERT_FLOAT_EQ(nanousd_length3f(v), 5.0f, 0.0001f);
    }
    {
        double v[3] = {3.0, 4.0, 0.0};
        ASSERT_FLOAT_EQ(nanousd_length3d(v), 5.0, 0.0001);
    }

    /* normalize */
    {
        float v[3] = {3.0f, 0.0f, 0.0f};
        float out[3] = {0};
        nanousd_normalize3f(v, out);
        ASSERT_FLOAT_EQ(out[0], 1.0f, 0.0001f);
        ASSERT_FLOAT_EQ(out[1], 0.0f, 0.0001f);
        ASSERT_FLOAT_EQ(out[2], 0.0f, 0.0001f);
    }

    /* cross product */
    {
        float x[3] = {1.0f, 0.0f, 0.0f};
        float y[3] = {0.0f, 1.0f, 0.0f};
        float out[3] = {0};
        nanousd_cross3f(x, y, out);
        ASSERT_FLOAT_EQ(out[0], 0.0f, 0.0001f);
        ASSERT_FLOAT_EQ(out[1], 0.0f, 0.0001f);
        ASSERT_FLOAT_EQ(out[2], 1.0f, 0.0001f); /* x cross y = z */
    }

    TEST_PASS();
}

static void test_math_matrix(void) {
    /* Identity * Identity = Identity */
    double id[16] = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    };
    double out[16] = {0};
    nanousd_mul_m4d(id, id, out);
    ASSERT(out[0] == 1.0 && out[5] == 1.0 && out[10] == 1.0 && out[15] == 1.0);
    ASSERT(out[1] == 0.0 && out[4] == 0.0);

    /* Transform point by identity */
    double p[3] = {1.0, 2.0, 3.0};
    double tp[3] = {0};
    nanousd_transform_point3d(id, p, tp);
    ASSERT_FLOAT_EQ(tp[0], 1.0, 0.0001);
    ASSERT_FLOAT_EQ(tp[1], 2.0, 0.0001);
    ASSERT_FLOAT_EQ(tp[2], 3.0, 0.0001);

    /* Translation matrix */
    double t[16] = {
        1,0,0,5,
        0,1,0,6,
        0,0,1,7,
        0,0,0,1
    };
    nanousd_transform_point3d(t, p, tp);
    ASSERT_FLOAT_EQ(tp[0], 6.0, 0.0001);
    ASSERT_FLOAT_EQ(tp[1], 8.0, 0.0001);
    ASSERT_FLOAT_EQ(tp[2], 10.0, 0.0001);

    TEST_PASS();
}

static void test_math_quaternion(void) {
    /* Identity quaternion [1,0,0,0] to matrix = identity */
    double qid[4] = {1.0, 0.0, 0.0, 0.0};
    double m[16] = {0};
    nanousd_quat_to_matrix(qid, m);
    ASSERT_FLOAT_EQ(m[0], 1.0, 0.0001);
    ASSERT_FLOAT_EQ(m[5], 1.0, 0.0001);
    ASSERT_FLOAT_EQ(m[10], 1.0, 0.0001);
    ASSERT_FLOAT_EQ(m[15], 1.0, 0.0001);

    /* Slerp between identity and itself = identity */
    double out[4] = {0};
    nanousd_quat_slerp(qid, qid, 0.5, out);
    ASSERT_FLOAT_EQ(out[0], 1.0, 0.0001);
    ASSERT_FLOAT_EQ(out[1], 0.0, 0.0001);
    ASSERT_FLOAT_EQ(out[2], 0.0, 0.0001);
    ASSERT_FLOAT_EQ(out[3], 0.0, 0.0001);

    /* Slerp t=0 returns first, t=1 returns second */
    double q90z[4] = {0.7071068, 0.0, 0.0, 0.7071068}; /* 90 deg around Z */
    nanousd_quat_slerp(qid, q90z, 0.0, out);
    ASSERT_FLOAT_EQ(out[0], 1.0, 0.001);
    ASSERT_FLOAT_EQ(out[3], 0.0, 0.001);
    nanousd_quat_slerp(qid, q90z, 1.0, out);
    ASSERT_FLOAT_EQ(out[0], 0.7071068, 0.001);
    ASSERT_FLOAT_EQ(out[3], 0.7071068, 0.001);

    TEST_PASS();
}

/* ============================================================
 * Specifier semantics (spec Section 9.1)
 *
 * Only prims with specifier Def or Class are populated on
 * the stage.  Over-only prims must NOT appear in traversal.
 * ============================================================ */

static void test_specifier_def(void) {
    NanousdStage stage = nanousd_open(usda_path("specifiers.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    /* "Defined" has specifier Def */
    NanousdPrim def = nanousd_primpath(stage, "/Defined");
    ASSERT(def != NULL);
    ASSERT(nanousd_isdefined(def) == 1);
    ASSERT(nanousd_isabstract(def) == 0);
    nanousd_freeprim(def);

    nanousd_close(stage);
    TEST_PASS();
}

static void test_specifier_class(void) {
    NanousdStage stage = nanousd_open(usda_path("specifiers.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    /* "Abstract" has specifier Class */
    NanousdPrim cls = nanousd_primpath(stage, "/Abstract");
    ASSERT(cls != NULL);
    ASSERT(nanousd_isabstract(cls) == 1);
    ASSERT(nanousd_isdefined(cls) == 1);  /* class is a defining specifier per spec */
    nanousd_freeprim(cls);

    nanousd_close(stage);
    TEST_PASS();
}

static void test_specifier_ancestor_state(void) {
    NanousdStage stage = nanousd_open(usda_path("specifiers.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    /* A child of a class prim is defined, but abstract because one ancestor
     * resolves to class. */
    NanousdPrim child = nanousd_primpath(stage, "/Abstract/AbstractChild");
    ASSERT(child != NULL);
    ASSERT(nanousd_isdefined(child) == 1);
    ASSERT(nanousd_isabstract(child) == 1);
    nanousd_freeprim(child);

    /* Descendants of an over-only ancestor are not part of the populated
     * stage, even if the descendant's own strongest specifier is def. */
    NanousdPrim overChild =
        nanousd_primpath(stage, "/DefinedWithOverChild/OverChild");
    ASSERT(overChild == NULL);

    NanousdPrim hiddenGrandchild =
        nanousd_primpath(stage, "/DefinedWithOverChild/OverChild/HiddenGrandchild");
    ASSERT(hiddenGrandchild == NULL);

    nanousd_close(stage);
    TEST_PASS();
}

static void test_specifier_over_not_populated(void) {
    NanousdStage stage = nanousd_open(usda_path("specifiers.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    /* Over-only prims should NOT be findable by path on the stage */
    NanousdPrim over = nanousd_primpath(stage, "/OverOnly");
    ASSERT(over == NULL);

    /* Verify via traversal: no prim named "OverOnly" should appear */
    int n = nanousd_nprims(stage);
    int i;
    for (i = 0; i < n; i++) {
        NanousdPrim p = nanousd_prim(stage, i);
        ASSERT(strcmp(nanousd_name(p), "OverOnly") != 0);
        nanousd_freeprim(p);
    }

    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Active/inactive filtering (spec Section 9.2)
 *
 * Prims with active=false and their entire subtrees are
 * excluded from stage population / traversal.
 * ============================================================ */

static void test_inactive_excluded(void) {
    NanousdStage stage = nanousd_open(usda_path("inactive.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    /* "Inactive" prim should NOT be populated */
    NanousdPrim inactive = nanousd_primpath(stage, "/Inactive");
    ASSERT(inactive == NULL);

    /* Its child should also not be reachable */
    NanousdPrim inactiveChild = nanousd_primpath(stage, "/Inactive/InactiveChild");
    ASSERT(inactiveChild == NULL);

    /* Active prims should be present */
    NanousdPrim active = nanousd_primpath(stage, "/Active");
    ASSERT(active != NULL);
    ASSERT(nanousd_isactive(active) == 1);

    NanousdPrim activeChild = nanousd_primpath(stage, "/Active/ActiveChild");
    ASSERT(activeChild != NULL);
    nanousd_freeprim(activeChild);
    nanousd_freeprim(active);

    NanousdPrim also = nanousd_primpath(stage, "/AlsoActive");
    ASSERT(also != NULL);
    nanousd_freeprim(also);

    /* Verify traversal count: Active, Active/ActiveChild, AlsoActive = 3 */
    int n = nanousd_nprims(stage);
    ASSERT(n == 3);

    /* No traversed prim should be named "Inactive" or "InactiveChild" */
    int i;
    for (i = 0; i < n; i++) {
        NanousdPrim p = nanousd_prim(stage, i);
        ASSERT(strcmp(nanousd_name(p), "Inactive") != 0);
        ASSERT(strcmp(nanousd_name(p), "InactiveChild") != 0);
        nanousd_freeprim(p);
    }

    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Composition — sublayer opinion strength (spec Section 6)
 *
 * When a stronger layer provides an opinion for an attribute,
 * it wins over the weaker layer's value.  Attributes only in
 * the weaker layer are still visible.
 * ============================================================ */

static void test_composition_sublayer(void) {
    NanousdStage stage = nanousd_open(usda_path("compose_overlay.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    /* "size" is defined in both layers; overlay (stronger) wins */
    int ok = 0;
    float size = nanousd_attribf(root, "size", &ok);
    ASSERT(ok == 1);
    ASSERT_FLOAT_EQ(size, 2.0f, 0.001f);

    /* "name" is only in the base layer; still visible */
    ok = 0;
    const char* name = nanousd_attribs(root, "name", &ok);
    ASSERT(ok == 1);
    ASSERT_STR_EQ(name, "from_base");

    /* "baseOnly" is only in the base layer */
    ok = 0;
    int bo = nanousd_attribi(root, "baseOnly", &ok);
    ASSERT(ok == 1);
    ASSERT(bo == 42);

    /* "visible" is only in the overlay */
    ok = 0;
    int vis = nanousd_attribb(root, "visible", &ok);
    ASSERT(ok == 1);
    ASSERT(vis == 1);

    /* Child from base layer is still reachable */
    NanousdPrim child = nanousd_childname(root, "Child");
    ASSERT(child != NULL);
    ASSERT_STR_EQ(nanousd_typename(child), "Mesh");
    ok = 0;
    int fc = nanousd_attribi(child, "faceCount", &ok);
    ASSERT(ok == 1);
    ASSERT(fc == 6);
    nanousd_freeprim(child);

    /* Prim defined only in the base layer is also reachable */
    NanousdPrim baseOnly = nanousd_primpath(stage, "/BaseOnly");
    ASSERT(baseOnly != NULL);
    ok = 0;
    float w = nanousd_attribf(baseOnly, "weight", &ok);
    ASSERT(ok == 1);
    ASSERT_FLOAT_EQ(w, 10.0f, 0.001f);
    nanousd_freeprim(baseOnly);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Composition — three sublayers (exercises parallel loading)
 *
 * Root layer sublayers A, B, C (listed strongest to weakest).
 * All four layers define /Root with different "source" and
 * "priority" values.  Each sublayer also defines a unique prim.
 * This test verifies correct opinion strength AND exercises the
 * parallel composition path (threshold = 2 sublayers).
 * ============================================================ */

static void test_composition_three_sublayers(void) {
    NanousdStage stage = nanousd_open(usda_path("compose_three_sublayers.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    /* Root layer (strongest) wins for "source" */
    int ok = 0;
    const char* src = nanousd_attribs(root, "source", &ok);
    ASSERT(ok == 1);
    ASSERT_STR_EQ(src, "root_layer");

    /* "priority" is not in root layer (only an over). A is the
     * strongest sublayer that provides it, so A's value wins. */
    ok = 0;
    float pri = nanousd_attribf(root, "priority", &ok);
    ASSERT(ok == 1);
    ASSERT_FLOAT_EQ(pri, 1.0f, 0.001f);

    /* Attributes unique to weaker sublayers still come through */
    ok = 0;
    float fb = nanousd_attribf(root, "from_b", &ok);
    ASSERT(ok == 1);
    ASSERT_FLOAT_EQ(fb, 42.0f, 0.001f);

    ok = 0;
    float fc = nanousd_attribf(root, "from_c", &ok);
    ASSERT(ok == 1);
    ASSERT_FLOAT_EQ(fc, 99.0f, 0.001f);

    /* Prims unique to each sublayer are all reachable */
    NanousdPrim onlyA = nanousd_primpath(stage, "/OnlyInA");
    ASSERT(onlyA != NULL);
    ok = 0;
    ASSERT(nanousd_attribi(onlyA, "value", &ok) == 100);
    ASSERT(ok == 1);
    nanousd_freeprim(onlyA);

    NanousdPrim onlyB = nanousd_primpath(stage, "/OnlyInB");
    ASSERT(onlyB != NULL);
    ok = 0;
    ASSERT(nanousd_attribi(onlyB, "value", &ok) == 200);
    ASSERT(ok == 1);
    nanousd_freeprim(onlyB);

    NanousdPrim onlyC = nanousd_primpath(stage, "/OnlyInC");
    ASSERT(onlyC != NULL);
    ok = 0;
    ASSERT(nanousd_attribi(onlyC, "value", &ok) == 300);
    ASSERT(ok == 1);
    nanousd_freeprim(onlyC);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Composition — reference opinion strength (spec Section 6)
 *
 * When a prim has prepend references, the referenced layer
 * provides opinions for attributes and child subtrees.
 * Local opinions (in the referencing layer) are stronger
 * and win over referenced opinions.
 * ============================================================ */

static void test_composition_reference(void) {
    NanousdStage stage = nanousd_open(usda_path("with_reference.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim myPrim = nanousd_primpath(stage, "/MyPrim");
    ASSERT(myPrim != NULL);

    /* Prim type comes from the referenced layer */
    ASSERT_STR_EQ(nanousd_typename(myPrim), "Xform");

    /* "label" is authored locally — local opinion wins over reference */
    int ok = 0;
    const char* label = nanousd_attribs(myPrim, "label", &ok);
    ASSERT(ok == 1);
    ASSERT_STR_EQ(label, "local_override");

    /* "height" is only in the referenced layer — still visible */
    ok = 0;
    float height = nanousd_attribf(myPrim, "height", &ok);
    ASSERT(ok == 1);
    ASSERT_FLOAT_EQ(height, 5.0f, 0.001f);

    /* Referenced child subtree is included */
    NanousdPrim geo = nanousd_childname(myPrim, "Geo");
    ASSERT(geo != NULL);
    ASSERT_STR_EQ(nanousd_typename(geo), "Mesh");

    ok = 0;
    int vc = nanousd_attribi(geo, "vertexCount", &ok);
    ASSERT(ok == 1);
    ASSERT(vc == 100);

    nanousd_freeprim(geo);
    nanousd_freeprim(myPrim);
    nanousd_close(stage);
    TEST_PASS();
}

/* Path-valued listOp fields are authored in the source namespace of a
 * composition arc. Verify observable relationship targets and attribute
 * connections through reference namespace mapping. */
static void test_composition_path_valued_listop_remap_references(void) {
    NanousdStage stage = nanousd_open(usda_path("path_remap_root.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim inst = nanousd_primpath(stage, "/Instance");
    ASSERT(inst != NULL);

    ASSERT(nanousd_nreltargets(inst, "material:binding") == 1);
    ASSERT_STR_EQ(nanousd_reltarget(inst, "material:binding", 0),
                  "/Instance/Materials/Preview");

    ASSERT(nanousd_nconnections(inst, "inputs:color") == 1);
    ASSERT_STR_EQ(nanousd_connection(inst, "inputs:color", 0),
                  "/Instance/Shader.outputs:color");

    /* External references have no identity mapping, so source-layer targets
     * outside /Model cannot transform into /Instance namespace. */
    ASSERT(nanousd_nreltargets(inst, "externalTarget") == 0);
    ASSERT(nanousd_nconnections(inst, "externalInput") == 0);

    nanousd_freeprim(inst);

    NanousdPrim same_path = nanousd_primpath(stage, "/Model");
    ASSERT(same_path != NULL);

    ASSERT(nanousd_nreltargets(same_path, "material:binding") == 1);
    ASSERT_STR_EQ(nanousd_reltarget(same_path, "material:binding", 0),
                  "/Model/Materials/Preview");

    ASSERT(nanousd_nconnections(same_path, "inputs:color") == 1);
    ASSERT_STR_EQ(nanousd_connection(same_path, "inputs:color", 0),
                  "/Model/Shader.outputs:color");

    ASSERT(nanousd_nreltargets(same_path, "externalTarget") == 0);
    ASSERT(nanousd_nconnections(same_path, "externalInput") == 0);

    nanousd_freeprim(same_path);

    NanousdPrim local = nanousd_primpath(stage, "/LocalInstance");
    ASSERT(local != NULL);

    ASSERT(nanousd_nreltargets(local, "localTarget") == 1);
    ASSERT_STR_EQ(nanousd_reltarget(local, "localTarget", 0), "/Outside");

    ASSERT(nanousd_nconnections(local, "localInput") == 1);
    ASSERT_STR_EQ(nanousd_connection(local, "localInput", 0),
                  "/Outside.outputs:value");

    nanousd_freeprim(local);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_composition_path_valued_listop_remap_payloads(void) {
    NanousdStage stage = nanousd_open(usda_path("path_remap_root.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim payload = nanousd_primpath(stage, "/PayloadInstance");
    ASSERT(payload != NULL);

    ASSERT(nanousd_nreltargets(payload, "payloadTarget") == 1);
    ASSERT_STR_EQ(nanousd_reltarget(payload, "payloadTarget", 0),
                  "/PayloadInstance/Materials/Preview");

    ASSERT(nanousd_nconnections(payload, "payloadInput") == 1);
    ASSERT_STR_EQ(nanousd_connection(payload, "payloadInput", 0),
                  "/PayloadInstance/Shader.outputs:value");

    ASSERT(nanousd_nreltargets(payload, "payloadExternalTarget") == 0);
    ASSERT(nanousd_nconnections(payload, "payloadExternalInput") == 0);

    nanousd_freeprim(payload);

    NanousdPrim same_payload = nanousd_primpath(stage, "/PayloadModel");
    ASSERT(same_payload != NULL);

    ASSERT(nanousd_nreltargets(same_payload, "payloadTarget") == 1);
    ASSERT_STR_EQ(nanousd_reltarget(same_payload, "payloadTarget", 0),
                  "/PayloadModel/Materials/Preview");

    ASSERT(nanousd_nconnections(same_payload, "payloadInput") == 1);
    ASSERT_STR_EQ(nanousd_connection(same_payload, "payloadInput", 0),
                  "/PayloadModel/Shader.outputs:value");

    ASSERT(nanousd_nreltargets(same_payload, "payloadExternalTarget") == 0);
    ASSERT(nanousd_nconnections(same_payload, "payloadExternalInput") == 0);

    nanousd_freeprim(same_payload);

    NanousdPrim local_payload = nanousd_primpath(stage, "/LocalPayloadInstance");
    ASSERT(local_payload != NULL);

    ASSERT(nanousd_nreltargets(local_payload, "localTarget") == 1);
    ASSERT_STR_EQ(nanousd_reltarget(local_payload, "localTarget", 0),
                  "/Outside");

    ASSERT(nanousd_nconnections(local_payload, "localInput") == 1);
    ASSERT_STR_EQ(nanousd_connection(local_payload, "localInput", 0),
                  "/Outside.outputs:value");

    nanousd_freeprim(local_payload);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_composition_path_valued_listop_remap_inherits_specializes(void) {
    NanousdStage stage = nanousd_open(usda_path("path_remap_root.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim inherit = nanousd_primpath(stage, "/InheritInstance");
    ASSERT(inherit != NULL);

    ASSERT(nanousd_nreltargets(inherit, "classTarget") == 1);
    ASSERT_STR_EQ(nanousd_reltarget(inherit, "classTarget", 0),
                  "/InheritInstance/Materials/Preview");

    ASSERT(nanousd_nconnections(inherit, "classInput") == 1);
    ASSERT_STR_EQ(nanousd_connection(inherit, "classInput", 0),
                  "/InheritInstance/Shader.outputs:value");

    ASSERT(nanousd_nreltargets(inherit, "classExternalTarget") == 1);
    ASSERT_STR_EQ(nanousd_reltarget(inherit, "classExternalTarget", 0),
                  "/Outside");

    ASSERT(nanousd_nconnections(inherit, "classExternalInput") == 1);
    ASSERT_STR_EQ(nanousd_connection(inherit, "classExternalInput", 0),
                  "/Outside.outputs:value");

    nanousd_freeprim(inherit);

    NanousdPrim specialize = nanousd_primpath(stage, "/SpecializeInstance");
    ASSERT(specialize != NULL);

    ASSERT(nanousd_nreltargets(specialize, "classTarget") == 1);
    ASSERT_STR_EQ(nanousd_reltarget(specialize, "classTarget", 0),
                  "/SpecializeInstance/Materials/Preview");

    ASSERT(nanousd_nconnections(specialize, "classInput") == 1);
    ASSERT_STR_EQ(nanousd_connection(specialize, "classInput", 0),
                  "/SpecializeInstance/Shader.outputs:value");

    ASSERT(nanousd_nreltargets(specialize, "classExternalTarget") == 1);
    ASSERT_STR_EQ(nanousd_reltarget(specialize, "classExternalTarget", 0),
                  "/Outside");

    ASSERT(nanousd_nconnections(specialize, "classExternalInput") == 1);
    ASSERT_STR_EQ(nanousd_connection(specialize, "classExternalInput", 0),
                  "/Outside.outputs:value");

    nanousd_freeprim(specialize);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_composition_chained_inherits(void) {
    NanousdStage stage = nanousd_open(usda_path("chained_inherits.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim a = nanousd_primpath(stage, "/Chained_A");
    ASSERT(a != NULL);

    int ok = 0;
    int bVal = nanousd_attribi(a, "bVal", &ok);
    ASSERT(ok == 1);
    ASSERT(bVal == 2);

    ok = 0;
    int cVal = nanousd_attribi(a, "cVal", &ok);
    ASSERT(ok == 1);
    ASSERT(cVal == 3);

    nanousd_freeprim(a);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_composition_implied_inherits(void) {
    NanousdStage stage = nanousd_open(usda_path("implied_inherits_root.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim asset = nanousd_primpath(stage, "/LeftGroup/Asset");
    ASSERT(asset != NULL);

    int ok = 0;
    ASSERT(nanousd_attribi(asset, "assetLocalVal", &ok) == 40);
    ASSERT(ok == 1);
    ok = 0;
    ASSERT(nanousd_attribi(asset, "assetClassVal", &ok) == 30);
    ASSERT(ok == 1);
    ok = 0;
    ASSERT(nanousd_attribi(asset, "groupClassVal", &ok) == 20);
    ASSERT(ok == 1);
    ok = 0;
    ASSERT(nanousd_attribi(asset, "rootClassVal", &ok) == 10);
    ASSERT(ok == 1);
    ok = 0;
    ASSERT(nanousd_attribi(asset, "conflictVal", &ok) == 10);
    ASSERT(ok == 1);

    NanousdPrim child = nanousd_primpath(stage, "/LeftGroup/Asset/SharedChild");
    ASSERT(child != NULL);
    ok = 0;
    ASSERT(nanousd_attribi(child, "assetChildVal", &ok) == 31);
    ASSERT(ok == 1);
    ok = 0;
    ASSERT(nanousd_attribi(child, "groupChildVal", &ok) == 21);
    ASSERT(ok == 1);
    ok = 0;
    ASSERT(nanousd_attribi(child, "rootChildVal", &ok) == 11);
    ASSERT(ok == 1);

    NanousdPrim no_upstream = nanousd_primpath(stage, "/LeftGroup/NoUpstream");
    ASSERT(no_upstream != NULL);
    ok = 0;
    ASSERT(nanousd_attribi(no_upstream, "assetOnlyClassVal", &ok) == 50);
    ASSERT(ok == 1);

    int ndiag = 0;
    NanousdDiagnostic* diags = nanousd_diagnostics(stage, &ndiag);
    ASSERT(ndiag == 0);
    nanousd_free_diagnostics(diags, ndiag);

    nanousd_freeprim(no_upstream);
    nanousd_freeprim(child);
    nanousd_freeprim(asset);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_composition_implied_specializes(void) {
    NanousdStage stage = nanousd_open(usda_path("implied_specializes_root.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim asset = nanousd_primpath(stage, "/LeftGroup/Asset");
    ASSERT(asset != NULL);

    int ok = 0;
    ASSERT(nanousd_attribi(asset, "assetLocalVal", &ok) == 40);
    ASSERT(ok == 1);
    ok = 0;
    ASSERT(nanousd_attribi(asset, "assetClassVal", &ok) == 30);
    ASSERT(ok == 1);
    ok = 0;
    ASSERT(nanousd_attribi(asset, "groupClassVal", &ok) == 20);
    ASSERT(ok == 1);
    ok = 0;
    ASSERT(nanousd_attribi(asset, "rootClassVal", &ok) == 10);
    ASSERT(ok == 1);
    ok = 0;
    ASSERT(nanousd_attribi(asset, "assetBaseVal", &ok) == 130);
    ASSERT(ok == 1);
    ok = 0;
    ASSERT(nanousd_attribi(asset, "groupBaseVal", &ok) == 120);
    ASSERT(ok == 1);
    ok = 0;
    ASSERT(nanousd_attribi(asset, "rootBaseVal", &ok) == 110);
    ASSERT(ok == 1);
    ok = 0;
    ASSERT(nanousd_attribi(asset, "classConflictVal", &ok) == 10);
    ASSERT(ok == 1);
    ok = 0;
    ASSERT(nanousd_attribi(asset, "baseConflictVal", &ok) == 110);
    ASSERT(ok == 1);
    ok = 0;
    ASSERT(nanousd_attribi(asset, "localBeatsSpecializesVal", &ok) == 40);
    ASSERT(ok == 1);

    NanousdPrim shared_child = nanousd_primpath(stage, "/LeftGroup/Asset/SharedChild");
    ASSERT(shared_child != NULL);
    ok = 0;
    ASSERT(nanousd_attribi(shared_child, "assetChildVal", &ok) == 31);
    ASSERT(ok == 1);
    ok = 0;
    ASSERT(nanousd_attribi(shared_child, "groupChildVal", &ok) == 21);
    ASSERT(ok == 1);
    ok = 0;
    ASSERT(nanousd_attribi(shared_child, "rootChildVal", &ok) == 11);
    ASSERT(ok == 1);

    NanousdPrim base_child = nanousd_primpath(stage, "/LeftGroup/Asset/BaseChild");
    ASSERT(base_child != NULL);
    ok = 0;
    ASSERT(nanousd_attribi(base_child, "assetBaseChildVal", &ok) == 131);
    ASSERT(ok == 1);
    ok = 0;
    ASSERT(nanousd_attribi(base_child, "groupBaseChildVal", &ok) == 121);
    ASSERT(ok == 1);
    ok = 0;
    ASSERT(nanousd_attribi(base_child, "rootBaseChildVal", &ok) == 111);
    ASSERT(ok == 1);

    NanousdPrim no_upstream = nanousd_primpath(stage, "/LeftGroup/NoUpstream");
    ASSERT(no_upstream != NULL);
    ok = 0;
    ASSERT(nanousd_attribi(no_upstream, "assetOnlyClassVal", &ok) == 50);
    ASSERT(ok == 1);

    int ndiag = 0;
    NanousdDiagnostic* diags = nanousd_diagnostics(stage, &ndiag);
    ASSERT(ndiag == 0);
    nanousd_free_diagnostics(diags, ndiag);

    nanousd_freeprim(no_upstream);
    nanousd_freeprim(base_child);
    nanousd_freeprim(shared_child);
    nanousd_freeprim(asset);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_composition_livrps_strength_ordering(void) {
    NanousdStage stage = nanousd_open(
        "tests/composition/livrps_strength_root.usda");
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    int ok = 0;
    NanousdPrim depth = nanousd_primpath(stage, "/NamespaceDepth/Child");
    ASSERT(depth != NULL);
    ASSERT(nanousd_attribi(depth, "namespaceDepthVal", &ok) == 20);
    ASSERT(ok == 1);
    nanousd_freeprim(depth);

    ok = 0;
    NanousdPrim sibling = nanousd_primpath(stage, "/SiblingRefs");
    ASSERT(sibling != NULL);
    ASSERT(nanousd_attribi(sibling, "siblingRefVal", &ok) == 10);
    ASSERT(ok == 1);
    nanousd_freeprim(sibling);

    ok = 0;
    NanousdPrim authored =
        nanousd_primpath(stage, "/AuthoredVsImplied");
    ASSERT(authored != NULL);
    ASSERT(nanousd_attribi(
        authored, "authoredVsImpliedVal", &ok) == 10);
    ASSERT(ok == 1);
    nanousd_freeprim(authored);

    nanousd_close(stage);
    TEST_PASS();
}

static void test_composition_path_valued_listop_remap_variants(void) {
    NanousdStage stage = nanousd_open(usda_path("path_remap_root.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim variant = nanousd_primpath(stage, "/VariantInstance");
    ASSERT(variant != NULL);

    ASSERT(nanousd_nreltargets(variant, "variantTarget") == 1);
    ASSERT_STR_EQ(nanousd_reltarget(variant, "variantTarget", 0),
                  "/VariantInstance/VariantChild");

    ASSERT(nanousd_nconnections(variant, "variantInput") == 1);
    ASSERT_STR_EQ(nanousd_connection(variant, "variantInput", 0),
                  "/VariantInstance/VariantChild.outputs:value");

    ASSERT(nanousd_nreltargets(variant, "variantExternalTarget") == 1);
    ASSERT_STR_EQ(nanousd_reltarget(variant, "variantExternalTarget", 0),
                  "/Outside");

    ASSERT(nanousd_nconnections(variant, "variantExternalInput") == 1);
    ASSERT_STR_EQ(nanousd_connection(variant, "variantExternalInput", 0),
                  "/Outside.outputs:value");

    nanousd_freeprim(variant);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_composition_relocates_reference_child(void) {
    NanousdStage stage = nanousd_open(usda_path("relocates_root.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    NanousdPrim old_child = nanousd_primpath(stage, "/Root/Child");
    ASSERT(old_child == NULL);

    NanousdPrim moved = nanousd_primpath(stage, "/Root/MovedChild");
    ASSERT(moved != NULL);

    int ok = 0;
    const char* source_label = nanousd_attribs(moved, "sourceLabel", &ok);
    ASSERT(ok == 1);
    ASSERT_STR_EQ(source_label, "from_ref_child");

    ok = 0;
    const char* local_label = nanousd_attribs(moved, "localLabel", &ok);
    ASSERT(ok == 1);
    ASSERT_STR_EQ(local_label, "from_root_over");

    ok = 0;
    (void)nanousd_attribs(moved, "ignoredSourceLabel", &ok);
    ASSERT(ok == 0);

    ASSERT(nanousd_nreltargets(root, "childTarget") == 1);
    ASSERT_STR_EQ(nanousd_reltarget(root, "childTarget", 0),
                  "/Root/MovedChild");

    ASSERT(nanousd_nconnections(root, "childInput") == 1);
    ASSERT_STR_EQ(nanousd_connection(root, "childInput", 0),
                  "/Root/MovedChild.outputs:value");

    nanousd_freeprim(moved);
    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

static void assert_nested_relocates_root(NanousdStage stage,
                                         const char* root_path) {
    NanousdPrim root = nanousd_primpath(stage, root_path);
    ASSERT(root != NULL);

    char old_child_path[128];
    snprintf(old_child_path, sizeof(old_child_path), "%s/Child", root_path);
    NanousdPrim old_child = nanousd_primpath(stage, old_child_path);
    ASSERT(old_child == NULL);

    char moved_path[128];
    snprintf(moved_path, sizeof(moved_path),
             "%s/RenamedChild", root_path);
    NanousdPrim moved = nanousd_primpath(stage, moved_path);
    ASSERT(moved != NULL);

    int ok = 0;
    const char* leaf_label = nanousd_attribs(moved, "leafLabel", &ok);
    ASSERT(ok == 1);
    ASSERT_STR_EQ(leaf_label, "from_leaf_child");

    ok = 0;
    const char* mid_label = nanousd_attribs(moved, "midLabel", &ok);
    ASSERT(ok == 1);
    ASSERT_STR_EQ(mid_label, "from_mid_target");

    ok = 0;
    (void)nanousd_attribs(moved, "ignoredMidSourceLabel", &ok);
    ASSERT(ok == 0);

    ok = 0;
    const char* winner = nanousd_attribs(moved, "winner", &ok);
    ASSERT(ok == 1);
    ASSERT_STR_EQ(winner, "mid");

    ok = 0;
    (void)nanousd_attribs(moved, "staleTargetLabel", &ok);
    ASSERT(ok == 0);

    char removed_path[160];
    snprintf(removed_path, sizeof(removed_path),
             "%s/RemoveMe", root_path);
    NanousdPrim removed = nanousd_primpath(stage, removed_path);
    ASSERT(removed == NULL);

    ASSERT(nanousd_nreltargets(root, "removedTarget") == 0);
    ASSERT(nanousd_nconnections(root, "removedInput") == 0);

    char invalid_grand_path[160];
    snprintf(invalid_grand_path, sizeof(invalid_grand_path),
             "%s/InvalidGrandChild", root_path);
    NanousdPrim invalid_grand =
        nanousd_primpath(stage, invalid_grand_path);
    ASSERT(invalid_grand == NULL);

    char grand_path[160];
    snprintf(grand_path, sizeof(grand_path),
             "%s/RenamedChild/GrandChild", root_path);
    NanousdPrim grand = nanousd_primpath(stage, grand_path);
    ASSERT(grand != NULL);

    ok = 0;
    const char* grand_label = nanousd_attribs(grand, "grandLabel", &ok);
    ASSERT(ok == 1);
    ASSERT_STR_EQ(grand_label, "from_leaf_grand");

    char valid_nested_path[160];
    snprintf(valid_nested_path, sizeof(valid_nested_path),
             "%s/RenamedChild/ValidNested", root_path);
    NanousdPrim valid_nested =
        nanousd_primpath(stage, valid_nested_path);
    ASSERT(valid_nested == NULL);

    char valid_nested_moved_path[192];
    snprintf(valid_nested_moved_path, sizeof(valid_nested_moved_path),
             "%s/RenamedChild/ValidNestedMoved", root_path);
    NanousdPrim valid_nested_moved =
        nanousd_primpath(stage, valid_nested_moved_path);
    ASSERT(valid_nested_moved != NULL);

    ok = 0;
    const char* valid_nested_label =
        nanousd_attribs(valid_nested_moved, "validNestedLabel", &ok);
    ASSERT(ok == 1);
    ASSERT_STR_EQ(valid_nested_label, "from_leaf_valid_nested");

    ASSERT(nanousd_nreltargets(root, "childTarget") == 1);
    ASSERT_STR_EQ(nanousd_reltarget(root, "childTarget", 0), moved_path);

    char expected_connection[160];
    snprintf(expected_connection, sizeof(expected_connection),
             "%s.outputs:value", moved_path);
    ASSERT(nanousd_nconnections(root, "childInput") == 1);
    ASSERT_STR_EQ(nanousd_connection(root, "childInput", 0),
                  expected_connection);

    nanousd_freeprim(valid_nested_moved);
    nanousd_freeprim(grand);
    nanousd_freeprim(moved);
    nanousd_freeprim(root);
}

static void test_composition_relocates_nested_layer_stack(void) {
    NanousdStage stage = nanousd_open(usda_path("relocates_nested_root.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    assert_nested_relocates_root(stage, "/RootRef");
    assert_nested_relocates_root(stage, "/RootPayload");

    int ndiag = 0;
    NanousdDiagnostic* diags = nanousd_diagnostics(stage, &ndiag);
    int found_ancestral_diag = 0;
    for (int i = 0; i < ndiag; ++i) {
        if (diags[i].arc_type == 8 &&
            diags[i].message &&
            strstr(diags[i].message, "ancestral relocated path") != NULL) {
            found_ancestral_diag = 1;
            break;
        }
    }
    ASSERT(found_ancestral_diag == 1);
    nanousd_free_diagnostics(diags, ndiag);

    nanousd_close(stage);
    TEST_PASS();
}

static void test_composition_relocates_write_roundtrip(void) {
    NanousdStage stage = nanousd_open(usda_path("relocates_nested_root.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    char tmp_usda[1024];
    char tmp_usdc[1024];
    snprintf(tmp_usda, sizeof(tmp_usda), "%s",
             tmp_path("relocates_flat_rt.usda"));
    snprintf(tmp_usdc, sizeof(tmp_usdc), "%s",
             tmp_path("relocates_flat_rt.usdc"));

    ASSERT(nanousd_write_usda(stage, tmp_usda) == 1);
    ASSERT(nanousd_write_usdc(stage, tmp_usdc) == 1);
    nanousd_close(stage);

    NanousdStage usda = nanousd_open(tmp_usda);
    ASSERT(usda != NULL && nanousd_isvalid(usda));
    assert_nested_relocates_root(usda, "/RootRef");
    assert_nested_relocates_root(usda, "/RootPayload");
    nanousd_close(usda);

    NanousdStage usdc = nanousd_open(tmp_usdc);
    ASSERT(usdc != NULL && nanousd_isvalid(usdc));
    assert_nested_relocates_root(usdc, "/RootRef");
    assert_nested_relocates_root(usdc, "/RootPayload");
    nanousd_close(usdc);

    remove(tmp_usda);
    remove(tmp_usdc);
    TEST_PASS();
}

/* ============================================================
 * Package format — USDZ read and packaged resource resolution
 *
 * The fixture is generated as a stored ZIP package so every backend
 * receives the same .usdz bytes. The root layer is the first central
 * directory entry and resolves a package-internal sublayer and reference.
 * ============================================================ */

static void test_package_usdz_read(void) {
    static const unsigned char root[] =
        "#usda 1.0\n"
        "(\n"
        "    defaultPrim = \"World\"\n"
        "    subLayers = [\n"
        "        @../layers/weak.usda@\n"
        "    ]\n"
        ")\n"
        "\n"
        "def \"World\" (\n"
        "    references = @../models/model.usda@</Model>\n"
        ")\n"
        "{\n"
        "    double local = 1\n"
        "    asset inputs:texture = @../textures/diffuse.png@\n"
        "    asset inputs:packedRemote = @https://example.com/assets/car.usdz[textures/paint.png]@\n"
        "}\n";
    static const unsigned char weak[] =
        "#usda 1.0\n"
        "over \"World\"\n"
        "{\n"
        "    double weak = 7\n"
        "}\n";
    static const unsigned char model[] =
        "#usda 1.0\n"
        "def \"Model\"\n"
        "{\n"
        "    double size = 5\n"
        "}\n";
    static const unsigned char texture_bytes[] = "placeholder texture bytes";
    ZipTestEntry entries[4];
    char pkg[1024];
    char explicit_id[1200];
    char expected_texture[1200];
    char resolved_texture[1200];
    unsigned char* texture_data = NULL;
    size_t texture_size = 0;
    int ok = 0;

    snprintf(pkg, sizeof(pkg), "%s", tmp_path("nanousd_compliance_package.usdz"));
    remove(pkg);

    entries[0].name = "scenes/root.usda";
    entries[0].data = root;
    entries[0].size = (uint32_t)(sizeof(root) - 1);
    entries[1].name = "layers/weak.usda";
    entries[1].data = weak;
    entries[1].size = (uint32_t)(sizeof(weak) - 1);
    entries[2].name = "models/model.usda";
    entries[2].data = model;
    entries[2].size = (uint32_t)(sizeof(model) - 1);
    entries[3].name = "textures/diffuse.png";
    entries[3].data = texture_bytes;
    entries[3].size = (uint32_t)(sizeof(texture_bytes) - 1);
    ASSERT(write_stored_zip(pkg, entries, 4) == 1);

    NanousdStage stage = nanousd_open(pkg);
    ASSERT(stage != NULL);
    ASSERT_MSG(nanousd_isvalid(stage) == 1, nanousd_error(stage));

    NanousdPrim world = nanousd_primpath(stage, "/World");
    ASSERT(world != NULL);

    ok = 0;
    ASSERT_FLOAT_EQ(nanousd_attribd(world, "local", &ok), 1.0, 0.001);
    ASSERT(ok == 1);
    ok = 0;
    ASSERT_FLOAT_EQ(nanousd_attribd(world, "weak", &ok), 7.0, 0.001);
    ASSERT(ok == 1);
    ok = 0;
    ASSERT_FLOAT_EQ(nanousd_attribd(world, "size", &ok), 5.0, 0.001);
    ASSERT(ok == 1);
    snprintf(expected_texture, sizeof(expected_texture),
             "%s[textures/diffuse.png]", pkg);
    ok = 0;
    const char* texture = nanousd_attribasset(world, "inputs:texture", &ok);
    ASSERT(ok == 1);
    ASSERT(texture != NULL);
    ASSERT_STR_EQ(texture, expected_texture);
    ASSERT(nanousd_stage_resolve_asset_path(stage, "../textures/diffuse.png",
                                            resolved_texture,
                                            sizeof(resolved_texture)) == 1);
    ASSERT_STR_EQ(resolved_texture, expected_texture);
    ASSERT(nanousd_read_asset_bytes(texture, &texture_data, &texture_size) == 1);
    ASSERT(texture_data != NULL);
    ASSERT(texture_size == sizeof(texture_bytes) - 1);
    ASSERT(memcmp(texture_data, texture_bytes, texture_size) == 0);
    nanousd_free_bytes(texture_data);
    texture_data = NULL;
    texture_size = 0;
    ok = 0;
    const char* packed_remote =
        nanousd_attribasset(world, "inputs:packedRemote", &ok);
    ASSERT(ok == 1);
    ASSERT(packed_remote != NULL);
    ASSERT_STR_EQ(packed_remote,
                  "https://example.com/assets/car.usdz[textures/paint.png]");

    nanousd_freeprim(world);
    nanousd_close(stage);

    snprintf(explicit_id, sizeof(explicit_id), "%s[models/model.usda]", pkg);
    NanousdStage model_stage = nanousd_open(explicit_id);
    ASSERT(model_stage != NULL);
    ASSERT_MSG(nanousd_isvalid(model_stage) == 1, nanousd_error(model_stage));

    NanousdPrim model_prim = nanousd_primpath(model_stage, "/Model");
    ASSERT(model_prim != NULL);
    ok = 0;
    ASSERT_FLOAT_EQ(nanousd_attribd(model_prim, "size", &ok), 5.0, 0.001);
    ASSERT(ok == 1);

    nanousd_freeprim(model_prim);
    nanousd_close(model_stage);
    remove(pkg);
    TEST_PASS();
}

static void test_package_usdz_write(void) {
    NanousdStage stage = nanousd_create();
    const char* tmp = tmp_path("nanousd_compliance_package_write.usdz");
    int ok = 0;

    ASSERT(stage != NULL && nanousd_isvalid(stage));
    ASSERT(nanousd_set_stage_metadata_token(stage, "defaultPrim", "Root") == 1);

    NanousdPrim root = nanousd_define_prim(stage, "/Root", "Xform");
    ASSERT(root != NULL);
    ASSERT(nanousd_create_attrib(root, "size", "double") == 1);
    ASSERT(nanousd_set_attribd(root, "size", 3.5) == 1);
    nanousd_freeprim(root);

    remove(tmp);
    ASSERT(nanousd_write_usdz(stage, tmp) == 1);
    ASSERT(file_starts_with(tmp, "PK\003\004", 4));
    nanousd_close(stage);

    NanousdStage reopened = nanousd_open(tmp);
    ASSERT(reopened != NULL && nanousd_isvalid(reopened));
    NanousdPrim reopened_root = nanousd_primpath(reopened, "/Root");
    ASSERT(reopened_root != NULL);
    ASSERT_FLOAT_EQ(nanousd_attribd(reopened_root, "size", &ok), 3.5, 1e-9);
    ASSERT(ok == 1);
    nanousd_freeprim(reopened_root);
    nanousd_close(reopened);

    ASSERT(nanousd_write_usdz(NULL, tmp) == 0);
    NanousdStage null_path_stage = nanousd_create();
    ASSERT(null_path_stage != NULL);
    ASSERT(nanousd_write_usdz(null_path_stage, NULL) == 0);
    nanousd_close(null_path_stage);

    remove(tmp);
    TEST_PASS();
}

static void test_file_uri_resource_open(void) {
    char path[1024];
    char uri[1200];
    int ok = 0;

    snprintf(path, sizeof(path), "%s", tmp_path("nanousd_file_uri_resource.usda"));
    remove(path);

    FILE* f = fopen(path, "wb");
    ASSERT(f != NULL);
    fputs("#usda 1.0\n"
          "\n"
          "def \"World\"\n"
          "{\n"
          "    double value = 2\n"
          "}\n",
          f);
    fclose(f);

    file_uri_from_path(path, uri, sizeof(uri));
    NanousdStage stage = nanousd_open(uri);
    ASSERT(stage != NULL);
    ASSERT_MSG(nanousd_isvalid(stage) == 1, nanousd_error(stage));

    NanousdPrim world = nanousd_primpath(stage, "/World");
    ASSERT(world != NULL);
    ASSERT_FLOAT_EQ(nanousd_attribd(world, "value", &ok), 2.0, 0.001);
    ASSERT(ok == 1);

    nanousd_freeprim(world);
    nanousd_close(stage);
    remove(path);
    TEST_PASS();
}

/* ============================================================
 * HasAPI / apiSchemas (spec Section 13)
 * ============================================================ */

static void test_hasapi(void) {
    NanousdStage stage = nanousd_open(usda_path("api_schemas.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim withAPI = nanousd_primpath(stage, "/WithAPI");
    ASSERT(withAPI != NULL);

    /* HasAPI should find the applied schema */
    ASSERT(nanousd_hasapi(withAPI, "CollectionAPI") == 1);

    /* Non-applied schema */
    ASSERT(nanousd_hasapi(withAPI, "ColorSpaceAPI") == 0);

    nanousd_freeprim(withAPI);

    /* Prim without apiSchemas */
    NanousdPrim noAPI = nanousd_primpath(stage, "/NoAPI");
    ASSERT(noAPI != NULL);
    ASSERT(nanousd_hasapi(noAPI, "CollectionAPI") == 0);
    nanousd_freeprim(noAPI);

    nanousd_close(stage);
    TEST_PASS();
}

/* Prim ListOp read — apiSchemas is a listop field on the prim */
static void test_prim_listop(void) {
    NanousdStage stage = nanousd_open(usda_path("api_schemas.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim withAPI = nanousd_primpath(stage, "/WithAPI");
    ASSERT(withAPI != NULL);

    NanousdListOp op = nanousd_prim_listop(withAPI, "apiSchemas");
    ASSERT(op != NULL);

    /* Should have at least 1 item */
    int n = nanousd_listop_nitems(op);
    ASSERT(n >= 1);

    /* First item should contain "CollectionAPI" */
    const char* item = nanousd_listop_item(op, 0);
    ASSERT(strstr(item, "CollectionAPI") != NULL);

    nanousd_listop_free(op);

    /* Prim without apiSchemas returns NULL */
    NanousdPrim noAPI = nanousd_primpath(stage, "/NoAPI");
    ASSERT(noAPI != NULL);
    NanousdListOp noOp = nanousd_prim_listop(noAPI, "apiSchemas");
    ASSERT(noOp == NULL);
    nanousd_freeprim(noAPI);

    nanousd_freeprim(withAPI);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Property ordering (spec Section 10.4)
 * ============================================================ */

static void test_property_order(void) {
    NanousdStage stage = nanousd_open(usda_path("property_order.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim ordered = nanousd_primpath(stage, "/Ordered");
    ASSERT(ordered != NULL);

    /* The prim has properties = ["charlie", "alpha", "bravo"]
     * so attribute enumeration should respect that order.
     * We check that the first 3 authored attributes appear in
     * the specified order. */
    int na = nanousd_nattribs(ordered);
    ASSERT(na >= 3);

    /* Collect the first 3 attribute names (copy immediately —
     * the returned pointer may be reused by subsequent calls) */
    char name0[64], name1[64], name2[64];
    snprintf(name0, sizeof(name0), "%s", nanousd_attribname(ordered, 0));
    snprintf(name1, sizeof(name1), "%s", nanousd_attribname(ordered, 1));
    snprintf(name2, sizeof(name2), "%s", nanousd_attribname(ordered, 2));

    ASSERT_STR_EQ(name0, "charlie");
    ASSERT_STR_EQ(name1, "alpha");
    ASSERT_STR_EQ(name2, "bravo");

    nanousd_freeprim(ordered);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Write operations
 * ============================================================ */

static void test_write_scalar(void) {
    NanousdStage stage = nanousd_open(usda_path("writable.usda"));
    ASSERT(nanousd_isvalid(stage));

    NanousdPrim root = nanousd_defaultprim(stage);
    ASSERT(root != NULL);

    /* Modify float */
    int ok = 0;
    float orig = nanousd_attribf(root, "height", &ok);
    ASSERT(ok && fabs(orig - 5.0f) < 0.001f);
    ASSERT(nanousd_set_attribf(root, "height", 99.0f) == 1);
    float updated = nanousd_attribf(root, "height", &ok);
    ASSERT(ok && fabs(updated - 99.0f) < 0.001f);

    /* Modify double */
    ASSERT(nanousd_set_attribd(root, "width", 42.0) == 1);
    double w = nanousd_attribd(root, "width", &ok);
    ASSERT(ok && fabs(w - 42.0) < 0.001);

    /* Modify int */
    ASSERT(nanousd_set_attribi(root, "count", 100) == 1);
    ASSERT(nanousd_attribi(root, "count", &ok) == 100 && ok);

    /* Modify string */
    ASSERT(nanousd_set_attribs(root, "label", "world") == 1);
    const char* s = nanousd_attribs(root, "label", &ok);
    ASSERT(ok);
    ASSERT_STR_EQ(s, "world");

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_write_vector(void) {
    NanousdStage stage = nanousd_open(usda_path("writable.usda"));
    ASSERT(nanousd_isvalid(stage));
    NanousdPrim root = nanousd_defaultprim(stage);
    ASSERT(root != NULL);

    float v3[3] = {10.0f, 20.0f, 30.0f};
    ASSERT(nanousd_set_attribv3f(root, "position", v3) == 1);

    float out[3] = {0};
    ASSERT(nanousd_attribv3f(root, "position", out) == 1);
    ASSERT_FLOAT_EQ(out[0], 10.0f, 0.001f);
    ASSERT_FLOAT_EQ(out[1], 20.0f, 0.001f);
    ASSERT_FLOAT_EQ(out[2], 30.0f, 0.001f);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_write_time_samples(void) {
    NanousdStage stage = nanousd_open(usda_path("writable.usda"));
    ASSERT(nanousd_isvalid(stage));
    NanousdPrim root = nanousd_defaultprim(stage);
    ASSERT(root != NULL);

    /* Existing animated attr has 2 samples */
    ASSERT(nanousd_hassamples(root, "animated") == 1);
    ASSERT(nanousd_nsamplekeys(root, "animated") == 2);

    /* Add a 3rd sample */
    ASSERT(nanousd_set_samplef(root, "animated", 3.0, 30.0f) == 1);
    ASSERT(nanousd_nsamplekeys(root, "animated") == 3);

    /* Read back the new sample */
    int ok = 0;
    float val = nanousd_samplef(root, "animated", 3.0, &ok);
    ASSERT(ok && fabs(val - 30.0f) < 0.001f);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_write_clear_block(void) {
    NanousdStage stage = nanousd_open(usda_path("writable.usda"));
    ASSERT(nanousd_isvalid(stage));
    NanousdPrim root = nanousd_defaultprim(stage);
    ASSERT(root != NULL);

    /* Clear default value */
    int ok = 0;
    nanousd_attribf(root, "height", &ok);
    ASSERT(ok);
    ASSERT(nanousd_clear_default(root, "height") == 1);
    nanousd_attribf(root, "height", &ok);
    ASSERT(!ok);  /* default was cleared */

    /* Clear time samples */
    ASSERT(nanousd_hassamples(root, "animated") == 1);
    ASSERT(nanousd_clear_samples(root, "animated") == 1);
    ASSERT(nanousd_hassamples(root, "animated") == 0);

    /* Block attribute */
    ASSERT(nanousd_block_attrib(root, "width") == 1);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_write_create_attrib(void) {
    NanousdStage stage = nanousd_open(usda_path("writable.usda"));
    ASSERT(nanousd_isvalid(stage));
    NanousdPrim root = nanousd_defaultprim(stage);
    ASSERT(root != NULL);

    /* Attribute does not exist */
    ASSERT(nanousd_hasattrib(root, "newAttr") == 0);

    /* Create it */
    ASSERT(nanousd_create_attrib(root, "newAttr", "float") == 1);
    ASSERT(nanousd_hasattrib(root, "newAttr") == 1);

    /* Set a value on it */
    ASSERT(nanousd_set_attribf(root, "newAttr", 7.5f) == 1);
    int ok = 0;
    float val = nanousd_attribf(root, "newAttr", &ok);
    ASSERT(ok && fabs(val - 7.5f) < 0.001f);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Specifier semantics — extended (spec Section 11.5)
 * ============================================================ */

/* Class prims are reachable via GetPrimAtPath and appear in children */
static void test_specifier_class_reachable(void) {
    NanousdStage stage = nanousd_open(usda_path("specifiers.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    /* Class prims should be findable by path */
    NanousdPrim cls = nanousd_primpath(stage, "/Abstract");
    ASSERT(cls != NULL);
    ASSERT(nanousd_isabstract(cls) == 1);
    ASSERT(nanousd_isdefined(cls) == 1);  /* class is a defining specifier */

    /* Should be able to read attributes on a class prim */
    int ok = 0;
    float val = nanousd_attribf(cls, "value", &ok);
    ASSERT(ok && fabs(val - 2.0f) < 0.001f);

    nanousd_freeprim(cls);
    nanousd_close(stage);
    TEST_PASS();
}

/* Over-only prims excluded from GetChildren */
static void test_specifier_over_not_in_children(void) {
    NanousdStage stage = nanousd_open(usda_path("specifiers.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    /* Over "OverOnly" should not appear in traversal. */
    int n = nanousd_nprims(stage);
    int i;
    for (i = 0; i < n; i++) {
        NanousdPrim p = nanousd_prim(stage, i);
        ASSERT(strcmp(nanousd_name(p), "OverOnly") != 0);
        nanousd_freeprim(p);
    }

    nanousd_close(stage);
    TEST_PASS();
}

/* Over that gains def through composition becomes defined */
static void test_specifier_over_gains_def(void) {
    NanousdStage stage = nanousd_open(usda_path("specifiers_composition.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    /* The root layer has "over GainsDef", but the sublayer has "def Xform GainsDef".
     * The resolved specifier should be def (first non-over wins). */
    NanousdPrim prim = nanousd_primpath(stage, "/GainsDef");
    ASSERT(prim != NULL);
    ASSERT(nanousd_isdefined(prim) == 1);

    /* Should be able to read the attribute from the over layer */
    int ok = 0;
    float overVal = nanousd_attribf(prim, "overValue", &ok);
    ASSERT(ok && fabs(overVal - 10.0f) < 0.001f);

    /* Should also see the base value from the sublayer */
    float baseVal = nanousd_attribf(prim, "baseValue", &ok);
    ASSERT(ok && fabs(baseVal - 5.0f) < 0.001f);

    /* Traversal handles should keep the strongest over-spec provenance even
     * though the prim is populated by a weaker def. The strongest over authors
     * propertyOrder, so the traversed handle should enumerate those authored
     * attributes first. */
    int n = nanousd_nprims(stage);
    int found = 0;
    for (int i = 0; i < n; i++) {
        NanousdPrim p = nanousd_prim(stage, i);
        if (strcmp(nanousd_path(p), "/GainsDef") == 0) {
            char name0[64], name1[64];
            snprintf(name0, sizeof(name0), "%s", nanousd_attribname(p, 0));
            snprintf(name1, sizeof(name1), "%s", nanousd_attribname(p, 1));
            ASSERT_STR_EQ(name0, "overValue");
            ASSERT_STR_EQ(name1, "baseValue");
            found = 1;
        }
        nanousd_freeprim(p);
    }
    ASSERT(found == 1);

    nanousd_freeprim(prim);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Active/inactive — extended (spec Section 9.2)
 * ============================================================ */

/* Active child under inactive parent is still excluded */
static void test_inactive_nested(void) {
    NanousdStage stage = nanousd_open(usda_path("inactive_nested.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    /* Parent is active */
    NanousdPrim parent = nanousd_primpath(stage, "/Parent");
    ASSERT(parent != NULL);

    /* Inactive child should not be reachable */
    NanousdPrim inactive = nanousd_primpath(stage, "/Parent/Inactive");
    ASSERT(inactive == NULL);

    /* Active child under the inactive subtree should also be excluded */
    NanousdPrim deep = nanousd_primpath(stage, "/Parent/Inactive/ActiveChild");
    ASSERT(deep == NULL);

    /* Active sibling should be reachable */
    NanousdPrim sibling = nanousd_primpath(stage, "/Parent/ActiveSibling");
    ASSERT(sibling != NULL);

    /* Children of parent: only ActiveSibling should appear */
    int nc = nanousd_nchildren(parent);
    ASSERT(nc == 1);
    NanousdPrim child = nanousd_child(parent, 0);
    ASSERT(child != NULL);
    ASSERT_STR_EQ(nanousd_name(child), "ActiveSibling");
    nanousd_freeprim(child);

    nanousd_freeprim(sibling);
    nanousd_freeprim(parent);
    nanousd_close(stage);
    TEST_PASS();
}

/* An inactive defaultPrim must not be returned by default-prim lookup. */
static void test_inactive_default_prim_excluded(void) {
    NanousdStage stage = nanousd_open(
        "tests/composition/stage_population_inactive_default.usda");
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim dp = nanousd_defaultprim(stage);
    ASSERT(dp == NULL);

    NanousdPrim inactive = nanousd_primpath(stage, "/InactiveDefault");
    ASSERT(inactive == NULL);

    NanousdPrim active = nanousd_primpath(stage, "/ActivePeer");
    ASSERT(active != NULL);
    nanousd_freeprim(active);

    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Value resolution (spec Section 12)
 * ============================================================ */

/* TimeSamples win over default for specific time queries */
static void test_value_timesample_over_default(void) {
    NanousdStage stage = nanousd_open(usda_path("value_resolution.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim prim = nanousd_primpath(stage, "/WithTimeSamples");
    ASSERT(prim != NULL);

    /* Has both default and timeSamples */
    int ok = 0;

    /* Reading the default value (no time) should return the authored default */
    float defVal = nanousd_attribf(prim, "anim", &ok);
    ASSERT(ok && fabs(defVal - 0.0f) < 0.001f);

    /* Reading at a sampled time should return the time sample, not the default */
    float tsVal = nanousd_samplef(prim, "anim", 1.0, &ok);
    ASSERT(ok && fabs(tsVal - 10.0f) < 0.001f);

    float tsVal2 = nanousd_samplef(prim, "anim", 24.0, &ok);
    ASSERT(ok && fabs(tsVal2 - 20.0f) < 0.001f);

    nanousd_freeprim(prim);
    nanousd_close(stage);
    TEST_PASS();
}

/* Blocked attribute: reading should return not-found */
static void test_value_block(void) {
    NanousdStage stage = nanousd_open(usda_path("value_resolution.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim prim = nanousd_primpath(stage, "/WithBlock");
    ASSERT(prim != NULL);

    /* The blocked attribute should exist... */
    ASSERT(nanousd_hasattrib(prim, "blocked") == 1);

    /* ...but reading its value should return not-found (ok == 0) */
    int ok = 1;
    nanousd_attribf(prim, "blocked", &ok);
    ASSERT(ok == 0);

    /* Non-blocked attribute should still be readable */
    float readable = nanousd_attribf(prim, "readable", &ok);
    ASSERT(ok && fabs(readable - 42.0f) < 0.001f);

    nanousd_freeprim(prim);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Composition semantics — extended
 * ============================================================ */

/* Fixture-level composed stage population check. The root fixture uses
 * sublayers, references, and payloads. The expected fixture is already
 * squashed: it contains the same composed objects and authored fields, but
 * no composition operators. Both sides are serialized through the public
 * flatten-to-USDA API to remove formatting noise before comparison. */
static void test_fixture_squashed_layer_compare(void) {
    ASSERT(compare_flattened_fixture_json(
        "fixture_compare/squash_root.usda",
        NULL, 0,
        "fixture_compare/squash_expected.json"));
    TEST_PASS();
}

/* Fixture-level population mask check. The mask includes a descendant prim;
 * the populated stage must include that prim and its ancestors, and must
 * exclude descendants, siblings, and unrelated roots outside the mask. */
static void test_fixture_population_mask_compare(void) {
    const char* masks[] = {"/World/Keep/Leaf"};
    ASSERT(compare_flattened_fixture_json(
        "fixture_compare/mask_root.usda",
        masks, 1,
        "fixture_compare/mask_expected.json"));
    TEST_PASS();
}

/* Fixture-level namespace remapping check for path-valued listOp fields.
 * The root fixture maps relationship targets and connections through both
 * references and payloads; source-layer targets outside the mapped namespace
 * are expected to disappear from the squashed layer. */
static void test_fixture_path_remap_compare(void) {
    ASSERT(compare_flattened_fixture_json(
        "fixture_compare/remap_root.usda",
        NULL, 0,
        "fixture_compare/remap_expected.json"));
    TEST_PASS();
}

/* Fixture-level sampled value resolution check. The root fixture includes
 * per-opinion default-vs-timeSamples ordering, spline baking, and value
 * clips. The expected fixture is the sampled layer: splines and clips are
 * absent, and timeSamples contain the resolved sampled values. */
static void test_fixture_sampled_layer_compare(void) {
    ASSERT(compare_flattened_fixture_json(
        "fixture_compare/sampled_root.usda",
        NULL, 0,
        "fixture_compare/sampled_expected.json"));
    TEST_PASS();
}

/* Metamorphic format check: once a fixture has been flattened/sampled,
 * writing it as USDC and reopening it must preserve the same normalized
 * scene object model. This does not rely on a second expected file. */
static void test_fixture_usdc_roundtrip_metamorphic(void) {
    const char* masks[] = {"/World/Keep/Leaf"};
    ASSERT(compare_flattened_fixture_usdc_roundtrip_json(
        "fixture_compare/squash_root.usda", NULL, 0));
    ASSERT(compare_flattened_fixture_usdc_roundtrip_json(
        "fixture_compare/mask_root.usda", masks, 1));
    ASSERT(compare_flattened_fixture_usdc_roundtrip_json(
        "fixture_compare/remap_root.usda", NULL, 0));
    ASSERT(compare_flattened_fixture_usdc_roundtrip_json(
        "fixture_compare/sampled_root.usda", NULL, 0));
    TEST_PASS();
}

static void test_generated_format_roundtrip_matrix(void) {
    char tmp_usda[1024];
    char tmp_usdc[1024];
    NanousdStage stage = nanousd_create();
    NanousdStage usda = NULL;
    NanousdStage usdc = NULL;
    char* usda_json = NULL;
    char* usdc_json = NULL;
    const char* target0[] = {"/Generated/P1"};
    const char* target1[] = {"/Generated/P0", "/Generated/P2"};

    ASSERT(stage != NULL && nanousd_isvalid(stage));
    NanousdPrim root = nanousd_define_prim(stage, "/Generated", "Scope");
    ASSERT(root != NULL);
    ASSERT(nanousd_create_attrib(root, "rootValue", "double") == 1);
    ASSERT(nanousd_set_attribd(root, "rootValue", 12.5) == 1);
    nanousd_freeprim(root);

    NanousdPrim p0 = nanousd_define_prim(stage, "/Generated/P0", "Scope");
    ASSERT(p0 != NULL);
    ASSERT(nanousd_create_attrib(p0, "label", "string") == 1);
    ASSERT(nanousd_set_attribs(p0, "label", "zero") == 1);
    ASSERT(nanousd_create_attrib(p0, "rank", "int") == 1);
    ASSERT(nanousd_set_attribi(p0, "rank", 0) == 1);
    ASSERT(nanousd_create_rel(p0, "next") == 1);
    ASSERT(nanousd_set_reltargets(p0, "next", target0, 1) == 1);
    nanousd_freeprim(p0);

    NanousdPrim p1 = nanousd_define_prim(stage, "/Generated/P1", "Scope");
    ASSERT(p1 != NULL);
    ASSERT(nanousd_create_attrib(p1, "label", "string") == 1);
    ASSERT(nanousd_set_attribs(p1, "label", "one") == 1);
    ASSERT(nanousd_create_attrib(p1, "weight", "float") == 1);
    ASSERT(nanousd_set_samplef(p1, "weight", 0.0, 1.0f) == 1);
    ASSERT(nanousd_set_samplef(p1, "weight", 10.0, 3.0f) == 1);
    ASSERT(nanousd_create_rel(p1, "peers") == 1);
    ASSERT(nanousd_set_reltargets(p1, "peers", target1, 2) == 1);
    nanousd_freeprim(p1);

    NanousdPrim p2 = nanousd_define_prim(stage, "/Generated/P2", "Scope");
    ASSERT(p2 != NULL);
    ASSERT(nanousd_create_attrib(p2, "enabled", "bool") == 1);
    ASSERT(nanousd_set_attribb(p2, "enabled", 1) == 1);
    nanousd_freeprim(p2);

    snprintf(tmp_usda, sizeof(tmp_usda), "%s",
             tmp_path("nanousd_generated_matrix.usda"));
    snprintf(tmp_usdc, sizeof(tmp_usdc), "%s",
             tmp_path("nanousd_generated_matrix.usdc"));
    remove(tmp_usda);
    remove(tmp_usdc);
    ASSERT(nanousd_write_usda(stage, tmp_usda) == 1);
    ASSERT(nanousd_write_usdc(stage, tmp_usdc) == 1);
    nanousd_close(stage);

    usda = nanousd_open(tmp_usda);
    ASSERT(usda != NULL && nanousd_isvalid(usda));
    usdc = nanousd_open(tmp_usdc);
    ASSERT(usdc != NULL && nanousd_isvalid(usdc));
    ASSERT(append_normalized_stage_json(usda, &usda_json));
    ASSERT(append_normalized_stage_json(usdc, &usdc_json));
    if (strcmp(usda_json, usdc_json) != 0) {
        print_text_diff(usdc_json, usda_json, "USDC JSON", "USDA JSON");
        ASSERT(0);
    }

    free(usda_json);
    free(usdc_json);
    nanousd_close(usda);
    nanousd_close(usdc);
    remove(tmp_usda);
    remove(tmp_usdc);
    TEST_PASS();
}

static void test_generated_adversarial_usda_rejections(void) {
    static const char* cases[] = {
        "#usda 1.0\n"
        "def Scope \"Bad\"\n"
        "{\n"
        "    string text = \"unterminated\n"
        "}\n",

        "#usda 1.0\n"
        "def Scope \"Bad\"\n"
        "{\n"
        "    uint count = -1\n"
        "}\n",

        "#usda 1.0\n"
        "def Scope \"Bad\"\n"
        "{\n"
        "    float value = 1e\n"
        "}\n",

        "#usda 1.0\n"
        "def Scope \"Bad\"\n"
        "{\n"
        "    uchar tiny = 256\n"
        "}\n",

        "#usda 1.0\n"
        "def Scope \"Bad\"\n"
        "{\n"
        "    asset broken = @line\n"
        "break@\n"
        "}\n",

        "#usda 1.0\n"
        "def Scope \"Bad\"\n"
        "{\n"
        "    float[] values = [1, 2\n"
        "}\n",

        "#usda 1.0\n"
        "def Scope \"Bad\"\n"
        "{\n"
        "    rel target = </Missing\n"
        "}\n",
    };

    for (int i = 0; i < (int)(sizeof(cases) / sizeof(cases[0])); ++i) {
        char tmp[1024];
        NanousdStage stage = NULL;
        snprintf(tmp, sizeof(tmp), "%s", tmp_path("nanousd_generated_bad.usda"));
        remove(tmp);
        ASSERT(write_text_file(tmp, cases[i]));
        stage = nanousd_open(tmp);
        ASSERT(stage == NULL || nanousd_isvalid(stage) == 0);
        if (stage) nanousd_close(stage);
        remove(tmp);
    }
    TEST_PASS();
}

/* Sublayer opinion strength: stronger layer's value wins */
static void test_composition_sublayer_strength(void) {
    NanousdStage stage = nanousd_open(usda_path("composition_strength.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    /* Strong layer has overridden=100, weak has overridden=1 — strong wins */
    int ok = 0;
    float val = nanousd_attribf(root, "overridden", &ok);
    ASSERT(ok && fabs(val - 100.0f) < 0.001f);

    /* String also: strong has "strong", weak has "weak" */
    const char* src = nanousd_attribs(root, "source", &ok);
    ASSERT(ok);
    ASSERT_STR_EQ(src, "strong");

    /* Property only in weaker layer should still be visible */
    float weakOnly = nanousd_attribf(root, "weakOnly", &ok);
    ASSERT(ok && fabs(weakOnly - 42.0f) < 0.001f);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

/* Reference opinion strength: local values override referenced */
static void test_composition_reference_strength(void) {
    NanousdStage stage = nanousd_open(usda_path("ref_strength.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim prim = nanousd_primpath(stage, "/MyPrim");
    ASSERT(prim != NULL);

    /* Local layer has height=99, ref has height=5 — local wins */
    int ok = 0;
    float height = nanousd_attribf(prim, "height", &ok);
    ASSERT(ok && fabs(height - 99.0f) < 0.001f);

    /* Local has label="local", ref has label="from_ref" — local wins */
    const char* label = nanousd_attribs(prim, "label", &ok);
    ASSERT(ok);
    ASSERT_STR_EQ(label, "local");

    /* Property only in reference should still be visible */
    float refOnly = nanousd_attribf(prim, "refOnly", &ok);
    ASSERT(ok && fabs(refOnly - 77.0f) < 0.001f);

    /* Child from reference should be inherited */
    NanousdPrim child = nanousd_childname(prim, "Child");
    ASSERT(child != NULL);
    int count = nanousd_attribi(child, "count", &ok);
    ASSERT(ok && count == 10);

    nanousd_freeprim(child);
    nanousd_freeprim(prim);
    nanousd_close(stage);
    TEST_PASS();
}

static int has_invalid_retiming_scale_diagnostic(NanousdStage stage, int arc_type) {
    int count = 0;
    int found = 0;
    int i;
    NanousdDiagnostic* diags = nanousd_diagnostics(stage, &count);
    for (i = 0; i < count; ++i) {
        if (diags[i].severity == 2 &&
            diags[i].category == 13 &&
            diags[i].arc_type == arc_type) {
            found = 1;
            break;
        }
    }
    nanousd_free_diagnostics(diags, count);
    return found;
}

static int check_invalid_retiming_scale_case(
        const char* file_path,
        const char* prim_path,
        int arc_type) {
    char flat_path[1024];
    int ok = 0;
    double v0, v10;
    NanousdPrim prim = NULL;
    NanousdStage flat = NULL;
    NanousdStage stage = nanousd_open(file_path);
    if (stage == NULL || !nanousd_isvalid(stage)) {
        if (stage) nanousd_close(stage);
        return 0;
    }
    if (!has_invalid_retiming_scale_diagnostic(stage, arc_type)) {
        nanousd_close(stage);
        return 0;
    }

    snprintf(flat_path, sizeof(flat_path), "%s", tmp_path("nanousd_invalid_retiming_flat.usda"));
    if (nanousd_write_usda(stage, flat_path) != 1) {
        nanousd_close(stage);
        return 0;
    }
    nanousd_close(stage);

    flat = nanousd_open(flat_path);
    if (flat == NULL || !nanousd_isvalid(flat)) {
        if (flat) nanousd_close(flat);
        return 0;
    }
    prim = nanousd_primpath(flat, prim_path);
    if (prim == NULL) {
        nanousd_close(flat);
        return 0;
    }

    v0 = nanousd_sampled(prim, "value", 0.0, &ok);
    if (ok != 1 || fabs(v0 - 0.0) >= 0.001) {
        nanousd_freeprim(prim);
        nanousd_close(flat);
        return 0;
    }

    v10 = nanousd_sampled(prim, "value", 10.0, &ok);
    if (ok != 1 || fabs(v10 - 10.0) >= 0.001) {
        nanousd_freeprim(prim);
        nanousd_close(flat);
        return 0;
    }

    nanousd_freeprim(prim);
    nanousd_close(flat);
    return 1;
}

static void test_composition_invalid_retiming_scale(void) {
    ASSERT_MSG(check_invalid_retiming_scale_case(
        "tests/composition/invalid_retiming_sublayer_root.usda",
        "/Source",
        0),
        "invalid sublayer retiming scale should diagnose and use default retiming");
    ASSERT_MSG(check_invalid_retiming_scale_case(
        "tests/composition/invalid_retiming_ref_root.usda",
        "/Prim",
        1),
        "invalid external reference retiming scale should diagnose and use default retiming");
    ASSERT_MSG(check_invalid_retiming_scale_case(
        "tests/composition/invalid_retiming_internal_ref.usda",
        "/Prim",
        1),
        "invalid internal reference retiming scale should diagnose and use default retiming");
    ASSERT_MSG(check_invalid_retiming_scale_case(
        "tests/composition/invalid_retiming_payload_root.usda",
        "/Prim",
        2),
        "invalid external payload retiming scale should diagnose and use default retiming");
    ASSERT_MSG(check_invalid_retiming_scale_case(
        "tests/composition/invalid_retiming_internal_payload.usda",
        "/Prim",
        2),
        "invalid internal payload retiming scale should diagnose and use default retiming");
    TEST_PASS();
}

/* apiSchemas combined across composition layers, not first-wins */
static void test_composition_apischemas_combined(void) {
    NanousdStage stage = nanousd_open(usda_path("composition_apischemas.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    NanousdListOp op = nanousd_prim_listop(root, "apiSchemas");
    ASSERT(op != NULL);

    /* Should have at least 2 items combined from both layers */
    int n = nanousd_listop_nitems(op);
    ASSERT(n >= 2);

    /* Both schemas should be present (order: stronger prepend first) */
    int foundLights = 0, foundShadows = 0;
    int i;
    for (i = 0; i < n; i++) {
        const char* item = nanousd_listop_item(op, i);
        if (strstr(item, "lights") != NULL) foundLights = 1;
        if (strstr(item, "shadows") != NULL) foundShadows = 1;
    }
    ASSERT(foundLights);
    ASSERT(foundShadows);

    nanousd_listop_free(op);
    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * ListOp — extended
 * ============================================================ */

/* Explicit overrides prepend/append */
static void test_listop_explicit_overrides(void) {
    const char* explicit_items[] = { "x", "y" };
    const char* prepend_items[] = { "a", "b" };
    NanousdListOp strong = nanousd_listop_create_explicit(explicit_items, 2);
    NanousdListOp weak = nanousd_listop_create(prepend_items, 2, NULL, 0, NULL, 0);
    ASSERT(strong != NULL && weak != NULL);

    NanousdListOp combined = nanousd_listop_combine(strong, weak);
    ASSERT(combined != NULL);

    /* Explicit in stronger completely overrides weaker's prepend */
    ASSERT(nanousd_listop_is_explicit(combined) == 1);
    ASSERT(nanousd_listop_nitems(combined) == 2);
    ASSERT_STR_EQ(nanousd_listop_item(combined, 0), "x");
    ASSERT_STR_EQ(nanousd_listop_item(combined, 1), "y");

    nanousd_listop_free(combined);
    nanousd_listop_free(strong);
    nanousd_listop_free(weak);
    TEST_PASS();
}

/* Prepend + delete interaction */
static void test_listop_prepend_delete(void) {
    const char* strong_prepend[] = { "new" };
    const char* strong_delete[] = { "b" };
    const char* weak_items[] = { "a", "b", "c" };

    NanousdListOp strong = nanousd_listop_create(strong_prepend, 1, NULL, 0,
                                              strong_delete, 1);
    NanousdListOp weak = nanousd_listop_create_explicit(weak_items, 3);
    ASSERT(strong != NULL && weak != NULL);

    NanousdListOp combined = nanousd_listop_combine(strong, weak);
    ASSERT(combined != NULL);

    /* Result: prepend "new" to ["a","b","c"], then delete "b"
     * → ["new", "a", "c"] */
    int n = nanousd_listop_nitems(combined);
    ASSERT(n == 3);
    ASSERT_STR_EQ(nanousd_listop_item(combined, 0), "new");
    ASSERT_STR_EQ(nanousd_listop_item(combined, 1), "a");
    ASSERT_STR_EQ(nanousd_listop_item(combined, 2), "c");

    nanousd_listop_free(combined);
    nanousd_listop_free(strong);
    nanousd_listop_free(weak);
    TEST_PASS();
}

/* ============================================================
 * Prim child ordering (spec Section 10.4)
 * ============================================================ */

static void test_prim_child_order(void) {
    NanousdStage stage = nanousd_open(usda_path("prim_order.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim parent = nanousd_primpath(stage, "/Parent");
    ASSERT(parent != NULL);

    /* reorder nameChildren = ["charlie", "alpha", "bravo"]
     * Children should appear in that order. */
    int nc = nanousd_nchildren(parent);
    ASSERT(nc == 3);

    char n0[64], n1[64], n2[64];
    {
        NanousdPrim c0 = nanousd_child(parent, 0);
        NanousdPrim c1 = nanousd_child(parent, 1);
        NanousdPrim c2 = nanousd_child(parent, 2);
        ASSERT(c0 != NULL && c1 != NULL && c2 != NULL);
        snprintf(n0, sizeof(n0), "%s", nanousd_name(c0));
        snprintf(n1, sizeof(n1), "%s", nanousd_name(c1));
        snprintf(n2, sizeof(n2), "%s", nanousd_name(c2));
        nanousd_freeprim(c0);
        nanousd_freeprim(c1);
        nanousd_freeprim(c2);
    }

    ASSERT_STR_EQ(n0, "charlie");
    ASSERT_STR_EQ(n1, "alpha");
    ASSERT_STR_EQ(n2, "bravo");

    NanousdPrim unordered = nanousd_primpath(stage, "/Unordered");
    ASSERT(unordered != NULL);
    ASSERT(nanousd_nchildren(unordered) == 3);
    NanousdPrim u0 = nanousd_child(unordered, 0);
    NanousdPrim u1 = nanousd_child(unordered, 1);
    NanousdPrim u2 = nanousd_child(unordered, 2);
    ASSERT(u0 != NULL && u1 != NULL && u2 != NULL);
    ASSERT_STR_EQ(nanousd_name(u0), "charlie");
    ASSERT_STR_EQ(nanousd_name(u1), "alpha");
    ASSERT_STR_EQ(nanousd_name(u2), "bravo");
    nanousd_freeprim(u0);
    nanousd_freeprim(u1);
    nanousd_freeprim(u2);
    nanousd_freeprim(unordered);

    nanousd_freeprim(parent);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_composed_prim_order_from_weaker_layer(void) {
    NanousdStage stage = nanousd_open(
        "tests/composition/stage_population_order_root.usda");
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim parent = nanousd_primpath(stage, "/Parent");
    ASSERT(parent != NULL);
    ASSERT(nanousd_nchildren(parent) == 3);

    NanousdPrim c0 = nanousd_child(parent, 0);
    NanousdPrim c1 = nanousd_child(parent, 1);
    NanousdPrim c2 = nanousd_child(parent, 2);
    ASSERT(c0 != NULL && c1 != NULL && c2 != NULL);
    ASSERT_STR_EQ(nanousd_name(c0), "charlie");
    ASSERT_STR_EQ(nanousd_name(c1), "alpha");
    ASSERT_STR_EQ(nanousd_name(c2), "bravo");
    nanousd_freeprim(c0);
    nanousd_freeprim(c1);
    nanousd_freeprim(c2);

    ASSERT(nanousd_nprims(stage) == 4);
    NanousdPrim p0 = nanousd_prim(stage, 0);
    NanousdPrim p1 = nanousd_prim(stage, 1);
    NanousdPrim p2 = nanousd_prim(stage, 2);
    NanousdPrim p3 = nanousd_prim(stage, 3);
    ASSERT(p0 != NULL && p1 != NULL && p2 != NULL && p3 != NULL);
    ASSERT_STR_EQ(nanousd_path(p0), "/Parent");
    ASSERT_STR_EQ(nanousd_path(p1), "/Parent/charlie");
    ASSERT_STR_EQ(nanousd_path(p2), "/Parent/alpha");
    ASSERT_STR_EQ(nanousd_path(p3), "/Parent/bravo");
    nanousd_freeprim(p0);
    nanousd_freeprim(p1);
    nanousd_freeprim(p2);
    nanousd_freeprim(p3);

    nanousd_freeprim(parent);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_root_prim_order(void) {
    NanousdStage stage = nanousd_open("tests/usda/reorder.usda");
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    ASSERT(nanousd_nprims(stage) == 6);
    NanousdPrim p0 = nanousd_prim(stage, 0);
    NanousdPrim p1 = nanousd_prim(stage, 1);
    NanousdPrim p2 = nanousd_prim(stage, 2);
    NanousdPrim p3 = nanousd_prim(stage, 3);
    NanousdPrim p4 = nanousd_prim(stage, 4);
    NanousdPrim p5 = nanousd_prim(stage, 5);
    ASSERT(p0 != NULL && p1 != NULL && p2 != NULL &&
           p3 != NULL && p4 != NULL && p5 != NULL);
    ASSERT_STR_EQ(nanousd_path(p0), "/B");
    ASSERT_STR_EQ(nanousd_path(p1), "/A");
    ASSERT_STR_EQ(nanousd_path(p2), "/A/Second");
    ASSERT_STR_EQ(nanousd_path(p3), "/A/First");
    ASSERT_STR_EQ(nanousd_path(p4), "/A/Third");
    ASSERT_STR_EQ(nanousd_path(p5), "/C");
    nanousd_freeprim(p0);
    nanousd_freeprim(p1);
    nanousd_freeprim(p2);
    nanousd_freeprim(p3);
    nanousd_freeprim(p4);
    nanousd_freeprim(p5);

    nanousd_close(stage);
    TEST_PASS();
}

/* Property ordering — partial reorder: listed properties first,
 * unlisted ones follow in path element order. */
static void test_property_order_partial(void) {
    NanousdStage stage = nanousd_open(usda_path("property_order.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    /* Unordered prim: properties use path element ordering. */
    NanousdPrim unordered = nanousd_primpath(stage, "/Unordered");
    ASSERT(unordered != NULL);

    int na = nanousd_nattribs(unordered);
    ASSERT(na >= 3);

    char u0[64], u1[64], u2[64];
    snprintf(u0, sizeof(u0), "%s", nanousd_attribname(unordered, 0));
    snprintf(u1, sizeof(u1), "%s", nanousd_attribname(unordered, 1));
    snprintf(u2, sizeof(u2), "%s", nanousd_attribname(unordered, 2));

    ASSERT_STR_EQ(u0, "alpha");
    ASSERT_STR_EQ(u1, "bravo");
    ASSERT_STR_EQ(u2, "charlie");

    nanousd_freeprim(unordered);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_property_order_from_weaker_layer(void) {
    NanousdStage stage = nanousd_open(usda_path("property_order_composed_root.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim ordered = nanousd_primpath(stage, "/Ordered");
    ASSERT(ordered != NULL);

    ASSERT(nanousd_nattribs(ordered) >= 4);

    char name0[64], name1[64], name2[64], name3[64];
    snprintf(name0, sizeof(name0), "%s", nanousd_attribname(ordered, 0));
    snprintf(name1, sizeof(name1), "%s", nanousd_attribname(ordered, 1));
    snprintf(name2, sizeof(name2), "%s", nanousd_attribname(ordered, 2));
    snprintf(name3, sizeof(name3), "%s", nanousd_attribname(ordered, 3));

    ASSERT_STR_EQ(name0, "charlie");
    ASSERT_STR_EQ(name1, "alpha");
    ASSERT_STR_EQ(name2, "bravo");
    ASSERT_STR_EQ(name3, "delta");

    nanousd_freeprim(ordered);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_property_path_element_ordering(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim root = nanousd_define_prim(stage, "/Root", "");
    ASSERT(root != NULL);

    const char* authored[] = {
        "foobar",
        "Foobar",
        "_foobar",
        "foo_bar",
        "foo001bar001abc",
        "foo001bar002abc",
        "foo0001bar0002xyz",
        "foo00001bar",
        "a0",
        "a\xC3\xBC",
        "ab",
    };
    for (int i = 0; i < 11; ++i) {
        ASSERT(nanousd_create_attrib(root, authored[i], "float") == 1);
    }

    const char* expected[] = {
        "_foobar",
        "a0",
        "ab",
        "a\xC3\xBC",
        "foo_bar",
        /* AOUSD 1.0.1's example places Foobar/foobar here, but the
         * normative bullets say ASCII numbers order before ASCII letters. */
        "foo00001bar",
        "foo001bar001abc",
        "foo001bar002abc",
        "foo0001bar0002xyz",
        "Foobar",
        "foobar",
    };

    ASSERT(nanousd_nattribs(root) == 11);
    for (int i = 0; i < 11; ++i) {
        ASSERT_STR_EQ(nanousd_attribname(root, i), expected[i]);
    }

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Write-then-read roundtrip on composed stages
 * ============================================================ */

/* Write scalar to a simple stage, re-read through the same handle */
static void test_write_read_scalar_roundtrip(void) {
    NanousdStage stage = nanousd_open(usda_path("writable.usda"));
    ASSERT(nanousd_isvalid(stage));
    NanousdPrim root = nanousd_defaultprim(stage);
    ASSERT(root != NULL);

    int ok = 0;

    /* Verify initial value */
    float orig = nanousd_attribf(root, "height", &ok);
    ASSERT(ok && fabs(orig - 5.0f) < 0.001f);

    /* Write + re-read through same handle — no reacquire needed */
    ASSERT(nanousd_set_attribf(root, "height", 123.0f) == 1);
    float updated = nanousd_attribf(root, "height", &ok);
    ASSERT(ok && fabs(updated - 123.0f) < 0.001f);

    /* String: write + immediate re-read */
    ASSERT(nanousd_set_attribs(root, "label", "modified") == 1);
    const char* s = nanousd_attribs(root, "label", &ok);
    ASSERT(ok);
    ASSERT_STR_EQ(s, "modified");

    /* Int: write + immediate re-read */
    ASSERT(nanousd_set_attribi(root, "count", 999) == 1);
    int ival = nanousd_attribi(root, "count", &ok);
    ASSERT(ok && ival == 999);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

/* Write vector to stage, re-read through same handle */
static void test_write_read_vector_roundtrip(void) {
    NanousdStage stage = nanousd_open(usda_path("writable.usda"));
    ASSERT(nanousd_isvalid(stage));
    NanousdPrim root = nanousd_defaultprim(stage);
    ASSERT(root != NULL);

    /* Write vec3f + immediate re-read */
    float v[3] = {7.0f, 8.0f, 9.0f};
    ASSERT(nanousd_set_attribv3f(root, "position", v) == 1);
    float out[3] = {0};
    ASSERT(nanousd_attribv3f(root, "position", out) == 1);
    ASSERT_FLOAT_EQ(out[0], 7.0f, 0.001f);
    ASSERT_FLOAT_EQ(out[1], 8.0f, 0.001f);
    ASSERT_FLOAT_EQ(out[2], 9.0f, 0.001f);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

/* On a composed (sublayered) stage: verify initial composed values,
 * write to the strongest layer, and verify the write takes effect
 * while weak-layer-only values remain visible. */
static void test_write_composed_sublayer(void) {
    NanousdStage stage = nanousd_open(usda_path("write_composed.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    int ok = 0;

    /* strong_val: strong layer has 100, weak has 1 → composed = 100 */
    float sv = nanousd_attribf(root, "strong_val", &ok);
    ASSERT(ok && fabs(sv - 100.0f) < 0.001f);

    /* weak_val: only in weak layer → composed = 42 */
    float wv = nanousd_attribf(root, "weak_val", &ok);
    ASSERT(ok && fabs(wv - 42.0f) < 0.001f);

    /* name: only in weak layer → "base" */
    const char* name = nanousd_attribs(root, "name", &ok);
    ASSERT(ok);
    ASSERT_STR_EQ(name, "base");

    /* Write new strong_val — immediately readable through same handle */
    ASSERT(nanousd_set_attribf(root, "strong_val", 500.0f) == 1);
    sv = nanousd_attribf(root, "strong_val", &ok);
    ASSERT(ok && fabs(sv - 500.0f) < 0.001f);

    /* weak_val should still be visible and unchanged */
    wv = nanousd_attribf(root, "weak_val", &ok);
    ASSERT(ok && fabs(wv - 42.0f) < 0.001f);

    /* Write to a previously weak-only attribute (writes to strong layer) */
    ASSERT(nanousd_set_attribs(root, "name", "overridden") == 1);
    name = nanousd_attribs(root, "name", &ok);
    ASSERT(ok);
    ASSERT_STR_EQ(name, "overridden");

    /* Child from weak layer should still be accessible */
    NanousdPrim child = nanousd_childname(root, "Child");
    ASSERT(child != NULL);
    int count = nanousd_attribi(child, "count", &ok);
    ASSERT(ok && count == 5);
    nanousd_freeprim(child);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

/* On a referenced stage: verify composed values, write to the
 * referencing (local) layer, and verify the new value overrides
 * while reference-only values remain visible. */
static void test_write_composed_reference(void) {
    NanousdStage stage = nanousd_open(usda_path("write_ref.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim prim = nanousd_primpath(stage, "/MyPrim");
    ASSERT(prim != NULL);

    int ok = 0;

    /* height: local has 99, ref has 5 → composed = 99 */
    float h = nanousd_attribf(prim, "height", &ok);
    ASSERT(ok && fabs(h - 99.0f) < 0.001f);

    /* base_val: only in reference → composed = 10 */
    float bv = nanousd_attribf(prim, "base_val", &ok);
    ASSERT(ok && fabs(bv - 10.0f) < 0.001f);

    /* label: only in reference → "from_template" */
    const char* lbl = nanousd_attribs(prim, "label", &ok);
    ASSERT(ok);
    ASSERT_STR_EQ(lbl, "from_template");

    /* Write new height — immediately readable through same handle */
    ASSERT(nanousd_set_attribf(prim, "height", 200.0f) == 1);
    h = nanousd_attribf(prim, "height", &ok);
    ASSERT(ok && fabs(h - 200.0f) < 0.001f);

    /* base_val from reference still visible */
    bv = nanousd_attribf(prim, "base_val", &ok);
    ASSERT(ok && fabs(bv - 10.0f) < 0.001f);

    nanousd_freeprim(prim);
    nanousd_close(stage);
    TEST_PASS();
}

/* Write a time sample to a composed stage, verify it reads back,
 * and verify the original time samples still work. */
static void test_write_composed_timesample(void) {
    NanousdStage stage = nanousd_open(usda_path("write_composed.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    int ok = 0;

    /* Verify existing time samples from weak layer */
    ASSERT(nanousd_hassamples(root, "animated") == 1);
    float ts1 = nanousd_samplef(root, "animated", 1.0, &ok);
    ASSERT(ok && fabs(ts1 - 10.0f) < 0.001f);
    float ts24 = nanousd_samplef(root, "animated", 24.0, &ok);
    ASSERT(ok && fabs(ts24 - 20.0f) < 0.001f);

    /* Add a new time sample — immediately readable */
    ASSERT(nanousd_set_samplef(root, "animated", 48.0, 30.0f) == 1);
    float ts48 = nanousd_samplef(root, "animated", 48.0, &ok);
    ASSERT(ok && fabs(ts48 - 30.0f) < 0.001f);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

/* Create a new attribute on a composed stage, write to it, re-read */
static void test_write_composed_create_attrib(void) {
    NanousdStage stage = nanousd_open(usda_path("write_composed.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    /* Attribute should not exist yet */
    ASSERT(nanousd_hasattrib(root, "brand_new") == 0);

    /* Create it */
    ASSERT(nanousd_create_attrib(root, "brand_new", "float") == 1);
    ASSERT(nanousd_hasattrib(root, "brand_new") == 1);

    /* Write a value — immediately readable */
    ASSERT(nanousd_set_attribf(root, "brand_new", 3.14f) == 1);
    int ok = 0;
    float val = nanousd_attribf(root, "brand_new", &ok);
    ASSERT(ok && fabs(val - 3.14f) < 0.01f);

    /* Other composed attributes still work */
    float wv = nanousd_attribf(root, "weak_val", &ok);
    ASSERT(ok && fabs(wv - 42.0f) < 0.001f);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

/* Block an attribute on a composed stage, verify it becomes unreadable,
 * then clear the block (by setting a new value) and verify it's readable again */
static void test_write_composed_block_unblock(void) {
    NanousdStage stage = nanousd_open(usda_path("write_composed.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    int ok = 0;

    /* Verify initial value from weak layer */
    float wv = nanousd_attribf(root, "weak_val", &ok);
    ASSERT(ok && fabs(wv - 42.0f) < 0.001f);

    /* Block it in strong layer — immediately unreadable */
    ASSERT(nanousd_block_attrib(root, "weak_val") == 1);
    nanousd_attribf(root, "weak_val", &ok);
    ASSERT(ok == 0);

    /* Set a new value to un-block */
    ASSERT(nanousd_set_attribf(root, "weak_val", 99.0f) == 1);

    /* Should be readable again with the new value — same handle */
    float restored = nanousd_attribf(root, "weak_val", &ok);
    ASSERT(ok && fabs(restored - 99.0f) < 0.001f);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Null safety
 * ============================================================ */

static void test_null_safety(void) {
    /* Stage functions */
    ASSERT(nanousd_nprims(NULL) == 0);
    ASSERT(nanousd_prim(NULL, 0) == NULL);
    ASSERT(nanousd_primpath(NULL, "/Root") == NULL);
    ASSERT(nanousd_defaultprim(NULL) == NULL);

    /* Prim functions */
    ASSERT(nanousd_nchildren(NULL) == 0);
    ASSERT(nanousd_child(NULL, 0) == NULL);
    ASSERT(nanousd_childname(NULL, "x") == NULL);
    ASSERT(nanousd_prim_isvalid(NULL) == 0);
    ASSERT(nanousd_isactive(NULL) == 0);
    ASSERT(nanousd_isdefined(NULL) == 0);
    ASSERT(nanousd_isabstract(NULL) == 0);
    ASSERT(nanousd_isinstanceable(NULL) == 0);
    ASSERT(nanousd_path(NULL)[0] == '\0');
    ASSERT(nanousd_name(NULL)[0] == '\0');
    ASSERT(nanousd_typename(NULL)[0] == '\0');
    ASSERT(nanousd_kind(NULL)[0] == '\0');

    /* Attribute functions */
    ASSERT(nanousd_nattribs(NULL) == 0);
    ASSERT(nanousd_hasattrib(NULL, "x") == 0);
    ASSERT(nanousd_attribtype(NULL, "x")[0] == '\0');

    /* Relationship functions */
    ASSERT(nanousd_hasrel(NULL, "x") == 0);
    ASSERT(nanousd_nreltargets(NULL, "x") == 0);

    /* Collection functions */
    ASSERT(nanousd_collection_nmembers(NULL, "x") == 0);
    ASSERT(nanousd_collection_member(NULL, "x", 0)[0] == '\0');
    ASSERT(nanousd_collection_contains(NULL, "x", "/Root") == 0);

    /* TimeSample functions */
    ASSERT(nanousd_hassamples(NULL, "x") == 0);
    ASSERT(nanousd_nsamplekeys(NULL, "x") == 0);

    /* Schema functions */
    ASSERT(nanousd_isa(NULL, "Mesh") == 0);
    ASSERT(nanousd_hasapi(NULL, "CollectionAPI") == 0);

    /* Free with NULL is safe */
    nanousd_freeprim(NULL);
    nanousd_close(NULL);
    nanousd_path_free(NULL);
    nanousd_listop_free(NULL);

    /* Path functions with NULL */
    ASSERT(nanousd_path_is_absolute(NULL) == 0);
    ASSERT(nanousd_path_is_root(NULL) == 0);
    ASSERT(nanousd_path_is_property(NULL) == 0);
    ASSERT(nanousd_path_str(NULL)[0] == '\0');
    ASSERT(nanousd_path_name(NULL)[0] == '\0');
    ASSERT(nanousd_path_parent(NULL) == NULL);
    ASSERT(nanousd_path_append_child(NULL, "x") == NULL);
    ASSERT(nanousd_path_append_property(NULL, "x") == NULL);

    /* ListOp functions with NULL */
    ASSERT(nanousd_listop_is_explicit(NULL) == 0);
    ASSERT(nanousd_listop_nitems(NULL) == 0);
    ASSERT(nanousd_listop_nprepended(NULL) == 0);
    ASSERT(nanousd_listop_nappended(NULL) == 0);
    ASSERT(nanousd_listop_ndeleted(NULL) == 0);
    ASSERT(nanousd_listop_combine(NULL, NULL) == NULL);

    TEST_PASS();
}

/* ============================================================
 * USDC Write Support
 * ============================================================ */

/* Set stage-level double and string metadata, read back via generic accessor */
static void test_write_stage_metadata(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    /* Set double metadata */
    ASSERT(nanousd_set_stage_metadatad(stage, "metersPerUnit", 1.0) == 1);
    ASSERT(nanousd_set_stage_metadatad(stage, "timeCodesPerSecond", 60.0) == 1);

    /* Set string metadata */
    ASSERT(nanousd_set_stage_metadatas(stage, "upAxis", "Z") == 1);

    /* Read back */
    int ok = 0;
    double mpu = nanousd_metadatad(stage, "metersPerUnit", &ok);
    ASSERT(ok == 1);
    ASSERT_FLOAT_EQ(mpu, 1.0, 1e-9);

    ok = 0;
    double tcps = nanousd_metadatad(stage, "timeCodesPerSecond", &ok);
    ASSERT(ok == 1);
    ASSERT_FLOAT_EQ(tcps, 60.0, 1e-9);

    ok = 0;
    const char* up = nanousd_metadatas(stage, "upAxis", &ok);
    ASSERT(ok == 1);
    ASSERT_STR_EQ(up, "Z");

    /* Null-safety */
    ASSERT(nanousd_set_stage_metadatad(NULL, "metersPerUnit", 1.0) == 0);
    ASSERT(nanousd_set_stage_metadatas(NULL, "upAxis", "Z") == 0);
    ASSERT(nanousd_set_stage_metadatad(stage, NULL, 1.0) == 0);
    ASSERT(nanousd_set_stage_metadatas(stage, NULL, "Z") == 0);
    ASSERT(nanousd_set_stage_metadatas(stage, "upAxis", NULL) == 0);

    nanousd_close(stage);
    TEST_PASS();
}

/* Token-typed stage metadata: upAxis must be Token, not String, per USD spec. */
static void test_write_stage_metadata_token(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    /* Set token-typed upAxis (the spec-correct type) */
    ASSERT(nanousd_set_stage_metadata_token(stage, "upAxis", "Y") == 1);
    ASSERT(nanousd_set_stage_metadatad(stage, "metersPerUnit", 1.0) == 1);

    /* Author a root prim so the file has content */
    NanousdPrim root = nanousd_define_prim(stage, "/World", "Xform");
    ASSERT(root != NULL);
    nanousd_freeprim(root);

    /* Round-trip through USDC */
    const char* tmp = tmp_path("nanousd_meta_token_roundtrip.usdc");
    remove(tmp);
    ASSERT(nanousd_write_usdc(stage, tmp) == 1);
    nanousd_close(stage);

    NanousdStage reopened = nanousd_open(tmp);
    ASSERT(reopened != NULL && nanousd_isvalid(reopened));
    NanousdPrim w = nanousd_primpath(reopened, "/World");
    ASSERT(w != NULL);
    nanousd_freeprim(w);
    nanousd_close(reopened);

    /* Null-safety */
    ASSERT(nanousd_set_stage_metadata_token(NULL, "upAxis", "Y") == 0);
    NanousdStage s2 = nanousd_create();
    ASSERT(nanousd_set_stage_metadata_token(s2, NULL, "Y") == 0);
    ASSERT(nanousd_set_stage_metadata_token(s2, "upAxis", NULL) == 0);
    nanousd_close(s2);

    TEST_PASS();
}

/* Array time sample reads: samplev2f, samplearrayf/d/i */
static void test_array_time_samples(void) {
    NanousdStage stage = nanousd_open("tests/capi/capi_comprehensive.usda");
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    /* Vec2f time samples */
    ASSERT(nanousd_hassamples(root, "uv") == 1);
    float uv[2];
    ASSERT(nanousd_samplev2f(root, "uv", 0.0, uv) == 1);
    ASSERT(uv[0] == 0.0f && uv[1] == 0.0f);
    ASSERT(nanousd_samplev2f(root, "uv", 100.0, uv) == 1);
    ASSERT(uv[0] > 0.99f && uv[1] > 0.49f);

    /* Float array time samples */
    ASSERT(nanousd_hassamples(root, "weights") == 1);
    float wbuf[3];
    int n = nanousd_samplearrayf(root, "weights", 0.0, wbuf, 3);
    ASSERT(n == 3);
    ASSERT(wbuf[0] == 1.0f && wbuf[1] == 0.0f && wbuf[2] == 0.0f);
    n = nanousd_samplearrayf(root, "weights", 100.0, wbuf, 3);
    ASSERT(n == 3);
    ASSERT(wbuf[0] == 0.0f && wbuf[1] == 1.0f);

    /* Double array time samples */
    ASSERT(nanousd_hassamples(root, "scales") == 1);
    double dbuf[2];
    n = nanousd_samplearrayd(root, "scales", 0.0, dbuf, 2);
    ASSERT(n == 2);
    ASSERT(dbuf[0] == 1.0 && dbuf[1] == 1.0);
    n = nanousd_samplearrayd(root, "scales", 100.0, dbuf, 2);
    ASSERT(n == 2);
    ASSERT(dbuf[0] == 2.0 && dbuf[1] == 3.0);

    /* Double-typed vector/matrix array time samples flatten via samplearrayd
     * (regression: these previously returned -1 and were silently dropped). */
    ASSERT(nanousd_hassamples(root, "dpts") == 1);
    double dvbuf[6];
    n = nanousd_samplearrayd(root, "dpts", 0.0, dvbuf, 6);
    ASSERT(n == 6);
    ASSERT(dvbuf[0] == 1.5 && dvbuf[1] == 2.5 && dvbuf[2] == 3.5);
    ASSERT(dvbuf[3] == 4.5 && dvbuf[4] == 5.5 && dvbuf[5] == 6.5);
    n = nanousd_samplearrayd(root, "dpts", 100.0, dvbuf, 6);
    ASSERT(n == 6);
    ASSERT(dvbuf[0] == 7.0 && dvbuf[5] == 12.0);
    ASSERT(nanousd_hassamples(root, "dmats") == 1);
    double dmbuf[16];
    n = nanousd_samplearrayd(root, "dmats", 0.0, dmbuf, 16);
    ASSERT(n == 16);
    ASSERT(dmbuf[0] == 2.0 && dmbuf[5] == 3.0 && dmbuf[10] == 4.0);
    ASSERT(dmbuf[12] == 5.0 && dmbuf[13] == 6.0 && dmbuf[14] == 7.0 && dmbuf[15] == 1.0);
    /* matrix3d (non-power-of-2 contiguous) flattens row-major */
    double dm3buf[9];
    n = nanousd_samplearrayd(root, "dm3", 0.0, dm3buf, 9);
    ASSERT(n == 9);
    ASSERT(dm3buf[0] == 1.0 && dm3buf[4] == 5.0 && dm3buf[8] == 9.0);
    /* quatd (distinct per-element copy path): 4 components, order-independent sum check */
    double dqbuf[4];
    n = nanousd_samplearrayd(root, "dquats", 0.0, dqbuf, 4);
    ASSERT(n == 4);
    {
        double qsum = dqbuf[0] + dqbuf[1] + dqbuf[2] + dqbuf[3];
        ASSERT(qsum > 0.999 && qsum < 1.001);
    }

    /* Int array time samples */
    ASSERT(nanousd_hassamples(root, "flags") == 1);
    int ibuf[3];
    n = nanousd_samplearrayi(root, "flags", 0.0, ibuf, 3);
    ASSERT(n == 3);
    ASSERT(ibuf[0] == 0 && ibuf[1] == 1 && ibuf[2] == 2);
    n = nanousd_samplearrayi(root, "flags", 100.0, ibuf, 3);
    ASSERT(n == 3);
    ASSERT(ibuf[0] == 3 && ibuf[1] == 4 && ibuf[2] == 5);

    /* Held extrapolation: past last sample returns the last sample's value */
    n = nanousd_samplearrayf(root, "weights", 999.0, wbuf, 3);
    ASSERT(n == 3);
    ASSERT(wbuf[0] == 0.0f && wbuf[1] == 1.0f && wbuf[2] == 0.0f);

    /* maxlen clamping */
    n = nanousd_samplearrayf(root, "weights", 0.0, wbuf, 2);
    ASSERT(n == 2);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

/* Write a stage to USDC, reopen it, verify the content round-trips correctly */
static void test_write_usdc_binary_magic_roundtrip(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    ASSERT(nanousd_set_stage_metadatad(stage, "metersPerUnit", 1.0) == 1);
    ASSERT(nanousd_set_stage_metadata_token(stage, "upAxis", "Z") == 1);

    NanousdPrim sphere = nanousd_define_prim(stage, "/World/Sphere", "Sphere");
    ASSERT(sphere != NULL);
    ASSERT(nanousd_create_attrib(sphere, "radius", "double") == 1);
    ASSERT(nanousd_set_attribd(sphere, "radius", 2.5) == 1);
    nanousd_freeprim(sphere);
    ASSERT(define_prim_ok(stage, "/World", "Xform"));

    const char* tmp = tmp_path("nanousd_binary_magic_roundtrip.usdc");
    remove(tmp);
    ASSERT(nanousd_write_usdc(stage, tmp) == 1);
    ASSERT(file_starts_with(tmp, "PXR-USDC", 8));
    nanousd_close(stage);

    NanousdStage s2 = nanousd_open(tmp);
    ASSERT(s2 != NULL && nanousd_isvalid(s2));
    int ok = 0;
    ASSERT_FLOAT_EQ(nanousd_metadatad(s2, "metersPerUnit", &ok), 1.0, 1e-9);
    ASSERT(ok == 1);
    NanousdPrim p = nanousd_primpath(s2, "/World/Sphere");
    ASSERT(p != NULL);
    ok = 0;
    ASSERT_FLOAT_EQ(nanousd_attribd(p, "radius", &ok), 2.5, 0.001);
    ASSERT(ok == 1);
    nanousd_freeprim(p);
    nanousd_close(s2);
    remove(tmp);
    TEST_PASS();
}

static void test_write_usdc_file_binary_roundtrip(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    ASSERT(nanousd_set_stage_metadatad(stage, "metersPerUnit", 0.01) == 1);
    NanousdPrim prim = nanousd_define_prim(stage, "/Root", "Xform");
    ASSERT(prim != NULL);
    ASSERT(nanousd_create_attrib(prim, "count", "int") == 1);
    ASSERT(nanousd_set_attribi(prim, "count", 77) == 1);
    nanousd_freeprim(prim);

    const char* tmp = tmp_path("nanousd_file_binary_roundtrip.usdc");
    remove(tmp);
    ASSERT(nanousd_write_usdc(stage, tmp) == 1);
    ASSERT(file_starts_with(tmp, "PXR-USDC", 8));
    nanousd_close(stage);

    NanousdStage s2 = nanousd_open(tmp);
    ASSERT(s2 != NULL && nanousd_isvalid(s2));
    NanousdPrim root = nanousd_primpath(s2, "/Root");
    ASSERT(root != NULL);
    int ok = 0;
    ASSERT(nanousd_attribi(root, "count", &ok) == 77);
    ASSERT(ok == 1);
    nanousd_freeprim(root);
    nanousd_close(s2);
    remove(tmp);
    TEST_PASS();
}

static void test_write_usdc_file_uri_binary(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim prim = nanousd_define_prim(stage, "/Root", "Xform");
    ASSERT(prim != NULL);
    ASSERT(nanousd_create_attrib(prim, "count", "int") == 1);
    ASSERT(nanousd_set_attribi(prim, "count", 91) == 1);
    nanousd_freeprim(prim);

    char path_buf[1024];
    char uri_buf[1200];
    snprintf(path_buf, sizeof(path_buf), "%s", tmp_path("nanousd_file_uri_binary.usdc"));
    remove(path_buf);
    file_uri_from_path(path_buf, uri_buf, sizeof(uri_buf));

    ASSERT(nanousd_write_usdc(stage, uri_buf) == 1);
    ASSERT(file_starts_with(path_buf, "PXR-USDC", 8));
    nanousd_close(stage);

    NanousdStage s2 = nanousd_open(uri_buf);
    ASSERT(s2 != NULL && nanousd_isvalid(s2));
    NanousdPrim root = nanousd_primpath(s2, "/Root");
    ASSERT(root != NULL);
    int ok = 0;
    ASSERT(nanousd_attribi(root, "count", &ok) == 91);
    ASSERT(ok == 1);
    nanousd_freeprim(root);
    nanousd_close(s2);
    remove(path_buf);
    TEST_PASS();
}

static void test_write_usdc_roundtrip(void) {
    /* Build the stage */
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    ASSERT(nanousd_set_stage_metadatad(stage, "metersPerUnit", 1.0) == 1);
    ASSERT(nanousd_set_stage_metadatas(stage, "upAxis", "Z") == 1);

    NanousdPrim root = nanousd_define_prim(stage, "/Root", "Xform");
    ASSERT(root != NULL);
    ASSERT(nanousd_create_attrib(root, "height", "float") == 1);
    ASSERT(nanousd_set_attribf(root, "height", 7.5f) == 1);

    NanousdPrim child = nanousd_define_prim(stage, "/Root/Child", "Sphere");
    ASSERT(child != NULL);
    nanousd_freeprim(child);
    nanousd_freeprim(root);

    /* Write to a temp file */
    const char* tmp = tmp_path("nanousd_compliance_roundtrip.usdc");
    ASSERT(nanousd_write_usdc(stage, tmp) == 1);
    nanousd_close(stage);

    /* Re-open and verify */
    NanousdStage s2 = nanousd_open(tmp);
    ASSERT(s2 != NULL && nanousd_isvalid(s2));

    /* Metadata */
    int ok = 0;
    double mpu = nanousd_metadatad(s2, "metersPerUnit", &ok);
    ASSERT(ok == 1);
    ASSERT_FLOAT_EQ(mpu, 1.0, 1e-9);

    ok = 0;
    const char* up = nanousd_metadatas(s2, "upAxis", &ok);
    ASSERT(ok == 1);
    ASSERT_STR_EQ(up, "Z");

    /* Prim hierarchy */
    NanousdPrim r2 = nanousd_primpath(s2, "/Root");
    ASSERT(r2 != NULL);

    ok = 0;
    float h = nanousd_attribf(r2, "height", &ok);
    ASSERT(ok == 1);
    ASSERT_FLOAT_EQ(h, 7.5f, 0.001f);

    NanousdPrim c2 = nanousd_childname(r2, "Child");
    ASSERT(c2 != NULL);
    ASSERT_STR_EQ(nanousd_typename(c2), "Sphere");

    nanousd_freeprim(c2);
    nanousd_freeprim(r2);
    nanousd_close(s2);
    remove(tmp);
    TEST_PASS();
}

/* ============================================================
 * USDA write tests
 * ============================================================ */

static void test_write_usda_null_safety(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);
    ASSERT(nanousd_write_usda(NULL, tmp_path("x.usda")) == 0);
    ASSERT(nanousd_write_usda(stage, NULL) == 0);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_write_usda_basic(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);

    ASSERT(nanousd_set_stage_metadatad(stage, "metersPerUnit", 0.01) == 1);
    ASSERT(nanousd_set_stage_metadatas(stage, "upAxis", "Y") == 1);

    NanousdPrim prim = nanousd_define_prim(stage, "/Box", "Cube");
    ASSERT(prim != NULL);
    /* Cube schema defines size as double */
    ASSERT(nanousd_set_attribd(prim, "size", 2.0) == 1);
    nanousd_freeprim(prim);

    const char* tmp = tmp_path("nanousd_compliance_usda_basic.usda");
    ASSERT(nanousd_write_usda(stage, tmp) == 1);
    nanousd_close(stage);

    NanousdStage s2 = nanousd_open(tmp);
    ASSERT(s2 != NULL && nanousd_isvalid(s2));

    NanousdPrim p2 = nanousd_primpath(s2, "/Box");
    ASSERT(p2 != NULL);
    ASSERT_STR_EQ(nanousd_typename(p2), "Cube");

    int ok = 0;
    double sz = nanousd_attribd(p2, "size", &ok);
    ASSERT(ok == 1);
    ASSERT(sz > 1.999 && sz < 2.001);

    nanousd_freeprim(p2);
    nanousd_close(s2);
    remove(tmp);
    TEST_PASS();
}

static void test_write_usda_roundtrip(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);

    ASSERT(nanousd_set_stage_metadatad(stage, "metersPerUnit", 1.0) == 1);
    ASSERT(nanousd_set_stage_metadatas(stage, "upAxis", "Z") == 1);

    NanousdPrim root = nanousd_define_prim(stage, "/Root", "Xform");
    ASSERT(root != NULL);
    ASSERT(nanousd_create_attrib(root, "height", "float") == 1);
    ASSERT(nanousd_set_attribf(root, "height", 7.5f) == 1);

    NanousdPrim child = nanousd_define_prim(stage, "/Root/Child", "Sphere");
    ASSERT(child != NULL);
    nanousd_freeprim(child);
    nanousd_freeprim(root);

    const char* tmp = tmp_path("nanousd_compliance_usda_roundtrip.usda");
    ASSERT(nanousd_write_usda(stage, tmp) == 1);
    nanousd_close(stage);

    NanousdStage s2 = nanousd_open(tmp);
    ASSERT(s2 != NULL && nanousd_isvalid(s2));

    int ok = 0;
    double mpu = nanousd_metadatad(s2, "metersPerUnit", &ok);
    ASSERT(ok == 1);
    ASSERT_FLOAT_EQ(mpu, 1.0, 1e-9);

    ok = 0;
    const char* up = nanousd_metadatas(s2, "upAxis", &ok);
    ASSERT(ok == 1);
    ASSERT_STR_EQ(up, "Z");

    NanousdPrim r2 = nanousd_primpath(s2, "/Root");
    ASSERT(r2 != NULL);
    ASSERT_STR_EQ(nanousd_typename(r2), "Xform");

    ok = 0;
    float h = nanousd_attribf(r2, "height", &ok);
    ASSERT(ok == 1);
    ASSERT_FLOAT_EQ(h, 7.5f, 0.001f);

    NanousdPrim c2 = nanousd_childname(r2, "Child");
    ASSERT(c2 != NULL);
    ASSERT_STR_EQ(nanousd_typename(c2), "Sphere");

    nanousd_freeprim(c2);
    nanousd_freeprim(r2);
    nanousd_close(s2);
    remove(tmp);
    TEST_PASS();
}

static void test_write_file_uri_resources(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);

    NanousdPrim prim = nanousd_define_prim(stage, "/Box", "Cube");
    ASSERT(prim != NULL);
    ASSERT(nanousd_set_attribd(prim, "size", 3.0) == 1);
    nanousd_freeprim(prim);

    char usda_path_buf[1024];
    char usdc_path_buf[1024];
    char usda_uri[1200];
    char usdc_uri[1200];
    snprintf(usda_path_buf, sizeof(usda_path_buf), "%s",
             tmp_path("nanousd_compliance_file_uri_write.usda"));
    snprintf(usdc_path_buf, sizeof(usdc_path_buf), "%s",
             tmp_path("nanousd_compliance_file_uri_write.usdc"));
    remove(usda_path_buf);
    remove(usdc_path_buf);

    file_uri_from_path(usda_path_buf, usda_uri, sizeof(usda_uri));
    file_uri_from_path(usdc_path_buf, usdc_uri, sizeof(usdc_uri));

    ASSERT(nanousd_write_usda(stage, usda_uri) == 1);
    ASSERT(nanousd_write_usdc(stage, usdc_uri) == 1);
    nanousd_close(stage);

    NanousdStage usda = nanousd_open(usda_uri);
    ASSERT(usda != NULL && nanousd_isvalid(usda));
    NanousdPrim usda_box = nanousd_primpath(usda, "/Box");
    ASSERT(usda_box != NULL);
    int ok = 0;
    ASSERT_FLOAT_EQ(nanousd_attribd(usda_box, "size", &ok), 3.0, 0.001);
    ASSERT(ok == 1);
    nanousd_freeprim(usda_box);
    nanousd_close(usda);

    NanousdStage usdc = nanousd_open(usdc_uri);
    ASSERT(usdc != NULL && nanousd_isvalid(usdc));
    NanousdPrim usdc_box = nanousd_primpath(usdc, "/Box");
    ASSERT(usdc_box != NULL);
    ok = 0;
    ASSERT_FLOAT_EQ(nanousd_attribd(usdc_box, "size", &ok), 3.0, 0.001);
    ASSERT(ok == 1);
    nanousd_freeprim(usdc_box);
    nanousd_close(usdc);

    remove(usda_path_buf);
    remove(usdc_path_buf);
    TEST_PASS();
}

/* nanousd_write_usda_string returns a malloc'd USDA string */
static void test_write_usda_string(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);

    ASSERT(nanousd_set_stage_metadatas(stage, "upAxis", "Y") == 1);

    NanousdPrim prim = nanousd_define_prim(stage, "/Cube", "Cube");
    ASSERT(prim != NULL);
    /* Cube schema defines size as double */
    ASSERT(nanousd_set_attribd(prim, "size", 3.0) == 1);
    nanousd_freeprim(prim);

    const char* usda = nanousd_write_usda_string(stage);
    ASSERT(usda != NULL);
    /* Should start with the USDA magic */
    ASSERT(strncmp(usda, "#usda 1.0", 9) == 0);

    /* Write to a temp file so we can reopen and verify round-trip */
    const char* tmp = tmp_path("nanousd_compliance_usda_string.usda");
    FILE* f = fopen(tmp, "wb");
    ASSERT(f != NULL);
    fputs(usda, f);
    fclose(f);
    nanousd_free_string(usda);
    nanousd_close(stage);

    NanousdStage s2 = nanousd_open(tmp);
    ASSERT(s2 != NULL && nanousd_isvalid(s2));

    NanousdPrim p2 = nanousd_primpath(s2, "/Cube");
    ASSERT(p2 != NULL);
    ASSERT_STR_EQ(nanousd_typename(p2), "Cube");

    int ok = 0;
    double sz = nanousd_attribd(p2, "size", &ok);
    ASSERT(ok == 1);
    ASSERT(sz > 2.999 && sz < 3.001);

    nanousd_freeprim(p2);
    nanousd_close(s2);
    remove(tmp);
    TEST_PASS();
}

/* nanousd_write_usda_string null safety */
static void test_write_usda_string_null(void) {
    ASSERT(nanousd_write_usda_string(NULL) == NULL);
    nanousd_free_string(NULL);  /* should not crash */
    TEST_PASS();
}

/* nanousd_define_prim_s creates a prim with the requested specifier */
static void test_define_prim_with_specifier(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);

    NanousdPrim over  = nanousd_define_prim_s(stage, "/Over",  NULL,   "over");
    NanousdPrim cls   = nanousd_define_prim_s(stage, "/Class", "Xform", "class");
    NanousdPrim def   = nanousd_define_prim_s(stage, "/Def",   "Sphere", "def");
    ASSERT(over  != NULL);
    ASSERT(cls   != NULL);
    ASSERT(def   != NULL);

    /* Write and re-read so specifier is encoded in the layer */
    const char* tmp = tmp_path("nanousd_define_prim_s.usdc");
    ASSERT(nanousd_write_usdc(stage, tmp) == 1);
    nanousd_freeprim(over);
    nanousd_freeprim(cls);
    nanousd_freeprim(def);
    nanousd_close(stage);

    NanousdStage s2 = nanousd_open(tmp);
    ASSERT(s2 != NULL && nanousd_isvalid(s2));

    /* "over" prim — not populated (no defining specifier in the stack) */
    NanousdPrim p_over = nanousd_primpath(s2, "/Over");
    ASSERT(p_over == NULL);   /* over-only prims are not populated */

    /* "class" prim — defined but abstract */
    NanousdPrim p_cls = nanousd_primpath(s2, "/Class");
    ASSERT(p_cls != NULL);
    ASSERT_STR_EQ(nanousd_typename(p_cls), "Xform");
    nanousd_freeprim(p_cls);

    /* "def" prim — normal concrete prim */
    NanousdPrim p_def = nanousd_primpath(s2, "/Def");
    ASSERT(p_def != NULL);
    ASSERT_STR_EQ(nanousd_typename(p_def), "Sphere");
    nanousd_freeprim(p_def);

    nanousd_close(s2);
    remove(tmp);
    TEST_PASS();
}

/*
 * Regression test: multi-root deep hierarchy prim count preserved through
 * USDC roundtrip.  The bug was that path element tokens for paths added
 * by spec-collection (e.g. reference/relationship path fields calling
 * EnsureAncestors) were missing from the TOKENS section because the token
 * pre-add pass ran before spec collection.  This caused the parser to find
 * an out-of-bounds token index in the PATHS jump-tree and abort early,
 * silently dropping most prims.
 */
static void test_write_usdc_multi_root_prim_count(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);

    /* Three top-level prims (multiple roots) with varying depths */
    ASSERT(define_prim_ok(stage, "/Environment", "Xform"));
    ASSERT(define_prim_ok(stage, "/Environment/defaultLight", "DistantLight"));

    ASSERT(define_prim_ok(stage, "/Render", "Xform"));
    ASSERT(define_prim_ok(stage, "/Render/OmniverseKit", "Xform"));
    ASSERT(define_prim_ok(stage, "/Render/OmniverseKit/HydraTextures", "Xform"));
    ASSERT(define_prim_ok(stage, "/Render/Vars", "Xform"));
    ASSERT(define_prim_ok(stage, "/Render/Vars/LdrColor", "RenderVar"));

    ASSERT(define_prim_ok(stage, "/World", "Xform"));
    ASSERT(define_prim_ok(stage, "/World/GroundPlane", "Xform"));
    ASSERT(define_prim_ok(stage, "/World/GroundPlane/CollisionMesh", "Mesh"));
    ASSERT(define_prim_ok(stage, "/World/GroundPlane/CollisionPlane", "Plane"));
    ASSERT(define_prim_ok(stage, "/World/Cube1", "Mesh"));
    ASSERT(define_prim_ok(stage, "/World/Cube2", "Mesh"));
    ASSERT(define_prim_ok(stage, "/World/Cube3", "Mesh"));
    ASSERT(define_prim_ok(stage, "/World/BigBase", "Mesh"));

    /* 15 prims total */

    const char* tmp = tmp_path("nanousd_multi_root_prim_count.usdc");
    ASSERT(nanousd_write_usdc(stage, tmp) == 1);
    nanousd_close(stage);

    NanousdStage s2 = nanousd_open(tmp);
    ASSERT(s2 != NULL && nanousd_isvalid(s2));

    /* All three roots and their descendants must survive the roundtrip */
    ASSERT(primpath_exists(s2, "/Environment"));
    ASSERT(primpath_exists(s2, "/Environment/defaultLight"));
    ASSERT(primpath_exists(s2, "/Render"));
    ASSERT(primpath_exists(s2, "/Render/Vars/LdrColor"));
    ASSERT(primpath_exists(s2, "/World"));
    ASSERT(primpath_exists(s2, "/World/GroundPlane/CollisionMesh"));
    ASSERT(primpath_exists(s2, "/World/Cube3"));
    ASSERT(primpath_exists(s2, "/World/BigBase"));

    ASSERT(nanousd_nprims(s2) == 15);

    nanousd_close(s2);
    remove(tmp);
    TEST_PASS();
}

/* USDC writer must emit `primChildren` / `propertyChildren` fields on every
 * spec that has children, so foreign readers (e.g. pxr) can populate the
 * prim hierarchy from spec §16.3.10.27 TokenVector entries — the spec at
 * §16.3.8.4.6 says spec data is a (path, form, fields) triple and pxr's
 * implementation reads children from `primChildren` rather than the PATHS
 * jump-tree.
 *
 * Round-tripping nanousd→USDC→nanousd must also keep the same prim count
 * even after the writer authors these new fields. This is a regression
 * test against re-encoding the field as a string-array via the unknown-
 * type fallback (which would corrupt the type on a USDC→USDC round-trip).
 */
static void test_write_usdc_prim_children_fields(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);

    ASSERT(define_prim_ok(stage, "/World", "Xform"));
    ASSERT(define_prim_ok(stage, "/World/Sphere", "Sphere"));
    ASSERT(define_prim_ok(stage, "/World/Cube", "Cube"));
    ASSERT(define_prim_ok(stage, "/World/Sphere/Inner", "Mesh"));

    const char* tmp = tmp_path("nanousd_prim_children_fields.usdc");
    ASSERT(nanousd_write_usdc(stage, tmp) == 1);
    nanousd_close(stage);

    /* Pass 1: open + walk should see all 4 prims plus correct hierarchy. */
    NanousdStage s2 = nanousd_open(tmp);
    ASSERT(s2 != NULL && nanousd_isvalid(s2));
    ASSERT(nanousd_nprims(s2) == 4);
    ASSERT(primpath_exists(s2, "/World"));
    ASSERT(primpath_exists(s2, "/World/Sphere"));
    ASSERT(primpath_exists(s2, "/World/Cube"));
    ASSERT(primpath_exists(s2, "/World/Sphere/Inner"));

    /* Pass 2: round-trip the parsed file to USDC again. The writer must not
     * duplicate primChildren or shift its CrateTypeId from TokenVector to
     * String-array — both would break pxr compatibility and might also
     * trip up nanousd's own parser. */
    const char* tmp2 = tmp_path("nanousd_prim_children_fields_rt.usdc");
    ASSERT(nanousd_write_usdc(s2, tmp2) == 1);
    nanousd_close(s2);

    NanousdStage s3 = nanousd_open(tmp2);
    ASSERT(s3 != NULL && nanousd_isvalid(s3));
    ASSERT(nanousd_nprims(s3) == 4);
    ASSERT(primpath_exists(s3, "/World/Sphere/Inner"));
    nanousd_close(s3);

    remove(tmp);
    remove(tmp2);
    TEST_PASS();
}

/* nanousd_set_specifier changes the specifier on an existing prim */
static void test_set_specifier(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);

    /* Start as "def", promote to "class" */
    NanousdPrim prim = nanousd_define_prim(stage, "/MyPrim", "Sphere");
    ASSERT(prim != NULL);
    ASSERT(nanousd_set_specifier(prim, "class") == 1);
    nanousd_freeprim(prim);

    const char* tmp = tmp_path("nanousd_set_specifier.usdc");
    ASSERT(nanousd_write_usdc(stage, tmp) == 1);
    nanousd_close(stage);

    NanousdStage s2 = nanousd_open(tmp);
    ASSERT(s2 != NULL && nanousd_isvalid(s2));

    NanousdPrim p2 = nanousd_primpath(s2, "/MyPrim");
    ASSERT(p2 != NULL);
    ASSERT_STR_EQ(nanousd_typename(p2), "Sphere");
    nanousd_freeprim(p2);

    nanousd_close(s2);
    remove(tmp);
    TEST_PASS();
}

/* Token/Asset attribute write + USDA roundtrip */
static void test_set_attrib_token_asset(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);

    NanousdPrim prim = nanousd_define_prim(stage, "/Root", "Xform");
    ASSERT(prim != NULL);

    /* Create and set a token scalar */
    ASSERT(nanousd_create_attrib(prim, "purpose", "token") == 1);
    ASSERT(nanousd_set_attrib_token(prim, "purpose", "render") == 1);

    /* Create and set an asset scalar */
    ASSERT(nanousd_create_attrib(prim, "ref", "asset") == 1);
    ASSERT(nanousd_set_attrib_asset(prim, "ref", "./model.usd") == 1);

    /* Create and set a token array */
    ASSERT(nanousd_create_attrib(prim, "tags", "token[]") == 1);
    const char* tags[] = {"visible", "shadow", "proxy"};
    ASSERT(nanousd_set_attribarraytokens(prim, "tags", tags, 3) == 1);

    nanousd_freeprim(prim);

    /* Write to USDA and roundtrip */
    const char* tmp = tmp_path("nanousd_token_asset_roundtrip.usda");
    ASSERT(nanousd_write_usda(stage, tmp) == 1);
    nanousd_close(stage);

    NanousdStage s2 = nanousd_open(tmp);
    ASSERT(s2 != NULL && nanousd_isvalid(s2));

    NanousdPrim p2 = nanousd_primpath(s2, "/Root");
    ASSERT(p2 != NULL);

    /* Token scalar: read back via dedicated token reader */
    int ok = 0;
    const char* purpose = nanousd_attrib_token(p2, "purpose", &ok);
    ASSERT(ok == 1);
    ASSERT_STR_EQ(purpose, "render");

    /* String reader must NOT see token-typed values */
    ok = 99;
    nanousd_attribs(p2, "purpose", &ok);
    ASSERT(ok == 0);

    /* Asset scalar: read back via asset reader — path is resolved on read,
     * so the returned value is the resolved (absolute) path, not the raw
     * authored "./model.usd".  Check that it contains "model.usd". */
    ok = 0;
    const char* ref = nanousd_attribasset(p2, "ref", &ok);
    ASSERT(ok == 1);
    ASSERT(ref != NULL && strstr(ref, "model.usd") != NULL);

    /* Token array: read back via dedicated token array reader */
    int len = nanousd_attribarraytokens_len(p2, "tags");
    ASSERT(len == 3);
    ASSERT_STR_EQ(nanousd_attribarraytokens(p2, "tags", 0), "visible");
    ASSERT_STR_EQ(nanousd_attribarraytokens(p2, "tags", 1), "shadow");
    ASSERT_STR_EQ(nanousd_attribarraytokens(p2, "tags", 2), "proxy");

    /* String array reader must NOT see token arrays */
    ASSERT(nanousd_attribarrays_len(p2, "tags") == -1);

    nanousd_freeprim(p2);
    nanousd_close(s2);
    remove(tmp);
    TEST_PASS();
}

/* In-memory token array read via dedicated reader */
static void test_set_attrib_token_array_readback(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);

    NanousdPrim prim = nanousd_define_prim(stage, "/Root", "Xform");
    ASSERT(prim != NULL);

    ASSERT(nanousd_create_attrib(prim, "ops", "token[]") == 1);
    const char* ops[] = {"xformOp:translate", "xformOp:scale"};
    ASSERT(nanousd_set_attribarraytokens(prim, "ops", ops, 2) == 1);

    /* Token array reader must work */
    ASSERT(nanousd_attribarraytokens_len(prim, "ops") == 2);
    ASSERT_STR_EQ(nanousd_attribarraytokens(prim, "ops", 0), "xformOp:translate");
    ASSERT_STR_EQ(nanousd_attribarraytokens(prim, "ops", 1), "xformOp:scale");

    /* String array reader must NOT see token arrays */
    ASSERT(nanousd_attribarrays_len(prim, "ops") == -1);

    /* Token scalar reader: nanousd_attribs must NOT see token-typed values */
    ASSERT(nanousd_create_attrib(prim, "kind", "token") == 1);
    ASSERT(nanousd_set_attrib_token(prim, "kind", "component") == 1);
    int ok = 99;
    nanousd_attribs(prim, "kind", &ok);
    ASSERT(ok == 0);

    nanousd_freeprim(prim);
    nanousd_close(stage);
    TEST_PASS();
}

/* Token/Asset setter null safety */
static void test_set_attrib_token_asset_null(void) {
    /* Null prim should return 0 */
    ASSERT(nanousd_set_attrib_token(NULL, "x", "v") == 0);
    ASSERT(nanousd_set_attrib_asset(NULL, "x", "v") == 0);
    ASSERT(nanousd_set_attribarraytokens(NULL, "x", NULL, 0) == 0);
    ASSERT(nanousd_attribarraytokens_len(NULL, "x") == -1);
    ASSERT(nanousd_attribarraytokens(NULL, "x", 0) == NULL);
    TEST_PASS();
}

/* Add reference — internal (same-layer) with composed read */
static void test_add_reference_internal(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);

    /* Source prim with attribute */
    NanousdPrim source = nanousd_define_prim(stage, "/Source", "Xform");
    ASSERT(source != NULL);
    ASSERT(nanousd_create_attrib(source, "mass", "float") == 1);
    ASSERT(nanousd_set_attribf(source, "mass", 2.5f) == 1);
    nanousd_freeprim(source);

    /* Target prim references source */
    NanousdPrim target = nanousd_define_prim(stage, "/Target", "Xform");
    ASSERT(target != NULL);
    ASSERT(nanousd_add_reference(target, NULL, "/Source") == 1);

    /* Composed read should see the referenced attribute */
    int ok = 0;
    float mass = nanousd_attribf(target, "mass", &ok);
    ASSERT(ok == 1);
    ASSERT_FLOAT_EQ(mass, 2.5f, 0.001f);

    nanousd_freeprim(target);
    nanousd_close(stage);
    TEST_PASS();
}

/* Add reference — null safety */
static void test_add_reference_null(void) {
    ASSERT(nanousd_add_reference(NULL, "./foo.usd", "/Prim") == 0);
    TEST_PASS();
}

/* ============================================================
 * Relationship Creation
 * ============================================================ */

static void test_create_rel(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);

    /* Schema-defined relationship: setting targets without create_rel should work
       because PhysicsJoint defines physics:body0 in its schema */
    NanousdPrim joint = nanousd_define_prim(stage, "/joint", "PhysicsJoint");
    ASSERT(joint != NULL);
    const char* targets[] = {"/body0"};
    ASSERT(nanousd_set_reltargets(joint, "physics:body0", targets, 1) == 1);
    ASSERT(nanousd_nreltargets(joint, "physics:body0") == 1);
    ASSERT_STR_EQ(nanousd_reltarget(joint, "physics:body0", 0), "/body0");
    nanousd_freeprim(joint);

    /* Non-schema relationship: setting targets before create should fail */
    NanousdPrim prim = nanousd_define_prim(stage, "/xform", "Xform");
    ASSERT(prim != NULL);
    ASSERT(nanousd_set_reltargets(prim, "custom:myRel", targets, 1) == 0);
    ASSERT(nanousd_add_reltarget(prim, "custom:myRel", "/body0") == 0);

    /* Create then set */
    ASSERT(nanousd_create_rel(prim, "custom:myRel") == 1);
    ASSERT(nanousd_set_reltargets(prim, "custom:myRel", targets, 1) == 1);
    ASSERT(nanousd_nreltargets(prim, "custom:myRel") == 1);
    ASSERT_STR_EQ(nanousd_reltarget(prim, "custom:myRel", 0), "/body0");

    /* Add target */
    ASSERT(nanousd_add_reltarget(prim, "custom:myRel", "/body1") == 1);
    ASSERT(nanousd_nreltargets(prim, "custom:myRel") == 2);

    nanousd_freeprim(prim);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_create_rel_null(void) {
    ASSERT(nanousd_create_rel(NULL, "rel") == 0);
    TEST_PASS();
}

/* ============================================================
 * Schema Registration
 * ============================================================ */

/* GeomModelAPI is a bundled applied API schema; applying it must expose its
 * model:* draw-mode / card properties (parity with OpenUSD UsdGeomModelAPI).
 * Regression guard for the ALab load gap where these were not surfaced. */
static void test_geommodelapi_schema_attrs(void) {
    NanousdStage stage = nanousd_open(usda_path("geommodelapi.usda"));
    ASSERT(stage != NULL);
    NanousdPrim prim = nanousd_primpath(stage, "/Thing");
    ASSERT(prim != NULL);
    ASSERT(nanousd_hasattrib(prim, "model:drawMode") == 1);
    ASSERT(nanousd_hasattrib(prim, "model:cardGeometry") == 1);
    ASSERT(nanousd_hasattrib(prim, "model:applyDrawMode") == 1);
    ASSERT(nanousd_hasattrib(prim, "model:drawModeColor") == 1);
    ASSERT(nanousd_hasattrib(prim, "model:cardTextureXPos") == 1);
    ASSERT(nanousd_hasattrib(prim, "model:cardTextureZNeg") == 1);
    nanousd_freeprim(prim);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_register_schemas_json(void) {
    const char* json =
        "{"
        "  \"schemas\": {"
        "    \"TestRegType\": {"
        "      \"schemaKind\": \"typed\","
        "      \"parent\": \"\","
        "      \"properties\": {"
        "        \"testProp\": {"
        "          \"type\": \"float\","
        "          \"fallback\": 1.5"
        "        }"
        "      }"
        "    }"
        "  }"
        "}";

    ASSERT(nanousd_register_schemas_json(json) == 1);

    /* Verify by creating a stage with this type and checking IsA */
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);
    NanousdPrim p = nanousd_define_prim(stage, "/TestObj", "TestRegType");
    ASSERT(p != NULL);
    ASSERT(nanousd_isa(p, "TestRegType") == 1);

    /* Verify fallback via attribute read (schema fallback for unset attrib) */
    {
        int ok = 0;
        float val = nanousd_attribf(p, "testProp", &ok);
        ASSERT(ok == 1);
        ASSERT(val > 1.4f && val < 1.6f);
    }

    nanousd_freeprim(p);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_schema_autoapplies(void) {
    const char* json =
        "{"
        "  \"schemas\": {"
        "    \"ComplianceBaseThing\": {"
        "      \"schemaKind\": \"typed\","
        "      \"properties\": {"
        "        \"base:flag\": { \"type\": \"bool\", \"fallback\": false }"
        "      }"
        "    },"
        "    \"ComplianceDerivedThing\": {"
        "      \"schemaKind\": \"typed\","
        "      \"parent\": \"ComplianceBaseThing\""
        "    },"
        "    \"ComplianceAutoAPI\": {"
        "      \"schemaKind\": \"singleApply\","
        "      \"autoApplies\": [\"ComplianceBaseThing\"],"
        "      \"properties\": {"
        "        \"auto:enabled\": { \"type\": \"bool\", \"fallback\": true }"
        "      }"
        "    }"
        "  }"
        "}";
    char path[1024];
    FILE* f = NULL;

    ASSERT(nanousd_register_schemas_json(json) == 1);

    snprintf(path, sizeof(path), "%s", tmp_path("nanousd_schema_autoapplies.usda"));
    remove(path);
    f = fopen(path, "wb");
    ASSERT(f != NULL);
    fputs("#usda 1.0\n"
          "\n"
          "def ComplianceDerivedThing \"AutoPrim\"\n"
          "{\n"
          "}\n"
          "\n"
          "def \"Untyped\"\n"
          "{\n"
          "}\n",
          f);
    fclose(f);

    NanousdStage stage = nanousd_open(path);
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim prim = nanousd_primpath(stage, "/AutoPrim");
    ASSERT(prim != NULL);
    ASSERT(nanousd_hasapi(prim, "ComplianceAutoAPI") == 1);
    ASSERT(nanousd_hasattrib(prim, "auto:enabled") == 1);
    {
        int ok = 0;
        int enabled = nanousd_attribb(prim, "auto:enabled", &ok);
        ASSERT(ok == 1);
        ASSERT(enabled == 1);
    }
    nanousd_freeprim(prim);

    NanousdPrim untyped = nanousd_primpath(stage, "/Untyped");
    ASSERT(untyped != NULL);
    ASSERT(nanousd_hasapi(untyped, "ComplianceAutoAPI") == 0);
    ASSERT(nanousd_hasattrib(untyped, "auto:enabled") == 0);
    nanousd_freeprim(untyped);

    nanousd_close(stage);
    remove(path);
    TEST_PASS();
}

static void test_register_schemas_json_null(void) {
    ASSERT(nanousd_register_schemas_json(NULL) == 0);
    ASSERT(nanousd_register_schemas_json("{bad json}") == 0);
    TEST_PASS();
}

static void test_schema_attribute_names_exclude_relationship_defs(void) {
    const char* json =
        "{"
        "  \"schemas\": {"
        "    \"TestAttrNameCacheType\": {"
        "      \"schemaKind\": \"typed\","
        "      \"parent\": \"\","
        "      \"properties\": {"
        "        \"cacheAttr\": { \"type\": \"double\" },"
        "        \"cacheRel\": { \"type\": \"rel\" }"
        "      }"
        "    }"
        "  }"
        "}";

    ASSERT(nanousd_register_schemas_json(json) == 1);

    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);
    NanousdPrim prim = nanousd_define_prim(stage, "/Obj", "TestAttrNameCacheType");
    ASSERT(prim != NULL);

    int n = nanousd_nattribs(prim);
    int sawAttr = 0;
    int sawRel = 0;
    for (int i = 0; i < n; ++i) {
        const char* name = nanousd_attribname(prim, i);
        if (name && strcmp(name, "cacheAttr") == 0) sawAttr = 1;
        if (name && strcmp(name, "cacheRel") == 0) sawRel = 1;
    }
    ASSERT(sawAttr == 1);
    ASSERT(sawRel == 0);

    nanousd_freeprim(prim);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Color Spaces
 * ============================================================ */

static void test_color_spaces(void) {
    NanousdStage stage = nanousd_open(usda_path("color_spaces.usda"));
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);
    ASSERT_STR_EQ(nanousd_prim_resolved_colorspace(root), "lin_rec709_scene");

    NanousdPrim override = nanousd_primpath(stage, "/Root/Override");
    ASSERT(override != NULL);
    ASSERT_STR_EQ(nanousd_prim_resolved_colorspace(override), "srgb_p3d65_scene");
    ASSERT_STR_EQ(nanousd_attrib_resolved_colorspace(override, "inputs:diffuseColor"),
                  "srgb_p3d65_scene");

    NanousdPrim inherited = nanousd_primpath(stage, "/Root/Inherited");
    ASSERT(inherited != NULL);
    ASSERT_STR_EQ(nanousd_prim_resolved_colorspace(inherited), "lin_rec709_scene");
    ASSERT_STR_EQ(nanousd_attrib_resolved_colorspace(inherited, "inputs:diffuseColor"),
                  "lin_rec709_scene");

    NanousdPrim authored = nanousd_primpath(stage, "/Root/Authored");
    ASSERT(authored != NULL);
    int ok = 0;
    ASSERT_STR_EQ(nanousd_attrib_colorspace(authored, "inputs:diffuseColor", &ok),
                  "srgb_rec709_scene");
    ASSERT(ok == 1);
    ASSERT_STR_EQ(nanousd_attrib_resolved_colorspace(authored, "inputs:diffuseColor"),
                  "srgb_rec709_scene");

    NanousdPrim texture = nanousd_primpath(stage, "/Root/Texture");
    ASSERT(texture != NULL);
    ASSERT_STR_EQ(nanousd_attrib_resolved_colorspace(texture, "inputs:file"),
                  "lin_rec709_scene");

    NanousdPrim plain = nanousd_primpath(stage, "/Plain");
    ASSERT(plain != NULL);
    ASSERT_STR_EQ(nanousd_prim_resolved_colorspace(plain), "");
    ok = 99;
    ASSERT_STR_EQ(nanousd_attrib_colorspace(plain, "color", &ok), "");
    ASSERT(ok == 0);
    ASSERT_STR_EQ(nanousd_attrib_resolved_colorspace(plain, "color"), "");

    nanousd_freeprim(root);
    nanousd_freeprim(override);
    nanousd_freeprim(inherited);
    nanousd_freeprim(authored);
    nanousd_freeprim(texture);
    nanousd_freeprim(plain);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_color_space_authoring(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);
    NanousdPrim prim = nanousd_define_prim(stage, "/Looks/Mat", "Material");
    ASSERT(prim != NULL);
    ASSERT(nanousd_create_attrib(prim, "inputs:diffuseColor", "color3f") == 1);

    int ok = 99;
    ASSERT_STR_EQ(nanousd_attrib_colorspace(prim, "inputs:diffuseColor", &ok), "");
    ASSERT(ok == 0);
    ASSERT(nanousd_set_attrib_colorspace(prim, "inputs:diffuseColor",
                                         "srgb_rec709_scene") == 1);
    ASSERT_STR_EQ(nanousd_attrib_colorspace(prim, "inputs:diffuseColor", &ok),
                  "srgb_rec709_scene");
    ASSERT(ok == 1);
    ASSERT_STR_EQ(nanousd_attrib_resolved_colorspace(prim, "inputs:diffuseColor"),
                  "srgb_rec709_scene");
    ASSERT(nanousd_clear_attrib_colorspace(prim, "inputs:diffuseColor") == 1);
    ok = 99;
    ASSERT_STR_EQ(nanousd_attrib_colorspace(prim, "inputs:diffuseColor", &ok), "");
    ASSERT(ok == 0);

    ASSERT(nanousd_apply_api(prim, "ColorSpaceAPI") == 1);
    ASSERT(nanousd_create_attrib(prim, "colorSpace:name", "token") == 1);
    ASSERT(nanousd_set_attrib_token(prim, "colorSpace:name", "lin_rec709_scene") == 1);
    ASSERT_STR_EQ(nanousd_prim_resolved_colorspace(prim), "lin_rec709_scene");
    ASSERT_STR_EQ(nanousd_attrib_resolved_colorspace(prim, "inputs:diffuseColor"),
                  "lin_rec709_scene");

    ASSERT(nanousd_set_attrib_colorspace(NULL, "x", "lin_rec709_scene") == 0);
    ASSERT(nanousd_set_attrib_colorspace(prim, NULL, "lin_rec709_scene") == 0);
    ASSERT(nanousd_set_attrib_colorspace(prim, "inputs:diffuseColor", NULL) == 0);
    ASSERT(nanousd_clear_attrib_colorspace(NULL, "x") == 0);
    ASSERT_STR_EQ(nanousd_prim_resolved_colorspace(NULL), "");
    ASSERT_STR_EQ(nanousd_attrib_resolved_colorspace(NULL, "x"), "");

    nanousd_freeprim(prim);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Prim Metadata
 * ============================================================ */

static void test_prim_metadata(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);
    NanousdPrim prim = nanousd_define_prim(stage, "/World", "Xform");
    ASSERT(prim != NULL);

    /* Set kind (token-typed) */
    ASSERT(nanousd_set_prim_metadata_token(prim, "kind", "assembly") == 1);
    int ok = 0;
    const char* kind = nanousd_prim_metadatas(prim, "kind", &ok);
    ASSERT(ok == 1);
    ASSERT(strcmp(kind, "assembly") == 0);

    /* Verify nanousd_kind returns the same */
    ASSERT(strcmp(nanousd_kind(prim), "assembly") == 0);

    /* Set documentation (string-typed) */
    ASSERT(nanousd_set_prim_metadatas(prim, "documentation", "test doc") == 1);
    ok = 0;
    const char* doc = nanousd_prim_metadatas(prim, "documentation", &ok);
    ASSERT(ok == 1);
    ASSERT(strcmp(doc, "test doc") == 0);

    /* Set and read double metadata */
    ASSERT(nanousd_set_prim_metadatad(prim, "hidden", 1.0) == 1);
    ok = 0;
    double hid = nanousd_prim_metadatad(prim, "hidden", &ok);
    ASSERT(ok == 1);
    ASSERT(hid == 1.0);

    /* Unset field returns ok=0 */
    ok = 99;
    nanousd_prim_metadatas(prim, "nonexistent", &ok);
    ASSERT(ok == 0);

    nanousd_freeprim(prim);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_prim_metadata_null(void) {
    ASSERT(nanousd_set_prim_metadatas(NULL, "kind", "x") == 0);
    ASSERT(nanousd_set_prim_metadata_token(NULL, "kind", "x") == 0);
    ASSERT(nanousd_set_prim_metadatad(NULL, "kind", 1.0) == 0);
    TEST_PASS();
}

/* ============================================================
 * Bulk array access (zero-copy pointer API)
 * ============================================================ */

static void test_bulk_array_float(void) {
    NanousdStage stage = nanousd_open(usda_path("bulk_and_types.usda"));
    ASSERT(stage != NULL);
    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    int count = 0;
    const float* data = nanousd_arraydataf(root, "floatArr", &count);
    ASSERT(data != NULL);
    ASSERT(count == 5);
    ASSERT_FLOAT_EQ(data[0], 1.0f, 1e-6);
    ASSERT_FLOAT_EQ(data[4], 5.0f, 1e-6);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_bulk_array_double(void) {
    NanousdStage stage = nanousd_open(usda_path("bulk_and_types.usda"));
    ASSERT(stage != NULL);
    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    int count = 0;
    const double* data = nanousd_arraydatad(root, "doubleArr", &count);
    ASSERT(data != NULL);
    ASSERT(count == 3);
    ASSERT_FLOAT_EQ(data[0], 10.0, 1e-9);
    ASSERT_FLOAT_EQ(data[2], 30.0, 1e-9);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_bulk_array_int(void) {
    NanousdStage stage = nanousd_open(usda_path("bulk_and_types.usda"));
    ASSERT(stage != NULL);
    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    int count = 0;
    const int* data = nanousd_arraydatai(root, "intArr", &count);
    ASSERT(data != NULL);
    ASSERT(count == 3);
    ASSERT(data[0] == 100);
    ASSERT(data[2] == 300);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_bulk_array_null(void) {
    int count = 99;
    ASSERT(nanousd_arraydataf(NULL, "x", &count) == NULL);
    ASSERT(nanousd_arraydatad(NULL, "x", &count) == NULL);
    ASSERT(nanousd_arraydatai(NULL, "x", &count) == NULL);
    TEST_PASS();
}

/* ============================================================
 * Vec array read/write
 * ============================================================ */

static void test_vec3f_array_read(void) {
    NanousdStage stage = nanousd_open(usda_path("bulk_and_types.usda"));
    ASSERT(stage != NULL);
    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    float buf[9];
    int n = nanousd_attribarrayv3f(root, "points", buf, 3);
    ASSERT(n == 3);
    ASSERT_FLOAT_EQ(buf[0], 1.0f, 1e-6);
    ASSERT_FLOAT_EQ(buf[1], 2.0f, 1e-6);
    ASSERT_FLOAT_EQ(buf[2], 3.0f, 1e-6);
    ASSERT_FLOAT_EQ(buf[6], 7.0f, 1e-6);
    ASSERT_FLOAT_EQ(buf[7], 8.0f, 1e-6);
    ASSERT_FLOAT_EQ(buf[8], 9.0f, 1e-6);

    /* Partial read */
    float small[6];
    n = nanousd_attribarrayv3f(root, "points", small, 2);
    ASSERT(n == 2);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_vec3d_array_read(void) {
    NanousdStage stage = nanousd_open(usda_path("bulk_and_types.usda"));
    ASSERT(stage != NULL);
    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    double buf[6];
    int n = nanousd_attribarrayv3d(root, "pointsD", buf, 2);
    ASSERT(n == 2);
    ASSERT_FLOAT_EQ(buf[0], 10.0, 1e-9);
    ASSERT_FLOAT_EQ(buf[3], 40.0, 1e-9);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_vec3f_array_write_roundtrip(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);
    NanousdPrim prim = nanousd_define_prim(stage, "/Test", "Mesh");
    ASSERT(prim != NULL);

    ASSERT(nanousd_create_attrib(prim, "pts", "point3f[]") == 1);
    float pts[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    ASSERT(nanousd_set_attribarrayv3f(prim, "pts", pts, 2) == 1);

    float readback[6];
    int n = nanousd_attribarrayv3f(prim, "pts", readback, 2);
    ASSERT(n == 2);
    ASSERT_FLOAT_EQ(readback[0], 1.0f, 1e-6);
    ASSERT_FLOAT_EQ(readback[5], 6.0f, 1e-6);

    nanousd_freeprim(prim);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Quaternion attributes
 * ============================================================ */

static void test_quatf_read(void) {
    NanousdStage stage = nanousd_open(usda_path("bulk_and_types.usda"));
    ASSERT(stage != NULL);
    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    float q[4];
    ASSERT(nanousd_attribqf(root, "orient", q) == 1);
    ASSERT_FLOAT_EQ(q[0], 1.0f, 1e-6);
    ASSERT_FLOAT_EQ(q[1], 0.0f, 1e-6);
    ASSERT_FLOAT_EQ(q[2], 0.0f, 1e-6);
    ASSERT_FLOAT_EQ(q[3], 0.0f, 1e-6);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_quatd_read(void) {
    NanousdStage stage = nanousd_open(usda_path("bulk_and_types.usda"));
    ASSERT(stage != NULL);
    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    double q[4];
    ASSERT(nanousd_attribqd(root, "orientD", q) == 1);
    ASSERT_FLOAT_EQ(q[0], 0.707, 1e-3);
    ASSERT_FLOAT_EQ(q[2], 0.707, 1e-3);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_quat_write_roundtrip(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);
    NanousdPrim prim = nanousd_define_prim(stage, "/Test", "Xform");
    ASSERT(prim != NULL);

    ASSERT(nanousd_create_attrib(prim, "q", "quatf") == 1);
    float qw[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    ASSERT(nanousd_set_attribqf(prim, "q", qw) == 1);

    float qr[4];
    ASSERT(nanousd_attribqf(prim, "q", qr) == 1);
    ASSERT_FLOAT_EQ(qr[0], 0.5f, 1e-6);
    ASSERT_FLOAT_EQ(qr[3], 0.5f, 1e-6);

    ASSERT(nanousd_create_attrib(prim, "qd", "quatd") == 1);
    double qdw[4] = {1.0, 0.0, 0.0, 0.0};
    ASSERT(nanousd_set_attribqd(prim, "qd", qdw) == 1);

    double qdr[4];
    ASSERT(nanousd_attribqd(prim, "qd", qdr) == 1);
    ASSERT_FLOAT_EQ(qdr[0], 1.0, 1e-9);

    nanousd_freeprim(prim);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Matrix3d attributes
 * ============================================================ */

static void test_matrix3d_read(void) {
    NanousdStage stage = nanousd_open(usda_path("bulk_and_types.usda"));
    ASSERT(stage != NULL);
    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    double m[9];
    ASSERT(nanousd_attribm3d(root, "inertia", m) == 1);
    /* Diagonal: 1, 2, 3 */
    ASSERT_FLOAT_EQ(m[0], 1.0, 1e-9);
    ASSERT_FLOAT_EQ(m[4], 2.0, 1e-9);
    ASSERT_FLOAT_EQ(m[8], 3.0, 1e-9);
    /* Off-diagonal should be 0 */
    ASSERT_FLOAT_EQ(m[1], 0.0, 1e-9);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_matrix3d_write_roundtrip(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);
    NanousdPrim prim = nanousd_define_prim(stage, "/Test", "Xform");
    ASSERT(prim != NULL);

    ASSERT(nanousd_create_attrib(prim, "mat", "matrix3d") == 1);
    double mw[9] = {1, 0, 0, 0, 2, 0, 0, 0, 3};
    ASSERT(nanousd_set_attribm3d(prim, "mat", mw) == 1);

    double mr[9];
    ASSERT(nanousd_attribm3d(prim, "mat", mr) == 1);
    ASSERT_FLOAT_EQ(mr[0], 1.0, 1e-9);
    ASSERT_FLOAT_EQ(mr[4], 2.0, 1e-9);
    ASSERT_FLOAT_EQ(mr[8], 3.0, 1e-9);

    nanousd_freeprim(prim);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * String array read/write
 * ============================================================ */

static void test_string_array_read(void) {
    NanousdStage stage = nanousd_open(usda_path("bulk_and_types.usda"));
    ASSERT(stage != NULL);
    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    int len = nanousd_attribarrays_len(root, "tags");
    ASSERT(len == 3);

    const char* s0 = nanousd_attribarrays(root, "tags", 0);
    ASSERT(s0 != NULL);
    ASSERT_STR_EQ(s0, "alpha");

    const char* s2 = nanousd_attribarrays(root, "tags", 2);
    ASSERT(s2 != NULL);
    ASSERT_STR_EQ(s2, "gamma");

    /* Out of bounds */
    ASSERT(nanousd_attribarrays(root, "tags", 3) == NULL);
    ASSERT(nanousd_attribarrays(root, "tags", -1) == NULL);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_string_array_write_roundtrip(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);
    NanousdPrim prim = nanousd_define_prim(stage, "/Test", "Scope");
    ASSERT(prim != NULL);

    ASSERT(nanousd_create_attrib(prim, "names", "string[]") == 1);
    const char* vals[3] = {"foo", "bar", "baz"};
    ASSERT(nanousd_set_attribarrays(prim, "names", vals, 3) == 1);

    ASSERT(nanousd_attribarrays_len(prim, "names") == 3);
    ASSERT_STR_EQ(nanousd_attribarrays(prim, "names", 0), "foo");
    ASSERT_STR_EQ(nanousd_attribarrays(prim, "names", 1), "bar");
    ASSERT_STR_EQ(nanousd_attribarrays(prim, "names", 2), "baz");

    nanousd_freeprim(prim);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * kilogramsPerUnit metadata
 * ============================================================ */

static void test_kilogramsperunit(void) {
    NanousdStage stage = nanousd_open(usda_path("bulk_and_types.usda"));
    ASSERT(stage != NULL);

    int ok = 0;
    double kpu = nanousd_metadatad(stage, "kilogramsPerUnit", &ok);
    ASSERT(ok == 1);
    ASSERT_FLOAT_EQ(kpu, 0.5, 1e-9);

    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Instancing (spec Section 11.3)
 * ============================================================ */

static void test_isinstance(void) {
    NanousdStage stage = nanousd_open("tests/composition/instancing_root.usda");
    ASSERT(nanousd_isvalid(stage));

    NanousdPrim instA = nanousd_primpath(stage, "/InstanceA");
    NanousdPrim instB = nanousd_primpath(stage, "/InstanceB");
    NanousdPrim nonInst = nanousd_primpath(stage, "/NonInstance");
    NanousdPrim noArc = nanousd_primpath(stage, "/InstanceableNoArc");

    ASSERT(instA != NULL);
    ASSERT(instB != NULL);
    ASSERT(nonInst != NULL);
    ASSERT(noArc != NULL);

    /* InstanceA and InstanceB are instances */
    ASSERT(nanousd_isinstance(instA) == 1);
    ASSERT(nanousd_isinstance(instB) == 1);

    /* NonInstance is not an instance */
    ASSERT(nanousd_isinstance(nonInst) == 0);

    /* Instanceable without arc is not an instance */
    ASSERT(nanousd_isinstanceable(noArc) == 1);
    ASSERT(nanousd_isinstance(noArc) == 0);

    nanousd_freeprim(instA);
    nanousd_freeprim(instB);
    nanousd_freeprim(nonInst);
    nanousd_freeprim(noArc);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_prototype_nav(void) {
    NanousdStage stage = nanousd_open("tests/composition/instancing_root.usda");
    ASSERT(nanousd_isvalid(stage));

    NanousdPrim instA = nanousd_primpath(stage, "/InstanceA");
    ASSERT(instA != NULL);
    ASSERT(nanousd_isinstance(instA) == 1);

    /* Get prototype */
    NanousdPrim proto = nanousd_prototype(instA);
    ASSERT(proto != NULL);
    ASSERT(nanousd_isprototype(proto) == 1);
    ASSERT(nanousd_isinprototype(proto) == 1);

    /* Prototype has instances */
    int n = nanousd_ninstances(proto);
    ASSERT(n == 2);

    /* Get first instance */
    NanousdPrim inst0 = nanousd_instance(proto, 0);
    ASSERT(inst0 != NULL);
    ASSERT(nanousd_isinstance(inst0) == 1);

    /* Null safety */
    ASSERT(nanousd_isinstance(NULL) == 0);
    ASSERT(nanousd_isprototype(NULL) == 0);
    ASSERT(nanousd_isinprototype(NULL) == 0);
    ASSERT(nanousd_prototype(NULL) == NULL);
    ASSERT(nanousd_ninstances(NULL) == 0);
    ASSERT(nanousd_instance(NULL, 0) == NULL);

    nanousd_freeprim(inst0);
    nanousd_freeprim(proto);
    nanousd_freeprim(instA);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_semantic_traversal_and_composition_queries(void) {
    NanousdStage stage = nanousd_open("tests/composition/instancing_root.usda");
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    int nflat = nanousd_traverse_flat(stage, NULL, 0);
    ASSERT(nflat > nanousd_nprims(stage));
    ASSERT(nflat > 0);
    NanousdFlatPrim* flat = (NanousdFlatPrim*)calloc((size_t)nflat, sizeof(NanousdFlatPrim));
    ASSERT(flat != NULL);
    ASSERT(nanousd_traverse_flat(stage, flat, nflat) == nflat);

    int sawInstA = 0;
    int sawInstB = 0;
    int sawProxyChild = 0;
    int sawProxyChildB = 0;
    for (int i = 0; i < nflat; ++i) {
        ASSERT(flat[i].struct_size == (int)sizeof(NanousdFlatPrim));
        ASSERT(flat[i].path != NULL);
        ASSERT(flat[i].type_name != NULL);
        if (strcmp(flat[i].path, "/InstanceA") == 0) {
            sawInstA = 1;
            ASSERT((flat[i].flags & NANOUSD_FLAT_PRIM_INSTANCE) != 0);
            ASSERT((flat[i].flags & NANOUSD_FLAT_PRIM_INSTANCE_PROXY) == 0);
        }
        if (strcmp(flat[i].path, "/InstanceB") == 0) {
            sawInstB = 1;
            ASSERT((flat[i].flags & NANOUSD_FLAT_PRIM_INSTANCE) != 0);
        }
        if (strcmp(flat[i].path, "/InstanceA/Child1") == 0) {
            sawProxyChild = 1;
            ASSERT((flat[i].flags & NANOUSD_FLAT_PRIM_INSTANCE) == 0);
            ASSERT((flat[i].flags & NANOUSD_FLAT_PRIM_INSTANCE_PROXY) != 0);
            ASSERT(flat[i].parent_index >= 0 && flat[i].parent_index < nflat);
            ASSERT_STR_EQ(flat[flat[i].parent_index].path, "/InstanceA");
        }
        if (strcmp(flat[i].path, "/InstanceB/Child1") == 0) {
            sawProxyChildB = 1;
            ASSERT((flat[i].flags & NANOUSD_FLAT_PRIM_INSTANCE_PROXY) != 0);
            ASSERT(flat[i].parent_index >= 0 && flat[i].parent_index < nflat);
            ASSERT_STR_EQ(flat[flat[i].parent_index].path, "/InstanceB");
        }
    }
    free(flat);
    ASSERT(sawInstA && sawInstB);
    ASSERT(sawProxyChild);
    ASSERT(sawProxyChildB);

    ASSERT(nanousd_stage_nprototypes(stage) == 2);
    NanousdPrim stageProto = nanousd_stage_prototype(stage, 0);
    ASSERT(stageProto != NULL);
    ASSERT(nanousd_isprototype(stageProto) == 1);

    NanousdPrim instA = nanousd_primpath(stage, "/InstanceA");
    NanousdPrim instB = nanousd_primpath(stage, "/InstanceB");
    NanousdPrim instC = nanousd_primpath(stage, "/InstanceC");
    ASSERT(instA != NULL && instB != NULL && instC != NULL);
    ASSERT(nanousd_isinstanceproxy(instA) == 0);

    NanousdPrim proxyChild = nanousd_primpath(stage, "/InstanceA/Child1");
    ASSERT(proxyChild != NULL);
    ASSERT_STR_EQ(nanousd_path(proxyChild), "/InstanceA/Child1");
    ASSERT(nanousd_isinstanceproxy(proxyChild) == 1);
    ASSERT(nanousd_isinprototype(proxyChild) == 0);
    NanousdPrim protoChild = nanousd_priminprototype(proxyChild);
    ASSERT(protoChild != NULL);
    ASSERT(nanousd_isinprototype(protoChild) == 1);
    ASSERT(strstr(nanousd_path(protoChild), "/Child1") != NULL);

    NanousdPrim proxyChildB = nanousd_primpath(stage, "/InstanceB/Child1");
    ASSERT(proxyChildB != NULL);
    ASSERT_STR_EQ(nanousd_path(proxyChildB), "/InstanceB/Child1");
    ASSERT(nanousd_isinstanceproxy(proxyChildB) == 1);
    ASSERT(nanousd_isinprototype(proxyChildB) == 0);

    NanousdPrim childFromInstance = nanousd_childname(instB, "Child1");
    ASSERT(childFromInstance != NULL);
    ASSERT_STR_EQ(nanousd_path(childFromInstance), "/InstanceB/Child1");
    ASSERT(nanousd_isinstanceproxy(childFromInstance) == 1);

    NanousdPrim parentFromProxy = nanousd_parent(childFromInstance);
    ASSERT(parentFromProxy != NULL);
    ASSERT_STR_EQ(nanousd_path(parentFromProxy), "/InstanceB");

    char keyA[1024];
    char keyB[1024];
    char keyC[1024];
    int keyALen = nanousd_instance_key(instA, keyA, sizeof(keyA));
    int keyBLen = nanousd_instance_key(instB, keyB, sizeof(keyB));
    int keyCLen = nanousd_instance_key(instC, keyC, sizeof(keyC));
    ASSERT(keyALen > 0);
    ASSERT(keyBLen == keyALen);
    ASSERT(keyCLen > 0);
    ASSERT(strcmp(keyA, keyB) == 0);
    ASSERT(strcmp(keyA, keyC) != 0);
    ASSERT(strstr(keyA, "instancing_ref.usda") != NULL);

    int narc = nanousd_ncomposition_arcs(instA);
    ASSERT(narc > 0);
    int sawReferenceArc = 0;
    for (int i = 0; i < narc; ++i) {
        NanousdCompositionArc arc;
        memset(&arc, 0, sizeof(arc));
        ASSERT(nanousd_composition_arc(instA, i, &arc) == 1);
        ASSERT(arc.struct_size == (int)sizeof(NanousdCompositionArc));
        if (arc.arc_type == NANOUSD_ARC_REFERENCE) {
            sawReferenceArc = 1;
            ASSERT((arc.flags & NANOUSD_COMPOSITION_ARC_DIRECT) != 0);
            ASSERT((arc.flags & NANOUSD_COMPOSITION_ARC_HAS_SOURCE_SPEC) != 0);
            ASSERT(arc.layer_path && strstr(arc.layer_path, "instancing_ref.usda") != NULL);
            ASSERT_STR_EQ(arc.source_path, "/Template");
            ASSERT_STR_EQ(arc.target_path, "/InstanceA");
        }
    }
    ASSERT(sawReferenceArc);

    NanousdPrim protoFromRoot = nanousd_priminprototype(instA);
    ASSERT(protoFromRoot != NULL);
    ASSERT(nanousd_isprototype(protoFromRoot) == 1);

    ASSERT(nanousd_stage_prototype(stage, -1) == NULL);
    ASSERT(nanousd_stage_prototype(stage, 9999) == NULL);
    ASSERT(nanousd_isinstanceproxy(NULL) == 0);
    ASSERT(nanousd_priminprototype(NULL) == NULL);
    ASSERT(nanousd_instance_key(NULL, keyA, sizeof(keyA)) == 0);
    ASSERT(nanousd_ncomposition_arcs(NULL) == 0);
    ASSERT(nanousd_composition_arc(NULL, 0, NULL) == 0);

    nanousd_freeprim(parentFromProxy);
    nanousd_freeprim(childFromInstance);
    nanousd_freeprim(proxyChildB);
    nanousd_freeprim(protoChild);
    nanousd_freeprim(proxyChild);
    nanousd_freeprim(protoFromRoot);
    nanousd_freeprim(instC);
    nanousd_freeprim(instB);
    nanousd_freeprim(instA);
    nanousd_freeprim(stageProto);
    nanousd_close(stage);
    TEST_PASS();
}

/* Variant composition (spec §11.2): selecting a variant from a set should
 * compose that variant's body into the prim's namespace — both local
 * opinions (attribute defaults authored under the variant) and descendant
 * prims defined inside the variant body. */
static void test_variant_basic_selection(void) {
    NanousdStage stage = nanousd_open("tests/composition/variant_basic.usda");
    ASSERT(nanousd_isvalid(stage));

    NanousdPrim model = nanousd_primpath(stage, "/Model");
    ASSERT(model != NULL);

    /* "baseline" is a local authored opinion — unaffected by variant. */
    int ok = 0;
    int baseline = nanousd_attribi(model, "baseline", &ok);
    ASSERT(ok == 1);
    ASSERT(baseline == 1);

    /* "fromVariant" comes from the selected variant body. The "high"
     * variant sets it to 100; the "low" variant sets it to 10. The file
     * selects "high" via variants = { string lod = "high" }. */
    ok = 0;
    int fromVariant = nanousd_attribi(model, "fromVariant", &ok);
    ASSERT(ok == 1);
    ASSERT(fromVariant == 100);

    /* The "high" variant body defines /Model/HighChild, which should now
     * appear as a child of /Model in the composed stage. */
    NanousdPrim highChild = nanousd_childname(model, "HighChild");
    ASSERT(highChild != NULL);

    ok = 0;
    int value = nanousd_attribi(highChild, "value", &ok);
    ASSERT(ok == 1);
    ASSERT(value == 7);

    nanousd_freeprim(highChild);
    nanousd_freeprim(model);
    nanousd_close(stage);
    TEST_PASS();
}

/* Port of OpenUSD PCP fixture testPcpMuseum_BasicNestedVariants — a variant
 * body that itself declares another variant set. Selecting the outer set's
 * variant must expand its body into the prim's namespace, after which the
 * work queue re-visits the newly-introduced descendant prim and resolves
 * ITS variant selection, producing a further descendant from the inner
 * variant's body.
 *
 * For /Foo (outer "which" = "A", inner "count" = "one"): the stage should
 * contain /Foo/A (from the outer variant body), /Foo/A/Number (declared
 * under "A"), and /Foo/A/Number/one (from Number's inner "count" = "one"
 * variant body). The unselected alternatives (/Foo/B, /Foo/A/Number/two)
 * should not appear. */
static void test_variant_nested_expansion(void) {
    NanousdStage stage = nanousd_open(
        "tests/composition/pcp_BasicNestedVariants.usda");
    ASSERT(nanousd_isvalid(stage));

    /* Outer variant "which" = "A" picks the /Foo/A body. */
    NanousdPrim foo = nanousd_primpath(stage, "/Foo");
    ASSERT(foo != NULL);
    NanousdPrim fooA = nanousd_childname(foo, "A");
    ASSERT(fooA != NULL);

    /* The unselected "B" variant must not have materialized its body. */
    NanousdPrim fooB = nanousd_childname(foo, "B");
    ASSERT(fooB == NULL);

    /* /Foo/A/Number is declared inside the "A" variant body and must
     * appear in the stage, along with its inner-variant expansion. */
    NanousdPrim number = nanousd_childname(fooA, "Number");
    ASSERT(number != NULL);

    NanousdPrim one = nanousd_childname(number, "one");
    ASSERT(one != NULL);

    /* The unselected "count" = "two" branch must not appear. */
    NanousdPrim two = nanousd_childname(number, "two");
    ASSERT(two == NULL);

    nanousd_freeprim(two);  /* NULL-safe */
    nanousd_freeprim(one);
    nanousd_freeprim(number);
    nanousd_freeprim(fooB);
    nanousd_freeprim(fooA);
    nanousd_freeprim(foo);
    nanousd_close(stage);
    TEST_PASS();
}

/* Variant body carrying a composition arc: selecting a variant that has
 * `references = @other.usda@` inside its body must chain — the variant
 * expands first, adding its reference opinion, then resolve picks up that
 * newly-attached reference and brings in the target's content. Spec §11.2
 * + §10.5 (LIVRPS) require variants to compose first so their opinions
 * are visible to downstream reference / payload resolution. */
static void test_variant_body_chained_reference(void) {
    NanousdStage stage = nanousd_open("tests/composition/variant_with_ref.usda");
    ASSERT(nanousd_isvalid(stage));

    NanousdPrim model = nanousd_primpath(stage, "/Model");
    ASSERT(model != NULL);

    /* The "high" variant body references variant_ref_target.usda, whose
     * defaultPrim contributes fromRef = 42 and a RefChild subprim. */
    int ok = 0;
    int fromRef = nanousd_attribi(model, "fromRef", &ok);
    ASSERT(ok == 1);
    ASSERT(fromRef == 42);

    NanousdPrim refChild = nanousd_childname(model, "RefChild");
    ASSERT(refChild != NULL);

    ok = 0;
    int refValue = nanousd_attribi(refChild, "refValue", &ok);
    ASSERT(ok == 1);
    ASSERT(refValue == 99);

    nanousd_freeprim(refChild);
    nanousd_freeprim(model);
    nanousd_close(stage);
    TEST_PASS();
}

/* Port of OpenUSD PCP fixture testPcpMuseum_BasicPayloadDiamond — /Root has
 * payloads to both A and B; A and B each payload C. Per spec §10.4 this
 * configuration is legal (no single chain of includes visits the same
 * layer stack twice), so composing /Root must:
 *   - succeed without producing a cycle diagnostic for the C payload,
 *   - make A's and B's local attributes visible on /Root (their payload
 *     targets), along with C's attributes chained through.
 */
static void test_payload_diamond_no_cycle(void) {
    NanousdStage stage = nanousd_open(
        "tests/composition/pcp_BasicPayloadDiamond_root.usda");
    ASSERT(nanousd_isvalid(stage));

    /* No payload-related error diagnostics should have been emitted —
     * in particular the diamond must not be flagged as a cycle. The
     * backend does not raise a cycle-category diagnostic, so here we
     * assert only that no payload diagnostic rose to Error severity. */
    int ndiag = 0;
    NanousdDiagnostic* diags = nanousd_diagnostics(stage, &ndiag);
    for (int i = 0; i < ndiag; i++) {
        /* Payload-related diagnostics are category 5 (MissingPayload) or
         * 6 (PayloadParseFail); neither should be Error (severity 2). */
        if (diags[i].category == 5 || diags[i].category == 6) {
            ASSERT(diags[i].severity < 2);
        }
    }
    nanousd_free_diagnostics(diags, ndiag);

    /* The payload list `[A, B]` brings in both targets; A contributes
     * A_attr, B contributes B_attr, and both transitively payload C
     * which contributes C_attr. All three should be visible as opinions
     * on /Root since payloads compose their target's opinions into the
     * including prim's namespace (spec §10.4). */
    NanousdPrim root = nanousd_primpath(stage, "/Root");
    ASSERT(root != NULL);

    ASSERT(nanousd_hasattrib(root, "A_attr") == 1);
    ASSERT(nanousd_hasattrib(root, "B_attr") == 1);
    ASSERT(nanousd_hasattrib(root, "C_attr") == 1);

    nanousd_freeprim(root);
    nanousd_close(stage);
    TEST_PASS();
}

/* Port of OpenUSD PCP fixture testPcpMuseum_BasicNestedPayload — exercises
 * three LIVRPS scenarios when payloads are nested within namespace.
 *
 * Scenario 1 — /Set: references set.usda, which payloads set_payload.usda.
 * set_payload declares /Set/Prop (which in turn references prop.usda,
 * whose payload prop_payload sets x = "from prop_payload") and also
 * *directly* overrides /Set/Prop/PropScope.x = "from set_payload". The
 * ancestor-arc opinion ("from set_payload") is authored on the upstream
 * payload arc, so it wins over the transitive reference->payload chain:
 * composed x is "from set_payload".
 *
 * Scenario 2 — /Set2: payloads set_payload.usda directly, AND /Set2/Prop
 * additionally payloads prop_payload.usda on its own. Both opinions
 * reach /Set2/Prop/PropScope.x through Payload arcs from the same layer
 * stack, so spec §10.5's namespace-depth tiebreak makes the descendant
 * /Set2/Prop payload stronger: composed x is "from prop_payload".
 *
 * Scenario 3 — /Set3: references set.usda, and locally overrides
 * /Set3/Prop/PropScope.x = "from root". Local opinions are stronger
 * than any payload-introduced opinion per LIVRPS, so x must be
 * "from root".
 */
static void test_payload_nested_livrps(void) {
    NanousdStage stage = nanousd_open(
        "tests/composition/pcp_BasicNestedPayload_root.usda");
    ASSERT(nanousd_isvalid(stage));

    /* Scenario 1: direct set_payload opinion wins over transitive
     * prop→prop_payload chain. */
    NanousdPrim s1 = nanousd_primpath(stage, "/Set/Prop/PropScope");
    ASSERT(s1 != NULL);
    int ok = 0;
    const char* x1 = nanousd_attribs(s1, "x", &ok);
    ASSERT(ok == 1);
    ASSERT(x1 != NULL);
    ASSERT(strcmp(x1, "from set_payload") == 0);
    nanousd_freeprim(s1);

    /* Scenario 2: the descendant /Set2/Prop payload is authored deeper
     * in namespace than /Set2's ancestor payload, so its opinion wins
     * among same-type Payload arcs. */
    NanousdPrim s2 = nanousd_primpath(stage, "/Set2/Prop/PropScope");
    ASSERT(s2 != NULL);
    ok = 0;
    const char* x2 = nanousd_attribs(s2, "x", &ok);
    ASSERT(ok == 1);
    ASSERT(x2 != NULL);
    ASSERT(strcmp(x2, "from prop_payload") == 0);
    nanousd_freeprim(s2);

    /* Scenario 3: local over on /Set3/Prop/PropScope wins over any
     * payload-introduced opinion. */
    NanousdPrim s3 = nanousd_primpath(stage, "/Set3/Prop/PropScope");
    ASSERT(s3 != NULL);
    ok = 0;
    const char* x3 = nanousd_attribs(s3, "x", &ok);
    ASSERT(ok == 1);
    ASSERT(x3 != NULL);
    ASSERT(strcmp(x3, "from root") == 0);
    nanousd_freeprim(s3);

    nanousd_close(stage);
    TEST_PASS();
}

/* Nested-reference instancing: instanceable prims authored in a referenced
 * library layer (each declared at a distinct path in that layer) must still
 * share a prototype when they all point at the same target asset. The
 * per-instance authoring in the library layer is a local override — part
 * of the parent's arc, not the instance's own composition — so it must
 * not differentiate prototypes. */
static void test_nested_ref_instancing_shares_prototype(void) {
    NanousdStage stage = nanousd_open("tests/composition/instancing_nested_root.usda");
    ASSERT(nanousd_isvalid(stage));

    NanousdPrim c1 = nanousd_primpath(stage, "/World/Env/Cone_01");
    NanousdPrim c2 = nanousd_primpath(stage, "/World/Env/Cone_02");
    NanousdPrim c3 = nanousd_primpath(stage, "/World/Env/Cone_03");
    ASSERT(c1 != NULL);
    ASSERT(c2 != NULL);
    ASSERT(c3 != NULL);

    ASSERT(nanousd_isinstance(c1) == 1);
    ASSERT(nanousd_isinstance(c2) == 1);
    ASSERT(nanousd_isinstance(c3) == 1);

    NanousdPrim p1 = nanousd_prototype(c1);
    NanousdPrim p2 = nanousd_prototype(c2);
    NanousdPrim p3 = nanousd_prototype(c3);
    ASSERT(p1 != NULL);
    ASSERT(p2 != NULL);
    ASSERT(p3 != NULL);

    /* All three must resolve to the same prototype path. */
    const char* pp1 = nanousd_path(p1);
    const char* pp2 = nanousd_path(p2);
    const char* pp3 = nanousd_path(p3);
    ASSERT(pp1 && pp2 && pp3);
    ASSERT(strcmp(pp1, pp2) == 0);
    ASSERT(strcmp(pp1, pp3) == 0);

    /* Exactly three instances under that single prototype. */
    ASSERT(nanousd_ninstances(p1) == 3);

    nanousd_freeprim(p3);
    nanousd_freeprim(p2);
    nanousd_freeprim(p1);
    nanousd_freeprim(c3);
    nanousd_freeprim(c2);
    nanousd_freeprim(c1);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_repeated_instanceable_reference_children(void) {
    NanousdStage stage = nanousd_open("tests/composition/instancing_root.usda");
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim instA = nanousd_primpath(stage, "/InstanceA");
    NanousdPrim instB = nanousd_primpath(stage, "/InstanceB");
    ASSERT(instA != NULL);
    ASSERT(instB != NULL);
    ASSERT(nanousd_isinstance(instA) == 1);
    ASSERT(nanousd_isinstance(instB) == 1);

    NanousdPrim protoA = nanousd_prototype(instA);
    NanousdPrim protoB = nanousd_prototype(instB);
    ASSERT(protoA != NULL);
    ASSERT(protoB != NULL);
    ASSERT(strcmp(nanousd_path(protoA), nanousd_path(protoB)) == 0);

    ASSERT(nanousd_nchildren(instB) == 2);
    NanousdPrim child0 = nanousd_child(instB, 0);
    NanousdPrim child1 = nanousd_child(instB, 1);
    ASSERT(child0 != NULL);
    ASSERT(child1 != NULL);

    int sawChild1 = 0;
    int sawChild2 = 0;
    const char* name0 = nanousd_name(child0);
    const char* name1 = nanousd_name(child1);
    if (name0 && strcmp(name0, "Child1") == 0) sawChild1 = 1;
    if (name0 && strcmp(name0, "Child2") == 0) sawChild2 = 1;
    if (name1 && strcmp(name1, "Child1") == 0) sawChild1 = 1;
    if (name1 && strcmp(name1, "Child2") == 0) sawChild2 = 1;
    ASSERT(sawChild1 && sawChild2);

    nanousd_freeprim(child1);
    nanousd_freeprim(child0);
    nanousd_freeprim(protoB);
    nanousd_freeprim(protoA);
    nanousd_freeprim(instB);
    nanousd_freeprim(instA);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_inactive_represented_instance_pruned(void) {
    NanousdStage stage = nanousd_open(
        "tests/composition/instancing_inactive_representative_root.usda");
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    NanousdPrim instA = nanousd_primpath(stage, "/InstanceA");
    NanousdPrim instB = nanousd_primpath(stage, "/InstanceB");
    ASSERT(instA != NULL);
    ASSERT(nanousd_isinstance(instA) == 1);
    ASSERT(instB == NULL);

    nanousd_freeprim(instA);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Variant API (spec §11.2)
 * ============================================================ */

/* Read-side queries on variant_basic.usda: one variant set "lod" with
 * variants "high" and "low", current selection "high". */
static void test_variant_api_read(void) {
    NanousdStage stage = nanousd_open("tests/composition/variant_basic.usda");
    ASSERT(nanousd_isvalid(stage));

    NanousdPrim model = nanousd_primpath(stage, "/Model");
    ASSERT(model != NULL);

    /* Variant sets */
    ASSERT(nanousd_nvariantsets(model) == 1);
    const char* setName = nanousd_variantsetname(model, 0);
    ASSERT(setName != NULL);
    ASSERT(strcmp(setName, "lod") == 0);
    ASSERT(nanousd_hasvariantset(model, "lod") == 1);
    ASSERT(nanousd_hasvariantset(model, "bogus") == 0);

    /* Out-of-range index returns empty string rather than crashing. */
    const char* oor = nanousd_variantsetname(model, 5);
    ASSERT(oor != NULL);
    ASSERT(oor[0] == '\0');

    /* Variants in set "lod" — both "high" and "low" are declared. Order is
     * authoring-order in the fixture; assert both are present regardless
     * of index assignment. */
    int n = nanousd_nvariants(model, "lod");
    ASSERT(n == 2);
    int sawHigh = 0, sawLow = 0;
    for (int i = 0; i < n; i++) {
        const char* vn = nanousd_variantname(model, "lod", i);
        ASSERT(vn != NULL);
        if (strcmp(vn, "high") == 0) sawHigh = 1;
        else if (strcmp(vn, "low") == 0) sawLow = 1;
    }
    ASSERT(sawHigh && sawLow);

    /* Unknown set returns 0 variants. */
    ASSERT(nanousd_nvariants(model, "bogus") == 0);

    /* Current selection. */
    const char* sel = nanousd_variantselection(model, "lod");
    ASSERT(sel != NULL);
    ASSERT(strcmp(sel, "high") == 0);

    /* Unauthored selection returns empty string. */
    const char* nosel = nanousd_variantselection(model, "bogus");
    ASSERT(nosel != NULL);
    ASSERT(nosel[0] == '\0');

    nanousd_freeprim(model);
    nanousd_close(stage);
    TEST_PASS();
}

/* Authoring: set a new variant selection on the root layer and verify the
 * selection reads back. variant_basic.usda selects "high"; flipping to
 * "low" should remove /Model/HighChild and change fromVariant from 100 to
 * 10. nanousd_setvariantselection self-recomposes internally and refreshes
 * the passed-in handle, so the new selection is visible immediately with no
 * separate recompose call. (Composed-body effects — HighChild disappearing,
 * fromVariant flipping — are covered by TestVariantApiSetAndRecompose in the
 * C++ suite.) */
static void test_variant_api_author(void) {
    NanousdStage stage = nanousd_open("tests/composition/variant_basic.usda");
    ASSERT(nanousd_isvalid(stage));

    NanousdPrim model = nanousd_primpath(stage, "/Model");
    ASSERT(model != NULL);

    /* Initial selection is "high". */
    const char* before = nanousd_variantselection(model, "lod");
    ASSERT(strcmp(before, "high") == 0);

    /* Switch to "low" on layer 0 (the root layer — in-memory override). */
    int ok = nanousd_setvariantselection(model, "lod", "low", 0);
    ASSERT(ok == 1);

    /* Re-query the selection against the same prim handle, which the setter
     * refreshed after its internal recompose. The author wrote "low" into the
     * root layer (layer 0, strongest opinion), so it reads back "low". */
    const char* after = nanousd_variantselection(model, "lod");
    ASSERT(after != NULL);
    ASSERT(strcmp(after, "low") == 0);

    /* Clearing the selection (empty variantName) erases the entry, so
     * GetVariantSelection falls through to any weaker opinion — in this
     * fixture there is none, so the result is "". */
    ok = nanousd_setvariantselection(model, "lod", "", 0);
    ASSERT(ok == 1);
    const char* cleared = nanousd_variantselection(model, "lod");
    ASSERT(cleared != NULL);
    ASSERT(cleared[0] == '\0');

    /* Out-of-range layer is rejected. */
    ok = nanousd_setvariantselection(model, "lod", "low", 999);
    ASSERT(ok == 0);

    nanousd_freeprim(model);
    nanousd_close(stage);
    TEST_PASS();
}

/* Instancing + variants (spec §11.2 + §11.3): two instanceable prims
 * that reference the same asset but select different variants must
 * NOT share a prototype — their composed bodies differ. Two instances
 * with the same reference AND same selection should share.
 *
 * Fixture instancing_variant_root.usda defines:
 *   /InstanceHigh, /InstanceHighAlt — ref target, lod=high
 *   /InstanceLow                    — ref target, lod=low
 */
static void test_instancing_variant_differentiates_prototype(void) {
    NanousdStage stage = nanousd_open(
        "tests/composition/instancing_variant_root.usda");
    ASSERT(nanousd_isvalid(stage));

    NanousdPrim hi  = nanousd_primpath(stage, "/InstanceHigh");
    NanousdPrim hi2 = nanousd_primpath(stage, "/InstanceHighAlt");
    NanousdPrim lo  = nanousd_primpath(stage, "/InstanceLow");
    ASSERT(hi != NULL);
    ASSERT(hi2 != NULL);
    ASSERT(lo != NULL);

    ASSERT(nanousd_isinstance(hi)  == 1);
    ASSERT(nanousd_isinstance(hi2) == 1);
    ASSERT(nanousd_isinstance(lo)  == 1);

    NanousdPrim hiProto  = nanousd_prototype(hi);
    NanousdPrim hi2Proto = nanousd_prototype(hi2);
    NanousdPrim loProto  = nanousd_prototype(lo);
    ASSERT(hiProto  != NULL);
    ASSERT(hi2Proto != NULL);
    ASSERT(loProto  != NULL);

    const char* hiPath  = nanousd_path(hiProto);
    const char* hi2Path = nanousd_path(hi2Proto);
    const char* loPath  = nanousd_path(loProto);
    ASSERT(hiPath && hi2Path && loPath);

    /* Same reference + same selection → same prototype. */
    ASSERT(strcmp(hiPath, hi2Path) == 0);

    /* Same reference + different selection → different prototype. */
    ASSERT(strcmp(hiPath, loPath) != 0);

    /* Cross-check content: the "high" prototype must carry the high-variant
     * body (fromVariant=100, HighChild subprim present); the "low" prototype
     * must carry the low-variant body (fromVariant=10, no HighChild). This
     * guards against a false-positive where the keys happen to differ but
     * both prototypes were populated from the same variant. */
    int ok = 0;
    int hiFromVariant = nanousd_attribi(hiProto, "fromVariant", &ok);
    ASSERT(ok == 1);
    ASSERT(hiFromVariant == 100);
    NanousdPrim hiHighChild = nanousd_childname(hiProto, "HighChild");
    ASSERT(hiHighChild != NULL);
    nanousd_freeprim(hiHighChild);

    ok = 0;
    int loFromVariant = nanousd_attribi(loProto, "fromVariant", &ok);
    ASSERT(ok == 1);
    ASSERT(loFromVariant == 10);
    NanousdPrim loHighChild = nanousd_childname(loProto, "HighChild");
    ASSERT(loHighChild == NULL);

    nanousd_freeprim(loProto);
    nanousd_freeprim(hi2Proto);
    nanousd_freeprim(hiProto);
    nanousd_freeprim(lo);
    nanousd_freeprim(hi2);
    nanousd_freeprim(hi);
    nanousd_close(stage);
    TEST_PASS();
}

/* Port of OpenUSD PCP fixture testPcpMuseum_SubrootReferenceAndVariants —
 * exercises ancestral variant selection on subroot references (spec
 * §11.2.5).
 *
 * Target layer: /Group has variantSelection {v=y}. Variant bodies at
 *   /Group{v=x}/Model (a = "v_x") and /Group{v=y}/Model (a = "v_y").
 *   /Group/Model does not exist at that exact path in the layer.
 *
 * Root layer:
 *   /SubrootRef references @target@</Group/Model> → must fold /Group's
 *     ancestral v=y and pull content from /Group{v=y}/Model, giving
 *     a = "v_y".
 *   /RootRef references @model@</Model> (that layer selects v=x) —
 *     composes modelAttr = "v=x".
 *   /RootRef/Child references @target@</Group/Model> — must use
 *     target.usda's /Group ancestral v=y independently, giving
 *     a = "v_y". The outer /RootRef's v=x does not leak in. */
static void test_ancestral_variant_subroot_reference(void) {
    NanousdStage stage = nanousd_open(
        "tests/composition/pcp_SubrootReferenceAndVariants_root.usda");
    ASSERT(nanousd_isvalid(stage));

    /* /SubrootRef: ancestral v=y on /Group picks /Group{v=y}/Model. */
    NanousdPrim sr = nanousd_primpath(stage, "/SubrootRef");
    ASSERT(sr != NULL);
    int ok = 0;
    const char* srA = nanousd_attribs(sr, "a", &ok);
    ASSERT(ok == 1);
    ASSERT(srA != NULL);
    ASSERT(strcmp(srA, "v_y") == 0);
    nanousd_freeprim(sr);

    /* /RootRef: its own @model@</Model> reference composes v=x, giving
     * modelAttr = "v=x". /Model's variant selection is direct (on the
     * target itself), not ancestral — exercised by ExpandVariantArcs. */
    NanousdPrim rr = nanousd_primpath(stage, "/RootRef");
    ASSERT(rr != NULL);
    ok = 0;
    const char* rrAttr = nanousd_attribs(rr, "modelAttr", &ok);
    ASSERT(ok == 1);
    ASSERT(rrAttr != NULL);
    ASSERT(strcmp(rrAttr, "v=x") == 0);
    nanousd_freeprim(rr);

    /* /RootRef/Child: independent chain — references target.usda's
     * /Group/Model, ancestor /Group picks v=y. a = "v_y" even though
     * /RootRef itself had v=x elsewhere. */
    NanousdPrim child = nanousd_primpath(stage, "/RootRef/Child");
    ASSERT(child != NULL);
    ok = 0;
    const char* childA = nanousd_attribs(child, "a", &ok);
    ASSERT(ok == 1);
    ASSERT(childA != NULL);
    ASSERT(strcmp(childA, "v_y") == 0);

    /* Guard against v=x leaking into /RootRef/Child from the outer
     * /RootRef's independent variant chain. */
    ASSERT(strcmp(childA, "v_x") != 0);
    nanousd_freeprim(child);

    /* Variant selections are not legal in authored reference target
     * paths. The defining prim still exists, but the invalid reference
     * contributes no target opinions and emits a dedicated diagnostic. */
    NanousdPrim invalid =
        nanousd_primpath(stage, "/InvalidSubrootRefWithVariantSelection");
    ASSERT(invalid != NULL);
    ok = 0;
    (void)nanousd_attribs(invalid, "a", &ok);
    ASSERT(ok == 0);
    nanousd_freeprim(invalid);

    int ndiag = 0;
    NanousdDiagnostic* diags = nanousd_diagnostics(stage, &ndiag);
    int saw_invalid_target = 0;
    for (int i = 0; i < ndiag; ++i) {
        if (diags[i].category == 11 &&  /* InvalidReferenceTarget */
            diags[i].arc_type == 1 &&   /* Reference */
            strcmp(diags[i].prim_path,
                   "/InvalidSubrootRefWithVariantSelection") == 0) {
            saw_invalid_target = 1;
            break;
        }
    }
    ASSERT(saw_invalid_target == 1);
    nanousd_free_diagnostics(diags, ndiag);

    nanousd_close(stage);
    TEST_PASS();
}

/* Dictionary metadata must combine across the opinion stack per
 * spec §12.2.5 + §6.6.2.1. Fixture layers:
 *   dict_resolution_root.usda   (stronger, sublayers base)
 *     customData = { fromRoot=true, sharedScalar=999,
 *                    nested = { rootOnly, sharedInNested=42 } }
 *   dict_resolution_base.usda   (weaker)
 *     customData = { fromBase=true, sharedScalar=10,
 *                    nested = { baseOnly, sharedInNested=100 } }
 * Expected combined result:
 *   fromRoot=true     (rule 4: only in root)
 *   fromBase=true     (rule 4: only in base)
 *   sharedScalar=999  (rule 2: scalar conflict, stronger wins)
 *   nested = {        (rule 3: both dicts → recurse)
 *     rootOnly        (rule 4 inside nested)
 *     baseOnly        (rule 4 inside nested)
 *     sharedInNested=42 (rule 2: stronger wins)
 *   }
 *
 * Read via FlattenStage + USDA string so we can string-match on the
 * serialized representation. The C API doesn't expose Dictionary
 * metadata directly, so flatten is the observable surface. */
static void test_metadata_dict_combine(void) {
    NanousdStage stage = nanousd_open(
        "tests/composition/dict_resolution_root.usda");
    ASSERT(nanousd_isvalid(stage));

    const char* usda = nanousd_write_usda_string(stage);
    ASSERT(usda != NULL);

    /* Keys unique to each layer must both appear. */
    ASSERT_MSG(strstr(usda, "fromRoot") != NULL,
               "combined customData must include root-only key");
    ASSERT_MSG(strstr(usda, "fromBase") != NULL,
               "combined customData must include base-only key");

    /* Scalar conflict — stronger (root's 999) wins over weaker (base's 10). */
    ASSERT_MSG(strstr(usda, "999") != NULL,
               "sharedScalar must resolve to root's value (999)");
    ASSERT_MSG(strstr(usda, " 10\n") == NULL && strstr(usda, "= 10\n") == NULL,
               "base's losing sharedScalar value (10) must not appear");

    /* Nested dict recursion. */
    ASSERT_MSG(strstr(usda, "rootOnly") != NULL,
               "nested recursion must keep root's unique key");
    ASSERT_MSG(strstr(usda, "baseOnly") != NULL,
               "nested recursion must keep base's unique key");
    ASSERT_MSG(strstr(usda, "42") != NULL,
               "sharedInNested must resolve to root's value (42)");
    ASSERT_MSG(strstr(usda, "100") == NULL,
               "base's losing sharedInNested value (100) must not appear");

    nanousd_free_string(usda);
    nanousd_close(stage);
    TEST_PASS();
}

/* Spec §12.2.6: relationship targets follow generic listop combining
 * across the opinion stack. Both layers use composable prepends; the
 * resolved target list must contain root's prepends first, then base's,
 * with shared items deduplicated by the listop combine rule. */
static void test_metadata_listop_relationship_targets_combine(void) {
    NanousdStage stage = nanousd_open(
        "tests/composition/listop_resolution_root.usda");
    ASSERT(nanousd_isvalid(stage));

    NanousdPrim source = nanousd_primpath(stage, "/Source");
    ASSERT(source != NULL);

    int n = nanousd_nreltargets(source, "relTargets");
    /* root prepends: StrongProp, SharedProp.
     * base prepends: WeakProp, SharedProp.
     * Combined (composable+composable): unique items in order —
     * StrongProp, SharedProp, WeakProp. */
    ASSERT(n == 3);

    int sawStrong = 0, sawShared = 0, sawWeak = 0;
    int strongIdx = -1, sharedIdx = -1, weakIdx = -1;
    for (int i = 0; i < n; i++) {
        const char* t = nanousd_reltarget(source, "relTargets", i);
        ASSERT(t != NULL);
        if (strstr(t, "StrongProp")) { sawStrong = 1; strongIdx = i; }
        else if (strstr(t, "SharedProp")) { sawShared = 1; sharedIdx = i; }
        else if (strstr(t, "WeakProp")) { sawWeak = 1; weakIdx = i; }
    }
    ASSERT(sawStrong && sawShared && sawWeak);
    /* Stronger prepends precede weaker prepends; shared item appears
     * once at its first position. */
    ASSERT(strongIdx < weakIdx);
    ASSERT(sharedIdx < weakIdx);

    nanousd_freeprim(source);
    nanousd_close(stage);
    TEST_PASS();
}

/* Spec §12.2.6 + §13.2.1.2: apiSchemas is a generic prim-level listop
 * metadata field. Read via the flattened USDA — combined apiSchemas
 * must include schemas from both layers. */
static void test_metadata_listop_apischemas_combine(void) {
    NanousdStage stage = nanousd_open(
        "tests/composition/listop_resolution_root.usda");
    ASSERT(nanousd_isvalid(stage));

    NanousdPrim source = nanousd_primpath(stage, "/Source");
    ASSERT(source != NULL);

    /* GetAppliedSchemas surface for apiSchemas is via the C API's
     * schema query path; here we observe via flatten so it exercises
     * the FlattenStage code path that picks up the registry's
     * isListOp flag. */
    const char* usda = nanousd_write_usda_string(stage);
    ASSERT(usda != NULL);
    ASSERT_MSG(strstr(usda, "RootAPI") != NULL,
               "combined apiSchemas must include root's RootAPI");
    ASSERT_MSG(strstr(usda, "BaseAPI") != NULL,
               "combined apiSchemas must include base's BaseAPI");

    nanousd_free_string(usda);
    nanousd_freeprim(source);
    nanousd_close(stage);
    TEST_PASS();
}

/* Spec §6.6.3.6 + §12.2.6: explicit stronger opinion prunes weaker
 * contributions. Root authors explicit listops; base's prepends must
 * NOT appear in the combined result. */
static void test_metadata_listop_explicit_prunes_weaker(void) {
    NanousdStage stage = nanousd_open(
        "tests/composition/listop_resolution_explicit_root.usda");
    ASSERT(nanousd_isvalid(stage));

    NanousdPrim source = nanousd_primpath(stage, "/Source");
    ASSERT(source != NULL);

    /* Relationship targets: only root's explicit OnlyProp survives.
     * Base's prepends (WeakProp, SharedProp) must be pruned. */
    int n = nanousd_nreltargets(source, "relTargets");
    ASSERT(n == 1);
    const char* t0 = nanousd_reltarget(source, "relTargets", 0);
    ASSERT(t0 != NULL);
    ASSERT_MSG(strstr(t0, "OnlyProp") != NULL,
               "explicit root must prune base's relTargets");

    /* apiSchemas: only root's OnlyRoot survives. */
    const char* usda = nanousd_write_usda_string(stage);
    ASSERT(usda != NULL);
    ASSERT_MSG(strstr(usda, "OnlyRoot") != NULL,
               "explicit apiSchemas must include root's value");
    ASSERT_MSG(strstr(usda, "BaseAPI") == NULL,
               "explicit root must prune base's BaseAPI from apiSchemas");

    nanousd_free_string(usda);
    nanousd_freeprim(source);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Diagnostics
 * ============================================================ */

static void test_diagnostics_missing_sublayer(void) {
    NanousdStage stage = nanousd_open("tests/usda/missing_sublayer.usda");
    ASSERT(stage != NULL);
    ASSERT(nanousd_isvalid(stage) == 1);  /* valid but degraded */

    int count = 0;
    NanousdDiagnostic* diags = nanousd_diagnostics(stage, &count);
    ASSERT(count >= 1);
    ASSERT(diags != NULL);

    /* Check first diagnostic has reasonable fields.
     * DefaultResolve returns a path even for nonexistent files (it doesn't
     * check existence), so the actual failure is a parse error (category 1)
     * rather than a resolution failure (category 0). */
    ASSERT(diags[0].severity == 2);  /* Error */
    ASSERT(diags[0].category == 1);  /* SublayerParseFail */
    ASSERT(diags[0].message != NULL);
    ASSERT(diags[0].message[0] != '\0');
    ASSERT(diags[0].arc_type == 0);  /* sublayer */

    nanousd_free_diagnostics(diags, count);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_diagnostics_json(void) {
    NanousdStage stage = nanousd_open("tests/usda/missing_sublayer.usda");
    ASSERT(stage != NULL);
    ASSERT(nanousd_isvalid(stage) == 1);

    const char* json = nanousd_diagnostics_json(stage);
    ASSERT(json != NULL);
    ASSERT(strnlen(json, 3) > 2);  /* more than "[]" */
    ASSERT(json[0] == '[');

    nanousd_free_string(json);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_diagnostics_clean_stage(void) {
    NanousdStage stage = nanousd_open(usda_path("stage_metadata.usda"));
    ASSERT(stage != NULL);
    ASSERT(nanousd_isvalid(stage) == 1);

    int count = -1;
    NanousdDiagnostic* diags = nanousd_diagnostics(stage, &count);
    ASSERT(count == 0);
    /* diags may be NULL when count is 0 */
    nanousd_free_diagnostics(diags, count);

    const char* json = nanousd_diagnostics_json(stage);
    ASSERT(json != NULL);
    ASSERT_STR_EQ(json, "[]");
    nanousd_free_string(json);

    nanousd_close(stage);
    TEST_PASS();
}

static void test_diagnostics_free_null(void) {
    /* Must not crash */
    nanousd_free_diagnostics(NULL, 0);
    nanousd_free_diagnostics(NULL, 5);
    TEST_PASS();
}

/* ============================================================
 * Per-layer spec / opinion queries (usdview panel parity)
 * ============================================================ */

static void test_layer_has_prim_spec(void) {
    NanousdStage stage = nanousd_open(usda_path("compose_three_sublayers.usda"));
    ASSERT(stage != NULL);
    ASSERT(nanousd_stage_n_layers(stage) >= 1);
    /* Root layer authors /Root */
    ASSERT(nanousd_layer_has_prim_spec(stage, 0, "/Root") == 1);
    /* Bogus path → no spec */
    ASSERT(nanousd_layer_has_prim_spec(stage, 0, "/DoesNotExist") == 0);
    /* Out-of-range layer index → 0 */
    ASSERT(nanousd_layer_has_prim_spec(stage, 9999, "/Root") == 0);
    /* NULL stage / NULL path → 0 */
    ASSERT(nanousd_layer_has_prim_spec(NULL, 0, "/Root") == 0);
    ASSERT(nanousd_layer_has_prim_spec(stage, 0, NULL) == 0);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_layer_has_attr_opinion(void) {
    NanousdStage stage = nanousd_open(usda_path("compose_three_sublayers.usda"));
    ASSERT(stage != NULL);
    /* Root layer authors /Root.source */
    ASSERT(nanousd_layer_has_attr_opinion(stage, 0, "/Root", "source") == 1);
    ASSERT(nanousd_layer_has_attr_opinion(stage, 0, "/Root", "missing") == 0);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_layer_attr_nsamples_no_samples(void) {
    NanousdStage stage = nanousd_open(usda_path("compose_three_sublayers.usda"));
    ASSERT(stage != NULL);
    /* Default-only attribute → 0 samples */
    ASSERT(nanousd_layer_attr_nsamples(stage, 0, "/Root", "source") == 0);
    nanousd_close(stage);
    TEST_PASS();
}

/* Regression: be_layer_attr_nsamples read the timeSamples field as a
 * TimeSamples value, but timeSamples are stored as a Dictionary, so it
 * returned 0 for every animated attribute. /Anim.height authors three
 * samples in the root layer. */
static void test_layer_attr_nsamples_authored(void) {
    NanousdStage stage = nanousd_open(usda_path("rel_metadata_and_samples.usda"));
    ASSERT(stage != NULL);
    ASSERT(nanousd_layer_attr_nsamples(stage, 0, "/Anim", "height") == 3);
    nanousd_close(stage);
    TEST_PASS();
}

/* Regression: nanousd_rel_metadatas gated its return on (ok && *ok), so
 * passing ok == NULL (a documented, optional out-param) silently discarded
 * a found value and returned "". Verify both ok-NULL and ok-non-NULL paths
 * return the authored relationship metadatum. */
static void test_rel_metadatas_null_ok(void) {
    NanousdStage stage = nanousd_open(usda_path("rel_metadata_and_samples.usda"));
    ASSERT(stage != NULL);
    NanousdPrim mesh = nanousd_primpath(stage, "/Mesh");
    ASSERT(mesh != NULL);

    /* Sanity: a non-NULL ok reports found and returns the authored token. */
    int ok = -1;
    const char* v = nanousd_rel_metadatas(mesh, "material:binding",
                                          "bindMaterialAs", &ok);
    ASSERT(ok == 1);
    ASSERT_STR_EQ(v, "strongerThanDescendants");

    /* The fix: ok == NULL must still return the value, not "". */
    const char* v2 = nanousd_rel_metadatas(mesh, "material:binding",
                                           "bindMaterialAs", NULL);
    ASSERT_STR_EQ(v2, "strongerThanDescendants");

    nanousd_freeprim(mesh);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_layer_n_sublayers(void) {
    NanousdStage stage = nanousd_open(usda_path("compose_three_sublayers.usda"));
    ASSERT(stage != NULL);
    /* Root authors three sublayers */
    int n = nanousd_layer_n_sublayers(stage, 0);
    ASSERT(n == 3);
    const char* a = nanousd_layer_sublayer_path(stage, 0, 0);
    const char* b = nanousd_layer_sublayer_path(stage, 0, 1);
    const char* c = nanousd_layer_sublayer_path(stage, 0, 2);
    ASSERT(a && strstr(a, "compose_layer_a.usda"));
    ASSERT(b && strstr(b, "compose_layer_b.usda"));
    ASSERT(c && strstr(c, "compose_layer_c.usda"));
    /* Out-of-range → empty string */
    ASSERT_STR_EQ(nanousd_layer_sublayer_path(stage, 0, 9), "");
    nanousd_close(stage);
    TEST_PASS();
}

static void test_layer_offset_root_identity(void) {
    NanousdStage stage = nanousd_open(usda_path("compose_three_sublayers.usda"));
    ASSERT(stage != NULL);
    double offset = -1.0, scale = -1.0;
    ASSERT(nanousd_layer_offset(stage, 0, &offset, &scale) == 1);
    /* Root layer is identity. */
    ASSERT(offset == 0.0);
    ASSERT(scale == 1.0);
    /* Pass NULLs — must not crash, still returns 1. */
    ASSERT(nanousd_layer_offset(stage, 0, NULL, NULL) == 1);
    nanousd_close(stage);
    TEST_PASS();
}

static void test_layer_prim_listop_references(void) {
    NanousdStage stage = nanousd_open(usda_path("with_reference.usda"));
    ASSERT(stage != NULL);
    /* Root layer authors `prepend references = @./ref_source.usda@` on
     * /MyPrim. Read it directly from layer 0 — bypasses compositional
     * combining (the per-layer view that the Composition tab needs). */
    NanousdListOp op = nanousd_layer_prim_listop(stage, 0, "/MyPrim", "references");
    ASSERT(op != NULL);
    ASSERT(nanousd_listop_nprepended(op) == 1);
    const char* item = nanousd_listop_prepended(op, 0);
    ASSERT(item != NULL);
    ASSERT(strstr(item, "ref_source.usda") != NULL);
    nanousd_listop_free(op);

    /* Field that's not authored → NULL */
    op = nanousd_layer_prim_listop(stage, 0, "/MyPrim", "specializes");
    ASSERT(op == NULL);
    nanousd_close(stage);
    TEST_PASS();
}

/* Per-layer opinion queries must follow a composition arc's path mapping.
 * with_reference.usda: /MyPrim references ref_source.usda (defaultPrim
 * "Template"), so opinions authored at /Template in the referenced layer
 * compose onto /MyPrim. A per-layer query against the referenced layer must
 * map the composed path /MyPrim back to the authored source path /Template;
 * looking up /MyPrim literally in that layer would miss every opinion behind
 * the arc. */
static void test_layer_opinion_behind_reference_arc(void) {
    NanousdStage stage = nanousd_open(usda_path("with_reference.usda"));
    ASSERT(stage != NULL);

    /* Locate the referenced layer in the composed layer stack. */
    int n = nanousd_stage_n_layers(stage);
    int ref_idx = -1;
    for (int i = 0; i < n; i++) {
        const char* p = nanousd_stage_layer_path(stage, i);
        if (p && strstr(p, "ref_source.usda")) { ref_idx = i; break; }
    }
    ASSERT(ref_idx > 0);  /* referenced layer is present and is not the root */

    /* Opinions authored at /Template are visible at composed /MyPrim. */
    ASSERT(nanousd_layer_has_prim_spec(stage, ref_idx, "/MyPrim") == 1);
    ASSERT(nanousd_layer_has_attr_opinion(stage, ref_idx, "/MyPrim", "height") == 1);
    ASSERT(nanousd_layer_has_attr_opinion(stage, ref_idx, "/MyPrim", "label") == 1);
    /* Child maps through the arc too (/MyPrim/Geo -> /Template/Geo). */
    ASSERT(nanousd_layer_has_prim_spec(stage, ref_idx, "/MyPrim/Geo") == 1);
    ASSERT(nanousd_layer_has_attr_opinion(stage, ref_idx, "/MyPrim/Geo",
                                          "vertexCount") == 1);

    /* The root layer authors the local override but not the arc's attrs. */
    ASSERT(nanousd_layer_has_attr_opinion(stage, 0, "/MyPrim", "label") == 1);
    ASSERT(nanousd_layer_has_attr_opinion(stage, 0, "/MyPrim", "height") == 0);

    nanousd_close(stage);
    TEST_PASS();
}

/* A selected variant body can author opinions in the same layer as the local
 * prim spec. Per-layer queries must check every source path contributed by the
 * layer, not just the local /Model path, or usdview's property stack misses
 * variant-authored fields like /Model.fromVariant. */
static void test_layer_opinion_inside_selected_variant(void) {
    NanousdStage stage = nanousd_open("tests/composition/variant_basic.usda");
    ASSERT(stage != NULL && nanousd_isvalid(stage));

    ASSERT(nanousd_layer_has_prim_spec(stage, 0, "/Model") == 1);
    ASSERT(nanousd_layer_has_attr_opinion(stage, 0, "/Model", "baseline") == 1);
    ASSERT(nanousd_layer_has_attr_opinion(stage, 0, "/Model", "fromVariant") == 1);

    ASSERT(nanousd_layer_has_prim_spec(stage, 0, "/Model/HighChild") == 1);
    ASSERT(nanousd_layer_has_attr_opinion(stage, 0, "/Model/HighChild", "value") == 1);

    nanousd_close(stage);
    TEST_PASS();
}

/* nanousd_nauthored_attribs / nanousd_authored_attribname report only the
 * authored attributes (vs nanousd_nattribs which also reports schema
 * fallbacks). These two methods were relocated to appended backend vtable
 * slots to preserve ABI, so this also exercises that the relocated slots
 * still dispatch to the right functions. */
static void test_nauthored_attribs(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);
    NanousdPrim prim = nanousd_define_prim(stage, "/P", "Xform");
    ASSERT(prim != NULL);
    ASSERT(nanousd_create_attrib(prim, "alpha", "float") == 1);
    ASSERT(nanousd_set_attribf(prim, "alpha", 1.0f) == 1);
    ASSERT(nanousd_create_attrib(prim, "beta", "float") == 1);
    ASSERT(nanousd_set_attribf(prim, "beta", 2.0f) == 1);

    int n = nanousd_nauthored_attribs(prim);
    ASSERT(n == 2);
    int seen_alpha = 0, seen_beta = 0;
    for (int i = 0; i < n; i++) {
        const char* nm = nanousd_authored_attribname(prim, i);
        ASSERT(nm != NULL);
        if (strcmp(nm, "alpha") == 0) seen_alpha = 1;
        else if (strcmp(nm, "beta") == 0) seen_beta = 1;
    }
    ASSERT(seen_alpha == 1);
    ASSERT(seen_beta == 1);

    /* Out-of-range index -> "", NULL prim -> 0 / "". */
    ASSERT_STR_EQ(nanousd_authored_attribname(prim, 99), "");
    ASSERT(nanousd_nauthored_attribs(NULL) == 0);
    ASSERT_STR_EQ(nanousd_authored_attribname(NULL, 0), "");

    nanousd_freeprim(prim);
    nanousd_close(stage);
    TEST_PASS();
}

/* nanousd_resolve_asset_path must treat an empty/NULL anchor as a valid
 * unanchored resolve (an absolute or URI asset resolves to itself). It used
 * to reject empty/NULL anchors outright, even though Stage::Open resolves a
 * root layer with an empty anchor. Only the asset path is required. */
static void test_resolve_asset_path_unanchored(void) {
    char out[1024];
#ifdef _WIN32
    const char* abs_asset = "C:/assets/textures/diffuse.png";
#else
    const char* abs_asset = "/assets/textures/diffuse.png";
#endif
    /* Absolute asset, empty anchor -> resolves to itself (was rejected). */
    ASSERT(nanousd_resolve_asset_path("", abs_asset, out, sizeof(out)) == 1);
    ASSERT_STR_EQ(out, abs_asset);
    /* NULL anchor behaves like empty. */
    ASSERT(nanousd_resolve_asset_path(NULL, abs_asset, out, sizeof(out)) == 1);
    ASSERT_STR_EQ(out, abs_asset);
    /* A URI asset also resolves with no anchor. */
    ASSERT(nanousd_resolve_asset_path("", "https://example.com/a.png",
                                      out, sizeof(out)) == 1);
    ASSERT_STR_EQ(out, "https://example.com/a.png");
    /* The asset path is still required: empty/NULL asset -> failure. */
    ASSERT(nanousd_resolve_asset_path("", "", out, sizeof(out)) == 0);
    ASSERT(nanousd_resolve_asset_path("", NULL, out, sizeof(out)) == 0);
    TEST_PASS();
}

/* ============================================================
 * Composition-arc authoring + prim-state writers (panel-c-api)
 *
 * These exercise spec §6.3.5 (composition arcs), §6.3.6 (active),
 * §6.6 (instanceable), §11.2 (variant sets), and §11.3 (variant
 * declaration). Each test uses only the public C API.
 * ============================================================ */

/* nanousd_add_payload — payload appears in per-layer listop view */
static void test_add_payload_basic(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);
    NanousdPrim prim = nanousd_define_prim(stage, "/Model", "Xform");
    ASSERT(prim != NULL);
    ASSERT(nanousd_add_payload(prim, "./extra.usda", "/Extra") == 1);
    nanousd_freeprim(prim);

    NanousdListOp op = nanousd_layer_prim_listop(stage, 0, "/Model", "payload");
    ASSERT(op != NULL);
    ASSERT(nanousd_listop_nprepended(op) == 1);
    nanousd_listop_free(op);
    nanousd_close(stage);
    TEST_PASS();
}

/* nanousd_add_inherit — inheritPaths listop populated */
static void test_add_inherit_basic(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);

    NanousdPrim klass = nanousd_define_prim_s(stage, "/_class_Pallet", "Xform", "class");
    ASSERT(klass != NULL);
    nanousd_freeprim(klass);

    NanousdPrim p = nanousd_define_prim(stage, "/Pallet1", "Xform");
    ASSERT(p != NULL);
    ASSERT(nanousd_add_inherit(p, "/_class_Pallet") == 1);
    nanousd_freeprim(p);

    NanousdListOp op = nanousd_layer_prim_listop(stage, 0, "/Pallet1", "inheritPaths");
    ASSERT(op != NULL);
    ASSERT(nanousd_listop_nprepended(op) == 1);
    nanousd_listop_free(op);

    /* Reject non-absolute paths */
    NanousdPrim p2 = nanousd_define_prim(stage, "/Pallet2", "Xform");
    ASSERT(nanousd_add_inherit(p2, "Relative") == 0);
    nanousd_freeprim(p2);

    nanousd_close(stage);
    TEST_PASS();
}

/* nanousd_add_specialize — specializes listop populated */
static void test_add_specialize_basic(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);
    NanousdPrim base = nanousd_define_prim_s(stage, "/_specialBase", "Xform", "class");
    ASSERT(base != NULL);
    nanousd_freeprim(base);

    NanousdPrim p = nanousd_define_prim(stage, "/Concrete", "Xform");
    ASSERT(p != NULL);
    ASSERT(nanousd_add_specialize(p, "/_specialBase") == 1);
    nanousd_freeprim(p);

    NanousdListOp op = nanousd_layer_prim_listop(stage, 0, "/Concrete", "specializes");
    ASSERT(op != NULL);
    ASSERT(nanousd_listop_nprepended(op) == 1);
    nanousd_listop_free(op);
    nanousd_close(stage);
    TEST_PASS();
}

/* Add a reference, then remove it — listop becomes empty */
static void test_remove_reference_basic(void) {
    NanousdStage stage = nanousd_create();
    ASSERT(stage != NULL);
    NanousdPrim src = nanousd_define_prim(stage, "/Source", "Xform");
    nanousd_freeprim(src);
    NanousdPrim p = nanousd_define_prim(stage, "/Target", "Xform");
    ASSERT(nanousd_add_reference(p, NULL, "/Source") == 1);

    NanousdListOp op1 = nanousd_layer_prim_listop(stage, 0, "/Target", "references");
    ASSERT(op1 != NULL);
    ASSERT(nanousd_listop_nprepended(op1) == 1);
    nanousd_listop_free(op1);

    ASSERT(nanousd_remove_reference(p, 0) == 1);
    nanousd_freeprim(p);

    NanousdListOp op2 = nanousd_layer_prim_listop(stage, 0, "/Target", "references");
    /* The field is still authored (now empty); accept either NULL or 0 prepended. */
    if (op2) {
        ASSERT(nanousd_listop_nprepended(op2) == 0);
        nanousd_listop_free(op2);
    }
    nanousd_close(stage);
    TEST_PASS();
}

/* Removal with bad arguments fails cleanly */
static void test_remove_listop_item_invalid(void) {
    NanousdStage stage = nanousd_create();
    NanousdPrim p = nanousd_define_prim(stage, "/X", "Xform");
    /* Field with no opinion */
    ASSERT(nanousd_remove_listop_item(p, "references", 1, 0) == 0);
    /* Bad listOpKind */
    ASSERT(nanousd_remove_listop_item(p, "references", 99, 0) == 0);
    /* Negative index */
    ASSERT(nanousd_remove_listop_item(p, "references", 1, -1) == 0);
    /* Null prim */
    ASSERT(nanousd_remove_listop_item(NULL, "references", 1, 0) == 0);
    nanousd_freeprim(p);
    nanousd_close(stage);
    TEST_PASS();
}

/* Active flag round-trip. Deactivation masks the prim from composed
 * traversal (spec §6.3.6) — primpath() returns no valid prim while the
 * authored opinion is false. Reactivation restores it. */
static void test_set_active_roundtrip(void) {
    NanousdStage stage = nanousd_create();
    NanousdPrim p = nanousd_define_prim(stage, "/Inert", "Xform");
    ASSERT(nanousd_isactive(p) == 1);
    nanousd_freeprim(p);

    NanousdPrim p2 = nanousd_primpath(stage, "/Inert");
    int n_before = nanousd_nprims(stage);
    int flat_before = nanousd_traverse_flat(stage, NULL, 0);
    ASSERT(n_before >= 1);
    ASSERT(flat_before >= 1);
    ASSERT(nanousd_set_active(p2, 0) == 1);

    /* Deactivated → masked from composed traversal */
    ASSERT(nanousd_nprims(stage) < n_before);
    ASSERT(nanousd_traverse_flat(stage, NULL, 0) < flat_before);
    NanousdPrim missing = nanousd_primpath(stage, "/Inert");
    ASSERT(missing == NULL || nanousd_prim_isvalid(missing) == 0);
    if (missing) nanousd_freeprim(missing);

    /* Authored value is visible in the per-layer view */
    ASSERT(nanousd_layer_has_prim_spec(stage, 0, "/Inert") == 1);

    /* Reactivate through the same handle. A deactivated prim is masked, so
     * primpath() cannot re-acquire it; the handle that deactivated it retains
     * the path needed to author active=1 back. */
    ASSERT(nanousd_set_active(p2, 1) == 1);
    nanousd_freeprim(p2);

    /* Visible in composed traversal again. */
    ASSERT(nanousd_nprims(stage) == n_before);
    ASSERT(nanousd_traverse_flat(stage, NULL, 0) == flat_before);
    NanousdPrim back = nanousd_primpath(stage, "/Inert");
    ASSERT(back != NULL && nanousd_prim_isvalid(back) == 1);
    ASSERT(nanousd_isactive(back) == 1);
    if (back) nanousd_freeprim(back);

    nanousd_close(stage);
    TEST_PASS();
}

/* Instanceable flag round-trip */
static void test_set_instanceable_roundtrip(void) {
    NanousdStage stage = nanousd_create();
    NanousdPrim p = nanousd_define_prim(stage, "/Inst", "Xform");
    ASSERT(nanousd_isinstanceable(p) == 0);
    ASSERT(nanousd_set_instanceable(p, 1) == 1);
    ASSERT(nanousd_isinstanceable(p) == 1);
    ASSERT(nanousd_set_instanceable(p, 0) == 1);
    ASSERT(nanousd_isinstanceable(p) == 0);
    nanousd_freeprim(p);
    nanousd_close(stage);
    TEST_PASS();
}

/* apply_api -> remove_api round-trip on apiSchemas listop */
static void test_remove_api_roundtrip(void) {
    NanousdStage stage = nanousd_create();
    NanousdPrim p = nanousd_define_prim(stage, "/Body", "Xform");
    ASSERT(nanousd_apply_api(p, "PhysicsRigidBodyAPI") == 1);
    ASSERT(nanousd_hasapi(p, "PhysicsRigidBodyAPI") == 1);
    ASSERT(nanousd_remove_api(p, "PhysicsRigidBodyAPI") == 1);
    ASSERT(nanousd_hasapi(p, "PhysicsRigidBodyAPI") == 0);
    /* Removing a non-applied schema is a no-op (returns 0). */
    ASSERT(nanousd_remove_api(p, "PhysicsRigidBodyAPI") == 0);
    nanousd_freeprim(p);
    nanousd_close(stage);
    TEST_PASS();
}

/* remove_prim removes the prim and any descendants */
static void test_remove_prim_basic(void) {
    NanousdStage stage = nanousd_create();
    NanousdPrim parent = nanousd_define_prim(stage, "/Group", "Xform");
    NanousdPrim child  = nanousd_define_prim(stage, "/Group/Box", "Cube");
    nanousd_freeprim(child);

    int n_before = nanousd_nprims(stage);
    ASSERT(n_before >= 2);
    ASSERT(nanousd_remove_prim(parent) == 1);
    nanousd_freeprim(parent);

    /* Both parent and child must be gone from composed traversal. */
    NanousdPrim missing1 = nanousd_primpath(stage, "/Group");
    NanousdPrim missing2 = nanousd_primpath(stage, "/Group/Box");
    ASSERT(missing1 == NULL || nanousd_prim_isvalid(missing1) == 0);
    ASSERT(missing2 == NULL || nanousd_prim_isvalid(missing2) == 0);
    if (missing1) nanousd_freeprim(missing1);
    if (missing2) nanousd_freeprim(missing2);

    nanousd_close(stage);
    TEST_PASS();
}

/* Variant-set declaration is idempotent and visible via the read API */
static void test_create_variantset_idempotent(void) {
    NanousdStage stage = nanousd_create();
    NanousdPrim p = nanousd_define_prim(stage, "/Asset", "Xform");
    ASSERT(nanousd_create_variantset(p, "shading") == 1);
    ASSERT(nanousd_create_variantset(p, "shading") == 1);  /* idempotent */
    ASSERT(nanousd_hasvariantset(p, "shading") == 1);
    ASSERT(nanousd_nvariantsets(p) >= 1);
    nanousd_freeprim(p);
    nanousd_close(stage);
    TEST_PASS();
}

/* Variants declared inside a set are enumerable */
static void test_create_variant_basic(void) {
    NanousdStage stage = nanousd_create();
    NanousdPrim p = nanousd_define_prim(stage, "/Asset", "Xform");
    ASSERT(nanousd_create_variant(p, "shading", "red")  == 1);
    ASSERT(nanousd_create_variant(p, "shading", "blue") == 1);
    /* nvariants reports both variants in the set */
    ASSERT(nanousd_nvariants(p, "shading") == 2);
    /* Both variant names should be enumerable (order not guaranteed). */
    int saw_red = 0, saw_blue = 0;
    for (int i = 0; i < nanousd_nvariants(p, "shading"); ++i) {
        const char* name = nanousd_variantname(p, "shading", i);
        if (name && strcmp(name, "red") == 0)  saw_red  = 1;
        if (name && strcmp(name, "blue") == 0) saw_blue = 1;
    }
    ASSERT(saw_red && saw_blue);
    nanousd_freeprim(p);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Asset path value resolution (spec §9.4, §12.3)
 * ============================================================ */

/* Cross-layer asset resolution: attribute from a referenced layer must be
 * resolved relative to that layer, not relative to the root layer. */
static void test_asset_resolve_cross_layer(void) {
    NanousdStage stage = nanousd_open(usda_path("asset_resolve_root.usda"));
    ASSERT(stage != NULL);
    ASSERT(nanousd_isvalid(stage) == 1);

    NanousdPrim prim = nanousd_primpath(stage, "/Model");
    ASSERT(prim != NULL);

    int ok = 0;
    const char* tex = nanousd_attribasset(prim, "inputs:texture", &ok);
    ASSERT(ok == 1);
    ASSERT(tex != NULL);
    /* The resolved path must be anchored to sub/asset_resolve_ref.usda,
     * so it must contain the "sub" directory segment and "diffuse.png". */
    ASSERT_MSG(strstr(tex, "sub") != NULL, "resolved path should contain 'sub'");
    ASSERT_MSG(strstr(tex, "diffuse.png") != NULL, "resolved path should contain 'diffuse.png'");

    nanousd_freeprim(prim);
    nanousd_close(stage);
    TEST_PASS();
}

/* Absolute asset paths must pass through resolution unchanged. */
static void test_asset_resolve_absolute_passthrough(void) {
    NanousdStage stage = nanousd_open(usda_path("asset_resolve_root.usda"));
    ASSERT(stage != NULL);
    ASSERT(nanousd_isvalid(stage) == 1);

    NanousdPrim prim = nanousd_primpath(stage, "/Model");
    ASSERT(prim != NULL);

    int ok = 0;
    const char* abs = nanousd_attribasset(prim, "inputs:absolute", &ok);
    ASSERT(ok == 1);
    ASSERT(abs != NULL);
    ASSERT_STR_EQ(abs, "/absolute/texture.png");

    nanousd_freeprim(prim);
    nanousd_close(stage);
    TEST_PASS();
}

/* Empty asset values must be returned as-is without resolution. */
static void test_asset_resolve_empty(void) {
    NanousdStage stage = nanousd_open(usda_path("asset_resolve_root.usda"));
    ASSERT(stage != NULL);
    ASSERT(nanousd_isvalid(stage) == 1);

    NanousdPrim prim = nanousd_primpath(stage, "/Model");
    ASSERT(prim != NULL);

    int ok = 0;
    const char* empty = nanousd_attribasset(prim, "inputs:empty", &ok);
    ASSERT(ok == 1);
    ASSERT(empty != NULL);
    ASSERT_STR_EQ(empty, "");

    nanousd_freeprim(prim);
    nanousd_close(stage);
    TEST_PASS();
}

/* Time-sampled asset values from a referenced layer must be resolved
 * against the layer that authored the timeSamples, not the root. */
static void test_asset_resolve_timesample(void) {
    /* The flatten path resolves timeSamples via FlattenStage.
     * Verify via the flattened USDA string output. */
    NanousdStage stage = nanousd_open(usda_path("asset_resolve_root.usda"));
    ASSERT(stage != NULL);
    ASSERT(nanousd_isvalid(stage) == 1);

    const char* usda = nanousd_write_usda_string(stage);
    ASSERT(usda != NULL);

    /* timeSamples come from sub/asset_resolve_ref.usda which has:
     *   asset inputs:animated.timeSamples = { 1: @./tex/frame001.exr@ ... }
     * After resolution, these should be anchored to the sub/ directory. */
    ASSERT_MSG(strstr(usda, "frame001.exr") != NULL,
              "flattened USDA should contain frame001.exr");
    ASSERT_MSG(strstr(usda, "frame002.exr") != NULL,
              "flattened USDA should contain frame002.exr");
    /* The resolved paths must contain the sub/ directory component. */
    ASSERT_MSG(strstr(usda, "sub") != NULL,
              "flattened time-sampled asset paths should be anchored to sub/");
    /* Must NOT contain the raw unresolved relative prefix. */
    ASSERT_MSG(strstr(usda, "@./tex/frame001") == NULL,
              "flattened USDA must not contain raw relative asset path");

    nanousd_free_string(usda);
    nanousd_close(stage);
    TEST_PASS();
}

/* Asset array elements from a referenced layer must each be resolved
 * against the anchor layer. */
static void test_asset_resolve_array(void) {
    NanousdStage stage = nanousd_open(usda_path("asset_resolve_root.usda"));
    ASSERT(stage != NULL);

    const char* usda = nanousd_write_usda_string(stage);
    ASSERT(usda != NULL);

    /* asset[] inputs:texarray = [@./tex/a.png@, @./tex/b.png@]
     * After flatten, both elements must be resolved to sub/tex/... */
    ASSERT_MSG(strstr(usda, "a.png") != NULL,
              "flattened USDA should contain a.png");
    ASSERT_MSG(strstr(usda, "b.png") != NULL,
              "flattened USDA should contain b.png");
    /* Must NOT contain raw relative paths */
    ASSERT_MSG(strstr(usda, "@./tex/a.png@") == NULL,
              "flattened asset array must not contain raw relative path");

    nanousd_free_string(usda);
    nanousd_close(stage);
    TEST_PASS();
}

/* Flattened output must be self-contained: all asset paths resolvable
 * from the flattened layer alone (absolute or relative to output). */
static void test_asset_resolve_flatten_selfcontained(void) {
    NanousdStage stage = nanousd_open(usda_path("asset_resolve_root.usda"));
    ASSERT(stage != NULL);

    const char* usda = nanousd_write_usda_string(stage);
    ASSERT(usda != NULL);

    /* Absolute path must survive flattening unchanged. */
    ASSERT_MSG(strstr(usda, "/absolute/texture.png") != NULL,
              "flattened USDA must preserve absolute asset path");
    /* The cross-layer relative path must be resolved (not raw ./tex/...). */
    ASSERT_MSG(strstr(usda, "diffuse.png") != NULL,
              "flattened USDA should contain diffuse.png");
    /* Empty asset must survive as @@ */
    ASSERT_MSG(strstr(usda, "@@") != NULL,
              "flattened USDA must preserve empty asset value");

    nanousd_free_string(usda);
    nanousd_close(stage);
    TEST_PASS();
}

/* Single-layer: relative paths resolve against the layer itself.
 * On read, the value should be resolved (absolute or rooted). */
static void test_asset_resolve_single_layer(void) {
    NanousdStage stage = nanousd_open(usda_path("asset_resolve_single.usda"));
    ASSERT(stage != NULL);
    ASSERT(nanousd_isvalid(stage) == 1);

    NanousdPrim prim = nanousd_primpath(stage, "/Mesh");
    ASSERT(prim != NULL);

    int ok = 0;
    const char* diff = nanousd_attribasset(prim, "inputs:diffuse", &ok);
    ASSERT(ok == 1);
    ASSERT(diff != NULL);
    /* Resolved path should contain metal.png and should not
     * be the raw relative string "./textures/metal.png". */
    ASSERT_MSG(strstr(diff, "metal.png") != NULL,
              "resolved path should contain metal.png");
    ASSERT_MSG(strstr(diff, "textures") != NULL,
              "resolved path should contain textures directory");

    /* Absolute path on single layer: unchanged */
    const char* abs = nanousd_attribasset(prim, "inputs:abs", &ok);
    ASSERT(ok == 1);
    ASSERT_STR_EQ(abs, "/global/shared.png");

    nanousd_freeprim(prim);
    nanousd_close(stage);
    TEST_PASS();
}

/* Write fidelity: overwriting an existing asset attribute and reading back
 * must preserve the authored identifier through the round-trip. */
static void test_asset_resolve_write_fidelity(void) {
    NanousdStage stage = nanousd_open(usda_path("asset_resolve_single.usda"));
    ASSERT(stage != NULL);

    NanousdPrim prim = nanousd_primpath(stage, "/Mesh");
    ASSERT(prim != NULL);

    /* Overwrite existing asset attribute with a new relative path */
    int rc = nanousd_set_attrib_asset(prim, "inputs:diffuse", "./renders/beauty.exr");
    ASSERT(rc == 1);

    /* Read it back — should be resolved but still contain beauty.exr */
    int ok = 0;
    const char* val = nanousd_attribasset(prim, "inputs:diffuse", &ok);
    ASSERT(ok == 1);
    ASSERT(val != NULL);
    ASSERT_MSG(strstr(val, "beauty.exr") != NULL,
              "read-back should contain beauty.exr");

    /* Flatten to string — the written path should appear resolved */
    const char* usda = nanousd_write_usda_string(stage);
    ASSERT(usda != NULL);
    ASSERT_MSG(strstr(usda, "beauty.exr") != NULL,
              "flattened output should contain the authored beauty.exr");

    nanousd_free_string(usda);
    nanousd_freeprim(prim);
    nanousd_close(stage);
    TEST_PASS();
}

/* Spec §5: Asset path values in metadata dictionaries must be resolved
 * against the anchor layer. This test verifies that customData on both
 * prims and attributes have their asset paths resolved when flattening
 * a multi-layer stage. */
static void test_asset_resolve_metadata(void) {
    NanousdStage stage = nanousd_open(usda_path("asset_resolve_metadata_root.usda"));
    ASSERT_MSG(nanousd_isvalid(stage) == 1, "stage should be valid");

    /* Flatten — all asset paths in metadata should be resolved against
     * the source layer (sub/asset_resolve_metadata_ref.usda) */
    const char* usda = nanousd_write_usda_string(stage);
    ASSERT(usda != NULL);

    /* Prim customData: thumbnail should be resolved from ./textures/thumb.png
     * relative to sub/ directory, so the flattened output should contain
     * sub/textures/thumb.png (not the raw ./textures/thumb.png) */
    ASSERT_MSG(strstr(usda, "sub/textures/thumb.png") != NULL ||
               strstr(usda, "textures/thumb.png") != NULL,
              "prim customData asset should be resolved relative to source layer");
    ASSERT_MSG(strstr(usda, "@./textures/thumb.png@") == NULL,
              "prim customData should not contain unresolved relative path");

    /* Attribute customData: preview should also be resolved */
    ASSERT_MSG(strstr(usda, "sub/textures/preview.png") != NULL ||
               strstr(usda, "textures/preview.png") != NULL,
              "attribute customData asset should be resolved relative to source layer");
    ASSERT_MSG(strstr(usda, "@./textures/preview.png@") == NULL,
              "attribute customData should not contain unresolved relative path");

    /* The attribute value itself should also be resolved */
    ASSERT_MSG(strstr(usda, "sub/textures/diffuse.exr") != NULL ||
               strstr(usda, "textures/diffuse.exr") != NULL,
              "attribute value should be resolved relative to source layer");

    nanousd_free_string(usda);
    nanousd_close(stage);
    TEST_PASS();
}

/* Absolute URI resource identifiers must be handled as absolute identifiers,
 * not as filesystem-relative paths anchored to the layer directory. */
static void test_asset_resolve_uri_passthrough(void) {
    NanousdStage stage = nanousd_open(usda_path("asset_resolve_uri.usda"));
    ASSERT(stage != NULL);
    ASSERT(nanousd_isvalid(stage) == 1);

    NanousdPrim prim = nanousd_primpath(stage, "/Mat");
    ASSERT(prim != NULL);

    int ok = 0;
    const char* uri = nanousd_attribasset(prim, "inputs:remote", &ok);
    ASSERT(ok == 1);
    ASSERT(uri != NULL);
    ASSERT_STR_EQ(uri, "https://example.com/assets/diffuse.png?color=red#preview");

    const char* local_file = nanousd_attribasset(prim, "inputs:localFile", &ok);
    ASSERT(ok == 1);
    ASSERT(local_file != NULL);
    ASSERT_STR_EQ(local_file, "/tmp/nanousd/local-texture.png");

    const char* remote_file = nanousd_attribasset(prim, "inputs:remoteFile", &ok);
    ASSERT(ok == 1);
    ASSERT(remote_file != NULL);
    ASSERT_STR_EQ(remote_file, "file://files.example.com/share/remote-texture.png");

    nanousd_freeprim(prim);
    nanousd_close(stage);
    TEST_PASS();
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    if (argc > 1) {
        g_usda_dir = argv[1];
    }

    printf("=== USD Core Specification Compliance Tests ===\n");
    printf("USDA directory: %s\n\n", g_usda_dir);

    printf("--- Stage Lifecycle ---\n");
    test_stage_open_valid();
    test_stage_open_null();
    test_stage_open_missing();
    test_stage_open_masked();
    test_stage_null_handle();

    printf("\n--- Stage Metadata ---\n");
    test_stage_metadata();
    test_stage_metadata_root_layer_only();
    test_stage_default_prim();
    test_stage_generic_metadata();
    test_stage_root_layer_path();

    printf("\n--- Prim Traversal & Hierarchy ---\n");
    test_prim_traversal();
    test_prim_hierarchy();
    test_define_prim_child_index_unique();
    test_define_prim_child_authored_order();
    test_prim_queries();

    printf("\n--- Scalar Attributes ---\n");
    test_scalar_attributes();

    printf("\n--- Vector Attributes ---\n");
    test_vec_attributes();
    test_matrix_attributes();
    test_foundational_type_names();

    printf("\n--- Array Attributes ---\n");
    test_array_attributes();

    printf("\n--- Time Samples ---\n");
    test_timesamples();

    printf("\n--- Relationships ---\n");
    test_relationships();

    printf("\n--- Collections ---\n");
    test_collections();

    printf("\n--- Specifier Semantics ---\n");
    test_specifier_def();
    test_specifier_class();
    test_specifier_ancestor_state();
    test_specifier_over_not_populated();
    test_specifier_class_reachable();
    test_specifier_over_not_in_children();
    test_specifier_over_gains_def();

    printf("\n--- Active/Inactive Filtering ---\n");
    test_inactive_excluded();
    test_inactive_nested();
    test_inactive_default_prim_excluded();

    printf("\n--- Value Resolution ---\n");
    test_value_timesample_over_default();
    test_value_block();

    printf("\n--- Fixture Layer Comparisons ---\n");
    test_fixture_squashed_layer_compare();
    test_fixture_population_mask_compare();
    test_fixture_path_remap_compare();
    test_fixture_sampled_layer_compare();
    test_fixture_usdc_roundtrip_metamorphic();
    test_generated_format_roundtrip_matrix();
    test_generated_adversarial_usda_rejections();

    printf("\n--- Composition ---\n");
    test_composition_sublayer();
    test_composition_three_sublayers();
    test_composition_reference();
    test_composition_path_valued_listop_remap_references();
    test_composition_path_valued_listop_remap_payloads();
    test_composition_path_valued_listop_remap_inherits_specializes();
    test_composition_chained_inherits();
    test_composition_implied_inherits();
    test_composition_implied_specializes();
    test_composition_livrps_strength_ordering();
    test_composition_path_valued_listop_remap_variants();
    test_composition_relocates_reference_child();
    test_composition_relocates_nested_layer_stack();
    test_composition_relocates_write_roundtrip();
    test_composition_sublayer_strength();
    test_composition_reference_strength();
    test_composition_invalid_retiming_scale();
    test_composition_apischemas_combined();

    printf("\n--- Package Format ---\n");
    test_package_usdz_read();
    test_package_usdz_write();
    test_file_uri_resource_open();

    printf("\n--- Schema Queries ---\n");
    test_schema_isa();
    test_hasapi();
    test_prim_listop();

    printf("\n--- Property & Prim Ordering ---\n");
    test_property_order();
    test_property_order_partial();
    test_property_order_from_weaker_layer();
    test_property_path_element_ordering();
    test_prim_child_order();
    test_composed_prim_order_from_weaker_layer();
    test_root_prim_order();

    printf("\n--- Path Operations ---\n");
    test_path_parse();
    test_path_operations();
    test_unicode_identifiers();
    test_usda_grammar_strictness();

    printf("\n--- ListOp Operations ---\n");
    test_listop_explicit();
    test_listop_composable();
    test_listop_combine();
    test_listop_explicit_overrides();
    test_listop_prepend_delete();

    printf("\n--- Math Utilities ---\n");
    test_math_vec3();
    test_math_matrix();
    test_math_quaternion();

    printf("\n--- Write Operations ---\n");
    test_write_scalar();
    test_write_vector();
    test_write_time_samples();
    test_write_clear_block();
    test_write_create_attrib();

    printf("\n--- Write-Read Roundtrip ---\n");
    test_write_read_scalar_roundtrip();
    test_write_read_vector_roundtrip();

    printf("\n--- Write on Composed Stages ---\n");
    test_write_composed_sublayer();
    test_write_composed_reference();
    test_write_composed_timesample();
    test_write_composed_create_attrib();
    test_write_composed_block_unblock();

    printf("\n--- Null Safety ---\n");
    test_null_safety();

    printf("\n--- Array Time Samples ---\n");
    test_array_time_samples();

    printf("\n--- USDA Write Support ---\n");
    test_write_usda_null_safety();
    test_write_usda_basic();
    test_write_usda_roundtrip();
    test_write_file_uri_resources();

    printf("\n--- USDA Write-to-String ---\n");
    test_write_usda_string();
    test_write_usda_string_null();

    printf("\n--- USDC Write Support ---\n");
    test_write_stage_metadata();
    test_write_stage_metadata_token();
    test_write_usdc_binary_magic_roundtrip();
    test_write_usdc_file_binary_roundtrip();
    test_write_usdc_file_uri_binary();
    test_write_usdc_roundtrip();
    test_write_usdc_multi_root_prim_count();
    test_write_usdc_prim_children_fields();
    test_define_prim_with_specifier();
    test_set_specifier();

    printf("\n--- Token/Asset Attribute Setters ---\n");
    test_set_attrib_token_asset();
    test_set_attrib_token_array_readback();
    test_set_attrib_token_asset_null();

    printf("\n--- Composition Arc Write ---\n");
    test_add_reference_internal();
    test_add_reference_null();

    printf("\n--- Relationship Creation ---\n");
    test_create_rel();
    test_create_rel_null();

    printf("\n--- Schema Registration ---\n");
    test_register_schemas_json();
    test_schema_autoapplies();
    test_register_schemas_json_null();
    test_schema_attribute_names_exclude_relationship_defs();

    printf("\n--- Color Spaces ---\n");
    test_color_spaces();
    test_color_space_authoring();

    printf("\n--- Prim Metadata ---\n");
    test_prim_metadata();
    test_prim_metadata_null();

    printf("\n--- Bulk Array Access ---\n");
    test_bulk_array_float();
    test_bulk_array_double();
    test_bulk_array_int();
    test_bulk_array_null();

    printf("\n--- Vec Array Read/Write ---\n");
    test_vec3f_array_read();
    test_vec3d_array_read();
    test_vec3f_array_write_roundtrip();

    printf("\n--- Quaternion Attributes ---\n");
    test_quatf_read();
    test_quatd_read();
    test_quat_write_roundtrip();

    printf("\n--- Matrix3d Attributes ---\n");
    test_matrix3d_read();
    test_matrix3d_write_roundtrip();

    printf("\n--- String Array Read/Write ---\n");
    test_string_array_read();
    test_string_array_write_roundtrip();

    printf("\n--- kilogramsPerUnit Metadata ---\n");
    test_kilogramsperunit();

    printf("\n--- Instancing ---\n");
    test_isinstance();
    test_prototype_nav();
    test_semantic_traversal_and_composition_queries();
    test_nested_ref_instancing_shares_prototype();
    test_repeated_instanceable_reference_children();
    test_inactive_represented_instance_pruned();

    printf("\n--- Variants ---\n");
    test_variant_basic_selection();
    test_variant_body_chained_reference();
    test_variant_nested_expansion();

    printf("\n--- Payloads ---\n");
    test_payload_diamond_no_cycle();
    test_payload_nested_livrps();

    printf("\n--- Variant API ---\n");
    test_variant_api_read();
    test_variant_api_author();
    test_instancing_variant_differentiates_prototype();
    test_ancestral_variant_subroot_reference();
    test_metadata_dict_combine();
    test_metadata_listop_relationship_targets_combine();
    test_metadata_listop_apischemas_combine();
    test_metadata_listop_explicit_prunes_weaker();

    printf("\n--- Diagnostics ---\n");
    test_diagnostics_missing_sublayer();
    test_diagnostics_json();
    test_diagnostics_clean_stage();
    test_diagnostics_free_null();

    printf("\n--- Asset Path Value Resolution ---\n");
    test_asset_resolve_cross_layer();
    test_asset_resolve_absolute_passthrough();
    test_asset_resolve_empty();
    test_asset_resolve_timesample();
    test_asset_resolve_array();
    test_asset_resolve_flatten_selfcontained();
    test_asset_resolve_single_layer();
    test_asset_resolve_write_fidelity();
    test_asset_resolve_metadata();
    test_asset_resolve_uri_passthrough();
    test_geommodelapi_schema_attrs();
    test_resolve_asset_path_unanchored();

    printf("\n--- Authored attribute enumeration ---\n");
    test_nauthored_attribs();

    printf("\n--- Per-layer spec / opinion queries ---\n");
    test_layer_has_prim_spec();
    test_layer_has_attr_opinion();
    test_layer_attr_nsamples_no_samples();
    test_layer_attr_nsamples_authored();
    test_rel_metadatas_null_ok();
    test_layer_n_sublayers();
    test_layer_offset_root_identity();
    test_layer_prim_listop_references();
    test_layer_opinion_behind_reference_arc();
    test_layer_opinion_inside_selected_variant();

    /* Composition-arc authoring + prim-state writers (panel-c-api branch) */
    test_add_payload_basic();
    test_add_inherit_basic();
    test_add_specialize_basic();
    test_remove_reference_basic();
    test_remove_listop_item_invalid();
    test_set_active_roundtrip();
    test_set_instanceable_roundtrip();
    test_remove_api_roundtrip();
    test_remove_prim_basic();
    test_create_variantset_idempotent();
    test_create_variant_basic();

    printf("\n=== Results: %d passed, %d failed out of %d tests ===\n",
           g_tests_passed, g_tests_failed, g_tests_run);

    return g_tests_failed > 0 ? 1 : 0;
}

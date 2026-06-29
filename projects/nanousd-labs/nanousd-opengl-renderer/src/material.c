// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * material.c -- Simplified material loader for the OpenGL renderer.
 *
 * Extracts material properties (base_color, roughness, metallic, etc.)
 * directly from USD prims via nanousdapi. Loads textures via stb_image.
 * No MaterialX, no shaderc, no SPIR-V compilation.
 */

#define _GNU_SOURCE  /* expose nftw / struct FTW / FTW_PHYS in <ftw.h> */
#include "material.h"
#include "mdl_bridge.h"
#include "scene.h"
#include <nanousd/nanousdapi.h>
#include "nu_parallel.h"
#ifdef NUSD_HAVE_PTEX
#include "ptex_material.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>
#include <dirent.h>
#include <sys/stat.h>
#include <ftw.h>
#include <limits.h>
#include <math.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

/* ---- Filename index (basename → full path) ----
 * NVIDIA SimReady scenes (Isaac warehouse, etc.) author texture paths
 * relative to the *prop* USD that lives in a subdirectory, not the
 * top-level USD that we opened. After referencing, the texture path
 * (`../Materials/Textures/T_*.png`) resolves to a sibling tree that
 * doesn't contain the file. To recover, we recursively walk the scene
 * tree once, build a basename→fullpath index, and use it as a fallback
 * when the literal path doesn't exist. Same strategy as
 * nanousd-vulkan-renderer's resolve_texture_path. */
typedef struct { char name[128]; char full[1024]; int root_rank; } FileIndexEntry;
static FileIndexEntry* g_file_index = NULL;
static int  g_file_index_n   = 0;
static int  g_file_index_cap = 0;
static char g_file_index_roots[32][1024];
static int  g_file_index_nroots = 0;
static int  g_file_index_current_root_rank = 0;

static int file_index_visit(const char* path, const struct stat* sb,
                             int typeflag, struct FTW* ftwbuf)
{
    (void)sb;
    if (typeflag != FTW_F) return 0;
    if (ftwbuf->level > 16) return 0;
    if (strstr(path, "/.thumbs/")) return 0;
    const char* base = strrchr(path, '/');
    base = base ? base + 1 : path;
    if (g_file_index_n >= g_file_index_cap) {
        int newcap = g_file_index_cap ? g_file_index_cap * 2 : 512;
        FileIndexEntry* nu = (FileIndexEntry*)realloc(
            g_file_index, (size_t)newcap * sizeof(FileIndexEntry));
        if (!nu) return 0;
        g_file_index = nu;
        g_file_index_cap = newcap;
    }
    FileIndexEntry* e = &g_file_index[g_file_index_n++];
    snprintf(e->name, sizeof(e->name), "%s", base);
    snprintf(e->full, sizeof(e->full), "%s", path);
    e->root_rank = g_file_index_current_root_rank;
    return 0;
}

static int file_index_cmp(const void* a, const void* b) {
    const FileIndexEntry* ea = (const FileIndexEntry*)a;
    const FileIndexEntry* eb = (const FileIndexEntry*)b;
    int name_cmp = strcmp(ea->name, eb->name);
    if (name_cmp != 0) return name_cmp;
    if (ea->root_rank != eb->root_rank)
        return ea->root_rank < eb->root_rank ? -1 : 1;
    return strcmp(ea->full, eb->full);
}

static int file_index_roots_match(char roots[][1024], int nroots) {
    if (nroots != g_file_index_nroots) return 0;
    for (int i = 0; i < nroots; i++) {
        if (strcmp(roots[i], g_file_index_roots[i]) != 0) return 0;
    }
    return 1;
}

static void file_index_ensure_many(char roots[][1024], int nroots) {
    if (!roots || nroots <= 0) return;
    if (nroots > 32) nroots = 32;
    if (file_index_roots_match(roots, nroots)) return;

    free(g_file_index);
    g_file_index = NULL;
    g_file_index_n = 0;
    g_file_index_cap = 0;
    g_file_index_nroots = nroots;
    for (int i = 0; i < nroots; i++)
        snprintf(g_file_index_roots[i], sizeof(g_file_index_roots[i]), "%s", roots[i]);

    /* nftw with FTW_PHYS to avoid following symlinks, fd budget 16. */
    for (int i = 0; i < nroots; i++) {
        g_file_index_current_root_rank = i;
        nftw(roots[i], file_index_visit, 16, FTW_PHYS);
    }
    if (g_file_index_n > 0)
        qsort(g_file_index, (size_t)g_file_index_n,
              sizeof(FileIndexEntry), file_index_cmp);
    fprintf(stderr, "material: indexed %d files across %d texture search roots\n",
            g_file_index_n, nroots);
}

static const char* file_index_lookup(const char* basename) {
    if (!g_file_index || g_file_index_n == 0 || !basename || !basename[0])
        return NULL;
    int lo = 0, hi = g_file_index_n;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (strcmp(g_file_index[mid].name, basename) < 0)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo >= g_file_index_n) return NULL;
    if (strcmp(g_file_index[lo].name, basename) != 0) return NULL;
    return g_file_index[lo].full;
}

static int file_exists(const char* path) {
    if (!path || !path[0]) return 0;
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static void path_parent(char* path);

static int dir_exists(const char* path) {
    if (!path || !path[0]) return 0;
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static int file_nonempty(const char* path) {
    if (!path || !path[0]) return 0;
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0;
}

static int env_enabled(const char* name) {
    const char* e = getenv(name);
    return e && e[0] && e[0] != '0' &&
           strcmp(e, "false") != 0 && strcmp(e, "False") != 0 &&
           strcmp(e, "off") != 0 && strcmp(e, "OFF") != 0 &&
           strcmp(e, "no") != 0 && strcmp(e, "NO") != 0;
}

static int is_http_url(const char* s) {
    return s && (strncmp(s, "https://", 8) == 0 ||
                 strncmp(s, "http://", 7) == 0);
}

static int remote_asset_fetch_allowed(const char* url) {
    if (env_enabled("NUSD_DISABLE_REMOTE_ASSET_FETCH")) return 0;
    if (env_enabled("NUSD_ENABLE_REMOTE_ASSET_FETCH")) return 1;
    /* Opt-in only: never trigger a network fetch on scene load by default,
     * not even the GB300 paint preset URL (previously auto-fetched without
     * the opt-in). Warn once so the env var is discoverable. */
    static int warned = 0;
    if (!warned && url && strncmp(url, "http", 4) == 0) {
        warned = 1;
        fprintf(stderr,
                "[nusd] remote asset fetch is opt-in; set "
                "NUSD_ENABLE_REMOTE_ASSET_FETCH=1 to download remote textures "
                "(e.g. DSX GB300 paint). Skipping remote assets this run.\n");
    }
    return 0;
}

static int remote_asset_cache_root(char* out, size_t out_size) {
    const char* env = getenv("NUSD_REMOTE_ASSET_CACHE");
    if (env && env[0]) {
        int n = snprintf(out, out_size, "%s", env);
        return n >= 0 && (size_t)n < out_size;
    }

    const char* xdg = getenv("XDG_CACHE_HOME");
    if (xdg && xdg[0]) {
        int n = snprintf(out, out_size, "%s/nusd_renderer/remote_assets", xdg);
        return n >= 0 && (size_t)n < out_size;
    }

    const char* home = getenv("HOME");
    if (home && home[0]) {
        int n = snprintf(out, out_size, "%s/.cache/nusd_renderer/remote_assets", home);
        return n >= 0 && (size_t)n < out_size;
    }

    int n = snprintf(out, out_size, "/tmp/nusd_renderer_remote_assets");
    return n >= 0 && (size_t)n < out_size;
}

static void sanitize_cache_segment(const char* begin, size_t len,
                                   char* out, size_t out_size) {
    size_t n = 0;
    for (size_t i = 0; i < len && n + 1 < out_size; i++) {
        unsigned char c = (unsigned char)begin[i];
        if (isalnum(c) || c == '.' || c == '_' || c == '-')
            out[n++] = (char)c;
        else
            out[n++] = '_';
    }
    out[n] = '\0';
    if (n == 0 || strcmp(out, ".") == 0 || strcmp(out, "..") == 0)
        snprintf(out, out_size, "_");
}

static int append_path_segment(char* path, size_t path_size,
                               const char* seg) {
    size_t len = strlen(path);
    size_t slen = strlen(seg);
    int need_slash = (len > 0 && path[len - 1] != '/');
    if (len + (need_slash ? 1 : 0) + slen + 1 > path_size) return 0;
    if (need_slash) path[len++] = '/';
    memcpy(path + len, seg, slen + 1);
    return 1;
}

static int remote_asset_cache_path(const char* url,
                                   char* out, size_t out_size) {
    if (!remote_asset_cache_root(out, out_size)) return 0;

    const char* p = strstr(url, "://");
    p = p ? p + 3 : url;
    int wrote_segment = 0;
    while (*p && *p != '?' && *p != '#') {
        while (*p == '/') p++;
        const char* begin = p;
        while (*p && *p != '/' && *p != '?' && *p != '#') p++;
        if (p == begin) continue;

        char seg[256];
        sanitize_cache_segment(begin, (size_t)(p - begin), seg, sizeof(seg));
        if (!append_path_segment(out, out_size, seg)) return 0;
        wrote_segment = 1;
    }
    return wrote_segment;
}

static int mkdir_p_for_file(const char* file_path) {
    char dir[1024];
    if (!file_path || strlen(file_path) >= sizeof(dir)) return 0;
    snprintf(dir, sizeof(dir), "%s", file_path);
    path_parent(dir);
    if (!dir[0] || dir_exists(dir)) return 1;

    for (char* p = dir + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (!dir_exists(dir) && mkdir(dir, 0755) != 0) {
            *p = '/';
            return 0;
        }
        *p = '/';
    }
    return dir_exists(dir) || mkdir(dir, 0755) == 0;
}

static int shell_quote(const char* s, char* out, size_t out_size) {
    size_t n = 0;
#define APPEND_CH(ch) do { if (n + 1 >= out_size) return 0; out[n++] = (ch); } while (0)
    APPEND_CH('\'');
    for (const char* p = s; p && *p; p++) {
        if (*p == '\'') {
            const char* esc = "'\\''";
            for (const char* q = esc; *q; q++) APPEND_CH(*q);
        } else {
            APPEND_CH(*p);
        }
    }
    APPEND_CH('\'');
    out[n] = '\0';
#undef APPEND_CH
    return 1;
}

static int resolve_remote_asset_to_cache(const char* url,
                                         char* out, size_t out_size) {
    char cached[1024];
    if (!remote_asset_cache_path(url, cached, sizeof(cached))) return 0;
    if (file_nonempty(cached)) {
        snprintf(out, out_size, "%s", cached);
        return 1;
    }

    if (!remote_asset_fetch_allowed(url)) return 0;
    if (!mkdir_p_for_file(cached)) return 0;

    char tmp[1024];
    if (snprintf(tmp, sizeof(tmp), "%s.tmp", cached) < 0 ||
        strlen(tmp) >= sizeof(tmp))
        return 0;
    remove(tmp);

    char qtmp[2048], qurl[2048];
    if (!shell_quote(tmp, qtmp, sizeof(qtmp)) ||
        !shell_quote(url, qurl, sizeof(qurl)))
        return 0;

    char cmd[8192];
    int n = snprintf(cmd, sizeof(cmd),
        "(command -v curl >/dev/null 2>&1 && "
        "curl --fail --silent --show-error --connect-timeout 10 --max-time 60 "
        "--max-filesize 8388608 "
        "-o %s %s) || "
        "(command -v wget >/dev/null 2>&1 && "
        "wget -q -T 30 --max-redirect=0 -Q 8m -O %s %s)",
        qtmp, qurl, qtmp, qurl);
    if (n < 0 || (size_t)n >= sizeof(cmd)) return 0;

    int rc = system(cmd);
    if (rc == 0 && file_nonempty(tmp)) {
        if (rename(tmp, cached) != 0) {
            remove(cached);
            if (rename(tmp, cached) != 0) {
                remove(tmp);
                return 0;
            }
        }
        if (file_nonempty(cached)) {
            if (getenv("NUSD_MAT_DIAG"))
                fprintf(stderr, "material: cached remote asset: %s -> %s\n",
                        url, cached);
            snprintf(out, out_size, "%s", cached);
            return 1;
        }
    }

    remove(tmp);
    return 0;
}

static void path_parent(char* path) {
    if (!path || !path[0]) return;
    size_t n = strlen(path);
    while (n > 1 && path[n - 1] == '/') path[--n] = '\0';
    char* slash = strrchr(path, '/');
    if (!slash) {
        path[0] = '\0';
        return;
    }
    if (slash == path) {
        path[1] = '\0';
        return;
    }
    *slash = '\0';
}

static void push_search_root(char roots[][1024], int* nroots, const char* path) {
    if (!path || !path[0] || !dir_exists(path)) return;
    for (int i = 0; i < *nroots; i++) {
        if (strcmp(roots[i], path) == 0) return;
    }
    if (*nroots >= 32) return;
    snprintf(roots[*nroots], 1024, "%s", path);
    (*nroots)++;
}

static int is_broad_search_root(const char* path)
{
    if (!path || !path[0]) return 1;
    if (strcmp(path, "/") == 0) return 1;
    if (strcmp(path, "/Users") == 0 || strcmp(path, "/home") == 0)
        return 1;
    const char* home = getenv("HOME");
    if (home && home[0] && strcmp(path, home) == 0)
        return 1;
    return 0;
}

static void add_search_path_list(const char* list,
                                 char roots[][1024],
                                 int* nroots,
                                 int materialx_resources)
{
    if (!list || !list[0]) return;

    const char* p = list;
    while (*p) {
        const char* sep = strchr(p, ':');
        size_t len = sep ? (size_t)(sep - p) : strlen(p);
        if (len > 0) {
            char entry[1024];
            if (len >= sizeof(entry)) len = sizeof(entry) - 1;
            memcpy(entry, p, len);
            entry[len] = '\0';

            push_search_root(roots, nroots, entry);
            if (materialx_resources) {
                char candidate[1024];
                int n = snprintf(candidate, sizeof(candidate),
                                 "%s/resources", entry);
                if (n >= 0 && (size_t)n < sizeof(candidate))
                    push_search_root(roots, nroots, candidate);

                char parent[1024];
                snprintf(parent, sizeof(parent), "%s", entry);
                path_parent(parent);
                if (parent[0]) {
                    push_search_root(roots, nroots, parent);
                    n = snprintf(candidate, sizeof(candidate),
                                 "%s/resources", parent);
                    if (n >= 0 && (size_t)n < sizeof(candidate))
                        push_search_root(roots, nroots, candidate);
                }
            }
        }
        if (!sep) break;
        p = sep + 1;
    }
}

static void collect_env_asset_search_roots(char roots[][1024], int* nroots)
{
    add_search_path_list(getenv("NUSD_ASSET_ROOTS"), roots, nroots, 0);
}

static void collect_materialx_resource_roots(char roots[][1024], int* nroots)
{
    /* Keep this in step with Vulkan's resolver. MaterialX standard-library
     * samples commonly refer to textures as resource-relative filenames;
     * checking the search entry, its resources/ child, and the parent variants
     * lets OpenGL resolve the same .mtlx graphs as the Vulkan backend. */
    add_search_path_list(getenv("MATERIALX_SEARCH_PATH"), roots, nroots, 1);
#ifdef MATERIALX_SEARCH_PATH
    add_search_path_list(MATERIALX_SEARCH_PATH, roots, nroots, 1);
#endif
}

static int collect_texture_search_roots(const char* scene_dir,
                                        char roots[][1024], int max_roots) {
    (void)max_roots;
    int nroots = 0;
    collect_env_asset_search_roots(roots, &nroots);

    if (scene_dir && scene_dir[0]) {
        char cur[1024];
        snprintf(cur, sizeof(cur), "%s", scene_dir);
        for (int up = 0; up <= 4 && cur[0]; up++) {
            if (is_broad_search_root(cur)) break;
            char candidate[1024];
            push_search_root(roots, &nroots, cur);
            int n = snprintf(candidate, sizeof(candidate), "%s/Library", cur);
            if (n >= 0 && (size_t)n < sizeof(candidate))
                push_search_root(roots, &nroots, candidate);
            if (!env_enabled("NUSD_DISABLE_DSX_ASSET_RESCUE")) {
                n = snprintf(candidate, sizeof(candidate),
                             "%s/DSX_BP/Library", cur);
                if (n >= 0 && (size_t)n < sizeof(candidate))
                    push_search_root(roots, &nroots, candidate);
            }
            path_parent(cur);
        }
    }

    collect_materialx_resource_roots(roots, &nroots);
    return nroots;
}

static int try_under_root(const char* root, const char* tex_path,
                          char* out, size_t out_size) {
    if (!root || !root[0] || !tex_path || !tex_path[0]) return 0;
    const char* rel = tex_path;
    if (rel[0] == '.' && rel[1] == '/') rel += 2;
    if (rel[0] == '/') return 0;

    char candidate[1024];
    int n = snprintf(candidate, sizeof(candidate), "%s/%s", root, rel);
    if (n < 0 || (size_t)n >= sizeof(candidate)) return 0;
    if (!file_exists(candidate)) return 0;
    if (strlen(candidate) >= out_size) return 0;
    snprintf(out, out_size, "%s", candidate);
    return 1;
}

static int try_asset_suffix_under_roots(const char* tex_path,
                                        char roots[][1024], int nroots,
                                        char* out, size_t out_size) {
    if (!tex_path || !tex_path[0]) return 0;

    char cleaned[1024];
    snprintf(cleaned, sizeof(cleaned), "%s", tex_path);
    size_t clen = strlen(cleaned);
    if (clen >= 2 && cleaned[0] == '@' && cleaned[clen - 1] == '@') {
        memmove(cleaned, cleaned + 1, clen - 2);
        cleaned[clen - 2] = '\0';
    }
    if (cleaned[0] == '.' && cleaned[1] == '/')
        memmove(cleaned, cleaned + 2, strlen(cleaned + 2) + 1);

    for (int r = 0; r < nroots; r++) {
        if (try_under_root(roots[r], cleaned, out, out_size)) return 1;
    }

    char stripped[1024];
    snprintf(stripped, sizeof(stripped), "%s", cleaned);
    while (strncmp(stripped, "../", 3) == 0)
        memmove(stripped, stripped + 3, strlen(stripped + 3) + 1);
    if (strcmp(stripped, cleaned) != 0) {
        for (int r = 0; r < nroots; r++) {
            if (try_under_root(roots[r], stripped, out, out_size)) return 1;
        }
    }

    if (stripped[0] != '/') {
        const char* materialx_example_roots[] = {
            "Materials/Examples/StandardSurface/",
            "Materials/Examples/OpenPbr/",
            "Materials/Examples/UsdPreviewSurface/",
            "Materials/Examples/GltfPbr/",
            "Materials/Examples/DisneyPrincipled/",
            "Materials/Examples/SimpleHair/",
            "Images/",
        };
        for (size_t m = 0; m < sizeof(materialx_example_roots) /
                               sizeof(materialx_example_roots[0]); m++) {
            char suffix[1024];
            int n = snprintf(suffix, sizeof(suffix), "%s%s",
                             materialx_example_roots[m], stripped);
            if (n < 0 || (size_t)n >= sizeof(suffix)) continue;
            for (int r = 0; r < nroots; r++) {
                if (try_under_root(roots[r], suffix, out, out_size)) return 1;
            }
        }
    }

    const char* markers[] = {
        "/Library/Materials/",
        "/resources/Materials/",
        "/Materials/",
        "/resources/Images/",
        "/textures/",
        "/Textures/",
        "/omniverse-content-production.s3.us-west-2.amazonaws.com/",
    };
    for (size_t m = 0; m < sizeof(markers) / sizeof(markers[0]); m++) {
        const char* hit = strstr(cleaned, markers[m]);
        if (!hit) continue;
        const char* suffix = hit + 1;
        for (int r = 0; r < nroots; r++) {
            if (try_under_root(roots[r], suffix, out, out_size)) return 1;
        }
    }
    return 0;
}

static int path_has_usd_layer_ext(const char* path) {
    const char* ext = strrchr(path ? path : "", '.');
    if (!ext) return 0;
    return strcasecmp(ext, ".usd") == 0 ||
           strcasecmp(ext, ".usda") == 0 ||
           strcasecmp(ext, ".usdc") == 0;
}

static void trim_asset_token(char* asset) {
    if (!asset) return;
    char* angle = strchr(asset, '<');
    if (angle) *angle = '\0';
    char* start = asset;
    while (*start && isspace((unsigned char)*start)) start++;
    if (start != asset) memmove(asset, start, strlen(start) + 1);
    size_t n = strlen(asset);
    while (n > 0 && isspace((unsigned char)asset[n - 1]))
        asset[--n] = '\0';
}

static int seen_layer_path(char seen[][1024], int nseen, const char* path) {
    for (int i = 0; i < nseen; i++) {
        if (strcmp(seen[i], path) == 0) return 1;
    }
    return 0;
}

static void collect_layer_asset_roots_recursive(const char* layer_path,
                                                int depth,
                                                char roots[][1024],
                                                int* nroots,
                                                char seen[][1024],
                                                int* nseen)
{
    if (!layer_path || !layer_path[0] || depth > 6) return;

    char layer_abs[PATH_MAX];
    char* rp = realpath(layer_path, layer_abs);
    if (!rp || !file_exists(layer_abs)) return;
    if (strlen(layer_abs) >= 1024) return;
    if (seen_layer_path(seen, *nseen, layer_abs)) return;
    if (*nseen < 64)
        snprintf(seen[(*nseen)++], 1024, "%s", layer_abs);

    char layer_dir[1024];
    snprintf(layer_dir, sizeof(layer_dir), "%s", layer_abs);
    path_parent(layer_dir);
    push_search_root(roots, nroots, layer_dir);

    if (depth >= 6 || !path_has_usd_layer_ext(layer_abs)) return;

    FILE* fp = fopen(layer_abs, "rb");
    if (!fp) return;

    /* Size-check the file we actually opened (fstat on the fd, not stat on
     * the name) so a concurrent rename/symlink swap cannot bypass the cap. */
    struct stat st;
    if (fstat(fileno(fp), &st) != 0 || st.st_size > 32L * 1024L * 1024L) {
        fclose(fp);
        return;
    }

    char line[4096];
    while (fgets(line, sizeof(line), fp)) {
        const char* pos = line;
        while ((pos = strchr(pos, '@')) != NULL) {
            const char* end = strchr(pos + 1, '@');
            if (!end) break;
            const char* begin = pos + 1;
            size_t len = (size_t)(end - begin);
            pos = end + 1;
            if (len == 0 || len >= 1024) continue;

            char asset[1024];
            memcpy(asset, begin, len);
            asset[len] = '\0';
            trim_asset_token(asset);
            if (!asset[0] || strncmp(asset, "anon:", 5) == 0 ||
                strstr(asset, "://"))
                continue;

            char asset_abs[1024];
            if (asset[0] == '/' || asset[0] == '\\') {
                snprintf(asset_abs, sizeof(asset_abs), "%s", asset);
            } else {
                int n = snprintf(asset_abs, sizeof(asset_abs), "%s/%s",
                                 layer_dir, asset);
                if (n < 0 || (size_t)n >= sizeof(asset_abs)) continue;
            }

            char asset_real[PATH_MAX];
            if (!realpath(asset_abs, asset_real) || !file_exists(asset_real))
                continue;
            if (strlen(asset_real) >= 1024) continue;

            char asset_dir[1024];
            snprintf(asset_dir, sizeof(asset_dir), "%s", asset_real);
            path_parent(asset_dir);
            push_search_root(roots, nroots, asset_dir);
            if (path_has_usd_layer_ext(asset_real))
                collect_layer_asset_roots_recursive(asset_real, depth + 1,
                                                    roots, nroots,
                                                    seen, nseen);
        }
    }
    fclose(fp);
}

static int collect_stage_asset_roots(void* stage,
                                     char roots[][1024], int max_roots)
{
    if (!stage || max_roots <= 0) return 0;
    const char* root_path =
        nanousd_stage_get_root_layer_path((NanousdStage)stage);
    if (!root_path || !root_path[0]) return 0;

    static char cache_key[1024];
    static char cache_roots[32][1024];
    static int cache_nroots = 0;

    if (strcmp(cache_key, root_path) != 0) {
        char seen[64][1024];
        int nseen = 0;
        cache_nroots = 0;
        cache_key[0] = '\0';
        collect_layer_asset_roots_recursive(root_path, 0, cache_roots,
                                            &cache_nroots, seen, &nseen);
        snprintf(cache_key, sizeof(cache_key), "%s", root_path);
        if (cache_nroots > 0) {
            fprintf(stderr,
                    "material: %d asset anchor dir(s) collected from %s\n",
                    cache_nroots, root_path);
        }
    }

    int ncopy = cache_nroots < max_roots ? cache_nroots : max_roots;
    for (int i = 0; i < ncopy; i++)
        snprintf(roots[i], 1024, "%s", cache_roots[i]);
    return ncopy;
}

/* ---- MDL file-content cache ----
 * Warehouse Materials all author `info:mdl:sourceAsset = @../Materials/MI_*.mdl@`,
 * but the same .mdl is referenced by many materials (e.g. all Barrel_* prims
 * point at MI_BarelPlasticA_02.mdl). Re-opening + slurping every .mdl per
 * material costs hundreds of ms. Cache each file's content keyed by
 * absolute path so repeats are free. Limit and clear at the end of
 * materials_load. */
typedef struct { char path[1024]; char* content; long size; } MdlEntry;
static MdlEntry* g_mdl_cache = NULL;
static int  g_mdl_cache_n = 0;
static int  g_mdl_cache_cap = 0;

static const char* mdl_cache_get(const char* abs_path, long* out_size) {
    if (!abs_path || !abs_path[0]) return NULL;
    for (int i = 0; i < g_mdl_cache_n; i++) {
        if (strcmp(g_mdl_cache[i].path, abs_path) == 0) {
            if (out_size) *out_size = g_mdl_cache[i].size;
            return g_mdl_cache[i].content;
        }
    }
    /* Miss — load. */
    FILE* fp = fopen(abs_path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0 || sz >= 1024 * 1024) { fclose(fp); return NULL; }
    char* buf = (char*)malloc((size_t)sz + 1);
    if (!buf) { fclose(fp); return NULL; }
    if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) {
        free(buf); fclose(fp); return NULL;
    }
    buf[sz] = '\0';
    fclose(fp);
    if (g_mdl_cache_n >= g_mdl_cache_cap) {
        int newcap = g_mdl_cache_cap ? g_mdl_cache_cap * 2 : 64;
        MdlEntry* nu = (MdlEntry*)realloc(
            g_mdl_cache, (size_t)newcap * sizeof(MdlEntry));
        if (!nu) { free(buf); return NULL; }
        g_mdl_cache = nu;
        g_mdl_cache_cap = newcap;
    }
    MdlEntry* e = &g_mdl_cache[g_mdl_cache_n++];
    snprintf(e->path, sizeof(e->path), "%s", abs_path);
    e->content = buf;
    e->size = sz;
    if (out_size) *out_size = sz;
    return buf;
}

static void mdl_cache_clear(void) {
    for (int i = 0; i < g_mdl_cache_n; i++) free(g_mdl_cache[i].content);
    free(g_mdl_cache);
    g_mdl_cache = NULL;
    g_mdl_cache_n = 0;
    g_mdl_cache_cap = 0;
}

/* Max texture resolution (GLES lightweight target) */
static int g_max_tex_size = 512;

void materials_set_max_tex_size(int size) {
    if (size > 0) g_max_tex_size = size;
}

/* ---- Texture loading ---- */

static int is_package_identifier_path(const char* path)
{
    size_t len;
    if (!path || !path[0]) return 0;
    len = strlen(path);
    return strchr(path, '[') != NULL && len > 0 && path[len - 1] == ']';
}

static int copy_path(char* out, size_t out_size, const char* path)
{
    size_t len;
    if (!out || out_size == 0 || !path) return 0;
    len = strlen(path);
    if (len + 1 > out_size) {
        out[0] = '\0';
        return 0;
    }
    memcpy(out, path, len + 1);
    return 1;
}

static unsigned char* load_texture(const char* path, int* w, int* h,
                                    void* stage)
{
    (void)stage;

    int channels;
    unsigned char* pixels = NULL;

    if (is_package_identifier_path(path)) {
        unsigned char* bytes = NULL;
        size_t byte_size = 0;
        if (nanousd_read_asset_bytes(path, &bytes, &byte_size) &&
            bytes && byte_size > 0) {
            pixels = stbi_load_from_memory(bytes, (int)byte_size,
                                           w, h, &channels, 4);
            if (pixels) {
                fprintf(stderr, "material: loaded package texture %dx%d: %s\n",
                        *w, *h, path);
            }
        }
        if (bytes) nanousd_free_bytes(bytes);
    }

    if (!pixels)
        pixels = stbi_load(path, w, h, &channels, 4); /* Force RGBA */
    if (!pixels) {
        fprintf(stderr, "material: failed to load texture: %s\n", path);
        return NULL;
    }

    /* Cap resolution for low VRAM */
    if (*w > g_max_tex_size || *h > g_max_tex_size) {
        /* Simple box downsample to fit within g_max_tex_size */
        int nw = *w, nh = *h;
        while (nw > g_max_tex_size || nh > g_max_tex_size) {
            nw = (nw + 1) / 2;
            nh = (nh + 1) / 2;
        }

        unsigned char* resized = (unsigned char*)malloc((size_t)nw * (size_t)nh * 4);
        if (!resized) {
            stbi_image_free(pixels);
            return NULL;
        }

        float sx = (float)*w / (float)nw;
        float sy = (float)*h / (float)nh;

        for (int y = 0; y < nh; y++) {
            for (int x = 0; x < nw; x++) {
                int ox = (int)(x * sx);
                int oy = (int)(y * sy);
                if (ox >= *w) ox = *w - 1;
                if (oy >= *h) oy = *h - 1;
                memcpy(resized + (y * nw + x) * 4,
                       pixels + (oy * *w + ox) * 4, 4);
            }
        }

        stbi_image_free(pixels);
        pixels = resized;
        *w = nw;
        *h = nh;
    }

    return pixels;
}

/* ---- Path resolution ---- */

static void resolve_texture_path_stage(const char* scene_dir, const char* tex_path,
                                       void* stage, char* out, size_t out_size)
{
    if (!tex_path || !tex_path[0]) {
        out[0] = '\0';
        return;
    }
    if (is_http_url(tex_path)) {
        if (!resolve_remote_asset_to_cache(tex_path, out, out_size))
            snprintf(out, out_size, "%s", tex_path);
        return;
    }
    if (is_package_identifier_path(tex_path)) {
        copy_path(out, out_size, tex_path);
        return;
    }
    if (stage) {
        char package_loc[2048];
        if (nanousd_stage_resolve_asset_path((NanousdStage)stage, tex_path,
                                             package_loc,
                                             sizeof(package_loc)) &&
            is_package_identifier_path(package_loc)) {
            copy_path(out, out_size, package_loc);
            return;
        }
    }

    char roots[32][1024];
    int nroots = 0;
    char stage_roots[32][1024];
    int nstage = collect_stage_asset_roots(stage, stage_roots, 32);
    for (int i = 0; i < nstage; i++)
        push_search_root(roots, &nroots, stage_roots[i]);
    char scene_roots[32][1024];
    int nscene = collect_texture_search_roots(scene_dir, scene_roots, 32);
    for (int i = 0; i < nscene; i++)
        push_search_root(roots, &nroots, scene_roots[i]);
    const char* base_dir = (scene_dir && scene_dir[0]) ? scene_dir : ".";

    /* Absolute path. DSX and collected Omniverse bundles may carry stale
     * absolute paths from the authoring workstation, so only return early
     * when the path exists in this checkout. */
    if (tex_path[0] == '/') {
        snprintf(out, out_size, "%s", tex_path);
        if (file_exists(out)) return;
        if (try_asset_suffix_under_roots(tex_path, roots, nroots, out, out_size))
            return;
    } else {
        /* Relative path — resolve against scene directory. POSIX fopen will
         * follow embedded `..` segments, so we don't normalize here. */
        int n = snprintf(out, out_size, "%s/%s", base_dir, tex_path);
        if (n < 0 || (size_t)n >= out_size) {
            fprintf(stderr, "material: resolved texture path truncated "
                            "(scene_dir=%s tex=%s)\n", base_dir, tex_path);
            out[0] = '\0';
            return;
        }
        if (file_exists(out)) return;

        for (int r = 0; r < nroots; r++) {
            if (try_under_root(roots[r], tex_path, out, out_size)) return;
        }
        if (try_asset_suffix_under_roots(tex_path, roots, nroots, out, out_size))
            return;
    }

    /* Hydra resolves asset paths against the *anchor layer* — the layer
     * that authored the value. nanousdapi doesn't expose per-attribute
     * layer-of-origin, so when our literal resolution misses (common with
     * referenced prop USDs that author `../Materials/...` paths, and DSX
     * top layers where Assembly sits beside Library), fall back to a basename
     * lookup in recursive indexes of likely package roots. */
    const char* slash = strrchr(tex_path, '/');
    const char* base = slash ? slash + 1 : tex_path;
    file_index_ensure_many(roots, nroots);
    const char* hit = file_index_lookup(base);
    if (hit && strlen(hit) < out_size) {
        snprintf(out, out_size, "%s", hit);
        return;
    }
}

static int ptex_asset_hint(const char* path)
{
    if (!path || !path[0]) return 0;
    char lower[1024];
    size_t n = strlen(path);
    if (n >= sizeof(lower)) n = sizeof(lower) - 1;
    for (size_t i = 0; i < n; i++)
        lower[i] = (char)tolower((unsigned char)path[i]);
    lower[n] = '\0';
    return strstr(lower, ".ptx") != NULL ||
           strstr(lower, ".ptex") != NULL ||
           strstr(lower, "ptex") != NULL;
}

static int material_try_fast_resolve_ptex_asset(const char* asset,
                                                const char* scene_dir,
                                                char out[512])
{
    if (!asset || !asset[0] || !out) return 0;
    out[0] = '\0';

    char cleaned[1024];
    snprintf(cleaned, sizeof(cleaned), "%s", asset);
    trim_asset_token(cleaned);
    size_t clen = strlen(cleaned);
    if (clen >= 2 && cleaned[0] == '@' && cleaned[clen - 1] == '@') {
        memmove(cleaned, cleaned + 1, clen - 2);
        cleaned[clen - 2] = '\0';
    }
    while (cleaned[0] == '.' && cleaned[1] == '/')
        memmove(cleaned, cleaned + 2, strlen(cleaned + 2) + 1);

    if (cleaned[0] == '/') {
        if (file_exists(cleaned)) {
            return copy_path(out, 512, cleaned);
        }
        return 0;
    }

    const char* base_dir = (scene_dir && scene_dir[0]) ? scene_dir : ".";
    char candidate[1024];
    int n = snprintf(candidate, sizeof(candidate), "%s/%s", base_dir, cleaned);
    if (n >= 0 && (size_t)n < sizeof(candidate) && file_exists(candidate)) {
        return copy_path(out, 512, candidate);
    }

    char stripped[1024];
    snprintf(stripped, sizeof(stripped), "%s", cleaned);
    while (strncmp(stripped, "../", 3) == 0)
        memmove(stripped, stripped + 3, strlen(stripped + 3) + 1);

    char cur[1024];
    snprintf(cur, sizeof(cur), "%s", base_dir);
    for (int up = 0; up < 6 && cur[0]; up++) {
        n = snprintf(candidate, sizeof(candidate), "%s/%s", cur, stripped);
        if (n >= 0 && (size_t)n < sizeof(candidate) && file_exists(candidate)) {
            return copy_path(out, 512, candidate);
        }
        path_parent(cur);
    }

    const char* markers[] = { "textures/", "Textures/" };
    for (size_t m = 0; m < sizeof(markers) / sizeof(markers[0]); m++) {
        const char* suffix = strstr(cleaned, markers[m]);
        if (!suffix) suffix = strstr(stripped, markers[m]);
        if (!suffix) continue;
        snprintf(cur, sizeof(cur), "%s", base_dir);
        for (int up = 0; up < 6 && cur[0]; up++) {
            n = snprintf(candidate, sizeof(candidate), "%s/%s", cur, suffix);
            if (n >= 0 && (size_t)n < sizeof(candidate) && file_exists(candidate)) {
                return copy_path(out, 512, candidate);
            }
            path_parent(cur);
        }
    }

    return 0;
}

static int material_try_resolve_ptex_asset(const char* asset,
                                           const char* scene_dir,
                                           void* stage,
                                           char out[512])
{
    if (!asset || !asset[0] || !out || !ptex_asset_hint(asset))
        return 0;
    if (material_try_fast_resolve_ptex_asset(asset, scene_dir, out))
        return 1;
    if (!env_enabled("NUSD_OPENGL_PTEX_FULL_RESOLVE"))
        return 0;
    char resolved[1024];
    resolved[0] = '\0';
    resolve_texture_path_stage(scene_dir, asset, stage, resolved,
                               sizeof(resolved));
    if (!resolved[0] || is_package_identifier_path(resolved) ||
        !file_exists(resolved))
        return 0;
    snprintf(out, 512, "%s", resolved);
    return 1;
}

#ifdef NUSD_HAVE_PTEX
typedef struct {
    char path[512];
    float rgb[3];
    int ok;
} PtexAverageCacheEntry;

static PtexAverageCacheEntry* g_ptex_average_cache = NULL;
static int g_ptex_average_cache_n = 0;
static int g_ptex_average_cache_cap = 0;
static int g_ptex_average_cache_hits = 0;
static int g_ptex_average_cache_loads = 0;
static int g_ptex_average_cache_failures = 0;
static int g_ptex_average_cache_skipped = 0;

static int ptex_average_env_int(const char* name, int fallback)
{
    const char* s = getenv(name);
    if (s && s[0]) {
        char* end = NULL;
        long v = strtol(s, &end, 10);
        if (end != s && v >= -1 && v <= 1000000)
            return (int)v;
    }
    return fallback;
}

static int ptex_average_sample_count(void)
{
    int samples = ptex_average_env_int("NUSD_OPENGL_PTEX_AVERAGE_SAMPLES", 1);
    return samples > 0 ? samples : 1;
}

static int ptex_average_max_unique(void)
{
    int max_unique = ptex_average_env_int(
        "NUSD_OPENGL_PTEX_AVERAGE_MAX_UNIQUE", 0);
    return max_unique < 0 ? INT_MAX : max_unique;
}

static void ptex_average_cache_clear(void)
{
    free(g_ptex_average_cache);
    g_ptex_average_cache = NULL;
    g_ptex_average_cache_n = 0;
    g_ptex_average_cache_cap = 0;
    g_ptex_average_cache_hits = 0;
    g_ptex_average_cache_loads = 0;
    g_ptex_average_cache_failures = 0;
    g_ptex_average_cache_skipped = 0;
}

static void ptex_average_cache_log(void)
{
    if (g_ptex_average_cache_n == 0 && g_ptex_average_cache_skipped == 0)
        return;
    fprintf(stderr,
            "material: Ptex average cache %d unique, %d loads, %d hits, "
            "%d failed, %d skipped (samples=%d, max_unique=%d)\n",
            g_ptex_average_cache_n, g_ptex_average_cache_loads,
            g_ptex_average_cache_hits, g_ptex_average_cache_failures,
            g_ptex_average_cache_skipped, ptex_average_sample_count(),
            ptex_average_max_unique());
}

static int ptex_average_cache_read(const char* path, float out_rgb[3])
{
    if (!path || !path[0] || !out_rgb) return 0;
    for (int i = 0; i < g_ptex_average_cache_n; i++) {
        if (strcmp(g_ptex_average_cache[i].path, path) == 0) {
            g_ptex_average_cache_hits++;
            if (!g_ptex_average_cache[i].ok)
                return 0;
            out_rgb[0] = g_ptex_average_cache[i].rgb[0];
            out_rgb[1] = g_ptex_average_cache[i].rgb[1];
            out_rgb[2] = g_ptex_average_cache[i].rgb[2];
            return 1;
        }
    }

    int max_unique = ptex_average_max_unique();
    if (g_ptex_average_cache_n >= max_unique) {
        g_ptex_average_cache_skipped++;
        return 0;
    }

    if (g_ptex_average_cache_n >= g_ptex_average_cache_cap) {
        int newcap = g_ptex_average_cache_cap ?
            g_ptex_average_cache_cap * 2 : 128;
        PtexAverageCacheEntry* nu = (PtexAverageCacheEntry*)realloc(
            g_ptex_average_cache,
            (size_t)newcap * sizeof(PtexAverageCacheEntry));
        if (!nu) {
            g_ptex_average_cache_skipped++;
            return 0;
        }
        g_ptex_average_cache = nu;
        g_ptex_average_cache_cap = newcap;
    }

    PtexAverageCacheEntry* e = &g_ptex_average_cache[g_ptex_average_cache_n++];
    memset(e, 0, sizeof(*e));
    snprintf(e->path, sizeof(e->path), "%s", path);
    g_ptex_average_cache_loads++;
    e->ok = nusd_ptex_sample_average_color(path, e->rgb,
                                           ptex_average_sample_count());
    if (!e->ok) {
        g_ptex_average_cache_failures++;
        return 0;
    }
    out_rgb[0] = e->rgb[0];
    out_rgb[1] = e->rgb[1];
    out_rgb[2] = e->rgb[2];
    return 1;
}
#endif

static int material_read_asset_string(NanousdPrim prim, const char* attr,
                                      char* out, size_t out_size)
{
    if (!prim || !attr || !out || out_size == 0) return 0;
    out[0] = '\0';
    int ok = 0;
    const char* value = nanousd_attribasset(prim, attr, &ok);
    if (!ok || !value || !value[0])
        value = nanousd_attribs(prim, attr, &ok);
    if (!ok || !value || !value[0])
        return 0;
    snprintf(out, out_size, "%s", value);
    return 1;
}

static int read_material_ptex_color_path(NanousdPrim material_prim,
                                         const char* scene_dir,
                                         void* stage,
                                         char out[512])
{
    if (!out) return 0;
    out[0] = '\0';
    if (!material_prim) return 0;

    const char* direct_attrs[] = {
        "inputs:surfaceMap",
        "inputs:baseColor",
        "inputs:diffuseColor",
    };
    for (int i = 0; i < 3; i++) {
        char asset[1024];
        if (material_read_asset_string(material_prim, direct_attrs[i],
                                       asset, sizeof(asset)) &&
            material_try_resolve_ptex_asset(asset, scene_dir, stage, out)) {
            return 1;
        }
    }

    int nchildren = nanousd_nchildren(material_prim);
    for (int i = 0; i < nchildren; i++) {
        NanousdPrim child = nanousd_child(material_prim, i);
        if (!child) continue;
        const char* child_attrs[] = {
            "inputs:surfaceMap",
            "inputs:file",
            "inputs:filename",
            "inputs:texture",
            "inputs:baseColor",
            "inputs:diffuseColor",
        };
        for (int a = 0; a < 6; a++) {
            char asset[1024];
            if (material_read_asset_string(child, child_attrs[a], asset,
                                           sizeof(asset)) &&
                material_try_resolve_ptex_asset(asset, scene_dir, stage, out)) {
                nanousd_freeprim(child);
                return 1;
            }
        }
        nanousd_freeprim(child);
    }
    return 0;
}

static void material_apply_ptex_average(SceneMaterial* mat,
                                        NanousdPrim material_prim,
                                        const char* scene_dir,
                                        void* stage)
{
    if (!mat) return;
    mat->ptex_color_path[0] = '\0';
    mat->has_ptex_average_color = 0;
    if (!read_material_ptex_color_path(material_prim, scene_dir, stage,
                                       mat->ptex_color_path))
        return;

#ifdef NUSD_HAVE_PTEX
    float avg[3];
    if (ptex_average_cache_read(mat->ptex_color_path, avg)) {
        mat->ptex_average_color[0] = avg[0];
        mat->ptex_average_color[1] = avg[1];
        mat->ptex_average_color[2] = avg[2];
        mat->has_ptex_average_color = 1;
        mat->params.base_color[0] = avg[0];
        mat->params.base_color[1] = avg[1];
        mat->params.base_color[2] = avg[2];
        mat->params.use_vertex_color = 0;
    }
#endif
}

/* ---- UDIM atlas stitcher ---- */

/*
 * UDIM layout: tile 1001 + col + row*10
 *   row 0: 1001, 1002, 1003, ...
 *   row 1: 1011, 1012, 1013, ...
 * Loads all existing tiles and composites into one atlas texture.
 * Returns pixels (caller must free) or NULL if no tiles found.
 */
static unsigned char* load_udim_atlas(const char* pattern, const char* scene_dir,
                                       void* stage,
                                       int* out_w, int* out_h,
                                       int* out_cols, int* out_rows)
{
    *out_w = 0; *out_h = 0; *out_cols = 0; *out_rows = 0;

    /* Find <UDIM> in pattern */
    const char* udim_pos = strstr(pattern, "<UDIM>");
    if (!udim_pos) return NULL;

    size_t prefix_len = (size_t)(udim_pos - pattern);
    const char* suffix = udim_pos + 6; /* strlen("<UDIM>") */

    /* Scan for existing tiles in range 1001-1100 */
    typedef struct { int col, row; unsigned char* pixels; int w, h; } UdimTile;
    UdimTile tiles[100];
    int ntiles = 0;
    int max_col = 0, max_row = 0;
    unsigned char present[100] = {0};
    char lower_pattern[512];
    int lpi = 0;
    for (; pattern[lpi] && lpi < (int)sizeof(lower_pattern) - 1; lpi++) {
        char c = pattern[lpi];
        lower_pattern[lpi] = (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c;
    }
    lower_pattern[lpi] = '\0';
    int opacity_udim = (strstr(lower_pattern, "opacity") ||
                        strstr(lower_pattern, "alpha")) ? 1 : 0;

    for (int row = 0; row < 10; row++) {
        for (int col = 0; col < 10; col++) {
            int tile_id = 1001 + col + row * 10;
            /* Build tile path: prefix + tile_id + suffix */
            char tile_rel[1024];
            snprintf(tile_rel, sizeof(tile_rel), "%.*s%d%s",
                     (int)prefix_len, pattern, tile_id, suffix);

            char tile_path[1024];
            resolve_texture_path_stage(scene_dir, tile_rel, stage,
                                       tile_path, sizeof(tile_path));

            /* Try loading silently — missing tiles are expected */
            int w, h, ch;
            unsigned char* px = stbi_load(tile_path, &w, &h, &ch, 4);
            if (px) {
                /* Cap tile resolution for low VRAM */
                if (w > g_max_tex_size || h > g_max_tex_size) {
                    int nw = w, nh = h;
                    while (nw > g_max_tex_size || nh > g_max_tex_size) {
                        nw = (nw + 1) / 2;
                        nh = (nh + 1) / 2;
                    }
                    unsigned char* resized = (unsigned char*)malloc((size_t)nw * (size_t)nh * 4);
                    if (resized) {
                        float sx = (float)w / (float)nw;
                        float sy = (float)h / (float)nh;
                        for (int y = 0; y < nh; y++) {
                            for (int x = 0; x < nw; x++) {
                                int ox = (int)(x * sx);
                                int oy = (int)(y * sy);
                                if (ox >= w) ox = w - 1;
                                if (oy >= h) oy = h - 1;
                                memcpy(resized + (y * nw + x) * 4,
                                       px + (oy * w + ox) * 4, 4);
                            }
                        }
                        stbi_image_free(px);
                        px = resized;
                        w = nw;
                        h = nh;
                    }
                }
                tiles[ntiles].col = col;
                tiles[ntiles].row = row;
                tiles[ntiles].pixels = px;
                tiles[ntiles].w = w;
                tiles[ntiles].h = h;
                ntiles++;
                present[row * 10 + col] = 1;
                if (col > max_col) max_col = col;
                if (row > max_row) max_row = row;
            }
        }
    }

    if (ntiles == 0) {
        fprintf(stderr, "material: UDIM no tiles found for %s\n", pattern);
        return NULL;
    }

    /* Use first tile's dimensions as reference */
    int tile_w = tiles[0].w;
    int tile_h = tiles[0].h;
    int cols = max_col + 1;
    int rows = max_row + 1;
    int atlas_w = tile_w * cols;
    int atlas_h = tile_h * rows;

    unsigned char* atlas = (unsigned char*)calloc((size_t)atlas_w * (size_t)atlas_h, 4);
    if (!atlas) {
        for (int i = 0; i < ntiles; i++) free(tiles[i].pixels);
        return NULL;
    }

    /* Blit tiles into atlas. UDIM row 0 is at the bottom in UV space,
     * but our texture is top-down, so flip vertically. */
    for (int i = 0; i < ntiles; i++) {
        int dst_x = tiles[i].col * tile_w;
        int dst_y = (rows - 1 - tiles[i].row) * tile_h;
        int copy_w = tiles[i].w < tile_w ? tiles[i].w : tile_w;
        int copy_h = tiles[i].h < tile_h ? tiles[i].h : tile_h;

        for (int y = 0; y < copy_h; y++) {
            memcpy(atlas + ((size_t)(dst_y + y) * atlas_w + dst_x) * 4,
                   tiles[i].pixels + ((size_t)y * tiles[i].w) * 4,
                   (size_t)copy_w * 4);
        }
        free(tiles[i].pixels);
    }

    int filled_missing = 0;
    if (!opacity_udim) {
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                if (present[row * 10 + col]) continue;

                int best_col = -1;
                int best_row = -1;
                int best_dist = 1 << 30;
                for (int sr = 0; sr < rows; sr++) {
                    for (int sc = 0; sc < cols; sc++) {
                        if (!present[sr * 10 + sc]) continue;
                        int dist = abs(sc - col) + abs(sr - row);
                        if (dist < best_dist) {
                            best_dist = dist;
                            best_col = sc;
                            best_row = sr;
                        }
                    }
                }
                if (best_col < 0) continue;

                int dst_x = col * tile_w;
                int dst_y = (rows - 1 - row) * tile_h;
                int src_x = best_col * tile_w;
                int src_y = (rows - 1 - best_row) * tile_h;
                for (int y = 0; y < tile_h; y++) {
                    memcpy(atlas + ((size_t)(dst_y + y) * atlas_w + dst_x) * 4,
                           atlas + ((size_t)(src_y + y) * atlas_w + src_x) * 4,
                           (size_t)tile_w * 4);
                }
                filled_missing++;
            }
        }
    }

    fprintf(stderr, "material: UDIM atlas %dx%d (%d tiles, %dx%d grid, filled %d missing) from %s\n",
            atlas_w, atlas_h, ntiles, cols, rows, filled_missing, pattern);

    *out_w = atlas_w;
    *out_h = atlas_h;
    *out_cols = cols;
    *out_rows = rows;
    return atlas;
}

/* ---- Find texture index or add new ----
 * Exposed (non-static) so material_mtlx.cpp can reuse the same dedup
 * + UDIM + size-cap logic when binding textures from .mtlx files. */
/* ---- Parallel texture pre-decode (NUSD_PARALLEL_DECODE) ---------------------
 * Mirror of the Vulkan renderer's pre-pass (3918b3b): decode image textures up
 * front across OpenMP threads (load_texture takes an already-resolved path, so
 * it is thread-safe), then find_or_add_texture consumes them. Bit-identical:
 * same load_texture output, just produced in parallel. NUSD_PARALLEL_DECODE=0
 * disables. The serial walk's prim-index composition (the larger cost) is
 * core-bound and unaffected. */
typedef struct { char path[1024]; unsigned char* pixels; int w, h; int consumed; } GlPredecoded;
static GlPredecoded* g_gl_predecoded = NULL;
static int    g_gl_predecoded_n = 0;
static double g_gl_predecode_ms = 0.0;

int find_or_add_texture(MaterialCollection* mc, const char* scene_dir,
                                const char* tex_path, void* stage)
{
    if (!tex_path || !tex_path[0]) return -1;

    /* Check for UDIM pattern */
    int is_udim = (strstr(tex_path, "<UDIM>") != NULL);

    char resolved[1024];
    if (is_udim) {
        /* For UDIM, use the pattern as the key */
        snprintf(resolved, sizeof(resolved), "%s", tex_path);
    } else {
        resolve_texture_path_stage(scene_dir, tex_path, stage,
                                   resolved, sizeof(resolved));
    }
    if (!resolved[0]) return -1;

    /* Check if already loaded */
    for (int i = 0; i < mc->ntextures; i++) {
        if (strcmp(mc->textures[i].path, resolved) == 0)
            return i;
    }

    /* Load texture (UDIM atlas or regular) */
    int w = 0, h = 0;
    unsigned char* pixels = NULL;
    int udim_cols = 0, udim_rows = 0;

    /* Parallel pre-decode consume: this texture may have been decoded up front
     * by the OpenMP pre-pass (see materials_load) — take ownership, skip decode. */
    if (!is_udim && g_gl_predecoded) {
        for (int i = 0; i < g_gl_predecoded_n; i++) {
            if (!g_gl_predecoded[i].consumed && g_gl_predecoded[i].pixels &&
                strcmp(g_gl_predecoded[i].path, resolved) == 0) {
                pixels = g_gl_predecoded[i].pixels;
                w = g_gl_predecoded[i].w; h = g_gl_predecoded[i].h;
                g_gl_predecoded[i].consumed = 1;
                break;
            }
        }
    }

    if (!pixels) {
        if (is_udim) {
            pixels = load_udim_atlas(tex_path, scene_dir, stage,
                                     &w, &h, &udim_cols, &udim_rows);
        } else {
            pixels = load_texture(resolved, &w, &h, stage);
        }
    }
    if (!pixels) return -1;

    /* Grow array */
    int idx = mc->ntextures;
    MaterialTexture* new_texs = (MaterialTexture*)realloc(
        mc->textures, (size_t)(idx + 1) * sizeof(MaterialTexture));
    if (!new_texs) {
        free(pixels);
        return -1;
    }
    mc->textures = new_texs;

    memset(&mc->textures[idx], 0, sizeof(MaterialTexture));
    mc->textures[idx].pixels = pixels;
    mc->textures[idx].width = w;
    mc->textures[idx].height = h;
    mc->textures[idx].udim_cols = udim_cols;
    mc->textures[idx].udim_rows = udim_rows;
    snprintf(mc->textures[idx].path, sizeof(mc->textures[idx].path), "%s", resolved);
    mc->ntextures = idx + 1;

    return idx;
}

/* ---- Inline MaterialX USD graph extraction ---- */

static int mtlx_info_is_standard_surface(const char* info_id) {
    return info_id &&
           (strstr(info_id, "ND_standard_surface") ||
            strstr(info_id, "standard_surface_surfaceshader"));
}

static int mtlx_info_is_open_pbr(const char* info_id) {
    return info_id &&
           (strstr(info_id, "ND_open_pbr_surface") ||
            strstr(info_id, "open_pbr_surface_surfaceshader"));
}

static int split_connection_path(const char* conn,
                                 char* prim_path, size_t prim_size,
                                 char* prop_name, size_t prop_size)
{
    if (prim_path && prim_size) prim_path[0] = '\0';
    if (prop_name && prop_size) prop_name[0] = '\0';
    if (!conn || !conn[0] || !prim_path || prim_size == 0) return 0;

    const char* begin = conn;
    const char* end = conn + strlen(conn);
    if (*begin == '<') {
        begin++;
        const char* close = strchr(begin, '>');
        if (close) end = close;
    }

    const char* dot = NULL;
    for (const char* p = end; p > begin; --p) {
        if (*(p - 1) == '.') {
            dot = p - 1;
            break;
        }
    }
    if (!dot) {
        size_t n = (size_t)(end - begin);
        if (n >= prim_size) n = prim_size - 1;
        memcpy(prim_path, begin, n);
        prim_path[n] = '\0';
        return n > 0;
    }

    size_t pn = (size_t)(dot - begin);
    if (pn >= prim_size) pn = prim_size - 1;
    memcpy(prim_path, begin, pn);
    prim_path[pn] = '\0';

    if (prop_name && prop_size) {
        size_t an = (size_t)(end - dot - 1);
        if (an >= prop_size) an = prop_size - 1;
        memcpy(prop_name, dot + 1, an);
        prop_name[an] = '\0';
    }
    return prim_path[0] != '\0';
}

static int first_connection(NanousdPrim prim, const char* attr,
                            char* out, size_t out_size)
{
    if (!out || out_size == 0) return 0;
    out[0] = '\0';
    if (!prim || !attr) return 0;
    if (nanousd_nconnections(prim, attr) <= 0) return 0;
    const char* c = nanousd_connection(prim, attr, 0);
    if (!c || !c[0]) return 0;
    snprintf(out, out_size, "%s", c);
    return 1;
}

static int read_attr_float(NanousdPrim prim, const char* attr, float* out)
{
    if (!prim || !attr || !out) return 0;
    int ok = 0;
    float f = nanousd_attribf(prim, attr, &ok);
    if (ok) {
        *out = f;
        return 1;
    }
    int i = nanousd_attribi(prim, attr, &ok);
    if (ok) {
        *out = (float)i;
        return 1;
    }
    float v3[3];
    if (nanousd_attribv3f(prim, attr, v3)) {
        *out = v3[0];
        return 1;
    }
    float v4[4];
    if (nanousd_attribv4f(prim, attr, v4)) {
        *out = v4[0];
        return 1;
    }
    return 0;
}

static int read_attr_color3(NanousdPrim prim, const char* attr, float out[3])
{
    if (!prim || !attr || !out) return 0;
    float v3[3];
    if (nanousd_attribv3f(prim, attr, v3)) {
        out[0] = v3[0]; out[1] = v3[1]; out[2] = v3[2];
        return 1;
    }
    float v4[4];
    if (nanousd_attribv4f(prim, attr, v4)) {
        out[0] = v4[0]; out[1] = v4[1]; out[2] = v4[2];
        return 1;
    }
    float f = 0.0f;
    if (read_attr_float(prim, attr, &f)) {
        out[0] = f; out[1] = f; out[2] = f;
        return 1;
    }
    return 0;
}

static int read_connected_attr_float(NanousdStage stage,
                                     NanousdPrim shader_prim,
                                     NanousdPrim material_prim,
                                     const char* input_name,
                                     float* out)
{
    char attr[128];
    snprintf(attr, sizeof(attr), "inputs:%s", input_name);
    if (read_attr_float(shader_prim, attr, out)) return 1;

    char conn[1024], prim_path[512], prop[128];
    if (first_connection(shader_prim, attr, conn, sizeof(conn)) &&
        split_connection_path(conn, prim_path, sizeof(prim_path),
                              prop, sizeof(prop))) {
        NanousdPrim src = nanousd_primpath(stage, prim_path);
        if (src) {
            int ok = prop[0] ? read_attr_float(src, prop, out) : 0;
            nanousd_freeprim(src);
            if (ok) return 1;
        }
    }
    return read_attr_float(material_prim, attr, out);
}

static int read_connected_attr_color3(NanousdStage stage,
                                      NanousdPrim shader_prim,
                                      NanousdPrim material_prim,
                                      const char* input_name,
                                      float out[3])
{
    char attr[128];
    snprintf(attr, sizeof(attr), "inputs:%s", input_name);
    if (read_attr_color3(shader_prim, attr, out)) return 1;

    char conn[1024], prim_path[512], prop[128];
    if (first_connection(shader_prim, attr, conn, sizeof(conn)) &&
        split_connection_path(conn, prim_path, sizeof(prim_path),
                              prop, sizeof(prop))) {
        NanousdPrim src = nanousd_primpath(stage, prim_path);
        if (src) {
            int ok = prop[0] ? read_attr_color3(src, prop, out) : 0;
            nanousd_freeprim(src);
            if (ok) return 1;
        }
    }
    return read_attr_color3(material_prim, attr, out);
}

static int resolve_mtlx_texture_file_from_target(NanousdStage stage,
                                                 const char* target,
                                                 int depth,
                                                 char* out_file,
                                                 size_t out_size,
                                                 int* is_normal_map,
                                                 float out_uvtiling[2],
                                                 int* has_uvtiling)
{
    if (!out_file || out_size == 0) return 0;
    out_file[0] = '\0';
    if (!stage || !target || !target[0] || depth > 8) return 0;

    char prim_path[512], prop[128];
    if (!split_connection_path(target, prim_path, sizeof(prim_path),
                               prop, sizeof(prop)))
        return 0;

    NanousdPrim src = nanousd_primpath(stage, prim_path);
    if (!src) return 0;

    int ok = 0;
    const char* info_id = nanousd_attrib_token(src, "info:id", &ok);
    int is_normal = ok && info_id && strstr(info_id, "normalmap");
    if (is_normal) {
        if (is_normal_map) *is_normal_map = 1;
        char inner[1024];
        int has_inner = first_connection(src, "inputs:in", inner, sizeof(inner));
        nanousd_freeprim(src);
        if (has_inner) {
            return resolve_mtlx_texture_file_from_target(stage, inner, depth + 1,
                                                         out_file, out_size,
                                                         is_normal_map, out_uvtiling,
                                                         has_uvtiling);
        }
        return 0;
    }

    const char* file = nanousd_attribasset(src, "inputs:file", &ok);
    if (!ok || !file || !file[0]) {
        ok = 0;
        file = nanousd_attribs(src, "inputs:file", &ok);
    }
    if (ok && file && file[0]) {
        snprintf(out_file, out_size, "%s", file);
        /* USD-authored MaterialX carries per-image UV tiling as inputs:uvtiling
         * (vector2 — the same custom image input the standalone .mtlx reader
         * consumes), so read it off the image node instead of silently ignoring
         * it on the USD-authored path. */
        if (out_uvtiling && has_uvtiling) {
            float uvt[2] = {1.0f, 1.0f};
            if (nanousd_attribv2f(src, "inputs:uvtiling", uvt)) {
                out_uvtiling[0] = uvt[0];
                out_uvtiling[1] = uvt[1];
                *has_uvtiling = 1;
            }
        }
        nanousd_freeprim(src);
        return 1;
    }

    if (prop[0]) {
        char next[1024];
        int has_next = first_connection(src, prop, next, sizeof(next));
        nanousd_freeprim(src);
        if (has_next) {
            return resolve_mtlx_texture_file_from_target(stage, next, depth + 1,
                                                         out_file, out_size,
                                                         is_normal_map, out_uvtiling,
                                                         has_uvtiling);
        }
        return 0;
    }

    nanousd_freeprim(src);
    return 0;
}

static int bind_mtlx_texture_input(MaterialCollection* mc,
                                   NanousdStage stage,
                                   NanousdPrim shader_prim,
                                   const char* input_name,
                                   int slot,
                                   MaterialParams* p,
                                   const char* scene_dir)
{
    char attr[128];
    snprintf(attr, sizeof(attr), "inputs:%s", input_name);
    char conn[1024];
    if (!first_connection(shader_prim, attr, conn, sizeof(conn))) return -1;

    char file[1024];
    int is_normal_map = 0;
    float uvtiling[2] = {1.0f, 1.0f};
    int has_uvtiling = 0;
    if (!resolve_mtlx_texture_file_from_target(stage, conn, 0, file,
                                               sizeof(file), &is_normal_map,
                                               uvtiling, &has_uvtiling))
        return -1;

    int tex_idx = find_or_add_texture(mc, scene_dir, file, stage);
    if (tex_idx < 0) return -1;

    if (slot >= 0 && p->tex_indices[slot] < 0)
        p->tex_indices[slot] = tex_idx;
    if (tex_idx >= 0 && tex_idx < mc->ntextures) {
        if (mc->textures[tex_idx].udim_cols > 0 &&
            mc->textures[tex_idx].udim_cols > (int)p->udim_scale_u)
            p->udim_scale_u = (float)mc->textures[tex_idx].udim_cols;
        if (mc->textures[tex_idx].udim_rows > 0 &&
            mc->textures[tex_idx].udim_rows > (int)p->udim_scale_v)
            p->udim_scale_v = (float)mc->textures[tex_idx].udim_rows;
    }
    (void)is_normal_map;

    /* One UV transform per material; the base-color image defines it
     * (deterministic priority, not the standalone path's order-dependent
     * last-write). mdl_uv_transform was initialized to identity, so a material
     * with no base-color uvtiling stays un-tiled. */
    if (slot == TEX_DIFFUSE_COLOR && has_uvtiling) {
        p->mdl_uv_transform[0] = uvtiling[0];
        p->mdl_uv_transform[1] = uvtiling[1];
    }
    return tex_idx;
}

static float clamp01(float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

static void init_materialx_surface_defaults(MaterialParams* p, int open_pbr)
{
    p->base_color[0] = 0.8f; p->base_color[1] = 0.8f;
    p->base_color[2] = 0.8f; p->base_color[3] = 1.0f;
    p->emissive_color[0] = 0.0f; p->emissive_color[1] = 0.0f;
    p->emissive_color[2] = 0.0f; p->emissive_color[3] = 1.0f;
    p->metallic = 0.0f;
    p->roughness = open_pbr ? 0.5f : 0.2f;
    p->opacity = 1.0f;
    p->ior = 1.5f;
    p->occlusion = 1.0f;
    p->clearcoat = 0.0f;
    p->clearcoat_roughness = 0.01f;
    p->normal_scale = 1.0f;
    p->udim_scale_u = 1.0f;
    p->udim_scale_v = 1.0f;
    p->use_vertex_color = 0;
    p->v_flip = 0;
    p->opacity_threshold = 0.0f;
    p->transmission_color[0] = 1.0f;
    p->transmission_color[1] = 1.0f;
    p->transmission_color[2] = 1.0f;
    p->transmission_color[3] = 1.0f;
    p->transmission_weight = 0.0f;
    p->transmission_ior = 0.0f;
    p->use_specular_workflow = 0;
    p->specular_color[0] = 0.0f;
    p->specular_color[1] = 0.0f;
    p->specular_color[2] = 0.0f;
    p->specular_color[3] = 0.0f;
    p->roughness_tex_scale = 1.0f;
    p->roughness_tex_bias = 0.0f;
    for (int i = 0; i < MAX_MATERIAL_TEXTURES; i++)
        p->tex_indices[i] = -1;
}

static void read_materialx_usd_surface(MaterialCollection* mc,
                                       NanousdStage stage,
                                       NanousdPrim material_prim,
                                       NanousdPrim shader_prim,
                                       MaterialParams* p,
                                       const char* scene_dir,
                                       int open_pbr)
{
    init_materialx_surface_defaults(p, open_pbr);
    if (!mc || !stage || !material_prim || !shader_prim) return;

    float c3[3];
    float f = 0.0f;

    if (read_connected_attr_color3(stage, shader_prim, material_prim,
                                   "base_color", c3)) {
        p->base_color[0] = c3[0];
        p->base_color[1] = c3[1];
        p->base_color[2] = c3[2];
    }

    if (open_pbr) {
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "base_metalness", &f))
            p->metallic = f;
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "specular_roughness", &f))
            p->roughness = f;
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "specular_ior", &f))
            p->ior = f;
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "coat_weight", &f))
            p->clearcoat = f;
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "coat_roughness", &f))
            p->clearcoat_roughness = f;
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "geometry_opacity", &f))
            p->opacity = f;
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "transmission_weight", &f))
            p->transmission_weight = f;
        if (read_connected_attr_color3(stage, shader_prim, material_prim,
                                       "transmission_color", c3)) {
            p->transmission_color[0] = c3[0];
            p->transmission_color[1] = c3[1];
            p->transmission_color[2] = c3[2];
        }
        if (read_connected_attr_color3(stage, shader_prim, material_prim,
                                       "emission_color", c3)) {
            p->emissive_color[0] = c3[0];
            p->emissive_color[1] = c3[1];
            p->emissive_color[2] = c3[2];
        }
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "emission_luminance", &f))
            p->emissive_color[3] = f;

        if (bind_mtlx_texture_input(mc, stage, shader_prim, "base_color",
                                    TEX_DIFFUSE_COLOR, p, scene_dir) >= 0) {
            p->base_color[0] = 1.0f;
            p->base_color[1] = 1.0f;
            p->base_color[2] = 1.0f;
        }
        bind_mtlx_texture_input(mc, stage, shader_prim, "base_metalness",
                                TEX_METALLIC, p, scene_dir);
        bind_mtlx_texture_input(mc, stage, shader_prim, "specular_roughness",
                                TEX_ROUGHNESS, p, scene_dir);
        bind_mtlx_texture_input(mc, stage, shader_prim, "geometry_normal",
                                TEX_NORMAL, p, scene_dir);
        bind_mtlx_texture_input(mc, stage, shader_prim, "geometry_opacity",
                                TEX_OPACITY, p, scene_dir);
        bind_mtlx_texture_input(mc, stage, shader_prim, "emission_color",
                                TEX_EMISSIVE_COLOR, p, scene_dir);
    } else {
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "metalness", &f))
            p->metallic = f;
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "specular_roughness", &f))
            p->roughness = f;
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "specular_IOR", &f)) {
            p->ior = f;
            p->transmission_ior = f;
        }
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "coat", &f))
            p->clearcoat = f;
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "coat_roughness", &f))
            p->clearcoat_roughness = f;
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "opacity", &f))
            p->opacity = f;
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "transmission", &f))
            p->transmission_weight = f;
        if (read_connected_attr_color3(stage, shader_prim, material_prim,
                                       "transmission_color", c3)) {
            p->transmission_color[0] = c3[0];
            p->transmission_color[1] = c3[1];
            p->transmission_color[2] = c3[2];
        }
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "emission", &f))
            p->emissive_color[3] = f;
        if (read_connected_attr_color3(stage, shader_prim, material_prim,
                                       "emission_color", c3)) {
            p->emissive_color[0] = c3[0] * p->emissive_color[3];
            p->emissive_color[1] = c3[1] * p->emissive_color[3];
            p->emissive_color[2] = c3[2] * p->emissive_color[3];
        }

        if (bind_mtlx_texture_input(mc, stage, shader_prim, "base_color",
                                    TEX_DIFFUSE_COLOR, p, scene_dir) >= 0) {
            p->base_color[0] = 1.0f;
            p->base_color[1] = 1.0f;
            p->base_color[2] = 1.0f;
        }
        bind_mtlx_texture_input(mc, stage, shader_prim, "metalness",
                                TEX_METALLIC, p, scene_dir);
        bind_mtlx_texture_input(mc, stage, shader_prim, "specular_roughness",
                                TEX_ROUGHNESS, p, scene_dir);
        bind_mtlx_texture_input(mc, stage, shader_prim, "normal",
                                TEX_NORMAL, p, scene_dir);
        bind_mtlx_texture_input(mc, stage, shader_prim, "opacity",
                                TEX_OPACITY, p, scene_dir);
        bind_mtlx_texture_input(mc, stage, shader_prim, "emission_color",
                                TEX_EMISSIVE_COLOR, p, scene_dir);
    }

    p->metallic = clamp01(p->metallic);
    p->roughness = clamp01(p->roughness);
    p->opacity = clamp01(p->opacity);

    fprintf(stderr,
            "material: read MaterialX USD %s params: base=(%.2f,%.2f,%.2f) "
            "metal=%.2f rough=%.2f opacity=%.2f slots D=%d N=%d R=%d M=%d E=%d OP=%d\n",
            open_pbr ? "OpenPBR" : "StandardSurface",
            p->base_color[0], p->base_color[1], p->base_color[2],
            p->metallic, p->roughness, p->opacity,
            p->tex_indices[TEX_DIFFUSE_COLOR],
            p->tex_indices[TEX_NORMAL],
            p->tex_indices[TEX_ROUGHNESS],
            p->tex_indices[TEX_METALLIC],
            p->tex_indices[TEX_EMISSIVE_COLOR],
            p->tex_indices[TEX_OPACITY]);
}

/* ---- Generic MDL sourceAsset extraction ---- */

static int mdl_has_any(const char* s, const char** keys, int nkeys)
{
    if (!s) return 0;
    for (int i = 0; i < nkeys; i++) {
        if (strstr(s, keys[i])) return 1;
    }
    return 0;
}

static void mdl_lower_copy(char* dst, size_t dst_size, const char* src)
{
    if (!dst || dst_size == 0) return;
    size_t i = 0;
    if (src) {
        for (; src[i] && i + 1 < dst_size; i++)
            dst[i] = (char)tolower((unsigned char)src[i]);
    }
    dst[i] = '\0';
}

static int mdl_has_image_extension(const char* path)
{
    char l[1024];
    mdl_lower_copy(l, sizeof(l), path);
    return strstr(l, ".png") || strstr(l, ".jpg") ||
           strstr(l, ".jpeg") || strstr(l, ".exr") ||
           strstr(l, ".tga") || strstr(l, ".bmp") ||
           strstr(l, ".hdr") || strstr(l, "<udim>");
}

static void mdl_material_name_from_subidentifier(const char* sub,
                                                 char* out, size_t out_size)
{
    if (!out || out_size == 0) return;
    out[0] = '\0';
    if (!sub || !sub[0]) return;

    const char* start = sub;
    const char* p = sub;
    while ((p = strstr(p, "::")) != NULL) {
        start = p + 2;
        p += 2;
    }
    const char* slash = strrchr(start, '/');
    if (slash) start = slash + 1;

    size_t n = 0;
    while (start[n] && start[n] != '(' && !isspace((unsigned char)start[n]))
        n++;
    if (n >= out_size) n = out_size - 1;
    memcpy(out, start, n);
    out[n] = '\0';
}

static int mdl_is_ident_char(char c)
{
    return isalnum((unsigned char)c) || c == '_';
}

static int mdl_find_matching_paren(const char* body, const char* open,
                                   const char** out_close)
{
    int depth = 0, in_string = 0, escape = 0;
    for (const char* p = open; *p; p++) {
        char c = *p;
        if (in_string) {
            if (escape) escape = 0;
            else if (c == '\\') escape = 1;
            else if (c == '"') in_string = 0;
            continue;
        }
        if (c == '"') {
            in_string = 1;
            continue;
        }
        if (c == '(') depth++;
        else if (c == ')') {
            depth--;
            if (depth == 0) {
                if (out_close) *out_close = p;
                return 1;
            }
        }
    }
    return 0;
}

static int mdl_find_export_material_params(const char* body,
                                           const char* material_name,
                                           const char** out_start,
                                           const char** out_end)
{
    if (!body || !out_start || !out_end) return 0;
    *out_start = NULL;
    *out_end = NULL;

    const char* first_start = NULL;
    const char* first_end = NULL;
    const char* p = body;
    const char* key = "export material";
    size_t key_len = strlen(key);
    while ((p = strstr(p, key)) != NULL) {
        const char* n = p + key_len;
        while (*n && isspace((unsigned char)*n)) n++;
        if (!isalpha((unsigned char)*n) && *n != '_') {
            p += key_len;
            continue;
        }
        const char* name_start = n;
        while (*n && mdl_is_ident_char(*n)) n++;
        size_t name_len = (size_t)(n - name_start);
        const char* open = strchr(n, '(');
        const char* close = NULL;
        if (!open || !mdl_find_matching_paren(body, open, &close)) {
            p += key_len;
            continue;
        }
        if (!first_start) {
            first_start = open + 1;
            first_end = close;
        }
        if (!material_name || !material_name[0] ||
            (strlen(material_name) == name_len &&
             strncmp(material_name, name_start, name_len) == 0)) {
            *out_start = open + 1;
            *out_end = close;
            return 1;
        }
        p = close + 1;
    }
    if (first_start) {
        *out_start = first_start;
        *out_end = first_end;
        return 1;
    }
    return 0;
}

static int mdl_param_name_before_equal(const char* param,
                                       char* out, size_t out_size)
{
    if (!param || !out || out_size == 0) return 0;
    out[0] = '\0';
    const char* eq = strchr(param, '=');
    if (!eq) return 0;
    const char* end = eq;
    while (end > param && isspace((unsigned char)*(end - 1))) end--;
    const char* start = end;
    while (start > param && mdl_is_ident_char(*(start - 1))) start--;
    if (start == end) return 0;
    size_t n = (size_t)(end - start);
    if (n >= out_size) n = out_size - 1;
    memcpy(out, start, n);
    out[n] = '\0';
    return 1;
}

static int mdl_first_quoted_after(const char* p, char* out, size_t out_size)
{
    if (!p || !out || out_size == 0) return 0;
    out[0] = '\0';
    const char* q1 = strchr(p, '"');
    if (!q1) return 0;
    const char* q2 = strchr(q1 + 1, '"');
    if (!q2 || q2 == q1 + 1) return 0;
    size_t n = (size_t)(q2 - q1 - 1);
    if (n >= out_size) n = out_size - 1;
    memcpy(out, q1 + 1, n);
    out[n] = '\0';
    return 1;
}

static int mdl_parse_texture_default(const char* param,
                                     char* out, size_t out_size)
{
    const char* eq = param ? strchr(param, '=') : NULL;
    if (!eq) return 0;
    const char* tex = strstr(eq + 1, "texture_2d");
    if (tex) return mdl_first_quoted_after(tex, out, out_size);
    return mdl_first_quoted_after(eq + 1, out, out_size);
}

static int mdl_parse_color_default(const char* param, float out[3])
{
    const char* eq = param ? strchr(param, '=') : NULL;
    if (!eq || !out) return 0;
    const char* c = strstr(eq + 1, "color");
    if (!c) return 0;
    const char* open = strchr(c, '(');
    if (!open) return 0;
    const char* p = open + 1;
    char* endp = NULL;
    float v[3];
    for (int i = 0; i < 3; i++) {
        while (*p && (isspace((unsigned char)*p) || *p == ',')) p++;
        if (i > 0 && *p == ')') {
            v[i] = v[0];
            if (i == 1) v[2] = v[0];
            break;
        }
        v[i] = strtof(p, &endp);
        if (endp == p) return 0;
        p = endp;
    }
    out[0] = v[0]; out[1] = v[1]; out[2] = v[2];
    return 1;
}

static int mdl_parse_float_default(const char* param, float* out)
{
    const char* p = param ? strchr(param, '=') : NULL;
    if (!p || !out) return 0;
    p++;
    while (*p && !(isdigit((unsigned char)*p) ||
                   *p == '-' || *p == '+' || *p == '.')) {
        if (*p == '"' || *p == '{') return 0;
        p++;
    }
    char* endp = NULL;
    float v = strtof(p, &endp);
    if (endp == p) return 0;
    *out = v;
    return 1;
}

static int mdl_classify_texture_slot(const char* name_hint, const char* tex_path,
                                     int* is_orm)
{
    if (is_orm) *is_orm = 0;
    if (!tex_path || !tex_path[0]) return -1;

    char lpath[1024], lhint[512];
    mdl_lower_copy(lpath, sizeof(lpath), tex_path);
    mdl_lower_copy(lhint, sizeof(lhint), name_hint);
    if (strstr(lpath, ".mdl")) return -1;

    const char* slash = strrchr(lpath, '/');
    const char* base = slash ? slash + 1 : lpath;
    char stem[256];
    snprintf(stem, sizeof(stem), "%s", base);
    char* dot = strrchr(stem, '.');
    if (dot) *dot = '\0';
    const char* us = strrchr(stem, '_');
    const char* suffix = us ? us + 1 : "";

    if (strcmp(suffix, "orm") == 0 || strstr(lpath, "_orm") ||
        strstr(lpath, "orm.") || strstr(lhint, "mergemap")) {
        if (is_orm) *is_orm = 1;
        return -1;
    }
    if (strcmp(suffix, "d") == 0 || strcmp(suffix, "c") == 0 ||
        strcmp(suffix, "diff") == 0 || strcmp(suffix, "diffuse") == 0 ||
        strcmp(suffix, "albedo") == 0 || strcmp(suffix, "basecolor") == 0 ||
        strstr(lpath, "albedo") || strstr(lpath, "diffuse") ||
        strstr(lpath, "basecolor") || strstr(lpath, "base_color") ||
        strstr(lhint, "albedo") || strstr(lhint, "diffuse") ||
        strstr(lhint, "basecolor") || strstr(lhint, "base_color"))
        return TEX_DIFFUSE_COLOR;
    if (strcmp(suffix, "n") == 0 || strcmp(suffix, "nrm") == 0 ||
        strcmp(suffix, "norm") == 0 || strstr(lpath, "normal") ||
        strstr(lhint, "normal"))
        return TEX_NORMAL;
    if (strcmp(suffix, "r") == 0 || strcmp(suffix, "rgh") == 0 ||
        strstr(lpath, "rough") || strstr(lhint, "rough"))
        return TEX_ROUGHNESS;
    if (strcmp(suffix, "m") == 0 || strcmp(suffix, "met") == 0 ||
        strstr(lpath, "metal") || strstr(lhint, "metal"))
        return TEX_METALLIC;
    if (strcmp(suffix, "e") == 0 || strcmp(suffix, "emi") == 0 ||
        strstr(lpath, "emissive") || strstr(lpath, "emission") ||
        strstr(lhint, "emissive") || strstr(lhint, "emission"))
        return TEX_EMISSIVE_COLOR;
    if (strcmp(suffix, "ao") == 0 || strcmp(suffix, "occ") == 0 ||
        strstr(lpath, "occlusion") || strstr(lpath, "_ao") ||
        strstr(lhint, "occlusion"))
        return TEX_OCCLUSION;
    if (strcmp(suffix, "o") == 0 || strcmp(suffix, "op") == 0 ||
        strcmp(suffix, "a") == 0 || strstr(lpath, "opacity") ||
        strstr(lpath, "alpha") || strstr(lhint, "opacity"))
        return TEX_OPACITY;
    return -1;
}

static int is_isaac_mdl_texture_input(const char* input)
{
    if (!input) return 0;
    return strcmp(input, "inputs:AlbedoTexture") == 0 ||
           strcmp(input, "inputs:BaseColor_Texture") == 0 ||
           strcmp(input, "inputs:BaseColor_Box") == 0 ||
           strcmp(input, "inputs:BaseColor_Plastic") == 0 ||
           strcmp(input, "inputs:TextureSelection") == 0 ||
           strcmp(input, "inputs:Text") == 0 ||
           strcmp(input, "inputs:MainNormalInput") == 0 ||
           strcmp(input, "inputs:NormalMap_Box") == 0 ||
           strcmp(input, "inputs:MergeMapInput") == 0 ||
           strcmp(input, "inputs:MultiMap_Box") == 0 ||
           strcmp(input, "inputs:MultiMap_Plastic") == 0 ||
           strcmp(input, "inputs:AlphaSelection") == 0;
}

static void mdl_assign_texture(MaterialCollection* mc, void* stage,
                               MaterialParams* p, const char* anchor_dir,
                               const char* name_hint, const char* tex_ref)
{
    if (!mdl_has_image_extension(tex_ref)) return;
    char lname[512];
    mdl_lower_copy(lname, sizeof(lname), name_hint);
    int is_orm = 0;
    int slot = mdl_classify_texture_slot(name_hint, tex_ref, &is_orm);
    const char* diffuse_keys[] = {"base", "diffuse", "albedo", "color", "tex", "texture"};
    if (slot < 0 && !is_orm && p->tex_indices[TEX_DIFFUSE_COLOR] < 0 &&
        mdl_has_any(lname, diffuse_keys, 6))
        slot = TEX_DIFFUSE_COLOR;

    if (is_orm) {
        int idx = find_or_add_texture(mc, anchor_dir, tex_ref, stage);
        if (idx >= 0) {
            if (p->tex_indices[TEX_OCCLUSION] < 0) p->tex_indices[TEX_OCCLUSION] = idx;
            if (p->tex_indices[TEX_ROUGHNESS] < 0) p->tex_indices[TEX_ROUGHNESS] = idx;
            if (p->tex_indices[TEX_METALLIC] < 0) p->tex_indices[TEX_METALLIC] = idx;
        }
    } else if (slot >= 0 && p->tex_indices[slot] < 0) {
        int idx = find_or_add_texture(mc, anchor_dir, tex_ref, stage);
        if (idx >= 0) p->tex_indices[slot] = idx;
    }
}

static float srgb_u8_to_linear(unsigned char v)
{
    float c = (float)v / 255.0f;
    if (c <= 0.04045f) return c / 12.92f;
    return powf((c + 0.055f) / 1.055f, 2.4f);
}

static unsigned char linear_to_srgb_u8(float v)
{
    v = clamp01(v);
    float c = (v <= 0.0031308f)
                  ? (12.92f * v)
                  : (1.055f * powf(v, 1.0f / 2.4f) - 0.055f);
    int q = (int)lroundf(clamp01(c) * 255.0f);
    if (q < 0) q = 0;
    if (q > 255) q = 255;
    return (unsigned char)q;
}

static int append_generated_texture(MaterialCollection* mc, const char* key,
                                    unsigned char* pixels, int width,
                                    int height, int is_srgb)
{
    if (!mc || !key || !key[0] || !pixels || width <= 0 || height <= 0)
        return -1;

    int idx = mc->ntextures;
    MaterialTexture* new_texs = (MaterialTexture*)realloc(
        mc->textures, (size_t)(idx + 1) * sizeof(MaterialTexture));
    if (!new_texs) return -1;

    mc->textures = new_texs;
    memset(&mc->textures[idx], 0, sizeof(MaterialTexture));
    mc->textures[idx].pixels = pixels;
    mc->textures[idx].width = width;
    mc->textures[idx].height = height;
    mc->textures[idx].is_srgb = is_srgb ? 1 : 0;
    snprintf(mc->textures[idx].path, sizeof(mc->textures[idx].path), "%s", key);
    mc->ntextures = idx + 1;
    return idx;
}

/* ---- MDL bake parallelism (std::thread via nu_parallel_for) --------------
 * These per-row bakes were OpenMP `#pragma omp parallel for` loops, which
 * compiled to no-ops (serial) on AppleClang. Each body is hoisted into a C
 * body-fn + context struct so it runs through the portable nu_parallel_for.
 * Output is bit-identical to the serial/OpenMP path (independent output rows,
 * read-only inputs). */
typedef struct {
    unsigned char* baked;
    const MaterialTexture* albedo;
    const MaterialTexture* mask;
    const float* color_albedo;
    int width, height;
} MdlMaskedBakeCtx;
static void mdl_masked_bake_row(int y, void* vctx) {
    const MdlMaskedBakeCtx* cx = (const MdlMaskedBakeCtx*)vctx;
    int my = (int)((int64_t)y * cx->mask->height / cx->height);
    if (my >= cx->mask->height) my = cx->mask->height - 1;
    for (int x = 0; x < cx->width; x++) {
        int mx = (int)((int64_t)x * cx->mask->width / cx->width);
        if (mx >= cx->mask->width) mx = cx->mask->width - 1;
        size_t ai = ((size_t)y * (size_t)cx->width + (size_t)x) * 4;
        size_t mi = ((size_t)my * (size_t)cx->mask->width + (size_t)mx) * 4;
        for (int c = 0; c < 3; c++) {
            float tex_c = srgb_u8_to_linear(cx->albedo->pixels[ai + c]);
            float mask_c = (float)cx->mask->pixels[mi + c] / 255.0f;
            float out_c = cx->color_albedo[c] * (1.0f - mask_c) + tex_c * mask_c;
            cx->baked[ai + c] = linear_to_srgb_u8(out_c);
        }
        cx->baked[ai + 3] = cx->albedo->pixels[ai + 3];
    }
}
typedef struct {
    unsigned char* baked;
    const MaterialTexture* albedo;
    const MaterialTexture* mask;
    const float* body_color;
    const float* handle_color;
    const float* cap_color;
    int width, height;
} MdlBodyMaskedBakeCtx;
static void mdl_body_masked_bake_row(int y, void* vctx) {
    const MdlBodyMaskedBakeCtx* cx = (const MdlBodyMaskedBakeCtx*)vctx;
    int my = (int)((int64_t)y * cx->mask->height / cx->height);
    if (my >= cx->mask->height) my = cx->mask->height - 1;
    for (int x = 0; x < cx->width; x++) {
        int mx = (int)((int64_t)x * cx->mask->width / cx->width);
        if (mx >= cx->mask->width) mx = cx->mask->width - 1;
        size_t ai = ((size_t)y * (size_t)cx->width + (size_t)x) * 4;
        size_t mi = ((size_t)my * (size_t)cx->mask->width + (size_t)mx) * 4;
        float r = (float)cx->mask->pixels[mi + 0] / 255.0f;
        float g = (float)cx->mask->pixels[mi + 1] / 255.0f;
        float b = (float)cx->mask->pixels[mi + 2] / 255.0f;
        float a = (float)cx->mask->pixels[mi + 3] / 255.0f;
        for (int c = 0; c < 3; c++) {
            float body_mix = cx->body_color[c] * r;
            float handle_mix = body_mix * (1.0f - g) + cx->handle_color[c] * g;
            float cap_mix = handle_mix * (1.0f - b) + cx->cap_color[c] * b;
            float tex_c = srgb_u8_to_linear(cx->albedo->pixels[ai + c]);
            float out_c = cap_mix * (1.0f - a) + tex_c * a;
            cx->baked[ai + c] = linear_to_srgb_u8(out_c);
        }
        cx->baked[ai + 3] = cx->albedo->pixels[ai + 3];
    }
}
typedef struct {
    unsigned char* baked;
    const MaterialTexture* color;
    const MaterialTexture* mask;
    float scale;
    int width, height;
} MdlEmissiveBakeCtx;
static void mdl_emissive_bake_row(int y, void* vctx) {
    const MdlEmissiveBakeCtx* cx = (const MdlEmissiveBakeCtx*)vctx;
    int my = (int)((int64_t)y * cx->mask->height / cx->height);
    if (my >= cx->mask->height) my = cx->mask->height - 1;
    for (int x = 0; x < cx->width; x++) {
        int mx = (int)((int64_t)x * cx->mask->width / cx->width);
        if (mx >= cx->mask->width) mx = cx->mask->width - 1;
        size_t ci = ((size_t)y * (size_t)cx->width + (size_t)x) * 4;
        size_t mi = ((size_t)my * (size_t)cx->mask->width + (size_t)mx) * 4;
        for (int c = 0; c < 3; c++) {
            float color_c = srgb_u8_to_linear(cx->color->pixels[ci + c]);
            float mask_c = (float)cx->mask->pixels[mi + c] / 255.0f;
            cx->baked[ci + c] = linear_to_srgb_u8(color_c * mask_c * cx->scale);
        }
        cx->baked[ci + 3] = 255;
    }
}

static int bake_mdl_masked_albedo_texture(MaterialCollection* mc, void* stage,
                                          const char* anchor_dir,
                                          const char* albedo_ref,
                                          const char* mask_ref,
                                          const float color_albedo[3])
{
    if (!mc || !anchor_dir || !color_albedo ||
        !mdl_has_image_extension(albedo_ref) ||
        !mdl_has_image_extension(mask_ref))
        return -1;

    char key[1024];
    snprintf(key, sizeof(key),
             "mdl_baked_masked_albedo:%s|%s|%.6g,%.6g,%.6g",
             albedo_ref, mask_ref,
             color_albedo[0], color_albedo[1], color_albedo[2]);
    for (int i = 0; i < mc->ntextures; i++) {
        if (strcmp(mc->textures[i].path, key) == 0) return i;
    }

    int albedo_idx = find_or_add_texture(mc, anchor_dir, albedo_ref, stage);
    int mask_idx = find_or_add_texture(mc, anchor_dir, mask_ref, stage);
    if (albedo_idx < 0 || mask_idx < 0 ||
        albedo_idx >= mc->ntextures || mask_idx >= mc->ntextures)
        return -1;

    const MaterialTexture* albedo = &mc->textures[albedo_idx];
    const MaterialTexture* mask = &mc->textures[mask_idx];
    if (!albedo->pixels || !mask->pixels ||
        albedo->width <= 0 || albedo->height <= 0 ||
        mask->width <= 0 || mask->height <= 0)
        return -1;

    int width = albedo->width;
    int height = albedo->height;
    unsigned char* baked = (unsigned char*)malloc((size_t)width *
                                                  (size_t)height * 4);
    if (!baked) return -1;

    /* Race-free per-row bake — parallelised via nu_parallel_for (std::thread;
     * the prior OpenMP pragma was a silent no-op on AppleClang). Bit-identical. */
    MdlMaskedBakeCtx _mbctx = { baked, albedo, mask, color_albedo, width, height };
    nu_parallel_for(height, mdl_masked_bake_row, &_mbctx);

    int idx = append_generated_texture(mc, key, baked, width, height, 1);
    if (idx < 0) {
        free(baked);
        return -1;
    }
    fprintf(stderr,
            "material:   MDL baked masked albedo: %s + %s -> diffuse slot\n",
            albedo_ref, mask_ref);
    return idx;
}

static int bake_mdl_body_masked_albedo_texture(MaterialCollection* mc,
                                               void* stage,
                                               const char* anchor_dir,
                                               const char* albedo_ref,
                                               const char* mask_ref,
                                               const float body_color[3],
                                               const float handle_color[3],
                                               const float cap_color[3])
{
    if (!mc || !anchor_dir || !body_color || !handle_color || !cap_color ||
        !mdl_has_image_extension(albedo_ref) ||
        !mdl_has_image_extension(mask_ref))
        return -1;

    char key[1024];
    snprintf(key, sizeof(key),
             "mdl_baked_body_masked_albedo:%s|%s|%.6g,%.6g,%.6g|"
             "%.6g,%.6g,%.6g|%.6g,%.6g,%.6g",
             albedo_ref, mask_ref,
             body_color[0], body_color[1], body_color[2],
             handle_color[0], handle_color[1], handle_color[2],
             cap_color[0], cap_color[1], cap_color[2]);
    for (int i = 0; i < mc->ntextures; i++) {
        if (strcmp(mc->textures[i].path, key) == 0) return i;
    }

    int albedo_idx = find_or_add_texture(mc, anchor_dir, albedo_ref, stage);
    int mask_idx = find_or_add_texture(mc, anchor_dir, mask_ref, stage);
    if (albedo_idx < 0 || mask_idx < 0 ||
        albedo_idx >= mc->ntextures || mask_idx >= mc->ntextures)
        return -1;

    const MaterialTexture* albedo = &mc->textures[albedo_idx];
    const MaterialTexture* mask = &mc->textures[mask_idx];
    if (!albedo->pixels || !mask->pixels ||
        albedo->width <= 0 || albedo->height <= 0 ||
        mask->width <= 0 || mask->height <= 0)
        return -1;

    int width = albedo->width;
    int height = albedo->height;
    unsigned char* baked = (unsigned char*)malloc((size_t)width *
                                                  (size_t)height * 4);
    if (!baked) return -1;

    /* Race-free per-row bake — parallelised via nu_parallel_for. Bit-identical. */
    MdlBodyMaskedBakeCtx _bbctx = { baked, albedo, mask, body_color,
                                    handle_color, cap_color, width, height };
    nu_parallel_for(height, mdl_body_masked_bake_row, &_bbctx);

    int idx = append_generated_texture(mc, key, baked, width, height, 1);
    if (idx < 0) {
        free(baked);
        return -1;
    }
    fprintf(stderr,
            "material:   MDL baked body-mask albedo: %s + %s -> diffuse slot\n",
            albedo_ref, mask_ref);
    return idx;
}

static int bake_mdl_emissive_product_texture(MaterialCollection* mc,
                                             void* stage,
                                             const char* anchor_dir,
                                             const char* color_ref,
                                             const char* mask_ref,
                                             float scale)
{
    if (!mc || !anchor_dir ||
        !mdl_has_image_extension(color_ref) ||
        !mdl_has_image_extension(mask_ref))
        return -1;

    char key[1024];
    snprintf(key, sizeof(key),
             "mdl_baked_emissive_product:%s|%s|%.6g",
             color_ref, mask_ref, scale);
    for (int i = 0; i < mc->ntextures; i++) {
        if (strcmp(mc->textures[i].path, key) == 0) return i;
    }

    int color_idx = find_or_add_texture(mc, anchor_dir, color_ref, stage);
    int mask_idx = find_or_add_texture(mc, anchor_dir, mask_ref, stage);
    if (color_idx < 0 || mask_idx < 0 ||
        color_idx >= mc->ntextures || mask_idx >= mc->ntextures)
        return -1;

    const MaterialTexture* color = &mc->textures[color_idx];
    const MaterialTexture* mask = &mc->textures[mask_idx];
    if (!color->pixels || !mask->pixels ||
        color->width <= 0 || color->height <= 0 ||
        mask->width <= 0 || mask->height <= 0)
        return -1;

    int width = color->width;
    int height = color->height;
    unsigned char* baked = (unsigned char*)malloc((size_t)width *
                                                  (size_t)height * 4);
    if (!baked) return -1;

    scale = scale > 0.0f ? scale : 0.0f;
    /* Race-free per-row bake — parallelised via nu_parallel_for. Bit-identical. */
    MdlEmissiveBakeCtx _ebctx = { baked, color, mask, scale, width, height };
    nu_parallel_for(height, mdl_emissive_bake_row, &_ebctx);

    int idx = append_generated_texture(mc, key, baked, width, height, 1);
    if (idx < 0) {
        free(baked);
        return -1;
    }
    fprintf(stderr,
            "material:   MDL baked emissive product: %s * %s -> emissive slot\n",
            color_ref, mask_ref);
    return idx;
}

static int mdl_find_first_texture_literal_for_slot(const char* body,
                                                   int wanted_slot,
                                                   char* out,
                                                   size_t out_size)
{
    if (!body || !out || out_size == 0) return 0;
    out[0] = '\0';
    const char* q = body;
    while ((q = strstr(q, "texture_2d")) != NULL) {
        char tex_ref[1024];
        if (!mdl_first_quoted_after(q, tex_ref, sizeof(tex_ref))) {
            q += 10;
            continue;
        }
        int is_orm = 0;
        int slot = mdl_classify_texture_slot(NULL, tex_ref, &is_orm);
        if (!is_orm && slot == wanted_slot) {
            snprintf(out, out_size, "%s", tex_ref);
            return 1;
        }
        q += 10;
    }
    return 0;
}

static int mdl_find_texture_literal_with_keywords(const char* body,
                                                  const char* const* keywords,
                                                  int keyword_count,
                                                  char* out,
                                                  size_t out_size)
{
    if (!body || !keywords || keyword_count <= 0 || !out || out_size == 0)
        return 0;
    out[0] = '\0';
    const char* q = body;
    while ((q = strstr(q, "texture_2d")) != NULL) {
        char tex_ref[1024];
        if (!mdl_first_quoted_after(q, tex_ref, sizeof(tex_ref))) {
            q += 10;
            continue;
        }

        const char* line = q;
        while (line > body && *(line - 1) != '\n') line--;
        const char* line_end = strchr(q, '\n');
        if (!line_end) line_end = q + strlen(q);

        char context[2048];
        size_t ref_len = strlen(tex_ref);
        size_t line_len = (size_t)(line_end - line);
        if (ref_len + line_len + 2 >= sizeof(context))
            line_len = sizeof(context) - ref_len - 2;
        memcpy(context, tex_ref, ref_len);
        context[ref_len] = ' ';
        memcpy(context + ref_len + 1, line, line_len);
        context[ref_len + 1 + line_len] = '\0';
        mdl_lower_copy(context, sizeof(context), context);

        for (int i = 0; i < keyword_count; i++) {
            if (keywords[i] && strstr(context, keywords[i])) {
                snprintf(out, out_size, "%s", tex_ref);
                return 1;
            }
        }
        q += 10;
    }
    return 0;
}

static int mdl_source_has_nonzero_emissive_color(const char* source)
{
    if (!source) return 0;
    const char* q = source;
    while ((q = strstr(q, "EmissiveColor_mdl")) != NULL) {
        const char* eq = strchr(q, '=');
        const char* semi = strchr(q, ';');
        if (eq && (!semi || eq < semi)) {
            const char* end = semi ? semi : q + strlen(q);
            char expr[512];
            size_t n = (size_t)(end - eq - 1);
            if (n >= sizeof(expr)) n = sizeof(expr) - 1;
            size_t w = 0;
            for (size_t i = 0; i < n && w + 1 < sizeof(expr); i++) {
                unsigned char c = (unsigned char)eq[1 + i];
                if (!isspace(c))
                    expr[w++] = (char)tolower(c);
            }
            expr[w] = '\0';
            return strcmp(expr, "float3(0.0,0.0,0.0)") != 0 &&
                   strcmp(expr, "float3(0,0,0)") != 0 &&
                   strcmp(expr, "color(0.0,0.0,0.0)") != 0 &&
                   strcmp(expr, "color(0,0,0)") != 0;
        }
        q += strlen("EmissiveColor_mdl");
    }
    return 0;
}

static int mdl_find_named_float_param_default(const char* begin,
                                              const char* end,
                                              const char* name,
                                              float* out)
{
    if (!begin || !end || begin >= end || !name || !out) return 0;
    size_t name_len = strlen(name);
    const char* q = begin;
    while (q < end && (q = strstr(q, name)) != NULL && q < end) {
        char before = (q > begin) ? *(q - 1) : '\0';
        char after = q[name_len];
        if ((before && mdl_is_ident_char(before)) ||
            (after && mdl_is_ident_char(after))) {
            q += name_len;
            continue;
        }

        const char* seg_end = strchr(q, ',');
        if (!seg_end || seg_end > end) seg_end = end;
        size_t n = (size_t)(seg_end - q);
        if (n == 0 || n >= 512) return 0;
        char segment[512];
        memcpy(segment, q, n);
        segment[n] = '\0';
        return mdl_parse_float_default(segment, out);
    }
    return 0;
}

static int mdl_read_asset_input(NanousdPrim shader_prim, const char* name,
                                char* out, size_t out_size)
{
    if (!shader_prim || !name || !out || out_size == 0) return 0;
    out[0] = '\0';
    int ok = 0;
    const char* value = nanousd_attribasset(shader_prim, name, &ok);
    if (!ok || !value || !value[0])
        value = nanousd_attribs(shader_prim, name, &ok);
    if (!ok || !value || !value[0]) return 0;
    snprintf(out, out_size, "%s", value);
    return 1;
}

static int mdl_read_color_input_any(NanousdPrim shader_prim,
                                    const char* const* names,
                                    int count, float out[3])
{
    if (!shader_prim || !names || !out) return 0;
    for (int i = 0; i < count; i++) {
        float v4[4];
        float v3[3];
        if (nanousd_attribv4f(shader_prim, names[i], v4)) {
            out[0] = v4[0];
            out[1] = v4[1];
            out[2] = v4[2];
            return 1;
        }
        if (nanousd_attribv3f(shader_prim, names[i], v3)) {
            out[0] = v3[0];
            out[1] = v3[1];
            out[2] = v3[2];
            return 1;
        }
    }
    return 0;
}

static int mdl_apply_isaac_mask_bakes(MaterialCollection* mc, void* stage,
                                      NanousdPrim shader_prim,
                                      MaterialParams* p,
                                      const char* scene_dir)
{
    if (!mc || !shader_prim || !p || !scene_dir) return 0;

    char albedo_ref[1024];
    char mask_ref[1024];
    if (!mdl_read_asset_input(shader_prim, "inputs:AlbedoTexture",
                              albedo_ref, sizeof(albedo_ref)) ||
        !mdl_read_asset_input(shader_prim, "inputs:MaskSelection",
                              mask_ref, sizeof(mask_ref)))
        return 0;

    float body[3] = {0.0f, 0.0f, 0.0f};
    float handle[3] = {0.0f, 0.0f, 0.0f};
    float cap[3] = {0.0f, 0.0f, 0.0f};
    float color_albedo[3] = {1.0f, 1.0f, 1.0f};

    const char* const body_names[] = {"inputs:Body"};
    const char* const handle_names[] = {"inputs:Handle"};
    const char* const cap_names[] = {"inputs:Cap"};
    const char* const color_names[] = {
        "inputs:ColorAlbedo",
        "inputs:BaseColor_Tint",
    };
    int has_body = mdl_read_color_input_any(
        shader_prim, body_names, (int)(sizeof(body_names) / sizeof(body_names[0])), body);
    int has_handle = mdl_read_color_input_any(
        shader_prim, handle_names, (int)(sizeof(handle_names) / sizeof(handle_names[0])), handle);
    int has_cap = mdl_read_color_input_any(
        shader_prim, cap_names, (int)(sizeof(cap_names) / sizeof(cap_names[0])), cap);
    int has_color_albedo = mdl_read_color_input_any(
        shader_prim, color_names, (int)(sizeof(color_names) / sizeof(color_names[0])),
        color_albedo);

    int baked_idx = -1;
    /* Keep this CPU-side bake aligned with Metal/Vulkan: IsaacSim MDLs use an
     * RGBA mask as control data for generated material variants, but the
     * fallback renderers all shade a compact UsdPreviewSurface-like texture
     * set. Baking here gives OpenGL the same diffuse texture contract that
     * Metal/Vulkan consume before backend-specific shading starts. */
    if (has_body || has_handle || has_cap) {
        baked_idx = bake_mdl_body_masked_albedo_texture(
            mc, stage, scene_dir, albedo_ref, mask_ref, body, handle, cap);
    } else if (has_color_albedo) {
        baked_idx = bake_mdl_masked_albedo_texture(
            mc, stage, scene_dir, albedo_ref, mask_ref, color_albedo);
    }

    if (baked_idx < 0) return 0;

    p->tex_indices[TEX_DIFFUSE_COLOR] = baked_idx;
    p->base_color[0] = 1.0f;
    p->base_color[1] = 1.0f;
    p->base_color[2] = 1.0f;
    p->v_flip = 0;
    return 1;
}

static void bind_metal_black_paint_remote_textures(MaterialCollection* mc,
                                                   void* stage,
                                                   MaterialParams* p,
                                                   const char* scene_dir)
{
    static const char* kBaseColor =
        "https://omniverse-content-production.s3.us-west-2.amazonaws.com/"
        "Materials/2023_2_1/Automotive/Surfacing/ov_metal_a_01/"
        "ov_metal_a_01_basecolor.jpg";
    static const char* kOrm =
        "https://omniverse-content-production.s3.us-west-2.amazonaws.com/"
        "Materials/2023_2_1/Automotive/Surfacing/ov_metal_a_01/"
        "ov_metal_a_01_orm.jpg";
    static const char* kNormal =
        "https://omniverse-content-production.s3.us-west-2.amazonaws.com/"
        "Materials/2023_2_1/Automotive/Surfacing/ov_metal_a_01/"
        "ov_metal_a_01_normal.jpg";

    int base_idx = find_or_add_texture(mc, scene_dir, kBaseColor, stage);
    if (base_idx >= 0) {
        p->tex_indices[TEX_DIFFUSE_COLOR] = base_idx;
        p->base_color[0] = 1.0f;
        p->base_color[1] = 1.0f;
        p->base_color[2] = 1.0f;
    }

    int normal_idx = find_or_add_texture(mc, scene_dir, kNormal, stage);
    if (normal_idx >= 0)
        p->tex_indices[TEX_NORMAL] = normal_idx;

    int orm_idx = find_or_add_texture(mc, scene_dir, kOrm, stage);
    if (orm_idx >= 0) {
        p->tex_indices[TEX_OCCLUSION] = orm_idx;
        p->tex_indices[TEX_ROUGHNESS] = orm_idx;
        p->tex_indices[TEX_METALLIC] = orm_idx;
    }

    if (getenv("NUSD_MAT_DIAG")) {
        fprintf(stderr,
                "[mat_diag] Metal_Black_Paint remote texture fallback: "
                "base=%d normal=%d orm=%d\n",
                base_idx, normal_idx, orm_idx);
    }
}

static void mdl_apply_color_param(MaterialParams* p, const char* lname,
                                  const float c[3])
{
    const char* emissive_keys[] = {"emiss", "emission"};
    const char* trans_keys[] = {"transmission_color", "transmission_tint"};
    const char* spec_keys[] = {"specular_color", "specular_tint"};
    const char* base_keys[] = {"base", "diffuse", "albedo", "tint", "color"};
    if (mdl_has_any(lname, emissive_keys, 2)) {
        p->emissive_color[0] = c[0];
        p->emissive_color[1] = c[1];
        p->emissive_color[2] = c[2];
        if (p->emissive_color[3] <= 0.0f) p->emissive_color[3] = 1.0f;
    } else if (mdl_has_any(lname, trans_keys, 2)) {
        p->transmission_color[0] = c[0];
        p->transmission_color[1] = c[1];
        p->transmission_color[2] = c[2];
    } else if (mdl_has_any(lname, spec_keys, 2)) {
        p->specular_color[0] = c[0];
        p->specular_color[1] = c[1];
        p->specular_color[2] = c[2];
        p->specular_color[3] = 1.0f;
        p->use_specular_workflow = 1;
    } else if (mdl_has_any(lname, base_keys, 5)) {
        p->base_color[0] = c[0];
        p->base_color[1] = c[1];
        p->base_color[2] = c[2];
    }
}

static void mdl_apply_float_param(MaterialParams* p, const char* lname, float v)
{
    if (strstr(lname, "texture") || strstr(lname, "image") ||
        strstr(lname, "map") || strstr(lname, "influence") ||
        strstr(lname, "enable") || strstr(lname, "flip_") ||
        strstr(lname, "tiling") || strstr(lname, "rotation") ||
        strstr(lname, "desaturation"))
        return;

    if (strstr(lname, "roughnessmin") || strstr(lname, "roughness_min") ||
        strstr(lname, "roughnessmax") || strstr(lname, "roughness_max"))
        return;

    const char* coat_rough_keys[] = {"clearcoat_roughness", "coat_roughness"};
    const char* rough_keys[] = {"roughness", "roughtness"};
    const char* metal_keys[] = {"metallic", "metalness", "base_metalness"};
    const char* spec_keys[] = {"specular_mdl", "specular"};
    const char* threshold_keys[] = {"opacity_threshold", "alpha_threshold", "cutout_threshold"};
    const char* opacity_keys[] = {"opacity", "alpha", "cutout"};
    const char* coat_keys[] = {"clearcoat", "coat_weight"};
    const char* ior_keys[] = {"ior", "specular_ior"};
    const char* normal_keys[] = {"normal_scale", "bump_factor", "bump_scale"};
    const char* trans_keys[] = {"transmission"};
    const char* emit_keys[] = {"emission", "emissive", "luminance", "intensity"};
    if (mdl_has_any(lname, coat_rough_keys, 2)) p->clearcoat_roughness = v;
    else if (mdl_has_any(lname, rough_keys, 2)) p->roughness = v;
    else if (mdl_has_any(lname, metal_keys, 3)) p->metallic = v;
    else if (mdl_has_any(lname, spec_keys, 2) &&
             !mdl_has_any(lname, ior_keys, 2)) {
        float f0 = fmaxf(0.0f, fminf(1.0f, v * 0.08f));
        p->specular_color[0] = f0;
        p->specular_color[1] = f0;
        p->specular_color[2] = f0;
        p->specular_color[3] = 1.0f;
        p->use_specular_workflow = 1;
    }
    else if (mdl_has_any(lname, threshold_keys, 3)) p->opacity_threshold = v;
    else if (mdl_has_any(lname, opacity_keys, 3)) p->opacity = v;
    else if (mdl_has_any(lname, coat_keys, 2)) p->clearcoat = v;
    else if (mdl_has_any(lname, ior_keys, 2)) p->ior = v;
    else if (mdl_has_any(lname, normal_keys, 3)) p->normal_scale = v;
    else if (mdl_has_any(lname, trans_keys, 1)) p->transmission_weight = v;
    else if (mdl_has_any(lname, emit_keys, 4)) p->emissive_color[3] = v;
}

static void mdl_apply_authored_shader_inputs(NanousdPrim shader_prim,
                                             MaterialParams* p)
{
    if (!shader_prim || !p) return;
    int na = nanousd_nattribs(shader_prim);
    for (int a = 0; a < na; a++) {
        const char* aname = nanousd_attribname(shader_prim, a);
        if (!aname || strncmp(aname, "inputs:", 7) != 0) continue;
        if (strstr(aname, ".connect")) continue;

        char lname[256];
        mdl_lower_copy(lname, sizeof(lname), aname + 7);
        float v4[4];
        float v3[3];
        if (nanousd_attribv4f(shader_prim, aname, v4)) {
            mdl_apply_color_param(p, lname, v4);
            continue;
        }
        if (nanousd_attribv3f(shader_prim, aname, v3)) {
            mdl_apply_color_param(p, lname, v3);
            continue;
        }

        int ok = 0;
        float f = nanousd_attribf(shader_prim, aname, &ok);
        if (ok) {
            mdl_apply_float_param(p, lname, f);
            continue;
        }
        int i = nanousd_attribi(shader_prim, aname, &ok);
        if (ok) {
            mdl_apply_float_param(p, lname, (float)i);
            continue;
        }
        int b = nanousd_attribb(shader_prim, aname, &ok);
        if (ok)
            mdl_apply_float_param(p, lname, b ? 1.0f : 0.0f);
    }
}

static int mdl_read_float_any(NanousdPrim shader_prim,
                              const char* const* names,
                              int count,
                              float* out)
{
    if (!shader_prim || !names || !out) return 0;
    for (int i = 0; i < count; ++i) {
        int ok = 0;
        float v = nanousd_attribf(shader_prim, names[i], &ok);
        if (ok) {
            *out = v;
            return 1;
        }
    }
    return 0;
}

static void mdl_apply_isaac_roughness_remap(NanousdPrim shader_prim,
                                            MaterialParams* p)
{
    if (!shader_prim || !p) return;

    float rmin = 0.0f, rmax = 0.0f;
    const char* const min_names[] = {
        "inputs:RoughnessMin",
        "inputs:Roughness_Min",
        "inputs:roughnessMin",
        "inputs:roughness_min",
    };
    const char* const max_names[] = {
        "inputs:RoughnessMax",
        "inputs:Roughness_Max",
        "inputs:roughnessMax",
        "inputs:roughness_max",
    };
    int okmin = mdl_read_float_any(
        shader_prim, min_names, (int)(sizeof(min_names) / sizeof(min_names[0])), &rmin);
    int okmax = mdl_read_float_any(
        shader_prim, max_names, (int)(sizeof(max_names) / sizeof(max_names[0])), &rmax);

    if (okmin && okmax) {
        p->roughness = 0.5f * (rmin + rmax);
        p->roughness_tex_bias = rmin;
        p->roughness_tex_scale = rmax - rmin;
    } else if (okmax) {
        p->roughness = rmax;
    } else if (okmin) {
        p->roughness = rmin;
    }

    {
        int ok = 0;
        float f = nanousd_attribf(shader_prim, "inputs:Roughness", &ok);
        if (ok) p->roughness = f;
    }
}

static int collect_mdl_shader_inputs(NanousdPrim shader_prim,
                                     NusdMdlInput* inputs,
                                     char names[][256],
                                     int max_inputs)
{
    if (!shader_prim || !inputs || !names || max_inputs <= 0) return 0;
    int count = 0;
    int na = nanousd_nattribs(shader_prim);
    for (int a = 0; a < na && count < max_inputs; a++) {
        const char* aname = nanousd_attribname(shader_prim, a);
        if (!aname || strncmp(aname, "inputs:", 7) != 0) continue;
        if (strstr(aname, ".connect")) continue;
        const char* pname = aname + 7;
        if (!pname[0]) continue;

        NusdMdlInput input;
        memset(&input, 0, sizeof(input));

        float v4[4];
        float v3[3];
        int ok = 0;
        if (nanousd_attribv4f(shader_prim, aname, v4)) {
            input.kind = NUSD_MDL_INPUT_COLOR;
            input.values[0] = v4[0];
            input.values[1] = v4[1];
            input.values[2] = v4[2];
            input.values[3] = v4[3];
        } else if (nanousd_attribv3f(shader_prim, aname, v3)) {
            input.kind = NUSD_MDL_INPUT_COLOR;
            input.values[0] = v3[0];
            input.values[1] = v3[1];
            input.values[2] = v3[2];
            input.values[3] = 1.0f;
        } else {
            int b = nanousd_attribb(shader_prim, aname, &ok);
            if (ok) {
                input.kind = NUSD_MDL_INPUT_BOOL;
                input.int_value = b ? 1 : 0;
                input.values[0] = b ? 1.0f : 0.0f;
            } else {
                int i = nanousd_attribi(shader_prim, aname, &ok);
                if (ok) {
                    input.kind = NUSD_MDL_INPUT_INT;
                    input.int_value = i;
                    input.values[0] = (float)i;
                } else {
                    float f = nanousd_attribf(shader_prim, aname, &ok);
                    if (!ok) continue;
                    input.kind = NUSD_MDL_INPUT_FLOAT;
                    input.values[0] = f;
                }
            }
        }

        snprintf(names[count], 256, "%s", pname);
        inputs[count] = input;
        inputs[count].name = names[count];
        count++;
    }
    return count;
}

static int apply_mdl_sdk_decoded_material(MaterialParams* p,
                                          MaterialCollection* mc,
                                          void* stage,
                                          const char* mdl_path,
                                          const char* subidentifier,
                                          const char* scene_dir,
                                          const NusdMdlInput* inputs,
                                          int input_count)
{
    if (!p || !mdl_path || !mdl_path[0]) return 0;
    char resolved_mdl[1024];
    resolve_texture_path_stage(scene_dir, mdl_path, stage,
                               resolved_mdl, sizeof(resolved_mdl));
    const char* bridge_path = resolved_mdl[0] ? resolved_mdl : mdl_path;
    NusdMdlDecoded decoded;
    if (!nusd_mdl_bridge_decode_with_inputs(
            bridge_path, subidentifier, scene_dir, inputs, input_count, &decoded))
        return 0;

    if (decoded.has_base_color) {
        p->base_color[0] = decoded.base_color[0];
        p->base_color[1] = decoded.base_color[1];
        p->base_color[2] = decoded.base_color[2];
        p->base_color[3] = decoded.base_color[3];
    }
    if (decoded.has_emissive_color) {
        p->emissive_color[0] = decoded.emissive_color[0];
        p->emissive_color[1] = decoded.emissive_color[1];
        p->emissive_color[2] = decoded.emissive_color[2];
        p->emissive_color[3] = decoded.emissive_color[3];
    }
    if (decoded.has_metallic) p->metallic = decoded.metallic;
    if (decoded.has_roughness) p->roughness = decoded.roughness;
    if (decoded.has_opacity) {
        p->opacity = decoded.opacity;
        p->base_color[3] = decoded.opacity;
    }
    if (decoded.has_ior) p->ior = decoded.ior;
    if (decoded.has_clearcoat) p->clearcoat = decoded.clearcoat;
    if (decoded.has_clearcoat_roughness)
        p->clearcoat_roughness = decoded.clearcoat_roughness;
    if (decoded.has_normal_scale) p->normal_scale = decoded.normal_scale;
    if (decoded.has_transmission_color) {
        p->transmission_color[0] = decoded.transmission_color[0];
        p->transmission_color[1] = decoded.transmission_color[1];
        p->transmission_color[2] = decoded.transmission_color[2];
        p->transmission_color[3] = decoded.transmission_color[3];
    }
    if (decoded.has_transmission_weight)
        p->transmission_weight = decoded.transmission_weight;
    if (decoded.has_transmission_ior)
        p->transmission_ior = decoded.transmission_ior;
    if (decoded.has_specular_color) {
        p->specular_color[0] = decoded.specular_color[0];
        p->specular_color[1] = decoded.specular_color[1];
        p->specular_color[2] = decoded.specular_color[2];
        p->specular_color[3] = decoded.specular_color[3];
    }
    if (decoded.has_specular_workflow)
        p->use_specular_workflow = decoded.use_specular_workflow;
    for (int i = 0; i < decoded.texture_count; ++i) {
        const NusdMdlDecodedTexture* tex = &decoded.textures[i];
        const char* tex_ref = tex->file_path[0] ? tex->file_path : tex->db_name;
        const char* hint = tex->name_hint[0] ? tex->name_hint : tex_ref;
        if (tex_ref && tex_ref[0])
            mdl_assign_texture(mc, stage, p, scene_dir, hint, tex_ref);
    }
    return 1;
}

static void mdl_parse_param_segment(MaterialCollection* mc, void* stage,
                                    MaterialParams* p, const char* anchor_dir,
                                    const char* begin, const char* end,
                                    int apply_constant_defaults)
{
    size_t n = (size_t)(end - begin);
    if (n == 0 || n >= 4096) return;
    char param[4096];
    memcpy(param, begin, n);
    param[n] = '\0';

    char name[256];
    if (!mdl_param_name_before_equal(param, name, sizeof(name))) return;
    char lname[256];
    mdl_lower_copy(lname, sizeof(lname), name);

    char tex_ref[1024];
    if (mdl_parse_texture_default(param, tex_ref, sizeof(tex_ref))) {
        mdl_assign_texture(mc, stage, p, anchor_dir, name, tex_ref);
        return;
    }

    if (!apply_constant_defaults) return;

    float c[3];
    if (mdl_parse_color_default(param, c)) {
        mdl_apply_color_param(p, lname, c);
        return;
    }

    float f = 0.0f;
    if (mdl_parse_float_default(param, &f))
        mdl_apply_float_param(p, lname, f);
}

static void mdl_parse_param_block(MaterialCollection* mc, void* stage,
                                  MaterialParams* p, const char* anchor_dir,
                                  const char* begin, const char* end,
                                  int apply_constant_defaults)
{
    const char* seg = begin;
    int in_string = 0, escape = 0, paren = 0, bracket = 0, brace = 0;
    for (const char* q = begin; q < end; q++) {
        char c = *q;
        if (in_string) {
            if (escape) escape = 0;
            else if (c == '\\') escape = 1;
            else if (c == '"') in_string = 0;
            continue;
        }
        if (c == '"') { in_string = 1; continue; }
        if (c == '(') paren++;
        else if (c == ')' && paren > 0) paren--;
        else if (c == '[') bracket++;
        else if (c == ']' && bracket > 0) bracket--;
        else if (c == '{') brace++;
        else if (c == '}' && brace > 0) brace--;
        else if (c == ',' && paren == 0 && bracket == 0 && brace == 0) {
            mdl_parse_param_segment(mc, stage, p, anchor_dir, seg, q,
                                    apply_constant_defaults);
            seg = q + 1;
        }
    }
    mdl_parse_param_segment(mc, stage, p, anchor_dir, seg, end,
                            apply_constant_defaults);
}

static void mdl_scan_texture_literals(MaterialCollection* mc, void* stage,
                                      MaterialParams* p, const char* anchor_dir,
                                      const char* body)
{
    const char* q = body;
    while ((q = strstr(q, "texture_2d")) != NULL) {
        char tex_ref[1024];
        if (!mdl_first_quoted_after(q, tex_ref, sizeof(tex_ref))) {
            q += 10;
            continue;
        }
        const char* line = q;
        while (line > body && *(line - 1) != '\n') line--;
        char hint[512];
        size_t n = (size_t)(q - line);
        if (n >= sizeof(hint)) n = sizeof(hint) - 1;
        memcpy(hint, line, n);
        hint[n] = '\0';
        mdl_assign_texture(mc, stage, p, anchor_dir, hint, tex_ref);
        q += 10;
    }
}

static int mdl_resolve_source_asset(const char* scene_dir, const char* mdl_path,
                                    void* stage, char* abs_mdl, size_t abs_size)
{
    if (!mdl_path || !mdl_path[0] || !abs_mdl || abs_size == 0) return 0;
    resolve_texture_path_stage(scene_dir, mdl_path, stage, abs_mdl, abs_size);
    return file_exists(abs_mdl);
}

static void apply_generic_mdl_source_material(MaterialCollection* mc, void* stage,
                                              NanousdPrim shader_prim,
                                              MaterialParams* p,
                                              const char* scene_dir,
                                              const char* mdl_path,
                                              int apply_constant_defaults)
{
    if (!mc || !shader_prim || !p || !scene_dir || !mdl_path || !mdl_path[0])
        return;

    char mdl_path_copy[1024];
    snprintf(mdl_path_copy, sizeof(mdl_path_copy), "%s", mdl_path);

    char abs_mdl[1024];
    if (!mdl_resolve_source_asset(scene_dir, mdl_path_copy, stage,
                                  abs_mdl, sizeof(abs_mdl)))
        return;

    long sz = 0;
    const char* body = mdl_cache_get(abs_mdl, &sz);
    if (!body || sz <= 0) return;

    if (strstr(body, "CustomizedUV0_mdl") &&
        strstr(body, "1.0-state::texture_coordinate(0).y")) {
        /* Same convention as Metal/Vulkan: several IsaacSim MDLs first flip
         * the USD UV (`CustomizedUV0 = (u, 1-v)`) and then sample
         * `1 - CustomizedUV0.y`, which cancels back to the original V. Keep
         * OpenGL from applying an extra backend-level v_flip on top. */
        p->v_flip = 0;
    }

    char mdl_dir[1024];
    snprintf(mdl_dir, sizeof(mdl_dir), "%s", abs_mdl);
    char* last = strrchr(mdl_dir, '/');
    if (last) *last = '\0';

    int ok = 0;
    const char* sub = nanousd_attrib_token(shader_prim,
                                           "info:mdl:sourceAsset:subIdentifier",
                                           &ok);
    char material_name[256];
    mdl_material_name_from_subidentifier(ok ? sub : NULL,
                                         material_name, sizeof(material_name));

    const char* begin = NULL;
    const char* end = NULL;
    if (mdl_find_export_material_params(body, material_name, &begin, &end))
        mdl_parse_param_block(mc, stage, p, mdl_dir, begin, end,
                              apply_constant_defaults);

    mdl_scan_texture_literals(mc, stage, p, mdl_dir, body);

    if (mdl_source_has_nonzero_emissive_color(body)) {
        char emissive_base[1024] = "";
        char emissive_mask[1024] = "";
        const char* const emissive_keywords[] = {
            "emiss",
            "stripe",
            "strip",
            "glow",
            "light",
        };
        if (p->tex_indices[TEX_DIFFUSE_COLOR] >= 0 &&
            p->tex_indices[TEX_DIFFUSE_COLOR] < mc->ntextures) {
            snprintf(emissive_base, sizeof(emissive_base), "%s",
                     mc->textures[p->tex_indices[TEX_DIFFUSE_COLOR]].path);
        } else {
            mdl_find_first_texture_literal_for_slot(
                body, TEX_DIFFUSE_COLOR, emissive_base, sizeof(emissive_base));
        }
        mdl_find_texture_literal_with_keywords(
            body, emissive_keywords,
            (int)(sizeof(emissive_keywords) / sizeof(emissive_keywords[0])),
            emissive_mask, sizeof(emissive_mask));

        float product_scale = 1.0f;
        const char* const param_names[] = {"inputs:Param"};
        if (!mdl_read_float_any(shader_prim, param_names, 1, &product_scale) &&
            begin && end) {
            mdl_find_named_float_param_default(begin, end, "Param",
                                               &product_scale);
        }

        int baked_idx = bake_mdl_emissive_product_texture(
            mc, stage, mdl_dir, emissive_base, emissive_mask, 1.0f);
        if (baked_idx >= 0) {
            p->tex_indices[TEX_EMISSIVE_COLOR] = baked_idx;
            p->emissive_color[0] = 0.0f;
            p->emissive_color[1] = 0.0f;
            p->emissive_color[2] = 0.0f;
            p->emissive_color[3] = product_scale;
        }
    }

    if (apply_constant_defaults)
        mdl_apply_authored_shader_inputs(shader_prim, p);

    if (getenv("NUSD_MAT_DIAG")) {
        const char* pp = nanousd_path(shader_prim);
        fprintf(stderr,
                "[mat_diag] generic MDL shader=%s asset=%s material=%s params=%s\n",
                pp ? pp : "?", mdl_path_copy,
                material_name[0] ? material_name : "<first>",
                begin ? "yes" : "no");
    }
}

static int resolve_surface_shader(NanousdStage stage, NanousdPrim mat_prim,
                                  NanousdPrim* out_shader)
{
    if (out_shader) *out_shader = NULL;
    if (!stage || !mat_prim || !out_shader) return 0;

    const char* outputs[] = {
        "outputs:surface",
        "outputs:mtlx:surface",
        "outputs:mdl:surface",
    };
    for (int i = 0; i < (int)(sizeof(outputs) / sizeof(outputs[0])); i++) {
        const char* name = outputs[i];
        char target[1024];
        target[0] = '\0';

        int ntargets = nanousd_nreltargets(mat_prim, name);
        if (ntargets > 0) {
            const char* rel = nanousd_reltarget(mat_prim, name, 0);
            if (rel && rel[0])
                snprintf(target, sizeof(target), "%s", rel);
        }
        if (!target[0] &&
            first_connection(mat_prim, name, target, sizeof(target)) == 0)
            continue;

        char shader_path[512], prop[128];
        if (!split_connection_path(target, shader_path, sizeof(shader_path),
                                   prop, sizeof(prop)))
            continue;
        NanousdPrim shader = nanousd_primpath(stage, shader_path);
        if (shader) {
            *out_shader = shader;
            return 1;
        }
    }
    return 0;
}

static int material_base_color_is_default_gray(const MaterialParams* p)
{
    return p &&
           p->base_color[0] == 0.7f &&
           p->base_color[1] == 0.7f &&
           p->base_color[2] == 0.7f;
}

static int shader_is_direct_child_of(NanousdPrim shader, NanousdPrim mat_prim)
{
    if (!shader || !mat_prim) return 0;
    NanousdPrim parent = nanousd_parent(shader);
    if (!parent) return 0;
    const char* pp = nanousd_path(parent);
    const char* mp = nanousd_path(mat_prim);
    int ret = (pp && mp && strcmp(pp, mp) == 0);
    nanousd_freeprim(parent);
    return ret;
}

static int try_extract_connected_surface_shader(MaterialCollection* mc,
                                                NanousdStage stage,
                                                NanousdPrim mat_prim,
                                                NanousdPrim shader_prim,
                                                MaterialParams* p,
                                                const char* scene_dir)
{
    if (!mc || !stage || !mat_prim || !shader_prim || !p || !scene_dir)
        return 0;

    int ok = 0;
    const char* info_id = nanousd_attrib_token(shader_prim, "info:id", &ok);
    if (ok && (mtlx_info_is_standard_surface(info_id) ||
               mtlx_info_is_open_pbr(info_id))) {
        read_materialx_usd_surface(mc, stage, mat_prim, shader_prim, p,
                                   scene_dir, mtlx_info_is_open_pbr(info_id));
        return 1;
    }

    int mdl_ok = 0;
    const char* mdl_path = nanousd_attribasset(shader_prim,
                                               "info:mdl:sourceAsset",
                                               &mdl_ok);
    if (!mdl_ok || !mdl_path || !mdl_path[0]) return 0;

    p->v_flip = 1;
    int sub_ok = 0;
    const char* sub = nanousd_attrib_token(
        shader_prim, "info:mdl:sourceAsset:subIdentifier", &sub_ok);
    char sub_buf[256];
    sub_buf[0] = '\0';
    if (sub_ok && sub) snprintf(sub_buf, sizeof(sub_buf), "%s", sub);

    NusdMdlInput mdl_inputs[128];
    char mdl_input_names[128][256];
    int n_mdl_inputs = collect_mdl_shader_inputs(shader_prim, mdl_inputs,
                                                 mdl_input_names, 128);
    int mdl_sdk_applied = apply_mdl_sdk_decoded_material(
        p, mc, stage, mdl_path, sub_buf[0] ? sub_buf : NULL, scene_dir,
        mdl_inputs, n_mdl_inputs);

    /* Authored USD shader inputs are explicit overrides. Keep applying them
     * even when the SDK produced a partial distilled result, because the flat
     * bridge may not expose every decoded MDL field yet. */
    mdl_apply_authored_shader_inputs(shader_prim, p);
    apply_generic_mdl_source_material(mc, stage, shader_prim, p, scene_dir,
                                      mdl_path, !mdl_sdk_applied);
    mdl_apply_isaac_roughness_remap(shader_prim, p);
    mdl_apply_isaac_mask_bakes(mc, stage, shader_prim, p, scene_dir);
    return 1;
}

/* ---- Extract material from a USD Material prim ---- */

static void extract_material(MaterialCollection* mc, void* stage,
                              NanousdPrim mat_prim, const char* scene_dir, int mat_idx)
{
    NanousdStage nstage = (NanousdStage)stage;
    SceneMaterial* mat = &mc->materials[mat_idx];
    MaterialParams* p = &mat->params;

    /* Defaults */
    p->base_color[0] = 0.7f; p->base_color[1] = 0.7f;
    p->base_color[2] = 0.7f; p->base_color[3] = 1.0f;
    p->emissive_color[0] = 0.0f; p->emissive_color[1] = 0.0f;
    p->emissive_color[2] = 0.0f; p->emissive_color[3] = 1.0f;
    p->metallic = 0.0f;
    p->roughness = 0.5f;
    p->opacity = 1.0f;
    p->ior = 1.5f;
    p->occlusion = 1.0f;
    p->clearcoat = 0.0f;
    p->clearcoat_roughness = 0.0f;
    p->normal_scale = 1.0f;
    p->udim_scale_u = 1.0f;
    p->udim_scale_v = 1.0f;
    p->use_vertex_color = 0;
    p->v_flip = 0;
    p->opacity_threshold = 0.0f;  /* default: alpha-blend mode (no discard) */
    p->transmission_color[0] = 1.0f;
    p->transmission_color[1] = 1.0f;
    p->transmission_color[2] = 1.0f;
    p->transmission_color[3] = 1.0f;
    p->transmission_weight = 0.0f;
    p->transmission_ior = 0.0f;
    p->use_specular_workflow = 0;
    p->specular_color[0] = 0.0f;
    p->specular_color[1] = 0.0f;
    p->specular_color[2] = 0.0f;
    p->specular_color[3] = 0.0f;
    p->roughness_tex_scale = 1.0f;
    p->roughness_tex_bias = 0.0f;
    for (int i = 0; i < MAX_MATERIAL_TEXTURES; i++)
        p->tex_indices[i] = -1;

    const char* mat_path = nanousd_path(mat_prim);
    if (mat_path) {
        snprintf(mat->prim_path, sizeof(mat->prim_path), "%s", mat_path);
        const char* slash = strrchr(mat_path, '/');
        snprintf(mat->name, sizeof(mat->name), "%s",
                 slash ? slash + 1 : mat_path);
    }

    char preferred_shader_path[512];
    preferred_shader_path[0] = '\0';
    NanousdPrim surface_shader = NULL;
    if (resolve_surface_shader(nstage, mat_prim, &surface_shader)) {
        const char* sp = nanousd_path(surface_shader);
        if (sp && sp[0])
            snprintf(preferred_shader_path, sizeof(preferred_shader_path),
                     "%s", sp);
        int use_connected = !shader_is_direct_child_of(surface_shader, mat_prim);
        if (use_connected &&
            try_extract_connected_surface_shader(mc, nstage, mat_prim,
                                                 surface_shader, p,
                                                 scene_dir)) {
            nanousd_freeprim(surface_shader);
            return;
        }
        nanousd_freeprim(surface_shader);
    }

    /* Find the Shader child prim (UsdPreviewSurface or OmniPBR) */
    int nch = nanousd_nchildren(mat_prim);
    for (int c = 0; c < nch; c++) {
        NanousdPrim child = nanousd_child(mat_prim, c);
        if (!child) continue;

        /* Accept both schema-resolved Shader (`isa`) and an exact typename
         * match: some packaged UsdPreviewSurface assets (e.g. Apple USDZ)
         * carry Shader-typed prims that nanousd's schema registry does not
         * resolve via isa(), which previously dropped all their textures.
         * Shader is a leaf type, so an exact typename match is equivalent
         * (see the Scope/Material typename fast-paths below). The
         * preferred_shader_path filter still selects the surface shader. */
        const char* child_type = nanousd_typename(child);
        if (nanousd_isa(child, "Shader") ||
            (child_type && strcmp(child_type, "Shader") == 0)) {
            if (preferred_shader_path[0]) {
                const char* cp = nanousd_path(child);
                if (!cp || strcmp(cp, preferred_shader_path) != 0) {
                    nanousd_freeprim(child);
                    continue;
                }
            }

            int ok;
            float v3[3];

            const char* info_id = nanousd_attrib_token(child, "info:id", &ok);
            if (ok && (mtlx_info_is_standard_surface(info_id) ||
                       mtlx_info_is_open_pbr(info_id))) {
                read_materialx_usd_surface(mc, nstage, mat_prim, child, p,
                                           scene_dir,
                                           mtlx_info_is_open_pbr(info_id));
                nanousd_freeprim(child);
                return;
            }

            /* Detect MDL upfront. MDL materials don't have UsdPreviewSurface-
             * style connections or UsdUVTexture child prims, so we skip
             * those probes entirely below — saves ~7 nreltargets +
             * nchildren walks × per-material × 2083 in the warehouse. */
            int mdl_ok_early = 0;
            const char* mdl_path_early = nanousd_attribasset(
                child, "info:mdl:sourceAsset", &mdl_ok_early);
            int is_mdl = (mdl_ok_early && mdl_path_early && mdl_path_early[0]);
            char mdl_path_buf[1024];
            mdl_path_buf[0] = '\0';
            int mdl_sdk_applied = 0;
            if (is_mdl) {
                snprintf(mdl_path_buf, sizeof(mdl_path_buf), "%s",
                         mdl_path_early);
                p->v_flip = 1;
                int sub_ok = 0;
                const char* sub = nanousd_attrib_token(
                    child, "info:mdl:sourceAsset:subIdentifier", &sub_ok);
                char sub_buf[256];
                sub_buf[0] = '\0';
                if (sub_ok && sub)
                    snprintf(sub_buf, sizeof(sub_buf), "%s", sub);
                NusdMdlInput mdl_inputs[128];
                char mdl_input_names[128][256];
                int n_mdl_inputs = collect_mdl_shader_inputs(
                    child, mdl_inputs, mdl_input_names, 128);
                mdl_sdk_applied = apply_mdl_sdk_decoded_material(
                    p, mc, stage, mdl_path_buf, sub_buf[0] ? sub_buf : NULL, scene_dir,
                    mdl_inputs, n_mdl_inputs);
            }

            /* Read base color. Three sources, in OmniPBR / UsdPreviewSurface
             * priority order:
             *   inputs:diffuseColor             (UsdPreviewSurface)
             *   inputs:diffuse_color_constant   (OmniPBR / MDL constant)
             *   inputs:diffuse_tint             (OmniPBR multiplier)
             *
             * Many warehouse assets author ONLY diffuse_tint and rely on
             * the MDL default for diffuse_color_constant (= white). If we
             * read only the first two, those assets render as the grey
             * fallback (e.g. "yellowpail" came out white instead of yellow).
             * Use whichever explicit color is set, then apply tint as a
             * multiplier so an asset that sets both still composes correctly. */
            float base[3] = { 1.0f, 1.0f, 1.0f };
            int  base_set = 0;
            if (nanousd_attribv3f(child, "inputs:diffuseColor", base) ||
                nanousd_attribv3f(child, "inputs:diffuse_color_constant", base)) {
                base_set = 1;
            }
            float tint[3] = { 1.0f, 1.0f, 1.0f };
            int  tint_set = nanousd_attribv3f(child, "inputs:diffuse_tint", tint) ? 1 : 0;
            if (base_set || tint_set) {
                p->base_color[0] = base[0] * tint[0];
                p->base_color[1] = base[1] * tint[1];
                p->base_color[2] = base[2] * tint[2];
            }

            ok = 0;
            float met = nanousd_attribf(child, "inputs:metallic", &ok);
            if (!ok) {
                met = nanousd_attribf(child, "inputs:metallic_constant", &ok);
                if (!ok) met = 0.0f;
            }
            p->metallic = met;

            ok = 0;
            float rough = nanousd_attribf(child, "inputs:roughness", &ok);
            if (!ok) {
                rough = nanousd_attribf(child, "inputs:roughness_constant", &ok);
                if (!ok) rough = 0.5f;
            }
            p->roughness = rough;

            ok = 0;
            float opac = nanousd_attribf(child, "inputs:opacity", &ok);
            if (ok) p->opacity = opac;

            /* opacityThreshold > 0 enables alpha-cutout (discard); 0 (default)
             * leaves the material in alpha-blend mode. The shader gates its
             * discard branch on > 0 so AlphaBlendModeTest's blend variants
             * don't lose their opaque areas. */
            ok = 0;
            float othr = nanousd_attribf(child, "inputs:opacityThreshold", &ok);
            if (ok) p->opacity_threshold = othr;

            ok = 0;
            float ior = nanousd_attribf(child, "inputs:ior", &ok);
            if (ok) p->ior = ior;

            if (nanousd_attribv3f(child, "inputs:emissiveColor", v3) ||
                nanousd_attribv3f(child, "inputs:emissive_color", v3)) {
                p->emissive_color[0] = v3[0];
                p->emissive_color[1] = v3[1];
                p->emissive_color[2] = v3[2];
                ok = 0;
                int bok = 0;
                int b = nanousd_attribb(child, "inputs:enable_emission", &bok);
                float enable_emission = nanousd_attribf(child, "inputs:enable_emission", &ok);
                int emission_enabled = bok ? (b != 0) : (!ok || enable_emission > 0.5f);
                ok = 0;
                float intensity = nanousd_attribf(child, "inputs:emissive_intensity", &ok);
                p->emissive_color[3] = emission_enabled ? (ok ? intensity : 1.0f) : 0.0f;
            }

            /* --- Connection-based texture resolution (like OpenUSD) ---
             * Follow explicit .connect attributes on shader inputs to find
             * upstream UsdUVTexture prims, then read their inputs:file.
             * This is authoritative — no name guessing needed.
             * Skipped for MDL: those materials don't have UsdUVTexture
             * connections, and probing is expensive on the warehouse. */
            if (!is_mdl) {
                struct { const char* input; int slot; } conn_inputs[] = {
                    {"inputs:diffuseColor",  TEX_DIFFUSE_COLOR},
                    {"inputs:normal",        TEX_NORMAL},
                    {"inputs:roughness",     TEX_ROUGHNESS},
                    {"inputs:metallic",      TEX_METALLIC},
                    {"inputs:emissiveColor", TEX_EMISSIVE_COLOR},
                    {"inputs:occlusion",     TEX_OCCLUSION},
                    {"inputs:opacity",       TEX_OPACITY},
                };
                int n_conn_inputs = (int)(sizeof(conn_inputs) / sizeof(conn_inputs[0]));

                for (int ci = 0; ci < n_conn_inputs; ci++) {
                    int nconn = nanousd_nconnections(child, conn_inputs[ci].input);
                    if (nconn <= 0) continue;

                    const char* conn_path = nanousd_connection(child, conn_inputs[ci].input, 0);
                    if (!conn_path || !conn_path[0]) continue;

                    /* conn_path is like:
                     * /Material/metallic_roughness_texture.outputs:b
                     * Extract prim path (before the dot) */
                    char prim_path_buf[512];
                    snprintf(prim_path_buf, sizeof(prim_path_buf), "%s", conn_path);
                    char* dot = strrchr(prim_path_buf, '.');
                    if (dot) *dot = '\0';

                    /* Look up the upstream texture prim */
                    NanousdPrim tex_prim = nanousd_primpath(stage, prim_path_buf);
                    if (!tex_prim) continue;

                    /* Read inputs:file from the texture prim */
                    ok = 0;
                    const char* tex_file = nanousd_attribasset(tex_prim, "inputs:file", &ok);
                    if (!ok || !tex_file || !tex_file[0]) {
                        ok = 0;
                        tex_file = nanousd_attribs(tex_prim, "inputs:file", &ok);
                    }
                    int slot_loaded = 0;
                    if (ok && tex_file && tex_file[0]) {
                        int tex_idx = find_or_add_texture(mc, scene_dir, tex_file, stage);
                        if (tex_idx >= 0) {
                            p->tex_indices[conn_inputs[ci].slot] = tex_idx;
                            slot_loaded = 1;
                        }
                    }
                    /* For the diffuse-color slot, fold UsdUVTexture
                     * inputs:scale into base_color so per-quad tints
                     * (TextureCoordinateTest yellow/orange/blue/green)
                     * survive into the shader's
                     * `tex_color * base_color` mix. When the upstream
                     * shader has no constant inputs:diffuseColor (only
                     * a connection), the loader defaulted base_color to
                     * (0.7, 0.7, 0.7) — that gray default is meant for
                     * untextured surfaces, so reset to white before
                     * applying scale. Mirrors the vulkan-renderer fix. */
                    if (conn_inputs[ci].slot == TEX_DIFFUSE_COLOR
                        && slot_loaded && !base_set) {
                        p->base_color[0] = 1.0f;
                        p->base_color[1] = 1.0f;
                        p->base_color[2] = 1.0f;
                    }
                    if (conn_inputs[ci].slot == TEX_DIFFUSE_COLOR && slot_loaded) {
                        float s4[4];
                        if (nanousd_attribv4f(tex_prim, "inputs:scale", s4)) {
                            p->base_color[0] *= s4[0];
                            p->base_color[1] *= s4[1];
                            p->base_color[2] *= s4[2];
                        }
                    }
                    nanousd_freeprim(tex_prim);
                }
            }

            /* --- Fallback: name-heuristic scan for materials without connections ---
             * Walk children of the Material for UsdUVTexture prims that weren't
             * reached via connections (e.g., OmniPBR with direct texture attrs).
             * Skipped for MDL — those materials don't have UsdUVTexture
             * children. */
            if (!is_mdl) {
                int nsh_ch = nanousd_nchildren(mat_prim);
                for (int tc = 0; tc < nsh_ch; tc++) {
                    NanousdPrim tex_child = nanousd_child(mat_prim, tc);
                    if (!tex_child) continue;

                    const char* tcp = nanousd_path(tex_child);
                    if (!tcp) { nanousd_freeprim(tex_child); continue; }

                    ok = 0;
                    const char* tex_file = nanousd_attribasset(tex_child, "inputs:file", &ok);
                    if (!ok || !tex_file || !tex_file[0]) {
                        ok = 0;
                        tex_file = nanousd_attribs(tex_child, "inputs:file", &ok);
                    }

                    if (ok && tex_file && tex_file[0]) {
                        int tex_idx = find_or_add_texture(mc, scene_dir, tex_file, stage);
                        /* Only fill empty slots via name heuristics */
                        if (p->tex_indices[TEX_DIFFUSE_COLOR] < 0 &&
                            (strstr(tcp, "diffuse") || strstr(tcp, "Diffuse") ||
                             strstr(tcp, "albedo") || strstr(tcp, "BaseColor"))) {
                            p->tex_indices[TEX_DIFFUSE_COLOR] = tex_idx;
                        }
                        if (p->tex_indices[TEX_NORMAL] < 0 &&
                            (strstr(tcp, "normal") || strstr(tcp, "Normal"))) {
                            p->tex_indices[TEX_NORMAL] = tex_idx;
                        }
                        if (p->tex_indices[TEX_ROUGHNESS] < 0 &&
                            (strstr(tcp, "roughness") || strstr(tcp, "Roughness"))) {
                            p->tex_indices[TEX_ROUGHNESS] = tex_idx;
                        }
                        if (p->tex_indices[TEX_METALLIC] < 0 &&
                            (strstr(tcp, "metallic") || strstr(tcp, "Metallic"))) {
                            p->tex_indices[TEX_METALLIC] = tex_idx;
                        }
                        if (p->tex_indices[TEX_EMISSIVE_COLOR] < 0 &&
                            (strstr(tcp, "emissive") || strstr(tcp, "Emissive"))) {
                            p->tex_indices[TEX_EMISSIVE_COLOR] = tex_idx;
                        }
                        if (p->tex_indices[TEX_OCCLUSION] < 0 &&
                            (strstr(tcp, "occlusion") || strstr(tcp, "ao"))) {
                            p->tex_indices[TEX_OCCLUSION] = tex_idx;
                        }
                    }
                    nanousd_freeprim(tex_child);
                }
            }

            /* --- MDL fallback ---
             * NVIDIA's SimReady warehouse assets commonly author only an
             * ``info:mdl:sourceAsset`` reference — no USD-side
             * inputs:diffuse_texture, no UsdUVTexture connections. The
             * MDL file itself names the textures inline as
             * ``<input>_texture: texture_2d("./relative/path.png", …)``.
             * Parse those lines so we can apply textures without an
             * MDL backend. Only fires when the USD-side connection /
             * top-level texture lookups below find nothing. */
            if (is_mdl) {
                /* mdl_path_early was read upfront — reuse rather than re-probing. */
                const char* mdl_path = mdl_path_buf;
                apply_generic_mdl_source_material(mc, stage, child, p,
                                                  scene_dir, mdl_path,
                                                  !mdl_sdk_applied);
                {
                    char abs_mdl[1024];
                    if (!mdl_resolve_source_asset(scene_dir, mdl_path, stage,
                                                  abs_mdl, sizeof(abs_mdl)))
                        abs_mdl[0] = '\0';
                    long sz = 0;
                    const char* buf = mdl_cache_get(abs_mdl, &sz);
                    if (buf) {
                        /* Compute the directory of the MDL — texture paths
                         * inside are ``./<rel>`` relative to the .mdl
                         * file's directory. */
                        char mdl_dir[1024];
                        snprintf(mdl_dir, sizeof(mdl_dir), "%s", abs_mdl);
                        char* last = strrchr(mdl_dir, '/');
                        if (last) *last = '\0';

                        struct { const char* key; int slot; } mdl_inputs[] = {
                            {"diffuse_texture",       TEX_DIFFUSE_COLOR},
                            {"normalmap_texture",     TEX_NORMAL},
                            {"reflectionroughness_texture", TEX_ROUGHNESS},
                            {"metallic_texture",      TEX_METALLIC},
                            {"ORM_texture",           TEX_ROUGHNESS},
                            {"emissive_color_texture",TEX_EMISSIVE_COLOR},
                            {"emissive_mask_texture", TEX_EMISSIVE_COLOR},
                            {"ao_texture",            TEX_OCCLUSION},
                            {"opacity_texture",       TEX_OPACITY},
                        };
                        int nmdl = (int)(sizeof(mdl_inputs)/sizeof(mdl_inputs[0]));
                        for (int mi = 0; mi < nmdl; mi++) {
                            if (p->tex_indices[mdl_inputs[mi].slot] >= 0) continue;
                            char needle[64];
                            snprintf(needle, sizeof(needle), "%s: texture_2d(\"",
                                     mdl_inputs[mi].key);
                            const char* hit = strstr(buf, needle);
                            if (!hit) continue;
                            const char* path_start = hit + strlen(needle);
                            const char* path_end = strchr(path_start, '"');
                            if (!path_end || path_end == path_start) continue;
                            size_t plen = (size_t)(path_end - path_start);
                            if (plen >= 512) continue;
                            char rel[512];
                            memcpy(rel, path_start, plen);
                            rel[plen] = '\0';
                            /* Strip leading ./ */
                            const char* relp = rel;
                            if (relp[0] == '.' && relp[1] == '/') relp += 2;
                            int tidx = find_or_add_texture(mc, mdl_dir, relp, stage);
                            if (tidx >= 0) {
                                p->tex_indices[mdl_inputs[mi].slot] = tidx;
                            }
                        }
                    }
                }
            }

            /* Texture inputs authored directly on the Shader prim. Two
             * naming families coexist:
             *   - OmniPBR:    inputs:diffuse_texture, _normalmap_texture,
             *                 _reflectionroughness_texture, ...
             *   - Isaac / UE: inputs:AlbedoTexture, inputs:BaseColor_Texture,
             *                 inputs:MainNormalInput, inputs:MergeMapInput
             *                 (ORM-packed). Adding the second set is what
             *                 unblocks the Isaac Simple_Warehouse — its
             *                 2083 materials don't expose any of the
             *                 OmniPBR names. Cf.
             *                 nanousd-vulkan-renderer/src/material.cpp.  */
            struct { const char* input; int slot; } tex_inputs[] = {
                /* OmniPBR */
                {"inputs:diffuse_texture",            TEX_DIFFUSE_COLOR},
                {"inputs:normalmap_texture",          TEX_NORMAL},
                {"inputs:reflectionroughness_texture",TEX_ROUGHNESS},
                {"inputs:metallic_texture",           TEX_METALLIC},
                {"inputs:ORM_texture",                TEX_ROUGHNESS},
                {"inputs:orm_texture",                TEX_ROUGHNESS},
                {"inputs:emissive_color_texture",     TEX_EMISSIVE_COLOR},
                {"inputs:emissive_mask_texture",      TEX_EMISSIVE_COLOR},
                {"inputs:ao_texture",                 TEX_OCCLUSION},
                {"inputs:opacity_texture",            TEX_OPACITY},
                /* OmniSurface / OmniPBRBase MDL */
                {"inputs:diffuse_reflection_color_image", TEX_DIFFUSE_COLOR},
                {"inputs:geometry_normal_image",      TEX_NORMAL},
                {"inputs:geometry_opacity_image",     TEX_OPACITY},
                {"inputs:specular_reflection_roughness_image", TEX_ROUGHNESS},
                {"inputs:metalness_image",            TEX_METALLIC},
                {"inputs:emission_color_image",       TEX_EMISSIVE_COLOR},
                /* Isaac / Unreal MDL */
                {"inputs:AlbedoTexture",              TEX_DIFFUSE_COLOR},
                {"inputs:BaseColor_Texture",          TEX_DIFFUSE_COLOR},
                {"inputs:BaseColor_Box",              TEX_DIFFUSE_COLOR},
                {"inputs:BaseColor_Plastic",          TEX_DIFFUSE_COLOR},
                {"inputs:TextureSelection",           TEX_DIFFUSE_COLOR},
                {"inputs:Text",                       TEX_DIFFUSE_COLOR},
                {"inputs:MainNormalInput",            TEX_NORMAL},
                {"inputs:NormalMap_Box",              TEX_NORMAL},
                {"inputs:MergeMapInput",              TEX_ROUGHNESS}, /* ORM-packed */
                {"inputs:MultiMap_Box",               TEX_ROUGHNESS}, /* ORM-packed */
                {"inputs:MultiMap_Plastic",           TEX_ROUGHNESS}, /* ORM-packed */
                {"inputs:AlphaSelection",             TEX_OPACITY},
            };
            const int ntex_inputs = (int)(sizeof(tex_inputs)/sizeof(tex_inputs[0]));
            int isaac_mdl_texture_inputs = 0;
            for (int ti = 0; ti < ntex_inputs; ti++) {
                int slot = tex_inputs[ti].slot;
                int explicit_mdl_input = is_mdl ? 1 : 0;
                if (!explicit_mdl_input && p->tex_indices[slot] >= 0) continue;
                ok = 0;
                const char* file = nanousd_attribasset(child, tex_inputs[ti].input, &ok);
                if (!ok || !file || !file[0])
                    file = nanousd_attribs(child, tex_inputs[ti].input, &ok);
                if (ok && file && file[0]) {
                    if (is_isaac_mdl_texture_input(tex_inputs[ti].input))
                        isaac_mdl_texture_inputs = 1;
                    int tex_idx = find_or_add_texture(mc, scene_dir, file, stage);
                    if (tex_idx >= 0) {
                        if (explicit_mdl_input || p->tex_indices[slot] < 0)
                            p->tex_indices[slot] = tex_idx;
                        if (slot == TEX_ROUGHNESS &&
                            (strstr(tex_inputs[ti].input, "ORM") ||
                             strstr(tex_inputs[ti].input, "orm") ||
                             strstr(tex_inputs[ti].input, "MergeMap") ||
                             strstr(file, "_ORM") || strstr(file, "_orm"))) {
                            if (explicit_mdl_input || p->tex_indices[TEX_OCCLUSION] < 0)
                                p->tex_indices[TEX_OCCLUSION] = tex_idx;
                            if (explicit_mdl_input || p->tex_indices[TEX_METALLIC] < 0)
                                p->tex_indices[TEX_METALLIC] = tex_idx;
                        }
                    }
                }
            }
            if (is_mdl && isaac_mdl_texture_inputs)
                p->v_flip = 0;

            /* Generic fallback: scan every asset-typed attribute on the
             * Shader prim. If the value looks like an image path
             * (.png/.jpg/.jpeg/.exr/.tga/.bmp/.hdr), classify by the
             * basename's underscore-suffix (`*_D` → diffuse, `*_N` →
             * normal, `*_ORM` → packed) or by attribute-name keywords.
             * Catches Isaac MDL materials that author texture paths under
             * arbitrary `inputs:Foo` names not in the table above. */
            int natt = nanousd_nattribs(child);
            for (int a = 0; a < natt; a++) {
                const char* aname = nanousd_attribname(child, a);
                if (!aname || strncmp(aname, "inputs:", 7) != 0) continue;
                int aok = 0;
                const char* aval = nanousd_attribasset(child, aname, &aok);
                if (!aok || !aval || !aval[0]) continue;

                /* Cheap extension test on the lowercased tail. */
                size_t vlen = strlen(aval);
                if (vlen < 4 || vlen >= 1024) continue;
                char tail[8]; int t = 0;
                for (int k = (int)vlen - 7; k < (int)vlen && t < 7; k++) {
                    if (k < 0) continue;
                    char c = aval[k];
                    tail[t++] = (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c;
                }
                tail[t] = '\0';
                if (!(strstr(tail, ".png")  || strstr(tail, ".jpg") ||
                      strstr(tail, ".jpeg") || strstr(tail, ".exr") ||
                      strstr(tail, ".tga")  || strstr(tail, ".bmp") ||
                      strstr(tail, ".hdr")))
                    continue;
                if (strstr(tail, ".mdl")) continue;

                /* Lowercase basename + attribute-name for keyword matching. */
                const char* slash = strrchr(aval, '/');
                const char* base = slash ? slash + 1 : aval;
                char lbase[256]; int li = 0;
                for (int k = 0; base[k] && li < 255; k++) {
                    char c = base[k];
                    lbase[li++] = (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c;
                }
                lbase[li] = '\0';
                char laname[256]; li = 0;
                for (int k = 0; aname[k] && li < 255; k++) {
                    char c = aname[k];
                    laname[li++] = (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c;
                }
                laname[li] = '\0';

                int slot = -1;
                if (strstr(lbase, "_orm") || strstr(lbase, "orm.") ||
                    strstr(laname, "mergemap")) {
                    slot = TEX_ROUGHNESS;  /* ORM detection below shares to AO/Metal */
                } else if (strstr(lbase, "albedo") || strstr(lbase, "diffuse") ||
                           strstr(lbase, "basecolor") || strstr(lbase, "base_color") ||
                           strstr(lbase, "_d.") || strstr(lbase, "_c.") ||
                           strstr(laname, "albedo") || strstr(laname, "basecolor") ||
                           strstr(laname, "diffuse")) {
                    slot = TEX_DIFFUSE_COLOR;
                } else if (strstr(lbase, "normal") || strstr(lbase, "_n.") ||
                           strstr(laname, "normal")) {
                    slot = TEX_NORMAL;
                } else if (strstr(lbase, "rough") || strstr(lbase, "_r.") ||
                           strstr(laname, "rough")) {
                    slot = TEX_ROUGHNESS;
                } else if (strstr(lbase, "metal") || strstr(lbase, "_m.") ||
                           strstr(laname, "metal")) {
                    slot = TEX_METALLIC;
                } else if (strstr(lbase, "emissive") || strstr(laname, "emissive") ||
                           strstr(laname, "emit")) {
                    slot = TEX_EMISSIVE_COLOR;
                } else if (strstr(lbase, "_ao") || strstr(lbase, "occlusion") ||
                           strstr(laname, "occlusion") || strstr(laname, "ambient")) {
                    slot = TEX_OCCLUSION;
                } else if (strstr(lbase, "opacity") || strstr(laname, "opacity")) {
                    slot = TEX_OPACITY;
                }
                if (slot < 0 || p->tex_indices[slot] >= 0) continue;

                int tex_idx = find_or_add_texture(mc, scene_dir, aval, stage);
                if (tex_idx < 0) continue;
                p->tex_indices[slot] = tex_idx;
                if (slot == TEX_ROUGHNESS &&
                    (strstr(lbase, "_orm") || strstr(lbase, "orm.") ||
                     strstr(laname, "mergemap"))) {
                    if (p->tex_indices[TEX_OCCLUSION] < 0)
                        p->tex_indices[TEX_OCCLUSION] = tex_idx;
                    if (p->tex_indices[TEX_METALLIC] < 0)
                        p->tex_indices[TEX_METALLIC] = tex_idx;
                }
            }

            /* OmniSurface/OmniPBR MDL graphs often use child file_texture
             * nodes with `inputs:texture` instead of UsdUVTexture
             * `inputs:file`. Match Metal/Vulkan's leaf scan so OpenGL can
             * pick up obvious albedo/normal/ORM leaves while we still avoid
             * evaluating the full MDL graph. */
            {
                NanousdPrim mat_parent = nanousd_parent(child);
                if (mat_parent) {
                    int nchildren = nanousd_nchildren(mat_parent);
                    for (int gc = 0; gc < nchildren; gc++) {
                        NanousdPrim graph_child = nanousd_child(mat_parent, gc);
                        if (!graph_child) continue;

                        int gok = 0;
                        const char* file = nanousd_attribasset(
                            graph_child, "inputs:texture", &gok);
                        if (!gok || !file || !file[0])
                            file = nanousd_attribasset(
                                graph_child, "inputs:file", &gok);
                        if (!gok || !file || !file[0] ||
                            !mdl_has_image_extension(file)) {
                            nanousd_freeprim(graph_child);
                            continue;
                        }

                        char hint[1536];
                        const char* child_path = nanousd_path(graph_child);
                        snprintf(hint, sizeof(hint), "%s %s",
                                 child_path ? child_path : "", file);

                        int is_orm = 0;
                        int slot = mdl_classify_texture_slot(hint, file, &is_orm);
                        if (!is_orm && slot < 0) {
                            nanousd_freeprim(graph_child);
                            continue;
                        }

                        int tex_idx = find_or_add_texture(mc, scene_dir, file, stage);
                        if (tex_idx >= 0) {
                            if (is_orm) {
                                if (p->tex_indices[TEX_OCCLUSION] < 0)
                                    p->tex_indices[TEX_OCCLUSION] = tex_idx;
                                if (p->tex_indices[TEX_ROUGHNESS] < 0)
                                    p->tex_indices[TEX_ROUGHNESS] = tex_idx;
                                if (p->tex_indices[TEX_METALLIC] < 0)
                                    p->tex_indices[TEX_METALLIC] = tex_idx;
                            } else if (p->tex_indices[slot] < 0) {
                                p->tex_indices[slot] = tex_idx;
                            }
                        }
                        nanousd_freeprim(graph_child);
                    }
                    nanousd_freeprim(mat_parent);
                }
            }

            /* --- ORM packed texture detection ---
             * NVIDIA assets commonly use a single _ORM texture: R=AO, G=Roughness, B=Metallic.
             * If any loaded texture path contains "ORM" or "_orm", share it across all three
             * channel slots so the shader can decode the packed channels. */
            {
                int orm_idx = -1;
                /* Check roughness slot first (most common binding for ORM) */
                if (p->tex_indices[TEX_ROUGHNESS] >= 0) {
                    const char* tp = mc->textures[p->tex_indices[TEX_ROUGHNESS]].path;
                    if (strstr(tp, "ORM") || strstr(tp, "orm") || strstr(tp, "_Orm"))
                        orm_idx = p->tex_indices[TEX_ROUGHNESS];
                }
                /* Also check all slots */
                if (orm_idx < 0) {
                    for (int s = 0; s < MAX_MATERIAL_TEXTURES; s++) {
                        if (p->tex_indices[s] < 0) continue;
                        const char* tp = mc->textures[p->tex_indices[s]].path;
                        if (strstr(tp, "_ORM") || strstr(tp, "_orm")) {
                            orm_idx = p->tex_indices[s];
                            break;
                        }
                    }
                }
                if (orm_idx >= 0) {
                    if (p->tex_indices[TEX_OCCLUSION] < 0)
                        p->tex_indices[TEX_OCCLUSION] = orm_idx;
                    if (p->tex_indices[TEX_ROUGHNESS] < 0)
                        p->tex_indices[TEX_ROUGHNESS] = orm_idx;
                    if (p->tex_indices[TEX_METALLIC] < 0)
                        p->tex_indices[TEX_METALLIC] = orm_idx;
                    fprintf(stderr, "material: ORM packed texture detected (idx=%d)\n", orm_idx);
                }
            }

            if (is_mdl)
                mdl_apply_isaac_mask_bakes(mc, stage, child, p, scene_dir);

            /* --- UDIM scale propagation ---
             * If any texture is a UDIM atlas, set the UV scale divisors. */
            for (int s = 0; s < MAX_MATERIAL_TEXTURES; s++) {
                int ti = p->tex_indices[s];
                if (ti < 0 || ti >= mc->ntextures) continue;
                if (mc->textures[ti].udim_cols > 0 &&
                    mc->textures[ti].udim_cols > (int)p->udim_scale_u)
                    p->udim_scale_u = (float)mc->textures[ti].udim_cols;
                if (mc->textures[ti].udim_rows > 0 &&
                    mc->textures[ti].udim_rows > (int)p->udim_scale_v)
                    p->udim_scale_v = (float)mc->textures[ti].udim_rows;
            }

            if (p->tex_indices[TEX_DIFFUSE_COLOR] >= 0) {
                float max_c = p->base_color[0];
                if (p->base_color[1] > max_c) max_c = p->base_color[1];
                if (p->base_color[2] > max_c) max_c = p->base_color[2];
                if ((is_mdl && !base_set && !tint_set &&
                     material_base_color_is_default_gray(p)) ||
                    max_c < 0.5f) {
                    p->base_color[0] = 1.0f;
                    p->base_color[1] = 1.0f;
                    p->base_color[2] = 1.0f;
                }
            }

            if (is_mdl) {
                int eb_ok = 0;
                int eb = nanousd_attribb(child, "inputs:enable_emission", &eb_ok);
                if (eb_ok && !eb) {
                    p->emissive_color[0] = 0.0f;
                    p->emissive_color[1] = 0.0f;
                    p->emissive_color[2] = 0.0f;
                    p->emissive_color[3] = 0.0f;
                }
            }

            char mdl_path_lower[1024];
            mdl_lower_copy(mdl_path_lower, sizeof(mdl_path_lower), mdl_path_buf);
            if (is_mdl && strstr(mdl_path_lower, "metal_black_paint")) {
                int has_any_texture = 0;
                for (int s = 0; s < MAX_MATERIAL_TEXTURES; s++) {
                    if (p->tex_indices[s] >= 0) {
                        has_any_texture = 1;
                        break;
                    }
                }
                if (!has_any_texture) {
                    /* DSX GB300 references NVIDIA's remote
                     * Metal_Black_Paint.mdl without packaged MDL source or
                     * USD-side parameters. Prefer OVRTX's remote texture set;
                     * if the network/cache path is unavailable, keep the
                     * scalar painted-metal preset instead of leaving the gray
                     * fallback material.
                     * NOTE: this hardcoded preset masks an MDL asset-resolution
                     * failure, so warn once rather than silently substituting. */
                    static int s_warned_mbp = 0;
                    if (!s_warned_mbp) {
                        s_warned_mbp = 1;
                        fprintf(stderr,
                                "[nusd] WARNING: Metal_Black_Paint MDL '%s' resolved "
                                "no textures; applying an empirical scalar painted-"
                                "metal fallback (real asset/remote textures "
                                "unavailable). Known hardcoded preset, not the "
                                "authored material.\n", mdl_path_buf);
                    }
                    p->base_color[0] = 0.012f;
                    p->base_color[1] = 0.014f;
                    p->base_color[2] = 0.016f;
                    p->metallic = 0.0f;
                    p->roughness = 0.42f;
                    p->clearcoat = 0.45f;
                    p->clearcoat_roughness = 0.22f;
                    bind_metal_black_paint_remote_textures(mc, stage, p,
                                                           scene_dir);
                }
            }
        }

        nanousd_freeprim(child);
    }
}

/* ---- Public API ---- */

/* ---- Profiling helper ---- */
#include <time.h>
static double mat_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int mtlx_parent_dir(const char* path, char* out, size_t out_size)
{
    if (!path || !path[0] || !out || out_size == 0) return 0;
    const char* slash = strrchr(path, '/');
    if (!slash || slash == path) return 0;
    size_t n = (size_t)(slash - path);
    if (n >= out_size) n = out_size - 1;
    memcpy(out, path, n);
    out[n] = '\0';
    return out[0] != '\0';
}

static int mtlx_scan_root(MaterialCollection* mc, const char* root,
                          void* stage)
{
    if (!root || !root[0]) return 0;
    return materials_scan_mtlx_directory(mc, root, stage);
}

typedef struct {
    char roots[256][PATH_MAX];
    int nroots;
} MtlxRootSet;

static int mtlx_scan_root_once(MaterialCollection* mc, MtlxRootSet* seen,
                               const char* root, void* stage)
{
    if (!root || !root[0]) return 0;

    char key[PATH_MAX];
    char resolved[PATH_MAX];
    if (realpath(root, resolved)) {
        snprintf(key, sizeof(key), "%s", resolved);
    } else {
        snprintf(key, sizeof(key), "%s", root);
    }

    if (seen) {
        for (int i = 0; i < seen->nroots; ++i) {
            if (strcmp(seen->roots[i], key) == 0) return 0;
        }
        if (seen->nroots < (int)(sizeof(seen->roots) / sizeof(seen->roots[0]))) {
            snprintf(seen->roots[seen->nroots++],
                     sizeof(seen->roots[0]), "%s", key);
        }
    }

    return mtlx_scan_root(mc, key, stage);
}

static int has_mtlx_suffix(const char* name)
{
    if (!name) return 0;
    size_t n = strlen(name);
    return n > 5 && strcasecmp(name + n - 5, ".mtlx") == 0;
}

static int dir_has_direct_mtlx(const char* root)
{
    if (!root || !root[0]) return 0;

    DIR* dir = opendir(root);
    if (!dir) return 0;

    int found = 0;
    struct dirent* ent = NULL;
    while ((ent = readdir(dir)) != NULL) {
        const char* name = ent->d_name;
        if (!name || name[0] == '.') continue;
        if (!has_mtlx_suffix(name)) continue;

        char path[PATH_MAX];
        int n = snprintf(path, sizeof(path), "%s/%s", root, name);
        if (n < 0 || (size_t)n >= sizeof(path)) continue;

        struct stat st;
        if (stat(path, &st) == 0 && S_ISREG(st.st_mode)) {
            found = 1;
            break;
        }
    }
    closedir(dir);
    return found;
}

static int mtlx_scan_implicit_root_once(MaterialCollection* mc,
                                        MtlxRootSet* seen,
                                        const char* root,
                                        void* stage)
{
    const char* recursive = getenv("NUSD_MTLX_SCAN_RECURSIVE");
    if (recursive && recursive[0] && recursive[0] != '0') {
        return mtlx_scan_root_once(mc, seen, root, stage);
    }

    /* Avoid loading unrelated MaterialX test assets when a plain USD lives at
     * a broad repository root. Implicit scans are intended for wrapper scenes
     * with sibling .mtlx files; use NUSD_MTLX_DIRS for wider searches. */
    if (!dir_has_direct_mtlx(root)) return 0;
    return mtlx_scan_root_once(mc, seen, root, stage);
}

static int mtlx_scan_env_roots(MaterialCollection* mc, MtlxRootSet* seen,
                               void* stage)
{
    const char* extra = getenv("NUSD_MTLX_DIRS");
    if (!extra || !extra[0]) return 0;

    int added = 0;
    const char* p = extra;
    while (*p) {
        const char* sep = strchr(p, ':');
        size_t n = sep ? (size_t)(sep - p) : strlen(p);
        if (n > 0) {
            char root[1024];
            if (n >= sizeof(root)) n = sizeof(root) - 1;
            memcpy(root, p, n);
            root[n] = '\0';
            added += mtlx_scan_root_once(mc, seen, root, stage);
        }
        if (!sep) break;
        p = sep + 1;
    }
    return added;
}

static int mtlx_scan_stage_layer_roots(MaterialCollection* mc, void* stage,
                                       MtlxRootSet* seen)
{
    if (!stage) return 0;

    int added = 0;
    int n_layers = nanousd_stage_n_layers(stage);
    for (int i = 0; i < n_layers; ++i) {
        const char* layer = nanousd_stage_layer_path(stage, i);
        char root[1024];
        if (mtlx_parent_dir(layer, root, sizeof(root))) {
            added += mtlx_scan_implicit_root_once(mc, seen, root, stage);
        }
    }
    return added;
}

static int mtlx_scan_all_roots(MaterialCollection* mc, void* stage,
                               const char* scene_dir)
{
    MtlxRootSet seen;
    memset(&seen, 0, sizeof(seen));

    int added = 0;
    added += mtlx_scan_implicit_root_once(mc, &seen, scene_dir, stage);
    added += mtlx_scan_stage_layer_roots(mc, stage, &seen);
    added += mtlx_scan_env_roots(mc, &seen, stage);
    return added;
}

static int gl_predecode_enabled(void) {
    const char* e = getenv("NUSD_PARALLEL_DECODE");
    return !(e && (e[0]=='0'||e[0]=='f'||e[0]=='F'||e[0]=='n'||e[0]=='N'));
}
/* DFS a material prim's subtree, collecting unique resolved image-asset paths. */
static void gl_collect_image_paths(void* stage, NanousdPrim prim, const char* scene_dir,
                                   char (*paths)[1024], int* count, int cap) {
    int na = nanousd_nattribs(prim);
    for (int a = 0; a < na && *count < cap; a++) {
        const char* an = nanousd_attribname(prim, a);
        if (!an) continue;
        int ok = 0;
        const char* av = nanousd_attribasset(prim, an, &ok);
        if (ok && av && av[0] && !strstr(av, "<UDIM>") &&
            !is_package_identifier_path(av) && mdl_has_image_extension(av)) {
            char resolved[1024];
            resolved[0] = '\0';
            resolve_texture_path_stage(scene_dir, av, stage, resolved, sizeof(resolved));
            if (resolved[0]) {
                int dup = 0;
                for (int k = 0; k < *count; k++)
                    if (strcmp(paths[k], resolved) == 0) { dup = 1; break; }
                if (!dup) { snprintf(paths[*count], 1024, "%s", resolved); (*count)++; }
            }
        }
    }
    int nc = nanousd_nchildren(prim);
    for (int c = 0; c < nc && *count < cap; c++) {
        NanousdPrim ch = nanousd_child(prim, c);
        if (ch) { gl_collect_image_paths(stage, ch, scene_dir, paths, count, cap);
                  nanousd_freeprim(ch); }
    }
}
/* Body for the parallel decode pre-pass: decode one already-resolved texture
 * path into g_gl_predecoded[i]. load_texture is the same call the OpenMP
 * version invoked in parallel, so it is safe under nu_parallel_for's workers. */
static void gl_decode_one(int i, void* ctx) {
    void* stage = ctx;
    int w = 0, h = 0;
    g_gl_predecoded[i].pixels = load_texture(g_gl_predecoded[i].path, &w, &h, stage);
    g_gl_predecoded[i].w = w;
    g_gl_predecoded[i].h = h;
}

/* Pre-pass: resolve all material image-asset paths serially, then decode the
 * unique set in parallel via nu_parallel_for (std::thread — portable; the prior
 * OpenMP pragma was a no-op on AppleClang). find_or_add_texture consumes them. */
static void gl_predecode_run(void* stage, const char* scene_dir) {
    if (!gl_predecode_enabled()) return;
    double pd0 = mat_now();
    int cap = 4096;
    char (*paths)[1024] = (char(*)[1024])malloc((size_t)cap * 1024);
    if (!paths) return;
    int n = 0;
    int nprims = nanousd_nprims(stage);
    for (int i = 0; i < nprims && n < cap; i++) {
        NanousdPrim prim = nanousd_prim(stage, i);
        if (!prim) continue;
        if (nanousd_isactive(prim) && nanousd_isa(prim, "Material"))
            gl_collect_image_paths(stage, prim, scene_dir, paths, &n, cap);
        nanousd_freeprim(prim);
    }
    if (n > 0) {
        g_gl_predecoded = (GlPredecoded*)calloc((size_t)n, sizeof(GlPredecoded));
        if (g_gl_predecoded) {
            g_gl_predecoded_n = n;
            for (int i = 0; i < n; i++)
                snprintf(g_gl_predecoded[i].path, sizeof(g_gl_predecoded[i].path),
                         "%s", paths[i]);
            nu_parallel_for(n, gl_decode_one, stage);
        }
    }
    free(paths);
    g_gl_predecode_ms = (mat_now() - pd0) * 1000.0;
    int okc = 0;
    for (int i = 0; i < g_gl_predecoded_n; i++) if (g_gl_predecoded[i].pixels) okc++;
    fprintf(stderr, "material: PARALLEL-DECODE %d/%d textures, %.1f ms wall\n",
            okc, g_gl_predecoded_n, g_gl_predecode_ms);
}
static void gl_predecode_free(void) {
    if (!g_gl_predecoded) return;
    for (int i = 0; i < g_gl_predecoded_n; i++)
        if (g_gl_predecoded[i].pixels && !g_gl_predecoded[i].consumed)
            stbi_image_free(g_gl_predecoded[i].pixels);
    free(g_gl_predecoded);
    g_gl_predecoded = NULL; g_gl_predecoded_n = 0;
}

/* Vote-based sRGB classification (mirrors vulkan). Slots known to carry color
 * authored as sRGB: TEX_DIFFUSE_COLOR, TEX_EMISSIVE_COLOR, TEX_OPACITY (alpha is
 * gamma-encoded with the color). All others are data (normal, roughness,
 * metallic, occlusion, displacement). When a single texture is referenced by
 * mixed roles across materials, "linear wins": sampling linear data as sRGB
 * corrupts normals + PBR, which is worse than the inverse.
 *
 * Runs on BOTH the inline-UsdShade and pure-.mtlx paths. MaterialX scenes (e.g.
 * the chess set, whose piece base_color images author colorspace="srgb_texture")
 * previously returned before this ran and uploaded diffuse as linear, washing
 * dark albedos to mid-grey — black pieces and dark board squares rendered pale. */
static void mc_classify_texture_srgb(MaterialCollection* mc)
{
    const int color_slot[MAX_MATERIAL_TEXTURES] = {
        /* TEX_DIFFUSE_COLOR  = 0 */ 1,
        /* TEX_NORMAL         = 1 */ 0,
        /* TEX_ROUGHNESS      = 2 */ 0,
        /* TEX_METALLIC       = 3 */ 0,
        /* TEX_EMISSIVE_COLOR = 4 */ 1,
        /* TEX_OCCLUSION      = 5 */ 0,
        /* TEX_OPACITY        = 6 */ 1,
        /* TEX_DISPLACEMENT   = 7 */ 0,
    };
    for (int t = 0; t < mc->ntextures; ++t) {
        int srgb_votes = 0, data_votes = 0;
        for (int m = 0; m < mc->nmaterials; ++m) {
            const int* tex_indices = mc->materials[m].params.tex_indices;
            for (int s = 0; s < MAX_MATERIAL_TEXTURES; ++s) {
                if (tex_indices[s] != t) continue;
                if (color_slot[s]) ++srgb_votes;
                else               ++data_votes;
            }
        }
        mc->textures[t].is_srgb = (data_votes == 0 && srgb_votes > 0) ? 1 : 0;
    }

    if (getenv("NUSD_MAT_DIAG")) {
        for (int m = 0; m < mc->nmaterials; ++m) {
            MaterialParams* p = &mc->materials[m].params;
            fprintf(stderr,
                    "[mat_diag] material[%d] path=%s base=(%.3f %.3f %.3f) "
                    "rough=%.3f metal=%.3f vflip=%d\n",
                    m, mc->materials[m].prim_path,
                    p->base_color[0], p->base_color[1], p->base_color[2],
                    p->roughness, p->metallic, p->v_flip);
            for (int s = 0; s < MAX_MATERIAL_TEXTURES; ++s) {
                int ti = p->tex_indices[s];
                if (ti < 0 || ti >= mc->ntextures) continue;
                fprintf(stderr,
                        "[mat_diag]   slot[%d] -> tex[%d] srgb=%d %s\n",
                        s, ti, mc->textures[ti].is_srgb,
                        mc->textures[ti].path);
            }
        }
    }
}

MaterialCollection* materials_load(void* stage, const char* scene_dir)
{
    if (!stage) return NULL;

    double t0 = mat_now();

    MaterialCollection* mc = (MaterialCollection*)calloc(1, sizeof(MaterialCollection));
    if (!mc) return NULL;
#ifdef NUSD_HAVE_PTEX
    ptex_average_cache_clear();
#endif

    /* Count Material prims */
    int nprims = nanousd_nprims(stage);
    int nmat = 0;
    for (int i = 0; i < nprims; i++) {
        NanousdPrim p = nanousd_prim(stage, i);
        if (!p) continue;
        if (nanousd_isa(p, "Material")) nmat++;
        nanousd_freeprim(p);
    }
    double t_count = mat_now();
    fprintf(stderr, "material: count Materials %.1f ms (%d / %d prims)\n",
            (t_count - t0) * 1000.0, nmat, nprims);

    /* Even when there are zero UsdShade Material prims, fall through:
     * the scene may bind materials via @file.mtlx@ references that
     * nanousd's parser doesn't follow. material_mtlx.cpp picks those
     * up below. */
    if (nmat == 0) {
        fprintf(stderr, "material: no inline UsdShade Material prims; "
                        "checking for scene-local .mtlx files\n");
        mtlx_scan_all_roots(mc, stage, scene_dir);
        if (mc->nmaterials > 0) {
            fprintf(stderr, "material: %d unique material(s) from .mtlx, "
                            "%d texture(s)\n",
                    mc->nmaterials, mc->ntextures);
        } else {
            fprintf(stderr, "material: no materials found\n");
        }
        mc_classify_texture_srgb(mc);
        return mc;
    }

    mc->materials = (SceneMaterial*)calloc((size_t)nmat, sizeof(SceneMaterial));
    if (!mc->materials) {
        free(mc);
        return NULL;
    }
    mc->nmaterials = nmat;

    /* Parallel texture pre-decode (before the serial extract walk). */
    gl_predecode_run(stage, scene_dir);

    /* Extract each material — deduplicated by leaf name. The warehouse has
     * 5912 Material prims but only ~1761 unique names (the rest are repeated
     * bindings/instances of the same material). Reading each unique material
     * once and copying its params for repeats cuts the per-material reader
     * work ~3.4x. Matches nanousd-vulkan-renderer's mat_name_to_idx dedup.
     * Each slot keeps its own prim_path (binding lookup resolves by path).
     * SceneMaterial is POD here (the only owned shader memory lives in
     * mc->unique_shaders), so the struct copy is safe. */
    int mat_idx = 0;
    int unique_reads = 0;
    int ptex_average_count = 0;
    for (int i = 0; i < nprims && mat_idx < nmat; i++) {
        NanousdPrim p = nanousd_prim(stage, i);
        if (!p) continue;
        if (nanousd_isa(p, "Material")) {
            const char* mname = nanousd_name(p);
            int dup = -1;
            for (int j = 0; j < mat_idx; j++) {
                if (strcmp(mc->materials[j].name, mname ? mname : "") == 0) {
                    dup = j; break;
                }
            }
            if (dup >= 0) {
                mc->materials[mat_idx] = mc->materials[dup];  /* params/tex/shader */
                const char* mp = nanousd_path(p);
                snprintf(mc->materials[mat_idx].prim_path,
                         sizeof(mc->materials[mat_idx].prim_path), "%s", mp ? mp : "");
            } else {
                extract_material(mc, stage, p, scene_dir, mat_idx);
                unique_reads++;
            }
            material_apply_ptex_average(&mc->materials[mat_idx], p,
                                        scene_dir, stage);
            if (mc->materials[mat_idx].has_ptex_average_color)
                ptex_average_count++;
            mat_idx++;
        }
        nanousd_freeprim(p);
    }
    fprintf(stderr, "material: %d unique materials read (from %d prims)\n",
            unique_reads, mat_idx);
    if (ptex_average_count > 0)
        fprintf(stderr, "material: applied Ptex average color to %d materials\n",
                ptex_average_count);
#ifdef NUSD_HAVE_PTEX
    ptex_average_cache_log();
#endif

    double t_extract = mat_now();
    fprintf(stderr, "material: extract_material total %.1f ms (%d materials, mdl cache %d files)\n",
            (t_extract - t_count) * 1000.0, mc->nmaterials, g_mdl_cache_n);
    gl_predecode_free();  /* free pre-decoded textures the walk didn't consume */

    /* Phase 2: pick up any standalone .mtlx files alongside the asset.
     * Mixed pipelines (UsdShade + MaterialX) merge here; pure-mtlx assets
     * already early-returned above. */
    int mtlx_added = 0;
    /* Scenes with inline UsdShade materials (Isaac warehouse, etc.) can
     * contain thousands of contributing layers. Avoid recursively probing
     * every layer for side-loaded MaterialX unless the caller explicitly
     * supplied extra roots. Pure MaterialX wrapper scenes still use the
     * nmat == 0 path above. */
    if (getenv("NUSD_MTLX_DIRS")) {
        MtlxRootSet seen;
        memset(&seen, 0, sizeof(seen));
        mtlx_added = mtlx_scan_env_roots(mc, &seen, stage);
    }
    if (mtlx_added > 0) {
        fprintf(stderr, "material: + %d material(s) from .mtlx files\n",
                mtlx_added);
    }

    mc_classify_texture_srgb(mc);

    fprintf(stderr, "material: loaded %d materials, %d textures\n",
            mc->nmaterials, mc->ntextures);
    mdl_cache_clear();
#ifdef NUSD_HAVE_PTEX
    ptex_average_cache_clear();
#endif
    return mc;
}

/* ---- Fast path: precomputed binding map ---- */

typedef struct {
    char path[512];
    int  mat_idx;
    int  stronger;
} BindingEntry;

static int compare_binding_path(const void* a, const void* b) {
    return strcmp(((const BindingEntry*)a)->path,
                  ((const BindingEntry*)b)->path);
}

static int material_binding_is_stronger_than_descendants(NanousdPrim prim,
                                                         const char* rel_name)
{
    int ok = 0;
    const char* strength =
        nanousd_rel_metadatas(prim, rel_name, "bindMaterialAs", &ok);
    return ok && strength &&
           strcmp(strength, "strongerThanDescendants") == 0;
}

static int material_index_for_binding_target(MaterialCollection* mc,
                                             NanousdPrim binding_prim,
                                             const char* target)
{
    if (!mc || !binding_prim || !target || !target[0]) return -1;

    /* Exact path match */
    for (int i = 0; i < mc->nmaterials; i++) {
        if (strcmp(mc->materials[i].prim_path, target) == 0)
            return i;
    }

    /* Path-remapping fallback: match by material name (last component) */
    const char* target_name = strrchr(target, '/');
    if (!target_name) return -1;
    target_name++;

    const char* mesh_path = nanousd_path(binding_prim);
    for (int i = 0; i < mc->nmaterials; i++) {
        if (strcmp(mc->materials[i].name, target_name) != 0)
            continue;
        if (mesh_path && strstr(mc->materials[i].prim_path, mesh_path) == NULL) {
            const char* mp = mc->materials[i].prim_path;
            const char* slash1 = strchr(mp + 1, '/');
            const char* slash2 = mesh_path ? strchr(mesh_path + 1, '/') : NULL;
            if (slash1 && slash2) {
                size_t len1 = (size_t)(slash1 - mp);
                size_t len2 = (size_t)(slash2 - mesh_path);
                if (len1 == len2 && memcmp(mp, mesh_path, len1) == 0)
                    return i;
            }
        } else {
            return i;
        }
    }

    /* If ancestor matching failed, just match by name */
    for (int i = 0; i < mc->nmaterials; i++) {
        if (strcmp(mc->materials[i].name, target_name) == 0)
            return i;
    }
    return -1;
}

static int material_binding_from_path_map(const BindingEntry* bindings,
                                          int nb,
                                          const char* mesh_path)
{
    if (!bindings || nb <= 0 || !mesh_path || !mesh_path[0])
        return -1;

    char* cur_path = strdup(mesh_path);
    if (!cur_path) return -1;

    int found = -1;
    while (cur_path[0]) {
        BindingEntry key;
        snprintf(key.path, sizeof(key.path), "%s", cur_path);
        BindingEntry* hit = (BindingEntry*)bsearch(
            &key, bindings, (size_t)nb, sizeof(BindingEntry),
            compare_binding_path);
        if (hit) {
            if (hit->stronger) {
                found = hit->mat_idx;
                break;
            }
            if (found < 0)
                found = hit->mat_idx;
        }

        if (strcmp(cur_path, "/") == 0)
            break;
        char* slash = strrchr(cur_path, '/');
        if (!slash)
            break;
        if (slash == cur_path)
            cur_path[1] = '\0';
        else
            *slash = '\0';
    }
    free(cur_path);
    return found;
}

int materials_assign_bindings(MaterialCollection* mc, void* stage,
                               void* meshes_void, int nmeshes)
{
    if (!mc || !stage || !meshes_void || nmeshes <= 0) return 0;
    SceneMesh* meshes = (SceneMesh*)meshes_void;
    if (mc->nmaterials == 0) {
        for (int k = 0; k < nmeshes; k++) meshes[k].material_index = -1;
        return 0;
    }

    double t0 = mat_now();
    int nprims = nanousd_nprims(stage);

    /* Build sorted material-path → idx index. .mtlx-derived materials
     * have an empty prim_path; instead they're indexed by name so the
     * binding resolver can fall back to matching the LAST path component
     * of the relationship target (e.g. </Foo/Materials/M_King_B> →
     * "M_King_B"). */
    BindingEntry* mat_idx_arr = (BindingEntry*)malloc(
        sizeof(BindingEntry) * (size_t)mc->nmaterials);
    BindingEntry* mat_name_arr = (BindingEntry*)malloc(
        sizeof(BindingEntry) * (size_t)mc->nmaterials);
    if (!mat_idx_arr || !mat_name_arr) {
        free(mat_idx_arr); free(mat_name_arr);
        return 0;
    }
    int n_path = 0, n_name = 0;
    for (int i = 0; i < mc->nmaterials; i++) {
        if (mc->materials[i].prim_path[0]) {
            snprintf(mat_idx_arr[n_path].path,
                     sizeof(mat_idx_arr[n_path].path), "%s",
                     mc->materials[i].prim_path);
            mat_idx_arr[n_path].mat_idx = i;
            mat_idx_arr[n_path].stronger = 0;
            n_path++;
        }
        if (mc->materials[i].name[0]) {
            snprintf(mat_name_arr[n_name].path,
                     sizeof(mat_name_arr[n_name].path), "%s",
                     mc->materials[i].name);
            mat_name_arr[n_name].mat_idx = i;
            mat_name_arr[n_name].stronger = 0;
            n_name++;
        }
    }
    qsort(mat_idx_arr,  (size_t)n_path, sizeof(BindingEntry),
          compare_binding_path);
    qsort(mat_name_arr, (size_t)n_name, sizeof(BindingEntry),
          compare_binding_path);

    /* Pass 1: walk all prims and collect implicit + explicit bindings.
     * Two patterns:
     *   (a) explicit `material:binding` rel on a prim (or alt name
     *       `rel:material:binding` that some authoring tools emit) →
     *       prim binds to target Material.
     *   (b) NVIDIA SimReady / Isaac convention: a prim has a child
     *       Scope (commonly named "Looks") that contains Material
     *       children. The prim and its descendants implicitly bind to
     *       the first Material in that Scope. The warehouse uses (b)
     *       almost exclusively. */
    BindingEntry* bindings = (BindingEntry*)malloc(
        sizeof(BindingEntry) * (size_t)(nprims * 2));
    if (!bindings) { free(mat_idx_arr); free(mat_name_arr); return 0; }
    int nb = 0;
    /* NUSD_BIND_PROFILE=1 — granular timing of Pass 1 sub-operations to
     * find where the DSX binding precompute spends its ~1150 s. */
    int bind_profile = getenv("NUSD_BIND_PROFILE") != NULL;
    double t_prim = 0, t_nrel = 0, t_scope = 0;
    long n_nrel_calls = 0, n_scope = 0;
    for (int i = 0; i < nprims; i++) {
        double _tp0 = bind_profile ? mat_now() : 0;
        NanousdPrim p = nanousd_prim(stage, i);
        if (bind_profile) t_prim += mat_now() - _tp0;
        if (!p) continue;

        /* (a) Explicit rel target. */
        const char* rel_name = "material:binding";
        double _tn0 = bind_profile ? mat_now() : 0;
        int n = nanousd_nreltargets(p, rel_name);
        n_nrel_calls++;
        if (n <= 0) {
            rel_name = "rel:material:binding";
            n = nanousd_nreltargets(p, rel_name);
            n_nrel_calls++;
        }
        if (bind_profile) t_nrel += mat_now() - _tn0;
        if (n > 0) {
            const char* target = nanousd_reltarget(p, rel_name, 0);
            if (target && target[0]) {
                BindingEntry key;
                snprintf(key.path, sizeof(key.path), "%s", target);
                BindingEntry* hit = (BindingEntry*)bsearch(
                    &key, mat_idx_arr, (size_t)n_path,
                    sizeof(BindingEntry), compare_binding_path);
                /* Fall back to name-match against .mtlx-derived materials
                 * whose prim_path is empty (the chess set's pattern: the
                 * binding target prim doesn't exist in the loaded stage,
                 * but its leaf name = the .mtlx surfacematerial name). */
                if (!hit) {
                    const char* slash = strrchr(target, '/');
                    const char* leaf = slash ? slash + 1 : target;
                    if (leaf && leaf[0]) {
                        BindingEntry nkey;
                        snprintf(nkey.path, sizeof(nkey.path), "%s", leaf);
                        hit = (BindingEntry*)bsearch(
                            &nkey, mat_name_arr, (size_t)n_name,
                            sizeof(BindingEntry), compare_binding_path);
                    }
                }
                if (hit) {
                    const char* prim_path = nanousd_path(p);
                    if (prim_path && prim_path[0]) {
                        BindingEntry* b = &bindings[nb++];
                        snprintf(b->path, sizeof(b->path), "%s", prim_path);
                        b->mat_idx = hit->mat_idx;
                        b->stronger =
                            material_binding_is_stronger_than_descendants(p, rel_name);
                    }
                }
            }
        }

        /* (b) Scope-with-Material-child inherited binding: if THIS prim
         * is a Scope with a Material child, attribute the binding to
         * the prim's *parent* (so all of the parent's other-children
         * meshes inherit it through the ancestor walk in Pass 2). */
        double _ts0 = bind_profile ? mat_now() : 0;
        /* `nanousd_isa(p, "Scope")` is schema-aware is-a resolution —
         * profiled at ~3.9 ms/call on DSX, which made this loop's
         * per-prim Scope check the dominant cost (~1150 s of the
         * binding precompute). `Scope` and `Material` are leaf schema
         * types with no subclasses, so an exact typename match is
         * equivalent and ~1000x cheaper (cached token string). */
        const char* p_type = nanousd_typename(p);
        if (p_type && strcmp(p_type, "Scope") == 0) {
            n_scope++;
            int mat_for_scope = -1;
            int nch = nanousd_nchildren(p);
            for (int c = 0; c < nch && mat_for_scope < 0; c++) {
                NanousdPrim child = nanousd_child(p, c);
                if (!child) continue;
                const char* c_type = nanousd_typename(child);
                if (c_type && strcmp(c_type, "Material") == 0) {
                    const char* cp = nanousd_path(child);
                    if (cp && cp[0]) {
                        BindingEntry key;
                        snprintf(key.path, sizeof(key.path), "%s", cp);
                        BindingEntry* hit = (BindingEntry*)bsearch(
                            &key, mat_idx_arr, (size_t)n_path,
                            sizeof(BindingEntry), compare_binding_path);
                        if (hit) mat_for_scope = hit->mat_idx;
                    }
                }
                nanousd_freeprim(child);
            }
            if (mat_for_scope >= 0) {
                NanousdPrim parent = nanousd_parent(p);
                if (parent) {
                    const char* pp = nanousd_path(parent);
                    if (pp && pp[0]) {
                        BindingEntry* b = &bindings[nb++];
                        snprintf(b->path, sizeof(b->path), "%s", pp);
                        b->mat_idx = mat_for_scope;
                        b->stronger = 0;
                    }
                    nanousd_freeprim(parent);
                }
            }
        }
        if (bind_profile) t_scope += mat_now() - _ts0;
        nanousd_freeprim(p);

        /* Incremental profile dump every 20k prims so a timed-out run
         * still yields a breakdown. */
        if (bind_profile && (i % 20000) == 19999) {
            fprintf(stderr,
                    "material: [bind_profile] @%d/%d  t_prim=%.1fs t_nrel=%.1fs "
                    "t_scope=%.1fs nrel_calls=%ld scopes=%ld\n",
                    i + 1, nprims, t_prim, t_nrel, t_scope,
                    n_nrel_calls, n_scope);
        }
    }
    if (bind_profile) {
        fprintf(stderr,
                "material: [bind_profile] nprims=%d nrel_calls=%ld scopes=%ld | "
                "t_prim=%.1fs t_nrel=%.1fs t_scope=%.1fs\n",
                nprims, n_nrel_calls, n_scope, t_prim, t_nrel, t_scope);
    }
    /* Sort, dedup keeping first hit (explicit wins over implicit because
     * we wrote explicit first per-prim). */
    qsort(bindings, (size_t)nb, sizeof(BindingEntry), compare_binding_path);
    int unique = 0;
    for (int i = 0; i < nb; i++) {
        if (unique == 0 ||
            strcmp(bindings[unique - 1].path, bindings[i].path) != 0) {
            bindings[unique++] = bindings[i];
        } else if (!bindings[unique - 1].stronger && bindings[i].stronger) {
            bindings[unique - 1] = bindings[i];
        }
    }
    nb = unique;

    double t1 = mat_now();
    fprintf(stderr,
            "material: bindings precompute %.1f ms (%d binding rels in stage)\n",
            (t1 - t0) * 1000.0, nb);

    /* Pass 2: per mesh, walk ancestor paths and bsearch the binding map.
     * OpenGL stores one material per SceneMesh just like Metal/Vulkan. USD
     * authored GeomSubsets therefore become additional SceneMesh records
     * whose paths are the subset prims. Resolving from mesh->path, rather
     * than assuming one scene mesh per UsdGeomMesh prim, keeps OpenGL aligned
     * with the split-mesh path and with USD material-binding strength rules. */
    int hit_count = 0;
    for (int mesh_idx = 0; mesh_idx < nmeshes; mesh_idx++) {
        int found = material_binding_from_path_map(
            bindings, nb, meshes[mesh_idx].path);
        meshes[mesh_idx].material_index = found;
        if (found >= 0) hit_count++;
    }

    free(bindings);
    free(mat_idx_arr);
    free(mat_name_arr);

    double t2 = mat_now();
    fprintf(stderr,
            "material: assign bindings to meshes %.1f ms (%d/%d resolved)\n",
            (t2 - t1) * 1000.0, hit_count, nmeshes);
    return hit_count;
}

int materials_find_binding(MaterialCollection* mc, void* stage,
                           void* mesh_prim)
{
    if (!mc || !mesh_prim || mc->nmaterials == 0) return -1;

    NanousdPrim prim = (NanousdPrim)mesh_prim;

    /* 1. Walk up prim hierarchy for material:binding (USD inheritance).
     * OVRTX/Hydra honors bindMaterialAs on the relationship itself:
     * closest binding normally wins, unless an ancestor is authored as
     * strongerThanDescendants. Keep this scalar resolver identical to the
     * precomputed OpenGL path-map resolver above and to the Metal/Vulkan
     * material semantics. */
    int closest_idx = -1;
    NanousdPrim cur = prim;
    while (cur) {
        int ntargets = nanousd_nreltargets(cur, "material:binding");
        const char* rel_name = "material:binding";
        if (ntargets <= 0) {
            ntargets = nanousd_nreltargets(cur, "rel:material:binding");
            rel_name = "rel:material:binding";
        }
        if (ntargets > 0) {
            const char* target = nanousd_reltarget(cur, rel_name, 0);
            if (target && target[0]) {
                int idx = material_index_for_binding_target(mc, cur, target);
                if (idx >= 0) {
                    if (material_binding_is_stronger_than_descendants(cur, rel_name)) {
                        if (cur != prim) nanousd_freeprim(cur);
                        return idx;
                    }
                    if (closest_idx < 0)
                        closest_idx = idx;
                }
            }
        }
        NanousdPrim next = nanousd_parent(cur);
        if (cur != prim) nanousd_freeprim(cur);
        cur = next;
    }
    if (closest_idx >= 0)
        return closest_idx;

    /* 2. Walk up to find variant selection for Black/White material preference */
    char variant_sel_buf[64] = {0};
    const char* variant_sel = NULL;
    cur = prim;
    while (cur) {
        const char* sel = nanousd_variantselection(cur, "shadingVariant");
        if (sel && sel[0]) {
            snprintf(variant_sel_buf, sizeof(variant_sel_buf), "%s", sel);
            variant_sel = variant_sel_buf;
            if (cur != prim) nanousd_freeprim(cur);
            break;
        }
        NanousdPrim next = nanousd_parent(cur);
        if (cur != prim) nanousd_freeprim(cur);
        cur = next;
    }

    /* 3. Fallback: find nearest ancestor Material sibling scope.
     * Use variant selection to pick the correct Black/White material. */
    cur = prim;
    while (cur) {
        NanousdPrim parent = nanousd_parent(cur);
        if (!parent) {
            if (cur != prim) nanousd_freeprim(cur);
            break;
        }
        int nch = nanousd_nchildren(parent);
        int found_idx = -1;
        for (int c = 0; c < nch && found_idx < 0; c++) {
            NanousdPrim sibling = nanousd_child(parent, c);
            if (!sibling) continue;
            const char* tn = nanousd_typename(sibling);
            if (!tn || strcmp(tn, "Scope") != 0) {
                nanousd_freeprim(sibling);
                continue;
            }
            int nmat = nanousd_nchildren(sibling);
            int first_match = -1;
            for (int m = 0; m < nmat; m++) {
                NanousdPrim mat_prim = nanousd_child(sibling, m);
                if (!mat_prim) continue;
                const char* mt = nanousd_typename(mat_prim);
                if (!mt || strcmp(mt, "Material") != 0) {
                    nanousd_freeprim(mat_prim);
                    continue;
                }
                const char* mat_path = nanousd_path(mat_prim);
                if (!mat_path) {
                    nanousd_freeprim(mat_prim);
                    continue;
                }

                int mat_idx = -1;
                for (int i = 0; i < mc->nmaterials; i++) {
                    if (strcmp(mc->materials[i].prim_path, mat_path) == 0) {
                        mat_idx = i;
                        break;
                    }
                }
                if (mat_idx < 0) {
                    nanousd_freeprim(mat_prim);
                    continue;
                }

                if (first_match < 0) first_match = mat_idx;

                if (variant_sel) {
                    const char* mn = nanousd_name(mat_prim);
                    if (mn) {
                        const char* suffix = (strcmp(variant_sel, "Black") == 0) ? "_B" :
                                             (strcmp(variant_sel, "White") == 0) ? "_W" : NULL;
                        if (suffix) {
                            size_t mn_len = strlen(mn);
                            size_t sf_len = strlen(suffix);
                            if (mn_len >= sf_len &&
                                strcmp(mn + mn_len - sf_len, suffix) == 0) {
                                found_idx = mat_idx;
                                nanousd_freeprim(mat_prim);
                                break;
                            }
                        }
                    }
                }
                nanousd_freeprim(mat_prim);
            }
            nanousd_freeprim(sibling);
            if (found_idx < 0 && first_match >= 0) found_idx = first_match;
        }
        if (found_idx >= 0) {
            if (cur != prim) nanousd_freeprim(cur);
            nanousd_freeprim(parent);
            return found_idx;
        }
        if (cur != prim) nanousd_freeprim(cur);
        cur = parent;
    }

    return -1;
}

void materials_free(MaterialCollection* mc)
{
    if (!mc) return;

    if (mc->textures) {
        for (int i = 0; i < mc->ntextures; i++) {
            if (mc->textures[i].pixels)
                free(mc->textures[i].pixels);
        }
        free(mc->textures);
    }

    if (mc->unique_shaders) {
        for (int i = 0; i < mc->nunique_shaders; i++) {
            free(mc->unique_shaders[i].vert_spv);
            free(mc->unique_shaders[i].frag_spv);
        }
        free(mc->unique_shaders);
    }

    free(mc->materials);
    free(mc);
}

/* No-ops: OpenGL renderer uses built-in PBR shaders, no MaterialX needed */
/* materialx_init / materialx_shutdown live in material_mtlx.cpp now —
 * they wrap the real MaterialX C++ library so .mtlx-bound scenes (chess
 * set, etc.) load with proper textures. */

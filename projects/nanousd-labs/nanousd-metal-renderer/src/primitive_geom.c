// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * primitive_geom.c — Tessellation for UsdGeom shape primitives.
 *
 * Each primitive is emitted with sharp (per-face) normals where edges
 * should be hard (Cube), and smooth (per-vertex outward) normals where
 * the surface is curved (Sphere body, Capsule hemispheres + body).
 *
 * Axis convention: USD spec — Cylinder/Capsule's `axis` token picks
 * which local axis runs from -h/2 to +h/2. Default is "Z".
 */

#include "primitive_geom.h"

#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================
 * USD attribute readers — small wrappers that fall back to USD spec
 * defaults when the attribute is absent.
 * ============================================================ */

static double read_double_attr(NanousdPrim prim, const char* name, double dflt)
{
    /* nanousd C API exposes scalar reads via attribf/attribd; size is
     * authored as DOUBLE in UsdGeomCube/Sphere/Capsule/Cylinder. Try
     * double first, fall back to float. */
    int ok = 0;
    double dv = nanousd_attribd(prim, name, &ok);
    if (ok) return dv;
    float fv = nanousd_attribf(prim, name, &ok);
    if (ok) return (double)fv;
    return dflt;
}

/* axis: USD authors as token, "X" / "Y" / "Z" (default "Z").
 * Returns 0/1/2 = X/Y/Z. */
static int read_axis_attr(NanousdPrim prim)
{
    int ok = 0;
    const char* tok = nanousd_attrib_token(prim, "axis", &ok);
    if (ok && tok) {
        if (!strcmp(tok, "X")) return 0;
        if (!strcmp(tok, "Y")) return 1;
    }
    return 2; /* Z (USD default) */
}

/* Permute (a, b, c) where c is the "axis" component so it lives in
 * out[axis], and a/b cycle into the other two coordinates. Mirrors the
 * standard USD convention for cylinder/capsule axis remapping. */
static void axis_permute(float a, float b, float c, int axis, float out[3])
{
    if (axis == 0)      { out[0] = c; out[1] = a; out[2] = b; }
    else if (axis == 1) { out[0] = b; out[1] = c; out[2] = a; }
    else                { out[0] = a; out[1] = b; out[2] = c; }
}

/* ============================================================
 * Cube — 24 vertices (4 per face) with face-aligned normals so the
 * edges stay sharp under smooth shading.
 * UsdGeomCube spec: vertices at ±size/2 on each axis. Default size=2.
 * ============================================================ */

static int gen_cube(NanousdPrim prim, Arena* arena, SceneMesh* m)
{
    double size = read_double_attr(prim, "size", 2.0);
    float h = (float)(size * 0.5);

    const int NV = 24;
    const int NI = 36;
    m->nvertices = NV;
    m->nindices  = NI;
    m->positions = (float*)arena_alloc(arena, (size_t)NV * 3 * sizeof(float), 16);
    m->normals   = (float*)arena_alloc(arena, (size_t)NV * 3 * sizeof(float), 16);
    m->indices   = (uint32_t*)arena_alloc(arena, (size_t)NI * sizeof(uint32_t), 16);
    if (!m->positions || !m->normals || !m->indices) return 0;

    /* 6 faces × 4 corners. Order: +X, -X, +Y, -Y, +Z, -Z. CCW from
     * outside so the auto-computed normals match the explicit ones. */
    static const float face_n[6][3] = {
        { 1, 0, 0}, {-1, 0, 0}, {0,  1, 0}, {0, -1, 0}, {0, 0,  1}, {0, 0, -1}
    };
    /* For each face, four corner offsets relative to face center (h units).
     * Each row is the four corners as (dx, dy, dz) using the face's
     * local up/right; we just hard-code with explicit signs. */
    static const float face_corners[6][4][3] = {
        /* +X */ {{ 1,-1,-1},{ 1, 1,-1},{ 1, 1, 1},{ 1,-1, 1}},
        /* -X */ {{-1, 1,-1},{-1,-1,-1},{-1,-1, 1},{-1, 1, 1}},
        /* +Y */ {{ 1, 1,-1},{-1, 1,-1},{-1, 1, 1},{ 1, 1, 1}},
        /* -Y */ {{-1,-1,-1},{ 1,-1,-1},{ 1,-1, 1},{-1,-1, 1}},
        /* +Z */ {{-1,-1, 1},{ 1,-1, 1},{ 1, 1, 1},{-1, 1, 1}},
        /* -Z */ {{ 1,-1,-1},{-1,-1,-1},{-1, 1,-1},{ 1, 1,-1}},
    };
    int v = 0;
    int idx = 0;
    for (int f = 0; f < 6; f++) {
        uint32_t base = (uint32_t)v;
        for (int c = 0; c < 4; c++) {
            m->positions[v * 3 + 0] = face_corners[f][c][0] * h;
            m->positions[v * 3 + 1] = face_corners[f][c][1] * h;
            m->positions[v * 3 + 2] = face_corners[f][c][2] * h;
            m->normals[v * 3 + 0] = face_n[f][0];
            m->normals[v * 3 + 1] = face_n[f][1];
            m->normals[v * 3 + 2] = face_n[f][2];
            v++;
        }
        m->indices[idx++] = base;
        m->indices[idx++] = base + 1;
        m->indices[idx++] = base + 2;
        m->indices[idx++] = base;
        m->indices[idx++] = base + 2;
        m->indices[idx++] = base + 3;
    }
    return 1;
}

/* ============================================================
 * Sphere — lat/long tessellation.
 * UsdGeomSphere spec: `radius` attr (default 1.0).
 * Tessellation density: 16 longitude segments × 8 latitude bands.
 * ============================================================ */

static int gen_sphere(NanousdPrim prim, Arena* arena, SceneMesh* m)
{
    double radius = read_double_attr(prim, "radius", 1.0);
    const int NLON = 24;
    const int NLAT = 12; /* latitude bands, including poles */

    /* Vertices: 2 poles + (NLAT-1) rings × NLON */
    const int NV = 2 + (NLAT - 1) * NLON;
    /* Triangles: 2 pole caps (NLON tris each) + (NLAT-2) quad rings (2*NLON tris each) */
    const int NTRI = 2 * NLON + (NLAT - 2) * 2 * NLON;
    const int NI = NTRI * 3;

    m->nvertices = NV;
    m->nindices  = NI;
    m->positions = (float*)arena_alloc(arena, (size_t)NV * 3 * sizeof(float), 16);
    m->normals   = (float*)arena_alloc(arena, (size_t)NV * 3 * sizeof(float), 16);
    m->indices   = (uint32_t*)arena_alloc(arena, (size_t)NI * sizeof(uint32_t), 16);
    if (!m->positions || !m->normals || !m->indices) return 0;

    /* Pole verts: 0 = +Z (north), 1 = -Z (south). */
    m->positions[0 * 3 + 0] = 0; m->positions[0 * 3 + 1] = 0; m->positions[0 * 3 + 2] = (float)radius;
    m->normals  [0 * 3 + 0] = 0; m->normals  [0 * 3 + 1] = 0; m->normals  [0 * 3 + 2] = 1;
    m->positions[1 * 3 + 0] = 0; m->positions[1 * 3 + 1] = 0; m->positions[1 * 3 + 2] = -(float)radius;
    m->normals  [1 * 3 + 0] = 0; m->normals  [1 * 3 + 1] = 0; m->normals  [1 * 3 + 2] = -1;

    /* Rings: ring r (1..NLAT-1) sits at polar angle θ_r = r·π/NLAT.
     * Ring vertex (r, j) lands at index 2 + (r-1)*NLON + j. */
    for (int r = 1; r < NLAT; r++) {
        double theta = (double)r * M_PI / (double)NLAT;
        double st = sin(theta), ct = cos(theta);
        for (int j = 0; j < NLON; j++) {
            double phi = (double)j * 2.0 * M_PI / (double)NLON;
            float nx = (float)(st * cos(phi));
            float ny = (float)(st * sin(phi));
            float nz = (float)(ct);
            int v = 2 + (r - 1) * NLON + j;
            m->normals  [v * 3 + 0] = nx;
            m->normals  [v * 3 + 1] = ny;
            m->normals  [v * 3 + 2] = nz;
            m->positions[v * 3 + 0] = (float)radius * nx;
            m->positions[v * 3 + 1] = (float)radius * ny;
            m->positions[v * 3 + 2] = (float)radius * nz;
        }
    }

    /* Indices */
    int idx = 0;
    /* North cap: pole 0 fans to ring 1. */
    for (int j = 0; j < NLON; j++) {
        int j1 = (j + 1) % NLON;
        m->indices[idx++] = 0;
        m->indices[idx++] = (uint32_t)(2 + j);
        m->indices[idx++] = (uint32_t)(2 + j1);
    }
    /* Quad rings between r and r+1, r in [1, NLAT-2] */
    for (int r = 1; r < NLAT - 1; r++) {
        int row0 = 2 + (r - 1) * NLON;
        int row1 = 2 + r * NLON;
        for (int j = 0; j < NLON; j++) {
            int j1 = (j + 1) % NLON;
            m->indices[idx++] = (uint32_t)(row0 + j);
            m->indices[idx++] = (uint32_t)(row1 + j);
            m->indices[idx++] = (uint32_t)(row1 + j1);
            m->indices[idx++] = (uint32_t)(row0 + j);
            m->indices[idx++] = (uint32_t)(row1 + j1);
            m->indices[idx++] = (uint32_t)(row0 + j1);
        }
    }
    /* South cap: pole 1 fans to ring NLAT-1 (last ring). */
    int last_row = 2 + (NLAT - 2) * NLON;
    for (int j = 0; j < NLON; j++) {
        int j1 = (j + 1) % NLON;
        m->indices[idx++] = 1;
        m->indices[idx++] = (uint32_t)(last_row + j1);
        m->indices[idx++] = (uint32_t)(last_row + j);
    }
    return 1;
}

/* ============================================================
 * Cylinder — capped tube.
 * UsdGeomCylinder spec: radius, height, axis. height = total length
 * along axis (-h/2 to +h/2). Default radius=1, height=2, axis=Z.
 * ============================================================ */

static int gen_cylinder(NanousdPrim prim, Arena* arena, SceneMesh* m)
{
    double radius = read_double_attr(prim, "radius", 1.0);
    double height = read_double_attr(prim, "height", 2.0);
    int axis = read_axis_attr(prim);
    const int NLON = 24;

    /* 2 cap centers + 2 rings (top/bottom) × NLON. Caps and sides share
     * positions but need different normals — duplicate so each side has
     * sharp shading at the rim. */
    const int NV = 2 + 2 * NLON  /* cap rim duplicates */
                     + 2 * NLON; /* side rim verts */
    const int NTRI = 2 * NLON    /* caps */
                     + 2 * NLON; /* side quads */
    const int NI = NTRI * 3;

    m->nvertices = NV;
    m->nindices  = NI;
    m->positions = (float*)arena_alloc(arena, (size_t)NV * 3 * sizeof(float), 16);
    m->normals   = (float*)arena_alloc(arena, (size_t)NV * 3 * sizeof(float), 16);
    m->indices   = (uint32_t*)arena_alloc(arena, (size_t)NI * sizeof(uint32_t), 16);
    if (!m->positions || !m->normals || !m->indices) return 0;

    float r = (float)radius;
    float hh = (float)(height * 0.5);

    /* Vertex layout:
     *   [0]               top cap center  (axis = +hh)
     *   [1]               bot cap center  (axis = -hh)
     *   [2 .. 2+NLON)     top cap rim, normals = +axis
     *   [2+NLON .. 2+2NLON)  bot cap rim, normals = -axis
     *   [2+2NLON .. 2+3NLON) side top rim, normals = outward radial
     *   [2+3NLON .. 2+4NLON) side bot rim, normals = outward radial
     */
    int top_center = 0, bot_center = 1;
    int top_cap_base = 2, bot_cap_base = top_cap_base + NLON;
    int side_top_base = bot_cap_base + NLON;
    int side_bot_base = side_top_base + NLON;

    /* Cap centers */
    { float p[3]; axis_permute(0, 0,  hh, axis, p);
      memcpy(&m->positions[top_center * 3], p, sizeof(p));
      float n[3]; axis_permute(0, 0, 1, axis, n);
      memcpy(&m->normals[top_center * 3], n, sizeof(n)); }
    { float p[3]; axis_permute(0, 0, -hh, axis, p);
      memcpy(&m->positions[bot_center * 3], p, sizeof(p));
      float n[3]; axis_permute(0, 0, -1, axis, n);
      memcpy(&m->normals[bot_center * 3], n, sizeof(n)); }

    /* Rim verts */
    for (int j = 0; j < NLON; j++) {
        double phi = (double)j * 2.0 * M_PI / (double)NLON;
        float cx = (float)cos(phi);
        float cy = (float)sin(phi);

        /* Top cap rim — normal +axis */
        { float p[3], n[3];
          axis_permute(cx * r, cy * r,  hh, axis, p);
          axis_permute(0, 0, 1, axis, n);
          memcpy(&m->positions[(top_cap_base + j) * 3], p, sizeof(p));
          memcpy(&m->normals  [(top_cap_base + j) * 3], n, sizeof(n)); }
        /* Bottom cap rim — normal -axis */
        { float p[3], n[3];
          axis_permute(cx * r, cy * r, -hh, axis, p);
          axis_permute(0, 0, -1, axis, n);
          memcpy(&m->positions[(bot_cap_base + j) * 3], p, sizeof(p));
          memcpy(&m->normals  [(bot_cap_base + j) * 3], n, sizeof(n)); }
        /* Side top rim — outward radial normal */
        { float p[3], n[3];
          axis_permute(cx * r, cy * r,  hh, axis, p);
          axis_permute(cx, cy, 0, axis, n);
          memcpy(&m->positions[(side_top_base + j) * 3], p, sizeof(p));
          memcpy(&m->normals  [(side_top_base + j) * 3], n, sizeof(n)); }
        /* Side bot rim — outward radial normal */
        { float p[3], n[3];
          axis_permute(cx * r, cy * r, -hh, axis, p);
          axis_permute(cx, cy, 0, axis, n);
          memcpy(&m->positions[(side_bot_base + j) * 3], p, sizeof(p));
          memcpy(&m->normals  [(side_bot_base + j) * 3], n, sizeof(n)); }
    }

    /* Indices */
    int idx = 0;
    /* Top cap fan */
    for (int j = 0; j < NLON; j++) {
        int j1 = (j + 1) % NLON;
        m->indices[idx++] = (uint32_t)top_center;
        m->indices[idx++] = (uint32_t)(top_cap_base + j);
        m->indices[idx++] = (uint32_t)(top_cap_base + j1);
    }
    /* Bottom cap fan (reversed winding) */
    for (int j = 0; j < NLON; j++) {
        int j1 = (j + 1) % NLON;
        m->indices[idx++] = (uint32_t)bot_center;
        m->indices[idx++] = (uint32_t)(bot_cap_base + j1);
        m->indices[idx++] = (uint32_t)(bot_cap_base + j);
    }
    /* Side quads */
    for (int j = 0; j < NLON; j++) {
        int j1 = (j + 1) % NLON;
        uint32_t t  = (uint32_t)(side_top_base + j);
        uint32_t t1 = (uint32_t)(side_top_base + j1);
        uint32_t b  = (uint32_t)(side_bot_base + j);
        uint32_t b1 = (uint32_t)(side_bot_base + j1);
        m->indices[idx++] = t;  m->indices[idx++] = b;  m->indices[idx++] = b1;
        m->indices[idx++] = t;  m->indices[idx++] = b1; m->indices[idx++] = t1;
    }
    return 1;
}

/* ============================================================
 * Capsule — cylinder body of length `height` with two hemispheres of
 * radius `radius` glued to the ends.
 * UsdGeomCapsule spec: radius (0.5), height (1.0), axis ("Z").
 * height is the length of the CYLINDRICAL SECTION (excluding hemis).
 * Tessellation: NLON=24, NLAT_HEMI=6.
 * ============================================================ */

static int gen_capsule(NanousdPrim prim, Arena* arena, SceneMesh* m)
{
    double radius = read_double_attr(prim, "radius", 0.5);
    double height = read_double_attr(prim, "height", 1.0);
    int axis = read_axis_attr(prim);

    const int NLON = 24;
    const int NLAT_HEMI = 6; /* latitude bands per hemisphere (≥ 2) */

    /* Hemisphere rings: 1 .. NLAT_HEMI (closest-to-equator). The
     * equator ring is shared with the cylinder rim. So per-hemisphere
     * we have 1 pole + (NLAT_HEMI - 1) intermediate rings, equator
     * ring belongs to the cylinder side.
     *
     * Total vertices:
     *   2 poles
     *   2 hemis × (NLAT_HEMI - 1) intermediate rings × NLON
     *   2 cylinder rims × NLON (equator top/bot)
     * Triangles:
     *   2 pole caps fans (NLON tris each)
     *   2 hemis × (NLAT_HEMI - 1) quad-ring layers × 2NLON tris
     *   1 cylinder side: 2 NLON tris (the equator ring shared)
     */
    const int rings_per_hemi = NLAT_HEMI - 1;
    const int NV  = 2 + 2 * rings_per_hemi * NLON + 2 * NLON;
    const int NTRI = 2 * NLON                                     /* pole fans */
                   + 2 * (NLAT_HEMI - 1) * 2 * NLON               /* hemi quad rings (incl. equator stitch) */
                   + 2 * NLON;                                    /* cylinder side */
    const int NI = NTRI * 3;

    m->nvertices = NV;
    m->nindices  = NI;
    m->positions = (float*)arena_alloc(arena, (size_t)NV * 3 * sizeof(float), 16);
    m->normals   = (float*)arena_alloc(arena, (size_t)NV * 3 * sizeof(float), 16);
    m->indices   = (uint32_t*)arena_alloc(arena, (size_t)NI * sizeof(uint32_t), 16);
    if (!m->positions || !m->normals || !m->indices) return 0;

    float r = (float)radius;
    float hh = (float)(height * 0.5); /* half-height of cylinder section */

    /* Vertex index layout:
     *   [0]                                       top pole
     *   [1]                                       bot pole
     *   [2 .. 2+rings_per_hemi*NLON)              top hemi rings (pole→equator-1)
     *   [next .. +rings_per_hemi*NLON)            bot hemi rings (pole→equator-1)
     *   [next .. +NLON)                           cylinder top rim (top hemi equator)
     *   [next .. +NLON)                           cylinder bot rim (bot hemi equator)
     */
    int top_pole = 0, bot_pole = 1;
    int top_hemi_base = 2;
    int bot_hemi_base = top_hemi_base + rings_per_hemi * NLON;
    int top_rim_base  = bot_hemi_base + rings_per_hemi * NLON;
    int bot_rim_base  = top_rim_base + NLON;

    /* Top pole vertex */
    { float p[3], n[3];
      axis_permute(0, 0, hh + r, axis, p);
      axis_permute(0, 0, 1, axis, n);
      memcpy(&m->positions[top_pole * 3], p, sizeof(p));
      memcpy(&m->normals  [top_pole * 3], n, sizeof(n)); }
    /* Bottom pole vertex */
    { float p[3], n[3];
      axis_permute(0, 0, -(hh + r), axis, p);
      axis_permute(0, 0, -1, axis, n);
      memcpy(&m->positions[bot_pole * 3], p, sizeof(p));
      memcpy(&m->normals  [bot_pole * 3], n, sizeof(n)); }

    /* Top hemi intermediate rings: ring k (1..rings_per_hemi)
     * Polar angle θ_k = k·π/(2·NLAT_HEMI), measured from +axis.
     * Position is offset by +hh along axis. */
    for (int k = 1; k <= rings_per_hemi; k++) {
        double theta = (double)k * (M_PI * 0.5) / (double)NLAT_HEMI;
        double st = sin(theta), ct = cos(theta);
        for (int j = 0; j < NLON; j++) {
            double phi = (double)j * 2.0 * M_PI / (double)NLON;
            float nx = (float)(st * cos(phi));
            float ny = (float)(st * sin(phi));
            float nz = (float)ct;
            int v = top_hemi_base + (k - 1) * NLON + j;
            float p[3], n[3];
            axis_permute(r * nx, r * ny, hh + r * nz, axis, p);
            axis_permute(nx, ny, nz, axis, n);
            memcpy(&m->positions[v * 3], p, sizeof(p));
            memcpy(&m->normals  [v * 3], n, sizeof(n));
        }
    }
    /* Bottom hemi: same but reflected. */
    for (int k = 1; k <= rings_per_hemi; k++) {
        double theta = (double)k * (M_PI * 0.5) / (double)NLAT_HEMI;
        double st = sin(theta), ct = cos(theta);
        for (int j = 0; j < NLON; j++) {
            double phi = (double)j * 2.0 * M_PI / (double)NLON;
            float nx = (float)(st * cos(phi));
            float ny = (float)(st * sin(phi));
            float nz = -(float)ct;
            int v = bot_hemi_base + (k - 1) * NLON + j;
            float p[3], n[3];
            axis_permute(r * nx, r * ny, -hh + r * nz, axis, p);
            axis_permute(nx, ny, nz, axis, n);
            memcpy(&m->positions[v * 3], p, sizeof(p));
            memcpy(&m->normals  [v * 3], n, sizeof(n));
        }
    }
    /* Cylinder rims: equator rings (top and bottom). */
    for (int j = 0; j < NLON; j++) {
        double phi = (double)j * 2.0 * M_PI / (double)NLON;
        float cx = (float)cos(phi);
        float cy = (float)sin(phi);
        { float p[3], n[3];
          axis_permute(r * cx, r * cy,  hh, axis, p);
          axis_permute(cx, cy, 0, axis, n);
          memcpy(&m->positions[(top_rim_base + j) * 3], p, sizeof(p));
          memcpy(&m->normals  [(top_rim_base + j) * 3], n, sizeof(n)); }
        { float p[3], n[3];
          axis_permute(r * cx, r * cy, -hh, axis, p);
          axis_permute(cx, cy, 0, axis, n);
          memcpy(&m->positions[(bot_rim_base + j) * 3], p, sizeof(p));
          memcpy(&m->normals  [(bot_rim_base + j) * 3], n, sizeof(n)); }
    }

    /* Indices */
    int idx = 0;
    /* Top hemi pole fan → ring 1 */
    {
        int ring1 = top_hemi_base;
        for (int j = 0; j < NLON; j++) {
            int j1 = (j + 1) % NLON;
            m->indices[idx++] = (uint32_t)top_pole;
            m->indices[idx++] = (uint32_t)(ring1 + j);
            m->indices[idx++] = (uint32_t)(ring1 + j1);
        }
    }
    /* Top hemi quad layers: ring k → ring k+1, for k=1..rings_per_hemi-1.
     * Then final layer: ring rings_per_hemi → top_rim_base (cylinder top). */
    for (int k = 1; k < rings_per_hemi; k++) {
        int row0 = top_hemi_base + (k - 1) * NLON;
        int row1 = top_hemi_base + k * NLON;
        for (int j = 0; j < NLON; j++) {
            int j1 = (j + 1) % NLON;
            m->indices[idx++] = (uint32_t)(row0 + j);
            m->indices[idx++] = (uint32_t)(row1 + j);
            m->indices[idx++] = (uint32_t)(row1 + j1);
            m->indices[idx++] = (uint32_t)(row0 + j);
            m->indices[idx++] = (uint32_t)(row1 + j1);
            m->indices[idx++] = (uint32_t)(row0 + j1);
        }
    }
    {
        /* Final top-hemi stitch to cylinder rim */
        int row0 = top_hemi_base + (rings_per_hemi - 1) * NLON;
        int row1 = top_rim_base;
        for (int j = 0; j < NLON; j++) {
            int j1 = (j + 1) % NLON;
            m->indices[idx++] = (uint32_t)(row0 + j);
            m->indices[idx++] = (uint32_t)(row1 + j);
            m->indices[idx++] = (uint32_t)(row1 + j1);
            m->indices[idx++] = (uint32_t)(row0 + j);
            m->indices[idx++] = (uint32_t)(row1 + j1);
            m->indices[idx++] = (uint32_t)(row0 + j1);
        }
    }

    /* Cylinder side: top rim → bot rim */
    {
        int rt = top_rim_base;
        int rb = bot_rim_base;
        for (int j = 0; j < NLON; j++) {
            int j1 = (j + 1) % NLON;
            m->indices[idx++] = (uint32_t)(rt + j);
            m->indices[idx++] = (uint32_t)(rb + j);
            m->indices[idx++] = (uint32_t)(rb + j1);
            m->indices[idx++] = (uint32_t)(rt + j);
            m->indices[idx++] = (uint32_t)(rb + j1);
            m->indices[idx++] = (uint32_t)(rt + j1);
        }
    }

    /* Bot hemi quad layers: bottom rim → ring rings_per_hemi → ring 1 → pole. */
    {
        /* First layer: cylinder bot rim → bot hemi ring rings_per_hemi-1 */
        int row0 = bot_rim_base;
        int row1 = bot_hemi_base + (rings_per_hemi - 1) * NLON;
        for (int j = 0; j < NLON; j++) {
            int j1 = (j + 1) % NLON;
            m->indices[idx++] = (uint32_t)(row0 + j);
            m->indices[idx++] = (uint32_t)(row1 + j);
            m->indices[idx++] = (uint32_t)(row1 + j1);
            m->indices[idx++] = (uint32_t)(row0 + j);
            m->indices[idx++] = (uint32_t)(row1 + j1);
            m->indices[idx++] = (uint32_t)(row0 + j1);
        }
    }
    for (int k = rings_per_hemi - 1; k >= 1; k--) {
        int row0 = bot_hemi_base + k * NLON;
        int row1 = bot_hemi_base + (k - 1) * NLON;
        for (int j = 0; j < NLON; j++) {
            int j1 = (j + 1) % NLON;
            m->indices[idx++] = (uint32_t)(row0 + j);
            m->indices[idx++] = (uint32_t)(row1 + j);
            m->indices[idx++] = (uint32_t)(row1 + j1);
            m->indices[idx++] = (uint32_t)(row0 + j);
            m->indices[idx++] = (uint32_t)(row1 + j1);
            m->indices[idx++] = (uint32_t)(row0 + j1);
        }
    }
    /* Bot hemi: ring 1 → pole */
    {
        int row = bot_hemi_base;
        for (int j = 0; j < NLON; j++) {
            int j1 = (j + 1) % NLON;
            m->indices[idx++] = (uint32_t)(row + j1);
            m->indices[idx++] = (uint32_t)(row + j);
            m->indices[idx++] = (uint32_t)bot_pole;
        }
    }
    return 1;
}

/* ============================================================
 * Public dispatch.
 * ============================================================ */

int load_primitive_geom(NanousdPrim prim,
                        const char* type_name,
                        Arena* arena,
                        SceneMesh* out_mesh)
{
    if (!type_name || !out_mesh || !arena) return 0;
    if (!strcmp(type_name, "Cube"))     return gen_cube(prim, arena, out_mesh);
    if (!strcmp(type_name, "Sphere"))   return gen_sphere(prim, arena, out_mesh);
    if (!strcmp(type_name, "Cylinder")) return gen_cylinder(prim, arena, out_mesh);
    if (!strcmp(type_name, "Capsule"))  return gen_capsule(prim, arena, out_mesh);
    return 0;
}

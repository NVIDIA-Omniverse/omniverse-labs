// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* cull_stats.c — measure the value of a per-meshlet frustum-cull draw path.
 *
 * The renderer already frustum-culls whole meshes (viewer.c). A per-meshlet
 * cull would only help meshes that *straddle* the frustum boundary — a mesh
 * fully inside draws whole, a mesh fully outside is already rejected. This
 * tool classifies every mesh as inside / straddling / outside several camera
 * frusta, then classifies each straddle mesh's meshlets (world-space sphere
 * vs frustum) to estimate the realized per-meshlet cull saving. Pure CPU.
 *
 * Usage: cull_stats <scene.usd>
 */
#include "scene.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --- minimal row-major 4x4 math (clip = M * v, v a column vector) --- */

static void mat4_mul(const float* a, const float* b, float* out)
{
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            float s = 0.0f;
            for (int k = 0; k < 4; k++) s += a[r*4+k] * b[k*4+c];
            out[r*4+c] = s;
        }
}

static void v3_sub(const float* a, const float* b, float* o)
{
    o[0]=a[0]-b[0]; o[1]=a[1]-b[1]; o[2]=a[2]-b[2];
}
static void v3_norm(float* v)
{
    float l = sqrtf(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    if (l > 1e-20f) { v[0]/=l; v[1]/=l; v[2]/=l; }
}
static void v3_cross(const float* a, const float* b, float* o)
{
    o[0]=a[1]*b[2]-a[2]*b[1];
    o[1]=a[2]*b[0]-a[0]*b[2];
    o[2]=a[0]*b[1]-a[1]*b[0];
}
static float v3_dot(const float* a, const float* b)
{
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

/* Transform an object-space point by a USD row-major world_xform
 * (v' = v . M, translation at M[12..14]). */
static void xform_point(const double* M, const float* p, float* o)
{
    for (int j = 0; j < 3; j++)
        o[j] = (float)((double)p[0]*M[j]   + (double)p[1]*M[4+j]
                     + (double)p[2]*M[8+j] + M[12+j]);
}

/* Largest basis-axis length of a USD world_xform — a conservative uniform
 * scale for transforming a bounding-sphere radius to world space. */
static float xform_max_scale(const double* M)
{
    float best = 0.0f;
    for (int i = 0; i < 3; i++) {
        double a = M[i*4+0], b = M[i*4+1], c = M[i*4+2];
        float len = (float)sqrt(a*a + b*b + c*c);
        if (len > best) best = len;
    }
    return best;
}

/* Row-major look-at: world -> camera (camera looks down -z). */
static void look_at(const float* eye, const float* center, const float* up_in,
                    float* v)
{
    float fwd[3], right[3], u[3], up[3];
    up[0]=up_in[0]; up[1]=up_in[1]; up[2]=up_in[2];
    v3_sub(center, eye, fwd); v3_norm(fwd);
    v3_cross(fwd, up, right);
    if (right[0]*right[0]+right[1]*right[1]+right[2]*right[2] < 1e-12f) {
        /* fwd parallel to up — fall back to an up axis not parallel to fwd. */
        up[0]=1; up[1]=0; up[2]=0;
        if (fabsf(fwd[0]) > 0.9f) { up[0]=0; up[1]=1; up[2]=0; }
        v3_cross(fwd, up, right);
    }
    v3_norm(right);
    v3_cross(right, fwd, u);
    v[0]=right[0]; v[1]=right[1]; v[2]=right[2]; v[3]=-v3_dot(right,eye);
    v[4]=u[0];     v[5]=u[1];     v[6]=u[2];     v[7]=-v3_dot(u,eye);
    v[8]=-fwd[0];  v[9]=-fwd[1];  v[10]=-fwd[2]; v[11]=v3_dot(fwd,eye);
    v[12]=0; v[13]=0; v[14]=0; v[15]=1;
}

/* Row-major GL perspective. */
static void perspective(float fovy_deg, float aspect, float zn, float zf,
                        float* p)
{
    float f = 1.0f / tanf(0.5f * fovy_deg * 3.14159265358979f / 180.0f);
    memset(p, 0, 16*sizeof(float));
    p[0]  = f / aspect;
    p[5]  = f;
    p[10] = (zf + zn) / (zn - zf);
    p[11] = (2.0f*zf*zn) / (zn - zf);
    p[14] = -1.0f;
}

/* Gribb-Hartmann: 6 frustum planes from a row-major VP. */
static void frustum_planes(const float* vp, float planes[6][4])
{
    const float* m0 = vp;      const float* m1 = vp+4;
    const float* m2 = vp+8;    const float* m3 = vp+12;
    for (int i = 0; i < 4; i++) {
        planes[0][i] = m3[i] + m0[i];   /* left   */
        planes[1][i] = m3[i] - m0[i];   /* right  */
        planes[2][i] = m3[i] + m1[i];   /* bottom */
        planes[3][i] = m3[i] - m1[i];   /* top    */
        planes[4][i] = m3[i] + m2[i];   /* near   */
        planes[5][i] = m3[i] - m2[i];   /* far    */
    }
}

enum { CLASS_INSIDE = 0, CLASS_STRADDLE = 1, CLASS_OUTSIDE = 2 };

/* Classify a world-space AABB against 6 planes (inside = a*x+b*y+c*z+d >= 0). */
static int classify_aabb(const float* mn, const float* mx,
                         const float planes[6][4])
{
    int straddle = 0;
    for (int p = 0; p < 6; p++) {
        float a = planes[p][0], b = planes[p][1], c = planes[p][2], d = planes[p][3];
        float px = a >= 0 ? mx[0] : mn[0];
        float py = b >= 0 ? mx[1] : mn[1];
        float pz = c >= 0 ? mx[2] : mn[2];
        if (a*px + b*py + c*pz + d < 0.0f) return CLASS_OUTSIDE;
        float nx = a >= 0 ? mn[0] : mx[0];
        float ny = b >= 0 ? mn[1] : mx[1];
        float nz = c >= 0 ? mn[2] : mx[2];
        if (a*nx + b*ny + c*nz + d < 0.0f) straddle = 1;
    }
    return straddle ? CLASS_STRADDLE : CLASS_INSIDE;
}

static void report(const char* label, const Scene* s, const float* vp)
{
    float planes[6][4];
    frustum_planes(vp, planes);

    /* Normalized plane copy — the meshlet bounding-sphere test needs a
     * metric (unit-normal) signed distance. */
    float nplanes[6][4];
    for (int p = 0; p < 6; p++) {
        float a=planes[p][0], b=planes[p][1], c=planes[p][2];
        float l = sqrtf(a*a + b*b + c*c);
        if (l < 1e-20f) l = 1.0f;
        nplanes[p][0]=a/l; nplanes[p][1]=b/l; nplanes[p][2]=c/l;
        nplanes[p][3]=planes[p][3]/l;
    }

    long n_in = 0, n_str = 0, n_out = 0;
    long long tri_in = 0, tri_str = 0;
    long long ml_str = 0;
    long long cull_ml = 0, cull_tris = 0;   /* per-meshlet cull of straddle meshes */
    long long ctr_in = 0, ctr_total = 0;    /* world-transform validation */
    for (int i = 0; i < s->nmeshes; i++) {
        const SceneMesh* m = &s->meshes[i];
        if (m->nindices == 0 || m->is_proto_only) continue;
        int cl = classify_aabb(m->bounds_min, m->bounds_max, planes);
        long long tris = m->nindices / 3;
        if (cl == CLASS_OUTSIDE) { n_out++; continue; }
        if (cl == CLASS_INSIDE)  { n_in++; tri_in += tris; continue; }

        /* Straddle — classify each meshlet (world-space sphere vs frustum).
         * A meshlet whose sphere is fully outside any plane is cullable;
         * conservative, since the sphere bounds every triangle in it. */
        n_str++; tri_str += tris; ml_str += m->meshlet_count;
        float msc = xform_max_scale(m->world_xform);
        for (uint32_t k = 0; k < m->meshlet_count; k++) {
            const SceneMeshlet* ml = &s->meshlets[m->meshlet_offset + k];
            float wc[3];
            xform_point(m->world_xform, ml->center, wc);
            float wr = ml->radius * msc;
            ctr_total++;
            if (wc[0] >= m->bounds_min[0]-1e-3f && wc[0] <= m->bounds_max[0]+1e-3f &&
                wc[1] >= m->bounds_min[1]-1e-3f && wc[1] <= m->bounds_max[1]+1e-3f &&
                wc[2] >= m->bounds_min[2]-1e-3f && wc[2] <= m->bounds_max[2]+1e-3f)
                ctr_in++;
            int outside = 0;
            for (int p = 0; p < 6; p++) {
                float d = nplanes[p][0]*wc[0] + nplanes[p][1]*wc[1]
                        + nplanes[p][2]*wc[2] + nplanes[p][3];
                if (d < -wr) { outside = 1; break; }
            }
            if (outside) { cull_ml++; cull_tris += ml->triangle_count; }
        }
    }
    long drawn = n_in + n_str;
    long long tri_drawn = tri_in + tri_str;
    double str_pct  = drawn ? 100.0*(double)n_str/(double)drawn : 0.0;
    double cull_pct = ml_str ? 100.0*(double)cull_ml/(double)ml_str : 0.0;
    double save_pct = tri_drawn ? 100.0*(double)cull_tris/(double)tri_drawn : 0.0;
    double valid    = ctr_total ? 100.0*(double)ctr_in/(double)ctr_total : 100.0;
    printf("[%s]\n", label);
    printf("  meshes:   %ld culled, %ld inside, %ld straddle  "
           "(%.1f%% of %ld drawn straddle)\n", n_out, n_in, n_str, str_pct, drawn);
    printf("  per-meshlet cull: %lld of %lld straddle-mesh meshlets fully "
           "outside (%.1f%%)\n", cull_ml, ml_str, cull_pct);
    printf("  realized saving:  %lld triangles = %.1f%% of the %lld submitted\n",
           cull_tris, save_pct, tri_drawn);
    printf("  validation: %.2f%% of meshlet centers within their mesh AABB\n",
           valid);
    printf("\n");
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: cull_stats <scene.usd>\n");
        return 2;
    }
    const char* usd = argv[1];
    setenv("NUSD_GEO_CACHE", "1", 1);   /* warm-load if a cache exists */

    Scene* s = scene_load(usd);
    if (!s || s->nmeshes <= 0) {
        fprintf(stderr, "FAIL: scene_load(%s)\n", usd);
        return 1;
    }

    const float* bmn = s->bounds_min;
    const float* bmx = s->bounds_max;
    float c[3] = { 0.5f*(bmn[0]+bmx[0]), 0.5f*(bmn[1]+bmx[1]),
                   0.5f*(bmn[2]+bmx[2]) };
    float ext[3] = { bmx[0]-bmn[0], bmx[1]-bmn[1], bmx[2]-bmn[2] };
    float radius = 0.5f * sqrtf(ext[0]*ext[0]+ext[1]*ext[1]+ext[2]*ext[2]);
    if (radius < 1e-6f) radius = 1.0f;
    /* up axis: 2 = Z-up (Omniverse/DSX), else Y-up. */
    float up[3] = { 0, 1, 0 };
    if (s->up_axis == 2) { up[0]=0; up[1]=0; up[2]=1; }

    printf("scene: %d meshes, bounds (%.2f %.2f %.2f)-(%.2f %.2f %.2f)\n",
           s->nmeshes, bmn[0],bmn[1],bmn[2], bmx[0],bmx[1],bmx[2]);
    printf("       center (%.2f %.2f %.2f), radius %.2f, %d meshlets total\n\n",
           c[0],c[1],c[2], radius, s->nmeshlets);

    float view[16], proj[16], vp[16];
    float eye[3], tgt[3];

    /* Camera 1 — whole-scene framing from an oblique angle (the offset has
     * all three axes, so it is never parallel to the up axis). Sanity check:
     * a whole-scene framing must cull ~0 — if it does not, the projection /
     * plane-extraction convention is wrong. */
    {
        float d[3] = { 0.40f, -0.84f, 0.36f };   /* oblique unit-ish dir */
        eye[0]=c[0]+5.0f*radius*d[0];
        eye[1]=c[1]+5.0f*radius*d[1];
        eye[2]=c[2]+5.0f*radius*d[2];
    }
    perspective(50.0f, 1.0f, 0.01f*radius, 20.0f*radius, proj);
    look_at(eye, c, up, view);
    mat4_mul(proj, view, vp);
    report("whole-scene (sanity: expect ~0 culled)", s, vp);

    /* Camera 2 — interior, looking along +X from the center. */
    eye[0]=c[0]; eye[1]=c[1]; eye[2]=c[2];
    tgt[0]=c[0]+radius; tgt[1]=c[1]; tgt[2]=c[2];
    perspective(60.0f, 1.0f, 0.001f*radius, 10.0f*radius, proj);
    look_at(eye, tgt, up, view);
    mat4_mul(proj, view, vp);
    report("interior, 60deg, looking +X from center", s, vp);

    /* Camera 3 — corner viewpoint looking back at the scene center. */
    eye[0]=bmx[0]+radius; eye[1]=bmx[1]+radius; eye[2]=bmx[2]+radius;
    perspective(45.0f, 1.0f, 0.01f*radius, 10.0f*radius, proj);
    look_at(eye, c, up, view);
    mat4_mul(proj, view, vp);
    report("corner, 45deg, looking at center", s, vp);

    /* Camera 4 — close interior, narrow FOV (a tight viewport). */
    eye[0]=c[0]-0.5f*radius; eye[1]=c[1]; eye[2]=c[2];
    tgt[0]=c[0]+radius; tgt[1]=c[1]; tgt[2]=c[2];
    perspective(25.0f, 1.0f, 0.001f*radius, 10.0f*radius, proj);
    look_at(eye, tgt, up, view);
    mat4_mul(proj, view, vp);
    report("close interior, 25deg narrow FOV", s, vp);

    scene_free(s);
    return 0;
}

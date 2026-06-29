/*
 * Verifies cubic BasisCurves segment extraction uses the authored basis for
 * both positions and widths. Ironwood's native-instance curves are cubic
 * B-splines; treating them as Catmull-Rom makes branches visibly displaced.
 */

#include "../src/scene.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int nearf(float a, float b)
{
    return fabsf(a - b) < 1.0e-5f;
}

static int nearf_eps(float a, float b, float eps)
{
    return fabsf(a - b) < eps;
}

static int checkf(const char* label, float got, float expected)
{
    if (!nearf(got, expected)) {
        fprintf(stderr, "FAIL: %s got %.8f expected %.8f\n", label, got, expected);
        return 0;
    }
    return 1;
}

static float sign_not_zero(float v)
{
    return v < 0.0f ? -1.0f : 1.0f;
}

static void decode_ribbon_normal(uint32_t mat_flags, float out[3])
{
    float x = (float)(mat_flags & SCENE_CURVE_SEG_OCT_MASK) *
              (1.0f / 32767.0f) * 2.0f - 1.0f;
    float y = (float)((mat_flags >> SCENE_CURVE_SEG_OCT_SHIFT_Y) &
                      SCENE_CURVE_SEG_OCT_MASK) *
              (1.0f / 32767.0f) * 2.0f - 1.0f;
    float z = 1.0f - fabsf(x) - fabsf(y);
    if (z < 0.0f) {
        float old_x = x;
        x = (1.0f - fabsf(y)) * sign_not_zero(old_x);
        y = (1.0f - fabsf(old_x)) * sign_not_zero(y);
    }
    float len = sqrtf(x*x + y*y + z*z);
    out[0] = x / len;
    out[1] = y / len;
    out[2] = z / len;
}

int main(void)
{
    setenv("NUSD_CURVE_SUBSEGS", "2", 1);

    float cvs[] = {
        0.0f, 0.0f, 0.0f,
        0.0f, 6.0f, 0.0f,
        6.0f, 6.0f, 0.0f,
        6.0f, 0.0f, 0.0f,
    };
    float widths[] = {
        0.0f, 0.0f, 6.0f, 0.0f,
    };
    float colors[] = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
    };
    float normals[] = {
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
    };
    int curve_vertex_counts[] = {4};

    SceneCurve curve = {0};
    curve.cvs = cvs;
    curve.widths = widths;
    curve.colors = colors;
    curve.normals = normals;
    curve.nv = 4;
    curve.ncurves_in_prim = 1;
    curve.curve_vertex_counts = curve_vertex_counts;
    curve.basis = CURVE_BASIS_BSPLINE;
    curve.wrap = CURVE_WRAP_NONPERIODIC;
    curve.type_is_cubic = 1;
    curve.has_normals = 1;
    for (int i = 0; i < 16; i++) {
        curve.world_xform[i] = (i % 5 == 0) ? 1.0 : 0.0;
        curve.normal_xform[i] = (i % 5 == 0) ? 1.0 : 0.0;
    }

    int nsegs = scene_curve_count_segments(&curve);
    if (nsegs != 2) {
        fprintf(stderr, "FAIL: scene_curve_count_segments got %d expected 2\n", nsegs);
        return 1;
    }

    SceneCurveSegment segs[2];
    int written = scene_curve_to_segments(&curve, segs);
    if (written != 2) {
        fprintf(stderr, "FAIL: scene_curve_to_segments got %d expected 2\n", written);
        return 1;
    }

    int ok = 1;
    ok &= checkf("seg0.p0.x", segs[0].p0[0], 1.0f);
    ok &= checkf("seg0.p0.y", segs[0].p0[1], 5.0f);
    ok &= checkf("seg0.p1.x", segs[0].p1[0], 3.0f);
    ok &= checkf("seg0.p1.y", segs[0].p1[1], 5.75f);
    ok &= checkf("seg0.r0", segs[0].r0, 0.5f);
    ok &= checkf("seg1.r0", segs[1].r0, 1.4375f);

    if ((segs[0].mat_flags & SCENE_CURVE_SEG_FLAG_RIBBON) == 0) {
        fprintf(stderr, "FAIL: seg0 missing ribbon flag\n");
        ok = 0;
    }
    if ((segs[0].mat_flags & SCENE_CURVE_SEG_FLAG_RIBBON_JOIN_PAD) == 0) {
        fprintf(stderr, "FAIL: seg0 missing ribbon join-pad flag\n");
        ok = 0;
    }
    float decoded_n[3];
    decode_ribbon_normal(segs[0].mat_flags, decoded_n);
    if (!nearf_eps(decoded_n[0], 0.0f, 1.0e-4f) ||
        !nearf_eps(decoded_n[1], 0.0f, 1.0e-4f) ||
        !nearf_eps(decoded_n[2], 1.0f, 1.0e-4f)) {
        fprintf(stderr, "FAIL: decoded ribbon normal got (%.8f, %.8f, %.8f)\n",
                decoded_n[0], decoded_n[1], decoded_n[2]);
        ok = 0;
    }

    float seg_colors[6];
    int ncolors = scene_curve_to_segment_colors(&curve, seg_colors);
    if (ncolors != 2) {
        fprintf(stderr, "FAIL: scene_curve_to_segment_colors got %d expected 2\n", ncolors);
        ok = 0;
    } else {
        ok &= checkf("seg0.color.r", seg_colors[0], 1.0f / 6.0f);
        ok &= checkf("seg1.color.r", seg_colors[3], 23.0f / 48.0f);
        ok &= checkf("seg0.color.g", seg_colors[1], 0.0f);
        ok &= checkf("seg1.color.b", seg_colors[5], 0.0f);
    }

    if (!ok) return 1;
    printf("test_scene_curve_basis: PASS\n");
    return 0;
}

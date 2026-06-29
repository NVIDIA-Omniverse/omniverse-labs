/*
 * Cross-check Moana element wrappers whose composed Storm view exposes
 * BasisCurves, including curves inside native instance proxies.
 *
 * This is CPU-only: it loads the scene, counts SceneCurve payloads, validates
 * B-spline topology/segment counts, and checks authored width statistics.
 */

#include "../src/scene.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char* name;
    const char* path;
    int expected_curves;
    long long expected_points;
    long long expected_subcurves;
    long long expected_segments;
    double expected_wmin;
    double expected_wmax;
    double expected_wmean;
    int expected_display_curves;  /* -1 = not asserted for this fixture */
    int expected_ribbon_curves;   /* curves with authored normals */
    int expected_st_curves;       /* curves with authored primvars:st */
} CurveAssetExpect;

static int file_exists(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static void test_setenv(const char* name, const char* value)
{
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static void test_unsetenv(const char* name)
{
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

static int close_enough(double got, double expected)
{
    double tol = fabs(expected) * 1.0e-6;
    if (tol < 1.0e-7) tol = 1.0e-7;
    return fabs(got - expected) <= tol;
}

static int expected_curve_subsegs(void)
{
    const char* env = getenv("NUSD_CURVE_SUBSEGS");
    if (!env || !env[0]) return 8;
    char* end = NULL;
    long v = strtol(env, &end, 10);
    if (end == env) return 8;
    if (v < 1) v = 1;
    if (v > 32) v = 32;
    return (int)v;
}

static int check_asset(const CurveAssetExpect* ex)
{
    Scene* scene = scene_load(ex->path);
    if (!scene) {
        fprintf(stderr, "FAIL[%s]: scene_load returned NULL for %s\n", ex->name, ex->path);
        return 0;
    }

    long long points = 0;
    long long widths = 0;
    long long subcurves = 0;
    long long segments = 0;
    int bspline_curves = 0;
    int cubic_nonperiodic_curves = 0;
    int display_colored_curves = 0;
    int ribbon_curves = 0;
    int textured_curve_prims = 0;
    double wsum = 0.0;
    float wmin = FLT_MAX;
    float wmax = -FLT_MAX;

    for (int i = 0; i < scene->ncurves; i++) {
        const SceneCurve* c = &scene->curves[i];
        points += c->nv;
        widths += c->nv;
        subcurves += c->ncurves_in_prim;
        segments += scene_curve_count_segments(c);
        if (c->basis == CURVE_BASIS_BSPLINE) bspline_curves++;
        if (c->type_is_cubic && c->wrap == CURVE_WRAP_NONPERIODIC)
            cubic_nonperiodic_curves++;
        if (c->has_display_color) display_colored_curves++;
        if (c->has_normals) ribbon_curves++;
        if (c->has_texcoords) textured_curve_prims++;
        if (!c->widths) {
            fprintf(stderr, "FAIL[%s]: curve %d has no widths\n", ex->name, i);
            scene_free(scene);
            return 0;
        }
        for (int v = 0; v < c->nv; v++) {
            float w = c->widths[v];
            if (w < wmin) wmin = w;
            if (w > wmax) wmax = w;
            wsum += (double)w;
        }
    }

    double wmean = widths ? wsum / (double)widths : 0.0;
    printf("moana_curve_asset[%s]: curves=%d points=%lld subcurves=%lld "
           "segments=%lld width[min,max,mean]=%.12g,%.12g,%.12g "
           "bspline=%d cubic_nonperiodic=%d displayColor=%d ribbons=%d st=%d\n",
           ex->name, scene->ncurves, points, subcurves, segments,
           (double)wmin, (double)wmax, wmean,
           bspline_curves, cubic_nonperiodic_curves,
           display_colored_curves, ribbon_curves, textured_curve_prims);
    fflush(stdout);

    int ok = 1;
    if (scene->ncurves != ex->expected_curves) {
        fprintf(stderr, "FAIL[%s]: curves got %d expected %d\n",
                ex->name, scene->ncurves, ex->expected_curves);
        ok = 0;
    }
    if (points != ex->expected_points || widths != ex->expected_points) {
        fprintf(stderr, "FAIL[%s]: points/widths got %lld/%lld expected %lld\n",
                ex->name, points, widths, ex->expected_points);
        ok = 0;
    }
    if (subcurves != ex->expected_subcurves) {
        fprintf(stderr, "FAIL[%s]: subcurves got %lld expected %lld\n",
                ex->name, subcurves, ex->expected_subcurves);
        ok = 0;
    }
    long long expected_segments =
        (ex->expected_segments / 8LL) * (long long)expected_curve_subsegs();
    if (segments != expected_segments) {
        fprintf(stderr, "FAIL[%s]: segments got %lld expected %lld\n",
                ex->name, segments, expected_segments);
        ok = 0;
    }
    if (bspline_curves != ex->expected_curves ||
        cubic_nonperiodic_curves != ex->expected_curves) {
        fprintf(stderr, "FAIL[%s]: expected all %d curves to be cubic nonperiodic bspline, got bspline=%d cubic_nonperiodic=%d\n",
                ex->name, ex->expected_curves, bspline_curves, cubic_nonperiodic_curves);
        ok = 0;
    }
    if (!close_enough((double)wmin, ex->expected_wmin) ||
        !close_enough((double)wmax, ex->expected_wmax) ||
        !close_enough(wmean, ex->expected_wmean)) {
        fprintf(stderr, "FAIL[%s]: width stats got %.12g,%.12g,%.12g expected %.12g,%.12g,%.12g\n",
                ex->name, (double)wmin, (double)wmax, wmean,
                ex->expected_wmin, ex->expected_wmax, ex->expected_wmean);
        ok = 0;
    }
    if (ex->expected_display_curves >= 0 &&
        display_colored_curves != ex->expected_display_curves) {
        fprintf(stderr, "FAIL[%s]: displayColor curves got %d expected %d\n",
                ex->name, display_colored_curves, ex->expected_display_curves);
        ok = 0;
    }
    if (ex->expected_ribbon_curves >= 0 &&
        ribbon_curves != ex->expected_ribbon_curves) {
        fprintf(stderr, "FAIL[%s]: ribbon curves got %d expected %d\n",
                ex->name, ribbon_curves, ex->expected_ribbon_curves);
        ok = 0;
    }
    if (ex->expected_st_curves >= 0 &&
        textured_curve_prims != ex->expected_st_curves) {
        fprintf(stderr, "FAIL[%s]: st curve prims got %d expected %d\n",
                ex->name, textured_curve_prims, ex->expected_st_curves);
        ok = 0;
    }

    scene_free(scene);
    return ok;
}

static int should_run_asset(const CurveAssetExpect* ex, int argc, char** argv)
{
    if (argc <= 1) return 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--all") || !strcmp(argv[i], ex->name))
            return 1;
    }
    return 0;
}

int main(int argc, char** argv)
{
    static const CurveAssetExpect assets[] = {
        { "isBeach", "$HOME/moana-island-scene-usd/island/usd/elements/isBeach/element.usda", 1, 64800LL, 5400LL, 388800LL, 0.0759999975562, 0.300000011921, 0.187999999151, -1, -1, -1 },
        { "isCoastline", "$HOME/moana-island-scene-usd/island/usd/elements/isCoastline/element.usda", 1, 4812LL, 401LL, 28872LL, 0.0759999975562, 0.300000011921, 0.187999999151, -1, -1, -1 },
        { "isDunesB", "$HOME/moana-island-scene-usd/island/usd/elements/isDunesB/element.usda", 62, 3383352LL, 494586LL, 15196752LL, 0.0, 4.06831789017, 0.103712459108, -1, -1, -1 },
        { "isIronwoodA1", "$HOME/moana-island-scene-usd/island/usd/elements/isIronwoodA1/element.usda", 56, 6870612LL, 1087974LL, 28853520LL, 0.0, 3.25, 0.0247535655693, 56, 0, 0 },
        { "isIronwoodB", "$HOME/moana-island-scene-usd/island/usd/elements/isIronwoodB/element.usda", 29, 3249679LL, 514748LL, 13643480LL, 0.0, 3.25, 0.0313807232053, 29, 0, 0 },
        { "isMountainB", "$HOME/moana-island-scene-usd/island/usd/elements/isMountainB/element.usda", 5, 25915475LL, 5183095LL, 82929520LL, 16.0000019073, 29.9999923706, 23.0059897881, -1, -1, -1 },
        { "isPalmRig", "$HOME/moana-island-scene-usd/island/usd/elements/isPalmRig/element.usda", 66, 3812952LL, 173316LL, 26344032LL, 0.0500000007451, 0.967431306839, 0.567872541025, 66, 66, 66 },
        { "isPandanusA", "$HOME/moana-island-scene-usd/island/usd/elements/isPandanusA/element.usda", 80, 2372336LL, 183720LL, 14569408LL, 0.00999999977648, 1.75, 1.25396259819, 80, 80, 80 },
    };

    const int nassets = (int)(sizeof(assets) / sizeof(assets[0]));
    for (int i = 0; i < nassets; i++) {
        if (!should_run_asset(&assets[i], argc, argv)) continue;
        if (!file_exists(assets[i].path)) {
            printf("SKIP: Moana curve asset missing: %s\n", assets[i].path);
            return 77;
        }
    }

    test_setenv("NUSD_FLAT_NATIVE_INSTANCE_TRAVERSAL", "1");
    test_setenv("NUSD_RENDER_PI_BATCHES", "1");
    test_setenv("NUSD_APPLY_METERS_PER_UNIT", "0");
    test_unsetenv("NUSD_NATIVE_CURVES");

    int ran = 0;
    for (int i = 0; i < nassets; i++) {
        if (!should_run_asset(&assets[i], argc, argv)) continue;
        ran++;
        if (!check_asset(&assets[i])) return 1;
    }

    if (ran == 0) {
        fprintf(stderr, "FAIL: no matching Moana curve asset requested\n");
        return 1;
    }

    printf("test_moana_curve_assets: PASS\n");
    return 0;
}

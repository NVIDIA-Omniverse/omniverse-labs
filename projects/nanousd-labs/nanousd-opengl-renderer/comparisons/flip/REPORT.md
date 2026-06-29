# OpenGL vs OVRTX — ϜLIP diff report

Each OpenGL backend output is compared to its OVRTX golden with the calibrated FLIP metric. Studio scenes: subject composited onto the golden background and foreground-masked, scored by mean FLIP. Full scenes: 95th-percentile FLIP (the mean is gameable — a tone-matched-but-murky render can beat a good one), plus orthogonal GMSD (structure), CIEDE2000 (exposure-suppressed colour), crushed-black and exposure gate legs. **Lower = closer to the golden.** Single-pass backends sit well above the path-traced golden even when they look right — read the *relative* ordering and the heatmaps, and judge full scenes by eye, not the scalar alone.

**Systematic finding:** vs the OVRTX golden, 12 OpenGL renders are brighter and 4 darker (single-pass lacks path-traced occlusion). Largest perceptual gap: `warehouse/warehouse_camB/opengl` at FLIP 0.866 (7% brighter, worst error center).

**Regression gate: ✅ PASS**  (tol +0.03; ceilings studio 0.95 / fullscene 0.99; +GMSD/crush/exposure legs on full scenes; 0 new)

## Largest diffs from golden

| pair | FLIP | GMSD | ΔE2000 | mode | diagnosis |
|---|---|---|---|---|---|
| `warehouse/warehouse_camB/opengl` | 0.866 | 0.226 | 8.40 | fullscene | 7% brighter, worst error center |
| `warehouse/warehouse_camA/opengl` | 0.820 | 0.211 | 7.72 | fullscene | 7% darker, warmer, worst error center |
| `chess/chess_set_camA/opengl` | 0.481 | 0.108 | 9.77 | studio | 11% darker, warmer, worst error center |
| `apple/fender_stratocaster_camB/opengl` | 0.449 | 0.076 | 8.99 | studio | 0% brighter, worst error lower-center |
| `apple/toy_drummer_camB/opengl` | 0.432 | 0.059 | 7.37 | studio | 3% brighter, cooler, worst error center |

## chess

| pair | backend | FLIP | status | baseline | diagnosis |
|---|---|---|---|---|---|
| chess_set camA | opengl | 0.481 | pass | 0.481 | 11% darker, warmer, worst error center |
| chess_set camB | opengl | 0.271 | pass | 0.271 | 12% brighter, worst error lower-left |

## apple

| pair | backend | FLIP | status | baseline | diagnosis |
|---|---|---|---|---|---|
| ball_soccerball camA | opengl | 0.351 | pass | 0.351 | 17% brighter, worst error lower-right |
| ball_soccerball camB | opengl | 0.341 | pass | 0.341 | 14% brighter, warmer, worst error lower-right |
| fender_stratocaster camA | opengl | 0.412 | pass | 0.412 | 10% brighter, cooler, worst error lower-center |
| fender_stratocaster camB | opengl | 0.449 | pass | 0.449 | 0% brighter, worst error lower-center |
| pancakes camA | opengl | 0.428 | pass | 0.428 | 8% brighter, worst error mid-left |
| pancakes camB | opengl | 0.431 | pass | 0.431 | 14% brighter, warmer, worst error center |
| robot camA | opengl | 0.393 | pass | 0.393 | 24% brighter, cooler, worst error center |
| robot camB | opengl | 0.377 | pass | 0.377 | 4% brighter, warmer, worst error mid-right |
| teapot camA | opengl | 0.401 | pass | 0.401 | 1% darker, worst error mid-left |
| teapot camB | opengl | 0.426 | pass | 0.426 | 5% darker, warmer, worst error mid-left |
| toy_drummer camA | opengl | 0.424 | pass | 0.424 | 10% brighter, cooler, worst error upper-center |
| toy_drummer camB | opengl | 0.432 | pass | 0.432 | 3% brighter, cooler, worst error center |

## warehouse

| pair | backend | FLIP | status | baseline | diagnosis |
|---|---|---|---|---|---|
| warehouse camA | opengl | 0.820 | pass | 0.820 | 7% darker, warmer, worst error center |
| warehouse camB | opengl | 0.866 | pass | 0.866 | 7% brighter, worst error center |

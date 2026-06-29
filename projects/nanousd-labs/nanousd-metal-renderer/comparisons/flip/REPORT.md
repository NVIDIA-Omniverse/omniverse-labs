# Metal vs OVRTX — ϜLIP diff report

Each Metal backend output is compared to its OVRTX golden with the calibrated FLIP metric. Studio scenes: subject composited onto the golden background and foreground-masked, scored by mean FLIP. Full scenes: 95th-percentile FLIP (the mean is gameable — a tone-matched-but-murky render can beat a good one), plus orthogonal GMSD (structure), CIEDE2000 (exposure-suppressed colour), crushed-black and exposure gate legs. **Lower = closer to the golden.** Single-pass backends sit well above the path-traced golden even when they look right — read the *relative* ordering and the heatmaps, and judge full scenes by eye, not the scalar alone.

**Systematic finding:** vs the OVRTX golden, 7 Metal renders are brighter and 25 darker (single-pass lacks path-traced occlusion). Largest perceptual gap: `warehouse/warehouse_camB/metal_raster` at FLIP 0.911 (50% brighter, cooler, worst error upper-center).

**Regression gate: ✅ PASS**  (tol +0.03; ceilings studio 0.95 / fullscene 0.99; +GMSD/crush/exposure legs on full scenes; 0 new)

## Largest diffs from golden

| pair | FLIP | GMSD | ΔE2000 | mode | diagnosis |
|---|---|---|---|---|---|
| `warehouse/warehouse_camB/metal_raster` | 0.911 | 0.232 | 12.66 | fullscene | 50% brighter, cooler, worst error upper-center |
| `warehouse/warehouse_camA/metal_raster` | 0.826 | 0.202 | 9.52 | fullscene | 7% brighter, cooler, worst error upper-left |
| `warehouse/warehouse_camB/metal_rt` | 0.781 | 0.195 | 7.61 | fullscene | 24% brighter, cooler, worst error upper-left |
| `warehouse/warehouse_camA/metal_rt` | 0.742 | 0.176 | 7.30 | fullscene | 18% brighter, warmer, worst error upper-right |
| `apple/teapot_camB/metal_raster` | 0.509 | 0.109 | 9.96 | studio | 10% darker, warmer, worst error mid-left |

## chess

| pair | backend | FLIP | status | baseline | diagnosis |
|---|---|---|---|---|---|
| chess_set camA | metal_raster | 0.434 | pass | 0.434 | 23% darker, warmer, worst error mid-right |
| chess_set camA | metal_rt | 0.448 | pass | 0.447 | 23% darker, warmer, worst error mid-right |
| chess_set camB | metal_raster | 0.242 | pass | 0.242 | 9% brighter, worst error lower-left |
| chess_set camB | metal_rt | 0.271 | pass | 0.271 | 1% darker, worst error lower-left |

## apple

| pair | backend | FLIP | status | baseline | diagnosis |
|---|---|---|---|---|---|
| ball_soccerball camA | metal_raster | 0.224 | pass | 0.224 | 6% darker, worst error upper-center |
| ball_soccerball camA | metal_rt | 0.238 | pass | 0.240 | 6% darker, warmer, worst error upper-center |
| ball_soccerball camB | metal_raster | 0.295 | pass | 0.295 | 0% darker, warmer, worst error lower-left |
| ball_soccerball camB | metal_rt | 0.284 | pass | 0.286 | 2% darker, warmer, worst error lower-left |
| fender_stratocaster camA | metal_raster | 0.300 | pass | 0.300 | 17% darker, warmer, worst error upper-center |
| fender_stratocaster camA | metal_rt | 0.303 | pass | 0.303 | 17% darker, warmer, worst error lower-center |
| fender_stratocaster camB | metal_raster | 0.358 | pass | 0.358 | 23% darker, warmer, worst error upper-center |
| fender_stratocaster camB | metal_rt | 0.367 | pass | 0.368 | 24% darker, warmer, worst error upper-center |
| pancakes camA | metal_raster | 0.378 | pass | 0.378 | 5% darker, worst error mid-left |
| pancakes camA | metal_rt | 0.302 | pass | 0.301 | 1% darker, warmer, worst error mid-left |
| pancakes camB | metal_raster | 0.400 | pass | 0.400 | 9% brighter, warmer, worst error center |
| pancakes camB | metal_rt | 0.382 | pass | 0.382 | 6% brighter, warmer, worst error center |
| robot camA | metal_raster | 0.236 | pass | 0.236 | 8% darker, worst error upper-center |
| robot camA | metal_rt | 0.218 | pass | 0.218 | 16% darker, warmer, worst error upper-center |
| robot camB | metal_raster | 0.303 | pass | 0.303 | 15% darker, warmer, worst error mid-right |
| robot camB | metal_rt | 0.278 | pass | 0.278 | 20% darker, warmer, worst error mid-right |
| teapot camA | metal_raster | 0.402 | pass | 0.402 | 11% darker, warmer, worst error mid-left |
| teapot camA | metal_rt | 0.316 | pass | 0.316 | 12% darker, warmer, worst error mid-left |
| teapot camB | metal_raster | 0.509 | pass | 0.509 | 10% darker, warmer, worst error mid-left |
| teapot camB | metal_rt | 0.392 | pass | 0.392 | 12% darker, warmer, worst error mid-left |
| toy_drummer camA | metal_raster | 0.368 | pass | 0.368 | 17% darker, worst error upper-center |
| toy_drummer camA | metal_rt | 0.331 | pass | 0.331 | 18% darker, warmer, worst error upper-center |
| toy_drummer camB | metal_raster | 0.352 | pass | 0.352 | 17% darker, warmer, worst error center |
| toy_drummer camB | metal_rt | 0.359 | pass | 0.359 | 22% darker, warmer, worst error upper-center |

## warehouse

| pair | backend | FLIP | status | baseline | diagnosis |
|---|---|---|---|---|---|
| warehouse camA | metal_raster | 0.826 | pass | 0.826 | 7% brighter, cooler, worst error upper-left |
| warehouse camA | metal_rt | 0.742 | pass | 0.742 | 18% brighter, warmer, worst error upper-right |
| warehouse camB | metal_raster | 0.911 | pass | 0.911 | 50% brighter, cooler, worst error upper-center |
| warehouse camB | metal_rt | 0.781 | pass | 0.781 | 24% brighter, cooler, worst error upper-left |

# Vulkan vs OVRTX — ϜLIP diff report

Each Vulkan backend output is compared to its OVRTX golden with the calibrated FLIP metric. Studio scenes: subject composited onto the golden background and foreground-masked, scored by mean FLIP. Full scenes: 95th-percentile FLIP (the mean is gameable — a tone-matched-but-murky render can beat a good one), plus orthogonal GMSD (structure), CIEDE2000 (exposure-suppressed colour), crushed-black and exposure gate legs. **Lower = closer to the golden.** Single-pass backends sit well above the path-traced golden even when they look right — read the *relative* ordering and the heatmaps, and judge full scenes by eye, not the scalar alone.

**Systematic finding:** vs the OVRTX golden, 10 Vulkan renders are brighter and 22 darker (single-pass lacks path-traced occlusion). Largest perceptual gap: `warehouse/warehouse_camB/vk_rt` at FLIP 0.961 (6% brighter, cooler, worst error center).

**Regression gate: ✅ PASS**  (tol +0.03; ceilings studio 0.95 / fullscene 0.99; +GMSD/crush/exposure legs on full scenes; 0 new)

## Largest diffs from golden

| pair | FLIP | GMSD | ΔE2000 | mode | diagnosis |
|---|---|---|---|---|---|
| `warehouse/warehouse_camB/vk_rt` | 0.961 | 0.242 | 9.94 | fullscene | 6% brighter, cooler, worst error center |
| `warehouse/warehouse_camA/vk_rt` | 0.924 | 0.213 | 7.25 | fullscene | 7% darker, cooler, worst error mid-right |
| `apple/teapot_camA/vk_rt` | 0.885 | 0.128 | 21.66 | studio | 62% darker, cooler, worst error center |
| `apple/teapot_camB/vk_rt` | 0.859 | 0.125 | 21.62 | studio | 62% darker, cooler, worst error mid-left |
| `apple/pancakes_camA/vk_rt` | 0.826 | 0.129 | 21.04 | studio | 65% darker, cooler, worst error lower-center |

## chess

| pair | backend | FLIP | status | baseline | diagnosis |
|---|---|---|---|---|---|
| chess_set camA | vk_raster | 0.468 | pass | 0.468 | 6% darker, cooler, worst error mid-left |
| chess_set camA | vk_rt | 0.657 | pass | 0.657 | 66% darker, worst error mid-right |
| chess_set camB | vk_raster | 0.340 | pass | 0.340 | 23% brighter, cooler, worst error center |
| chess_set camB | vk_rt | 0.517 | pass | 0.517 | 54% darker, cooler, worst error center |

## apple

| pair | backend | FLIP | status | baseline | diagnosis |
|---|---|---|---|---|---|
| ball_soccerball camA | vk_raster | 0.313 | pass | 0.313 | 5% brighter, cooler, worst error mid-left |
| ball_soccerball camA | vk_rt | 0.517 | pass | 0.517 | 35% darker, cooler, worst error lower-left |
| ball_soccerball camB | vk_raster | 0.314 | pass | 0.314 | 0% darker, cooler, worst error mid-left |
| ball_soccerball camB | vk_rt | 0.594 | pass | 0.594 | 42% darker, cooler, worst error lower-right |
| fender_stratocaster camA | vk_raster | 0.423 | pass | 0.423 | 5% brighter, cooler, worst error upper-center |
| fender_stratocaster camA | vk_rt | 0.739 | pass | 0.739 | 61% darker, cooler, worst error upper-center |
| fender_stratocaster camB | vk_raster | 0.448 | pass | 0.448 | 5% darker, cooler, worst error upper-center |
| fender_stratocaster camB | vk_rt | 0.751 | pass | 0.751 | 64% darker, cooler, worst error upper-center |
| pancakes camA | vk_raster | 0.558 | pass | 0.558 | 8% darker, cooler, worst error mid-left |
| pancakes camA | vk_rt | 0.826 | pass | 0.826 | 65% darker, cooler, worst error lower-center |
| pancakes camB | vk_raster | 0.466 | pass | 0.466 | 2% darker, cooler, worst error lower-right |
| pancakes camB | vk_rt | 0.746 | pass | 0.746 | 59% darker, cooler, worst error lower-right |
| robot camA | vk_raster | 0.399 | pass | 0.399 | 32% brighter, cooler, worst error mid-left |
| robot camA | vk_rt | 0.566 | pass | 0.566 | 63% darker, cooler, worst error upper-center |
| robot camB | vk_raster | 0.382 | pass | 0.382 | 13% brighter, cooler, worst error mid-right |
| robot camB | vk_rt | 0.618 | pass | 0.618 | 66% darker, cooler, worst error upper-center |
| teapot camA | vk_raster | 0.473 | pass | 0.473 | 17% darker, cooler, worst error mid-left |
| teapot camA | vk_rt | 0.885 | pass | 0.885 | 62% darker, cooler, worst error center |
| teapot camB | vk_raster | 0.473 | pass | 0.473 | 22% darker, cooler, worst error mid-left |
| teapot camB | vk_rt | 0.859 | pass | 0.859 | 62% darker, cooler, worst error mid-left |
| toy_drummer camA | vk_raster | 0.460 | pass | 0.460 | 8% brighter, cooler, worst error upper-center |
| toy_drummer camA | vk_rt | 0.720 | pass | 0.720 | 63% darker, cooler, worst error upper-center |
| toy_drummer camB | vk_raster | 0.436 | pass | 0.436 | 6% brighter, cooler, worst error center |
| toy_drummer camB | vk_rt | 0.691 | pass | 0.691 | 63% darker, worst error center |

## warehouse

| pair | backend | FLIP | status | baseline | diagnosis |
|---|---|---|---|---|---|
| warehouse camA | vk_raster | 0.768 | pass | 0.768 | 4% brighter, worst error center |
| warehouse camA | vk_rt | 0.924 | pass | 0.924 | 7% darker, cooler, worst error mid-right |
| warehouse camB | vk_raster | 0.811 | pass | 0.811 | 23% brighter, cooler, worst error center |
| warehouse camB | vk_rt | 0.961 | pass | 0.961 | 6% brighter, cooler, worst error center |

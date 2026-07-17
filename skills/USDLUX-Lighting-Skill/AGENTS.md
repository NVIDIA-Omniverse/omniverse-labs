# AGENTS.md — omniverse-usdlux-lighting-skill

> Read this file fully before starting any task in this repo.
> It is written for AI agents (Claude Code, Cursor, Codex, and similar).

---

## What This Repo Is

An agentic skill for creating and upgrading physically accurate lighting in USD scenes
using the UsdLux 2505 specification in NVIDIA Omniverse 109.0+.

The skill enables agents to set up physically correct lighting on any USD scene
programmatically, without requiring Kit UI interaction.

---

## What the Skill Does

Given a USD scene, the skill:

1. Scans geometry to find light fixture prims — by name keywords, vertex clustering
   on merged meshes, and PointInstancer traversal
2. Calculates physically correct lumen output per fixture using the Lumen Method
   (Room Index → UF → reflectance scale → scene complexity factor)
3. Samples surface reflectances from materials or displayColor to refine the UF
4. Creates UsdLux lights (RectLight, SphereLight, CylinderLight) at each fixture
   position with physical schemas applied from `omni.usd.schema.physicallight`
5. Deactivates existing legacy lights (new lights at fixture positions get UsdLux 2505;
   existing DomeLights are upgraded in place when windows are detected)
6. Detects glass/window geometry and creates or upgrades a DomeLight
7. Falls back to a ceiling grid if no fixture geometry is found
8. Warns when per-fixture lumens exceed realistic indoor lighting ranges
9. Writes everything to a non-destructive sublayer — never touches the original scene
10. Supports one-step calibration via `measured_lux`: scales all lights to a measured
    PT AOV Illuminance reading

The entry point for executing the skill is `usdlux-lighting-skill.md`.
The canonical implementation is `run_lighting_skill.py`.

---

## Key Domain Knowledge

### UsdLux Versions
- `2411` — legacy (default for older scenes)
- `2505` — current physical behavior target (Kit 109.0+)

Upgrade all lights at once (Omniverse Script Editor):
```python
omni.kit.commands.execute("UpgradeUsdLuxLights")
```

> **Note:** 2505 is current as of 2026. Verify the latest version in the Omniverse
> release notes before authoring new lights.

### Physical Light Schemas (`omni.usd.schema.physicallight`)

This Kit extension registers the physical light schemas. It must be loaded for
`AddAppliedSchema()` to work. The script handles enabling it automatically when
running inside Kit.

Key schemas and their primary attributes:

| Schema | Apply to | Key attribute |
|--------|----------|--------------|
| `PhotometricAreaLightAPI` | Rect/Sphere/Cylinder/Disc lights | `photometric:power` (lumens) |
| `PhotometricDomeLightAPI` | Dome lights | `photometric:illuminance` (lux) |
| `PhotometricDistantLightAPI` | Distant lights | `photometric:illuminance` (lux) |
| `PhysicalLightIlluminantAPI` | All light types | `physical:illuminant`, `physical:colorTemperature` |

### intensity / exposure behavior
- `intensity` is a multiplier on Power — keep at `1.0` for physical accuracy
- `exposure` is photographic stops — keep at `0.0` for physical accuracy
- After applying physical schemas, Power is the authoritative brightness control

### Lumen Method
```
Power_per_fixture = (target_lux × floor_area) / (N × UF × MF) × complexity_factor

Room Index  = (L × W) / (H_mount × (L + W))
UF          = 0.35 – 0.65 based on RI, then scaled by surface reflectances
MF          = 0.8 (maintenance factor)
complexity  = empirical RTX occlusion correction (1.10 – 1.75 by scene type + RI)
```

### Realistic lumen ranges
| Fixture type | Lumens |
|-------------|--------|
| Office panel / troffer | 3,000 – 6,000 lm |
| Industrial high-bay LED | 15,000 – 40,000 lm |
| Maximum indoor LED (specialty) | ~100,000 lm |
| Above 100,000 lm | Unrealistic for indoor — skill prints a warning |

### Calibration workflow
1. Run the skill → lights are written to the sublayer
2. Open scene in Omniverse → switch to **RTX Path Tracing**
3. **Set camera tone mapping first** — the script prints a `CAMERA EXPOSURE CHECK` block
   at the end of every real (non-dry-run) run with recommended Film ISO, F-stop, and
   Exposure Time. Apply these in **Render Settings → Post Processing → Tone Mapping**
   before using Debug View. Without this, the scene will look too dark or too bright
   even though the lights are physically correct.
4. **Debug View → PT AOV Illuminance** — click floor surfaces to read lux
5. If the reading is off target, re-run with `measured_lux=<reading>` to scale all lights
   to the exact target in one step

---

## Execution Environment

The skill runs in any environment with Kit Python available:

- **Kit Script Editor (Omniverse Composer)** — use the `run_skill()` function directly;
  set the KIT CONFIG block variables at the bottom of `run_lighting_skill.py`
- **Kit headless (shell/terminal)** — `kit --exec run_lighting_skill.py --no-window -- scene.usd`
- **Standalone Python with usd-core** — supports dry-run and analysis; writing physical
  schemas requires Kit (schemas not registered outside Kit's extension system)

See `usdlux-lighting-skill.md` Pre-flight section for step-by-step paths.

---

## Repo Structure

```
README.md                        # root overview (points here)
skills/
  USDLUX-Lighting-Skill/
    AGENTS.md                    # this file — read first
    README.md                    # human-facing overview
    usdlux-lighting-skill.md     # THE SKILL — agent execution guide + embedded script
    run_lighting_skill.py        # canonical implementation (v4.3)
    PhysLighting_BeforeAfter.png # before/after render comparison
```

---

## Related Resources

- [NVIDIA Omniverse Lighting documentation](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/lighting.html)
- [UsdLux specification](https://openusd.org/release/api/usd_lux_page_front.html)
- Omniverse Kit 109.0+ required for `omni.usd.schema.physicallight`

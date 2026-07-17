# UsdLux Physical Lighting Skill

> Agent skill for NVIDIA Omniverse Kit 109.0+.
> Upgrades legacy USD scene lighting to physically accurate UsdLux 2505.
>
> Source: NVIDIA Omniverse Materials and Rendering documentation
> https://docs.omniverse.nvidia.com/materials-and-rendering/latest/lighting.html#physical-lighting

---

## HOW THIS SKILL WORKS

This skill has two parts:

1. **Execution section (below)** — the agent follows this. Pre-flight, decision logic, run the script, interpret output, validate. No physics knowledge required here.
2. **Reference section** — 12 steps of domain knowledge. The agent reads these only when it needs to understand *why* something is happening, debug a result, or explain something to the user.

The Python script (`run_lighting_skill.py`) does all the actual work. The execution section directs the agent to run it correctly. Do not duplicate what the script already does — just follow the steps.

---

## EXECUTION

### Pre-flight Checklist

Work through these in order. Do not proceed until all are resolved.

**1. Do you have a scene path?**
- User provided one → use it.
- Not provided → ask: *"Which USD scene should I apply the lighting skill to? I need a file path (.usd, .usda, .usdc, or .usdz)."*
- Do NOT proceed without a scene.

**2. Which execution context are you in?**

Both paths require Kit. The distinction is *how the script is invoked* — not whether the machine is local, a VM, or a cloud node. A VM with Kit's UI open and its display streamed to your screen is still a Script Editor context.

Mechanical test: if you can successfully run `import omni.usd` inside a running Kit Python context (e.g., you are already in the Script Editor console), use the Script Editor path. If you are invoking from a shell, use the Kit Terminal path.

| Context | How to tell | Path to take |
|--------|------------|-------------|
| Kit Script Editor (Kit UI is open — local, VM, or streamed desktop) | `import omni.usd` works in the current Python environment and a stage can be opened interactively | → **Script Editor path** |
| Kit Terminal (invoking from a shell) | You have shell/SSH access to a machine with Kit installed | → **Kit Terminal path** |

> **Note:** Plain Python with `usd-core` (pip) is not needed here. The NVIDIA physical light schemas (`PhotometricAreaLightAPI`, etc.) require Kit's extension system, which `usd-core` doesn't include. Use Kit Python in both paths below.
>
> **Streaming VMs:** If Kit is running on a remote VM and the display is streamed to your local machine (RDP, Parsec, Omniverse Streaming), choose the path based on how you interact with Kit — Script Editor if you are working in the UI, Kit Terminal if you have a shell session on that VM.

**3. Write `run_lighting_skill.py` to disk.**

The full script is embedded at the end of this skill file under **EMBEDDED SCRIPT**. Derive the target path from the scene location:

```python
import os

scene_path = "/path/to/your/scene.usd"          # ← from pre-flight step 1
script_path = os.path.join(
    os.path.dirname(os.path.abspath(scene_path)),
    "run_lighting_skill.py"
)

with open(script_path, "w") as f:
    f.write(EMBEDDED_SCRIPT_CONTENT)             # ← content from EMBEDDED SCRIPT section

print(f"Script written to: {script_path}")
```

If the scene directory is not writable (e.g., read-only NFS mount), write to a temp directory instead and use that path in the steps below:

```python
import tempfile, os
script_path = os.path.join(tempfile.gettempdir(), "run_lighting_skill.py")
```

Then use `script_path` in the steps below.

---

### Script Editor Path (inside Omniverse Composer)

Use this when running interactively inside Omniverse Composer or Kit with a UI.

**Step K1 — Open the scene in Composer first.**
The script works on the stage that is already open. Do not open a second stage inside the script.

**Step K2 — Check required extensions.**
Run this once in the Script Editor before anything else:

```python
import omni.kit.app
mgr = omni.kit.app.get_app().get_extension_manager()
for ext in ["omni.usd.schema.physicallight", "omni.usd.schema.omniverse"]:
    if not mgr.is_extension_enabled(ext):
        mgr.set_extension_enabled_immediate(ext, True)
        print(f"Enabled: {ext}")
    else:
        print(f"Already enabled: {ext}")
```

If either extension fails to enable, stop and tell the user which extension is missing and what Kit version they are on.

**Step K3 — Configure and run the skill script.**
Load the script into Kit's Python context with `exec`. Paste this into the Script Editor console:

```python
exec(open(script_path).read())   # script_path from pre-flight step 3
```

This is equivalent to opening the file in Script Editor's file browser but requires no UI navigation. To override defaults, edit the KIT CONFIG block in the written `.py` file before running. The block is at the bottom of the script:

```python
# ── KIT CONFIG — edit these before hitting Run ───────────────
_DRY_RUN          = True    # Set False to write files
_COLOR_TEMP       = 4000.0  # Kelvin, fixture lights
_DOME_COLOR_TEMP  = 6500.0  # Kelvin, dome light
_DOME_ILLUMINANCE = 0.0     # lux, dome light (0 = auto: 400 lux sunset/clear sky)
_FORCE_DOME       = False   # True = always create DomeLight
_NO_DOME          = False   # True = never create DomeLight
_TARGET_LUX       = None    # None = auto-detect from scene name
_MEASURED_LUX     = None    # Set to PT AOV Illuminance reading to calibrate (e.g. 220.0)
_REFLECTANCES     = None    # Set to (ceil, wall, floor) tuple to override sampling (e.g. (0.7, 0.5, 0.2))
_OUTPUT_FILE      = None    # Set to a file path to save log output
# ─────────────────────────────────────────────────────────────
```

Run with `_DRY_RUN = True` first. Read the dry-run output (see Interpret Output below). If it looks correct, set `_DRY_RUN = False` and run again.

---

### Kit Terminal Path (Shell — Local, VM, or Horde)

Use this when invoking from a shell. The machine can be a local workstation, any virtual machine (with or without a streaming display), or a cloud node like Horde DGXC. Kit runs as a full application — same extension system and commands as Script Editor, just no interactive UI.

**Step H1 — Verify Kit is installed and locate the binary.**

**If Isaac Sim is installed or available (e.g., as a container on a Horde VM), use it — Isaac Sim ships with a full Kit including all required extensions.** Check first:

```bash
# Isaac Sim — container (most common on Horde/cloud VMs):
/isaac-sim/kit/kit --version

# Isaac Sim — local install:
~/.local/share/ov/pkg/isaac-sim-*/kit/kit --version
```

Other Kit sources if Isaac Sim is not available:

```bash
~/.local/share/ov/pkg/code-*/kit/kit --version          # Omniverse Code (Linux)
~/kit-sdk/kit --version                                  # Kit SDK (Linux)

# Windows:
"C:\Users\<user>\AppData\Local\ov\pkg\code-2023.2.5\kit\kit.exe" --version
```

If the path is still not known, search for it:

```bash
# Linux:
find / -name 'kit' -type f 2>/dev/null | grep -v proc | grep -v snap | head -10

# Windows (PowerShell):
Get-ChildItem -Path C:\ -Recurse -Filter "kit.exe" -ErrorAction SilentlyContinue | Select-Object FullName
```

If `kit` is not found at all, Kit is not installed. Install Isaac Sim, Omniverse Code, or the Kit SDK on the machine first.

**Step H2 — Dry-run first.**

```bash
# Linux / Horde VM:
/path/to/kit/kit --exec run_lighting_skill.py --no-window -- <scene.usd> --dry-run

# Windows local:
"C:\...\kit\kit.exe" --exec run_lighting_skill.py --no-window -- <scene.usd> --dry-run
```

**Step H3 — Execute.**

```bash
/path/to/kit/kit --exec run_lighting_skill.py --no-window -- <scene.usd>
```

Optional flags (same for both paths):
- `--color-temp 4000` — fixture color temperature in Kelvin
- `--dome-color-temp 6500` — dome light color temperature
- `--dome-illuminance 5000` — dome light illuminance in lux
- `--target-lux 300` — override auto-detected lux target
- `--force-dome` — create DomeLight even without windows detected
- `--no-dome` — skip DomeLight even if windows detected
- `--output-layer lighting.usd` — custom sublayer filename

`kit --exec` starts Kit as a full headless application. The extension manager initializes, `omni.usd.schema.physicallight` loads, and `UpgradeUsdLuxLights` is available — identical to the Script Editor path.

---

### Interpret Output

After the dry-run, check these in the output before executing:

| Output line | What to verify |
|-------------|---------------|
| `Scene bounds (m): X × Y × Z` | Bounds should be plausible real-world size (e.g. 50m warehouse). If < 0.5m, unit mismatch — see **Scale fix procedure** below. |
| `Fixture positions detected: N` | If 0, no fixture geometry was found — see **Zero fixtures procedure** below. |
| `Existing lights deactivated: N` | `0` is fine — scene had no legacy lights. `> 0` means legacy lights were found and hidden (not deleted). If deactivated >> fixture count (e.g., 2000 deactivated, 24 created), that is expected: lights without fixture geometry are not recreated by design. Tell the user how many were deactivated vs. recreated so they are not surprised. |
| `Robot/vehicle excluded: N` | Lights on robots/forklifts — correct that these are skipped. |
| `Power range: X lm` | Cross-check against Reference Step 8 table. A warehouse fixture should be 15,000–60,000 lm for high-bay LEDs. |
| `DomeLight: Yes/No` | Yes = windows detected or force_dome. No = interior-only scene. |
| `unitsResolve detected` | Normal — geometry authored in cm but stage is in meters. Clustering works in local space and handles this correctly. |

If anything else looks wrong, check the Reference section for the relevant step and adjust `--target-lux` or scene geometry before executing.

---

**Zero fixtures procedure:**
If `Fixture positions detected: 0`, do not proceed to execute. Instead:
1. Traverse the stage and print all mesh prim names:
   ```python
   from pxr import Usd
   stage = Usd.Stage.Open(scene_path)
   meshes = [str(p.GetPath()) for p in stage.Traverse() if p.GetTypeName() == "Mesh"]
   print("\n".join(meshes))
   ```
2. Compare names against the keyword list in Reference Step 10b.
3. If mesh names are close to a keyword (e.g., contain `"ceiling_unit"` or `"lamp_housing"`), re-run with `--extra-keyword <term>` if the flag exists, or ask the user if those prims represent light fixtures. If yes, ask them to rename to include a recognized keyword (e.g., add `"fixture"` or `"luminaire"` to the prim name).
4. If still 0 after renaming, the geometry may be inside a PointInstancer or a deeply nested reference. Ask the user to identify which prims represent light fixtures so you can update the keyword list in Reference Step 10b.

---

**Scale fix procedure:**
If `Scene bounds < 0.5m`, the stage has a unit mismatch. Check Reference Step 0 to confirm which case applies, then fix programmatically:

```python
from pxr import Usd
stage = Usd.Stage.Open(scene_path)
# Most common case: geometry authored in cm, metersPerUnit incorrectly set to 1.0
stage.SetMetadata("metersPerUnit", 0.01)
stage.GetRootLayer().Save()
print("metersPerUnit set to 0.01 — re-run the dry-run to verify bounds.")
```

After saving, re-run the dry-run. Bounds should now read in plausible meters.

---

### Validate in Compositor

After running (not dry-run):

1. Open the scene in Omniverse Composer if not already open.
2. Switch renderer: **RTX - Interactive (Path Tracing)**
3. **Render Settings → Post Processing → Tone Mapping** — set camera exposure to match scene lux (see camera recommendations in script output).
4. **Debug View → PT AOV Illuminance** — enable and select in viewport header.
5. Click floor surfaces — read lux values in the **Illuminance Value** field.
6. Compare to target: warehouse = 200–300 lux, office = 300–500 lux, retail = 500–1000 lux.

If the scene looks dark but lux reads correct → camera exposure is wrong, not the lights. Follow the camera recommendations in the script output.
If lux reads too low → re-run with `--target-lux <higher_value>` or check fixture count.

**If you ran via Kit Terminal (no Composer open):** Steps 1–6 require a viewport. Either:
- Open the output scene (`physical_lighting_setup.usd` sublayered under your scene) in Omniverse Composer on any machine and follow steps 1–6 above.
- Or use ovrtx to render a validation frame with the PT AOV Illuminance output and read lux values from the rendered image.

Tell the user the scene is ready and share the output sublayer path.

---

### What the Script Does (Summary)

The agent does not need to re-implement any of this. The script handles it.

1. **Scale validation** — checks world-space bounds, halts if scene is implausibly small.
2. **Sublayer setup** — creates `physical_lighting_setup.usd` as a sublayer. Never modifies the original scene.
3. **Fixture detection** — finds fixture meshes by name keywords. Handles merged meshes (vertex clustering), single meshes, and PointInstancers. Deduplicates sub-mesh positions so one fixture = one light.
4. **Glass/window detection** — finds building windows → creates or upgrades DomeLight. Filters out prop glass (forklift windshields, extinguisher glass, lamp covers) by name context and minimum area (0.5 m²).
5. **Power calculation** — Lumen Method with Room Index. Splits ceiling vs wall fixtures and assigns different power to each group.
6. **Deactivate existing lights** — all existing architectural lights are deactivated and hidden. Robot/vehicle lights are excluded. If the scene has no existing lights, this step is a no-op — the script continues normally.
7. **Create new lights** — SphereLights, RectLights, or CylinderLights at fixture positions, with physical schemas applied.
8. **DomeLight** — upgrades existing DomeLight if found; creates new one if windows detected and none exists; deactivates existing if no windows.
9. **Camera diagnostics** — reads current tone mapping settings and recommends correct exposure for the target lux.
10. **Save** — standalone: saves to disk. Kit: uses `omni.usd.get_context().save_stage()` to avoid Composer fetch-changes dialogs.

---

---

## REFERENCE

The agent reads these steps when it needs to understand or explain domain concepts. These are not execution steps — the script handles execution.

---

## Step 0 — Validate Scene Scale

Before adding or editing any lights, check that the scene's unit metadata and geometry bounds are consistent. A scale mismatch means any light values you author will be physically wrong.

### What to check

1. `metersPerUnit` — the stage metadata that defines how many meters one USD unit represents.
2. Real-world bounds — the actual geometry size converted to meters.
3. Light dimensions — width/height/radius of existing lights relative to scene size.
4. `photometric:illuminance:distance` on any existing lights — must be plausible in meters.

### Scale mismatch — how to fix

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| Scene is 100× too small | Geometry authored in cm, `metersPerUnit` = 1.0 | Set `metersPerUnit = 0.01` and remove any compensating `xformOp:scale:unitsResolve` |
| Scene is 100× too large | Geometry authored in m, `metersPerUnit` = 0.01 | Set `metersPerUnit = 1.0` |
| Light dimensions huge relative to scene | Light inherited wrong default | Rescale light width/height/radius to match scene |

The `xformOp:scale:unitsResolve` pattern is normal in Omniverse — geometry authored in cm while the stage declares `metersPerUnit = 1.0`. World-space bounds are correct. Local vertex data is in the original unit. The script handles this automatically.

---

## Step 1 — Check the UsdLux Version

Every light prim has `omni:rtx:usdluxVersion`:

- `2411` — legacy behavior (default for older scenes)
- `2505` — current physical behavior (Kit 109.0+)

Read the version on each light:

```python
from pxr import Usd, UsdLux

stage = Usd.Stage.Open("path/to/scene.usd")
for prim in stage.Traverse():
    if prim.HasAPI(UsdLux.LightAPI):
        attr = prim.GetAttribute("omni:rtx:usdluxVersion")
        version = attr.Get() if attr.IsValid() else None
        print(f"{prim.GetPath()} → version: {version}")
```

---

## Step 2 — Upgrade to UsdLux 2505

### Upgrade all lights at once (Kit)

```python
import omni.kit.commands

# Test in session layer first — does not modify the saved file
omni.kit.commands.execute("UpgradeUsdLuxLights", use_session_layer=True)

# If result looks correct, upgrade in root layer
omni.kit.commands.execute("UpgradeUsdLuxLights", use_session_layer=False)
```

Always make a backup or use version control before running the root-layer upgrade.

### Upgrade a single light (standalone pxr)

```python
prim.GetAttribute("omni:rtx:usdluxVersion").Set(2505)
stage.Save()
```

### Significant behavior changes in 2505

- **Dome lights:** Default orientation changed. The default camera now looks at the horizon instead of the "bottom" of the HDRI image.
- **Distant lights:** RTX now respects `bool inputs:normalize`.
- **Area lights:** `inputs:shaping:cone:softness` mapping changed. RTX now respects `inputs:shaping:ies:angleScale`.

---

## Step 3 — Enable Required Extensions (Kit only)

The physical light schemas (`PhotometricAreaLightAPI`, `PhotometricDomeLightAPI`) are NVIDIA-specific schemas registered by the `omni.usd.schema.physicallight` extension. Without it loaded, `AddAppliedSchema()` silently fails.

```python
import omni.kit.app
mgr = omni.kit.app.get_app().get_extension_manager()
for ext in ["omni.usd.schema.physicallight", "omni.usd.schema.omniverse"]:
    if not mgr.is_extension_enabled(ext):
        mgr.set_extension_enabled_immediate(ext, True)
```

To enable manually: **Developer → Extensions** → search `omni.usd.schema.physicallight` → Install → enable **AUTOLOAD**.

UI schema editing also requires `omni.kit.widget.schema_api` — install and enable via Extensions.

---

## Step 4 — Physical Schemas Overview

| Schema | Apply to | Purpose |
|--------|----------|---------|
| `PhotometricAreaLightAPI` | Area lights (Rect, Sphere, Disc, Cylinder) | Brightness in lumens or lux |
| `PhotometricDomeLightAPI` | Dome lights | Brightness as upward illuminance in lux |
| `PhotometricDistantLightAPI` | Distant lights | Brightness as perpendicular illuminance in lux |
| `PhysicalLightIlluminantAPI` | All light types | Color spectrum and color temperature |
| `ShapingAPI` | Area lights | Angular distribution, cone control, IES profiles |

> Applying `PhotometricAreaLightAPI` automatically also applies `PhysicalLightIlluminantAPI`.

---

## Step 5 — Schema Attribute Reference

### PhotometricAreaLightAPI

| Attribute | Type | Default | Unit | Description |
|-----------|------|---------|------|-------------|
| `photometric:power` | float | 1600.0 | lumens (lm) | Total photometric power |
| `photometric:illuminance` | float | 0.0 | lux (lx) | Illuminance at distance. Only active when distance > 0. |
| `photometric:illuminance:distance` | float | 0.0 | meters (m) | Set to 0 to use Power mode. |

Use either **Power** or **Illuminance + Distance** — not both. When distance > 0, Power is ignored.

### PhotometricDomeLightAPI

| Attribute | Type | Default | Unit | Description |
|-----------|------|---------|------|-------------|
| `photometric:illuminance` | float | 10000.0 | lux | Illuminance on upward-facing surface |

### PhotometricDistantLightAPI

| Attribute | Type | Default | Unit | Description |
|-----------|------|---------|------|-------------|
| `photometric:illuminance` | float | 10000.0 | lux | Illuminance on surface perpendicular to light direction |

### PhysicalLightIlluminantAPI

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `physical:illuminant` | token | `white` | `white`, `blackbody`, `illuminantD`, `custom` |
| `physical:colorTemperature` | float | — | Kelvin — used with `blackbody` and `illuminantD` |

`illuminantD` = CIE D-series daylight (e.g. D65 = 6504K). Use for LED and fluorescent sources.
`blackbody` = thermal emitter. Use for incandescent and halogen sources.

### ShapingAPI

| Attribute | Type | Range | Description |
|-----------|------|-------|-------------|
| `inputs:shaping:cone:angle` | float | 0–180° | Half-angle of spotlight cone |
| `inputs:shaping:cone:softness` | float | 0–1 | Penumbra softness |
| `inputs:shaping:ies:file` | asset | — | Path to IES photometric profile |
| `inputs:shaping:ies:angleScale` | float | — | Expand/squash IES distribution |

---

## Step 6 — Handling Legacy intensity

### Why you cannot convert legacy intensity to lumens

In legacy UsdLux (2411), `inputs:intensity` is a dimensionless renderer multiplier with no physical unit. There is no reliable formula to convert it to lumens.

**Do not attempt `photometric:power = intensity × some_factor`.**

### Correct upgrade sequence

1. Run `UpgradeUsdLuxLights` (Step 2). This remaps legacy intensity under 2505 rules to preserve the existing visual look.
2. Apply `PhotometricAreaLightAPI`.
3. Set `photometric:power` to a physically realistic value from the Lumen Method (Step 8) — NOT derived from old intensity.
4. Reset `intensity = 1.0` and `exposure = 0.0` so Power is authoritative.
5. Validate with Debug View (Step 12).

After physical schemas are applied, `intensity` and `exposure` still act as multipliers on top of Power. Keep both at defaults (1.0 / 0.0) unless intentionally scaling.

---

## Step 7 — Apply Schemas via Python

```python
from pxr import Usd, UsdLux

stage = Usd.Stage.Open("path/to/scene.usd")
prim = stage.GetPrimAtPath("/World/Lights/RectLight")

prim.ApplyAPI("PhotometricAreaLightAPI")  # also applies PhysicalLightIlluminantAPI
prim.GetAttribute("omni:rtx:usdluxVersion").Set(2505)
prim.GetAttribute("inputs:intensity").Set(1.0)
prim.GetAttribute("inputs:exposure").Set(0.0)
prim.GetAttribute("photometric:power").Set(4000.0)
prim.GetAttribute("physical:illuminant").Set("illuminantD")
prim.GetAttribute("physical:colorTemperature").Set(4000.0)

stage.Save()
```

For Distant lights:

```python
prim.ApplyAPI("PhotometricDistantLightAPI")
prim.ApplyAPI("PhysicalLightIlluminantAPI")
prim.GetAttribute("omni:rtx:usdluxVersion").Set(2505)
prim.GetAttribute("inputs:intensity").Set(1.0)
prim.GetAttribute("inputs:exposure").Set(0.0)
prim.GetAttribute("photometric:illuminance").Set(10000.0)  # lux, perpendicular surface
prim.GetAttribute("physical:illuminant").Set("illuminantD")
prim.GetAttribute("physical:colorTemperature").Set(6500.0)
stage.Save()
```

---

## Step 8 — Power Calculation (Lumen Method)

```
Power_per_fixture = (E_target × A_floor) / (N × UF × MF)

E_target = target illuminance (lux)
A_floor  = total floor area (m²)
N        = number of fixtures
UF       = utilization factor (0.35–0.65, based on Room Index)
MF       = maintenance factor (0.8 typical)

Room Index = (L × W) / (H_mount × (L + W))
  RI < 1  → UF = 0.35
  RI 1–2  → UF = 0.45
  RI 2–3  → UF = 0.55
  RI > 3  → UF = 0.65
```

### Target illuminance by scene type

| Scene type | Lux target | Detect from prim names |
|------------|-----------|----------------------|
| Warehouse / industrial | 200–300 | `warehouse`, `factory`, `hangar`, `industrial` |
| Corridors / entrance lobbies | 100–200 | — (sub-type of industrial) |
| Office / conference | 300–500 | `office`, `desk`, `cubicle`, `conference` |
| Retail / showroom | 500–1000 | `retail`, `store`, `shop`, `showroom` |
| Studio / stage | 800–1500 | `studio`, `stage`, `set` |
| TV studio / operating theatre | 1,000 | — (sub-type of studio) |
| Exterior / parking | 50–150 | `exterior`, `outdoor`, `street`, `parking` |

### Real-world fixture reference

| Fixture type | Typical lumens | Color temp |
|-------------|---------------|-----------|
| Industrial high-bay LED (basic) | 15,000–25,000 lm | 4000–5000K |
| Industrial high-bay LED (high output) | 30,000–60,000 lm | 4000–5000K |
| Office panel light | 3,000–6,000 lm | 4000–6500K |
| Retail track spot | 2,000–4,000 lm | 2700–3000K |
| Household bulb | 800–1,600 lm | 2700–3000K |

---

## Step 9 — Setting Brightness for Dome Lights

Apply `PhotometricDomeLightAPI` and set `photometric:illuminance` to the outdoor sky brightness at the HDRI capture location.

| Lux | Scenario |
|-----|---------|
| 0.25 | Full moon |
| 400 | Sunset, clear sky |
| 10,000 | Midday overcast |
| 100,000 | Midday direct sunlight |

> **RTX limitation — interior scenes:** Dome lights illuminate every surface with line-of-sight to the sky. If the building geometry does not fully seal the roof and walls (common in USD assets), outdoor sky values flood the interior. Use `--dome-illuminance` to reduce the dome contribution, or `--no-dome` to disable it entirely. Validate with Debug View → PT AOV Illuminance: if floor lux is dominated by the dome rather than fixtures, reduce `--dome-illuminance`.

```python
dome.GetPrim().ApplyAPI("PhotometricDomeLightAPI")
dome.GetPrim().GetAttribute("omni:rtx:usdluxVersion").Set(2505)
dome.GetPrim().GetAttribute("photometric:illuminance").Set(10000.0)  # midday overcast
dome.GetPrim().GetAttribute("physical:illuminant").Set("illuminantD")
dome.GetPrim().GetAttribute("physical:colorTemperature").Set(6500.0)
```

> When upgrading an EXISTING DomeLight, do NOT set `omni:rtx:usdluxVersion = 2505` — changing the version alters HDRI texture mapping and rotates the sky. Only set it when creating a new DomeLight from scratch.

---

## Step 10 — IES Profiles

IES profiles simulate real fixture light distribution from manufacturer data.

```python
prim.ApplyAPI("PhysicalLightIlluminantAPI")
prim.GetAttribute("inputs:shaping:ies:file").Set("path/to/fixture.ies")
prim.GetAttribute("inputs:intensity").Set(1.0)
prim.GetAttribute("inputs:shaping:ies:angleScale").Set(1.0)  # 1.0 = no change
```

When `PhotometricAreaLightAPI` is also applied, it overrides IES brightness via `photometric:power` but preserves the IES distribution pattern.

IES sources: manufacturer datasheets, https://ieslibrary.com

---

## Step 10b — Fixture Detection Logic

The script uses three-tier logic to decide whether a geometry prim is a light fixture:

**Tier 1 — Hard false positives (always block):**
Fire alarms, mechanical parts, pipe, conduit — these block even if "lamp" is in the name.
Examples: `firelamp`, `sm_firelamp`, `sm_fusebox`, `conduit`, `pipe_`

**Tier 2 — Explicit fixture keywords (always pass, override soft blocks):**
`lamp`, `luminaire`, `luminary`, `ceilinglamp`, `walllamp`, `downlight`, `recessed`, `troffer`, `chandelier`, `pendant`, `sconce`, `lantern`, `bulb`, `globe`

**Tier 3 — Soft false positives (block general keywords but not explicit):**
Structural surfaces: `sm_wall`, `sm_ceiling`, `sm_floor`, `sm_roof`
Example: `sm_walllamp` passes (explicit "lamp" wins), `sm_wall_a01` does not.

**Tier 4 — General fixture keywords (pass if no soft block):**
`light_fixture`, `panel_light`, `spotlight`, `sm_lamp`, `fluorescent`, etc.

### Light type inference from fixture name

| Fixture keyword | USD light type |
|----------------|---------------|
| `tube`, `strip`, `neon`, `fluorescent`, `linear` | CylinderLight |
| `spot`, `downlight`, `cone`, `projector`, `flood` | SphereLight + ShapingAPI cone |
| `panel`, `troffer`, `ceilinglamp`, `walllamp`, `disk` | RectLight |
| `bulb`, `globe`, `pendant`, `chandelier`, `lantern` | SphereLight |
| Fixture in ceiling zone (height ≥ 60% of building) | RectLight (panel default) |
| Anything else | SphereLight (safe default) |

### Glass/window detection

The script triggers a DomeLight if building windows are present. It filters out prop glass:
- Prop context exclusions: `forklift`, `extinguisher`, `camera`, `vehicle`, `robot`, `sensor`
- Minimum area: 0.5 m² (filters windshields and small gauge glass)

---

## Step 11 — Scenes with No Lights

If the scene has no lights and no fixture geometry, create a minimal physically accurate baseline:

```python
from pxr import Usd, UsdLux

stage = Usd.Stage.Open("path/to/scene.usd")

# Dome light — midday overcast sky
dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
dome.GetPrim().ApplyAPI("PhotometricDomeLightAPI")
dome.GetPrim().GetAttribute("omni:rtx:usdluxVersion").Set(2505)
dome.GetIntensityAttr().Set(1.0)
dome.GetExposureAttr().Set(0.0)
dome.GetPrim().GetAttribute("photometric:illuminance").Set(10000.0)
dome.GetPrim().GetAttribute("physical:illuminant").Set("illuminantD")
dome.GetPrim().GetAttribute("physical:colorTemperature").Set(6500.0)

# Key light — 4000 lm LED panel
rect = UsdLux.RectLight.Define(stage, "/World/Lights/KeyLight")
rect.GetPrim().ApplyAPI("PhotometricAreaLightAPI")
rect.GetPrim().GetAttribute("omni:rtx:usdluxVersion").Set(2505)
rect.GetIntensityAttr().Set(1.0)
rect.GetExposureAttr().Set(0.0)
rect.GetPrim().GetAttribute("photometric:power").Set(4000.0)
rect.GetPrim().GetAttribute("physical:illuminant").Set("illuminantD")
rect.GetPrim().GetAttribute("physical:colorTemperature").Set(5600.0)

stage.Save()
```

---

## Reporting Results (AI Agents)

After the script finishes, **always include the full camera exposure block in your response to the user.** Do not summarize or omit it.

The script prints a `CAMERA EXPOSURE CHECK` section at the end of every real (non-dry-run) execution. It contains the Film ISO, F-stop, and Camera Exposure Time the user must set in Composer before validating with Debug View. If you drop this block from your response, the user has no way to correctly calibrate their viewport and the render will look wrong — they will not know why.

**Required output format after a successful run:**

```
Lights created: <N>
Existing deactivated: <N>
DomeLight: Yes / No
Output: <path to physical_lighting_setup.usd>

Camera settings (Render Settings → Post Processing → Tone Mapping):
  Film ISO:              <value>
  F-stop:                f/<value>
  Camera Exposure Time:  <value>s
  → Effective EV: <value>

Validate: Debug View → PT AOV Illuminance → click floor surfaces → target <N> lux
```

If the script printed `(Camera settings not readable — running outside Kit)`, still relay the recommended ISO/F-stop/exposure values that follow that line — they are always printed regardless.

---

## Step 12 — Validate with Debug View

1. Switch renderer to **RTX - Interactive (Path Tracing)**
2. **Render Settings → Post Processing → Tone Mapping** — set Film ISO, F-stop, and Exposure Time to match the camera recommendations in the script output.
3. **Debug View → PT AOV Illuminance** — enable and select in viewport header.
4. Click surfaces in the viewport to read lux values.

Expected ranges: office = 300–500 lux, retail = 500–1000 lux, industrial = 200–300 lux.

**If the scene looks dark but lux reads correctly:** Camera exposure is too low — follow the script's camera recommendations. Do NOT increase light power.

**If lux reads too low everywhere:** Re-run with `--target-lux <higher_value>`.

**If lux reads too high in spots near fixtures:** Normal — lux values directly under a point source are higher than the average floor illuminance. Move to the floor midpoint between fixtures for a representative reading.

---

## Common Errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| Light appears unchanged after schema | `intensity` or `exposure` not reset | Set both to `1.0` / `0.0` |
| `photometric:power` has no effect | `photometric:illuminance:distance` is non-zero | Set `photometric:illuminance:distance` to `0` |
| Schema attributes missing in UI | `omni.kit.widget.schema_api` not loaded | Enable extension via Developer → Extensions |
| `AddAppliedSchema` silently does nothing | `omni.usd.schema.physicallight` not loaded | Enable extension (Execution Step K2) |
| Dome HDRI appears rotated | Upgraded from pre-2505 with version changed | Do not set `omni:rtx:usdluxVersion` when upgrading existing DomeLights |
| Color temperature has no visual effect | Wrong `physical:illuminant` token | Use `blackbody` or `illuminantD` — `white` ignores color temperature |
| Scene too dark, lux reads correct | Camera exposure miscalibrated | Follow camera recommendations in script output — do not change light power |
| Fixture count = 0 in output | No prim names match fixture keywords | Check prim names against Step 10b keyword tables |

---

## Related Resources

- [NVIDIA Omniverse Lighting documentation](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/lighting.html)
- [UsdLux specification](https://openusd.org/release/api/usd_lux_page_front.html)
- [IES Library](https://ieslibrary.com)
- `Physical_Lighting_Guide_BACKUP_2026-04-28.md` — primary domain reference (this repo)

---

---

## EMBEDDED SCRIPT


> **Version sync:** This embedded script is **v4.3**. If you edit `run_lighting_skill.py` externally (bug fixes, tuning), update this embedded block too — they must stay in sync. The version string is in the docstring at the top of the script.

```python
"""
UsdLux Physical Lighting Skill — Canonical Implementation (v4.3)

Usage:
    python run_lighting_skill.py <scene.usd> [options]

Features:
- Adaptive vertex clustering detects individual fixtures in merged meshes
- Works with any unit system (cm, m, mm) — clusters in local space, outputs world space
- Handles xformOp:scale:unitsResolve transparently
- Detects glass/window geometry (by name + material binding) → auto-creates DomeLight
- PointInstancer support for properly instanced fixtures
- Y-up and Z-up scene support
- Dry-run mode for preview without writing
- CLI arguments for color temp, target lux, dome settings
- Power distributed correctly across N detected fixtures

Requires NVIDIA Omniverse Kit 109.0+ — physical light schemas need Kit's extension system.
The skill.md file explains WHEN and WHY to use this; this script is HOW.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import math
import argparse
from collections import Counter

# pxr import — works in Kit (already on path) or standalone Python.
# Falls back to the VM packman path if pxr is not found.
try:
    from pxr import Usd, UsdGeom, UsdLux, Sdf, Gf
except ImportError:
    _USD_LIB = os.path.expanduser(
        "~/.cache/packman/chk/usd.py312.manylinux_2_35_x86_64.stock.release/"
        "0.25.11.kit.1-gl.18239+b8f43314/lib/python"
    )
    sys.path.insert(0, _USD_LIB)
    from pxr import Usd, UsdGeom, UsdLux, Sdf, Gf

# ============================================================
# CONFIG
# ============================================================
LIGHTING_LAYER_NAME = "physical_lighting_setup.usd"


def find_sublayer_folder(root_layer):
    """
    Detect if the scene uses a subfolder for USD sublayers.
    Looks at existing subLayerPaths and sibling directories.
    Returns the relative subfolder path (e.g. 'SubUSDs/') or '' if root-level.
    """
    root_dir = os.path.dirname(root_layer.realPath)

    # Check existing sublayer paths for a pattern
    for sub_path in root_layer.subLayerPaths:
        dirname = os.path.dirname(sub_path)
        if dirname:  # sublayer is in a subfolder
            full_dir = os.path.join(root_dir, dirname)
            if os.path.isdir(full_dir):
                return dirname

    # Check for common USD subfolder conventions
    for candidate in ['SubUSDs', 'sublayers', 'layers', 'usd']:
        candidate_path = os.path.join(root_dir, candidate)
        if os.path.isdir(candidate_path):
            # Verify it contains .usd files (not just textures)
            has_usd = any(
                f.endswith(('.usd', '.usda', '.usdc'))
                for f in os.listdir(candidate_path)
                if os.path.isfile(os.path.join(candidate_path, f))
            )
            if has_usd:
                return candidate

    return ''  # no subfolder detected, use root level


def parse_args():
    """Parse CLI arguments. Scene path is required."""
    parser = argparse.ArgumentParser(
        description="UsdLux Physical Lighting Skill — automated lighting setup for USD scenes.",
        epilog="Example: python run_lighting_skill.py /path/to/scene.usd --dry-run",
    )
    parser.add_argument(
        "scene", help="Path to the USD scene file (.usd, .usda, .usdc, .usdz)"
    )
    parser.add_argument(
        "--output-layer", "-o", default=LIGHTING_LAYER_NAME,
        help=f"Name for the lighting sublayer (default: {LIGHTING_LAYER_NAME})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Analyze and report without writing any files"
    )
    parser.add_argument(
        "--color-temp", type=float, default=4000.0,
        help="Color temperature for fixture lights in Kelvin (default: 4000)"
    )
    parser.add_argument(
        "--dome-color-temp", type=float, default=6500.0,
        help="Color temperature for dome light in Kelvin (default: 6500)"
    )
    parser.add_argument(
        "--dome-illuminance", type=float, default=0.0,
        help="DomeLight illuminance in lux (default: auto — 400 lux sunset/clear sky; RTX handles glass transmission physically)"
    )
    parser.add_argument(
        "--no-dome", action="store_true",
        help="Skip DomeLight creation even if glass/windows detected"
    )
    parser.add_argument(
        "--force-dome", action="store_true",
        help="Create DomeLight even without glass/window geometry"
    )
    parser.add_argument(
        "--target-lux", type=float, default=None,
        help="Override automatic lux target (auto-detected from scene type)"
    )
    parser.add_argument(
        "--measured-lux", type=float, default=None,
        help="Measured floor illuminance in lux from PT AOV Illuminance. "
             "Scales all existing lights by (target/measured) without re-scanning. "
             "Run the skill normally first, then re-run with this flag to calibrate."
    )
    parser.add_argument(
        "--window-size", choices=["small", "medium", "large"], default=None,
        help="Hint for window coverage when using --force-dome. Reduces fixture power "
             "by the estimated indoor daylight contribution: "
             "small (~5%% wall area, e.g. narrow strips) = ~16 lux reduction, "
             "medium (~15%% wall area, standard industrial) = ~48 lux reduction, "
             "large (~30%% wall area, curtain wall / floor-to-ceiling) = ~96 lux reduction."
    )
    parser.add_argument(
        "--dome-contribution", type=float, default=None,
        help="Direct override for estimated indoor daylight contribution in lux "
             "(subtracted from fixture target when --force-dome is used). "
             "Use instead of --window-size when you have a measured or known value."
    )
    return parser.parse_args()

# ============================================================
# FIXTURE DETECTION
# ============================================================
# Keywords that ALWAYS identify a light fixture.
# Checked BEFORE false positives — these win even if a false positive also matches.
# e.g. "sm_walllamp" contains "sm_wall" (false positive) but also "lamp" (explicit) → fixture.
EXPLICIT_FIXTURE_KEYWORDS = [
    "lamp", "luminaire", "luminary",
    "ceilinglamp", "walllamp", "ceiling_light", "wall_light",
    "downlight", "recessed", "troffer", "chandelier",
    "pendant", "sconce", "lantern", "bulb", "globe",
    # Compound LED fixture names — unambiguous, cannot collide with "ledge" etc.
    "led_strip", "led_panel", "led_bar", "led_light", "led_lamp",
    # Tube lighting explicitly identified as lighting (not HVAC/structural)
    "tube_light", "tube_lamp", "led_tube", "fluoro_tube",
    # Emitter geometry — used by DCC tools / procedural pipelines for light source geo
    "lightemitter", "light_emitter", "emitterlight", "emitter_light",
]

# General fixture keywords — checked against leaf name only (after soft false positives clear).
# "led" here: leaf "sm_led_*" passes; leaf "sm_wall_a01_ledge_01" is blocked first by "wall_a".
# Bare "linear" is intentionally excluded — too broad (linear_rail, linear_actuator, etc.).
FIXTURE_KEYWORDS = [
    "light_fixture", "panel_light", "spotlight", "projector",
    "neon", "fluorescent", "linear_light",
    "led",
    "floodlight", "street_light", "mast_light", "post_light",
    "sm_lamp",
]

# Hard false positives — block EVERYTHING including explicit keywords.
# Use for safety/emergency equipment and mechanical parts that contain "lamp" or "light".
HARD_FALSE_POSITIVES = [
    "firelamp", "sm_firelamp",          # fire alarm / emergency lamp
    "sm_wiring", "sm_fusebox", "sm_fan",
    "firetube", "sm_decorelement",
    "_sprayer", "_mounts", "_bolts", "_grille", "_bracket",
    "conduit", "duct_", "pipe_",
    "electricalsystem",  # control panel / electrical box displays
    "_stem",        # lamp pole / post — structural, not the emitter
    "_attachment",  # lamp mount / cap — structural, not the emitter
]

# Soft false positives — block general keywords but NOT explicit fixture keywords.
# e.g. "sm_wall_a01" is blocked, but "sm_walllamp_a01" is not.
# sm_tube_a/b/c are structural by default but yield to "tube_light" / "tube_lamp" explicit names.
FALSE_POSITIVE_PATTERNS = [
    "roomwalls", "room_walls", "sm_ceiling", "sm_wall", "sm_floor",
    "sm_roof", "wall_a", "wall_b", "ceiling_a", "floor_a",
    "sm_tube_a", "sm_tube_b", "sm_tube_c",
]

GLASS_KEYWORDS = [
    "glass", "window", "glazing", "pane", "skylight",
]

# Glass prim names that belong to props, not windows.
# Presence of these words in the prim path means it is NOT a building window.
GLASS_PROP_CONTEXT = [
    "forklift", "extinguisher", "camera", "firelamp", "firebox",
    "ceilinglamp", "walllamp", "headlight", "taillight",
    "instrument", "gauge", "monitor", "screen", "vehicle",
    "robot", "sensor", "lens",
]

# Minimum world-space area (m²) for a glass prim to count as a window.
# Forklift windshield ~0.3 m², fire extinguisher glass ~0.01 m².
# A real window is typically >= 0.5 m².
GLASS_MIN_AREA_M2 = 0.5

SCENE_KEYWORDS = {
    "industrial": ["warehouse", "factory", "hangar", "industrial", "plant"],
    "office": ["office", "desk", "cubicle", "conference", "meeting"],
    "retail": ["retail", "store", "shop", "showroom", "display"],
    "studio": ["studio", "soundstage", "filmset", "film", "photo"],
    "exterior": ["exterior", "outdoor", "street", "parking", "yard"],
}
TARGET_LUX = {
    "industrial": 250,
    "office": 400,
    "retail": 750,
    "studio": 1000,
    "exterior": 100,
    "unknown": 300,
}


def is_fixture(prim):
    """
    Return True if this geometry prim looks like a light fixture.

    Three-tier logic:
    1. Hard false positives (firelamp, pipes, mechanical) → always False
    2. Explicit fixture keywords (lamp, walllamp, ceilinglamp, ...) → always True
    3. Soft false positives (sm_wall, sm_ceiling structural parts) → False
    4. General fixture keywords → True

    This means sm_walllamp passes (lamp overrides sm_wall),
    but sm_firelamp does not (firelamp is a hard false positive).
    """
    name = prim.GetName().lower()
    path = str(prim.GetPath()).lower()
    # Tier 1: hard false positives — block everything (leaf name only)
    if any(fp in name for fp in HARD_FALSE_POSITIVES):
        return False
    # Tier 2: explicit fixture keywords — check full path so that generic leaf
    # names like Section0/Section1 inside SM_LampCeilingA_* are still detected.
    if any(k in path for k in EXPLICIT_FIXTURE_KEYWORDS):
        return True
    # Tier 3: soft false positives — block general keywords (leaf name only)
    if any(fp in name for fp in FALSE_POSITIVE_PATTERNS):
        return False
    # Tier 4: general fixture keywords (leaf name only to avoid over-matching)
    return any(k in name for k in FIXTURE_KEYWORDS)


# Keywords identifying robot / vehicle / mobile equipment paths.
# Lights on these prims are excluded from upgrade — logged but not touched.
ROBOT_VEHICLE_KEYWORDS = [
    "nova_carter", "carter", "forklift", "vehicle", "robot",
    "drone", "agv", "amr", "jetbot", "legged", "quadruped",
    "manipulator", "excavator", "truck", "van_", "car_",
]


def is_robot_vehicle_light(lp):
    """Return True if this light belongs to a robot, vehicle, or mobile equipment."""
    path = str(lp.GetPath()).lower()
    return any(k in path for k in ROBOT_VEHICLE_KEYWORDS)


def is_glass(prim, bbox_cache=None, mpu=1.0):
    """
    Return True if this geometry prim looks like a building window or skylight.

    Rejects:
    - Glass that is part of a prop (forklift, fire extinguisher, camera, lamp cover, etc.)
    - Glass geometry that is too small to be a window (< GLASS_MIN_AREA_M2)
    """
    name = prim.GetName().lower()
    path = str(prim.GetPath()).lower()
    combined = name + " " + path

    # Must have a glass keyword
    if not any(k in combined for k in GLASS_KEYWORDS):
        return False

    # Reject doors (usually opaque)
    if "door" in name:
        return False

    # Reject prop glass — check both prim name and full path for context
    if any(ctx in combined for ctx in GLASS_PROP_CONTEXT):
        return False

    # Size check: window must be large enough to matter.
    # Use largest single dimension (not area) so flat thin panels (wall/roof glass
    # with 1cm depth but 2m face) are not discarded by a misleading area product.
    # Exception: if the prim name itself contains "window" it is almost certainly
    # architectural glass — accept it regardless of computed size.
    if "window" not in name and bbox_cache is not None:
        try:
            wb = bbox_cache.ComputeWorldBound(prim)
            size = wb.ComputeAlignedRange().GetSize()
            dims = sorted([abs(size[0]) * mpu, abs(size[1]) * mpu, abs(size[2]) * mpu])
            if dims[2] < GLASS_MIN_AREA_M2:  # largest dimension < 0.5 m → too small
                return False
        except Exception:
            pass  # if bbox fails, fall through to name-based decision

    return True


def has_emissive_material(prim, threshold=0.01):
    """
    Return True if this prim's bound material has a non-trivial emissive output.

    Walks all Shader children of the material directly — this avoids the MDL vs
    UsdPreviewSurface surface-terminal mismatch (MDL uses outputs:mdl:surface,
    not outputs:surface, so GetSurfaceOutput() finds nothing for MDL materials).

    Detection rules per shader found:
      OmniPBR / MDL:       enable_emission=True  AND  emissive_intensity > threshold
      UsdPreviewSurface:   emissiveColor non-black  OR  connected to a texture
      MDL without flag:    emissive_color non-black (direct value, not texture attribute)

    Note: OmniPBR stores emissive_color_texture as a plain asset attribute, not a
    UsdShade connection, so HasConnectedSource() does not apply — the authoritative
    signal for OmniPBR is enable_emission + emissive_intensity.
    """
    try:
        from pxr import UsdShade
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material:
            return False
        for child in material.GetPrim().GetAllChildren():
            if child.GetTypeName() != "Shader":
                continue
            shader = UsdShade.Shader(child)
            # OmniPBR / MDL: enable_emission is the authoritative on/off switch.
            # emissive_color_texture overrides emissive_color so the direct color
            # value may be near-black even when the material is visibly emissive.
            ee = shader.GetInput('enable_emission')
            if ee and ee.Get():
                ei = shader.GetInput('emissive_intensity')
                if ei:
                    v = ei.Get()
                    if v is not None and float(v) > threshold:
                        return True
            # UsdPreviewSurface: emissiveColor as direct value or texture connection
            ec = shader.GetInput('emissiveColor')
            if ec:
                v = ec.Get()
                if v and max(v) > threshold:
                    return True
                if ec.HasConnectedSource():
                    return True
            # MDL without enable flag: emissive_color direct (non-texture) value
            emc = shader.GetInput('emissive_color')
            if emc:
                v = emc.Get()
                if v and hasattr(v, '__iter__') and max(v) > threshold:
                    return True
    except Exception:
        pass
    return False


def infer_light_type(prim):
    """Return (light_class_name, hints) based on fixture prim."""
    return infer_light_type_from_name(prim.GetName() + " " + str(prim.GetPath()))


def _classify_bbox_shape(dims_m):
    """Return 'RectLight', 'CylinderLight', or 'SphereLight' from 3 bbox dimensions in metres."""
    if not dims_m or len(dims_m) < 3 or min(dims_m) <= 0:
        return None
    dims = sorted(dims_m)                       # [smallest, mid, largest]
    small, mid, large = dims
    if large <= 0:
        return None
    if small / large < 0.25:                    # flat panel: thinnest axis < 25% of widest
        return "RectLight"
    if large / max(mid, 0.001) > 2.5:          # elongated tube
        return "CylinderLight"
    return "SphereLight"


def infer_light_type_from_name(name, height_m=None, ceiling_h=None, fmeta=None):
    """
    Return (light_class_name, hints) based on fixture name and optional height.

    Rules:
    1. Tube/strip/linear → CylinderLight
    2. Spot/downlight/cone → SphereLight with cone shaping
    3. Panel/disk/ceiling/wall lamp → RectLight (flat fixture)
    4. Bulb/globe/pendant/chandelier → SphereLight (round fixture)
    5. Height fallback: if fixture is in the upper 40% of the building
       (ceiling zone), default to RectLight — ceiling fixtures are
       almost always flat panels, not bulbs.
    6. Bbox shape analysis: if fmeta has a bbox_ltype from 3D bounding box
       aspect-ratio classification, use it before falling back to SphereLight.
    7. Otherwise → SphereLight (safe default for unknown floor/accent lights)
    """
    name = name.lower()

    # Elongated tube fixtures
    if any(k in name for k in ["tube", "strip", "neon", "fluorescent", "linear"]):
        return "CylinderLight", {"type": "cylinder"}

    # Directional spot fixtures
    if any(k in name for k in ["spot", "downlight", "cone", "projector", "flood"]):
        return "SphereLight", {"type": "spot", "cone": True}

    # Flat panel / ceiling / wall / street fixtures → RectLight
    if any(k in name for k in ["panel", "troffer", "rect", "sconce", "disk",
                               "ceilinglight", "ceilinglamp",
                               "walllight", "walllamp",
                               "officelight", "officedisk",
                               "overhead", "soffit",
                               "streetlamp", "streetlight"]):
        return "RectLight", {"type": "rect"}

    # Explicitly round/globe fixtures → SphereLight
    if any(k in name for k in ["bulb", "globe", "pendant", "chandelier", "lantern"]):
        return "SphereLight", {"type": "sphere"}

    # Height-based fallback: ceiling zone fixtures are panels by default
    if height_m is not None and ceiling_h is not None:
        if height_m >= ceiling_h * 0.6:
            return "RectLight", {"type": "rect"}

    # Bbox shape analysis: 3D aspect-ratio classification from detection-time world bbox
    if fmeta and fmeta.get('bbox_ltype'):
        _blt = fmeta['bbox_ltype']
        return _blt, {"type": _blt.replace('Light', '').lower()}

    return "SphereLight", {"type": "sphere"}


def infer_scene_type(stage):
    # Include file path so "warehouse" in the filename counts even if prim paths don't.
    file_hint = os.path.basename(stage.GetRootLayer().realPath).lower()
    all_names = file_hint + " " + " ".join(str(p.GetPath()).lower() for p in stage.Traverse())
    for scene_type, keywords in SCENE_KEYWORDS.items():
        if any(k in all_names for k in keywords):
            return scene_type
    return "unknown"


def get_world_position(prim):
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        return None
    xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return Gf.Vec3d(xform.ExtractTranslation())


def proximity_threshold(max_scene_dim_m):
    """
    Proximity threshold for matching fixture geometry to existing lights.
    Scales with scene size to handle both small and large warehouses:
    - Small scenes (~22m): ~2m threshold
    - Medium scenes (~50m): ~3m threshold
    - Large scenes (~130m): ~5m threshold
    Capped at 5m to avoid wrong cross-room matches.
    """
    return max(1.5, min(max_scene_dim_m * 0.03, 5.0))


def deduplicate_positions(positions, threshold_m):
    """
    Merge positions within threshold_m of each other into one.
    Uses highest-Z position as representative (emitter surface).
    positions: list of (x, y, z, name) or (x, y, z, name, meta)
    """
    if not positions:
        return []
    merged = []
    used = set()
    for i, pos_i in enumerate(positions):
        if i in used:
            continue
        x1, y1, z1 = pos_i[0], pos_i[1], pos_i[2]
        cluster = [pos_i]
        used.add(i)
        for j, pos_j in enumerate(positions):
            if j in used:
                continue
            x2, y2, z2 = pos_j[0], pos_j[1], pos_j[2]
            if math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2) <= threshold_m:
                cluster.append(pos_j)
                used.add(j)
        merged.append(max(cluster, key=lambda p: p[2]))
    return merged


# Sub-mesh suffixes in order of preference for light placement.
# Body and cover are merged meshes that cluster correctly.
# Screws, cables, decals, corners are structural hardware — ignored if better option exists.
_SUBMESH_PRIORITY = ["_led", "_glass", "_body", "_cover", "_wire"]
_SUBMESH_IGNORE   = ["_screws", "_cables", "_decals", "_corners",
                     "_mount", "_device", "_bolts", "_mounts", "_bracket"]


def select_best_submesh_positions(raw_positions):
    """
    Group raw sub-mesh positions by their parent fixture Xform path,
    then keep only positions from the highest-priority sub-mesh.

    This means: one parent Xform = one light source cluster,
    regardless of how many sub-meshes (body, cover, led, screws, cables...)
    were detected underneath it.

    Args:
        raw_positions: list of (x, y, z, name, parent_path)

    Returns:
        Deduplicated list of (x, y, z, name) — one cluster per fixture parent
    """
    if not raw_positions:
        return []

    # Group by parent path
    groups = {}   # parent_path -> [(x, y, z, name, priority, meta)]
    for entry in raw_positions:
        x, y, z, name, parent_path = entry[0], entry[1], entry[2], entry[3], entry[4]
        meta = entry[5] if len(entry) > 5 else None
        name_lower = name.lower()
        # Skip known structural/hardware sub-mesh types entirely
        if any(s in name_lower for s in _SUBMESH_IGNORE):
            continue
        # Calculate sub-mesh priority (lower index = preferred)
        pri = len(_SUBMESH_PRIORITY)
        for i, suffix in enumerate(_SUBMESH_PRIORITY):
            if suffix in name_lower:
                pri = i
                break
        if parent_path not in groups:
            groups[parent_path] = []
        groups[parent_path].append((x, y, z, name, pri, meta))

    # For each parent, keep positions from the best sub-mesh type only
    final = []
    for parent_path, entries in groups.items():
        best_pri = min(e[4] for e in entries)
        # Carry meta through as the 5th element of each position tuple
        best = [(x, y, z, n, meta) for x, y, z, n, p, meta in entries if p == best_pri]
        # Secondary tiebreaker: when naming priority is equal (e.g. generic names like
        # Section0/Section1 that don't match any suffix), prefer submeshes that went
        # through geometric analysis (elongated/clustered — have meta) over single-point
        # fallbacks (meta=None). This avoids ceiling-mount attachment pieces being kept
        # alongside the actual emitter submesh.
        has_geo = [e for e in best if e[4] is not None]
        if has_geo:
            best = has_geo
        # Final proximity dedup within this group (handles any remaining duplicates)
        best = deduplicate_positions(best, threshold_m=0.15)
        final.extend(best)

    return final


# ============================================================
# VERTEX CLUSTERING — DETECT INDIVIDUAL FIXTURES IN MERGED MESH
# ============================================================
def estimate_bucket_size(points, up_axis="Z"):
    """
    Estimate an appropriate bucket size for fixture clustering.

    Strategy: sample vertex positions on the floor-plane axes,
    find the gap pattern between dense regions. The bucket size
    should be smaller than the gap between fixtures but larger
    than a single fixture's footprint.

    Falls back to span/20 if no clear pattern is found.
    """
    if not points or len(points) < 100:
        return 100  # safe fallback

    # Determine which axes are the floor plane
    if up_axis == "Z":
        ax0 = [p[0] for p in points]  # X
        ax1 = [p[1] for p in points]  # Y
    else:  # Y-up
        ax0 = [p[0] for p in points]  # X
        ax1 = [p[2] for p in points]  # Z

    span0 = max(ax0) - min(ax0)
    span1 = max(ax1) - min(ax1)
    max_span = max(span0, span1)

    if max_span < 1:
        return 1  # degenerate mesh

    # Heuristic: try bucket = span / 50 first, count clusters.
    # If we get a reasonable fixture count (3+), that's good.
    # Otherwise progressively increase bucket size.
    # Target: bucket should be ~1/3 the spacing between fixtures.
    # For a 5400-unit span with 9 fixtures: spacing ~600, bucket ~200.
    # For a 54-unit span (meters) with 9 fixtures: spacing ~6, bucket ~2.
    # General: span / 30 is a reasonable starting point.
    bucket = max_span / 30.0

    # Round to a clean number for consistent bucketing
    # Find nearest power-of-10 magnitude
    magnitude = 10 ** int(math.log10(max(bucket, 0.1)))
    bucket = round(bucket / magnitude) * magnitude
    bucket = max(bucket, 1.0)  # never less than 1 unit

    return bucket


def split_elongated_sub_units(points, long_local, min_gap_frac=0.10, long_dim_m=None):
    """
    Split a merged elongated mesh into individual sub-units by finding large vertex gaps
    along the long axis.  Returns one dict per sub-unit with actual vertex bounds:
        {'long_min', 'long_max', 'short_min', 'short_max'}   (all in local coordinates)
    long_local: 'X' uses p[0] as long coord, 'Y' uses p[1].
    Returns [single full-bbox dict] when no significant gap is found.
    """
    if long_local == 'X':
        long_of  = lambda p: p[0]
        short_of = lambda p: p[1]
    else:
        long_of  = lambda p: p[1]
        short_of = lambda p: p[0]

    sorted_pts = sorted(points, key=long_of)
    if len(sorted_pts) < 4:
        lc = [long_of(p) for p in sorted_pts]
        sc = [short_of(p) for p in sorted_pts]
        return [{'long_min': min(lc), 'long_max': max(lc),
                 'short_min': min(sc), 'short_max': max(sc)}]

    total_span = long_of(sorted_pts[-1]) - long_of(sorted_pts[0])
    if total_span < 1.0:
        lc = [long_of(p) for p in sorted_pts]
        sc = [short_of(p) for p in sorted_pts]
        return [{'long_min': min(lc), 'long_max': max(lc),
                 'short_min': min(sc), 'short_max': max(sc)}]

    gap_threshold = total_span * min_gap_frac

    # Partition sorted points into groups separated by large gaps
    groups = [[sorted_pts[0]]]
    for pt in sorted_pts[1:]:
        if long_of(pt) - long_of(groups[-1][-1]) > gap_threshold:
            groups.append([pt])
        else:
            groups[-1].append(pt)

    # Hollow-frame guard: if exactly 2 groups with a >70% interior gap, the mesh may be a
    # single rectangular panel whose vertices live only on the two end-caps (e.g. a diffuser
    # cover modelled as a chamfered box). The gap is the empty interior of one fixture, not
    # spacing between two separate fixtures — treat as 1 unit.
    # Guard is skipped when the total fixture is >2.5m: a single architectural light fixture
    # is never that long, so if two vertex clusters span >2.5m they are separate lamp heads
    # (e.g. a 2-lamp wall strip where each LED face is a tiny dense cluster at each end).
    if len(groups) == 2:
        interior_gap = long_of(groups[1][0]) - long_of(groups[0][-1])
        if interior_gap > total_span * 0.70:
            if long_dim_m is None or long_dim_m <= 2.5:
                all_pts = groups[0] + groups[1]
                lc = [long_of(p) for p in all_pts]
                sc = [short_of(p) for p in all_pts]
                return [{'long_min': min(lc), 'long_max': max(lc),
                         'short_min': min(sc), 'short_max': max(sc)}]

    result = []
    for group in groups:
        lc = [long_of(p) for p in group]
        sc = [short_of(p) for p in group]
        result.append({'long_min': min(lc), 'long_max': max(lc),
                       'short_min': min(sc), 'short_max': max(sc)})
    return result


def find_panels_by_gap_analysis(points, world_xform, mpu, up_axis="Z", min_gap_frac=0.10):
    """
    Detect individual glass / LED panel faces in a merged fixture mesh.

    Two-level detection:
      Level 1 — Coarse split: find large gaps along each floor-plane axis.
        * Exactly 1 significant gap  → split into 2 halves (each half may contain
          multiple side-by-side panes that are detected at level 2).
        * Multiple bimodal gaps      → split at the large-gap tier only (small gaps
          are intra-pane vertex edges).
        * Multiple equal gaps        → uniform array (no panel structure) → return []
          so the caller falls back to flood-fill.
      Level 2 — Sub-pane detection within each coarse half:
        * On the "single-large-gap" axis: consecutive-pair grouping — each pair
          of adjacent unique coordinate values defines one glass pane face.
        * On the "bimodal" axis: use group min/max as the full panel extent
          (the bimodal gaps already separated individual pane heights).

    Returns list of (x, y, z, {'dim0_m': ..., 'dim1_m': ...}) in world space,
    or [] when no multi-panel structure is found.
    """
    if not points or len(points) < 4:
        return []

    if up_axis == "Z":
        ax0 = [p[0] for p in points]  # X
        ax1 = [p[1] for p in points]  # Y
        ax2 = [p[2] for p in points]
    else:
        ax0 = [p[0] for p in points]
        ax1 = [p[2] for p in points]
        ax2 = [p[1] for p in points]

    cz_local = min(ax2)

    def _coarse_split(sorted_coords, total_span, min_frac):
        """
        Find coarse coordinate groups separated by dominant gaps.
        Returns (groups, n_sig) where groups is a list of value-lists and
        n_sig is the count of significant gaps that triggered the split.
        n_sig == 1 → single large gap axis (sub-pane pairing needed within each group).
        n_sig > 1  → bimodal axis (group min/max = full panel extent).
        Returns (None, 0) when no clear structure exists.
        """
        threshold = total_span * min_frac
        sig_gaps = [sorted_coords[i] - sorted_coords[i-1]
                    for i in range(1, len(sorted_coords))
                    if sorted_coords[i] - sorted_coords[i-1] > threshold]
        if not sig_gaps:
            return None, 0
        max_g, min_g = max(sig_gaps), min(sig_gaps)
        # Multiple equal gaps = uniform array (lamp grid, not panel structure)
        if len(sig_gaps) > 1 and max_g <= 2.0 * min_g:
            return None, 0
        effective = max_g / 2.0
        groups = [[sorted_coords[0]]]
        for i in range(1, len(sorted_coords)):
            if sorted_coords[i] - sorted_coords[i-1] >= effective:
                groups.append([sorted_coords[i]])
            else:
                groups[-1].append(sorted_coords[i])
        if len(groups) < 2:
            return None, 0
        return groups, len(sig_gaps)

    def _subpane_pairs(group_coords):
        """
        Within a coarse group on the "single-gap" axis, find individual glass pane
        faces by pairing consecutive unique coordinate values.
        Each consecutive pair (val[2k], val[2k+1]) defines one pane's two edge values.
        Returns list of {'center': ..., 'min': ..., 'max': ...}.
        """
        span = max(group_coords) - min(group_coords) if len(group_coords) > 1 else 0.0
        # Tolerance = 0.1% of group span (merges float-precision near-duplicates while
        # preserving genuinely distinct edge positions)
        tol = max(span * 0.001, 1e-6)
        sorted_vals = sorted(group_coords)
        uniq = []
        for v in sorted_vals:
            if not uniq or v - uniq[-1] > tol:
                uniq.append(v)
        if len(uniq) < 2:
            v = uniq[0] if uniq else 0.0
            return [{'center': v, 'min': v, 'max': v}]
        if len(uniq) % 2 == 0:
            panes = []
            for k in range(0, len(uniq), 2):
                panes.append({'center': (uniq[k] + uniq[k+1]) / 2.0,
                              'min': uniq[k], 'max': uniq[k+1]})
            return panes
        else:
            # Odd number: treat full group extent as one pane
            return [{'center': (uniq[0] + uniq[-1]) / 2.0,
                     'min': uniq[0], 'max': uniq[-1]}]

    # --- Level 1: coarse split on both floor-plane axes ---
    sorted_ax0 = sorted(ax0)
    sorted_ax1 = sorted(ax1)
    span_ax0 = sorted_ax0[-1] - sorted_ax0[0]
    span_ax1 = sorted_ax1[-1] - sorted_ax1[0]

    groups_ax0, nsig0 = _coarse_split(sorted_ax0, span_ax0, min_gap_frac)
    groups_ax1, nsig1 = _coarse_split(sorted_ax1, span_ax1, min_gap_frac)

    if groups_ax0 is None and groups_ax1 is None:
        return []  # No panel structure — caller uses flood-fill

    # --- Level 2: sub-pane detection within each coarse group ---
    # "Single-gap" axis → sub-pane pairs. "Bimodal" axis → group min/max.
    if groups_ax0 is not None:
        if nsig0 == 1:
            # Sub-pane pairing: each coarse half has N individual pane faces
            panes_ax0 = []
            for g in groups_ax0:
                panes_ax0.extend(_subpane_pairs(g))
        else:
            # Bimodal: each group IS one complete pane (min-to-max extent)
            panes_ax0 = [{'center': (min(g) + max(g)) / 2.0,
                          'min': min(g), 'max': max(g)}
                         for g in groups_ax0]
    else:
        panes_ax0 = [{'center': (sorted_ax0[0] + sorted_ax0[-1]) / 2.0,
                      'min': sorted_ax0[0], 'max': sorted_ax0[-1]}]

    if groups_ax1 is not None:
        if nsig1 == 1:
            panes_ax1 = []
            for g in groups_ax1:
                panes_ax1.extend(_subpane_pairs(g))
        else:
            panes_ax1 = [{'center': (min(g) + max(g)) / 2.0,
                          'min': min(g), 'max': max(g)}
                         for g in groups_ax1]
    else:
        panes_ax1 = [{'center': (sorted_ax1[0] + sorted_ax1[-1]) / 2.0,
                      'min': sorted_ax1[0], 'max': sorted_ax1[-1]}]

    total_panels = len(panes_ax0) * len(panes_ax1)
    if total_panels <= 1:
        return []

    # --- Generate one RectLight position per (ax0_pane × ax1_pane) combination ---
    # Rotation of the lamp assembly's local X axis in the world floor plane.
    # This is the same for every panel in the mesh — it comes from the parent xform chain.
    # We store it so the RectLight can be rotated to match the assembly orientation.
    _o  = world_xform.Transform(Gf.Vec3d(0, 0, 0))
    _xv = world_xform.Transform(Gf.Vec3d(1, 0, 0))
    _dx = _xv - _o
    if up_axis == "Z":
        _dlen = math.sqrt(_dx[0] ** 2 + _dx[1] ** 2)
        rot_up_deg = math.degrees(math.atan2(_dx[1], _dx[0])) if _dlen > 1e-6 else 0.0
    else:
        _dlen = math.sqrt(_dx[0] ** 2 + _dx[2] ** 2)
        rot_up_deg = math.degrees(math.atan2(_dx[0], _dx[2])) if _dlen > 1e-6 else 0.0
    results = []
    for px in panes_ax0:
        for py in panes_ax1:
            if up_axis == "Z":
                ctr_local = Gf.Vec3d(px['center'], py['center'], cz_local)
                wa  = world_xform.Transform(Gf.Vec3d(px['min'], 0, 0))
                wb  = world_xform.Transform(Gf.Vec3d(px['max'], 0, 0))
                wa2 = world_xform.Transform(Gf.Vec3d(0, py['min'], 0))
                wb2 = world_xform.Transform(Gf.Vec3d(0, py['max'], 0))
            else:
                ctr_local = Gf.Vec3d(px['center'], cz_local, py['center'])
                wa  = world_xform.Transform(Gf.Vec3d(px['min'], 0, 0))
                wb  = world_xform.Transform(Gf.Vec3d(px['max'], 0, 0))
                wa2 = world_xform.Transform(Gf.Vec3d(0, 0, py['min']))
                wb2 = world_xform.Transform(Gf.Vec3d(0, 0, py['max']))
            ctr_world = world_xform.Transform(ctr_local)
            dim0_m = max((wb  - wa ).GetLength() * mpu, 0.05)
            dim1_m = max((wb2 - wa2).GetLength() * mpu, 0.05)
            # Pull light 5 cm below the emitter surface so the source sits
            # just outside the glass face rather than flush with it.
            pull = 0.05 / mpu
            if up_axis == "Z":
                results.append((ctr_world[0], ctr_world[1], ctr_world[2] - pull,
                                {'dim0_m': dim0_m, 'dim1_m': dim1_m, 'rot_up_deg': rot_up_deg}))
            else:
                results.append((ctr_world[0], ctr_world[1] - pull, ctr_world[2],
                                {'dim0_m': dim0_m, 'dim1_m': dim1_m, 'rot_up_deg': rot_up_deg}))

    return results


def merge_aligned_cluster_pairs(centers, up_axis="Z", axis_tol=0.35):
    """
    Merge cluster pairs that are the two end-caps of one elongated fixture.

    Each fluorescent tube produces two clusters — one per end — because the
    vertex density in the middle of the tube is too sparse for the flood-fill.
    Two clusters belong to the same tube when they share the same height (Z)
    and same position on one floor axis, differing only along the other axis.

    Returns a new list where each such pair is replaced by a single entry
    whose position is the midpoint and whose dim0_m spans the full tube length.
    """
    if not centers or len(centers) < 2:
        return centers

    ia_up = 2 if up_axis == "Z" else 1   # index of the "up" axis

    used = [False] * len(centers)
    merged = []

    for i in range(len(centers)):
        if used[i]:
            continue
        ci = centers[i]
        best_j, best_long_dist = -1, float('inf')

        for j in range(len(centers)):
            if j == i or used[j]:
                continue
            cj = centers[j]

            # Must share height
            if abs(ci[ia_up] - cj[ia_up]) > axis_tol:
                continue

            # Floor-plane axes
            if up_axis == "Z":
                da0 = abs(ci[0] - cj[0])   # X separation
                da1 = abs(ci[1] - cj[1])   # Y separation
            else:
                da0 = abs(ci[0] - cj[0])   # X separation
                da1 = abs(ci[2] - cj[2])   # Z separation

            # Valid pair: one floor axis nearly identical, the other is the lamp axis
            aligned = (da0 < axis_tol < da1) or (da1 < axis_tol < da0)
            if not aligned:
                continue

            long_dist = max(da0, da1)
            if long_dist < best_long_dist:
                best_long_dist = long_dist
                best_j = j

        if best_j >= 0:
            cj = centers[best_j]
            used[i] = used[best_j] = True

            mx = (ci[0] + cj[0]) / 2
            my = (ci[1] + cj[1]) / 2
            mz = (ci[2] + cj[2]) / 2

            mi = (ci[3] or {}) if len(ci) > 3 else {}
            mj = (cj[3] or {}) if len(cj) > 3 else {}

            if up_axis == "Z":
                da0 = abs(ci[0] - cj[0])
                da1 = abs(ci[1] - cj[1])
            else:
                da0 = abs(ci[0] - cj[0])
                da1 = abs(ci[2] - cj[2])

            # World-space union AABB of the two end-cap clusters: spans
            # tip-to-tip along the tube and edge-to-edge across the housing.
            # Exact and rotation-invariant — no axis assumptions needed.
            if 'wxmin_m' in mi and 'wxmin_m' in mj:
                w_dx = max(mi['wxmax_m'], mj['wxmax_m']) - min(mi['wxmin_m'], mj['wxmin_m'])
                w_dy = max(mi['wymax_m'], mj['wymax_m']) - min(mi['wymin_m'], mj['wymin_m'])
                long_m  = max(w_dx, w_dy)
                short_m = min(w_dx, w_dy)
            else:
                # Fallback when world bbox is unavailable: rotation-invariant min/max
                d_small = min(mi.get('dim0_m') or 0.05, mi.get('dim1_m') or 0.05)
                d_large = max(mi.get('dim0_m') or mj.get('dim0_m') or 0.05,
                              mi.get('dim1_m') or mj.get('dim1_m') or 0.05)
                da = da0 if da0 >= da1 else da1
                long_m  = da + d_small
                short_m = d_large

            merged.append((mx, my, mz,
                           {'dim0_m': long_m, 'dim1_m': short_m, 'from_cluster': True}))
        else:
            used[i] = True
            merged.append(ci)

    return merged


def find_flat_panels_in_mesh(mesh, mpu, world_xform, angle_deg=25.0,
                              min_area_m2=0.0001, max_panels=20):
    """
    Detect flat emitting panels in a single mesh prim.

    Algorithm:
    1. BFS connected components — adjacent faces with normals within angle_deg
    2. Normal-direction grouping — group components by mean normal (within 15°).
       Select the directional group with the most total area — those are the
       emitting panels (all facing the same way, e.g. forward-facing LED array).
    3. Secondary area gap (3×) within the selected group to drop tiny remnants.
    4. Safety cap — if >max_panels survive, return [] (fall through to single fixture).

    Returns [(cx, cy, cz, meta), ...] in world space with len >= 2, or [].
    """
    _tc      = Usd.TimeCode.Default()
    points   = mesh.GetPointsAttr().Get(_tc)
    fvc_attr = mesh.GetFaceVertexCountsAttr().Get(_tc)
    fvi_attr = mesh.GetFaceVertexIndicesAttr().Get(_tc)
    if not (points and fvc_attr and fvi_attr):
        return []

    nf = len(fvc_attr)
    cos_adj = math.cos(math.radians(angle_deg))

    # Precompute face → vertex-index offset (O(1) random access in BFS)
    fvi_off = [0] * (nf + 1)
    for fi in range(nf):
        fvi_off[fi + 1] = fvi_off[fi] + fvc_attr[fi]

    # Per-face: unit normal, centroid and half-|cross| area (local space)
    face_n = [None] * nf
    face_c = [None] * nf
    face_a = [0.0]  * nf

    for fi in range(nf):
        fc  = fvc_attr[fi]
        off = fvi_off[fi]
        if fc < 3:
            continue
        p0 = points[fvi_attr[off]]
        p1 = points[fvi_attr[off + 1]]
        p2 = points[fvi_attr[off + 2]]
        e1x = p1[0]-p0[0]; e1y = p1[1]-p0[1]; e1z = p1[2]-p0[2]
        e2x = p2[0]-p0[0]; e2y = p2[1]-p0[1]; e2z = p2[2]-p0[2]
        nx = e1y*e2z - e1z*e2y
        ny = e1z*e2x - e1x*e2z
        nz = e1x*e2y - e1y*e2x
        nl = (nx*nx + ny*ny + nz*nz) ** 0.5
        if nl < 1e-10:
            continue
        face_n[fi] = (nx/nl, ny/nl, nz/nl)
        face_a[fi] = nl * 0.5
        face_c[fi] = (
            sum(points[fvi_attr[off + j]][0] for j in range(fc)) / fc,
            sum(points[fvi_attr[off + j]][1] for j in range(fc)) / fc,
            sum(points[fvi_attr[off + j]][2] for j in range(fc)) / fc,
        )

    # Build edge → [face_index, ...] adjacency
    edge_adj = {}
    for fi in range(nf):
        if face_n[fi] is None:
            continue
        fc  = fvc_attr[fi]
        off = fvi_off[fi]
        for j in range(fc):
            v0 = fvi_attr[off + j]
            v1 = fvi_attr[off + (j + 1) % fc]
            key = (v0, v1) if v0 < v1 else (v1, v0)
            edge_adj.setdefault(key, []).append(fi)

    # BFS: connected components of adjacent faces with similar normals
    comp  = [-1] * nf
    comps = []
    for start in range(nf):
        if face_n[start] is None or comp[start] >= 0:
            continue
        cid   = len(comps)
        comps.append([])
        stack = [start]
        comp[start] = cid
        while stack:
            fi  = stack.pop()
            comps[cid].append(fi)
            fc  = fvc_attr[fi]
            off = fvi_off[fi]
            for j in range(fc):
                v0  = fvi_attr[off + j]
                v1  = fvi_attr[off + (j + 1) % fc]
                key = (v0, v1) if v0 < v1 else (v1, v0)
                for nfi in edge_adj.get(key, []):
                    if comp[nfi] >= 0 or face_n[nfi] is None:
                        continue
                    n1 = face_n[fi]; n2 = face_n[nfi]
                    if n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2] >= cos_adj:
                        comp[nfi] = cid
                        stack.append(nfi)

    # Build candidate list with actual face area (m²) and area-weighted mean normal
    candidates = []
    for face_list in comps:
        tot_a = sum(face_a[fi] for fi in face_list)
        actual_m2 = tot_a * mpu * mpu
        if actual_m2 < min_area_m2:
            continue
        anx = any_ = anz = 0.0
        for fi in face_list:
            w = face_a[fi]; n = face_n[fi]
            anx += n[0]*w; any_ += n[1]*w; anz += n[2]*w
        anl = (anx*anx + any_*any_ + anz*anz) ** 0.5
        if anl < 1e-10:
            continue
        anx /= anl; any_ /= anl; anz /= anl
        candidates.append((actual_m2, face_list, tot_a, anx, any_, anz))

    if len(candidates) < 2:
        return []

    # Sort descending by area; pre-limit to keep grouping O(N²) manageable
    candidates.sort(key=lambda c: c[0], reverse=True)
    candidates = candidates[:max_panels * 5]

    # Modal area clustering on ALL candidates first (±0.5%).
    # Finds the sub-group of equal-area components with the greatest total area.
    # Running this BEFORE direction-grouping ensures all same-size panels (e.g.
    # 8 LED emitters spread across a curved array) are captured in one pass,
    # even if their normals span more than the direction-grouping tolerance.
    AREA_TOL = 0.005
    used_mb  = [False] * len(candidates)
    modal_bins = []
    for i in range(len(candidates)):
        if used_mb[i]:
            continue
        mb = [i]; used_mb[i] = True
        ai = candidates[i][0]
        for j in range(i + 1, len(candidates)):
            if used_mb[j]:
                continue
            aj = candidates[j][0]
            if abs(ai - aj) / max(ai, 1e-15) <= AREA_TOL:
                mb.append(j); used_mb[j] = True
        modal_bins.append(mb)

    best_modal = max(modal_bins, key=lambda b: sum(candidates[i][0] for i in b))
    selected   = sorted([candidates[i] for i in best_modal], key=lambda c: -c[0])

    if len(selected) < 2:
        return []

    # Secondary area gap: drop tiny outliers if there's a 3× drop in area.
    cutoff = len(selected)
    for i in range(1, len(selected)):
        if selected[i - 1][0] / max(selected[i][0], 1e-15) >= 3.0:
            cutoff = i
            break
    selected = selected[:min(cutoff, max_panels)]

    if len(selected) < 2:
        return []

    # Safety cap
    if len(selected) > max_panels:
        return []

    # Convert surviving candidates to world-space panels
    panels = []
    for actual_m2, face_list, tot_a, anx, any_, anz in selected:
        # Area-weighted centroid (local)
        lcx = lcy = lcz = 0.0
        for fi in face_list:
            w = face_a[fi]; c = face_c[fi]
            lcx += c[0]*w; lcy += c[1]*w; lcz += c[2]*w
        lcx /= tot_a; lcy /= tot_a; lcz /= tot_a

        # Transform centroid and normal to world space
        wc  = world_xform.Transform(Gf.Vec3d(lcx, lcy, lcz))
        wnd = world_xform.TransformDir(Gf.Vec3d(anx, any_, anz))
        wnl = (wnd[0]**2 + wnd[1]**2 + wnd[2]**2) ** 0.5
        if wnl < 1e-10:
            continue
        wn = (wnd[0]/wnl, wnd[1]/wnl, wnd[2]/wnl)

        # In-plane extent: build (u, v) perpendicular to local mean normal
        if abs(anx) < 0.9:
            bx, by, bz = 1.0, 0.0, 0.0
        else:
            bx, by, bz = 0.0, 1.0, 0.0
        ux = any_*bz - anz*by; uy = anz*bx - anx*bz; uz = anx*by - any_*bx
        ul = (ux*ux + uy*uy + uz*uz) ** 0.5
        if ul < 1e-10:
            continue
        ux /= ul; uy /= ul; uz /= ul
        vx = any_*uz - anz*uy; vy = anz*ux - anx*uz; vz = anx*uy - any_*ux

        pu = [face_c[fi][0]*ux + face_c[fi][1]*uy + face_c[fi][2]*uz for fi in face_list]
        pv = [face_c[fi][0]*vx + face_c[fi][1]*vy + face_c[fi][2]*vz for fi in face_list]

        span_u_m = (max(pu) - min(pu)) * mpu
        span_v_m = (max(pv) - min(pv)) * mpu
        # Margin: mean face diameter so a single-face panel still has a non-zero size
        margin_m = math.sqrt(actual_m2 / max(len(face_list), 1))
        dim0_m   = max(span_u_m + margin_m, 0.05)
        dim1_m   = max(span_v_m + margin_m, 0.05)

        panels.append((wc[0], wc[1], wc[2], {
            'dim0_m':          dim0_m,
            'dim1_m':          dim1_m,
            'from_cluster':    True,
            'from_flat_panel': True,
            'bbox_ltype':      'RectLight',
            'panel_normal':    wn,
        }))

    return panels if len(panels) >= 2 else []


def find_fixture_centers_in_mesh(prim, mpu, up_axis="Z", bucket_size_local=None,
                                 world_xform_override=None):
    """
    Detect individual fixture positions within a merged mesh by clustering vertices.

    Strategy:
    1. Adaptively determine bucket size from mesh geometry (or use override)
    2. Bucket all vertices into a 2D grid in LOCAL space (floor plane)
       - Local space avoids issues with parent xforms that scale cm->m
    3. Flood-fill adjacent buckets to find connected clusters
    4. Transform cluster centroids to WORLD space for light placement

    Args:
        prim: The mesh prim containing merged fixture geometry
        mpu: meters per unit (used for world-space output)
        up_axis: "Z" or "Y" — determines which 2D plane to cluster on
        bucket_size_local: Override grid cell size. If None, auto-detected.

    Returns:
        List of (x, y, z, bbox_meta) in WORLD space (one per detected fixture).
        bbox_meta is a dict with 'dim0_m' and 'dim1_m' — the cluster's floor-plane extents
        in world-space metres (axis-0 = X or X, axis-1 = Y or Z depending on up_axis).
    """
    mesh = UsdGeom.Mesh(prim)
    if not mesh:
        return []

    points = mesh.GetPointsAttr().Get()
    if not points or len(points) < 10:
        return []

    # Auto-detect bucket size if not provided
    if bucket_size_local is None:
        bucket_size_local = estimate_bucket_size(points, up_axis)

    # Get world transform for final position output
    if world_xform_override is not None:
        world_xform = world_xform_override
    else:
        xformable = UsdGeom.Xformable(prim)
        world_xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    # Bucket vertices in LOCAL space on the floor plane
    buckets = {}  # (b0, b1) -> {'count': N, 'x_sum': F, 'y_sum': F, 'z_sum': F, ...}
    for p in points:
        if up_axis == "Z":
            b0 = int(round(p[0] / bucket_size_local))
            b1 = int(round(p[1] / bucket_size_local))
            fp0, fp1 = p[0], p[1]
        else:  # Y-up: cluster on XZ
            b0 = int(round(p[0] / bucket_size_local))
            b1 = int(round(p[2] / bucket_size_local))
            fp0, fp1 = p[0], p[2]

        if (b0, b1) not in buckets:
            buckets[(b0, b1)] = {"count": 0, "x_sum": 0.0, "y_sum": 0.0, "z_sum": 0.0,
                                  "z_min": float("inf"),
                                  "p0_min": float("inf"), "p0_max": float("-inf"),
                                  "p1_min": float("inf"), "p1_max": float("-inf")}
        bkt = buckets[(b0, b1)]
        bkt["count"] += 1
        bkt["x_sum"] += p[0]
        bkt["y_sum"] += p[1]
        bkt["z_sum"] += p[2]
        bkt["z_min"]  = min(bkt["z_min"],  p[2])
        bkt["p0_min"] = min(bkt["p0_min"], fp0)
        bkt["p0_max"] = max(bkt["p0_max"], fp0)
        bkt["p1_min"] = min(bkt["p1_min"], fp1)
        bkt["p1_max"] = max(bkt["p1_max"], fp1)

    # Filter: only keep buckets with significant vertex density
    if not buckets:
        return []

    avg_count = len(points) / len(buckets)
    # Threshold: at least 30% of average or 50 verts (lower floor for small meshes)
    threshold = max(avg_count * 0.3, 50)

    significant = {k: v for k, v in buckets.items() if v["count"] >= threshold}

    if not significant:
        # Fallback: just use the mesh center, transformed to world
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        center_local = Gf.Vec3d(
            sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)
        )
        center_world = world_xform.Transform(center_local)
        return [(center_world[0], center_world[1], center_world[2], None)]

    # Flood-fill to find connected clusters
    visited = set()
    fixture_centers = []

    def flood_fill(start):
        stack = [start]
        cluster = []
        while stack:
            pos = stack.pop()
            if pos in visited or pos not in significant:
                continue
            visited.add(pos)
            cluster.append(pos)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (pos[0] + dx, pos[1] + dy)
                    if neighbor not in visited and neighbor in significant:
                        stack.append(neighbor)
        return cluster

    for key in sorted(significant.keys()):
        if key not in visited:
            cluster = flood_fill(key)
            if cluster:
                # Weighted centroid XY in LOCAL space; Z = cluster minimum (bottom of fixture).
                # Using centroid Z placed pendant lamps mid-cable; min Z gives the bulb/panel face.
                total_verts = sum(significant[k]["count"] for k in cluster)
                cx_local = sum(significant[k]["x_sum"] for k in cluster) / total_verts
                cy_local = sum(significant[k]["y_sum"] for k in cluster) / total_verts
                cz_local = min(significant[k]["z_min"] for k in cluster)

                # Cluster local bounding box on the floor plane
                p0_min = min(significant[k]["p0_min"] for k in cluster)
                p0_max = max(significant[k]["p0_max"] for k in cluster)
                p1_min = min(significant[k]["p1_min"] for k in cluster)
                p1_max = max(significant[k]["p1_max"] for k in cluster)

                # Convert local spans to world metres via axis-endpoint transforms
                _wa = world_xform.Transform(Gf.Vec3d(p0_min, 0, 0))
                _wb = world_xform.Transform(Gf.Vec3d(p0_max, 0, 0))
                dim0_m = (_wb - _wa).GetLength() * mpu
                if up_axis == "Z":
                    _wa2 = world_xform.Transform(Gf.Vec3d(0, p1_min, 0))
                    _wb2 = world_xform.Transform(Gf.Vec3d(0, p1_max, 0))
                else:
                    _wa2 = world_xform.Transform(Gf.Vec3d(0, 0, p1_min))
                    _wb2 = world_xform.Transform(Gf.Vec3d(0, 0, p1_max))
                dim1_m = (_wb2 - _wa2).GetLength() * mpu

                # Four floor-plane corners of the cluster local bbox → world AABB.
                # Stored in metres so merge_aligned_cluster_pairs needs no mpu.
                if up_axis == "Z":
                    _bc = [Gf.Vec3d(p0_min,p1_min,cz_local), Gf.Vec3d(p0_max,p1_min,cz_local),
                           Gf.Vec3d(p0_min,p1_max,cz_local), Gf.Vec3d(p0_max,p1_max,cz_local)]
                else:
                    _bc = [Gf.Vec3d(p0_min,cz_local,p1_min), Gf.Vec3d(p0_max,cz_local,p1_min),
                           Gf.Vec3d(p0_min,cz_local,p1_max), Gf.Vec3d(p0_max,cz_local,p1_max)]
                _wc = [world_xform.Transform(c) for c in _bc]
                bbox_meta = {
                    'dim0_m': dim0_m, 'dim1_m': dim1_m,
                    'wxmin_m': min(c[0] for c in _wc) * mpu,
                    'wxmax_m': max(c[0] for c in _wc) * mpu,
                    'wymin_m': min(c[1] for c in _wc) * mpu,
                    'wymax_m': max(c[1] for c in _wc) * mpu,
                }

                # Transform to WORLD space
                world_pos = world_xform.Transform(
                    Gf.Vec3d(cx_local, cy_local, cz_local)
                )
                fixture_centers.append(
                    (world_pos[0], world_pos[1], world_pos[2], bbox_meta)
                )

    return fixture_centers


# ============================================================
# GLASS/WINDOW DETECTION — DOME LIGHT TRIGGER
# ============================================================
def check_glass_geometry(stage, bbox_cache, mpu):
    """
    Check if the scene has significant glass/window geometry.

    Uses three-step filtering:
    1. Name-based detection (glass/window/pane/skylight keywords)
    2. Prop context rejection (forklift, camera, extinguisher, lamp cover, etc.)
    3. Minimum size threshold — glass must be >= GLASS_MIN_AREA_M2 to count as a window

    Also checks material bindings for glass/transparent materials,
    but applies the same prop context and size filters.

    Returns (has_glass, glass_info) where glass_info is a summary string.
    """
    from pxr import UsdShade

    glass_prims = []

    # Collect glass material paths — but pre-filter out prop material names
    glass_material_paths = set()
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Material":
            name_lower = prim.GetName().lower()
            path_lower = str(prim.GetPath()).lower()
            combined = name_lower + " " + path_lower
            if any(k in combined for k in ["glass", "window", "transparent", "clear"]):
                # Skip prop materials
                if not any(ctx in combined for ctx in GLASS_PROP_CONTEXT):
                    glass_material_paths.add(str(prim.GetPath()))

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue

        # Method 1: name-based detection with prop filter + size check
        if is_glass(prim, bbox_cache=bbox_cache, mpu=mpu):
            glass_prims.append(prim)
            continue

        # Method 2: material binding detection (also size-checked)
        if glass_material_paths:
            try:
                binding = UsdShade.MaterialBindingAPI(prim)
                mat, _ = binding.ComputeBoundMaterial()
                if mat and str(mat.GetPath()) in glass_material_paths:
                    # Apply size check for material-matched glass too
                    path_lower = str(prim.GetPath()).lower()
                    if not any(ctx in path_lower for ctx in GLASS_PROP_CONTEXT):
                        prim_name_lower = prim.GetName().lower()
                        if "window" in prim_name_lower:
                            glass_prims.append(prim)  # window in name = always architectural
                        else:
                            try:
                                wb = bbox_cache.ComputeWorldBound(prim)
                                size = wb.ComputeAlignedRange().GetSize()
                                dims = sorted([abs(size[0]) * mpu, abs(size[1]) * mpu,
                                               abs(size[2]) * mpu])
                                if dims[2] >= GLASS_MIN_AREA_M2:  # largest dim >= 0.5 m
                                    glass_prims.append(prim)
                            except Exception:
                                pass
            except Exception:
                pass

    # Parent-Xform keyword pass: find Xform prims whose NAME contains glass keywords,
    # then add their descendant Gprim meshes. Handles NV_June scenes where "windowGlass_4"
    # (Xform) is the identifier, not the leaf mesh "NV_JuneMesh_406_____000".
    # Restricted to unambiguously-glass Xform names to avoid windowFrame / windowSeal / pillar.
    _GLASS_XFORM_KW = ["windowglass", "glasspanel", "glazing", "skylight",
                        "glassup", "glass_pane", "glassroof", "roofwindow"]
    _xform_glass_paths = {str(p.GetPath()) for p in glass_prims}
    for _gxp in stage.Traverse():
        if not _gxp.IsA(UsdGeom.Xform):
            continue
        _gxpname = _gxp.GetName().lower()
        _gxppath = str(_gxp.GetPath()).lower()
        if not any(k in _gxpname for k in _GLASS_XFORM_KW):
            continue
        if any(ctx in _gxppath for ctx in GLASS_PROP_CONTEXT):
            continue
        _gprim_found_g = False
        for _gdesc in Usd.PrimRange(_gxp):
            if _gdesc == _gxp:
                continue
            if not _gdesc.IsA(UsdGeom.Gprim):
                continue
            _gdpath = str(_gdesc.GetPath())
            if _gdpath in _xform_glass_paths:
                continue
            # Apply size check when bbox is available
            try:
                _gwb = bbox_cache.ComputeWorldBound(_gdesc)
                _gsize = _gwb.ComputeAlignedRange().GetSize()
                _gdims = sorted([abs(_gsize[0]) * mpu, abs(_gsize[1]) * mpu,
                                 abs(_gsize[2]) * mpu])
                if _gdims[2] < GLASS_MIN_AREA_M2:
                    continue
            except Exception:
                pass
            _xform_glass_paths.add(_gdpath)
            glass_prims.append(_gdesc)
            _gprim_found_g = True
        # Fallback: instance children (USD-instanced glass, e.g. NV_June windowGlass_1 Xform)
        if not _gprim_found_g:
            for _gchild in _gxp.GetChildren():
                if not _gchild.IsInstance():
                    continue
                _gcpath = str(_gchild.GetPath())
                if _gcpath in _xform_glass_paths:
                    continue
                # Size check using BBoxCache on the instance
                try:
                    _gwb = bbox_cache.ComputeWorldBound(_gchild)
                    _grng = _gwb.ComputeAlignedRange()
                    if not _grng.IsEmpty():
                        _gsize = _grng.GetSize()
                        _gdims = sorted([abs(_gsize[0]) * mpu, abs(_gsize[1]) * mpu,
                                         abs(_gsize[2]) * mpu])
                        if _gdims[2] < GLASS_MIN_AREA_M2:
                            continue
                except Exception:
                    pass
                _xform_glass_paths.add(_gcpath)
                glass_prims.append(_gchild)

    if not glass_prims:
        return False, "No glass/window geometry detected (prop glass filtered out)."

    info_parts = []
    for gp in glass_prims:
        mesh = UsdGeom.Mesh(gp)
        if mesh:
            points = mesh.GetPointsAttr().Get()
            vert_count = len(points) if points else 0
            info_parts.append(f"{gp.GetName()} ({vert_count} verts)")
        else:
            info_parts.append(gp.GetName())

    return True, f"Glass/window geometry found: {', '.join(info_parts)}"


# ============================================================
# POWER CALCULATION
# ============================================================
def _get_UF(room_index):
    """Utilization factor from Room Index lookup."""
    if room_index < 1:   return 0.35
    elif room_index < 2: return 0.45
    elif room_index < 3: return 0.55
    else:                return 0.65


def _room_index(L, W, h_mount):
    """Standard Room Index formula."""
    if h_mount > 0 and (L + W) > 0:
        return (L * W) / (h_mount * (L + W))
    return 1.0


def _prim_diffuse_info(prim):
    """
    Return (luminance, tier) or None.
    Tries bound material shader first, then displayColor.
    luminance is CIE Y [0..1] from linear sRGB.
    """
    from pxr import UsdShade
    try:
        mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if mat:
            surface, _, _ = mat.ComputeSurfaceSource()
            if surface:
                for name in ('diffuseColor', 'albedo', 'baseColor', 'albedoColor',
                             'diffuse_reflection_color', 'diffuse_color'):
                    inp = surface.GetInput(name)
                    if inp and inp.IsValid():
                        val = inp.Get()
                        if val is not None:
                            try:
                                r, g, b = float(val[0]), float(val[1]), float(val[2])
                                return (0.2126*r + 0.7152*g + 0.0722*b, 'material')
                            except Exception:
                                pass
    except Exception:
        pass
    for attr_name in ('primvars:displayColor', 'displayColor'):
        attr = prim.GetAttribute(attr_name)
        if attr and attr.IsValid():
            val = attr.Get()
            if val is not None:
                try:
                    c = val[0] if (hasattr(val, '__len__') and
                                   not isinstance(val, (int, float))) else val
                    r, g, b = float(c[0]), float(c[1]), float(c[2])
                    return (0.2126*r + 0.7152*g + 0.0722*b, 'displayColor')
                except Exception:
                    pass
    return None


def _sample_scene_reflectances(stage, up_axis, combined, scene_type="industrial"):
    """
    Estimate ceiling / wall / floor reflectances using a 3-tier fallback:
      Tier 1 — bound material shader diffuseColor / albedo (most accurate)
      Tier 2 — USD displayColor attribute (no full materials)
      Tier 3 — conservative scene-type defaults (no color data at all)

    Classifies the 30 largest Gprims by bounding-box footprint area into
    ceiling (top 20% of height), floor (bottom 20%), or wall (middle 60%)
    zones, then averages the luminance per zone.

    Returns (ceiling_r, wall_r, floor_r, tier_str, detail_lines).
    """
    DEFAULTS = {
        "industrial":  (0.50, 0.35, 0.10),
        "warehouse":   (0.50, 0.35, 0.10),
        "office":      (0.70, 0.55, 0.25),
        "retail":      (0.70, 0.55, 0.30),
        "residential": (0.65, 0.50, 0.20),
    }
    def_c, def_w, def_f = DEFAULTS.get(scene_type, (0.50, 0.40, 0.15))

    if combined.IsEmpty():
        return def_c, def_w, def_f, 'default', \
               [f"No geometry — using {scene_type} scene defaults."]

    h_axis = 2 if up_axis == "Z" else 1
    h_min  = combined.GetMin()[h_axis]
    h_max  = combined.GetMax()[h_axis]
    h_range = h_max - h_min
    if h_range <= 0:
        return def_c, def_w, def_f, 'default', \
               [f"No height range — using {scene_type} scene defaults."]

    floor_top = h_min + h_range * 0.20
    ceil_bot  = h_max - h_range * 0.20

    _bc = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=True)
    candidates = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        try:
            wb  = _bc.ComputeWorldBound(prim)
            rng = wb.ComputeAlignedRange()
            if rng.IsEmpty():
                continue
            sz   = rng.GetSize()
            area = sz[0] * sz[1] if up_axis == "Z" else sz[0] * sz[2]
            h_ctr = (rng.GetMin()[h_axis] + rng.GetMax()[h_axis]) * 0.5
            candidates.append((area, h_ctr, prim))
        except Exception:
            pass
        if len(candidates) >= 500:
            break

    candidates.sort(key=lambda x: x[0], reverse=True)
    zones = {'ceiling': [], 'wall': [], 'floor': []}
    for area, h_ctr, prim in candidates[:30]:
        if   h_ctr >= ceil_bot:  zones['ceiling'].append(prim)
        elif h_ctr <= floor_top: zones['floor'].append(prim)
        else:                    zones['wall'].append(prim)

    def _zone_avg(prims):
        mat_v, dc_v = [], []
        for p in prims:
            info = _prim_diffuse_info(p)
            if info:
                lum, t = info
                (mat_v if t == 'material' else dc_v).append(lum)
        if mat_v: return sum(mat_v)/len(mat_v), 'material',     len(mat_v)
        if dc_v:  return sum(dc_v) /len(dc_v),  'displayColor', len(dc_v)
        return None, None, 0

    c_val, c_t, c_n = _zone_avg(zones['ceiling'])
    w_val, w_t, w_n = _zone_avg(zones['wall'])
    f_val, f_t, f_n = _zone_avg(zones['floor'])

    c_r = c_val if c_val is not None else def_c
    w_r = w_val if w_val is not None else def_w
    f_r = f_val if f_val is not None else def_f

    n_mat = sum(1 for t in (c_t, w_t, f_t) if t == 'material')
    n_any = sum(1 for t in (c_t, w_t, f_t) if t is not None)

    if n_mat >= 2:
        tier   = 'material'
        source = f"material colors sampled from {c_n+w_n+f_n} surface shader(s)"
    elif n_any >= 2:
        tier   = 'displayColor'
        source = f"displayColor from {c_n+w_n+f_n} prim(s) (no full materials — less accurate)"
    else:
        tier   = 'default'
        c_r, w_r, f_r = def_c, def_w, def_f
        source = f"no color data found — using conservative {scene_type} defaults"

    detail = [
        f"Reflectance source: {source}.",
        f"Assumed reflectances — ceiling: {c_r:.0%}  walls: {w_r:.0%}  floor: {f_r:.0%}",
    ]
    return c_r, w_r, f_r, tier, detail


def _uf_reflectance_adjusted(ri, ceiling_r, wall_r, floor_r):
    """
    Scale the RI-based UF by the ratio of actual surface reflectances to the
    standard reference values (ceiling 70%, walls 50%, floor 20%) that the
    basic RI lookup table assumes.

    Returns (adjusted_UF, scale_factor).
    """
    base_UF = _get_UF(ri)
    ref = 0.50 * 0.70 + 0.35 * 0.50 + 0.15 * 0.20   # = 0.555 (standard reference)
    act = 0.50 * ceiling_r + 0.35 * wall_r + 0.15 * floor_r
    scale = (act / ref) if ref > 0 else 1.0
    scale = max(0.30, min(1.50, scale))
    return round(base_UF * scale, 3), round(scale, 3)


def _scene_complexity_factor(scene_type, room_index):
    """
    Empirical multiplier for the gap between the analytical lumen method
    (empty rectangular room) and RTX path tracing (geometry occlusion,
    machinery blocking inter-reflections, realistic light absorption).

    Applied on top of the reflectance scale. Derived from real scene tests;
    the measured_lux calibration loop closes any remaining gap.
    """
    TABLE = [
        # (scene_type_keyword, ri_max, factor)
        # Dense industrial: machinery heavily blocks inter-reflections
        ("industrial",  1.0, 1.75),
        ("industrial",  2.0, 1.40),
        ("industrial",  9.9, 1.20),
        # Warehouse: tall space, more open but long light path
        ("warehouse",   1.0, 1.50),
        ("warehouse",   2.0, 1.25),
        ("warehouse",   9.9, 1.10),
        # Lighter / more open scene types
        ("office",      9.9, 1.10),
        ("retail",      9.9, 1.12),
        ("residential", 9.9, 1.10),
    ]
    st = (scene_type or "").lower()
    for keyword, ri_max, factor in TABLE:
        if keyword in st and room_index <= ri_max:
            return factor
    return 1.20  # conservative default for unknown scene types


# ============================================================
# SCHEMA APPLICATION HELPERS
# ============================================================
def _deactivate_and_hide(prim):
    """
    Deactivate a light AND hide it from the viewport.

    Two separate USD operations, both written to the current edit target
    (the lighting sublayer):
    - SetActive(False)          → stops light computation / rendering
    - MakeInvisible()           → hides the gizmo/icon in the viewport

    SetActive alone leaves the light icon visible. MakeInvisible alone
    would hide it visually but not stop it rendering. Both together give
    the clean 'off and hidden' result.
    """
    prim.SetActive(False)
    UsdGeom.Imageable(prim).MakeInvisible()


def apply_area_light_schemas(prim, power, color_temp=4000.0, illuminant="illuminantD"):
    """Apply PhotometricAreaLightAPI + PhysicalLightIlluminantAPI and set values."""
    prim.AddAppliedSchema("PhotometricAreaLightAPI")
    prim.AddAppliedSchema("PhysicalLightIlluminantAPI")

    ver_attr = prim.GetAttribute("omni:rtx:usdluxVersion")
    if not ver_attr.IsValid():
        ver_attr = prim.CreateAttribute("omni:rtx:usdluxVersion", Sdf.ValueTypeNames.Int)
    ver_attr.Set(2505)

    int_attr = prim.GetAttribute("inputs:intensity")
    if int_attr.IsValid():
        int_attr.Set(1.0)
    exp_attr = prim.GetAttribute("inputs:exposure")
    if exp_attr.IsValid():
        exp_attr.Set(0.0)

    pw_attr = prim.GetAttribute("photometric:power")
    if not pw_attr.IsValid():
        pw_attr = prim.CreateAttribute("photometric:power", Sdf.ValueTypeNames.Float)
    pw_attr.Set(float(power))

    dist_attr = prim.GetAttribute("photometric:illuminance:distance")
    if not dist_attr.IsValid():
        dist_attr = prim.CreateAttribute(
            "photometric:illuminance:distance", Sdf.ValueTypeNames.Float
        )
    dist_attr.Set(0.0)

    ill_attr = prim.GetAttribute("physical:illuminant")
    if not ill_attr.IsValid():
        ill_attr = prim.CreateAttribute("physical:illuminant", Sdf.ValueTypeNames.Token)
    ill_attr.Set(illuminant)

    ct_attr = prim.GetAttribute("physical:colorTemperature")
    if not ct_attr.IsValid():
        ct_attr = prim.CreateAttribute(
            "physical:colorTemperature", Sdf.ValueTypeNames.Float
        )
    ct_attr.Set(float(color_temp))


def apply_dome_light_schemas(prim, illuminance=10000.0, color_temp=6500.0,
                             set_version=True):
    """Apply PhotometricDomeLightAPI + PhysicalLightIlluminantAPI for dome lights.

    set_version: if True, sets omni:rtx:usdluxVersion = 2505.
                 Set False when upgrading an EXISTING DomeLight — changing the
                 UsdLux version alters how the HDRI texture is mapped, which
                 rotates the sky even if xformOps haven't changed.
                 Only set True when CREATING a new DomeLight from scratch.
    """
    prim.AddAppliedSchema("PhotometricDomeLightAPI")
    prim.AddAppliedSchema("PhysicalLightIlluminantAPI")

    if set_version:
        ver_attr = prim.GetAttribute("omni:rtx:usdluxVersion")
        if not ver_attr.IsValid():
            ver_attr = prim.CreateAttribute("omni:rtx:usdluxVersion", Sdf.ValueTypeNames.Int)
        ver_attr.Set(2505)

    int_attr = prim.GetAttribute("inputs:intensity")
    if int_attr.IsValid():
        int_attr.Set(1.0)
    exp_attr = prim.GetAttribute("inputs:exposure")
    if exp_attr.IsValid():
        exp_attr.Set(0.0)

    ill_attr = prim.GetAttribute("photometric:illuminance")
    if not ill_attr.IsValid():
        ill_attr = prim.CreateAttribute(
            "photometric:illuminance", Sdf.ValueTypeNames.Float
        )
    ill_attr.Set(float(illuminance))

    il_attr = prim.GetAttribute("physical:illuminant")
    if not il_attr.IsValid():
        il_attr = prim.CreateAttribute("physical:illuminant", Sdf.ValueTypeNames.Token)
    il_attr.Set("illuminantD")

    ct_attr = prim.GetAttribute("physical:colorTemperature")
    if not ct_attr.IsValid():
        ct_attr = prim.CreateAttribute(
            "physical:colorTemperature", Sdf.ValueTypeNames.Float
        )
    ct_attr.Set(float(color_temp))


# ============================================================
# CAMERA EXPOSURE DIAGNOSTIC
# ============================================================
def _camera_ev(iso, fstop, exposure_time):
    """
    Calculate Exposure Value from camera settings.
    EV = log2(f² / t) adjusted for ISO.
    EV 10 = bright outdoor sun. EV 7 = indoor 500 lux. EV 5 = dim indoor.
    """
    import math
    if exposure_time <= 0 or fstop <= 0 or iso <= 0:
        return None
    ev_base = math.log2((fstop ** 2) / exposure_time)
    ev_iso  = math.log2(iso / 100.0)
    return ev_base - ev_iso   # effective EV at this ISO


def _target_ev(lux):
    """
    EV needed for correct exposure at a given illuminance (lux).
    Based on incident light metering: EV = log2(lux / 2.5).
    """
    import math
    return math.log2(max(lux, 1) / 2.5)


def _recommend_camera(lux):
    """
    Return recommended camera settings (ISO, f-stop, exposure_time) for
    a physically lit scene at the given floor illuminance.

    Presets cover typical indoor ranges. The goal is EV ≈ log2(lux/2.5).
    Calibrated for Omniverse RTX with default responsivity ~1.1 (empirically
    validated: 250 lux warehouse scene correct at ISO 400, f/2.8, 1/60s).
    All ISO values are ~2 stops lower than standard photography tables to
    account for RTX tone-mapper defaults.
    """
    # (lux_min, lux_max, iso, fstop, exp_time, label)
    PRESETS = [
        (  1,   30, 1600, 2.8, 1/30,  "very dim indoor / emergency lighting"),
        ( 30,  100,  800, 2.8, 1/30,  "dim indoor / corridor"),
        (100,  300,  400, 2.8, 1/60,  "normal indoor / warehouse / office"),
        (300,  750,  200, 2.8, 1/60,  "bright indoor / retail / studio"),
        (750, 2000,  100, 2.8, 1/60,  "very bright indoor / TV studio"),
        (2000, 1e9,  100, 5.6, 1/125, "near-outdoor / sunlit atrium"),
    ]
    for lo, hi, iso, fstop, exp, label in PRESETS:
        if lo <= lux < hi:
            return iso, fstop, exp, label
    return 800, 2.8, 1/60, "normal indoor"


def _log_scene_camera(stage, log_fn):
    """
    Read the bound camera from the scene's customLayerData and log its
    prim path plus any authored exposure attributes (ISO, f-stop, time).
    """
    cam_path = None
    try:
        root = stage.GetRootLayer()
        cd = root.customLayerData
        cam_path = (cd.get("cameraSettings") or {}).get("boundCamera")
    except Exception:
        pass

    if not cam_path:
        log_fn("  Scene camera:   (no bound camera found in customLayerData)")
        return

    log_fn(f"  Scene camera:   {cam_path}")

    cam_prim = stage.GetPrimAtPath(cam_path)
    if not cam_prim or not cam_prim.IsValid():
        log_fn("  (camera prim not found on stage)")
        return

    def _get(attr_name):
        a = cam_prim.GetAttribute(attr_name)
        return a.Get() if a and a.IsValid() else None

    iso   = _get("exposure:iso")
    fstop = _get("exposure:fStop")
    exp   = _get("exposure:time")

    if iso or fstop or exp:
        log_fn("  Camera exposure attributes (from prim):")
        if iso   is not None: log_fn(f"    Film ISO:      {iso}")
        if fstop is not None: log_fn(f"    F-stop:        f/{fstop}")
        if exp   is not None: log_fn(f"    Exposure time: {exp:.4f}s  (1/{round(1/exp) if exp else '?'})")
    else:
        log_fn("  (no exposure:iso / exposure:fStop / exposure:time on camera prim)")


def _log_camera_recommendations(target_lux, log_fn):
    """
    Read current Omniverse tone mapping settings (if in Kit), compare
    against what's needed for target_lux, and log recommendations.

    IMPORTANT: If the scene looks dark after running this skill, do NOT
    increase light power — the physics values are calibrated. Adjust the
    camera/tonemapping settings instead.
    """
    import math

    log_fn("\n" + "-" * 60)
    log_fn("  CAMERA EXPOSURE CHECK")
    log_fn("-" * 60)
    log_fn(f"  If the scene looks too dark or too bright, adjust the")
    log_fn(f"  camera/tonemapping settings — do NOT change light power.")
    log_fn(f"  The physical lumen values are calibrated for {target_lux} lux.")

    # Try to read current settings from Kit's carb settings
    current_iso = current_fstop = current_exp = None
    try:
        import carb.settings as _cs
        s = _cs.get_settings()
        current_iso  = s.get("/rtx/post/tonemap/filmIso")
        current_fstop = s.get("/rtx/post/tonemap/fStop")
        current_exp  = s.get("/rtx/post/tonemap/cameraExposureTime")
    except Exception:
        pass

    if current_iso and current_fstop and current_exp:
        ev_current = _camera_ev(current_iso, current_fstop, current_exp)
        ev_target  = _target_ev(target_lux)
        delta = ev_target - ev_current if ev_current is not None else None

        log_fn(f"\n  Current camera settings detected:")
        log_fn(f"    Film ISO:      {current_iso}")
        log_fn(f"    F-stop:        f/{current_fstop}")
        log_fn(f"    Exposure time: {current_exp:.4f}s  (1/{round(1/current_exp)})")
        if ev_current is not None:
            log_fn(f"    Effective EV:  {ev_current:.1f}")

        if delta is not None:
            if delta < -0.5:
                stops = abs(delta)
                log_fn(f"\n  WARNING: Camera is {stops:.1f} stops OVEREXPOSED for {target_lux} lux.")
                log_fn(f"  The scene will look blown out. Reduce ISO or close the aperture.")
            elif delta > 0.5:
                stops = delta
                log_fn(f"\n  WARNING: Camera is {stops:.1f} stops UNDEREXPOSED for {target_lux} lux.")
                log_fn(f"  This is why the scene looks dark — the lighting is correct.")
                log_fn(f"  Do NOT increase light power. Adjust camera settings instead.")
            else:
                log_fn(f"\n  Camera exposure looks correctly calibrated for {target_lux} lux.")
    else:
        log_fn(f"\n  (Camera settings not readable — running outside Kit or settings unavailable)")

    # Always output recommended settings regardless
    rec_iso, rec_fstop, rec_exp, rec_label = _recommend_camera(target_lux)
    ev_rec = _camera_ev(rec_iso, rec_fstop, rec_exp)
    log_fn(f"\n  Recommended settings for {target_lux} lux ({rec_label}):")
    log_fn(f"    Render Settings -> Post Processing -> Tone Mapping:")
    log_fn(f"      Film ISO:              {rec_iso}")
    log_fn(f"      F-stop:                f/{rec_fstop}")
    log_fn(f"      Camera Exposure Time:  {rec_exp:.4f}s  (1/{round(1/rec_exp)})")
    if ev_rec is not None:
        log_fn(f"      → Effective EV: {ev_rec:.1f}")
    log_fn(f"\n  For Path Tracing: same settings apply.")
    log_fn(f"  Validate with: Debug View -> PT AOV Illuminance")
    log_fn(f"  Target reading on floor surfaces: {target_lux} lux ± 20%")
    log_fn("-" * 60)


# ============================================================
# CORE SKILL — works in Kit, CLI, or any Python with pxr
# ============================================================
def run_skill(stage,
              dry_run=True,
              color_temp=4000.0,
              dome_color_temp=6500.0,
              dome_illuminance=0.0,
              no_dome=False,
              force_dome=False,
              target_lux=None,
              output_layer=LIGHTING_LAYER_NAME,
              output_file=None,
              measured_lux=None,
              reflectances=None,
              window_size=None,
              dome_contribution=None):
    """
    Run the UsdLux Physical Lighting Skill on an already-open stage.

    Works anywhere pxr is available:
    - Kit Script Editor: pass omni.usd.get_context().get_stage()
    - CLI: called by main() after opening the stage from a file path
    - VM agent: same as CLI

    Args:
        stage:           Usd.Stage — the scene to process
        dry_run:         If True, analyze and report only — no files written
        color_temp:      Kelvin for fixture lights (default 4000K)
        dome_color_temp: Kelvin for dome light (default 6500K daylight)
        dome_illuminance: Lux for dome light (0 = auto: 400 lux sunset/clear sky; RTX handles glass transmission physically — fixtures always target full lux)
        no_dome:         Skip DomeLight even if windows detected
        force_dome:      Create DomeLight even without windows
        window_size:     'small'|'medium'|'large' — hint for indoor daylight contribution
                         when force_dome=True. Reduces fixture target by estimated
                         contribution: small≈16 lux, medium≈48 lux, large≈96 lux.
        dome_contribution: Direct override in lux for indoor daylight contribution
                         (takes precedence over window_size).
        target_lux:      Override auto-detected illuminance target
        output_layer:    Filename for the lighting sublayer
        output_file:     Optional path to write full log output (useful in Kit)
        measured_lux:    Measured PT AOV Illuminance reading from Omniverse viewport.
                         When provided, skips fixture detection and scales all
                         photometric:power values in the existing lighting layer by
                         (target_lux / measured_lux) for one-step calibration.
        reflectances:    Optional tuple (ceiling_r, wall_r, floor_r) to override
                         automatic surface reflectance sampling. Values 0.0–1.0.
                         Useful when materials are MDL-only and tier-1/2 sampling
                         returns scene-type defaults.

    Returns:
        dict with keys: fixture_count, light_count, power_lm, dome_created
    """
    lighting_layer_name = output_layer
    log_lines = []

    def log(msg=""):
        log_lines.append(str(msg))
        print(msg)

    log("\n" + "=" * 60)
    log("  UsdLux Physical Lighting Skill v4.3")
    log("  • Adaptive merged-mesh fixture clustering")
    log("  • Glass/window -> DomeLight auto-detection")
    log("  • PointInstancer support  • Y-up and Z-up scenes")
    log(f"  DRY_RUN = {dry_run}")
    log("=" * 60)

    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    up_axis = UsdGeom.GetStageUpAxis(stage) or "Z"
    log(f"  metersPerUnit: {mpu}  upAxis: {up_axis}")
    log(f"  sublayers: {stage.GetRootLayer().subLayerPaths}")

    # --- Step 0: Validate scale ---
    log("\n=== Step 0: Scale Validation ===")
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=True
    )
    combined = Gf.Range3d()
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Gprim):
            wb = bbox_cache.ComputeWorldBound(prim)
            combined = Gf.Range3d.GetUnion(combined, wb.ComputeAlignedRange())

    if combined.IsEmpty():
        log("  ERROR: No geometry found!")
        return {"error": "no_geometry"}

    size_usd = combined.GetSize()
    size_m = [size_usd[i] * mpu for i in range(3)]
    max_dim = max(size_m)
    threshold_m = proximity_threshold(max_dim)
    log(f"  Scene bounds (m): {size_m[0]:.1f} x {size_m[1]:.1f} x {size_m[2]:.1f}")
    log(f"  Largest dimension: {max_dim:.1f} m  Proximity threshold: {threshold_m:.2f} m")

    if max_dim < 0.5:
        log("  HALT: Scene is too small — likely a unit mismatch!")
        return {"error": "scale_too_small"}

    # Check for unitsResolve xform
    units_resolve_detected = False
    for prim in stage.Traverse():
        xf = UsdGeom.Xformable(prim)
        if not xf:
            continue
        for op in xf.GetOrderedXformOps():
            if "unitsResolve" in op.GetOpName():
                scale_val = op.Get()
                if scale_val is not None:
                    s = scale_val[0] if hasattr(scale_val, '__getitem__') else float(scale_val)
                    if abs(s - 1.0) > 1e-6:
                        units_resolve_detected = True
                        log(f"  unitsResolve on '{prim.GetPath()}': scale = {scale_val}")
                        if abs(s - 0.01) < 1e-6:
                            log(f"  -> Geometry in CENTIMETERS, scaled to meters by unitsResolve.")
                        break
        if units_resolve_detected:
            break

    if units_resolve_detected:
        log("  Fixture clustering works in local vertex space, outputs world space.")
    else:
        log("  No unitsResolve transforms — consistent units throughout.")
    log("  Scale looks correct.\n")

    # --- Create/open lighting layer ---
    log("=== Step 2: Lighting Layer ===")
    root_layer = stage.GetRootLayer()
    root_dir = os.path.dirname(root_layer.realPath)

    sublayer_folder = find_sublayer_folder(root_layer)
    if sublayer_folder:
        layer_rel_path = sublayer_folder + "/" + lighting_layer_name
        layer_abs_path = os.path.join(root_dir, sublayer_folder, lighting_layer_name)
        log(f"  Detected sublayer folder: {sublayer_folder}/")
    else:
        layer_rel_path = lighting_layer_name
        layer_abs_path = os.path.join(root_dir, lighting_layer_name)

    # --- Calibration mode: scale existing lights, skip fixture scanning ---
    if measured_lux is not None and measured_lux > 0:
        _cal_scene_type = infer_scene_type(stage)
        _cal_target     = target_lux or TARGET_LUX.get(_cal_scene_type, 300)
        _cal_scale      = _cal_target / measured_lux
        log(f"\n  CALIBRATION MODE")
        log(f"  Measured illuminance: {measured_lux:.0f} lux")
        log(f"  Target illuminance:   {_cal_target:.0f} lux")
        log(f"  Scale factor:         {_cal_scale:.3f}× applied to all fixture lights")
        log(f"  (DomeLights are sky illuminance — not scaled)")

        if dry_run:
            log(f"  [DRY RUN] Would scale lights by {_cal_scale:.3f}×")
            log("=" * 60 + "\n")
            return {"calibrated_dry_run": True, "scale": _cal_scale,
                    "target_lux": _cal_target}

        if not os.path.exists(layer_abs_path):
            log(f"  ERROR: No lighting layer at {layer_abs_path}")
            log(f"  Run the skill normally first, then re-run with measured_lux=<reading>.")
            log("=" * 60 + "\n")
            return {"error": "no_lighting_layer"}

        # Use Sdf.Layer directly — photometric:power is a custom attribute
        # from PhotometricAreaLightAPI (Omniverse extension) which isn't
        # registered in standalone Python, so Usd.Stage attribute lookup
        # won't find it. Sdf reads raw property values regardless of schema.
        _cal_layer = Sdf.Layer.FindOrOpen(layer_abs_path)
        if not _cal_layer:
            log(f"  ERROR: Could not open {layer_abs_path}")
            log("=" * 60 + "\n")
            return {"error": "cannot_open_layer"}

        _cal_scaled = 0
        _cal_skipped_dome = 0

        def _visit_cal(path):
            nonlocal _cal_scaled, _cal_skipped_dome
            if not path.IsPrimPropertyPath():
                return
            prop_name = path.name
            if prop_name == 'photometric:power':
                spec = _cal_layer.GetObjectAtPath(path)
                if isinstance(spec, Sdf.AttributeSpec):
                    cur = spec.default
                    if cur is not None and cur > 0:
                        new_val = float(round(cur * _cal_scale / 100) * 100)
                        spec.default = new_val
                        log(f"    {path.GetParentPath().name}: "
                            f"{cur:.0f} → {new_val:.0f} lm")
                        _cal_scaled += 1
            elif prop_name == 'photometric:illuminance':
                _cal_skipped_dome += 1

        _cal_layer.Traverse(Sdf.Path('/'), _visit_cal)
        _cal_layer.Save()
        root_layer.Save()

        log(f"\n  Scaled {_cal_scaled} fixture light(s). Layer saved.")
        if _cal_skipped_dome:
            log(f"  Skipped {_cal_skipped_dome} DomeLight(s) — sky illuminance unchanged.")
        log(f"  Re-check PT AOV Illuminance. Target: {_cal_target:.0f} lux ± 20%.")
        log("=" * 60 + "\n")
        return {"calibrated": _cal_scaled, "scale": _cal_scale,
                "target_lux": _cal_target}

    if dry_run:
        log(f"  [DRY RUN] Would create/clear: {layer_rel_path}\n")
    else:
        layer_dir = os.path.dirname(layer_abs_path)
        if layer_dir and not os.path.exists(layer_dir):
            os.makedirs(layer_dir, exist_ok=True)

        if os.path.exists(layer_abs_path):
            lighting_layer = Sdf.Layer.FindOrOpen(layer_abs_path)
            lighting_layer.Clear()
            log(f"  Cleared and reopened: {layer_rel_path}")
        else:
            lighting_layer = Sdf.Layer.CreateNew(layer_abs_path)
            log(f"  Created: {layer_rel_path}")

        if layer_rel_path not in root_layer.subLayerPaths:
            for existing in list(root_layer.subLayerPaths):
                if lighting_layer_name in existing:
                    root_layer.subLayerPaths.remove(existing)
            root_layer.subLayerPaths.insert(0, layer_rel_path)
            log(f"  Added as sublayer[0]: {layer_rel_path}")

        stage.SetEditTarget(lighting_layer)
        log("  Edit target -> lighting layer\n")

    # --- Fixture detection with vertex clustering ---
    log("=== Fixture Detection (Vertex Clustering) ===")
    fixtures_prims = [
        p for p in stage.Traverse() if p.IsA(UsdGeom.Gprim) and is_fixture(p)
    ]

    # Emissive material pass — catches fixtures not identified by name/path conventions.
    # Size guard (5 m max dimension) prevents floors, walls, and large decorative surfaces
    # from being treated as fixture geometry.
    # Keyword guard: prim name OR bound material name must contain a light-related term so
    # that props with incidental emissive materials (cardboard boxes, pallets, rack shelves,
    # floor markings) are not mistaken for light fixtures.
    _EMISSIVE_LIGHT_KW = (
        "lamp", "light", "neon", "led", "glow", "fluoro",
        "luminaire", "emitter", "fixture", "bulb", "lantern",
    )
    named_paths = {str(p.GetPath()) for p in fixtures_prims}
    emissive_found = []
    for p in stage.Traverse():
        if not p.IsA(UsdGeom.Gprim):
            continue
        if str(p.GetPath()) in named_paths:
            continue
        # Fast keyword pre-filter before the expensive shader walk
        _pname = p.GetName().lower()
        _ppath = str(p.GetPath()).lower()
        # Respect hard false positives even when an emissive keyword is present
        # (e.g. sm_fusebox_a03_led01 has "led" but its parent is a fusebox).
        if any(fp in _pname or fp in _ppath for fp in HARD_FALSE_POSITIVES):
            continue
        try:
            from pxr import UsdShade as _UsdShade
            _mat, _ = _UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()
            _mname = _mat.GetPrim().GetName().lower() if _mat else ""
        except Exception:
            _mname = ""
        if not any(kw in _pname or kw in _mname for kw in _EMISSIVE_LIGHT_KW):
            continue
        if not has_emissive_material(p):
            continue
        mesh = UsdGeom.Mesh(p)
        if mesh:
            pts = mesh.GetPointsAttr().Get()
            if pts and len(pts) >= 3:
                xf = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                world_pts = [xf.Transform(Gf.Vec3d(pt[0], pt[1], pt[2])) for pt in pts[:200]]
                max_dim = max(
                    (max(v[i] for v in world_pts) - min(v[i] for v in world_pts)) * mpu
                    for i in range(3)
                )
                if max_dim > 5.0:
                    continue
        emissive_found.append(p)
        log(f"  Emissive material fixture: {p.GetPath()}")
    if emissive_found:
        log(f"  -> {len(emissive_found)} additional fixture(s) from emissive materials")
    fixtures_prims = fixtures_prims + emissive_found

    # Parent-Xform keyword pass: find Xform prims whose NAME contains fixture keywords,
    # then add their descendant Gprim meshes. Handles scenes (e.g. NV_June) where the
    # meaningful keyword lives in the container Xform name ("lightEmitter_6") rather than
    # in the generic leaf mesh name ("NV_JuneMesh_462_____000").
    # Also handles USD-instanced geometry: when Gprim descendants are hidden behind instance
    # prims (IsInstance=True), the instance Xforms themselves are added — their world position
    # is valid and the processing loop falls back to BBoxCache centroid for them.
    # Guard: only run when previous passes found nothing — avoids double-counting in scenes
    # where the normal Gprim keyword pass already detects everything correctly.
    if not fixtures_prims:
        _xform_fix_paths = set()
        _xform_fix = []
        for _xp in stage.Traverse():
            if not _xp.IsA(UsdGeom.Xform):
                continue
            _xpname = _xp.GetName().lower()
            if any(fp in _xpname for fp in HARD_FALSE_POSITIVES):
                continue
            if not any(k in _xpname for k in EXPLICIT_FIXTURE_KEYWORDS):
                continue
            # Try Gprim descendants first (non-instanced scenes)
            _gprim_found = False
            for _desc in Usd.PrimRange(_xp):
                if _desc == _xp:
                    continue
                if not _desc.IsA(UsdGeom.Gprim):
                    continue
                _dpath = str(_desc.GetPath())
                if _dpath in _xform_fix_paths:
                    continue
                _xform_fix_paths.add(_dpath)
                _xform_fix.append(_desc)
                _gprim_found = True
            # Fallback: if no Gprim descendants (USD-instanced), add direct instance children.
            # Instance Xforms carry correct world position; BBoxCache centroid used in loop.
            # Size guard: require max BBox dimension >= 0.15 m to skip tiny machine indicator LEDs.
            if not _gprim_found:
                for _child in _xp.GetChildren():
                    if not _child.IsInstance():
                        continue
                    _cpath = str(_child.GetPath())
                    if _cpath in _xform_fix_paths:
                        continue
                    try:
                        _cwb = bbox_cache.ComputeWorldBound(_child)
                        _crng = _cwb.ComputeAlignedRange()
                        if not _crng.IsEmpty():
                            _csz = _crng.GetSize()
                            _cmaxdim = max(abs(_csz[i]) * mpu for i in range(3))
                            if _cmaxdim < 0.15:
                                continue
                    except Exception:
                        pass
                    _xform_fix_paths.add(_cpath)
                    _xform_fix.append(_child)
        if _xform_fix:
            log(f"  -> {len(_xform_fix)} fixture(s) found via parent-Xform keyword scan")
            fixtures_prims = fixtures_prims + _xform_fix

    # Last-resort emissive scan: if STILL no fixtures, drop the light-keyword name guard and
    # accept any emissive Gprim that isn't a hard false positive and isn't too large.
    # This handles scenes where fixture names follow conventions we don't know (e.g. NV_June*).
    if not fixtures_prims:
        log("  [last-resort] 0 fixtures after keyword+emissive passes — scanning ALL emissive geo")
        named_paths_lr = set()  # nothing in fixtures_prims yet
        for p in stage.Traverse():
            if not p.IsA(UsdGeom.Gprim):
                continue
            _pname = p.GetName().lower()
            _ppath = str(p.GetPath()).lower()
            if any(fp in _pname or fp in _ppath for fp in HARD_FALSE_POSITIVES):
                continue
            if not has_emissive_material(p):
                continue
            mesh = UsdGeom.Mesh(p)
            if mesh:
                pts = mesh.GetPointsAttr().Get()
                if pts and len(pts) >= 3:
                    xf = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    world_pts = [xf.Transform(Gf.Vec3d(pt[0], pt[1], pt[2])) for pt in pts[:200]]
                    max_dim = max(
                        (max(v[i] for v in world_pts) - min(v[i] for v in world_pts)) * mpu
                        for i in range(3)
                    )
                    if max_dim > 5.0:
                        continue
            fixtures_prims.append(p)
            log(f"  [last-resort] emissive fixture: {p.GetPath()}")
        if fixtures_prims:
            log(f"  [last-resort] -> {len(fixtures_prims)} fixture(s) found via emissive fallback")

    # raw_positions: (x, y, z, name, parent_path)
    # parent_path is the USD path of the fixture Xform that owns this sub-mesh.
    # All sub-meshes of the same lamp (body, cover, led, screws...) share the same parent_path.
    raw_positions = []

    # PointInstancer support
    for prim in stage.Traverse():
        if prim.GetTypeName() == "PointInstancer":
            instancer = UsdGeom.PointInstancer(prim)
            protos = instancer.GetPrototypesRel().GetTargets()
            has_fixture_proto = any(
                any(k in stage.GetPrimAtPath(p).GetName().lower() for k in FIXTURE_KEYWORDS)
                for p in protos if stage.GetPrimAtPath(p).IsValid()
            )
            if has_fixture_proto:
                positions = instancer.GetPositionsAttr().Get()
                if positions:
                    log(f"\n  PointInstancer: {prim.GetPath()} -> {len(positions)} instance(s)")
                    xformable = UsdGeom.Xformable(prim)
                    world_xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    parent_path = str(prim.GetPath().GetParentPath())
                    for pos in positions:
                        wp = world_xform.Transform(Gf.Vec3d(pos[0], pos[1], pos[2]))
                        raw_positions.append((wp[0], wp[1], wp[2], prim.GetName(), parent_path, None))

    for fp in fixtures_prims:
        # parent_path = the fixture Xform that owns this sub-mesh
        parent_path = str(fp.GetPath().GetParentPath())

        mesh = UsdGeom.Mesh(fp)
        if not mesh:
            # USD instance prims (IsInstance=True) carry no geometry directly — their mesh
            # lives in a prototype.  Prototype-based analysis fails because the prototype
            # uses scene-internal millimetre coords and a complex rotation, which confuses
            # the elongation / sub-unit logic.  BBoxCache already resolves the full transform
            # chain (including root scale=0.001 and all rotations) so it gives the correct
            # world-space position and face dimensions directly.
            if fp.IsInstance():
                _inst_handled = False
                try:
                    # First: try prototype clustering to find individual sub-fixtures
                    # (e.g. a prototype with 6 lamp geometries needs 6 lights, not 1).
                    _proto = fp.GetPrototype()
                    if _proto:
                        for _pp in Usd.PrimRange(_proto):
                            if _pp.IsA(UsdGeom.Mesh):
                                _pmesh_xf = UsdGeom.Xformable(_pp).ComputeLocalToWorldTransform(
                                    Usd.TimeCode.Default())
                                _inst_xf  = UsdGeom.Xformable(fp).ComputeLocalToWorldTransform(
                                    Usd.TimeCode.Default())
                                _composed = _pmesh_xf * _inst_xf
                                _centers = find_fixture_centers_in_mesh(
                                    _pp, mpu, up_axis=up_axis,
                                    world_xform_override=_composed)
                                if len(_centers) > 1:
                                    _centers = merge_aligned_cluster_pairs(
                                        _centers, up_axis=up_axis)
                                if len(_centers) > 1:
                                    log(f"\n  {fp.GetName()} (instance, {len(_centers)}"
                                        f" sub-fixtures via prototype clustering)")
                                    for _ce in _centers:
                                        _cx, _cy, _cz = _ce[0], _ce[1], _ce[2]
                                        _cm = _ce[3] if len(_ce) > 3 else None
                                        if _cm:
                                            _cm = {**_cm, 'from_cluster': True}
                                        raw_positions.append(
                                            (_cx, _cy, _cz, fp.GetName(), parent_path, _cm))
                                    _inst_handled = True
                                break
                except Exception:
                    pass

                if not _inst_handled:
                    # Single fixture or clustering failed — use BBoxCache for position + size.
                    try:
                        _iwb = bbox_cache.ComputeWorldBound(fp)
                        _irng = _iwb.ComputeAlignedRange()
                        if not _irng.IsEmpty():
                            _ictr = _irng.GetMidpoint()
                            _isz  = _irng.GetSize()
                            _axes  = sorted(range(3), key=lambda i: abs(_isz[i]), reverse=True)
                            _dim0  = abs(_isz[_axes[0]]) * mpu
                            _dim1  = abs(_isz[_axes[1]]) * mpu
                            _dim2  = abs(_isz[_axes[2]]) * mpu
                            _fn_ax   = _axes[2]
                            _fn_sign = -1.0 if _isz[_fn_ax] > 0 else 1.0
                            fn_w = tuple(_fn_sign if j == _fn_ax else 0.0 for j in range(3))
                            _fz = fn_w[2]; _dot_fn = -_fz
                            _cx = fn_w[1]; _cy = -fn_w[0]; _c_len = (_cx**2 + _cy**2)**0.5
                            if _c_len < 1e-6:
                                _face_quat = (1.,0.,0.,0.) if _dot_fn > 0 else (0.,1.,0.,0.)
                            else:
                                _ang = math.acos(max(-1., min(1., _dot_fn)))
                                _sh  = math.sin(_ang / 2)
                                _face_quat = (math.cos(_ang/2),
                                              _cx/_c_len*_sh, _cy/_c_len*_sh, 0.)
                            _long_axis = 'X' if abs(_isz[0]) >= abs(_isz[1]) else 'Y'
                            _inst_meta = {
                                'elongated':   True,
                                'long_dim':    _dim0 / mpu,
                                'short_dim':   _dim1 / mpu,
                                'face_height': _dim1 / mpu,
                                'long_dim_m':  _dim0,
                                'short_dim_m': _dim1,
                                'long_axis':   _long_axis,
                                'world_quat':  (1., 0., 0., 0.),
                                'face_quat':   _face_quat,
                            }
                            raw_positions.append((_ictr[0], _ictr[1], _ictr[2],
                                                  fp.GetName(), parent_path, _inst_meta))
                            log(f"\n  {fp.GetName()} (instance bbox {_dim0:.2f}×{_dim1:.2f}m"
                                f" depth={_dim2:.3f}m) -> 1 light")
                            _inst_handled = True
                    except Exception:
                        pass
                if _inst_handled:
                    continue
            pos = get_world_position(fp)
            if pos:
                _inst_single_meta = None
                try:
                    _wb = bbox_cache.ComputeWorldBound(fp)
                    _rng = _wb.ComputeAlignedRange()
                    if not _rng.IsEmpty():
                        _sz = _rng.GetSize()
                        _blt = _classify_bbox_shape([abs(_sz[i]) * mpu for i in range(3)])
                        if _blt:
                            _inst_single_meta = {'bbox_ltype': _blt}
                except Exception:
                    pass
                raw_positions.append((pos[0], pos[1], pos[2], fp.GetName(), parent_path, _inst_single_meta))
            continue

        points = mesh.GetPointsAttr().Get()
        if not points:
            continue

        if up_axis == "Z":
            ax0 = [p[0] for p in points]  # floor X
            ax1 = [p[1] for p in points]  # floor Y
            ax2 = [p[2] for p in points]  # up Z
        else:
            ax0 = [p[0] for p in points]  # floor X
            ax1 = [p[2] for p in points]  # floor Z
            ax2 = [p[1] for p in points]  # up Y

        span_0 = max(ax0) - min(ax0)
        span_1 = max(ax1) - min(ax1)
        span_ax2 = max(ax2) - min(ax2)
        xformable = UsdGeom.Xformable(fp)
        world_xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        # Compute world-space spans by transforming axis endpoints directly.
        # This is correct under any rotation — the old abs(GetRow(0)[0]) trick
        # returns ~0 for fixtures rotated 90° around Z, causing missed elongated detections.
        if up_axis == "Z":
            _w0a = world_xform.Transform(Gf.Vec3d(min(ax0), 0, 0))
            _w0b = world_xform.Transform(Gf.Vec3d(max(ax0), 0, 0))
            _w1a = world_xform.Transform(Gf.Vec3d(0, min(ax1), 0))
            _w1b = world_xform.Transform(Gf.Vec3d(0, max(ax1), 0))
        else:
            _w0a = world_xform.Transform(Gf.Vec3d(min(ax0), 0, 0))
            _w0b = world_xform.Transform(Gf.Vec3d(max(ax0), 0, 0))
            _w1a = world_xform.Transform(Gf.Vec3d(0, 0, min(ax1)))
            _w1b = world_xform.Transform(Gf.Vec3d(0, 0, max(ax1)))
        span_0_world_m = (_w0b - _w0a).GetLength() * mpu
        span_1_world_m = (_w1b - _w1a).GetLength() * mpu
        max_span_world_m = max(span_0_world_m, span_1_world_m)
        long_dim_m  = max_span_world_m
        short_dim_m = min(span_0_world_m, span_1_world_m)
        aspect = long_dim_m / max(short_dim_m, 0.001)

        # Elongated fixture: one horizontal dimension >= 3× the other.
        # Treat as a single linear emitter — skip clustering, use bbox + meta for sizing.
        is_elongated = long_dim_m >= 0.3 and aspect >= 2.5
        is_merged    = max_span_world_m > 5.0 and not is_elongated

        if is_elongated:
            # Determine which local axis is long (ax0=X, ax1=Y/floor-Y)
            long_local  = 'X' if span_0 >= span_1 else 'Y'
            short_local = 'Y' if long_local == 'X' else 'X'

            min_ax0, max_ax0 = min(ax0), max(ax0)
            min_ax1, max_ax1 = min(ax1), max(ax1)
            cz_local = min(ax2)  # bottom face / emitter surface

            # Determine world long axis
            local_long_vec = (Gf.Vec3d(1, 0, 0) if long_local == 'X'
                              else (Gf.Vec3d(0, 1, 0) if up_axis == "Z"
                                    else Gf.Vec3d(0, 0, 1)))
            world_long = world_xform.TransformDir(local_long_vec)
            long_axis = 'X' if abs(world_long[0]) >= abs(world_long[1]) else 'Y'

            # Split by vertex gaps to find actual per-unit bounds
            sub_units = split_elongated_sub_units(points, long_local, long_dim_m=long_dim_m)
            n_units = len(sub_units)

            # Face normal: prefer authored mesh normals, fallback to bounding-box axis.
            # For elongated fixtures the long-axis normals (end caps) dominate the raw average,
            # so we filter them out first — only normals NOT parallel to the long axis matter.
            _mesh_normals = mesh.GetNormalsAttr().Get()
            _fn_from_normals = False
            if _mesh_normals and len(_mesh_normals) >= 3:
                # Determine the local long-axis direction to exclude
                _long_ax = Gf.Vec3d(1, 0, 0) if long_local == 'X' else Gf.Vec3d(0, 1, 0)
                _anx = _any_ = _anz = 0.0
                _cnt = 0
                for _n in _mesh_normals:
                    _d = abs(_n[0]*_long_ax[0] + _n[1]*_long_ax[1] + _n[2]*_long_ax[2])
                    if _d < 0.7:    # keep normals not mostly along the long axis
                        _anx += _n[0]; _any_ += _n[1]; _anz += _n[2]; _cnt += 1
                if _cnt >= 3:
                    _anx /= _cnt; _any_ /= _cnt; _anz /= _cnt
                    _nl = (_anx**2 + _any_**2 + _anz**2) ** 0.5
                    if _nl > 0.15:   # meaningful direction (not cancelled out)
                        fn_local = Gf.Vec3d(_anx/_nl, _any_/_nl, _anz/_nl)
                        _fn_from_normals = True
                        log(f"  DEBUG normals {fp.GetName()}: face_n=({_anx/_nl:.3f},{_any_/_nl:.3f},{_anz/_nl:.3f}) from {_cnt}/{len(_mesh_normals)} filtered normals")

            if not _fn_from_normals:
                # Try area-weighted triangle normals (detects asymmetry between front/back faces).
                _fvc = mesh.GetFaceVertexCountsAttr().Get()
                _fvi = mesh.GetFaceVertexIndicesAttr().Get()
                if _fvc and _fvi:
                    _awn = [0.0, 0.0, 0.0]
                    _vi = 0
                    for _fc in _fvc:
                        if _fc >= 3:
                            _p0 = points[_fvi[_vi]]
                            _p1 = points[_fvi[_vi+1]]
                            _p2 = points[_fvi[_vi+2]]
                            _e1 = (_p1[0]-_p0[0], _p1[1]-_p0[1], _p1[2]-_p0[2])
                            _e2 = (_p2[0]-_p0[0], _p2[1]-_p0[1], _p2[2]-_p0[2])
                            _awn[0] += _e1[1]*_e2[2] - _e1[2]*_e2[1]
                            _awn[1] += _e1[2]*_e2[0] - _e1[0]*_e2[2]
                            _awn[2] += _e1[0]*_e2[1] - _e1[1]*_e2[0]
                        _vi += _fc
                    _awl = (_awn[0]**2 + _awn[1]**2 + _awn[2]**2) ** 0.5
                    if _awl > 1e-6:
                        fn_local = Gf.Vec3d(_awn[0]/_awl, _awn[1]/_awl, _awn[2]/_awl)
                        _fn_from_normals = True
                        log(f"  DEBUG tri {fp.GetName()}: area_weighted_n=({_awn[0]/_awl:.3f},{_awn[1]/_awl:.3f},{_awn[2]/_awl:.3f})")

            if not _fn_from_normals:
                # Final fallback: thinnest bounding-box axis as face normal
                if up_axis == "Z":
                    _thin_dirs = [(span_0,    Gf.Vec3d(-1,  0,  0)),
                                  (span_1,    Gf.Vec3d( 0, -1,  0)),
                                  (span_ax2,  Gf.Vec3d( 0,  0, -1))]
                else:
                    _thin_dirs = [(span_0,    Gf.Vec3d(-1,  0,  0)),
                                  (span_1,    Gf.Vec3d( 0,  0, -1)),
                                  (span_ax2,  Gf.Vec3d( 0, -1,  0))]
                fn_local = min(_thin_dirs, key=lambda e: e[0])[1]
                log(f"  DEBUG bbox {fp.GetName()}: all normals cancelled, fn_local={fn_local}")
            fn_w_raw = world_xform.TransformDir(fn_local)
            _fn_len = (fn_w_raw[0]**2 + fn_w_raw[1]**2 + fn_w_raw[2]**2) ** 0.5
            fn_w = (fn_w_raw[0]/_fn_len, fn_w_raw[1]/_fn_len, fn_w_raw[2]/_fn_len) \
                   if _fn_len > 1e-8 else (0.0, 0.0, -1.0)
            # Quaternion rotating RectLight default emit (-Z) to fn_w
            # dot((0,0,-1), fn_w) = -fz
            _dot_fn = -fn_w[2]
            # cross((0,0,-1), (fx,fy,fz)) = (0*fz-(-1)*fy, (-1)*fx-0*fz, 0*fy-0*fx) = (fy,-fx,0)
            _cx = fn_w[1]; _cy = -fn_w[0]; _cz = 0.0
            _c_len = (_cx**2 + _cy**2 + _cz**2) ** 0.5
            if _c_len < 1e-6:
                _face_quat = (1.0, 0.0, 0.0, 0.0) if _dot_fn > 0 else (0.0, 1.0, 0.0, 0.0)
            else:
                _angle = math.acos(max(-1.0, min(1.0, _dot_fn)))
                _sh = math.sin(_angle / 2)
                _face_quat = (math.cos(_angle / 2),
                              _cx/_c_len * _sh, _cy/_c_len * _sh, _cz/_c_len * _sh)
            # Thin axis name for face-aware position and size computation
            if fn_local[0] != 0:
                thin_local_name = 'X'
            elif fn_local[1] != 0:
                thin_local_name = 'Y'
            else:
                thin_local_name = 'Z'

            # Scale factors: local-unit → world metres along each axis
            _origin_w = world_xform.Transform(Gf.Vec3d(0, 0, 0))
            if up_axis == "Z":
                _lx_w = world_xform.Transform(Gf.Vec3d(1, 0, 0))
                _ly_w = world_xform.Transform(Gf.Vec3d(0, 1, 0))
                _lz_w = world_xform.Transform(Gf.Vec3d(0, 0, 1))
            else:
                _lx_w = world_xform.Transform(Gf.Vec3d(1, 0, 0))
                _ly_w = world_xform.Transform(Gf.Vec3d(0, 0, 1))
                _lz_w = world_xform.Transform(Gf.Vec3d(0, 1, 0))
            scale_ax0 = (_lx_w - _origin_w).GetLength() * mpu  # local X → metres
            scale_ax1 = (_ly_w - _origin_w).GetLength() * mpu  # local Y (or Z) → metres
            scale_ax2 = (_lz_w - _origin_w).GetLength() * mpu  # local Z (or Y) → metres
            span_ax2_world_m = span_ax2 * scale_ax2

            # Compute face height geometrically from bbox spans and face normal.
            # For a tilted panel with fn_local=(fx,0,fz), height_dir=(-fz/|fxz|, 0, fx/|fxz|).
            # face_height = span_along_axis / |height_dir_component_on_that_axis|.
            # Taking min of X-derived and Z-derived estimates handles housings where the full Z
            # span is much larger than the emitting face (e.g. wall_straight_33 type fixtures).
            _fn_xz_len = (fn_local[0]**2 + fn_local[2]**2) ** 0.5
            if _fn_xz_len > 1e-6:
                _fn_xz_x = fn_local[0] / _fn_xz_len
                _fn_xz_z = fn_local[2] / _fn_xz_len
                _fh_inf = float('inf')
                _fh_from_x = (span_0 * scale_ax0 / abs(_fn_xz_z)) if abs(_fn_xz_z) > 0.2 else _fh_inf
                _fh_from_z = (span_ax2 * scale_ax2 / abs(_fn_xz_x)) if abs(_fn_xz_x) > 0.2 else _fh_inf
                _face_height_from_geo_m = min(_fh_from_x, _fh_from_z)
                if _face_height_from_geo_m == _fh_inf:
                    _face_height_from_geo_m = span_ax2_world_m
            else:
                _face_height_from_geo_m = span_ax2_world_m

            # Front face center in local space: offset the bbox center along fn_local to the
            # emitting face plane.  Works for flat panels (offset ≈ 0) and thick housings alike.
            _fn_projs  = [fn_local[0]*p[0] + fn_local[1]*p[1] + fn_local[2]*p[2] for p in points]
            _fp_max    = max(_fn_projs)
            _bbox_cx   = (min(ax0) + max(ax0)) / 2
            _bbox_cy   = (min(ax1) + max(ax1)) / 2
            _bbox_cz   = (min(ax2) + max(ax2)) / 2
            _bbox_proj = fn_local[0]*_bbox_cx + fn_local[1]*_bbox_cy + fn_local[2]*_bbox_cz
            _fp_offset = _fp_max - _bbox_proj
            _fc_x      = _bbox_cx + _fp_offset * fn_local[0]
            _fc_y      = _bbox_cy + _fp_offset * fn_local[1]
            _fc_z      = _bbox_cz + _fp_offset * fn_local[2]

            for unit in sub_units:
                # Per-unit centre in local coords
                if long_local == 'X':
                    cx_local = (unit['long_min']  + unit['long_max'])  / 2
                    cy_local = (unit['short_min'] + unit['short_max']) / 2
                    unit_long_dim_m  = max((unit['long_max']  - unit['long_min'])  * scale_ax0, 0.05)
                    unit_short_dim_m = max((unit['short_max'] - unit['short_min']) * scale_ax1, 0.05)
                else:
                    cx_local = (unit['short_min'] + unit['short_max']) / 2
                    cy_local = (unit['long_min']  + unit['long_max'])  / 2
                    unit_long_dim_m  = max((unit['long_max']  - unit['long_min'])  * scale_ax1, 0.05)
                    unit_short_dim_m = max((unit['short_max'] - unit['short_min']) * scale_ax0, 0.05)

                # Place light at front face centroid (non-long axes) + per-unit long-axis center.
                # _fc_x/_fc_y/_fc_z hold the local centroid of the emitting face, computed above.
                if up_axis == "Z":
                    if long_local == 'Y':
                        center_local = Gf.Vec3d(_fc_x, cy_local, _fc_z)
                    else:
                        center_local = Gf.Vec3d(cx_local, _fc_y, _fc_z)
                else:
                    if long_local == 'Y':
                        center_local = Gf.Vec3d(_fc_x, _fc_y, cy_local)
                    else:
                        center_local = Gf.Vec3d(cx_local, _fc_y, _fc_z)
                center_world = world_xform.Transform(center_local)

                # RectLight face height: for side-facing panels use geometric front-face extent,
                # not the full Z span (which includes housing depth for thick fixtures).
                if thin_local_name in ('X', 'Y'):
                    unit_face_height_m = max(_face_height_from_geo_m, 0.05)
                else:
                    unit_face_height_m = unit_short_dim_m

                rot = world_xform.ExtractRotation()
                rq  = rot.GetQuat()
                meta = {
                    'elongated':        True,
                    'long_dim':         unit_long_dim_m / mpu,
                    'short_dim':        unit_short_dim_m / mpu,
                    'face_height':      unit_face_height_m / mpu,
                    'long_dim_m':       unit_long_dim_m,
                    'short_dim_m':      unit_short_dim_m,
                    'long_axis':        long_axis,
                    'world_quat':       (float(rq.GetReal()),
                                        float(rq.GetImaginary()[0]),
                                        float(rq.GetImaginary()[1]),
                                        float(rq.GetImaginary()[2])),
                    'face_quat':        _face_quat,
                }
                raw_positions.append((center_world[0], center_world[1], center_world[2],
                                       fp.GetName(), parent_path, meta))

            log(f"\n  {fp.GetName()} (elongated {long_dim_m:.2f}×{short_dim_m:.2f}m,"
                f" axis={long_axis}) -> {n_units} linear fixture(s)")

        elif is_merged:
            # Try gap-based panel detection first — works on sparse emitter meshes
            # (glass panes, flat LED arrays) where flood-fill finds only 1 cluster.
            panels = find_panels_by_gap_analysis(points, world_xform, mpu, up_axis=up_axis)
            if len(panels) > 1:
                centers = panels
                log(f"\n  {fp.GetName()} ({max_span_world_m:.1f}m merged) -> {len(centers)} panel(s) via gap analysis")
            else:
                centers = find_fixture_centers_in_mesh(fp, mpu, up_axis=up_axis)
                log(f"\n  {fp.GetName()} ({max_span_world_m:.1f}m merged) -> {len(centers)} instance(s) via clustering")
            if not centers and fp.IsInstance():
                # find_fixture_centers_in_mesh cannot traverse instance prims.
                # Fall back to BBoxCache centroid + full bbox extent as one large area light.
                try:
                    _iwb = bbox_cache.ComputeWorldBound(fp)
                    _irng = _iwb.ComputeAlignedRange()
                    if not _irng.IsEmpty():
                        _ictr = _irng.GetMidpoint()
                        _isz = _irng.GetSize()
                        _dims = sorted([abs(_isz[i]) * mpu for i in range(3)], reverse=True)
                        _fb_meta = {'dim0_m': _dims[0], 'dim1_m': _dims[1], 'from_cluster': True}
                        centers = [(_ictr[0], _ictr[1], _ictr[2], _fb_meta)]
                        log(f"  -> 1 bbox fallback at ({_ictr[0]:.2f},{_ictr[1]:.2f},{_ictr[2]:.2f})"
                            f" {_dims[0]:.2f}×{_dims[1]:.2f}m")
                except Exception:
                    pass
            for entry in centers:
                cx, cy, cz = entry[0], entry[1], entry[2]
                cluster_bbox = entry[3] if len(entry) > 3 else None
                if cluster_bbox and cluster_bbox.get('dim0_m') and cluster_bbox.get('dim1_m'):
                    cluster_meta = {**cluster_bbox, 'from_cluster': True}
                else:
                    cluster_meta = None
                raw_positions.append((cx, cy, cz, fp.GetName(), parent_path, cluster_meta))

        else:
            # Try multi-panel detection: find flat connected face groups by normal similarity.
            # Returns >=2 panels or [] to fall through to the single-fixture path.
            _flat_panels = []
            try:
                _flat_panels = find_flat_panels_in_mesh(mesh, mpu, world_xform)
            except Exception as _fpe:
                log(f"    flat panel: exception — {_fpe}")

            if _flat_panels:
                for _px, _py, _pz, _pmeta in _flat_panels:
                    raw_positions.append((_px, _py, _pz, fp.GetName(), parent_path, _pmeta))
                log(f"\n  {fp.GetName()} -> {len(_flat_panels)} flat panel(s) [bbox→RectLight each]")
            else:
                # Single fixture: place at the bottom face of the mesh (emitter surface),
                # not the centroid. For a pendant globe this is the bottom of the glass;
                # for a wall sconce it is the lowest point of the fixture body.
                # Then pull 5 cm further below the surface so the source is not inside geo.
                cx_local = sum(p[0] for p in points) / len(points)
                cy_local = (sum(p[1] for p in points) / len(points)
                            if up_axis == "Z"
                            else sum(p[2] for p in points) / len(points))
                bottom_up = min(ax2)  # lowest vertex along the up axis (local space)
                pull = 0.05 / mpu
                if up_axis == "Z":
                    bottom_local = Gf.Vec3d(cx_local, cy_local, bottom_up)
                    bw = world_xform.Transform(bottom_local)
                    pos = (bw[0], bw[1], bw[2] - pull)
                else:
                    bottom_local = Gf.Vec3d(cx_local, bottom_up, cy_local)
                    bw = world_xform.Transform(bottom_local)
                    pos = (bw[0], bw[1] - pull, bw[2])
                _single_meta = None
                try:
                    _wb = bbox_cache.ComputeWorldBound(fp)
                    _rng = _wb.ComputeAlignedRange()
                    if not _rng.IsEmpty():
                        _sz = _rng.GetSize()
                        _blt = _classify_bbox_shape([abs(_sz[i]) * mpu for i in range(3)])
                        if _blt:
                            _single_meta = {'bbox_ltype': _blt}
                except Exception:
                    pass
                raw_positions.append((pos[0], pos[1], pos[2], fp.GetName(), parent_path, _single_meta))
                log(f"\n  {fp.GetName()} -> 1 instance at {pos}"
                    + (f" [bbox→{_single_meta['bbox_ltype']}]" if _single_meta else ""))

    # Group by parent fixture Xform and select best sub-mesh type.
    # One parent Xform = one physical fixture = correct set of light positions.
    all_fixture_positions = select_best_submesh_positions(raw_positions)

    log(f"\n  Raw sub-mesh entries:   {len(raw_positions)}")
    log(f"  After parent grouping:  {len(all_fixture_positions)} fixture position(s)")

    if not all_fixture_positions:
        log("\n  *** WARNING: No light fixtures detected in this scene. ***")
        log("  The skill searched for:")
        log("    - Mesh/Gprim names containing: light, lamp, emitter, luminaire, fixture, etc.")
        log("    - Emissive materials (emissiveColor > 0)")
        log("    - Parent Xform names containing: lightEmitter, windowGlass, etc.")
        log("    - USD instance prims (IsInstance=True) via prototype analysis")
        log("  If your scene has fixtures, check that their mesh or material names match")
        log("  one of these conventions, or that they use emissive materials.")
        log("  Generating fallback ceiling grid to achieve ≥250 lux.")

        # Room extents
        _fb_min = combined.GetMin()
        _fb_max = combined.GetMax()
        if up_axis == "Z":
            _fb_L, _fb_W, _fb_H = size_m[0], size_m[1], size_m[2]
        else:
            _fb_L, _fb_W, _fb_H = size_m[0], size_m[2], size_m[1]

        # Mounting height above 0.85m work plane; grid spacing = 1× mount height
        _fb_h_m  = max(_fb_H - 0.85, 1.0)
        _fb_spc  = _fb_h_m
        _fb_nx   = max(1, round(_fb_L / _fb_spc))
        _fb_ny   = max(1, round(_fb_W / _fb_spc))
        _fb_N    = _fb_nx * _fb_ny

        # Lumen estimate (informational — standard power calc uses real N at runtime)
        _fb_RI    = (_fb_L * _fb_W) / (_fb_h_m * (_fb_L + _fb_W)) if _fb_h_m > 0 and (_fb_L + _fb_W) > 0 else 1.0
        _fb_UF    = min(0.3 + _fb_RI * 0.15, 0.75)
        _fb_est   = max(1000.0, round(250.0 * _fb_L * _fb_W / (_fb_UF * 0.8 * _fb_N)))
        _fb_panel = 0.6  # 0.6×0.6 m standard LED panel

        log(f"  Fallback grid: {_fb_nx}×{_fb_ny} = {_fb_N} RectLights "
            f"({_fb_panel:.1f}×{_fb_panel:.1f}m) at H={_fb_H:.2f}m")
        log(f"  RI={_fb_RI:.2f} | UF={_fb_UF:.2f} | ~{_fb_est:.0f} lm/panel "
            f"for 250 lux target")

        # Grid positions in scene units, 5 cm below ceiling
        _fb_xstep = (_fb_max[0] - _fb_min[0]) / _fb_nx
        if up_axis == "Z":
            _fb_ystep = (_fb_max[1] - _fb_min[1]) / _fb_ny
            _fb_ceil  = _fb_max[2] - 0.05 / mpu
        else:
            _fb_ystep = (_fb_max[2] - _fb_min[2]) / _fb_ny
            _fb_ceil  = _fb_max[1] - 0.05 / mpu

        _fb_meta = {'dim0_m': _fb_panel, 'dim1_m': _fb_panel,
                    'from_cluster': True, 'fallback_grid': True}
        for _ix in range(_fb_nx):
            _gx = _fb_min[0] + _fb_xstep * (_ix + 0.5)
            for _iy in range(_fb_ny):
                if up_axis == "Z":
                    _gy = _fb_min[1] + _fb_ystep * (_iy + 0.5)
                    _gz = _fb_ceil
                else:
                    _gy = _fb_ceil
                    _gz = _fb_min[2] + _fb_ystep * (_iy + 0.5)
                all_fixture_positions.append(
                    (_gx, _gy, _gz, 'FALLBACK_CEILING_PANEL', _fb_meta))

    for i, (x, y, z, name, *_) in enumerate(all_fixture_positions):
        log(f"    [{i:02d}] {name} at ({x:.2f}, {y:.2f}, {z:.2f})")

    # --- Glass/window detection ---
    log("\n=== Glass/Window Detection ===")
    if no_dome:
        has_glass = False
        log("  (Skipped — no_dome=True)")
    elif force_dome:
        has_glass = True
        log("  (Forced — force_dome=True)")
    else:
        has_glass, glass_info = check_glass_geometry(stage, bbox_cache, mpu)
        log(f"  {glass_info}")
    if has_glass:
        log("  -> Will create DomeLight")

    # --- Classify existing lights ---
    log("\n=== Step 3: Upgrade Existing / Create New Lights ===")
    existing_lights = [p for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI)]

    robot_lights = []         # robot/vehicle lights — log but never touch
    existing_dome_lights = [] # DomeLights — handled by dome logic, not orphan logic
    architectural_with_pos = []  # regular lights eligible for fixture matching

    for lp in existing_lights:
        if is_robot_vehicle_light(lp):
            robot_lights.append(lp)
        elif lp.GetTypeName() == "DomeLight":
            # DomeLights have no fixture geometry — handled by dome logic below
            existing_dome_lights.append(lp)
        else:
            pos = get_world_position(lp)
            if pos:
                architectural_with_pos.append((lp, pos))

    if robot_lights:
        log(f"\n  Robot/vehicle lights (excluded — not upgraded): {len(robot_lights)}")
        for lp in robot_lights[:5]:
            log(f"    - {lp.GetPath()}")
        if len(robot_lights) > 5:
            log(f"    ... and {len(robot_lights) - 5} more")

    log(f"  Existing architectural lights: {len(architectural_with_pos)} → all will be deactivated+hidden")
    log(f"  Existing DomeLights:          {len(existing_dome_lights)}")
    log(f"  New lights to create:         {len(all_fixture_positions)} (one per fixture position)")

    # --- Power calculation — use FIXTURE POSITIONS as N, not existing light count ---
    # Fixture positions = lights that will actually be active after deactivating orphans.
    # Using existing light count (2054) over-dilutes the lumen budget — each active light
    # ends up with too little power because the calculation assumes all 2054 are on.
    log("\n=== Power Calculation ===")
    scene_type = infer_scene_type(stage)
    base_target_lux = target_lux or TARGET_LUX.get(scene_type, 300)
    effective_target_lux = base_target_lux

    # Dome contribution: when --force-dome is used with --window-size or --dome-contribution,
    # reduce fixture target by estimated indoor daylight contribution so fixtures are not
    # oversized. Coverage ratios: small=5%, medium=15%, large=30% of wall area; glass
    # transmission assumed 0.8. dome_contribution overrides window_size directly.
    _dome_contrib_lux  = 0.0
    _dome_contrib_note = None
    _WINDOW_COVERAGE   = {"small": 0.05, "medium": 0.15, "large": 0.30}
    _WINDOW_LABEL      = {
        "small":  "narrow strip windows (~5% wall area)",
        "medium": "standard windows (~15% wall area)",
        "large":  "floor-to-ceiling / curtain wall (~30% wall area)",
    }
    if force_dome or has_glass:
        _eff_dome_illum = dome_illuminance if dome_illuminance > 0 else 400.0
        if dome_contribution is not None:
            _dome_contrib_lux  = max(0.0, dome_contribution)
            _dome_contrib_note = f"direct override ({_dome_contrib_lux:.0f} lux)"
        elif window_size is not None:
            _coverage          = _WINDOW_COVERAGE[window_size]
            _dome_contrib_lux  = round(_eff_dome_illum * _coverage * 0.8)
            _dome_contrib_note = (
                f"{window_size} windows — {_WINDOW_LABEL[window_size]} "
                f"→ {_dome_contrib_lux:.0f} lux indoor contribution "
                f"({_eff_dome_illum:.0f} lux sky × {_coverage*100:.0f}% coverage × 0.8 transmission)"
            )
        if _dome_contrib_lux > 0:
            effective_target_lux = max(base_target_lux * 0.20, base_target_lux - _dome_contrib_lux)
            log(f"  Dome contribution ({_dome_contrib_note}):")
            log(f"    Fixture target reduced: {base_target_lux:.0f} lux → {effective_target_lux:.0f} lux")
            log(f"    (DomeLight covers the remaining ~{_dome_contrib_lux:.0f} lux)")

    # Dome illuminance = outdoor sky brightness at the HDRI capture location (Step 9 table):
    #   400 lux    — sunset / clear sky  (default — adds realistic ambient without flooding)
    #   10000 lux  — midday overcast
    #   100000 lux — midday direct sunlight
    # NOTE: RTX dome lights illuminate every surface with line-of-sight to the sky.
    # If the building geometry does not fully seal roof/walls, the dome floods the
    # interior. Use --dome-illuminance to reduce, or --no-dome to disable entirely.
    # Validate: Debug View -> PT AOV Illuminance — if floor reads dome-dominated
    # (much higher than fixture target), reduce --dome-illuminance.
    if has_glass:
        if dome_illuminance <= 0:
            dome_illuminance = 400.0
            log(f"  (Dome illuminance auto: {dome_illuminance} lux — sunset/clear sky, see Step 9 table)")
    elif force_dome and dome_illuminance <= 0:
        dome_illuminance = 400.0
        log(f"  (Dome illuminance auto: {dome_illuminance} lux — sunset/clear sky, forced)")

    _fl_area  = size_m[0] * size_m[1] if up_axis == "Z" else size_m[0] * size_m[2]
    _L        = size_m[0]
    _W        = size_m[1] if up_axis == "Z" else size_m[2]
    _ceil_h   = size_m[2] if up_axis == "Z" else size_m[1]
    _ht       = _ceil_h * 0.6  # ceiling/wall split threshold
    _MF       = 0.8

    # Split fixture positions into ceiling (high) and wall (low) groups
    _fix_high = [(x, y, z) for x, y, z, n, *_ in all_fixture_positions
                 if (z if up_axis == "Z" else y) * mpu >= _ht]
    _fix_low  = [(x, y, z) for x, y, z, n, *_ in all_fixture_positions
                 if (z if up_axis == "Z" else y) * mpu < _ht]

    # --- Reflectance-aware UF + scene complexity factor -----------------
    if reflectances is not None:
        _refl_c, _refl_w, _refl_f = reflectances
        _refl_detail = [
            f"Reflectance source: user override.",
            f"Assumed reflectances — ceiling: {_refl_c:.0%}  "
            f"walls: {_refl_w:.0%}  floor: {_refl_f:.0%}",
        ]
    else:
        _refl_c, _refl_w, _refl_f, _refl_tier, _refl_detail = \
            _sample_scene_reflectances(stage, up_axis, combined, scene_type)

    # Use actual average fixture height for RI — more accurate than the split threshold
    if _fix_high:
        _h_preview = sum((z if up_axis == "Z" else y) * mpu for x, y, z in _fix_high) / len(_fix_high)
    elif _fix_low:
        _h_preview = sum((z if up_axis == "Z" else y) * mpu for x, y, z in _fix_low) / len(_fix_low)
    else:
        _h_preview = _ceil_h * 0.6
    _ri_preview  = _room_index(_L, _W, _h_preview)
    _complexity  = _scene_complexity_factor(scene_type, _ri_preview)

    log(f"  N (active fixtures): {len(all_fixture_positions)} "
        f"({len(_fix_high)} ceiling, {len(_fix_low)} wall)")

    log(f"\n  LUMEN CALCULATION METHOD")
    log(f"  " + "-" * 56)
    for _dl in _refl_detail:
        log(f"  {_dl}")
    log(f"  Scene complexity factor: {_complexity:.2f}×  "
        f"({scene_type}, RI≈{_ri_preview:.1f})")
    log(f"    Corrects for geometry occlusion / machinery blocking inter-reflections")
    log(f"    in RTX path tracing vs the simple empty-room the lumen method assumes.")
    log(f"  Formula: lm = (target lux × floor area) / (N × UF × MF) × complexity")
    log(f"    MF={_MF}  UF=RI-lookup × reflectance-scale  complexity={_complexity:.2f}×")
    log(f"  Reference reflectances the RI table assumes: "
        f"ceiling 70%  walls 50%  floor 20%")
    log(f"  → If the scene still looks too dark/bright after this run:")
    log(f"    Check PT AOV Illuminance on the floor, then re-run with:")
    log(f"    measured_lux=<your reading>  to scale lights to the exact target.")
    log(f"  " + "-" * 56)

    if _fix_high:
        _h_avg  = sum((z if up_axis == "Z" else y) * mpu for x, y, z in _fix_high) / len(_fix_high)
        _ri     = _room_index(_L, _W, _h_avg)
        _UF, _refl_scale = _uf_reflectance_adjusted(_ri, _refl_c, _refl_w, _refl_f)

        if _fix_low:
            _h_low  = sum((z if up_axis == "Z" else y) * mpu for x, y, z in _fix_low) / len(_fix_low)
            _ratio  = (_h_low / _h_avg) ** 2
            # Weighted effective N: HIGH and LOW combined must deliver the full target.
            # Using only N_high made HIGH alone cover the whole room, then LOW doubled it.
            _N_eff      = len(_fix_high) * 1.0 + len(_fix_low) * _ratio
            _power_high = max(round((_fl_area * effective_target_lux) /
                                    (_N_eff * _UF * _MF) * _complexity / 100) * 100, 100)
            _power_low  = max(round(_power_high * _ratio / 100) * 100, 100)
            log(f"  HIGH ({len(_fix_high)} fixtures, avg {_h_avg:.1f}m): "
                f"RI={_ri:.1f}, UF={_UF} (refl {_refl_scale:.2f}×), "
                f"complexity {_complexity:.2f}×, N_eff={_N_eff:.1f} → {_power_high} lm")
            log(f"  LOW  ({len(_fix_low)} fixtures, avg {_h_low:.1f}m): "
                f"scale={_ratio:.2f} × {_power_high} → {_power_low} lm")
        else:
            _power_high = max(round((_fl_area * effective_target_lux) /
                                    (len(_fix_high) * _UF * _MF) * _complexity / 100) * 100, 100)
            _power_low  = _power_high
            log(f"  HIGH ({len(_fix_high)} fixtures, avg {_h_avg:.1f}m): "
                f"RI={_ri:.1f}, UF={_UF} (refl {_refl_scale:.2f}×), "
                f"complexity {_complexity:.2f}× → {_power_high} lm")
    else:
        # Wall-only scene
        if _fix_low:
            _h_avg  = sum((z if up_axis == "Z" else y) * mpu for x, y, z in _fix_low) / len(_fix_low)
            _ri     = _room_index(_L, _W, _h_avg)
            _UF, _refl_scale = _uf_reflectance_adjusted(_ri, _refl_c, _refl_w, _refl_f)
            _power_low  = max(round((_fl_area * effective_target_lux) /
                                    (len(_fix_low) * _UF * _MF) * _complexity / 100) * 100, 100)
            _power_high = _power_low
            log(f"  WALL-ONLY ({len(_fix_low)} fixtures, avg {_h_avg:.1f}m): "
                f"RI={_ri:.1f}, UF={_UF} (refl {_refl_scale:.2f}×), "
                f"complexity {_complexity:.2f}× → {_power_low} lm")
        else:
            _power_high = _power_low = 4000
            log(f"  No fixture positions — fallback 4000 lm")

    default_power = _power_high  # ceiling zone; wall fixtures use _power_low

    deactivated_paths = set()
    created_lights = []

    # --- Pass 1: deactivate + hide ALL existing architectural lights ---
    log(f"\n  Deactivating + hiding all {len(architectural_with_pos)} architectural lights.")
    if dry_run:
        log(f"  [DRY RUN] Would deactivate+hide {len(architectural_with_pos)} lights")
    else:
        for lp, pos in architectural_with_pos:
            _deactivate_and_hide(lp)
            deactivated_paths.add(str(lp.GetPath()))
        log(f"  Done — {len(deactivated_paths)} lights deactivated+hidden.")

    # --- Pass 2: create new lights at every fixture position ---
    log(f"\n  Creating {len(all_fixture_positions)} new lights at fixture positions.")
    _ceiling_h = size_m[2] if up_axis == "Z" else size_m[1]

    # If all fixtures share one mesh name AND none are elongated, determine light type once.
    # Prevents height-based fallback from creating mixed SphereLight/RectLight for
    # physically identical fixtures that hang at different heights.
    # Skip when elongated fixtures are present — they each carry their own sizing meta.
    _unique_names   = {fname for _, _, _, fname, *_ in all_fixture_positions}
    _has_elongated  = any(
        (fmeta or {}).get('elongated')
        for *_, fmeta in all_fixture_positions
    )
    if len(_unique_names) == 1 and not _has_elongated:
        _uname = next(iter(_unique_names))
        _uniform_ltype, _uniform_hints = infer_light_type_from_name(_uname)
        _bbox_resolved = False
        # Name didn't resolve to a specific shape — try bbox from the first fixture with meta
        if _uniform_ltype == "SphereLight":
            for *_, _fm in all_fixture_positions:
                if _fm and _fm.get('bbox_ltype'):
                    _blt = _fm['bbox_ltype']
                    _uniform_ltype = _blt
                    _uniform_hints = {"type": _blt.replace('Light', '').lower()}
                    _bbox_resolved = True
                    break
        if _bbox_resolved:
            log(f"  Single fixture type '{_uname}' -> bbox shape analysis -> uniform {_uniform_ltype}")
        else:
            log(f"  Single fixture type '{_uname}' -> uniform {_uniform_ltype} for all")
    else:
        _uniform_ltype, _uniform_hints = None, None

    for i, fixture_entry in enumerate(all_fixture_positions):
        fx, fy, fz, fname = fixture_entry[0], fixture_entry[1], fixture_entry[2], fixture_entry[3]
        fmeta = fixture_entry[4] if len(fixture_entry) > 4 else None

        # Pick power based on fixture height
        fix_h_m = (fz if up_axis == "Z" else fy) * mpu
        light_power = _power_high if fix_h_m >= _ht else _power_low

        # Resolve light type — elongated fixtures override uniform type
        if fmeta and fmeta.get('elongated'):
            light_type, hints = infer_light_type_from_name(fname, height_m=fix_h_m,
                                                            ceiling_h=_ceiling_h)
            # If name doesn't identify a specific shape, default elongated → CylinderLight
            if light_type == "SphereLight":
                light_type, hints = "CylinderLight", {"type": "cylinder"}
        elif _uniform_ltype:
            light_type, hints = _uniform_ltype, _uniform_hints
        else:
            light_type, hints = infer_light_type_from_name(fname, height_m=fix_h_m,
                                                            ceiling_h=_ceiling_h,
                                                            fmeta=fmeta)

        if dry_run:
            size_note = ""
            if fmeta and fmeta.get('elongated'):
                size_note = (f" [{fmeta['long_dim_m']:.2f}×{fmeta['short_dim_m']:.2f}m"
                             f" axis={fmeta['long_axis']}]")
            elif fmeta and fmeta.get('from_cluster'):
                size_note = f" [cluster {fmeta['dim0_m']:.2f}×{fmeta['dim1_m']:.2f}m]"
            log(f"  [DRY RUN] CREATE {light_type} at ({fx:.2f}, {fy:.2f}, {fz:.2f}) "
                f"{light_power} lm — '{fname}'{size_note}")
        else:
            lights_scope = stage.GetPrimAtPath("/World/Lights")
            if not lights_scope.IsValid():
                UsdGeom.Scope.Define(stage, "/World/Lights")

            light_path = Sdf.Path(f"/World/Lights/{fname}_Light_{i:02d}")

            if light_type == "RectLight":
                light = UsdLux.RectLight.Define(stage, light_path)
                if fmeta and fmeta.get('elongated'):
                    long_a = fmeta.get('long_axis', 'X')
                    ld = float(fmeta['long_dim'])
                    # face_height: Z span of the mesh face (for side-facing panels where
                    # the emitting face is in the XZ or YZ plane, not the floor plane).
                    # Falls back to short_dim for flat ceiling/floor strips (thin in Z).
                    fh = float(fmeta.get('face_height', fmeta['short_dim']))
                    # RectLight: width=X, height=Y — swap if long axis is Y
                    light.GetWidthAttr().Set(ld if long_a == 'X' else fh)
                    light.GetHeightAttr().Set(fh if long_a == 'X' else ld)
                elif fmeta and fmeta.get('from_cluster'):
                    # Cluster bbox: dim0_m=X span, dim1_m=Y (or Z) span
                    d0 = max(float(fmeta['dim0_m']) / mpu, 0.05 / mpu)
                    d1 = max(float(fmeta['dim1_m']) / mpu, 0.05 / mpu)
                    light.GetWidthAttr().Set(d0)
                    light.GetHeightAttr().Set(d1)
                else:
                    light.GetWidthAttr().Set(float(0.5 / mpu))
                    light.GetHeightAttr().Set(float(0.5 / mpu))
            elif light_type == "CylinderLight":
                light = UsdLux.CylinderLight.Define(stage, light_path)
                if fmeta and fmeta.get('elongated'):
                    light.GetLengthAttr().Set(float(fmeta['long_dim']))
                    light.GetRadiusAttr().Set(float(fmeta['short_dim'] / 2))
                else:
                    light.GetLengthAttr().Set(float(1.2 / mpu))
                    light.GetRadiusAttr().Set(float(0.025 / mpu))
            else:  # SphereLight
                light = UsdLux.SphereLight.Define(stage, light_path)
                light.GetRadiusAttr().Set(float(0.15 / mpu))

            xform = UsdGeom.Xformable(light.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3d(fx, fy, fz))

            # Orient elongated lights by matching the actual LED face direction from vertices.
            # RectLight: use face_quat (derived from thinnest mesh axis = face normal).
            # CylinderLight: use world_quat (aligns cylinder long-axis with fixture orientation).
            if fmeta and fmeta.get('elongated'):
                if light_type == "RectLight" and fmeta.get('face_quat') is not None:
                    w, xi, yi, zi = fmeta['face_quat']
                    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
                        Gf.Quatd(w, Gf.Vec3d(xi, yi, zi)))
                elif fmeta.get('world_quat') is not None:
                    w, xi, yi, zi = fmeta['world_quat']
                    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
                        Gf.Quatd(w, Gf.Vec3d(xi, yi, zi)))

            # Rotate panel RectLights to match the lamp assembly's world orientation.
            # rot_up_deg is the angle of the assembly's local X axis in the world floor plane.
            # Without this, all panel lights are axis-aligned even when the lamp is rotated.
            if (light_type == "RectLight" and fmeta and fmeta.get('from_cluster')
                    and fmeta.get('rot_up_deg') is not None):
                if up_axis == "Z":
                    xform.AddRotateZOp().Set(float(fmeta['rot_up_deg']))
                else:
                    xform.AddRotateYOp().Set(float(fmeta['rot_up_deg']))

            if hints.get("cone"):
                light.GetPrim().AddAppliedSchema("ShapingAPI")
                light.GetPrim().CreateAttribute(
                    "inputs:shaping:cone:angle", Sdf.ValueTypeNames.Float).Set(30.0)
                light.GetPrim().CreateAttribute(
                    "inputs:shaping:cone:softness", Sdf.ValueTypeNames.Float).Set(0.2)

            apply_area_light_schemas(light.GetPrim(), light_power, color_temp=color_temp)
            created_lights.append(str(light_path))

    if not dry_run:
        log(f"  Created {len(created_lights)} lights.")

    _deactivated_n = len(deactivated_paths) if not dry_run else len(architectural_with_pos)
    _created_n     = len(created_lights)    if not dry_run else len(all_fixture_positions)
    prefix = "[DRY RUN] Would: " if dry_run else ""
    log(f"\n  Summary ({('dry run' if dry_run else 'applied')}):")
    log(f"    {prefix}deactivate+hide {_deactivated_n} existing lights")
    log(f"    {prefix}create          {_created_n} new lights at fixture positions")
    log(f"    excluded        {len(robot_lights)} robot/vehicle lights (not touched)")

    # --- DomeLight ---
    # Logic:
    # - has_glass (windows detected) or force_dome → need a DomeLight
    #   * If one already exists → upgrade it in place
    #   * If none exists → create a new one
    # - no glass, no force → no DomeLight needed
    #   * If existing DomeLights present → deactivate them (orphaned)
    dome_created = False
    need_dome = has_glass or force_dome

    log(f"\n=== DomeLight ===")
    if existing_dome_lights:
        log(f"  Existing DomeLight(s) in scene: {len(existing_dome_lights)}")

    if need_dome:
        if existing_dome_lights:
            # Upgrade existing DomeLight(s) — keep in place, apply physical schema
            for dome_lp in existing_dome_lights:
                if dry_run:
                    log(f"  [DRY RUN] UPGRADE existing DomeLight: {dome_lp.GetPath()}")
                    log(f"            -> {dome_illuminance} lux, {dome_color_temp}K")
                else:
                    apply_dome_light_schemas(dome_lp, illuminance=dome_illuminance,
                                             color_temp=dome_color_temp,
                                             set_version=False)  # preserve rotation
                    log(f"  UPGRADED DomeLight: {dome_lp.GetPath()} "
                        f"({dome_illuminance} lux, {dome_color_temp}K)")
            dome_created = True
        else:
            # No existing DomeLight — create one
            dome_path = "/World/Lights/Environment_DomeLight"
            if dry_run:
                log(f"  [DRY RUN] Would create DomeLight at {dome_illuminance} lux, {dome_color_temp}K")
                dome_created = True
            else:
                dome = UsdLux.DomeLight.Define(stage, dome_path)
                apply_dome_light_schemas(dome.GetPrim(),
                                         illuminance=dome_illuminance,
                                         color_temp=dome_color_temp)
                log(f"  Created DomeLight: {dome_illuminance} lux, {dome_color_temp}K")
                dome_created = True
    else:
        if existing_dome_lights:
            # No windows — deactivate existing DomeLights (no physical basis)
            log(f"  No glass/windows detected — deactivating existing DomeLight(s).")
            for dome_lp in existing_dome_lights:
                if dry_run:
                    log(f"  [DRY RUN] DEACTIVATE DomeLight (no windows): {dome_lp.GetPath()}")
                else:
                    dome_lp.SetActive(False)
                    log(f"  DEACTIVATED DomeLight (no windows): {dome_lp.GetPath()}")
        else:
            log(f"  No windows detected and no existing DomeLight — nothing to do.")

    # --- Save ---
    if dry_run:
        log(f"\n=== [DRY RUN] No files written ===")
    else:
        log(f"\n=== Saving ===")
        # Detect Kit: if omni.usd is available, we're inside Composer.
        # In that case, do NOT call layer.Save() — Composer has the stage
        # open and saving from the script triggers a "fetch changes" dialog.
        # Instead, let the user save via Ctrl+S after reviewing the result.
        _in_kit = False
        try:
            import omni.usd as _omni_check
            _in_kit = True
        except ImportError:
            pass

        if _in_kit:
            # Reset edit target so root layer is active again.
            stage.SetEditTarget(root_layer)
            log(f"  Running inside Kit — changes applied to live stage.")
            log(f"  Edit target reset to root layer.")
            # Use Kit's own save — saves all layers including the lighting
            # sublayer, without triggering Composer's "fetch changes" dialog.
            _omni_check.get_context().save_stage()
            log(f"  Stage saved via Kit (all layers including {layer_rel_path}).")
        else:
            # Standalone / VM — save to disk as usual.
            lighting_layer.Save()
            root_layer.Save()
            log(f"  Lighting layer: {layer_abs_path}")
            log(f"  Root layer updated (sublayer: {layer_rel_path})")

    # --- Summary ---
    log("\n" + "=" * 60)
    log(f"  {'[DRY RUN] ' if dry_run else ''}COMPLETE — v4.3")
    unique_powers = sorted({_power_high, _power_low})
    power_summary = (f"{unique_powers[0]} – {unique_powers[-1]} lm"
                     if len(unique_powers) > 1 else f"{unique_powers[0]} lm")
    log(f"  Fixture positions detected: {len(all_fixture_positions)}")
    log(f"  Existing lights deactivated: {_deactivated_n}{' [dry run]' if dry_run else ''}")
    log(f"  New lights created:          {_created_n}{' [dry run]' if dry_run else ''}")
    log(f"  Robot/vehicle excluded:      {len(robot_lights)}")
    log(f"  Power range:                {power_summary}")

    # Warn when per-fixture lumens exceed realistic indoor architectural limits.
    _max_power = max(_power_high, _power_low)
    if _max_power > 60_000:
        _n_fix = max(len(all_fixture_positions), 1)
        _m2_per_fix = _fl_area / _n_fix
        log("")
        if _max_power > 100_000:
            log("  !! UNREALISTIC LUMEN WARNING !!")
        else:
            log("  ! HIGH LUMEN WARNING")
        log(f"  {_max_power:,.0f} lm per fixture is outside the range of real indoor")
        log(f"  architectural lighting:")
        log(f"    Standard high-bay LED:  15,000 – 40,000 lm")
        log(f"    Maximum indoor LED:     ~60,000 – 100,000 lm (specialty only)")
        log(f"    This scene:             {_n_fix} fixture(s) covering {_fl_area:,.0f} m²")
        log(f"                            ({_m2_per_fix:,.0f} m² per fixture)")
        log(f"")
        log(f"  WHY: Too few fixture positions were detected relative to the scene size.")
        log(f"  The skill is compensating by making each fixture extremely bright.")
        log(f"")
        log(f"  WHAT TO DO:")
        log(f"  1. Check if the scene has ceiling fixture geometry with recognizable")
        log(f"     names (e.g. 'lamp', 'luminaire', 'ceiling_light', 'panel').")
        log(f"     If so, re-run — the skill will detect and use them.")
        log(f"  2. If no ceiling fixtures exist in the scene geometry, consider adding")
        log(f"     them in Omniverse, or re-run with --force-dome to add ambient")
        log(f"     sky contribution so fixture power can be reduced.")
        log(f"  3. Use --measured-lux after checking PT AOV Illuminance to calibrate")
        log(f"     the actual output — the math will be wrong but the result can still")
        log(f"     look correct in RTX.")
        log(f"  NOTE: RTX path tracing renders physically regardless of fixture count,")
        log(f"  so you can still get a good-looking result — but the scene would need")
        log(f"  ~{int(_fl_area / 30):,} fixtures at 25,000 lm each for realistic density.")

    if _dome_contrib_lux > 0:
        log("")
        log(f"  DOME CONTRIBUTION APPLIED")
        log(f"  Window hint:   {_dome_contrib_note}")
        log(f"  Fixture target reduced from {base_target_lux:.0f} lux → {effective_target_lux:.0f} lux")
        log(f"  The DomeLight is expected to supply the remaining ~{_dome_contrib_lux:.0f} lux.")
        log(f"  Validate in Omniverse: Debug View → PT AOV Illuminance")
        log(f"    Floor should read ~{base_target_lux:.0f} lux total (fixtures + dome combined).")
        log(f"  If the scene reads too dark:  use --window-size large  or increase --dome-contribution")
        log(f"  If the scene reads too bright: use --window-size small  or decrease --dome-contribution")

    log(f"  Color temperature:          {color_temp}K")
    log(f"  DomeLight:                  {'Yes' if dome_created else 'No'}")
    if dome_created:
        log(f"  Dome illuminance:           {dome_illuminance} lux")
    if not dry_run:
        scene_type = infer_scene_type(stage)
        t = target_lux or TARGET_LUX.get(scene_type, 300)
        log(f"\n  Next: Composer -> RTX Path Tracing -> Debug View -> PT AOV Illuminance")
        log(f"  Target: {t} lux on work surfaces ({scene_type})")
        _log_scene_camera(stage, log)
        _log_camera_recommendations(t, log)
    log("=" * 60 + "\n")

    # Write log to file if requested (useful in Kit Script Editor)
    if output_file:
        try:
            with open(output_file, "w") as f:
                f.write("\n".join(log_lines))
            print(f"Log written to: {output_file}")
        except Exception as e:
            print(f"Could not write log file: {e}")

    return {
        "fixture_count":    len(all_fixture_positions),
        "upgraded":         0,  # v4 deactivates+recreates; no in-place upgrade
        "deactivated":      len(deactivated_paths),
        "created":          len(created_lights),
        "robot_excluded":   len(robot_lights),
        "power_lm":         default_power,
        "dome_created":     dome_created,
    }


# ============================================================
# CLI ENTRY POINT — for VM agent and terminal use
# ============================================================
def main():
    args = parse_args()
    scene_path = os.path.abspath(args.scene)
    if not os.path.exists(scene_path):
        print(f"ERROR: Scene file not found: {scene_path}")
        sys.exit(1)
    stage = Usd.Stage.Open(scene_path)
    if not stage:
        print("ERROR: Could not open stage!")
        sys.exit(1)
    result = run_skill(
        stage,
        dry_run=args.dry_run,
        color_temp=args.color_temp,
        dome_color_temp=args.dome_color_temp,
        dome_illuminance=args.dome_illuminance,
        no_dome=args.no_dome,
        force_dome=args.force_dome,
        target_lux=args.target_lux,
        output_layer=args.output_layer,
        measured_lux=args.measured_lux,
        window_size=args.window_size,
        dome_contribution=args.dome_contribution,
    )
    if "error" in result:
        sys.exit(1)


try:
    # Kit Script Editor check — omni.usd only exists inside Omniverse/Kit.
    # This import succeeds in Kit and fails everywhere else.
    import omni.usd as _omni_usd

    # ── KIT CONFIG — edit these before hitting Run ───────────────
    _DRY_RUN          = True    # Set False to write files
    _COLOR_TEMP       = 4000.0  # Kelvin, fixture lights
    _DOME_COLOR_TEMP  = 6500.0  # Kelvin, dome light
    _DOME_ILLUMINANCE = 0.0     # lux, dome light (0 = auto: 400 lux sunset/clear sky)
    _FORCE_DOME       = False
    _NO_DOME          = False
    _TARGET_LUX       = None    # None = auto-detect from scene name
    _MEASURED_LUX     = None    # Set to PT AOV Illuminance reading to calibrate (e.g. 220.0)
    _REFLECTANCES     = None    # Set to (ceil, wall, floor) tuple to override sampling (e.g. (0.7, 0.5, 0.2))
    _WINDOW_SIZE      = None    # 'small'|'medium'|'large' — reduces fixture power when force_dome=True
    _DOME_CONTRIBUTION= None    # Direct override in lux for dome indoor contribution (e.g. 60.0)
    _OUTPUT_FILE      = None    # Set to a path like r"C:\Users\...\Desktop\output.txt" to save log
    # ─────────────────────────────────────────────────────────────

    _stage = _omni_usd.get_context().get_stage()
    if _stage:
        run_skill(
            _stage,
            dry_run=_DRY_RUN,
            color_temp=_COLOR_TEMP,
            dome_color_temp=_DOME_COLOR_TEMP,
            dome_illuminance=_DOME_ILLUMINANCE,
            force_dome=_FORCE_DOME,
            no_dome=_NO_DOME,
            target_lux=_TARGET_LUX,
            measured_lux=_MEASURED_LUX,
            reflectances=_REFLECTANCES,
            window_size=_WINDOW_SIZE,
            dome_contribution=_DOME_CONTRIBUTION,
            output_file=_OUTPUT_FILE,
        )
    else:
        print("No stage open in Composer. Open a USD scene first.")

except ImportError:
    # Not in Kit — run as CLI
    if __name__ == "__main__":
        main()

```

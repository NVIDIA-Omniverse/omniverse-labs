---
name: blender-community-skill-bootstrap
description: Locate and install optional community Blender production skills from their upstream GitHub repository instead of rebundling them in this catalog. Use when an agent needs detailed generic recipes for Blender modeling, materials, UVs, lighting, cameras, rendering, animation, export, full-scene orchestration, or multi-skill harmonization and those skills are not already installed.
---

# Blender community skill bootstrap

Use this catalog's `blender-python-execution` for the safe MCP/bpy transaction
contract. Obtain optional generic production recipes directly from their
upstream owner rather than copying them into this catalog.

## Upstream source

- Repository: `RobLe3/cc-blender-skill`
- Known reviewed revision: `11016c9a5847897491dde935c346571bd7548e3d`
- Upstream license: MIT; inspect the pinned
  `https://github.com/RobLe3/cc-blender-skill/blob/11016c9a5847897491dde935c346571bd7548e3d/LICENSE`
  before installing or redistributing it.
- Skill paths: `plugin/skills/<skill-name>`

Treat the revision as a reproducibility pin, not an endorsement or claim of
local authorship. If the user requests the latest upstream version, inspect its
current revision, license, and changed skill paths before installing it.

## Install only what the request needs

| Need | Upstream skill path |
| --- | --- |
| Natural-language Blender orchestration | `plugin/skills/text-to-blender` |
| Multi-skill precedence and handoffs | `plugin/skills/blender-skill-harmonizer` |
| Production ordering and critique | `plugin/skills/blender-pro-workflow` |
| Mesh creation and modifiers | `plugin/skills/blender-modeling` |
| Principled/PBR materials | `plugin/skills/blender-materials` |
| UVs, images, atlases, and baking | `plugin/skills/blender-uv-texturing` |
| Blender light rigs and worlds | `plugin/skills/blender-lighting` |
| Cameras and composition | `plugin/skills/blender-cameras` |
| Cycles/EEVEE renders | `plugin/skills/blender-rendering` |
| Keyframes, actions, and motion | `plugin/skills/blender-animation` |
| GLB, FBX, OBJ, STL, or USD export | `plugin/skills/blender-export` |

For a complete authored-scene request, install `text-to-blender`,
`blender-skill-harmonizer`, and only the focused production skills required by
the deliverable. Do not install the entire repository automatically.

## Codex installation workflow

1. Check whether each required skill is already available and preflight every
   destination basename before a multi-skill install. Do not overwrite an
   existing skill directory or silently replace a locally modified copy. If any
   destination conflicts, stop or install only the explicitly approved
   non-conflicting subset.
2. Tell the user which upstream repository, revision, paths, and license will be
   used. Installing requires network access and writes to the user's Codex skill
   directory.
3. Invoke `$skill-installer` and ask it to install from GitHub with:
   - repo: `RobLe3/cc-blender-skill`
   - ref: `11016c9a5847897491dde935c346571bd7548e3d`
   - path: the selected `plugin/skills/<skill-name>` entries
4. Let `skill-installer` request any required network approval and perform the
   installation. Do not reproduce its downloader in this skill.
5. Load `$skill-creator` and run its standard `quick_validate.py` against each
   installed skill folder. Resolve the helper through that system skill; do not
   assume its filesystem location. After a batch failure, inspect and report
   each destination separately because some skills may already have installed.
6. Tell the user the installed skills become available on the next turn. Do not
   claim they were used in the current turn unless they were already installed
   and loaded.

Equivalent helper invocation, for an environment that exposes the
`skill-installer` script path, is:

```text
install-skill-from-github.py \
  --repo RobLe3/cc-blender-skill \
  --ref 11016c9a5847897491dde935c346571bd7548e3d \
  --path plugin/skills/blender-modeling plugin/skills/blender-materials
```

Use `$skill-installer` rather than assuming the helper's filesystem location.

## Local execution and safety overlay

Upstream recipes do not override this catalog's current safety rules:

- preserve the user's scene and work in a caller-owned copy before destructive
  operations;
- execute through `blender-python-execution` as bounded, idempotent
  transactions with structured and visual checks;
- inspect Blender version compatibility before using version-specific RNA;
- never block Blender's main thread with `time.sleep()`;
- keep viewport screenshots distinct from native OVRTX evidence;
- use the documented OVRTX/OVPhysX add-on and native-client workflows rather
  than creating another bridge or transport.

If installation is declined or unavailable, continue only with locally
installed capabilities. State the missing production skill instead of inventing
an unreviewed Blender recipe or claiming that guidance-only instructions were
executed.

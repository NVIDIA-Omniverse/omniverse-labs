# Adding a New Sample to ov-labs

This guide explains how to create a well-structured sample in the `samples/` tree. Every sample lives in its own folder and must include a `README.md` with the YAML frontmatter block defined below.

---

## 1. Folder Structure

Create a directory under `samples/` using kebab-case matching your sample name:

```
samples/
└── your-sample-name/
    ├── README.md          # Required — includes YAML frontmatter (see below)
    ├── source/            # Python / C++ source code
    │   └── extension/     # Extension source, if applicable
    ├── skills/            # Agent-callable skill definitions
    ├── scripts/           # Standalone scripts, batch runners, OmniGraph nodes
    ├── data/              # USD stages, textures, meshes, and other asset files
    ├── config/            # .toml / .json / .yaml configuration files
    ├── docs/              # Screenshots, diagrams, supplemental docs
    └── tools/             # Build helpers, packaging scripts (optional)
```

**Rules:**
- `source/` holds all runnable code. No loose `.py` files in the root.
- `skills/` holds agent-callable skill definitions — required when `type: skill` or `skills_included: true`.
- `data/` holds all binary or large assets — keep source-controllable text assets (`.usda`) preferred over binary (`.usdc`/`.usdz`) where practical.
- `scripts/` is for batch scripts or standalone snippets that are not part of a full extension.
- Delete any scaffolded folders you do not use — empty directories add noise.

---

## 2. README Frontmatter

Every sample's `README.md` **must** open with this YAML block (between the `---` fences). Fill in every field; use `null` only if a field genuinely does not apply.

```yaml
---
# ──────────────────────────────────────────────────────────────────────────────
# Sample Metadata  —  required in every samples/<name>/README.md
# ──────────────────────────────────────────────────────────────────────────────

name: "Human-Readable Sample Name"

description: >
  One to three sentences. What does this sample do, and why would someone use it?
  Focus on the outcome, not the implementation.

version: "1.0.0"                  # SemVer; bump on breaking changes

type: snippet                     # See valid values below ↓

complexity: beginner              # beginner | intermediate | advanced

omniverse_libraries:              # List every Omniverse pip package used (e.g. ovrtx==0.2.0)
  - ovrtx
  # - ovphysx
  # - ovstage
  # - ovui
  # - ovstorage
  # - ovstream

omniverse_library_version_min: "2023.2.0"  # Earliest Omniverse library version tested against

tested_platforms:                 # Remove any that were NOT tested
  - Windows
  - Linux

tags:                             # Free-form keywords for search/filtering
  - usd
  - animation

assets_included: false            # true if data/ contains USD stages or media

skills_included: false            # true if scripts/ contains skill definitions

author: "Team or Individual Name"
maintainer: "team-alias@nvidia.com"
---
```

### `type` — Valid Values

| Value | When to use |
|---|---|
| `snippet` | A focused, self-contained code fragment demonstrating one API or technique. Typically < 100 lines. |
| `example` | A complete, runnable application or extension that shows end-to-end usage of one or more features. |
| `demo` | A polished, presentation-ready scene or workflow — meant to be shown, not modified. Often includes assets. |
| `template` | Boilerplate starter code intended to be copied and extended. Includes comments/TODOs guiding customization. |
| `tutorial` | Step-by-step learning sample with numbered stages and explanatory comments throughout the code. |
| `benchmark` | Measures and reports performance characteristics. Produces reproducible metrics. |
| `skill` | An agent-callable skill definition — a focused capability (tool, action, or workflow step) designed to be invoked programmatically rather than run standalone. |

---

## 3. README Body

After the frontmatter, structure your README with these sections (in order). Use only the sections that apply.

```markdown
# Sample Name

Short one-line tagline.

## Overview

Expand on the description: what problem does this solve, what will the user see or learn?

## Prerequisites

- Omniverse libraries >= 2023.2.0
- Python 3.10+
- Any required external downloads or licenses

## Setup

Step-by-step instructions to get the sample running. Prefer numbered lists.

1. Clone the repo (if not already): ...
2. Open a Kit application or `kit.exe` with: ...
3. Enable the extension: ...

## Usage

How to interact with the sample once it is running. Include screenshots in `docs/` and
reference them here: `![description](docs/screenshot.png)`

## Code Walkthrough  *(for snippet / tutorial types)*

Brief explanation of key files and the logic flow. Link to specific source files.

## Assets  *(if assets_included: true)*

Describe the contents of `data/` — what USD stages are included and how they are structured.

## Skills  *(if skills_included: true)*

List and describe any skill definitions in `skills/`.

## Known Limitations

Anything the user should be aware of that is intentionally out of scope.

## License

[Apache 2.0](../../LICENSE) unless otherwise noted in individual files.
```

---

## 4. Checklist Before Opening a PR

- [ ] Folder name is kebab-case and matches the `name` field (lowercased, spaces → hyphens)
- [ ] `README.md` frontmatter is present and all required fields are filled
- [ ] `type` is one of the valid values in the table above
- [ ] No loose files at the sample root (code in `source/`, assets in `data/`)
- [ ] Sample runs without errors on at least one listed platform
- [ ] `omniverse_library_version_min` and `tested_platforms` reflect what was actually tested
- [ ] No hardcoded absolute paths or machine-specific config values
- [ ] Large binary assets (> 5 MB) are discussed with a maintainer before committing

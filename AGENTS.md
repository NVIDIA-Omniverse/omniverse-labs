# AGENTS.md — omniverse-labs

Guidance for AI coding agents working in this repository.

## What this repo is

**omniverse-labs** is NVIDIA Omniverse’s public sketchpad: early samples, integration experiments, reference patterns, and proof-of-concept apps that compile and run but have **no support SLA**. Code may change without notice; presence here does not imply a product roadmap commitment.

**Audience:** developers evaluating or building on Omniverse libraries (`ov-render`, `ov-physics`, `ov-storage`, etc.) who want forkable starting points, not polished production SDKs.

**License:** Apache 2.0 ([LICENSE](LICENSE)).

---

## Repository layout

```
omniverse-labs/
├── README.md                 # Human-facing overview
├── AGENTS.md                 # This file
├── _unpublished/             # Draft articles & samples (NOT deployed to Pages)
├── samples/                  # Runnable samples (one folder per sample)
│   ├── README.md             # Sample catalog rules
│   └── NEW_SAMPLE_TEMPLATE.md  # Authoritative scaffold + frontmatter spec
├── docs/                     # Static GitHub Pages site (project showcase)
│   ├── README.md             # How to publish / preview the site
│   ├── data/projects.json    # Home grid metadata
│   ├── projects/<slug>/      # Per-project HTML articles
│   └── assets/               # CSS, JS, card images
└── .github/workflows/pages.yml  # Deploys docs/ to GitHub Pages on push to main
```

| Area | Purpose | When to touch |
|------|---------|----------------|
| `samples/` | Self-contained runnable code | Adding or fixing experiments, skills, scripts |
| `_unpublished/` | Draft placeholders (not on Pages) | WIP before explicit publish; see `_unpublished/README.md` |
| `docs/` | Public marketing/showcase site | New write-ups, cards on the home grid |
| Root `README.md` | Repo positioning | Rarely; keep in sync with major scope changes |

Samples and site projects are **independent**: a sample can exist without a `docs/` card, and a `docs/` article can link to external or future code.

---

## Working on samples

### Before you change code

1. Read the sample’s `README.md` (YAML frontmatter + body).
2. Note **Status**: `Working` | `Partial` | `Concept` — do not claim end-to-end success for `Concept` samples.
3. Install only the Omniverse pip packages listed in frontmatter (`omniverse_libraries`), using **pip package names** (e.g. `ovrtx`, `ovphysx`), not Kit extension IDs.
4. Respect platform notes in `tested_platforms`; many samples need a GPU or extra credentials (S3, NGC) — the README states this.

### Required sample structure

Follow [`samples/NEW_SAMPLE_TEMPLATE.md`](samples/NEW_SAMPLE_TEMPLATE.md). Summary:

- Folder: `samples/<kebab-case-name>/`
- **No loose files at sample root** — code in `source/`, assets in `data/`, config in `config/`, optional `skills/`, `scripts/`, `docs/`, `tools/`
- Every `README.md` **must** start with the YAML frontmatter block defined in the template
- **Self-contained** — no imports from other sample directories
- If something cannot run, set status to `Concept` and document what broke

### Sample types (`type` in frontmatter)

`snippet` | `example` | `demo` | `template` | `tutorial` | `benchmark` | `skill`

Use `skills/` and `skills_included: true` when the sample ships agent-callable skill definitions.

### Creating a new sample

```bash
# From repo root — copy scaffold if _template exists, else create layout manually
cp -r samples/_template samples/your-sample-name   # when _template is present
```

Then fill frontmatter, implement under `source/`, and complete the pre-PR checklist in `NEW_SAMPLE_TEMPLATE.md`.

### Agent-specific expectations

- Prefer **minimal diffs** that match existing layout and naming in sibling samples.
- Do not add cross-sample shared libraries at repo root without maintainer direction.
- Do not commit secrets (`.env`, API keys, machine-specific absolute paths).
- Binary assets **> 5 MB**: flag for maintainer review before adding.
- Update `version` in frontmatter on breaking changes (SemVer).
- When adding skills, keep definitions in `skills/` and document them under README **Skills**.

---

## Working on the docs site (`docs/`)

Static site; no build step beyond GitHub Pages uploading `docs/` as-is.

### Local preview

```bash
python -m http.server 8080 --directory docs
```

Open `http://localhost:8080/`. Cards load from `docs/data/projects.json` via `docs/assets/js/main.js`.

### Add a home-grid project or article

1. Copy `docs/projects/_template.html` → `docs/projects/<slug>/index.html`
2. Add an entry to `docs/data/projects.json` (`slug`, `title`, `subtitle`, `type`, `status`, `date`, `image`, `accent`, `links`, etc.)
3. Add `docs/assets/images/<card>.svg` (or PNG)
4. Keep contributor-only guides off the public grid — surface **about**, **samples**, and real experiments

**Project `type` values in JSON:** `article` | `project` | `sample` | `tool` | `concept` (see `TYPE_LABELS` in `main.js`).

**Status in JSON:** `working` | `partial` | `concept` (lowercase; drives badge CSS).

Do not add internal-only or placeholder cards to the home grid.

### Unpublished drafts (`_unpublished/`)

Draft articles and samples live under [`_unpublished/`](_unpublished/README.md). GitHub Pages **only** uploads `docs/`, so `_unpublished/` never appears on the public site.

**Agents and contributors must not:**

- Copy draft JSON from `_unpublished/articles/projects.json` into `docs/data/projects.json` unless publishing.
- Copy draft HTML into `docs/projects/` unless publishing.
- Move `_unpublished/samples/*` into `samples/` unless publishing.

**To publish**, follow the explicit checklists in [`_unpublished/README.md`](_unpublished/README.md) (move folders, merge one JSON object, add card image, preview locally).

---

## Commands agents commonly need

| Task | Command / location |
|------|-------------------|
| Preview site | `python -m http.server 8080 --directory docs` |
| Run a sample | See that sample’s README (`python …`, Kit, etc.) |
| Install Omniverse libs | `pip install <package>` per sample README |
| Site deploy | Automatic on push to `main` via `.github/workflows/pages.yml` |

There is no monorepo `package.json` or root `Makefile` — each sample documents its own run path.

---

## Git and PR conventions

- **Do not commit** unless the user explicitly asks.
- **Do not push** or open PRs unless asked.
- Match existing commit style from `git log` when committing.
- Never skip hooks (`--no-verify`) unless the user requests it.
- Prefer focused PRs: sample code + its README, or docs site changes, described clearly in the PR body.

---

## What not to do

- Treat samples as stable APIs or promise backward compatibility.
- Use this repo as a stand-in for production support (point users to library repos / NVIDIA Developer Forums).
- Organize samples by date — organize by **what they demonstrate**.
- Add broken samples without `Concept` status and documented limitations.
- Put runnable code only in `docs/` HTML — runnable code belongs under `samples/`.
- Publish `_unpublished/` content by copying it into `docs/` or `samples/` without using the publish checklist — that exposes WIP on GitHub Pages or in the public samples tree.

---

## Quick reference links

| Document | Use for |
|----------|---------|
| [README.md](README.md) | Repo purpose and audience |
| [samples/README.md](samples/README.md) | Sample README requirements (status, libraries, architecture) |
| [samples/NEW_SAMPLE_TEMPLATE.md](samples/NEW_SAMPLE_TEMPLATE.md) | Folder layout, frontmatter fields, PR checklist |
| [docs/README.md](docs/README.md) | GitHub Pages setup and publishing steps |

---

## Success criteria for agent tasks

- **Sample work:** README frontmatter complete, structure matches template, status honest, runs on at least one declared platform (or marked `Concept` with explanation).
- **Site work:** `projects.json` valid JSON, new page paths match `links.page`, images resolve from `docs/`, local preview shows the new card.
- **Fixes:** Root cause addressed with smallest change; no unrelated refactors.

When unsure about scope (new sample vs. site card vs. both), ask the user or follow the checklist in `NEW_SAMPLE_TEMPLATE.md` and `docs/README.md`.

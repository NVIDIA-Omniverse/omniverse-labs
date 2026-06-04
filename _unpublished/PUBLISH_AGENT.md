# Agent script: publish `_unpublished/` → GitHub Pages

Copy the **Teammate prompt** block below into Cursor (or any coding agent) after cloning [ov-labs](https://github.com/nvidia-omniverse/omniverse-labs). Replace the placeholders with the item you are publishing.

Authoritative human docs: [`_unpublished/README.md`](README.md), [`docs/README.md`](../docs/README.md), [`AGENTS.md`](../AGENTS.md).

---

## Teammate prompt (copy from here)

```text
You are working in the ov-labs repository. Publish ONE item from `_unpublished/` to the public tree so it appears on GitHub Pages after merge to `main`.

**Target**
- Kind: [article | sample]
- Slug / folder name: <kebab-case-slug>   (e.g. `ptc-article`, `ptc-demo`)
- Add home-grid card: [yes | no]           (articles: usually yes; samples: only if we want a site write-up)
- Remove draft copy after publish: [yes | no]   (default: yes once the public copy is complete)

**Rules**
- Read and follow `_unpublished/PUBLISH_AGENT.md` end to end before editing.
- Read `_unpublished/README.md`, `docs/README.md`, and `AGENTS.md` for repo conventions.
- Do NOT commit or push unless I explicitly ask.
- Do NOT copy the entire `_unpublished/articles/projects.json` into `docs/data/projects.json` — merge ONE project object only.
- Do NOT add internal-only or placeholder cards (no “how to publish” cards on the home grid).
- Honest status: use `working` | `partial` | `concept` in JSON (lowercase); sample README **Status** must match reality.

**When done, report**
1. Files created/moved/deleted (paths)
2. Whether local preview was run and what you checked
3. Suggested PR title and 2–3 sentence summary
4. Anything still blocked (missing card art, content stubs, sample not runnable, etc.)
```

---

## Agent instructions (follow in order)

### 0. Confirm scope

1. List `_unpublished/articles/` and `_unpublished/samples/` and verify the requested slug exists.
2. If the slug is missing, stop and tell the user how to add a placeholder (see `_unpublished/README.md` → “Add another unpublished placeholder”).
3. Read the draft:
   - **Article:** `_unpublished/articles/<slug>/index.html` and the matching object in `_unpublished/articles/projects.json`.
   - **Sample:** `_unpublished/samples/<slug>/` (especially `README.md` frontmatter and **Status**).

### 1. Publish an **article** (site page + optional home card)

**Goal:** `docs/projects/<slug>/index.html` is served on GitHub Pages; optional card in `docs/data/projects.json`.

| Step | Action |
|------|--------|
| 1 | Finish article HTML. If the draft is a stub, use `docs/projects/_template.html` as the shell and merge in real content from the draft. |
| 2 | Ensure paths work from `docs/projects/<slug>/`: `href="../../assets/css/main.css"`, `data-base="../../"`, images `../../assets/images/...`, home link `../../`. |
| 3 | Copy the corporate policy `<nav>` from `docs/assets/partials/corporate-footer-policies.html` into the page footer (match `docs/projects/about-ov-labs/index.html` or `_template.html`). |
| 4 | Write the page to `docs/projects/<slug>/index.html` (create the directory). |
| 5 | If adding a home card: copy **one** object from `_unpublished/articles/projects.json` → append to `projects` array in `docs/data/projects.json`. Update `title`, `subtitle`, `status`, `date`, `image`, `links.page` (`projects/<slug>/`). Remove “placeholder” / “draft” wording from subtitle. |
| 6 | Add `docs/assets/images/<card-name>.svg` (or PNG) if not reusing an existing card; set `image` in JSON to `assets/images/<card-name>.svg`. |
| 7 | Add a `<url>` block to `docs/sitemap.xml`: `https://nvidia-omniverse.github.io/omniverse-labs/projects/<slug>/` with `<lastmod>` = project `date` from JSON. |
| 8 | Remove “UNPUBLISHED” / draft-only comments from the published HTML. |
| 9 | If user asked to remove draft: delete `_unpublished/articles/<slug>/` and remove that project object from `_unpublished/articles/projects.json`; update the table in `_unpublished/README.md` if that slug was listed. |

**Article JSON reference**

- `type`: `article` (or `project` / `sample` / `tool` / `concept` per `docs/assets/js/main.js` `TYPE_LABELS`)
- `status`: `working` | `partial` | `concept` (lowercase)
- `links.page`: `projects/<slug>/` (no leading slash)
- `featured`: usually `false` unless the user requests it

### 2. Publish a **sample** (runnable code in `samples/`)

**Goal:** `samples/<slug>/` exists with valid layout per `samples/NEW_SAMPLE_TEMPLATE.md`.

| Step | Action |
|------|--------|
| 1 | Verify folder layout: code under `source/`, assets under `data/`, no loose files at sample root except `README.md`. |
| 2 | Update README frontmatter and body: remove “Unpublished” / `_unpublished/` warnings; fix license link to `../../LICENSE` from `samples/<slug>/`. |
| 3 | Set **Status** to `Working` or `Partial` only if the sample actually runs on a declared platform; otherwise keep `Concept` and document what is broken. |
| 4 | **Move** (preferred) or copy the entire folder to `samples/<slug>/` with the same kebab-case name. |
| 5 | Delete `_unpublished/samples/<slug>/` when the move is complete; update `_unpublished/README.md` table if listed. |
| 6 | **Optional site card** (only if user requested): follow **§1** article steps using `type: "sample"` in JSON, write-up under `docs/projects/<slug>/`, link to repo in `links.repo` if appropriate. |

Samples do **not** automatically appear on the home grid; a card is optional.

### 3. Validate before handoff

Run from repo root:

```bash
python -m http.server 8080 --directory docs
```

Open `http://localhost:8080/` and confirm:

- [ ] `docs/data/projects.json` is valid JSON (no trailing commas; only intended new entry added).
- [ ] New card appears on the home grid (if a card was added).
- [ ] Article page loads; CSS and card image resolve (no 404 in network tab).
- [ ] Footer includes NVIDIA policy links.
- [ ] `sitemap.xml` includes the new project URL (if a site page was published).

For samples, skim `samples/<slug>/README.md` against `samples/NEW_SAMPLE_TEMPLATE.md` checklist.

### 4. Git / PR (only when the user asks)

- Do not commit unless explicitly requested.
- Suggested PR title pattern: `docs: publish <slug> article` or `samples: publish <slug>`.
- PR body should list: what moved from `_unpublished/`, new paths, preview steps, honest status, and any follow-ups (card artwork, runnable gaps).

---

## Path cheat sheet

| Draft location | Published location | Home grid |
|----------------|-------------------|-----------|
| `_unpublished/articles/<slug>/index.html` | `docs/projects/<slug>/index.html` | Entry in `docs/data/projects.json` |
| `_unpublished/samples/<slug>/` | `samples/<slug>/` | Optional: also `docs/projects/<slug>/` + JSON |

GitHub Pages deploys **only** `docs/`. The `samples/` tree is on GitHub.com but not on the site unless you add a card and project page.

---

## Do not

- Merge `_unpublished/articles/projects.json` wholesale into `docs/data/projects.json`.
- Publish placeholder subtitles (“Draft article — not on the public site…”) to the live grid.
- Claim `working` / `Working` when content or code is still a stub.
- Add publishing guides or internal docs as home-grid projects.
- Commit secrets, `.env`, or machine-specific absolute paths.

---

## Current placeholders (update when publishing)

See the table in [`_unpublished/README.md`](README.md). After you publish an item, remove its row from that table if the draft folder was deleted.

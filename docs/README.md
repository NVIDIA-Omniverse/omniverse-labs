# Omniverse Labs site (GitHub Pages)

Static site served from this `docs/` folder — a public project showcase inspired by [Shopify Spatial Commerce Projects](https://shopify.github.io/spatial-commerce-projects/). NVIDIA teams publish here for external developers; contributor steps stay in this README, not on the home grid.

## Local preview

From the repo root:

```bash
# Python 3
python -m http.server 8080 --directory docs
```

Open [http://localhost:8080/](http://localhost:8080/).

## Enable GitHub Pages (once)

1. Repo **Settings** → **Pages**
2. **Build and deployment** → Source: **Deploy from a branch**
3. Branch: **`main`**, folder: **`/docs`**
4. Merge to `main`; GitHub publishes the contents of this folder directly

Alternatively, use **GitHub Actions** as the source and keep `.github/workflows/pages.yml` (it runs `scripts/sync-docs-site.py` before upload).

Live URL (default): `https://nvidia-omniverse.github.io/omniverse-labs/`

## How `docs/` relates to `articles/` and `samples/`

GitHub Pages **only serves files inside `docs/`**. Source markdown and runnable code stay at the repo root:

| Repo path | Role on the site |
|-----------|------------------|
| `articles/` | Authoring source; article HTML and card images are copied into `docs/` |
| `samples/` | Runnable code; sample overview pages and card images are copied into `docs/` |
| `docs/data/projects.json` | Home grid card definitions (loaded by `assets/js/main.js`) |
| `docs/projects/<slug>/` | Public article or sample overview pages |

Before you push site changes, sync assets from the repo root into `docs/`:

```bash
python scripts/sync-docs-site.py
git add docs/
git commit -m "sync docs site from articles/ and samples/"
git push
```

The sync script reads `docs/data/content-manifest.json` and copies card images plus article assets into `docs/assets/images/`. It also bumps `site.dataVersion` in `projects.json` so browsers fetch fresh card data.

## Publish a new project or article (contributors)

Contributor steps are documented here (not promoted on the public home grid):

Summary:

1. Add your article under `articles/<name>/` or sample under `samples/<name>/`
2. Copy `projects/_template.html` → `projects/<slug>/index.html`
3. Add an entry to `data/projects.json` and `data/content-manifest.json`
4. Run `python scripts/sync-docs-site.py` from the repo root (copies images into `docs/`)
5. Add `assets/images/<your-card>.svg` (or PNG) if not produced by sync
6. Add a `<url>` entry to `sitemap.xml` (use the project `date` for `<lastmod>`)
7. Copy the corporate policy `<nav>` from `assets/partials/corporate-footer-policies.html` into the page footer (required on every public page; use localized nvidia.com URLs if you ship translated regions)
8. Open a PR

Do not add internal-only cards (publishing guides, placeholders). The home grid should surface **about**, **samples**, and real experiments as they land.

## Draft content (`_unpublished/`)

Work-in-progress articles and samples live in [`../_unpublished/`](../_unpublished/README.md) at the repo root. That folder is **not** part of the Pages deploy. Do not add draft entries to `data/projects.json` until you follow the publish checklist in `_unpublished/README.md`.

# Unpublished placeholders (not on GitHub Pages)

Work-in-progress **articles** and **samples** live here until someone explicitly publishes them.

**Agent-assisted publish:** Share [`PUBLISH_AGENT.md`](PUBLISH_AGENT.md) with your teammate — it contains a copy-paste Cursor prompt and a full checklist for moving one slug from `_unpublished/` to `docs/` and/or `samples/`.

## Why this folder exists

GitHub Pages deploys only the [`docs/`](../docs/) tree. Nothing under `_unpublished/` is served on the public site, and nothing here is linked from [`docs/data/projects.json`](../docs/data/projects.json) (the home grid).

**For humans and AI agents:** Do **not** add entries from this folder to `docs/data/projects.json`, and do **not** copy HTML into `docs/projects/` until the content is ready. Editing placeholders here is fine; promoting them without review is not.

| Location | Visible on GitHub.com (repo) | Visible on GitHub Pages (public site) |
|----------|------------------------------|----------------------------------------|
| `_unpublished/` | Yes (clone / browse) | **No** |
| `docs/projects/` + `projects.json` | Yes | **Yes** |
| `samples/<name>/` | Yes | Only if you add a site card |

---

## Placeholders in this tree

| Kind | Slug / folder | Owner action |
|------|---------------|--------------|
| Article | [`articles/ovrtx-agent-ready/`](articles/ovrtx-agent-ready/) | Publish article (below) |
| Article | [`articles/ptc-article/`](articles/ptc-article/) | Publish article (below) |
| Sample | [`samples/ov-hackathon-00/`](samples/ov-hackathon-00/) | Publish sample (below) |
| Sample | [`samples/ov-hackathon-01/`](samples/ov-hackathon-01/) | Publish sample (below) |
| Sample | [`samples/ov-hackathon-02/`](samples/ov-hackathon-02/) | Publish sample (below) |
| Sample | [`samples/ptc-demo/`](samples/ptc-demo/) | Publish sample (below) |

Draft home-grid metadata for the two articles is in [`articles/projects.json`](articles/projects.json) — copy **one object at a time** into `docs/data/projects.json` when publishing.

---

## Publish an article (explicit checklist)

1. Finish content in `_unpublished/articles/<slug>/index.html` (or rewrite after copy).
2. **Move or copy** the folder to `docs/projects/<slug>/index.html` (paths in HTML must use `../../assets/...` like other project pages).
3. Copy the matching object from `_unpublished/articles/projects.json` into `docs/data/projects.json` — update `title`, `subtitle`, `status`, `date`, and `links.page` if needed.
4. Add `docs/assets/images/<card>.svg` (or PNG) and set `image` in the JSON entry.
5. Preview: `python -m http.server 8080 --directory docs` → confirm the card appears and the article loads.
6. Open a PR. Optionally delete the `_unpublished/articles/<slug>/` copy after merge if you no longer need the draft.

---

## Publish a sample (explicit checklist)

1. Implement code under `_unpublished/samples/<name>/` (`source/`, `data/`, etc.) per [`samples/NEW_SAMPLE_TEMPLATE.md`](../samples/NEW_SAMPLE_TEMPLATE.md).
2. Set **Status** in the README to `Working` or `Partial` when it truly runs (keep `Concept` until then).
3. **Move** the entire folder to `samples/<name>/` (same kebab-case name).
4. Optional: add a home-grid card — copy article publish steps using `type: "sample"` in `projects.json` and a `docs/projects/<slug>/` write-up if you want a site page.
5. Open a PR. Remove the `_unpublished/samples/<name>/` folder when the move is complete.

---

## Add another unpublished placeholder

1. For an article: copy `docs/projects/_template.html` → `_unpublished/articles/<slug>/index.html`, add a stub object to `_unpublished/articles/projects.json`, list it in the table above.
2. For a sample: copy an existing `_unpublished/samples/*` folder, rename, update README frontmatter, list it in the table above.

Never append draft JSON directly to `docs/data/projects.json`.

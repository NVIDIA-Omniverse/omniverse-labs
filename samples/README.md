# Samples

Each sample lives in its own directory. Browse the folders below to explore what's available, or follow the guide to contribute a new one.

**Draft samples** (not listed here until published) live in [`../_unpublished/samples/`](../_unpublished/README.md). Move a folder into `samples/` only when ready — see `_unpublished/README.md`.

---

## Creating a new sample

Use [`NEW_SAMPLE_TEMPLATE.md`](NEW_SAMPLE_TEMPLATE.md) as your guide. It covers:

1. **Folder structure** — where source, skills, scripts, data, config, and docs go
2. **README frontmatter** — required YAML block with name, type, libraries, version, and more
3. **README body sections** — standard ordering (Overview, Prerequisites, Setup, Usage, etc.)
4. **Pre-PR checklist** — what reviewers will check before merging

To get started, copy the folder scaffold and fill in your sample's `README.md` frontmatter:

```bash
cp -r _template samples/your-sample-name
```

Or create the structure manually following the layout in `NEW_SAMPLE_TEMPLATE.md`.

---

## Every sample README must include

### Status

First line after the title — be honest:

```
**Status:** Working | Partial | Concept
```

- **Working** — runs end-to-end on at least one tested platform.
- **Partial** — some steps work; known gaps are documented.
- **Concept** — does not run; here for reference or WIP.

### Libraries used

List every Omniverse library the sample touches with its role. Use the pip package name (e.g. `ovrtx`, `ovphysx`):

| Library | Role in this sample |
|---|---|
| `ovrtx` | Renders the USD scene via RTX ray tracing |
| `ovphysx` | Runs the rigid-body simulation |

If the sample uses Warp, Newton, or an external SDK, include those too.

### Architecture

A diagram or a short written walkthrough — whichever is clearer for this sample. Save diagrams to `docs/architecture.png` (or `.svg`) and embed them:

```
![Architecture](docs/architecture.png)
```

If you write it out, describe the data flow: what goes in, what each library does with it, what comes out. Skip this section only if the sample is trivially single-library.

### Extending this sample

Tell the developer what's easy to change and what isn't:

- Which file or function to edit to swap in a different USD scene, model, or dataset
- How to connect a different Omniverse Library in place of one used here
- Any known limitations that would require non-trivial rework to get past

---

## Rules

- **Self-contained.** Each sample directory must run on its own. No cross-sample imports.
- **No broken samples.** If a sample stops working and you can't fix it, mark it `Concept` and note what broke.
- **Frontmatter required.** Every sample `README.md` must open with the YAML block defined in `NEW_SAMPLE_TEMPLATE.md`.
- **Use pip package names.** Reference Omniverse libraries by their pip name (`ovrtx`, `ovphysx`, etc.), not kit extension IDs.
- **No loose files at the root.** Code goes in `source/`, skills in `skills/`, assets in `data/`. See `NEW_SAMPLE_TEMPLATE.md` for the full layout.


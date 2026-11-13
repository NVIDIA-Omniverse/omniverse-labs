# Semantic labels

The campus carries [SimReady](https://github.com/NVIDIA/simready-foundation)
semantic labels so that renderer output can become ML-ready ground truth (for
example segmentation and bounding boxes written by `ovannotators`). Labels live
in a sparse override layer, `data/scene-overrides/r52-semantic-labels.usda`. The
application composes it with the other scene corrections. Campus artwork is never
edited.

## What is labeled

| SemanticsLabelsAPI instance | Applied to | Values | SimReady requirement |
| --- | --- | --- | --- |
| `wikidata_qcode` | Objects (highest labeled prim; descendants inherit) | [Wikidata](https://www.wikidata.org) Q-codes, for example `Q42948` (wall) | SL.001, SL.003, SL.QCODE.001 |
| `wikidata_class` | The same prims | The Q-code's Wikidata English label, for example `wall` | Convention of NVIDIA SimReady assets |
| `material` | Materials bound by rendered geometry | Material words, for example `wood` | SL.MAT.001 |

All values are static (SL.TIME.001). The closed vocabulary, including each
class's Q-code, is in `config/semantic-taxonomy.json`. The labeling follows the
convention of the SimReady exhibit assets, which already carry their own labels.
The tools detect those labels and never relabel them.

Models the application references at run time (the car, the IRB4600 robot arms
and the robot guide) are labeled where they are referenced, by
`source/nvidia/ov_library_lab/semantics.py`, using the same taxonomy.

## How labels are produced

The tools in `tools/semantic_labeler/` run as a pipeline. Each stage writes to
`data/semantics/`, which Git ignores.

1. `enumerate_prims.py <stage>`: walks the stage and applies the deterministic
   rules in `config/semantic-rules.json`. Rules match prim-name words, referenced
   assets and bound materials. Material rules label rendered materials. The
   remaining object-sized prims become *units*. Units that are instances of the
   same object share a group.
2. `render_crops.py <labels.json>`: draws a geometry thumbnail of each group
   (the unit in orange among its neighbors). It needs no GPU.
3. `vlm_label.py <labels.json>`: asks a vision-language model to choose a class
   for each group from the closed vocabulary, using the thumbnail and the prim's
   name, size and materials. Low-confidence or `unknown` answers go to
   `needs_review`. Answers are cached, so reruns only pay for new groups.
4. `review.py <labels.json>`: writes an HTML review sheet. Record corrections in
   `config/semantic-overrides.json`; overrides win over rules and the model.
5. `author_layer.py <labels.json>`: writes the override layer.
6. `coverage.py <stage> --layer <layer>`: checks SL.001, SL.QCODE.001,
   SL.MAT.001 and SL.TIME.001 and lists the largest unlabeled geometry.
7. `verify_qcodes.py`: confirms every taxonomy Q-code resolves on wikidata.org
   to its recorded label. Run it after you edit the taxonomy.

Model proposals are suggestions until someone reviews them. Rules and reviewed
overrides are the source of truth.

### Running the VLM stage

`vlm_label.py` works with any OpenAI-compatible chat-completions endpoint whose
model accepts images. Configure it with environment variables:

| Variable | Meaning |
| --- | --- |
| `VLM_API_URL` | Full chat-completions URL, for example `https://<host>/v1/chat/completions` |
| `VLM_MODEL` | Model ID served by that endpoint |
| `VLM_API_KEY` | Bearer token. It is read from the environment and never written to disk |
| `VLM_EXTRA_BODY` | Optional JSON merged into each request. For example, `{"chat_template_kwargs": {"enable_thinking": false}}` turns off a reasoning model's thinking, which makes answers faster and cheaper |

The model receives prim names, sizes, material names and geometry thumbnails.
Before you send unpublished assets to a hosted model, confirm that this is
allowed.

### Regenerating the layer

```bash
python tools/semantic_labeler/enumerate_prims.py data/campus/<revision>/campus-lab.usda
python tools/semantic_labeler/render_crops.py data/semantics/campus-lab.labels.json
python tools/semantic_labeler/vlm_label.py data/semantics/campus-lab.labels.json
python tools/semantic_labeler/review.py data/semantics/campus-lab.labels.json
python tools/semantic_labeler/author_layer.py data/semantics/campus-lab.labels.json
python tools/semantic_labeler/coverage.py data/campus/<revision>/campus-lab.usda --layer data/scene-overrides/r52-semantic-labels.usda
```

If you add overrides, rerun `enumerate_prims.py` and `vlm_label.py` (cached),
then `author_layer.py`.

## Limits

- Geometry that the application builds procedurally at run time (room floors,
  placards, fixtures and their materials) is not labeled yet. The campus,
  referenced models and SimReady exhibits are labeled.
- Labels from the model are not yet reviewed. Treat them as proposals until
  review records corrections in `config/semantic-overrides.json`.
- Thumbnails show shape and context, not appearance. An `ovrtx` render saved
  under the same file name in `data/semantics/crops/` can replace a thumbnail.
- `coverage.py` is a quick local check. Use the official `simready-validate`
  tool before claiming SimReady conformance.

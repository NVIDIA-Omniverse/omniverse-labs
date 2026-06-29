# Tutorial: walk the graph, generate a parser feature

A hands-on introduction to working in `usd-developer-skillgraph`. By the end you
will have run the core loop yourself — *something fails, you reach for the right
skill and contract, and you watch it pass* — and you'll know the shape of how to
make your own contribution.

**Time:** ~10 minutes. **Prerequisites:** Python 3 (`py` on Windows, `python3`
on Linux/macOS). No build step, no agent required for the first exercise.

---

## 1. The mental model

This component does not contain a USD implementation. It contains the *durable,
human-curated inputs* an agent uses to **generate** one — and the deterministic
harness that judges what the agent produced.

```
        USD Core Spec   (the durable input, pinned to v1.0.1)
                |
                v
   +-------------------------------------------------+
   |  the skill graph   (durable, human-curated)      |
   |    skills/      how to build each piece          |
   |    contracts/   what "correct" means             |
   |    goldens/     executable proof of correctness  |
   +-------------------------------------------------+
                |   an agent walks the graph
                v
        generated/   (disposable, .gitignored code)
                |   harness/score.py  +  harness/regen_graph.py
                v
        deterministic PASS / FAIL against the goldens
```

The three durable artifacts are the asset; the generated code is disposable. You
improve the project by improving **skills, contracts, and goldens** — not by
hand-editing generated code. Everything below makes that concrete.

---

## 2. Anatomy: a handful of skills and contracts

The unit of the graph is a **node**: a skill plus the contracts and goldens that
bound it. A **scope** names a set of nodes that compose into a deliverable. We'll
use the smallest end-to-end scope, `usda-single-layer` — open one USDA text
layer and emit its document model. Open `graph/scopes/usda-single-layer.yaml`:

```yaml
goal: >
  Generate a target implementation that opens one resolved USDA text layer
  resource, returns a target-native document-model Layer, and emits canonical
  layer dump JSON only through the validation adapter.
required_nodes:
  - usd-identifiers-and-names
  - usd-foundational-values
  - usd-paths
  - usd-listops-authored
  - usd-document-model
  - usda-lexical-format
  - usda-value-parser
  - usda-spec-parser
  - usd-resource-protocol
  - usda-layer-open
  - usd-layer-open
```

Eleven *required* nodes, in dependency order. (When you run `regen_graph.py` in
§4, the expanded dependency closure shows **twelve**: it pulls in `usd-tokens`,
which several of these nodes consume but the scope doesn't list explicitly.)
Three of the required nodes, top to bottom:

**`usd-identifiers-and-names`** (no dependencies) — the foundation. Open
`skills/usd-identifiers-and-names/SKILL.md`. A skill is prose, structured into:
*Spec Sources* (which pinned spec sections it derives from), *Provides* (what it
owns — prim names, property names, the identifier-scanner capability…),
*Contract* (the binding rules — "Identifiers are UTF-8 / Unicode code point
sequences constrained by the spec's XID start/continue rules"), *Boundary
Guards* (what it must **not** do — "Defer path parsing to `usd-paths`"), and
*Test Obligations* (what proves it correct, ending in a golden file). The
machine-checkable half lives in `contracts/capabilities/identifier-scanner.json`.

**`usd-document-model`** (depends on tokens, identifiers, values, paths, listOps)
— layer/spec/field storage after syntax is accepted. Its `SKILL.md` Contract
owns `contracts/handles/document-model.handle.json` and states a rule we'll use
in a moment: *"The parser must keep child-list fields in agreement with created
specs"* and *"Required field authoring for parsed specs."*

**`usda-spec-parser`** (depends on the lexer, value parser, document model, paths)
— the node that actually walks USDA text and builds the document-model layer.

How do these compose into a checkable result? Through a **golden**. Open
`goldens/integration/usda-single-layer/basic.json` and find the `one-root-prim`
case — input USDA on the left, expected document model on the right:

```jsonc
"input":  { "usda": "#usda 1.0\n\ndef Xform \"World\"\n{\n}\n" },
"expected": {
  "ok": true, "diagnostics": [],
  "layer": {
    "fields": { "primChildren": {"type": "token[]", "value": ["World"]} },
    "specs": {
      "/":      { "kind": "layer", "fields": { "primChildren": {"type": "token[]", "value": ["World"]} } },
      "/World": { "kind": "prim",  "fields": {
        "specifier":          {"type": "specifier", "value": "def"},     // <- usd-document-model (specifier domain value)
        "typeName":           {"type": "token", "value": "Xform"},        // <- usd-identifiers-and-names + tokens
        "primChildren":       {"type": "token[]", "value": []},           // <- usd-document-model (child-list maintenance)
        "propertyChildren":   {"type": "token[]", "value": []},           // <- usd-document-model
        "variantSetChildren": {"type": "token[]", "value": []}            // <- usd-document-model
      } }
    }
  }
}
```

Every expected field traces back to the skill that owns it. The golden is the
contract made executable — `contracts/layer-dump.schema.json` defines the *shape*
of that JSON, and the golden pins the *values* for a specific input.

---

## 3. Exercise: something fails → use the contract → it passes

You don't need an agent for this. We ship a deliberately tiny, hand-written
stand-in parser so you can run the loop yourself in one sitting:
`docs/tutorial/mini_dump.py`. (This is **not** how the real parser is built — see
§4, *The real workflow*, below — it's a ~50-line teaching prop that only
understands the two shapes in `docs/tutorial/hello.golden.json`.)

**Score it against the golden.** From the repo root:

```bash
py harness/score.py --goldens docs/tutorial/hello.golden.json \
  --dump-cmd "py docs/tutorial/mini_dump.py"            # Windows
# python3 harness/score.py --goldens docs/tutorial/hello.golden.json \
#   --dump-cmd "python3 docs/tutorial/mini_dump.py"     # Linux / macOS
```

You'll see something fail:

```text
Scoring suite: tutorial-hello
Cases: 2
PASS empty-layer
FAIL one-root-prim: $.layer.specs./World.fields: key mismatch missing=['primChildren', 'propertyChildren', 'variantSetChildren'] extra=[]
OVERALL: 1/2 (50.0%)
```

The harness tells you *exactly* what's wrong: the `/World` prim spec is missing
its child-list fields. **That's the contract talking.** Don't guess — go read the
owner. `skills/usd-document-model/SKILL.md` says, under *Provides* and *Contract*:
"Child list maintenance" and "Required field authoring for parsed specs." Every
parsed prim must author `primChildren`, `propertyChildren`, and
`variantSetChildren`, even when empty.

**Make the fix.** Open `docs/tutorial/mini_dump.py`, find the `TUTORIAL TODO`
block, and author the three child lists the contract requires:

```python
"typeName": {"type": "token", "value": type_name},
"primChildren": token_array([]),
"propertyChildren": token_array([]),
"variantSetChildren": token_array([]),
```

**Re-run the score:**

```text
PASS empty-layer
PASS one-root-prim
OVERALL: 2/2 (100.0%)
```

Green. You just did, by hand and on a micro-slice, exactly what the graph does at
scale: a golden defined correctness, a contract said where the answer lived, and
the harness verified the result deterministically — no human judgment in the
loop.

---

## 4. The real workflow: let your agent generate it

The stand-in above cuts every corner. The real `usda-single-layer` parser is
**generated by a coding agent** from the scope's full subgraph (the twelve-node
closure from §2), and a generated target passes all six cases in `basic.json`
(see `docs/first-python-target.md`).

> **On a fresh clone this section does not run yet — by design.** Two directories
> are intentionally absent until you create them: `generated/` (disposable agent
> output, `.gitignored`) and `spec/pinned/` (the materialized spec excerpts the
> agent reads). So step 3 below fails with *file not found* until you first
> materialize the pinned spec (see [`spec/README.md`](../../spec/README.md)) and
> then run the agent generation in step 2. "Passes all six cases" describes a
> *generated* tree — reaching green here is the point of doing the generation,
> not a precondition for it.

The loop:

```bash
# 1. Inspect the plan: which nodes/artifacts are missing or stale.
py harness/regen_graph.py --scope usda-single-layer --target python

# 2. Generation is an agent task, not a script. Point your coding agent at
#    REGEN_TASK.md plus the skills, contracts, and goldens it names. It writes
#    the disposable target under generated/python/ (dump_layer.py + node modules).

# 3. Score the generated parser against the real goldens.
py harness/score.py --goldens goldens/integration/usda-single-layer/basic.json \
  --dump-cmd "py generated/python/dump_layer.py"

# 4. Record the manifest and run full deterministic scope validation
#    (score + benchmarks + contract lint + cross-checks).
py harness/regen_graph.py --scope usda-single-layer --target python --record-existing --validate
```

`regen_graph.py` never calls an LLM — it plans, fingerprints, and validates. The
generation step is the agent's; everything that decides *accepted or not* is
deterministic. See `docs/graph-driven-regeneration.md` for the node model and the
`missing` / `unrecorded` / `stale` / `ready` status model.

---

## 5. Now make your own contribution

You've seen the whole loop. A contribution is the same loop, pointed at something
new:

1. **Express the gap as a golden.** Add a case (a USDA feature, an edge case, a
   diagnostic) to a goldens suite. Run `score.py` — watch it fail. The failure is
   your spec for the work.
2. **Encode the answer in a skill and contract.** Update the owning
   `skills/<node>/SKILL.md` and its `contracts/…` so the rule is durable and
   machine-checkable — derived from the USD Core Spec, with a `spec_refs` /
   `Spec Sources` citation.
3. **Regenerate and validate.** Regenerate the affected nodes (the §4 workflow)
   and run `regen_graph.py … --validate` until it's green.

Highest-leverage contributions, and where they land:

- **Skills, contracts, and goldens** → here, in this component. They improve how code
  is generated and define what "correct" means.
- **Spec ambiguity** you hit while writing a golden → file it upstream on the
  [AOUSD `specifications-public`](https://github.com/aousd/specifications-public)
  repo. A fix to the spec benefits every implementation.

This component is published as part of an experiment under Omniverse Labs and
isn't set up to take inbound pull requests; the contributions above land in your
own work — skills you write to generate against the spec, and ambiguity reports
filed upstream.

### Where to go next

- `graph/scopes/usda-single-layer.yaml` — the full scope you just toured.
- `REGEN_TASK.md` — the agent prompt for generating a target.
- `docs/graph-driven-regeneration.md` — node model, dependency vs consumption, status model.
- `docs/first-python-target.md` — the first generated Python target and its benchmark findings.

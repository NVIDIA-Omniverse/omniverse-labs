# nanousd FAQ

Why this experiment exists, what it is and isn't, and how to engage — the
questions that don't change as the code does. If yours isn't here, [open an
issue](https://github.com/NVIDIA-Omniverse/omniverse-labs/issues).

> **Where this sits, and how it stays true.** The [umbrella README](README.md)
> says where to start. [**Where we are**](STATE-OF-THE-FLEET.md) says what works
> today, where it's going, and the gaps between. The
> [skill-graph quickstart](usd-developer-skillgraph/) lets you run the method
> yourself. **This page is for the *why* and the hard questions** — the durable
> stuff that doesn't move: why this exists, what it is and isn't, where we stand
> with OpenUSD and AOUSD, how to engage.

---

## Why this exists

### Why are you doing this?

Because USD's reach is expanding into new kinds of software — robotics,
simulation, embedded and edge runtimes — and that expansion is bigger than any
single implementation or group can carry. The Core Specification is the shared,
durable contract that makes it possible; an implementation is not — it's a set of
choices about footprint, language, and scope. So this is an invitation, grounded
in the standard, to the existing USD community and to newcomers: treat the
*implementation* as something you regenerate against your own constraints, from
the spec, with agents doing the mechanical work — and find out whether doing that
surfaces gaps in the spec that nobody catches while everyone vendors the same
reference code.

Where this is going: the spec stays durable, and **multiple purpose-built
implementations** get generated against it, all passing the same spec-derived
tests. *Specs are durable. Agents are elastic.* The deliverable we actually care
about is the **method** — the skill graph, the contracts, the compliance tests —
not this particular pile of repos. If the method works, the repos are
regenerable. If it doesn't, we'll have learned where it breaks in public.
[Where we are](STATE-OF-THE-FLEET.md) is the current read on how far along that is.

### Is this a product? Is NVIDIA going to support it / will it disappear?

It is **not a product**, not a pre-product, and not a committed roadmap. It's an
experiment published under [`omniverse-labs`](../..) —
samples and ideas, not something with a support contract or a guaranteed
destination. The honest current state of every piece is in
[Where we are](STATE-OF-THE-FLEET.md).

### Why call it "experimental"?

It's literally true: the
methodology is a harness still being invented, and we're learning in public what
generates cleanly, what doesn't, and what "a skill" even is as a durable artifact.
The label marks where we are in the journey.

---

## Where this stands with OpenUSD

### Are you abandoning, forking, or replacing OpenUSD?

No. Your OpenUSD stack keeps working untouched — use what fits, open discussions
around what doesn't. No codebase owns USD; the goal is multiple implementations
passing the same spec-derived tests. If you live comfortably inside OpenUSD
today, none of this asks you to leave.

### What about <my favorite OpenUSD feature> — Hydra, file-format plugins, imaging, X?

The question that sorts every case isn't "is it in nanousd" — it's **is the thing
you need a Core Spec capability, or an OpenUSD implementation detail?**

Start from the scope statement: **nanousd is the data layer** — the scenegraph you
parse, compose, query, and write, and that a runtime, renderer, or any consumer
reads *through*. That's a layer boundary, not a feature count. With that in mind,
your feature lands in one of three places:

- **A different layer of this fleet.** Imaging and rendering aren't absent — they
  live *on top* of the data layer. The Vulkan / Metal / OpenGL backends and
  [`nanousdview`](nanousdview/) are the fleet's imaging layer. So imaging isn't
  "use OpenUSD instead"; it's a separate layer that consumes nanousd, and it's in
  the fleet — it just isn't the data layer's job.
- **A spec capability reached through an OpenUSD-specific mechanism.** Sometimes
  the *mechanism* is an OpenUSD implementation detail but the *capability* it
  produces is squarely in the spec's data model. File-format plugins are the
  clean case: nanousd needn't replicate OpenUSD's plugin system, but "read foreign
  data and surface it as Core Spec-compliant document data" is a legitimate,
  welcome request — served however nanousd chooses. [File it](https://github.com/NVIDIA-Omniverse/omniverse-labs/issues).
- **An OpenUSD implementation detail, not a spec capability.** Hydra, the plugin
  registry and discovery machinery, a particular imaging architecture — these are
  specific choices OpenUSD made, not Core Spec capabilities, so they're genuinely
  not nanousd goals and not feature requests. If you think you need one, the
  useful question is *what spec capability do you actually need underneath it?* —
  which usually lands back in one of the buckets above.

Whether a given spec capability is *built yet* — schemas included — is the moving
part; its current state is in [Where we are](STATE-OF-THE-FLEET.md), not here. Domain schemas
and re-targetings outside the spec live in stacks generated around the core, or
in your own fork.

---

## Using it today

### What can I use today, and what's still rough?

Some of it is solid and some of it is mid-flight, and we don't blur the two.
Rather than make a claim here that goes stale the day after it's written, the
current, honest, per-repo state — what works, what's disclaimed, what's a known
gap — is in [Where we are](STATE-OF-THE-FLEET.md) and each repo's
README. Read those before you build on anything. If something there is wrong or
out of date, that's a bug — [tell us](https://github.com/NVIDIA-Omniverse/omniverse-labs/issues).

### Why use `nanousd` if the skill graph doesn't generate it yet?

Today these are **two tracks that meet at the C ABI**, not one generating the
other: the flagship `nanousd` was built directly
with agents (working the codebase the conventional way), and is further along;
the skill graph regenerates a narrower slice directly from the spec. So:

- If you want a **portable C-ABI USD runtime** to build on, use the flagship for
  what it is today — its value doesn't depend on how it was produced. Its real
  state is in [Where we are](STATE-OF-THE-FLEET.md).
- If you're here for the **experiment** — generating implementations from the
  spec — the skill graph is the part to watch, and the
  [quickstart](usd-developer-skillgraph/) lets you walk it yourself. The honest
  edge of what it generates is documented, not hidden.

Where this is going is the two tracks **converging** — what's built directly
today becomes something the method can regenerate from the spec. Closing that gap
is the work. The check that keeps it honest is mechanical: the graph's generated
backend runs through the same compliance suite as the flagship, cross-checked
against OpenUSD, so disagreements between the implementations surface as CI
signals.

### Is the API stable? Can I pin to it?

No — it's pre-1.0 and experimental. The C API is stable *in shape* (opaque
handles, a narrow function list), but individual functions can change or be
removed before 1.0. If you need reproducibility right now, pin to a specific
commit. Pre-1.0 is the maturity signal, deliberately; 1.0 is gated on full Core
Spec coverage and stable language APIs, not on a date.

### Why should I care about the C ABI if I write Python (or Rust, or Go)?

Because a thin binding over a C ABI keeps your language environment yours. When a
USD binding layer is coupled to a specific Python ABI, upgrading Python means
rebuilding the whole stack against it. A thin binding layer over a stable C ABI
doesn't care what version you're on — you rebuild the binding (seconds), not the
USD stack (hours). The same property holds for any language: the C ABI carries
the contract, so bindings stay small.

---

## The agent-generated part

### Is the agent-generated code any good?

Generated code is only as trustworthy as the tests that pin it, so that's where
the discipline is. Every change runs against a compliance suite, and the same
suite runs against the OpenUSD reference through the same C ABI — so a divergence
between implementations is a CI signal, not a surprise you find in production. If
you find a case where they disagree, that's a spec-ambiguity report and we want
it. And the design decisions — performance, tradeoffs, architecture — are made by
people, not the agent.

### Are agents replacing USD developers?

No. Agents take the mechanical spec-to-implementation work — the part that's
closer to transcription than design. Humans own performance, tradeoffs, and the
hard design calls. The thing agents replace is the *babysitting*, not the
developer. We're also not claiming agents beat humans at this; we're claiming
they can do the rote part well enough that the durable human work becomes the
spec, the contracts, and the tests.

### Is generating a single-layer parser a meaningful result?

Atomic features generate well today; **cohesion across many features is the
frontier**. A single-layer parser working end-to-end through the
same compliance harness as the flagship is a real result, not the finish line.
The named gaps are listed openly in [Where we are](STATE-OF-THE-FLEET.md). Want to see the loop
yourself? The [quickstart](usd-developer-skillgraph/) walks you through the
scoring micro-loop by hand against the contracts — the part you can run on a
fresh checkout in minutes.

---

## Contributing and governance

### Why publish it before it's finished?

Because the point is to do this in the open, where the gaps are visible and the
method can be pulled on by people who aren't us. A polished drop six months from
now would hide exactly the thing the experiment is about: where a
spec-plus-agents methodology holds and where it breaks. The state docs name the
unfinished parts so you're not misled into thinking it's done.

### Can I send a pull request?

Not against these directories. The fleet is published together as one experiment
under Omniverse Labs, and much of the implementation is regenerable output — a
hand patch to generated code gets overwritten on the next regeneration. The
invitation is to the **method**, not to patch the output (see below).

### Then what can I actually contribute?

The invitation is to the **method**, and the highest-leverage contributions are
inputs to the code, not patches to it:

- **Skills, contracts, and goldens** → into
  [`usd-developer-skillgraph`](usd-developer-skillgraph/). These improve how code
  is generated and define what "correct" means. A prose description of an
  uncovered corner case is a real contribution — you don't have to write the
  fixture.
- **Spec-ambiguity reports** → filed upstream on the
  [AOUSD specifications repo](https://github.com/aousd/specifications-public). A
  fix to the spec helps every implementation.
- **Your own implementations and stacks** → generated from the skill graph in
  your repos. The skills you write to get there are what come back here.

Adopting the generated code is welcome too — but the activation is for people who
want to write skills, generate their own implementations, and build new stacks.
The [quickstart](usd-developer-skillgraph/) is the front door to all of this.

---

*Part of [`nanousd-labs`](./README.md) — experimental work under `omniverse-labs`,
not a product. If something here is wrong or out of date,
[file an issue](https://github.com/NVIDIA-Omniverse/omniverse-labs/issues).*

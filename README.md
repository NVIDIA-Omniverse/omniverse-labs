# omniverse-labs

Experimental samples and ideas from the NVIDIA Omniverse team. A Sketchpad built by NVIDIA Omniverse Teams.

**Browse projects and articles:** once GitHub Pages is enabled, the showcase lives at `https://<org>.github.io/omniverse-labs/` (see [docs/README.md](docs/README.md) to publish).

---

## What this is

`omniverse-labs` is where the Omniverse team publishes work that isn't ready to be a shipped library or a supported SDK feature, but is too useful to keep internal. You'll find:

- **Early samples** — working code for workflows we're exploring, posted before we know if they'll ship
- **Integration experiments** — combinations of Omniverse Libraries, Warp, Newton, and external tools that we've tried but haven't productized
- **Reference patterns** — agent-generatable starting points for tasks that come up repeatedly in ISV conversations
- **Proof-of-concept apps** — small, runnable things that demonstrate a new capability or answer a "can it do X?" question

Everything here compiles and runs. Nothing here has a support SLA.

---

## Who this is for

Developers who are:

- Building on Omniverse Libraries (`ov-render`, `ov-physics`, `ov-storage`, etc.) and want to see how real workflows are assembled
- Evaluating whether Omniverse is the right foundation for their use case
- Looking for starting points they can fork and adapt, not polished end-to-end samples
- Willing to tell us what they think — what works, what doesn't, what's missing

If you need production-ready examples with full API documentation, start with the individual library repos. Come here when you want to see how pieces fit together or when you want to poke at something early.

---

## What you'll find

Each sample lives in its own directory and is self-contained. Every directory includes:

- `README.md` — what it does, what it needs, how to run it
- Working code (Python, C, or both)
- An honest note on status: **working** / **partial** / **concept**

Samples are not organized by date. They're organized by what they demonstrate.

---

## How to give feedback

Feedback is the whole point.

**If something works well and you'd like to see it become a first-class feature:** open an issue and say so. Tell us what you built with it.

**If something is broken or misleading:** open an issue. Include the sample name and what you expected vs. what happened.

**If you have a question we haven't answered:** open a discussion. If the same question comes up more than twice, we'll update the sample.

**If you want to propose a new sample topic:** open an issue with the label `idea`. Describe the workflow, what libraries it would use, and who you think would run it. We read these.

We won't commit to shipping any of this. We will commit to reading what you write.

---

## What this is not

- A support channel for production issues — use the individual library repos or NVIDIA Developer Forums for that
- A stable API surface — anything here can change without notice
- A signal that a feature is on the roadmap — presence in omniverse-labs means we're interested, not that we've committed

---

## Running samples

Most samples require at least one Omniverse Library. Check the sample's own README for exact requirements. Common setup:

```bash
# Install the core library the sample uses — for example:
pip install ovrtx
pip install ovphysx

# Then run:
python examples/your-sample/main.py
```

Some samples require a GPU. Some require additional credentials (S3, NGC). The sample README will say.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

Samples in this repo are provided as-is. NVIDIA makes no warranties about fitness for a particular purpose or continued maintenance of any sample.

---

*Built by the NVIDIA Omniverse team. Questions? Open an issue.*

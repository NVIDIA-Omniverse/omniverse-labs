# Codex skills for this experiment

Use the canonical repo-local skill:

- [`nanousd-real-to-sim`](../../../agentic-skills/nanousd-real-to-sim/SKILL.md)

It composes these existing renderer skills:

- `renderer-simulation-loop` for transform and per-step update discipline.
- `verification-led-development` for the smallest meaningful local gates.
- `portable-build-runtime-paths` for checkout-relative renderer discovery.
- `nanousd-renderer-scene-extraction` for USDA and coordinate semantics.

Do not duplicate the skill body in this directory. Keep one source of truth in
`projects/nanousd-labs/agentic-skills`.

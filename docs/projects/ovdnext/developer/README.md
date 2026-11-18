<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# OVD Viewer developer and AI guide

Start with the **[interactive field guide](interactive/index.html)**: learn what
an OVD contains and follow file-recording, ingestion and live-TCP flow diagrams.
Then select recorded rigid-body geometry, inspect articulation limits and drives, and explore
contact pairs, normals, separations and impulses. Scrub 40 real frame samples,
follow object references and inspect the actual PhysX attribute names, native
types and values. [Data notes and recipes](interactive/DATA.md) explain the
capture identities and decoding. All four labs, explanations and Windows/Linux
commands remain on one scrolling page; only the full raw attribute dumps are
collapsed. A concrete body/shape walkthrough reads mass and contact offset. A box/ground
diagram explains contact-generation distance and resting separation. In the native
viewer, **select or focus the matching Pre-sim frame** to display recorded contacts.
The guide also covers TCP recording selection while streaming and side-loading
evidence JSON, with native viewer steps and a measured CPU-control example. Open the guide on the Labs website, or from a local checkout in a browser;
GitHub's file view displays the HTML source. Its inspection examples use local
assets and work offline without installation or a server. The hosted edition
includes the Labs site's corporate consent and analytics scripts; local-file
and localhost previews do not load them. Keep the complete `developer/` directory so the linked references and skills
remain available. The screenshot uses the actual OVD Viewer light-theme UI.

This companion documents **OVD Viewer 0.1 (build v0.2.21)**, application
source `cd83af8e7c6211be71cc2a82541738e0b1f2018f`. It is separate from the
Windows/Linux application ZIPs. Extracting or installing this documentation
does not require modifying those ZIPs or their extracted application files.

| Task | Guide |
| --- | --- |
| Inspect recorded bodies, joints and contacts visually | [Interactive field guide](interactive/index.html) |
| Read mass and contact offset from a body and its shape | [Concrete walkthrough](interactive/index.html#flow) |
| Change live recording scope over TCP | [Recording selection](interactive/index.html#live-selection) |
| Import external findings beside an OVD | [Evidence side loading](interactive/index.html#evidence-import) |
| Use the packaged runtime to inspect OVD without a viewer | [Headless inspection, example and API reference](skills/ovdnext-inspect/references/headless.md) |
| Understand the architecture, build source and run checks | [Source development](skills/ovdnext-develop/references/development.md) |
| Give an AI agent an OVD investigation task | [ovdnext-inspect skill](skills/ovdnext-inspect/SKILL.md) |
| Give an AI agent a source-development task | [ovdnext-develop skill](skills/ovdnext-develop/SKILL.md) |

The supported release scope is the bundled viewer and documented headless
**OVD** functions. PVD3 is untested and excluded. These guides describe the
current interfaces, not a stable SDK or a promise of backward compatibility.
Source builds require dependencies that are not all publicly accessible;
using the released runtime does not require those source checkouts.

## Start with your package

Verify the downloaded application ZIP against its supplied checksum before
extracting it. The v0.2.21 identities are:

| Archive | SHA-256 |
| --- | --- |
| `OvdNext-windows-x86_64-0.2.21.zip` | `b12a623f2a3b974f1d5d4e9a7607311439957eac12316917cef164b80bb0da84` |
| `OvdNext-linux-x86_64-0.2.21.zip` | `e23e30dd5e9eb4a88fa7d418285290bdb0cc37792c054b321648fa0c481a0eac` |

Keep `python/`, `app/` and `lib/` together. Use the bundled interpreter for
the headless example. A system Python, `pip install ovdnext`, or an unrelated
`ovd-inspect` executable is not a substitute for this runtime. Keep your
scripts, captures, outputs and temporary work outside the package.

## Skills

For recording and producer-side capture control, agents should start with the
existing [PhysX SDK OVD skills](https://github.com/NVIDIA-Omniverse/PhysX/blob/7d6f6ac8b83e1610bdf715f1f2f1d6b718aee586/physx/SKILLS.md):

- [ovd-record](https://github.com/NVIDIA-Omniverse/PhysX/blob/7d6f6ac8b83e1610bdf715f1f2f1d6b718aee586/physx/skills/ovd-record/SKILL.md): finite files, late sampling,
  object/data selection and TCP discovery/control APIs.
- [ovd-inspect](https://github.com/NVIDIA-Omniverse/PhysX/blob/7d6f6ac8b83e1610bdf715f1f2f1d6b718aee586/physx/skills/ovd-inspect/SKILL.md): discover saved SDK captures and
  inspect supported typed pose/velocity values with provenance.
- [ovd-compare](https://github.com/NVIDIA-Omniverse/PhysX/blob/7d6f6ac8b83e1610bdf715f1f2f1d6b718aee586/physx/skills/ovd-compare/SKILL.md): align two captures and explain
  measured differences; the executable is `ovd-inspect compare`.

Use the copy in the matching SDK checkout or installation (`share/physx/SKILLS.md`).
The links point to a verified public PhysX SDK revision. Use the skills
from the SDK version installed with your producer; they do not change the
v0.2.21 viewer's pinned build inputs. Those tools are
separate from OVD Viewer: its packaged-runtime companion below supplies scalar,
reference and bounded-property queries such as the executed mass/contact-offset
example. Do not invent SDK scalar or selection CLI flags. The SDK skill's
annotation-export caveat is separate from this viewer's version-matched Evidence
import contract; use the documented formats and verification requirements.

Every field-guide chapter has an inspection case and data: recorded geometry,
attribute queries, a retained TCP session, CPU evidence points, availability
comparisons or packaged formatter outputs. TCP and formatter controls replay
recorded results; they do not execute a live producer in the browser.

Both skills use ordinary `SKILL.md` files with YAML frontmatter and local
Markdown references. No MCP server, service, credentials, executable installer
or additional Python package is needed to read them.

- **ovdnext-inspect:** inspect saved OVD captures, use the packaged native
  binding and report bounded, reproducible findings.
- **ovdnext-develop:** navigate the source, identify build prerequisites,
  select relevant verification and preserve the Labs import boundary.

For an agent supporting this skill format, copy the **entire individual
skill directory**, including `references/`, into its configured skill
directory. Keep the two skills as siblings. For a project using
`.agents/skills/`, the result is `.agents/skills/ovdnext-inspect/SKILL.md`
and `.agents/skills/ovdnext-develop/SKILL.md`. Do this in your working project,
not inside the extracted runtime or the pinned Labs source snapshot.
Otherwise, point the agent directly at the relevant `SKILL.md` and its
reference. Installing the skill does not authorize network capture, data
uploads, application modifications or release publication.

Example requests:

> Use ovdnext-inspect with this v0.2.21 package and saved OVD. List scenes
> and frames, then inspect the object I select. Report the capture hash,
> scene, frame, property metadata and any incomplete reads.

> Use ovdnext-develop to locate the property-inspection path and identify
> the prerequisites and focused checks for a proposed change.

The physics labs use `multi_geom.ovd`, `articulation.ovd` and a recorded contact
fixture, extracted with the exact Windows v0.2.21 runtime. The optional capture
fundamentals retain the small `test_scene.ovd` exercise. Provenance is included
in each lab. They illustrate recorded state; they do not simulate physics or
parse user captures in the browser. Live recording and application changes
are separate.

The teaching structure is inspired by the concept-first OvPhysX guides and
Stefano Cristiano's OvSpatial documentation; the page, diagrams and controls
here are original. The tested Markdown reference remains the detailed API
and agent source of truth.

The documentation is CC-BY-4.0; the companion's JavaScript/CSS carry
Apache-2.0 headers. Application and dependency licensing is
unchanged; retain the application's notices and licensing/source companion.

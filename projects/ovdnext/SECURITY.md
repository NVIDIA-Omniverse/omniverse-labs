# Security Policy: OvdNext

## Reporting a Vulnerability

If you discover a potential security vulnerability in OvdNext, please **do not
open a public issue, merge request, or discussion**.

- **Web (preferred):** [NVIDIA Vulnerability Disclosure Program](https://www.nvidia.com/en-us/security/)
- **Email:** [psirt@nvidia.com](mailto:psirt@nvidia.com). For sensitive reports,
  use the [NVIDIA public PGP key](https://www.nvidia.com/en-us/security/pgp-key).
- **GitHub/GitLab:** Use this repository's **Security** tab and private vulnerability
  reporting flow when available.

Include the following information:

- OvdNext version, branch, or commit SHA
- Affected component and vulnerability type
- Reproduction steps and required input or environment
- Proof-of-concept data or code, if available
- Expected impact, including whether the issue affects confidentiality,
  integrity, or availability

NVIDIA PSIRT will acknowledge the report, validate the issue and severity,
coordinate a fix, and publish a security bulletin when appropriate.

## Security Architecture & Context

OvdNext is an OvUI/ModernGL desktop application with a C++ inspection runtime
for receiving, recording, opening and visualizing PhysX Visual Debugger (PVD3)
and OmniPVD (OVD) diagnostic data. The Python interface and headless inspection
entry points use native object models and parsers in `src/PVDCore` through
`src/PVDSession`. Legacy `.pvd` archives use the bounded Python stdlib ZIP
extractor in `python/ovd_ui/ingestion.py`; `.pxd2` captures use the PVD event
pipeline with internal `.pxd3` backing files. OVD files and live streams are
applied to an in-memory DOM. No Qt, Qwt or bundled minizip implementation is used.

This software operates at the **Application** level. Its primary security
responsibility is to consume diagnostic telemetry without allowing malformed
files or streams to corrupt memory, escape intended filesystem locations, or
exhaust host resources.

OvdNext is intended for external distribution as a debugging tool for local,
remote-development, and data-center environments. Captures may contain
proprietary simulation geometry, object state, messages, filenames, and source
locations.

### Interfaces and Trust Boundaries

- `OvdLiveSource` accepts live OVD TCP streams, normally on port 5425, through
  the cancellable native Winsock/POSIX TCP helper and PVDRuntime parser. Commands are
  applied to `OmniPvdDOMState`; consumed bytes can be saved as `.ovd` recordings.
- Capable producers accept bounded capture-selection requests from the viewer.
  These select recorded telemetry, not arbitrary executable or administrative
  commands. Legacy PVD3 parsing remains a file-input path, not a live listener.
- The viewer opens `.pvd`, `.pxd2`, and `.ovd` files selected by the user.
  These inputs are parser inputs and must not be assumed well formed; `.pxd3`
  is an internal backing format rather than a normal UI input.
- `.pvd` files are ZIP archives whose entries are validated and extracted into
  an owned temporary directory before the inner capture is opened.
- OvdNext does not require an application account or producer identity. Live
  streams intentionally support anonymous producers on localhost, remote
  development machines, and data-center networks, and listeners intentionally
  remain reachable on all configured interfaces.
- The live protocol intentionally provides no authentication, authorization,
  encryption, or peer identity validation. Network reachability, segmentation,
  source restrictions, and any transport protection are deployment and firewall
  responsibilities.
- The protocol is observational telemetry and exposes no intentional command-
  execution or administrative control plane. This does not make parser input
  intrinsically safe: malformed data can still trigger crashes, resource
  exhaustion, or memory-safety defects in the native viewer.

### Threat Model

1. **Native parser memory corruption:** Crafted PVD event groups, property
   arrays, OVD commands, or short property samples can reach native allocation,
   copy, and typed-read paths in `PvdCommStreamParsing`, `PvdObjectModel`,
   `PvdDomParser`, and the property viewer. Integer overflow, truncation, or
   missing bounds checks can cause out-of-bounds reads or writes.
2. **Archive extraction outside the application directory:** A crafted `.pvd`
   ZIP entry can contain absolute or parent-directory components. Extraction in
   `python/ovd_ui/ingestion.py` must ensure every output path remains beneath
   the owned extraction directory and respect size/free-space limits.
3. **Memory and CPU exhaustion:** A producer can send very large event groups or
   an indefinitely growing sequence of objects, classes, attributes, and frames.
   Queueing, DOM growth, indexing, and rendering can consume excessive memory or
   stall the UI if resource limits and failure handling are insufficient.
4. **Local temporary-file manipulation:** Predictable names, non-atomic creation,
   unsafe path copies, or stale files in the application-data directory can let
   another local process redirect or interfere with backing-file writes.
5. **Shutdown and concurrency faults:** Live ingestion runs on a worker while the
   UI reads the shared DOM and shutdown closes the socket to unblock reads.
   Incorrect synchronization or violations of the stream close/read contract can
   cause races, hangs, or use-after-free behavior.
6. **Build and dependency integrity:** OvdNext incorporates CPython, OvUI,
   ModernGL, Pillow, pybind11, PhysX/PxShared and OmniPVD runtime/DOM code.
   Compromised or stale build inputs can affect the shipped parser and viewer.

### Accepted Risks

Previously triaged threats retained by team decision:

- **Plaintext live telemetry:** A party with access to the network path may
  observe simulation data or alter bytes before they reach the viewer. —
  *Accepted (2026-07-20): The debugging protocol intentionally provides no
  transport confidentiality or integrity layer; deployments that require those
  controls provide them through network segmentation, VPNs, or equivalent
  infrastructure.*
- **Anonymous live telemetry:** Any producer that can reach a configured live
  listener can submit diagnostic input without an application identity. —
  *Accepted (2026-07-20): Remote development and data-center streaming is a
  supported product use case; producer identity is intentionally outside the
  observational telemetry protocol, and reachability and source restrictions
  are deployment/firewall responsibilities. This does not waive parser
  robustness requirements.*

### Critical Security Assumptions

- OvdNext runs as a non-privileged desktop user and relies on the host operating
  system for process isolation, filesystem permissions, and memory protection.
- Operators deliberately choose which networks can reach the live listener and
  accept that OvdNext itself does not authenticate producers or encrypt traffic.
- Diagnostic streams and captures may contain proprietary simulation state,
  object names, messages, filenames, and source locations; operators protect
  captures and network paths according to the sensitivity of that data.
- Live OVD clips stay in process memory until removed or the viewer closes.
  Explicit saves and optional auto-save create persistent recordings; users
  control their destinations and retention.
- Anonymous streaming is not a reason to trust stream contents. All file and
  network inputs remain untrusted parser input and require defensive validation.
- OvdNext does not intentionally execute commands received from a producer, but
  native-parser defects can still affect the viewer process and host resources.
- Packaged binaries and their Python/OvUI/OmniPVD dependencies are obtained from
  trusted build and distribution channels and kept at supported revisions.
- Temporary and extracted files must stay inside owned working directories and
  must not follow attacker-controlled paths outside them.

## Intentional Protocol Behavior and Accepted Deployment Risk

The following behavior is intentional and should not, by itself, be treated as a
product defect:

- listening for anonymous live OVD producers on all configured interfaces;
- accepting streams from localhost, remote development systems, or data centers;
- operating the diagnostic protocol without an application login, producer
  authentication, or transport encryption.

These choices make the debugging stream easy to connect across supported
development environments. Network exposure and transport protection are owned by
the deployment. Reports that demonstrate a parser, memory-safety, resource-
exhaustion, filesystem, or boundary violation remain in scope regardless of the
producer's lack of identity.

## Security Update Guidance

Use the latest maintained OvdNext build and its matching OmniPVD runtime/DOM
dependencies. When sharing a capture for investigation, use an approved secure
transfer channel if it may contain confidential telemetry. Security fixes should
include a regression test where practical and should be validated against both
file and live-stream entry paths when they share parser code.

---
name: ovrtx-multi-session-deployment
description: Plan and validate multiple independent Blender and OVRTX jobs on available GPU hosts or local workers. Use when renders, viewport captures, or simulations must run concurrently with isolated inputs, outputs, and runtime ownership.
license: "Apache-2.0"
metadata:
  author: "Max Bickley"
  version: "0.1"
  team: "omniverse"
  domain: "physical-ai"
  tags:
    - blender
    - omniverse
    - ovrtx
---
# OVRTX multi-session deployment

Use this optional skill only when concurrency is required. Follow the installed
deployment tool's and add-on's documented setup, scheduling, and cleanup interfaces.

## When to Use

Use when renders, viewport captures, or simulations must run concurrently with isolated inputs, outputs, and runtime ownership.

## Matrix and isolation

Define one row per job: unique name, input `.blend`/fixture hash, source/add-on versions, GPU/host, capture type, camera/frame range, samples/resolution, timeout, and output prefix. Give each row one Blender/runtime owner and isolated config/cache/output directories. Never collapse rows onto one session when independent state matters.

Preflight available host capacity, authenticated scheduler access, Blender/add-on installation, runtime capability, and an explicit cleanup operation. Package immutable inputs once when the scheduler supports shared storage; keep Blender library paths separate from runtime library paths.
Run each job with least-privilege user-scoped access and isolated writable
config, cache, temporary, and output directories. Do not place credentials,
tokens, or restricted URLs in job manifests or command arguments; use the
deployment system's supported secret mechanism.

## Instructions

Run one smallest fixed-image smoke per row before UI or long animation. Require structured `pass`, exit 0, nonblank image, dimensions, samples, logs, and checksums. Capture UI only after the image gate. Collect manifests, logs, native outputs, first/middle/last frames, and per-row status (`pass`, `partial`, `failed`, `blocked-capacity`, or `blocked-runtime`).

Verify artifacts before stopping/deleting sessions. End with every temporary worker terminated or explicitly handed off; report cleanup status and never claim a shared runtime was isolated without evidence.
Use relative or sanitized paths in shareable manifests, and process logs,
screenshots, and reports through `blender-sanitized-support-bundle` before
sharing.

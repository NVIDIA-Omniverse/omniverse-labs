# AGENTS.md — OneDrive Storage Service (ovstorage/onedrive-example)

Onboarding guide for coding agents starting work in this project. Read this
first, then load the skills and README sections referenced below before acting.

## What this project is

A **stateless storage service** that connects **OneDrive for Business** to
applications using the NVIDIA Omniverse Storage API. It accepts a per-user
Bearer token from the Kit Client Library, resolves the user's OneDrive drive,
and forwards operations to the **Microsoft Graph API** over both **gRPC and
REST**. This is **sample reference code** — Docker/Helm packaging are starting
points, not production drop-ins.

Full detail lives in [`README.md`](README.md). Skim it once before making
changes; it is the source of truth for behavior, config, and limitations.

## Load the skills first (do this before deployment/auth/integration work)

Skills for this project live under [`skills/`](skills/). Each is a self-contained
folder with a `SKILL.md` entry point and supporting `references/` files. **Treat
the relevant `SKILL.md` as required reading — load it into context before you
plan or answer**, rather than reasoning from memory.

| Skill | Load its `SKILL.md` when the task involves… |
|---|---|
| [`skills/onedrive-deployment/SKILL.md`](skills/onedrive-deployment/SKILL.md) | Deploying, configuring, or integrating the service: Entra ID / Azure AD setup (tenant ID, client ID, client secret), Microsoft Graph delegated permissions, choosing **Bearer pass-through vs. On-Behalf-Of (OBO)** auth, Kit App Streaming wiring (discovery endpoints, redirect URIs, ingress auth extension), end-to-end validation, or troubleshooting auth/deploy failures. It also explains the Graph API implementation and the OBO token lifecycle. |

If a request matches any of those triggers, open the `SKILL.md` and follow its
workflow and `references/` links instead of improvising. Check `skills/` for
newly added skills before assuming this is the only one.

## The one decision that shapes everything: auth model

Almost every config choice follows from which auth model is deployed:

- **Bearer pass-through** (default) — token arrives already Graph-scoped; the
  service forwards it as-is. No client secret. Good for local/dev.
- **On-Behalf-Of (OBO)** — token arrives app-scoped (validated by an ingress
  auth extension against your app's audience); the service exchanges it for a
  Graph-scoped token and caches it. Requires `AZURE_CLIENT_SECRET`. This is the
  common **Kit App Streaming** topology.

OBO activates only when `USE_OBO_FLOW=true` **and** `AZURE_CLIENT_SECRET` is set
— setting the flag without the secret is a hard startup error. See the
deployment skill for the full decision rule.

## Repository layout

| Path | Description |
|---|---|
| `omni_onedrive_service/` | Service source (OneDrive backend, gRPC/REST layers, CLI) |
| `omni_onedrive_service/src/omni_onedrive_storage_service/` | The Python package — `onedrive/`, `grpc_service/`, `rest_service/`, `backends/` |
| `proto/` | gRPC Protocol Buffer definitions (**reference** — generated stubs live under `src/nvidia/`) |
| `openapi/` | REST OpenAPI specifications (reference) |
| `helm/` | Sample Helm chart (Deployment, Service, optional Contour HTTPProxy ingress) |
| `docker-compose.yml` | Docker Compose service definition |
| `skills/` | Project skills — see the table above |
| `generate_protos.sh` | Regenerate Python protobuf stubs from `.proto` files |

## Quick start (local, Bearer pass-through)

```bash
cd omni_onedrive_service && uv sync           # pip install -e . also works
cp ../.env.example .env                        # set AZURE_TENANT_ID + OIDC_CLIENT_ID
uv run omni-onedrive-storage-service onedrive  # REST :8011  gRPC :50051
```

Smoke test — `/api/v1/auth-config` is the **only** unauthenticated endpoint:

```bash
curl http://localhost:8011/api/v1/auth-config   # expect HTTP 200
curl -i http://localhost:8011/api/v1/services    # expect HTTP 401 without a token — this is normal
```

A `401` from any path other than `/api/v1/auth-config` is expected, not a
failure — it is the signal that triggers Kit's interactive login.

## Conventions & gotchas for agents

- **No thread-local/global state.** The Bearer token and resolved drive ID flow
  through `contextvars` (`onedrive/request_context.py`). Preserve this pattern.
- **`proto/` and generated stubs are reference material.** Don't hand-edit the
  `*_pb2.py` stubs under `src/nvidia/`; run `./generate_protos.sh` if the API
  surface changes.
- **Known no-ops / limits** (see README "Known Limitations"): custom metadata is
  a no-op (OneDrive has none), only OneDrive **for Business** is supported (not
  personal accounts / SharePoint libraries), single-tenant only.
- **Sample code, not a product.** Adapt Dockerfile/Helm to your own registry,
  ingress, and TLS; don't assume the defaults are deployment-ready.

## Pointers

- Behavior, config reference, Azure AD setup → [`README.md`](README.md)
- Deployment / auth / App Streaming / troubleshooting → [`skills/onedrive-deployment/SKILL.md`](skills/onedrive-deployment/SKILL.md)
- Helm values reference → [`helm/README.md`](helm/README.md)

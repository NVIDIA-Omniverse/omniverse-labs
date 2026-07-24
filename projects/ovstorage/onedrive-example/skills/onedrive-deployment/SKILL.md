---
name: onedrive-deployment
description: >-
  Guide for deploying, configuring, and integrating the OneDrive Storage Service
  for NVIDIA Omniverse with Kit App Streaming. Use when configuring Entra ID /
  Azure AD authentication (tenant ID, client ID, client secret, Graph API
  endpoints), setting up Microsoft Graph delegated permissions, choosing between
  Bearer pass-through and On-Behalf-Of (OBO) token flows, wiring the service into
  Kit App Streaming (discovery endpoints, redirect URIs, ingress auth
  extension), validating end-to-end connectivity, or troubleshooting deployment
  failures (invalid/expired credentials, missing Graph permissions, token
  acquisition/exchange failures, tenant ID / redirect URI mismatches). Also
  explains the underlying Graph API implementation and the OBO token lifecycle
  (acquisition, caching, expiry, renewal) and why each auth component is required.
---

# OneDrive Storage Service — Deployment & Kit App Streaming Integration

This skill guides you through configuring, deploying, and integrating the
**OneDrive Storage Service for NVIDIA Omniverse**. The service is a stateless
translator: it accepts a per-user Bearer token, resolves the user's OneDrive for
Business drive, and forwards operations to the **Microsoft Graph API** over both
gRPC and REST.

Everything in this skill is grounded in the service source under
`omni_onedrive_service/src/omni_onedrive_storage_service/` and the packaging in
`helm/`, `docker-compose.yml`, and `.env.example`.

## The one decision that shapes everything: which auth model?

The service supports **two authentication models**, and almost every other
configuration choice follows from which one you deploy. Both are first-class.

| Model | What the service does | Client secret? | When to use |
|---|---|---|---|
| **Bearer pass-through** (default) | Forwards the user's token **as-is** to Graph. Kit owns login *and* token refresh. | No | The token reaching the service is already Graph-scoped (audience = `https://graph.microsoft.com`). Simplest path; good for local/dev and deployments without a token-validating proxy. |
| **On-Behalf-Of (OBO)** | Exchanges the incoming **app-scoped** user token for a **Graph-scoped** token via Azure AD, then caches it. | **Yes** (`AZURE_CLIENT_SECRET`) | The token reaching the service is scoped to *your app* — typically because an **ingress auth extension** validated it against your app's audience. This is the common **Kit App Streaming** topology. |

**Decision rule:** If requests pass through an ingress/proxy that validates
tokens against *your application's* audience (the `httpProxy.authExtension` path
in the Helm chart), you **must** use OBO — a Graph call with an app-audience
token is rejected. If tokens arrive already Graph-scoped, use pass-through. See
[references/architecture.md](references/architecture.md#choosing-a-model) for the
full reasoning.

OBO activates when `USE_OBO_FLOW=true` **and** `AZURE_CLIENT_SECRET` is set
(`onedrive/__init__.py:84-94`). Setting `USE_OBO_FLOW=true` without a secret is a
hard startup error.

## Deployment workflow

Follow these in order. Each links to a detailed reference.

1. **Prerequisites & Entra ID app registration** — tenant ID, client ID,
   redirect URIs, and the public-vs-confidential client choice.
   → [references/prerequisites-and-app-registration.md](references/prerequisites-and-app-registration.md)

2. **Configuration: env vars, secrets, credentials** — the full variable/flag
   reference for both auth models, secret handling, and TLS.
   → [references/configuration.md](references/configuration.md)

3. **Microsoft Graph permissions** — the delegated permissions the service
   actually uses, the `.default` scope, and admin consent.
   → [references/permissions.md](references/permissions.md)

4. **Kit App Streaming & Kit Desktop integration** — discovery endpoints,
   `SERVICE_PUBLIC_URL`, CORS, redirect-URI matching, the ingress auth extension,
   end-to-end connectivity validation, and a Desktop-vs-App-Streaming topology
   comparison with Desktop-specific setup/validation.
   → [references/app-streaming-integration.md](references/app-streaming-integration.md)

5. **Validation & troubleshooting** — post-deploy verification plus a
   symptom → cause → fix matrix mapped to the actual error responses the code
   returns.
   → [references/validation-and-troubleshooting.md](references/validation-and-troubleshooting.md)

## Understanding the system (the "why")

- **Architecture & Graph API implementation** — request pipeline, drive
  resolution, file operations, retry/rate-limit behavior, and *why each auth
  component is required*.
  → [references/architecture.md](references/architecture.md)

- **OBO token lifecycle** — acquisition, SHA-256-keyed caching, the expiry
  safety margin, and renewal. (Section within the architecture reference.)
  → [references/architecture.md](references/architecture.md#obo-token-lifecycle)

- **FAQ, capabilities, limitations & external docs** — what the service can and
  cannot do, plus links to Microsoft Graph, OneDrive, and NVIDIA App Streaming
  documentation.
  → [references/faq-and-links.md](references/faq-and-links.md)

## Quick reference: minimum to run

**Bearer pass-through (local/dev):**

```bash
cp .env.example .env   # set AZURE_TENANT_ID and OIDC_CLIENT_ID
cd omni_onedrive_service && uv sync
uv run omni-onedrive-storage-service onedrive
# REST http://localhost:8011  |  gRPC localhost:50051
```

**OBO (App Streaming / ingress-fronted):** additionally set
`USE_OBO_FLOW=true` and `AZURE_CLIENT_SECRET=<secret>`. See
[references/configuration.md](references/configuration.md).

**Smoke test:**

```bash
# /api/v1/auth-config is the ONLY unauthenticated path — expect 200
curl http://localhost:8011/api/v1/auth-config

# Everything else, including /api/v1/services, requires a Bearer token.
# Without one it returns 401 — expected, not a failure.
curl -i http://localhost:8011/api/v1/services
```

A `200` from `/api/v1/auth-config` with a valid `openid_configuration` URL means
the service is up and advertising the right tenant. **Only** `/api/v1/auth-config`
(and its `/api/v1alpha` alias) is served without a token; every other path —
**including `/api/v1/services`** — returns `401` without a Bearer token. That 401
is the signal that triggers Kit's interactive login, not a deployment failure
(`onedrive/middleware.py:38, 64-88`).

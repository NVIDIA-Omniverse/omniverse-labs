# OneDrive Storage Service for NVIDIA Omniverse

A storage service that connects **OneDrive for Business** to applications using the NVIDIA Omniverse Storage API. It accepts per-user Bearer tokens from the Kit Client Library, forwards them to the Microsoft Graph API, and exposes files via both gRPC and REST interfaces.

This repository is **sample reference code**: it contains the Python service, API contracts, and example Docker/Helm packaging so you can build, run, and extend your own OneDrive-backed storage integration. The provided Dockerfile and Helm chart are starting points — adapt them to your own registry, ingress, and deployment strategy.

> **Deploying or integrating with Kit App Streaming?** A step-by-step guide covering Entra ID setup, the Bearer pass-through vs. On-Behalf-Of (OBO) auth models, Graph permissions, App Streaming integration, and troubleshooting lives in the [`onedrive-deployment` skill](skills/onedrive-deployment/SKILL.md). Skills for this repository are located under [`skills/`](skills/).

## How It Works

```
Kit (local or streaming)       OneDrive Storage Service          Microsoft Graph
       │                              │                               │
       │  1. GET /api/v1/auth-config  │                               │
       │ ◀────────────────────────────│                               │
       │  (returns OIDC config)       │                               │
       │                              │                               │
       │  2. User logs in via browser │                               │
       │     (Entra AD OIDC flow)     │                               │
       │                              │                               │
       │  3. Bearer token on requests │                               │
       │ ────────────────────────────▶│  4. Forwards token            │
       │                              │ ────────────────────────────▶ │
       │                              │  (GET /me/drive, files, etc.) │
       │  5. Storage API response     │                               │
       │ ◀────────────────────────────│ ◀──────────────────────────── │
```

The Kit Client Library handles OAuth/OIDC login (opening a browser for the user). This service is a stateless translator between the Omniverse Storage API and Microsoft Graph.

### Request Pipeline (internal)

1. **Middleware** (`middleware.py`) extracts Bearer token from the request header
2. **Drive resolution** calls Graph `/me/drive` to get the user's drive ID (cached by token hash in `request_context.py`)
3. Both values are stored in `contextvars` — no thread-local or global state
4. **OneDrive provider** (`onedrive_provider.py`) translates Storage API calls to Graph API calls via `graph_client.py`
5. **Path cache** (`path_mapper.py`) avoids redundant Graph lookups for recently accessed paths

## Features

- Full gRPC and REST API (v1alpha and v1beta)
- Per-user OneDrive access via Bearer token pass-through
- File operations: read, write, stat, enumerate, delete, copy, move
- Folder operations: list, create, delete
- Version history via Graph API versions endpoint
- Redirect-based downloads using pre-authenticated Graph API URLs
- Large file uploads via chunked upload sessions
- Discovery endpoints for Kit and Streaming Portal integration

## Supported Storage

| Storage Type | Supported |
|---|---|
| OneDrive for Business (per-user) | Yes |
| OneDrive Personal (consumer) | No |
| SharePoint shared libraries | No (requires different auth model) |

## Prerequisites

- Python 3.10+
- [UV](https://docs.astral.sh/uv/) package manager (`pip` also works — see fallback below)
- An **Azure AD (Entra ID) app registration** in your tenant — see [Azure AD Setup](#azure-ad-app-setup)

## Quick Start

### 1. Install

```bash
cd omni_onedrive_service
uv sync
```

<details>
<summary>Using pip instead of UV</summary>

```bash
cd omni_onedrive_service
pip install -e .
```

</details>

### 2. Configure

Create a `.env` file with your values (see `.env.example` at repo root for a fully commented template):

```bash
cp ../.env.example .env
# Edit .env — you need AZURE_TENANT_ID and OIDC_CLIENT_ID at minimum
```

| Variable | Required | Description |
|---|---|---|
| `AZURE_TENANT_ID` | Yes | Azure AD directory (tenant) ID |
| `OIDC_CLIENT_ID` | Yes | Public application (client) ID for Kit/Portal OIDC login |
| `OIDC_SCOPES` | No | OAuth scope string (default: `openid {client_id}/.default offline_access`). The `{client_id}` placeholder is replaced with `OIDC_CLIENT_ID` at startup. Requires delegated permissions (e.g. `Files.ReadWrite`, `User.Read`) on the app registration. |
| `ONEDRIVE_BASE_URI` | No | Base URI for resource addresses (default: `onedrive://me`) |
| `GRPC_SERVER_PORT` | No | gRPC server port (default: `50051`) |
| `HTTP_SERVER_PORT` | No | REST server port (default: `8011`) |

### 3. Start the service

```bash
uv run omni-onedrive-storage-service onedrive
```

The service starts on:
- **REST**: `http://localhost:8011`
- **gRPC**: `localhost:50051`
- **Discovery**: `http://localhost:8011/api/v1/services` and `http://localhost:8011/api/v1/auth-config`

### 4. Verify

```bash
# /api/v1/auth-config is the ONLY unauthenticated endpoint — expect HTTP 200
curl http://localhost:8011/api/v1/auth-config

# /api/v1/services requires a Bearer token. Without one it returns HTTP 401
# (expected — this is not a deployment failure).
curl -i http://localhost:8011/api/v1/services

# Query the service catalog with a token:
curl http://localhost:8011/api/v1/services -H "Authorization: Bearer $TOKEN"
```

### 5. Connect from Kit

In Kit 111.0.0+, add a storage endpoint pointing to:
```
http://localhost:8011
```

Kit's Client Library will fetch `/api/v1/auth-config`, open a browser for Entra AD login, and use the resulting token for all subsequent file operations.

## Run with Docker

### Using docker-compose (recommended)

```bash
# Make sure .env is configured (see step 2 above)
docker compose up --build
```

### Manual docker build and run

```bash
# Build (run from repo root)
docker build -f omni_onedrive_service/Dockerfile -t onedrive-storage-service .

# Run (pass env vars from your .env file)
docker run --env-file .env -p 8011:8011 -p 50051:50051 onedrive-storage-service
```

The container exposes:
- **REST**: port 8011
- **gRPC**: port 50051

## Deploy to Kubernetes (Helm)

A sample Helm chart is provided under `helm/`. It packages the Deployment, Service, and an optional Contour `HTTPProxy` for ingress. See [`helm/README.md`](helm/README.md) for the full values reference and examples.

```bash
helm install onedrive-storage ./helm \
  --set image.tag=<version> \
  --set env.AZURE_TENANT_ID="your-tenant-id" \
  --set env.OIDC_CLIENT_ID="your-client-id"
```

Treat the chart as a starting point and adapt it to your own registry, ingress, and TLS setup.

### Portal / HTTPS Deployment

When deploying behind a reverse proxy or for the Streaming Portal:

- Terminate TLS at the proxy; the service itself listens on plain HTTP/gRPC.
- Set `ONEDRIVE_BASE_URI` if the public-facing URL differs from localhost defaults.
- Ensure the **Redirect URI** in your Azure AD app registration matches the public URL that users' browsers will be redirected to after login.

## Azure AD App Setup

This service requires a **public client** app registration in your Entra ID tenant (no client secret). Kit and the Streaming Portal use this registration to perform the OIDC browser login flow.

1. Go to [Azure Portal](https://portal.azure.com) > **Microsoft Entra ID** > **App registrations**
2. Click **New registration**, give it a name
3. Set **Supported account types** to your organization (single tenant)
4. Add **Redirect URIs** appropriate for your Kit/Portal deployment
5. Note the **Application (client) ID** and **Directory (tenant) ID**
6. Go to **API permissions** > **Add a permission** > **Microsoft Graph** > **Delegated permissions**:
   - Add `Files.ReadWrite` (user's own OneDrive files)
   - Add `User.Read` (needed for `/me/drive` resolution)
7. Click **Grant admin consent** for the permissions above
8. No client secret is needed — Kit performs the browser-based OIDC flow

## Resource Addressing

Files are addressed using URIs with the `onedrive://me` scheme:

```
onedrive://me/path/to/file.usd          # latest version
onedrive://me/path/to/file.usd;2        # specific version (by index)
```

The `me` authority mirrors Microsoft Graph's `/me/` convention — it resolves to the authenticated user's OneDrive at runtime.

Resource identities (returned by write/stat operations) are opaque, immutable identifiers using the `onedrive-id://` scheme. Clients should not parse them.

## Service Modes

The service starts both gRPC and REST servers by default. To run only one protocol:

```bash
uv run omni-onedrive-storage-service --no-grpc onedrive   # REST only
uv run omni-onedrive-storage-service --no-rest onedrive   # gRPC only
```

## Configuration Reference

| Flag | Environment Variable | Default | Description |
|---|---|---|---|
| `--tenant-id` | `AZURE_TENANT_ID` | (required) | Azure AD tenant ID |
| `--oidc-client-id` | `OIDC_CLIENT_ID` | (required) | Public OIDC client ID for Kit/Portal auth |
| `--oidc-scopes` | `OIDC_SCOPES` | `openid {client_id}/.default offline_access` | OAuth scope string (`{client_id}` replaced with `--oidc-client-id` at startup) |
| `--base-uri` | `ONEDRIVE_BASE_URI` | `onedrive://me` | Base URI for resource addresses |
| `--cache-ttl` | | `300` | Path and drive cache TTL in seconds |
| `--grpc-port` | `GRPC_SERVER_PORT` | `50051` | gRPC server port |
| `--http-port` | `HTTP_SERVER_PORT` | `8011` | REST server port |

## Known Limitations

- **No custom metadata**: OneDrive does not support arbitrary user-defined metadata on files. Metadata get/set/delete operations are no-ops.
- **Version size accuracy**: The Graph API may report `size: 0` for some historical versions.
- **Concurrent writes**: Rapid parallel writes to the same file may trigger ETag conflicts. The service retries with exponential backoff (up to 5 attempts).
- **Consumer OneDrive**: Only OneDrive for Business is supported. Personal Microsoft accounts are not supported.
- **Single tenant**: The service is configured for one Entra AD tenant. Multi-tenant support would require changing the OpenID configuration URL to use `/common` or `/organizations`.

## Skills

Repository skills live under [`skills/`](skills/) at the repo root. Each skill is
a self-contained folder with a `SKILL.md` entry point and supporting reference
files.

| Skill | Description |
|---|---|
| [`skills/onedrive-deployment/`](skills/onedrive-deployment/SKILL.md) | Deploying, configuring, and integrating the service with Kit App Streaming — Entra ID setup, Bearer pass-through vs. OBO auth, Graph permissions, connectivity validation, and troubleshooting |

## Repository Layout

| Path | Description |
|---|---|
| `omni_onedrive_service/` | Service source code (OneDrive backend, gRPC/REST layers, CLI) |
| `proto/` | gRPC Protocol Buffer definitions (reference) |
| `openapi/` | REST OpenAPI specifications (reference) |
| `helm/` | Sample Helm chart (Deployment, Service, optional HTTPProxy ingress) |
| `docker-compose.yml` | Docker Compose service definition |
| `.env.example` | Environment variable template |
| `skills/` | Repository skills (see below) |
| `skills/onedrive-deployment/` | Deployment & Kit App Streaming integration guide (auth models, Graph permissions, troubleshooting) |
| `generate_protos.sh` | Regenerate Python protobuf stubs from `.proto` files |

### Proto and OpenAPI Specs

The `proto/` and `openapi/` directories contain the Omniverse Storage API interface definitions. They are **reference material** — you do not need to modify them to run the service. If you change the API surface, regenerate Python stubs with:

```bash
./generate_protos.sh
```

## License

See `LICENSE.txt` and `PRODUCT_TERMS_OMNIVERSE.txt` for license information.

# Configuration: Environment Variables, Secrets & Credentials

Every setting is available as both a CLI flag and an environment variable
(source: `onedrive/__init__.py`, `helm/values.yaml`, `.env.example`). Precedence
follows Typer: an explicit flag overrides the env var.

> **`.env` quoting:** Do **not** wrap values in quotes. Docker `--env-file`
> treats quotes as literal characters and will corrupt URLs and auth config
> (`.env.example:6-7`).

## Core settings (both auth models)

| Flag | Env var | Default | Required | Purpose |
|---|---|---|---|---|
| `--tenant-id` | `AZURE_TENANT_ID` | — | **Yes** | Entra ID tenant. Builds the OpenID configuration URL advertised to Kit. |
| `--oidc-client-id` | `OIDC_CLIENT_ID` | — | **Yes** | Public app (client) ID advertised for Kit/portal login. |
| `--oidc-scopes` | `OIDC_SCOPES` | `openid {client_id}/.default offline_access` | No | Scope string advertised to Kit. `{client_id}` is interpolated at startup. `.default` grants whatever delegated permissions are on the app registration; `offline_access` lets Kit obtain refresh tokens. |
| `--base-uri` | `ONEDRIVE_BASE_URI` | `onedrive://me` | No | Base URI for resource addresses. |
| `--cache-ttl` | *(none)* | `300` | No | TTL (seconds) for the path **and** drive-ID caches. |
| `--grpc-port` | `GRPC_SERVER_PORT` | `50051` | No | gRPC listen port. |
| `--http-port` | `HTTP_SERVER_PORT` | `8011` | No | REST listen port. |

## OBO settings (On-Behalf-Of model only)

| Flag | Env var | Default | Purpose |
|---|---|---|---|
| `--use-obo-flow` | `USE_OBO_FLOW` | `false` | Enable OBO token exchange. |
| `--client-secret` | `AZURE_CLIENT_SECRET` | — | Confidential-client secret. **Required when OBO is on** — startup fails otherwise (`__init__.py:85-86`). |
| `--obo-timeout` | `OBO_TIMEOUT` | `10.0` | Timeout (s) for exchange calls to Azure AD. |
| `--obo-cache-maxsize` | `OBO_CACHE_MAXSIZE` | `1000` | Max exchanged tokens cached. |
| `--obo-cache-ttl` | `OBO_CACHE_TTL` | `3000` | **Upper bound** (s) on cached token lifetime. Each token is additionally capped at its real `expires_in` minus a 300 s safety margin, so a cached token never outlives its true validity (`auth.py:37, 172-195`). See the [token lifecycle](architecture.md#obo-token-lifecycle). |

Enabling OBO (source of truth `onedrive/__init__.py:84-94`):

```bash
# Local / Docker
USE_OBO_FLOW=true
AZURE_CLIENT_SECRET=<your-secret>

# CLI form
uv run omni-onedrive-storage-service onedrive \
  --tenant-id "$AZURE_TENANT_ID" --oidc-client-id "$OIDC_CLIENT_ID" \
  --use-obo-flow --client-secret "$AZURE_CLIENT_SECRET"
```

## Deployment / networking settings

| Env var | Default | Purpose |
|---|---|---|
| `SERVICE_PUBLIC_URL` | *(empty)* | Public base URL when behind an ingress. Used to build the REST **and** gRPC URLs in the `/api/v1/services` discovery response (`discovery.py:29-45`). Set this to your ingress host in App Streaming deployments. |
| `CORS_ALLOWED_ORIGINS` | `*` | Comma-separated allowed browser origins. Credentials are enabled **only** with explicit origins; a wildcard disables them (wildcard + credentials is forbidden). Lock this to your portal origin(s) in production (`values.yaml:33-37`). |
| `MAX_UPLOAD_SIZE_BYTES` | `536870912` (512 MiB) | Max bytes buffered in memory for a single body upload. Bounds per-request memory. Size pod memory limits relative to this × upload concurrency (`values.yaml:30-32, 58-67`). |

## Transport security (TLS)

Bearer tokens travel in HTTP headers and gRPC metadata, so **all traffic must be
encrypted in transit** (`helm/README.md:107-118`). Two models:

1. **TLS terminated at the ingress (recommended).** Set `httpProxy.tls.secretName`
   so Contour terminates TLS for REST and gRPC (h2c) at the edge. Protect
   in-cluster pod traffic with NetworkPolicies and/or a mesh. Don't expose the
   pod's plaintext ports to untrusted networks.
2. **TLS directly on the gRPC server.** Mount cert/key and set
   `GRPC_TLS_CERT_PATH` + `GRPC_TLS_KEY_PATH`; optionally `GRPC_TLS_CLIENT_CA_PATH`
   for mTLS. Unset → the server logs a warning and listens plaintext (only safe
   behind model 1).

## Secret handling

The OBO client secret is held in the service's **process memory** for the
lifetime of the process — this is inherent to the confidential-client OBO flow
(`auth.py:49-52`). Consequences and mitigations:

- **Never** put the secret in plain Helm values or a committed file. `.env` is
  gitignored (`.env.example:3-4`); for Kubernetes use a Secret:

  ```bash
  kubectl create secret generic onedrive-storage-obo \
    --from-literal=client-secret="<secret>" -n onedrive

  helm install onedrive-storage ./helm \
    --set obo.enabled=true --set obo.secretName="onedrive-storage-obo"
  ```

  Helm keys: `obo.enabled`, `obo.secretName`, `obo.secretKey` (default
  `client-secret`) — `helm/README.md:74-105`, `values.yaml:39-42`.

- **Rotate regularly** (Azure Portal → App registrations → Certificates &
  secrets). The service reduces exposure by evicting cached Graph tokens 300 s
  before they truly expire (`auth.py:34-37`), but the secret itself resides in
  memory until process exit.

- **Prefer** a secrets manager or workload identity federation where your
  platform supports it.

## Where to configure it

| Method | How | Reference |
|---|---|---|
| Local / source | `.env` at repo root + `uv run …` | `README.md:81-116`, `.env.example` |
| Docker Compose | `.env` + `docker compose up --build` | `README.md:129-134` |
| Docker (manual) | `docker run --env-file .env -p 8011:8011 -p 50051:50051 …` | `README.md:136-144` |
| Kubernetes (Helm) | `--set env.*`, `obo.*`, `httpProxy.*` | `helm/README.md` |

Next: [permissions.md](permissions.md).

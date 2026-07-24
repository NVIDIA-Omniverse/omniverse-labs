# Changelog

## 0.2.1

* Added sample container packaging: multi-stage `Dockerfile` under `omni_onedrive_service/` and a root `docker-compose.yml` for local runs (`docker compose up --build`).
* Added sample Helm chart under `helm/` (Deployment, Service, optional Contour `HTTPProxy` ingress, OBO secret wiring, configurable resource limits and health probes). The default `image.repository` is a generic placeholder — set it to the registry where you publish the image built from the Dockerfile, and pin a tag with `--set image.tag=<version>` (falls back to the chart `appVersion` when unset).
* Fixed an unsafe CORS configuration. The REST app combined `allow_origins=["*"]` with `allow_credentials=True`, which makes Starlette reflect any request `Origin` and permit credentialed cross-origin requests from any site. Allowed origins are now configurable via `CORS_ALLOWED_ORIGINS` (comma-separated, default `*`), and credentials are only enabled when explicit origins are configured (a wildcard now disables credentials). The service authenticates via the Bearer token in the `Authorization` header — not a CORS credential — so Kit clients are unaffected.
* Added optional TLS for the gRPC server via `GRPC_TLS_CERT_PATH` and `GRPC_TLS_KEY_PATH`, with optional `GRPC_TLS_CLIENT_CA_PATH` to require client certificates (mTLS). When TLS is not configured, the server logs a warning that Bearer tokens are sent over a plaintext connection.
* Added an in-memory upload size limit to prevent a denial-of-service. The OneDrive backend buffers the entire object in memory before writing it, so an authenticated client could stream an arbitrarily large (or size-misdeclared) upload and exhaust process memory. Both the gRPC `Write` handler and the REST body-upload path now enforce a cap (fail fast on the declared size and abort once the accumulated/streamed bytes exceed the limit — gRPC `RESOURCE_EXHAUSTED`, REST `413`). Configurable via `MAX_UPLOAD_SIZE_BYTES` (default 512 MiB).
* Added a `threading.Lock` guarding all access to the OBO token cache (`auth.py`), matching the existing drive-cache lock in `request_context.py`. `cachetools` caches are not thread-safe and mutate internal state on reads (expiry), so concurrent gRPC worker / `asyncio.to_thread` requests could corrupt the cache or serve the wrong user's Graph token.
* Added a fail-closed startup guard: the service now refuses to start (exits non-zero) if any enabled transport (REST or gRPC) would be served without Bearer token authentication wired up, so the service can never be exposed unauthenticated.
* Reduced OBO token-exchange failure logging at ERROR level to the machine-readable error code and HTTP status only. Azure AD's `error_description` (and raw response body) — which can carry correlation IDs, timestamps, and partial claim details — is now logged at DEBUG level only.
* Fixed a cache-key collision risk that could cause cross-user access. The per-user drive cache (`request_context.py`) and the OBO token cache (`auth.py`) keyed entries on a 64-bit-truncated SHA-256 of the Bearer token; a collision between two users' tokens could return the wrong user's `drive_id` or cached Graph token, executing operations against another user's OneDrive. Both now use the full SHA-256 digest.
* Hardened OBO (On-Behalf-Of) token caching so a cached Microsoft Graph token can never outlive its real validity. Cached tokens are now bound to each token's reported `expires_in` minus a safety margin (default 300s), capped by `OBO_CACHE_TTL` (now an upper bound rather than a fixed TTL). This shortens the in-memory exposure window of cached tokens and prevents serving near-expired tokens.
* Fixed intermittent `Result.ERROR` / "Error listing directory" failures when browsing folders (reported for large folders such as `Microsoft Teams Chat Files`). Two causes: (1) `list_stat` issued one Graph `versions` call *per file* (an N+1) purely to record the latest version id in each entry's identity — this is removed; listing entries now use the `current` version sentinel (read/stat of a `current` identity already targets the latest version), drastically reducing Graph calls for populated folders. (2) The general Graph request retry ignored the `Retry-After` header and backed off exponentially inside the throttling window, exhausting its budget and surfacing tenacity's `RetryError` as a generic client error; it now honors `Retry-After` (capped at 60s, bounded by attempts and total delay, clamped non-negative) and re-raises the underlying `RateLimitError`, matching the existing download retry loops. Optimistic-locking checks (`is_version_latest` and the gRPC/REST `previous_version` comparisons) now accept the `current` sentinel, so reusing a `list_stat` entry's identity as `previous_version` no longer triggers a spurious conflict.
* Fixed a path double-encoding bug that caused `ERROR_NOT_FOUND` when browsing or accessing files and folders whose names contain reserved URI characters (e.g. spaces such as `New project` or `Office Lens`). Inbound resource addresses are now URL-decoded once at the service boundary, and outbound addresses are re-encoded exactly once across the OneDrive backend and the gRPC/REST service layers.
* Fixed handling of filenames containing a semicolon (`;`): they are no longer mistaken for a `;<version>` suffix and truncated. Version suffixes are now only stripped when they are numeric.

## 0.2.0

* **BREAKING**: Changed default `OIDC_SCOPES` from `Files.ReadWrite.All User.Read offline_access` to `openid {client_id}/.default offline_access`. The `{client_id}` placeholder is replaced with `OIDC_CLIENT_ID` at startup. This narrows the default permission surface to only what is configured on the Azure AD app registration. Existing integrations must ensure their app registration has delegated permissions (e.g. `Files.ReadWrite`, `User.Read`) with admin consent, or set `OIDC_SCOPES` explicitly to restore the previous behavior.
* Added On-Behalf-Of (OBO) token exchange. When `AZURE_CLIENT_SECRET` is set, the service exchanges incoming app-scoped user tokens for Microsoft Graph tokens via Azure AD's OBO flow. This is required when an upstream auth layer validates tokens against the app's audience. Falls back to direct token pass-through when no client secret is configured.
* Renamed Helm values key `ingress` → `httpProxy` to reflect the Contour `HTTPProxy` CRD (template renamed from `ingress.yaml` to `httpproxy.yaml`).
* Added optional external authorization on the HTTPProxy via `httpProxy.authExtension`. When enabled, Contour calls a separately deployed `ExtensionService` to validate Bearer tokens at the proxy level. Discovery endpoints (`/api/v1/auth-config`) bypass authorization so Kit can bootstrap OIDC.

## 0.1.2

* Removed stale upstream Sphinx documentation (`docs/`). Documentation is now `README.md` + `openapi/` + `proto/`.

## 0.1.1

* Added Contour `HTTPProxy` ingress template with path-based routing (gRPC h2c + REST on a single host).
* Added `SERVICE_PUBLIC_URL` env var — discovery endpoints return the public URL instead of `localhost` when set.
* Added HTTPProxy values (`enabled`, `host`, `tls`, `annotations`) to the Helm chart.

## 0.1.0

Initial release of the OneDrive Storage Service for NVIDIA Omniverse.

* OneDrive for Business backend implementing the Omniverse USD Storage API (gRPC + REST).
* Multi-user support via Azure AD / Entra ID Bearer tokens — each user accesses their own OneDrive.
* Sample Docker image packaging (`omni_onedrive_service/Dockerfile`, `docker-compose.yml`).
* UV / Ruff / ty toolchain; PEP 621 project metadata.

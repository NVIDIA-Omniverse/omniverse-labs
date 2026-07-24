# Deployment Validation & Troubleshooting

## Post-deployment verification checklist

1. **Process/pod healthy.** Liveness and readiness probe `GET /api/v1/auth-config`
   (`values.yaml:69-81`). A `CrashLoopBackOff` with OBO on usually means
   `USE_OBO_FLOW=true` but no `AZURE_CLIENT_SECRET` — startup raises
   `ValueError` (`__init__.py:85-86`).
2. **Discovery correct.** `curl /api/v1/auth-config` (no token) → correct tenant
   in `openid_configuration`. Then `curl /api/v1/services` **with a Bearer
   token** → public URLs, not localhost. `/api/v1/services` requires
   authentication — without a token it returns `401`, which is expected.
3. **Auth enforced.** Every path except `/api/v1/auth-config` returns `401`
   without a token (`middleware.py:38`).
4. **Authenticated round-trip** succeeds with a real user token (see
   [app-streaming-integration.md](app-streaming-integration.md#end-to-end-connectivity-validation)).
5. **TLS everywhere.** No plaintext exposure of pod ports; ingress terminates TLS
   (`helm/README.md:107-118`).
6. **OBO caching** confirmed via logs (exchange logged once, then served from
   cache).

## Symptom → cause → fix

The responses below are exactly what the code emits, so you can match on log
lines and HTTP codes.

### 401 — "Authorization header with Bearer token is required"
- **Source:** `middleware.py:83-88` (gRPC: `UNAUTHENTICATED`, `middleware.py:159-164`).
- **Cause:** No `Authorization: Bearer <token>` header/metadata reached the
  service. On real paths this is **expected** pre-login and triggers Kit's flow.
- **Fix:** If unexpected, check the client is attaching the token; check an
  ingress/auth extension isn't stripping the header.

### 403 — "Access denied to OneDrive. Check token permissions."
- **Source:** `PermissionError` → `middleware.py:98-102` (gRPC: `PERMISSION_DENIED`).
- **Cause:** Token is valid but lacks the needed Graph permission, **or** an OBO
  exchange was refused.
- **Fix:** Confirm delegated `Files.ReadWrite` + `User.Read` are added **and
  admin-consented** ([permissions.md](permissions.md)). For OBO, see the
  exchange-failure row below.

### 401 — "Failed to resolve OneDrive. Token may be invalid or expired."
- **Source:** generic exception in drive resolution → `middleware.py:103-108`.
- **Cause:** `GET /me/drive` failed — expired/invalid token, wrong audience
  (app-scoped token with pass-through instead of OBO), or a Graph outage.
- **Fix:** Re-acquire the token. If the token is app-scoped (validated by an auth
  extension), you need **OBO** — see below.

### OBO token exchange failed — `PermissionError: Token exchange failed: <error_code>`
- **Source:** `auth.py:140-152`. The machine-readable `error` code logs at
  **ERROR**; the full `error_description` (correlation IDs, timestamps) logs only
  at **DEBUG** — enable DEBUG to see details.
- **Common `error_code` values and causes:**
  | Code | Meaning / fix |
  |---|---|
  | `invalid_client` | Wrong or **expired client secret**. Rotate/replace `AZURE_CLIENT_SECRET`. |
  | `invalid_grant` | The incoming user token isn't a valid OBO assertion (wrong audience, expired, or not issued for this app). Check the auth extension config and the app's *Expose an API* / audience. |
  | `invalid_scope` / consent errors | Delegated Graph permissions missing or not admin-consented. |
  | `invalid_request` | Malformed request — often a tenant/authority mismatch. |
- **Fix:** Verify `AZURE_TENANT_ID`, `OIDC_CLIENT_ID`, and `AZURE_CLIENT_SECRET`
  all belong to the **same** app registration and tenant.

### Login fails before reaching the service — redirect URI mismatch
- **Symptom:** Entra ID shows `AADSTS50011` / `redirect_uri` mismatch in the
  browser; the service logs nothing (the request never arrives).
- **Cause:** The app registration's Redirect URI ≠ the public URL users are
  redirected to (portal/`SERVICE_PUBLIC_URL`).
- **Fix:** Add the exact public URL as a Redirect URI ([prerequisites](prerequisites-and-app-registration.md)); ensure `SERVICE_PUBLIC_URL` matches.

### Wrong tenant / authority
- **Symptom:** `openid_configuration` points at the wrong tenant; users from the
  right tenant can't sign in.
- **Cause:** `AZURE_TENANT_ID` incorrect. The service is single-tenant
  (`README.md:226`).
- **Fix:** Set the correct Directory (tenant) ID. For multi-tenant you'd need
  `/common` or `/organizations` — out of scope for this build.

### 429 / throttling — "Rate limited. Retry after N seconds."
- **Source:** `RateLimitError` (`exceptions.py:29-34`). The client honors Graph's
  `Retry-After` (capped at 60 s) with exponential-backoff fallback
  (`graph_client.py:59-76`).
- **Fix:** Usually self-heals. Persistent throttling → reduce request volume or
  increase `--cache-ttl` to cut redundant Graph lookups.

### 404 — item / version not found
- **Source:** `ItemNotFoundError` / `VersionNotFoundError` (`exceptions.py:37-51`).
- **Cause:** Path/version doesn't exist, or a stale path cache entry. Fix the
  path; stale entries expire per `--cache-ttl`.

### ETag conflicts on rapid writes
- **Cause:** Parallel writes to the same file. The service retries with
  exponential backoff (up to 5 attempts) (`README.md:224`).
- **Fix:** Serialize writes to the same file where possible.

### CORS errors in the browser/portal
- **Cause:** `CORS_ALLOWED_ORIGINS` doesn't include the portal origin, or a
  wildcard is combined with credentials (disallowed).
- **Fix:** Set explicit portal origin(s) (`values.yaml:33-37`).

### gRPC/REST connection refused
- **Cause:** Discovery advertised `localhost` (missing `SERVICE_PUBLIC_URL`),
  wrong ports, or plaintext-vs-TLS mismatch at the ingress.
- **Fix:** Set `SERVICE_PUBLIC_URL`; verify ports `8011`/`50051` and the TLS model
  ([configuration.md](configuration.md#transport-security-tls)).

## Turning on detail

Raise the service log level to `DEBUG` to surface OBO `error_description`,
`OBO token exchange successful`, drive-resolution, and gRPC per-method logs
(`auth.py:146-158`, `middleware.py:154-193`). Keep DEBUG off in production —
descriptions can carry correlation IDs and partial claim data (`auth.py:143-145`).

Next: [architecture.md](architecture.md).

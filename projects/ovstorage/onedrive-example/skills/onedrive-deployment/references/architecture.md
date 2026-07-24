# Architecture, Graph API Implementation & Token Lifecycle

This reference explains *how the service works* and *why each authentication
component is required* — so you configure it with understanding, not by rote.

## Design in one sentence

The service is a **stateless per-user translator** between the Omniverse Storage
API (gRPC + REST) and the Microsoft Graph API: it never stores files or standing
user credentials; it carries the caller's identity through each request and
speaks Graph on their behalf.

## Request pipeline

Source: `README.md:29-35`, `middleware.py`, `request_context.py`,
`onedrive_provider.py`, `graph_client.py`, `path_mapper.py`.

```
Client (Kit / portal)
   │  Authorization: Bearer <token>
   ▼
BearerTokenMiddleware (REST) / BearerTokenInterceptor (gRPC)   middleware.py
   │  1. extract token  → contextvar request_token
   │  2. resolve drive  → GET /me/drive  (cached by token hash)
   │     store drive_id → contextvar request_drive_id
   ▼
OneDriveStorageProvider                                         onedrive_provider.py
   │  translate Storage API op → Graph API op
   │  (path cache avoids repeat lookups — path_mapper.py)
   ▼
GraphClient  → Microsoft Graph                                  graph_client.py
   │  get_authorization_header(token):                          auth.py
   │    • pass-through: use token as-is
   │    • OBO: exchange for a Graph token, then use that
   ▼
Microsoft Graph  (/me/drive, items, content, versions, upload sessions)
```

Per-request identity flows through **`contextvars`**, not globals or
thread-locals (`request_context.py:24-25`) — so concurrent requests can never
bleed one user's token/drive into another's. Streaming gRPC responses re-set the
context vars around each `next()` because the interceptor's `finally` resets them
before the generator is iterated (`middleware.py:113-144`).

## Why each auth component is required

| Component | Why it exists | What breaks without it |
|---|---|---|
| **Tenant ID** | Builds the OpenID configuration URL Kit uses to find the Entra ID login endpoint (`discovery.py:63`). | Kit can't discover where to authenticate; wrong tenant → wrong users. |
| **Client ID** | Identifies the app requesting tokens; advertised to Kit/portal (`discovery.py:69-72`). | No client to log in against; OIDC can't start. |
| **Delegated permissions** | Scope the user token so Graph authorizes file ops and `/me/drive` (`permissions.md`). | `403` on files, or drive resolution fails. |
| **Redirect URI** | Where Entra ID returns the user after login; must match the public URL. | `redirect_uri` mismatch at login, before the service is reached. |
| **Client secret** (OBO only) | Authenticates the service as a **confidential client** so Azure AD will perform the OBO exchange (`auth.py:118-159`). | Can't exchange app-scoped tokens → Graph rejects app-audience tokens. |
| **Bearer token** (per request) | The user's proof of identity; the sole authorization for every Graph call. | `401`; the service has no other way to act as the user. |

## File & API operations

`GraphClient` (`graph_client.py`) wraps Graph OneDrive operations behind the
storage provider: read/stat/enumerate, write, delete, copy/move, folder
create/list/delete, and version history via the Graph `versions` endpoint. Notable
implementation details:

- **Drive resolution:** every request first resolves the user's drive with
  `GET /me/drive`, cached by token hash (`middleware.py:51-59`).
- **Downloads:** streamed in 1 MB chunks; may use pre-authenticated Graph
  download URLs via redirect (`graph_client.py:50-51`, `README.md:44`).
- **Large uploads:** files over 4 MB use Graph **upload sessions** (chunked)
  (`graph_client.py:53-54`, `README.md:46`). In-memory body uploads are bounded
  by `MAX_UPLOAD_SIZE_BYTES`.
- **Rate limiting (429):** honors Graph's `Retry-After` (capped at 60 s) with
  exponential-backoff fallback via `tenacity` (`graph_client.py:59-76`).
- **Concurrency:** rapid writes to one file can hit ETag conflicts; retried with
  exponential backoff up to 5 attempts (`README.md:224`).
- **Typed errors:** `GraphApiError`, `RateLimitError`, `ItemNotFoundError`,
  `VersionNotFoundError` (`exceptions.py`) map cleanly to HTTP/gRPC statuses.

## Choosing a model

- **Pass-through** is correct when the token arriving at the service already has
  Graph as its **audience** (`aud = https://graph.microsoft.com`). The service
  simply attaches it (`auth.py:198-207`). Kit performs login and refresh.
- **OBO** is correct — and required — when the token's audience is **your app**,
  which happens when an ingress auth extension validates it against the app. Graph
  rejects app-audience tokens, so the service must exchange the token for a
  Graph-scoped one before calling Graph (`auth.py:14-21`).

The tell: **is there a proxy/auth extension validating tokens against your app's
audience?** Yes → OBO. No → pass-through. OBO activates when `USE_OBO_FLOW=true`
and `AZURE_CLIENT_SECRET` is set (`__init__.py:84-94`).

## Token lifecycle by model

### Pass-through
The service does **not** acquire, cache, or refresh user tokens. Kit's Client
Library owns the entire lifecycle: interactive login, storing the token, and
**refreshing** it (via the `offline_access` refresh token) when it expires. The
service only reads the token off each request and forwards it. If a token
expires, the *next* request simply carries Kit's refreshed token.

### OBO token lifecycle

When OBO is on, the service manages a **second** token — the Graph token it gets
in exchange for the user's app token. Lifecycle (`auth.py`):

1. **Acquisition.** On a cache miss, `POST` to
   `https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token` with
   `grant_type=jwt-bearer`, `requested_token_use=on_behalf_of`, the user token as
   `assertion`, and `scope=https://graph.microsoft.com/.default`
   (`auth.py:118-159`). This is the OBO flow
   ([Microsoft docs](https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-on-behalf-of-flow)).

2. **Caching.** The Graph token is cached in a thread-safe `TLRUCache`
   (`auth.py:74-78`) keyed by the **full SHA-256 digest** of the *user* token
   (`auth.py:111-115`). The full digest matters: a truncated key could collide
   two different users' tokens and hand one user another user's Graph token — the
   comment calls this out explicitly as a cross-user access risk.

3. **Expiration handling.** Cached TTL is the **smaller** of the configured
   `OBO_CACHE_TTL` upper bound and the token's real `expires_in` minus a **300 s
   safety margin** (`OBO_TOKEN_EXPIRY_MARGIN_SECONDS`, `auth.py:34-37, 172-195`).
   So a cached token is always evicted *before* it truly expires — the service
   never serves a near-dead token, and the in-memory exposure window stays
   shorter than the token's real validity. Tokens already within the margin of
   expiry are not cached at all (`auth.py:190-191`).

4. **Renewal.** There is no background refresh. On the next request after
   eviction, the cache miss simply triggers a fresh exchange (step 1). Because
   the incoming user token is itself refreshed by Kit, the service always
   exchanges a currently-valid assertion.

5. **Thread safety.** `cachetools` caches mutate internal state even on reads
   (expiry bookkeeping) and aren't thread-safe, so all cache access is guarded by
   a lock (`auth.py:75-78, 162-195`) — concurrent gRPC workers /
   `asyncio.to_thread` calls can't corrupt it or cross user tokens.

### Drive-ID cache (both models)

Independently of tokens, resolved drive IDs are cached in a `TTLCache`
(`request_context.py:27-54`), also keyed by the full SHA-256 of the token and
lock-guarded, with TTL from `--cache-ttl` (default 300 s). This avoids a
`GET /me/drive` on every request. Same anti-collision reasoning as the OBO cache.

## Security properties worth knowing

- **No cross-user leakage:** identity via contextvars; all caches keyed by full
  token digest under locks.
- **Bounded secret exposure:** the OBO client secret lives in process memory only
  (inherent to confidential-client OBO); cached Graph tokens are evicted early
  (`auth.py:34-37`). Rotate the secret regularly.
- **Least logging:** OBO failures log only the machine-readable error code at
  ERROR; sensitive `error_description` is DEBUG-only (`auth.py:143-152`).

Next: [faq-and-links.md](faq-and-links.md).

# Kit App Streaming Integration & End-to-End Connectivity

**Kit App Streaming** is NVIDIA's hosted service that runs an Omniverse **Kit**
application in the cloud and streams its rendered output to a user's browser or
thin client. The Kit app runs server-side; the user interacts with a video
stream through a **Web Streaming Portal**. In this topology the Kit app uses its
**Client Library** to call storage backends like this service to read/write USD
files in the user's OneDrive.

This service is built to slot into that stack. What follows is the integration
**contract this repository implements**. The portal/streaming-instance
provisioning side is NVIDIA-hosted infrastructure — this reference describes what
your service must expose and links out for the hosted side.

## Kit Desktop vs. Kit App Streaming

The same storage contract (discovery + Bearer token) serves both **Kit Desktop**
(Kit running on the user's own workstation, e.g. USD Composer) and **Kit App
Streaming** (Kit running in the cloud). The difference is topology, and it drives
the auth-model choice:

| | Kit Desktop (local) | Kit App Streaming (hosted) |
|---|---|---|
| Where Kit runs | User's workstation | NVIDIA-hosted cloud |
| Front end | Kit desktop UI | Web Streaming Portal (`navigator`) |
| Typical auth model | **Pass-through** (usually no ingress auth extension) | **OBO** (ingress auth extension validates app-audience tokens) |
| Redirect URI | The redirect URI Kit's Client Library expects (often a loopback) | The public portal URL |
| Service URL | A host/port the workstation can reach (e.g. `http://localhost:8011`) | `SERVICE_PUBLIC_URL` behind ingress |

### Kit Desktop setup

1. Run the service somewhere the workstation can reach it (locally, or a shared
   host).
2. In **Kit 111.0.0+**, add a storage endpoint pointing at the service base URL,
   e.g. `http://localhost:8011` (`README.md:118-125`).
3. Kit's Client Library fetches `/api/v1/auth-config`, opens a browser for Entra
   ID login, and attaches the resulting token to every request.
4. Ensure the app registration's Redirect URI matches what Kit's Client Library
   uses.

### Kit Desktop validation

- `curl http://<service>/api/v1/auth-config` → `200` with the correct tenant.
- In Kit, connect the storage endpoint and confirm the browser login completes.
- Open and save a USD file in the user's OneDrive from Kit — this exercises token
  handling, drive resolution (`GET /me/drive`), and file operations end to end.
- If the workstation's token already targets Graph (no ingress auth extension),
  keep **pass-through**; do **not** enable OBO. Enable OBO only if an auth
  extension issues app-audience tokens (see below).

The rest of this reference covers the App Streaming topology in detail.

## The integration contract

### 1. Discovery endpoints

Kit and the portal use two discovery endpoints. **Only `/api/v1/auth-config` is
unauthenticated** — it is the bootstrap endpoint clients read *before* they have
a token. `/api/v1/services` **requires a Bearer token** like every other path
(`onedrive/discovery.py`, `middleware.py:38`).

- **`GET /api/v1/auth-config`** (also `/api/v1alpha/auth-config`) — **no token
  required.** Returns:
  ```json
  {
    "openid_configuration": "https://login.microsoftonline.com/<tenant>/v2.0/.well-known/openid-configuration",
    "clients": {
      "default":        { "client_id": "<OIDC_CLIENT_ID>", "scope": "<resolved scope>" },
      "client_library": { "client_id": "<OIDC_CLIENT_ID>", "scope": "<resolved scope>" },
      "navigator":      { "client_id": "<OIDC_CLIENT_ID>", "scope": "<resolved scope>" }
    }
  }
  ```
  `client_library` is Kit's library; `navigator` is the streaming portal front end
  (`discovery.py:74-81`). `{client_id}` in the scope is interpolated at startup.

- **`GET /api/v1/services`** — **Bearer token required.** Returns the storage
  service's REST and gRPC URLs. These come from `SERVICE_PUBLIC_URL` when set,
  else localhost (`discovery.py:29-45, 85-96`). An unauthenticated request
  returns `401`, so inspect it *with* a token.

`/api/v1/auth-config` (and its `/api/v1alpha` alias) is the **only** path that
skips authentication (`middleware.py:38, 64-78`) — that is all a client needs to
bootstrap OIDC. Every other path, `/api/v1/services` included, is authenticated.

### 2. `SERVICE_PUBLIC_URL` behind ingress

In a streamed deployment the service sits behind an ingress at a public hostname,
not localhost. Set `SERVICE_PUBLIC_URL` to that hostname so the discovery
response advertises reachable REST/gRPC URLs (`discovery.py:32-41`). Without it,
clients receive `localhost` URLs and cannot connect.

### 3. Redirect URI matching

The **Redirect URI** in your Entra ID app registration must match the public URL
the user's browser is redirected to after login (`README.md:163-169`). In App
Streaming that is the **portal URL**. A mismatch produces an Entra ID error at
login (`redirect_uri` mismatch) before the service is ever reached.

### 4. CORS for browser clients

The portal is browser-based, so CORS must permit its origin. Default is `*`;
lock it to the portal origin(s) in production via `CORS_ALLOWED_ORIGINS`
(`values.yaml:33-37`, `__main__.py:170`). Credentials are only enabled with
explicit origins.

### 5. Ingress auth extension → why OBO

App Streaming deployments commonly front the service with a Contour `HTTPProxy`
plus an **external authorization (auth) extension** that validates Bearer tokens
at the edge — defense in depth (`helm/README.md:51-72`, `values.yaml:44-55`).

Here is the crux: when that extension validates tokens against **your app's**
audience, the token reaching the service is **app-scoped**, not Graph-scoped. A
Graph call with an app-audience token is rejected. The service therefore must
run **OBO** to exchange the app token for a Graph token
(`auth.py:14-21, 118-159`). This is *why* OBO exists and why the App Streaming
topology and OBO go together.

The auth extension is deployed **separately** (NVIDIA's `envoy-auth-extension`
example, your own, or another proxy). The chart only wires the `HTTPProxy` to
reference it (`helm/README.md:53-55`). Discovery endpoints automatically bypass
the extension so bootstrap still works (`helm/README.md:72`).

Enable it:

```bash
helm install onedrive-storage ./helm \
  --set httpProxy.enabled=true \
  --set httpProxy.host="storage.example.com" \
  --set httpProxy.tls.secretName="storage-tls" \
  --set httpProxy.authExtension.enabled=true \
  --set httpProxy.authExtension.extensionRef.name="auth-extension" \
  --set obo.enabled=true --set obo.secretName="onedrive-storage-obo"
```

## End-to-end connectivity validation

Work outward from the service. Each step isolates one layer.

**Step 1 — Auth-config reachable without a token.**
```bash
curl -s https://storage.example.com/api/v1/auth-config | jq .
```
Expect `200`. Verify `openid_configuration` has the **correct tenant**. This is
the only endpoint that should answer without a token.

**Step 2 — Auth is enforced everywhere else.**
```bash
# /api/v1/services WITHOUT a token → 401 (expected; it is not a bootstrap path)
curl -s -o /dev/null -w '%{http_code}\n' https://storage.example.com/api/v1/services
# any file/folder path without a token → 401
```
A `401` here is expected and correct — it's the trigger for Kit's interactive
login (`middleware.py:38, 64-88`). Then confirm the service catalog **with** a
token:
```bash
curl -s https://storage.example.com/api/v1/services \
  -H "Authorization: Bearer $USER_TOKEN" | jq .
```
Verify the `services` URLs are the **public** host (not localhost) — if they show
localhost, `SERVICE_PUBLIC_URL` is unset.

**Step 3 — Authenticated round-trip.** Obtain a user token (via the portal login,
or a test token for the app), then:
```bash
curl -s https://storage.example.com/<file-list-path> \
  -H "Authorization: Bearer $USER_TOKEN"
```
Success means: the token was accepted, drive resolution (`GET /me/drive`)
succeeded, and — in OBO mode — the token exchange worked. Failures map to the
[troubleshooting matrix](validation-and-troubleshooting.md).

**Step 4 — OBO exchange healthy (OBO only).** With `logging` at DEBUG you should
see `OBO token exchange enabled` at startup (`auth.py:104`) and
`OBO token exchange successful` on first use (`auth.py:158`). Repeated requests
by the same user should **not** re-log an exchange until the cached token nears
expiry — that confirms caching works.

**Step 5 — Portal → Kit → service.** Launch a streamed Kit session from the
portal, sign in, and perform a file open/save against OneDrive. This exercises
redirect-URI matching, CORS, the auth extension, and OBO together.

## Links for the hosted side

- NVIDIA Omniverse Kit App Streaming: https://docs.omniverse.nvidia.com/
- See [faq-and-links.md](faq-and-links.md) for the full link set.

Next: [validation-and-troubleshooting.md](validation-and-troubleshooting.md).

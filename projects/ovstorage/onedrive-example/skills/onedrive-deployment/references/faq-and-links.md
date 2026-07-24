# FAQ, Capabilities, Limitations & External Documentation

## Capabilities

- Full gRPC **and** REST APIs (v1alpha and v1beta).
- Per-user OneDrive for Business access via Bearer token (pass-through or OBO).
- File ops: read, write, stat, enumerate, delete, copy, move.
- Folder ops: list, create, delete.
- Version history via the Graph `versions` endpoint.
- Redirect-based downloads using pre-authenticated Graph URLs.
- Large-file uploads via chunked upload sessions (>4 MB).
- Discovery endpoints for Kit Client Library and the Web Streaming Portal.

(Source: `README.md:37-46`.)

## Limitations (verbatim from source)

| Limitation | Detail |
|---|---|
| **OneDrive for Business only** | Consumer/personal Microsoft accounts are not supported (`README.md:225`, `onedrive/__init__.py:18-24`). |
| **No SharePoint shared libraries** | Requires a different auth model; out of scope. |
| **Single tenant** | One Entra ID tenant. Multi-tenant would need the OpenID URL to use `/common` or `/organizations` (`README.md:226`). |
| **No custom metadata** | OneDrive has no arbitrary user metadata; metadata get/set/delete are no-ops (`README.md:222`). |
| **Version size accuracy** | Graph may report `size: 0` for some historical versions (`README.md:223`). |
| **Concurrent writes** | Rapid parallel writes to one file can hit ETag conflicts; retried up to 5× with backoff (`README.md:224`). |

## FAQ

**Q: Do I need a client secret?**
Only for OBO. Pass-through uses a public client with no secret. See the decision
table in `SKILL.md` and [architecture.md](architecture.md#choosing-a-model).

**Q: How do I know whether to use pass-through or OBO?**
Is there a proxy/auth extension validating tokens against *your app's* audience?
Yes → OBO (required). No → pass-through. See
[app-streaming-integration.md](app-streaming-integration.md#5-ingress-auth-extension--why-obo).

**Q: Does the service refresh user tokens?**
In pass-through, no — Kit does (via `offline_access`). In OBO, the service manages
only the exchanged *Graph* token's cache lifecycle, not the user's session; it
re-exchanges on cache miss. See [token lifecycle](architecture.md#token-lifecycle-by-model).

**Q: Why `Files.ReadWrite` and `User.Read` specifically?**
`Files.ReadWrite` for file ops; `User.Read` to resolve `/me/drive`. See
[permissions.md](permissions.md).

**Q: What does `.default` do in the scope?**
It tells Entra ID to issue whatever delegated permissions are consented on the
app registration — so you manage permissions in the portal, not in service
config. See [permissions.md](permissions.md#the-default-scope).

**Q: The `services` discovery response shows `localhost` in production — why?**
`SERVICE_PUBLIC_URL` is unset. Set it to your ingress host
([configuration.md](configuration.md#deployment--networking-settings)). Note
`/api/v1/services` requires a Bearer token — query it with
`-H "Authorization: Bearer $TOKEN"`; without one it returns `401`.

**Q: Which endpoints are unauthenticated?**
Only `/api/v1/auth-config` (and its `/api/v1alpha` alias) — it is the bootstrap
endpoint Kit reads before login, and Helm health probes use it. **Every** other
path, `/api/v1/services` included, requires a Bearer token and returns `401`
without one (`middleware.py:38, 64-88`). A `401` on those paths is expected, not
a deployment failure.

**Q: Login fails with a redirect_uri error before hitting the service.**
The app registration Redirect URI doesn't match your public/portal URL. See
[troubleshooting](validation-and-troubleshooting.md#login-fails-before-reaching-the-service--redirect-uri-mismatch).

**Q: Can I run only REST or only gRPC?**
Yes: `--no-grpc` (REST only) or `--no-rest` (gRPC only) (`README.md:199-206`).

**Q: How are files addressed?**
`onedrive://me/path/to/file.usd`, optionally `;N` for a specific version. IDs
returned by write/stat use the opaque `onedrive-id://` scheme — don't parse them
(`README.md:186-197`).

## External documentation

**Microsoft Graph / Entra ID**
- Microsoft Graph overview — https://learn.microsoft.com/en-us/graph/overview
- OneDrive / DriveItem in Graph — https://learn.microsoft.com/en-us/graph/api/resources/onedrive
- Graph permissions reference — https://learn.microsoft.com/en-us/graph/permissions-reference
- On-Behalf-Of flow — https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-on-behalf-of-flow
- Register an app in Entra ID — https://learn.microsoft.com/en-us/entra/identity-platform/quickstart-register-app
- Upload large files (upload sessions) — https://learn.microsoft.com/en-us/graph/api/driveitem-createuploadsession
- Throttling / `Retry-After` — https://learn.microsoft.com/en-us/graph/throttling
- Azure Portal — https://portal.azure.com

**NVIDIA Omniverse**
- Omniverse documentation — https://docs.omniverse.nvidia.com/
- Kit App Streaming — https://docs.omniverse.nvidia.com/

**In-repo references**
- `README.md` — top-level service overview and quick start
- `helm/README.md` — chart values, TLS, auth extension, OBO secret
- `.env.example` — annotated env var template
- `proto/` and `openapi/` — Storage API contracts

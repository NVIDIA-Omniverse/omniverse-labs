# Microsoft Graph Permissions

The service uses **delegated** Microsoft Graph permissions — it acts *as the
signed-in user*, never as an application with standing access to everyone's
files. This is why every request needs a per-user token and why the service can
only ever touch the calling user's own OneDrive.

## Permissions the service requires

| Permission | Type | Why it is required |
|---|---|---|
| `Files.ReadWrite` | Delegated | Read and write the user's own OneDrive files — the core file/folder operations. Without it, reads may work but writes/deletes fail with `403`. |
| `User.Read` | Delegated | Resolve the user's drive via Graph `GET /me/drive` at the start of every request pipeline (`middleware.py:51-59`). Without it, drive resolution fails and requests return `401`/`403` before any file op. |

Add both under **API permissions → Microsoft Graph → Delegated permissions**,
then **Grant admin consent** (`README.md:180-183`).

> **OBO deployments:** the Helm OBO guidance lists `Files.ReadWrite.All` +
> `User.Read` (`helm/README.md:103`). Use the `.All` variant if your OBO
> deployment needs it for your tenant's setup; `Files.ReadWrite` (own files) is
> the minimum for per-user OneDrive access. Both are delegated. Confirm against
> your tenant's admin-consent policy.

### Recommended permission set by deployment

The service only ever accesses the calling user's own drive, so the *minimum*
that works is the same everywhere. The difference is which set your deployment
standardizes on:

| Deployment | Recommended delegated permissions | Rationale |
|---|---|---|
| **Kit Desktop / pass-through** | `Files.ReadWrite` + `User.Read` | Minimum for per-user own-file access; no broader grant needed. |
| **App Streaming / OBO** | `Files.ReadWrite.All` + `User.Read` | Matches the Helm OBO guidance (`helm/README.md:103`) and the setup most OBO deployments are validated against. |

> **Team decision to ratify:** the repo currently documents `Files.ReadWrite` in
> the top-level README and `Files.ReadWrite.All` in the Helm OBO guide. Pick one
> permission set for your supported deployment and align README, Helm docs, and
> this skill on it. Until then, the table above is the recommended default; both
> are delegated and both require admin consent.

## The `.default` scope

The default scope string is `openid {client_id}/.default offline_access`
(`__init__.py:71, 129`):

- **`{client_id}/.default`** — tells Entra ID to issue a token carrying *whatever
  delegated permissions are already consented on the app registration*. This is
  why you manage permissions in the portal, not in the service config: change the
  app registration and the granted scopes follow, no redeploy needed.
- **`openid`** — OIDC sign-in.
- **`offline_access`** — lets Kit obtain a **refresh token** so it can renew the
  user's session without re-prompting. (In pass-through mode this refresh is
  Kit's responsibility, not the service's — see
  [architecture.md](architecture.md#token-lifecycle-by-model).)

To pin explicit Graph scopes instead of `.default`, override `OIDC_SCOPES`, e.g.:

```bash
OIDC_SCOPES="Files.ReadWrite.All User.Read offline_access"
```

## OBO scope

In OBO mode the service requests a Graph token with scope
`https://graph.microsoft.com/.default` during the exchange (`auth.py:134`). The
effective permissions are still the delegated ones consented on the app
registration — OBO changes the token's *audience* (to Graph), not the user's
underlying rights.

## Out of scope

This service targets **per-user OneDrive for Business only**. SharePoint shared
libraries and app-only (client-credentials) access are **not** part of this
integration and are intentionally not covered here.

Next: [app-streaming-integration.md](app-streaming-integration.md).

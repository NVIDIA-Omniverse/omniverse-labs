# Prerequisites & Entra ID App Registration

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | For running from source (`README.md:58`). |
| [UV](https://docs.astral.sh/uv/) package manager | `pip install -e .` also works. |
| Docker / Docker Compose | Optional — for containerized runs. |
| Kubernetes + Helm 3.x | Optional — for cluster deployment. Contour ingress controller if using `HTTPProxy` (`helm/README.md:5-11`). |
| An **Entra ID (Azure AD) app registration** | In the tenant whose users' OneDrive you will access. Details below. |
| A **Microsoft 365 / OneDrive for Business** tenant | Consumer/personal OneDrive is **not** supported. |

## Why an app registration is required at all

The service never asks users for passwords. Instead it relies on Entra ID as the
identity provider:

- The **client ID** and **tenant ID** are advertised to Kit via the
  `/api/v1/auth-config` discovery endpoint so Kit knows *where* and *as whom* to
  log the user in.
- Kit (or the App Streaming portal) performs the OAuth/OIDC browser login against
  Entra ID and obtains a **user Bearer token**.
- That token is what authorizes every Graph API call. No token, no access — the
  service holds no standing credentials to users' files.

So the app registration is the trust anchor: it defines the client that may
request tokens, which permissions those tokens can carry, and (for OBO) the
confidential-client secret used to exchange tokens.

## Choose your client type first

Your auth model (see `SKILL.md`) determines the app registration shape.

| | **Public client** (pass-through) | **Confidential client** (OBO) |
|---|---|---|
| Client secret | None | **Required** — created under *Certificates & secrets* |
| Who calls Graph with what audience | Kit's token is already Graph-scoped | Service exchanges an app-scoped token for a Graph token |
| Typical trigger | Direct/local use, no token-validating proxy | Ingress auth extension validates tokens against the app audience (App Streaming) |

You can register a single app that serves both, but a confidential client is
only needed when you run OBO.

## Step-by-step registration

1. Go to [Azure Portal](https://portal.azure.com) → **Microsoft Entra ID** →
   **App registrations** → **New registration**.
2. Give it a name. Set **Supported account types** to your organization
   (**single tenant**). The service is single-tenant by design; multi-tenant
   would require pointing the OpenID configuration URL at `/common` or
   `/organizations` (`README.md:226`).
3. Add **Redirect URIs** matching where users' browsers land after login:
   - For **local Kit**, use the redirect URI Kit's Client Library expects.
   - For **App Streaming / portal**, use the **public portal URL** — it must
     match `SERVICE_PUBLIC_URL` / your ingress host (`README.md:163-169`). A
     mismatch here is one of the most common failure modes; see
     [validation-and-troubleshooting.md](validation-and-troubleshooting.md).
4. Record the **Application (client) ID** → `OIDC_CLIENT_ID` / `AZURE_TENANT_ID`
   from **Directory (tenant) ID**.
5. **API permissions** → **Add a permission** → **Microsoft Graph** →
   **Delegated permissions**, then add the permissions in
   [permissions.md](permissions.md).
6. Click **Grant admin consent**.
7. **For OBO only:** **Certificates & secrets** → **New client secret**. Copy the
   value immediately (shown once) → `AZURE_CLIENT_SECRET`. Note the expiry and
   set a rotation reminder; see secret handling in
   [configuration.md](configuration.md#secret-handling).

## What you should have after this section

- `AZURE_TENANT_ID` — Directory (tenant) ID
- `OIDC_CLIENT_ID` — Application (client) ID
- Redirect URI(s) registered and matching your deployment's public URL
- Delegated Graph permissions added **and admin-consented**
- (OBO only) `AZURE_CLIENT_SECRET` — a client secret value

Continue to [configuration.md](configuration.md).

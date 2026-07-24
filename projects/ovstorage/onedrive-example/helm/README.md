# OneDrive Storage Service Helm Chart

Helm chart for deploying the OneDrive Storage Service on Kubernetes.

## Prerequisites

- Kubernetes cluster
- Helm 3.x installed
- An image built from `omni_onedrive_service/Dockerfile` and pushed to a registry your cluster can pull from
- An image pull secret in the target namespace if your registry requires authentication (name set via `imagePullSecretName`, default `regcred`)
- [Contour](https://projectcontour.io/) ingress controller (if enabling HTTPProxy)

## Installation

```bash
helm install onedrive-storage ./helm \
  --set env.AZURE_TENANT_ID="your-tenant-id" \
  --set env.OIDC_CLIENT_ID="your-client-id"
```

## Configuration

Edit `values.yaml` or pass `--set` overrides. Required values:

| Key | Description |
|-----|-------------|
| `env.AZURE_TENANT_ID` | Azure AD tenant (directory) ID |
| `env.OIDC_CLIENT_ID` | Public OIDC client ID from app registration |
| `env.OIDC_SCOPES` | OAuth scope string (default: `openid {client_id}/.default offline_access`). `{client_id}` is replaced with `OIDC_CLIENT_ID` at startup. Requires delegated permissions on the app registration. |
| `image.repository` | Container image registry/repository (set to where you publish the built image) |
| `image.tag` | Image tag (default: chart `appVersion`). Pin a concrete build with `--set image.tag=<version>` |

### HTTPProxy (Contour)

The chart includes a Contour `HTTPProxy` template for ingress routing. Enable it with:

```bash
helm install onedrive-storage ./helm \
  --set httpProxy.enabled=true \
  --set httpProxy.host="storage.example.com" \
  --set httpProxy.tls.secretName="storage-tls"
```

| Key | Description |
|-----|-------------|
| `httpProxy.enabled` | Create an HTTPProxy resource (default: `false`) |
| `httpProxy.host` | Virtual host FQDN |
| `httpProxy.tls.secretName` | TLS secret for HTTPS (optional) |
| `httpProxy.annotations` | Additional annotations on the HTTPProxy resource |

### External Authorization (auth extension)

The HTTPProxy can optionally require external authorization via a Contour `ExtensionService`. This validates Bearer tokens at the proxy level before requests reach the service — defense in depth.

The auth extension is **deployed separately**. You can use NVIDIA's `envoy-auth-extension` example, write your own, or use a different proxy/firewall solution. The chart only wires the HTTPProxy to reference it.

```bash
helm install onedrive-storage ./helm \
  --set httpProxy.enabled=true \
  --set httpProxy.host="storage.example.com" \
  --set httpProxy.authExtension.enabled=true \
  --set httpProxy.authExtension.extensionRef.name="auth-extension"
```

| Key | Description |
|-----|-------------|
| `httpProxy.authExtension.enabled` | Enable external authorization on the HTTPProxy (default: `false`) |
| `httpProxy.authExtension.extensionRef.name` | Name of the Contour `ExtensionService` to call for authorization |
| `httpProxy.authExtension.extensionRef.namespace` | Namespace of the ExtensionService (default: same namespace as HTTPProxy) |
| `httpProxy.authExtension.responseTimeout` | Timeout for authorization calls (default: `2s`) |

When enabled, the discovery endpoints (`/api/v1/auth-config`, `/api/v1alpha/auth-config`) automatically bypass authorization so Kit clients can bootstrap OIDC without a token.

### On-Behalf-Of (OBO) Token Exchange

When the auth extension validates tokens against the app's audience (not Graph's), the service must exchange incoming user tokens for Microsoft Graph tokens. This is done via Azure AD's [On-Behalf-Of flow](https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-on-behalf-of-flow).

First, create a Kubernetes Secret with your Azure AD client secret:

```bash
kubectl create secret generic onedrive-storage-obo \
  --from-literal=client-secret="your-client-secret-value" \
  -n onedrive
```

Then enable OBO in the Helm release:

```bash
helm install onedrive-storage ./helm \
  --set obo.enabled=true \
  --set obo.secretName="onedrive-storage-obo"
```

| Key | Description |
|-----|-------------|
| `obo.enabled` | Enable OBO token exchange (default: `false`) |
| `obo.secretName` | Name of the Kubernetes Secret containing the client secret (default: `<release>-obo`) |
| `obo.secretKey` | Key within the Secret (default: `client-secret`) |
| `env.OBO_TIMEOUT` | Timeout in seconds for OBO exchange requests to Azure AD (default: `10`) |
| `env.OBO_CACHE_MAXSIZE` | Maximum number of exchanged tokens to cache (default: `1000`) |
| `env.OBO_CACHE_TTL` | Upper bound (seconds) for cached OBO token lifetime (default: `3000`). Each token is additionally capped at its real expiry minus a safety margin, so a cached token never outlives its actual validity. |

The app registration requires a client secret (Azure Portal > App registrations > Certificates & secrets) and delegated Microsoft Graph permissions (`Files.ReadWrite.All`, `User.Read`) with admin consent.

> **Secret handling:** the client secret is held in the service's process memory for the confidential-client OBO exchange. Rotate it regularly (Azure Portal > App registrations > Certificates & secrets), keep it in a Kubernetes Secret (never in plain values), and prefer a secrets manager or workload identity federation where available.

### Transport Security (TLS)

Bearer tokens travel in HTTP headers and gRPC metadata, so all traffic must be encrypted in transit. Two supported models:

1. **TLS termination at the ingress (recommended, chart default).** Set `httpProxy.tls.secretName` so Contour terminates TLS for both REST and gRPC (h2c) at the edge. In-cluster traffic between the proxy and the pod is plaintext, so restrict pod-to-pod access with NetworkPolicies and/or a service mesh (mTLS). Do not expose the pod's REST/gRPC ports directly to untrusted networks.
2. **TLS directly on the gRPC server.** Mount a certificate and private key into the pod and set `GRPC_TLS_CERT_PATH` and `GRPC_TLS_KEY_PATH`; the server then uses `add_secure_port`. Optionally set `GRPC_TLS_CLIENT_CA_PATH` to require and verify client certificates (mTLS). When these are unset the gRPC server logs a warning and listens on a plaintext port (safe only behind model 1).

| Key | Description |
|-----|-------------|
| `env.GRPC_TLS_CERT_PATH` | Path to the gRPC server certificate chain (PEM). Enables gRPC TLS when set with the key. |
| `env.GRPC_TLS_KEY_PATH` | Path to the gRPC server private key (PEM). |
| `env.GRPC_TLS_CLIENT_CA_PATH` | Optional CA bundle (PEM) to require and verify client certificates (mTLS). |

## Uninstallation

```bash
helm uninstall onedrive-storage
```

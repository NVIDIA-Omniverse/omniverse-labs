## Security

NVIDIA is dedicated to the security and trust of our software products and services, including all source code repositories managed through our organization.

If you need to report a security issue, please use the appropriate contact points outlined below. **Please do not report security vulnerabilities through GitHub/GitLab.**

## Reporting Potential Security Vulnerability in an NVIDIA Product

To report a potential security vulnerability in any NVIDIA product:
- Web: [Security Vulnerability Submission Form](https://www.nvidia.com/object/submit-security-vulnerability.html)
- E-Mail: psirt@nvidia.com
    - We encourage you to use the following PGP key for secure email communication: [NVIDIA public PGP Key for communication](https://www.nvidia.com/en-us/security/pgp-key)
    - Please include the following information:
   	 - Product/Driver name and version/branch that contains the vulnerability
   	 - Type of vulnerability (code execution, denial of service, buffer overflow, etc.)
   	 - Instructions to reproduce the vulnerability
   	 - Proof-of-concept or exploit code
   	 - Potential impact of the vulnerability, including how an attacker could exploit the vulnerability

While NVIDIA currently does not have a bug bounty program, we do offer acknowledgement when an externally reported security issue is addressed under our coordinated vulnerability disclosure policy. Please visit our [Product Security Incident Response Team (PSIRT)](https://www.nvidia.com/en-us/security/psirt-policies/) policies page for more information.

## NVIDIA Product Security

For all security-related concerns, please visit NVIDIA's Product Security portal at https://www.nvidia.com/en-us/security

## Design Notes

### Intentionally unauthenticated discovery endpoints

The discovery endpoints `/api/v1/auth-config` and `/api/v1alpha/auth-config`
(and the corresponding `/services` endpoints) are intentionally served **without
authentication**. They are the OIDC bootstrap: a client must fetch the OpenID
configuration URL, the public OIDC `client_id`, and scopes *before* it can
obtain a Bearer token, so requiring a token here would make sign-in impossible.

The values returned — the Azure AD `tenant_id` (embedded in the
`openid_configuration` URL) and the **public** OIDC `client_id` and scopes — are
not secrets. They are inherently exposed during any interactive OAuth sign-in
(the browser is redirected to `https://login.microsoftonline.com/{tenant_id}/...`
with the `client_id` in the query string) and are the same values distributed to
every client application.

These endpoints return only static configuration and perform no backend or
Microsoft Graph calls. To limit reconnaissance or abuse, apply rate limiting and
request monitoring to these routes at your ingress / API gateway (e.g. Contour,
an API gateway, or a WAF) rather than in the service itself. All other REST and
gRPC endpoints require a valid Bearer token.

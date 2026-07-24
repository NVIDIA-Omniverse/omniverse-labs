# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Bearer token auth for OneDrive storage backend.

Supports two modes:
1. Pass-through: Forwards the user's token directly to Graph API (requires
   the token to already have Graph as its audience).
2. On-Behalf-Of (OBO): Exchanges the user's app-scoped token for a Graph
   token via Azure AD's OBO flow. Required when the auth extension validates
   tokens against the app's audience and the service needs to call Graph.

OBO mode is enabled by setting USE_OBO_FLOW=true, which additionally requires
AZURE_CLIENT_SECRET to be configured; startup fails if the flag is set without a
secret. Setting AZURE_CLIENT_SECRET alone does not enable OBO.
"""

import hashlib
import logging
import threading
from dataclasses import dataclass

import requests
from cachetools import TLRUCache

logger = logging.getLogger(__name__)

# Subtracted from a token's real lifetime (`expires_in`) before caching it, so a
# cached token is always evicted before it actually expires. This avoids serving
# a near-expired token and keeps the in-memory exposure window shorter than the
# token's real validity (limiting what a process memory dump could reveal).
OBO_TOKEN_EXPIRY_MARGIN_SECONDS = 300


@dataclass(frozen=True)
class OboConfig:
    """Azure AD credentials required for On-Behalf-Of token exchange.

    ``cache_ttl`` is an *upper bound* on how long an exchanged Graph token is
    cached. Each token is additionally capped at its own reported ``expires_in``
    minus ``OBO_TOKEN_EXPIRY_MARGIN_SECONDS``, so a cached token can never
    outlive its real validity.

    Note: ``client_secret`` is held in process memory for the lifetime of the
    service (required by the confidential-client OBO flow). Rotate it regularly
    and prefer sourcing it from a secrets manager / short-lived credential where
    possible.
    """

    tenant_id: str
    client_id: str
    client_secret: str
    timeout: float = 10.0
    cache_maxsize: int = 1000
    cache_ttl: int = 3000

    @property
    def token_url(self) -> str:
        return f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"


def _obo_ttu(_key: str, value: tuple[str, int], now: float) -> float:
    """Per-entry expiry for the OBO token cache (see ``_set_cached_obo_token``)."""
    _token, ttl = value
    return now + ttl


_obo_config: OboConfig | None = None
_obo_token_cache: TLRUCache[str, tuple[str, int]] = TLRUCache(maxsize=1000, ttu=_obo_ttu)
# Guards all access to _obo_token_cache: cachetools caches are not thread-safe
# and mutate internal state even on reads (expiry), so concurrent gRPC worker /
# asyncio.to_thread requests could corrupt it or serve the wrong user's token.
_obo_cache_lock = threading.Lock()


def configure_obo(
    tenant_id: str,
    client_id: str,
    client_secret: str,
    timeout: float = 10.0,
    cache_maxsize: int = 1000,
    cache_ttl: int = 3000,
) -> None:
    """Enable OBO flow by providing Azure AD credentials.

    Must be called at startup before any requests are processed.
    """
    global _obo_config, _obo_token_cache
    _obo_config = OboConfig(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
        timeout=timeout,
        cache_maxsize=cache_maxsize,
        cache_ttl=cache_ttl,
    )
    with _obo_cache_lock:
        _obo_token_cache = TLRUCache(maxsize=cache_maxsize, ttu=_obo_ttu)
    logger.info("OBO token exchange enabled (client_secret configured)")


def _obo_enabled() -> bool:
    return _obo_config is not None


def _cache_key(token: str) -> str:
    # Use the full SHA-256 digest: a truncated key risks a collision between two
    # different users' tokens, which would return another user's cached Graph
    # token (cross-user access).
    return hashlib.sha256(token.encode()).hexdigest()


def _exchange_token_obo(user_token: str) -> str:
    """Exchange a user token for a Graph API token via Azure AD OBO flow.

    See: https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-on-behalf-of-flow
    """
    assert _obo_config is not None

    cached = _get_cached_obo_token(user_token)
    if cached is not None:
        return cached

    data = {
        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
        "client_id": _obo_config.client_id,
        "client_secret": _obo_config.client_secret,
        "assertion": user_token,
        "scope": "https://graph.microsoft.com/.default",
        "requested_token_use": "on_behalf_of",
    }

    response = requests.post(_obo_config.token_url, data=data, timeout=_obo_config.timeout)

    if not response.ok:
        error_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        error_code = error_data.get("error", "unknown_error")
        # error_description / raw body can carry correlation IDs, timestamps, and
        # partial claim details — log only the machine-readable code + status at
        # ERROR level, and keep the full description at DEBUG.
        logger.error(f"OBO token exchange failed [{response.status_code}]: {error_code}")
        logger.debug(
            "OBO token exchange failure detail [%s]: %s",
            response.status_code,
            error_data.get("error_description", response.text),
        )
        raise PermissionError(f"Token exchange failed: {error_code}")

    token_data = response.json()
    access_token = token_data["access_token"]

    _set_cached_obo_token(user_token, access_token, token_data.get("expires_in"))
    logger.debug("OBO token exchange successful")
    return access_token


def _get_cached_obo_token(user_token: str) -> str | None:
    key = _cache_key(user_token)
    with _obo_cache_lock:
        entry = _obo_token_cache.get(key)
    if entry is None:
        return None
    token, _ttl = entry
    return token


def _set_cached_obo_token(user_token: str, graph_token: str, expires_in: int | str | None = None) -> None:
    """Cache a Graph token, bounding its lifetime by the token's real expiry.

    The cache TTL is the smaller of the configured cap (``cache_ttl``) and the
    token's actual ``expires_in`` minus ``OBO_TOKEN_EXPIRY_MARGIN_SECONDS``, so a
    cached token can never outlive its real validity. Tokens already within the
    safety margin of expiry are not cached.
    """
    assert _obo_config is not None

    ttl = _obo_config.cache_ttl
    if expires_in is not None:
        try:
            lifetime = int(expires_in) - OBO_TOKEN_EXPIRY_MARGIN_SECONDS
        except (TypeError, ValueError):
            lifetime = ttl
        ttl = min(ttl, lifetime)

    if ttl <= 0:
        return

    key = _cache_key(user_token)
    with _obo_cache_lock:
        _obo_token_cache[key] = (graph_token, ttl)


def get_authorization_header(token: str) -> dict[str, str]:
    """Get the Authorization header for Graph API calls.

    If OBO is configured, exchanges the user's token for a Graph token.
    Otherwise, passes through the user's token directly.
    """
    if _obo_enabled():
        token = _exchange_token_obo(token)

    return {"Authorization": f"Bearer {token}"}

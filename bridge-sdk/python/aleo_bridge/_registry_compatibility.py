"""Accept old saved transfers only when their reviewed deployment remains identical.

A plan or checkpoint records the registry version it was prepared against. A newer client
accepts it when the version is its own, or when the version is a reviewed prior snapshot AND
the route's deployment fingerprint (route + both assets + both chains' protocol domain) still
equals what that snapshot pinned — so a label can never smuggle a changed deployment, and a
saved discovery-only plan cannot become executable merely by being loaded against a snapshot
that activated its route. An older client never accepts a newer version (upgrade recovery
services before applications that write checkpoints with the new registry).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from .errors import BridgeError

if TYPE_CHECKING:
    from .registry import Registry

LEGACY_VERSION = "2026-08-31.solana-deposits.1"
PREVIOUS_VERSION = "2026-09-28.cctp-arc.1"
CURRENT_VERSION = "2026-09-30.hyperlane-bat-usdg-zec.1"


def route_fingerprint(registry: Registry, route_id: str) -> str:
    route = registry.route(route_id)
    assets = [registry.asset(route.source_asset_id), registry.asset(route.destination_asset_id)]
    chains: list[dict[str, Any]] = []
    for asset in assets:
        chain = asdict(registry.chain(asset.chain_id))
        domains = chain["protocol_domains"]
        # Sepolia's original deployment used Ethereum's Circle domain; it is now explicit.
        if chain["id"] == "sepolia" and route.protocol == "xreserve" and "xreserve" not in domains:
            domains["xreserve"] = 0
        chain["protocol_domains"] = {route.protocol: domains[route.protocol]} if route.protocol in domains else {}
        chains.append(chain)
    snapshot = {"route": asdict(route), "assets": [asdict(a) for a in assets], "chains": chains}
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def is_registry_version_compatible(registry: Registry, version: str | None, route_id: str | None) -> bool:
    """Whether a plan/checkpoint saved under *version* may use *registry*'s *route_id*.

    Exact version: yes. A reviewed prior snapshot (:data:`LEGACY_VERSION`, :data:`PREVIOUS_VERSION`):
    only when that snapshot pinned the route and the live fingerprint still matches it. Anything
    else — an unknown label, a newer version than this client's, a route the snapshot did not
    have (or had with a different deployment / availability) — is incompatible.
    """
    if route_id is None or version is None:
        return False
    if version == registry.version:
        return True
    pinned = KNOWN_ROUTE_HASHES.get(version)
    if pinned is None or route_id not in pinned:
        return False
    try:
        return route_fingerprint(registry, route_id) == pinned[route_id]
    except (BridgeError, TypeError, ValueError, KeyError):
        return False


# Generated from the captured Python registry at 1c0935dbd6532319f0b5e3967320dbcbe0827a23
# (tests/fixtures/registry-2026-08-31-python.json).
LEGACY_ROUTE_HASHES: dict[str, str] = {
    "xreserve:ethereum/usdc->aleo/usdcx": "4ed8a326a8fe3cb284648e17b9e319a1b484abc2013a85dd77d257120fd2276f",
    "xreserve:aleo/usdcx->ethereum/usdc": "09da7909d1bf2cf378dc2554669b0245a400eb658b601ee2b3509389a91d2069",
    "xreserve:sepolia/usdc->aleo-testnet/usdcx": "b0c21c2818c11fd0f44eb4b756f35f67eb269a0ae3561cceda9b13ef000701b0",
    "xreserve:aleo-testnet/usdcx->sepolia/usdc": "f98f69e1c00b1b235aba49d1bfaf0dba293749141463683dedcf3ede3ab2c0fb",
    "hyperlane:ethereum/eth->aleo/eth": "20373a6cd79a46cfa81f15c96afff141391b8d63c5590a1c66a39e952526e701",
    "hyperlane:aleo/eth->ethereum/eth": "f8a90521d79b34e3a5e34608f0857c365d0a2dd7f0a4fd38bab87c0f201fd2fa",
    "hyperlane:ethereum/wbtc->aleo/wbtc": "9ebcfd2052464d50952a7a0df297dac6c949839473f51afe733c3a32e6b55188",
    "hyperlane:aleo/wbtc->ethereum/wbtc": "a815111fe419154f1dba77b34f49a491b59593811c38eb4b1e6439880bc050f9",
    "hyperlane:ethereum/usdt->aleo/usdt": "17126cfffa855c9e56173290a196596b178753e61b3c00ec9ffe401bf4395770",
    "hyperlane:aleo/usdt->ethereum/usdt": "40aeb94f3c007034cb2235f6f65fcd16a0a34d33e129d891b69d0c606b1e0b83",
    "hyperlane:solana/sol->aleo/sol": "f8dccf00884bf41dc4ae702ec40366dacc47c65c2aa5985e7638ad11139098fc",
    "hyperlane:aleo/sol->solana/sol": "ae205673cb91518ec0e6aeea2a465bd15450e880809ee642226640690e461f80",
    "hyperlane:aleo/aleo->ethereum/aleo": "c940558ced6095dae81a41ab135ae0dece301d753321bf5530052c3927a91e24",
    "hyperlane:ethereum/aleo->aleo/aleo": "748b567a881f9de2d5802cb02d23e699dbd47fb17966522260cfce17f83fe4e4",
    "hyperlane:aleo/aleo->solana/aleo": "71a29407848f09605d8d611f678a0531803a40e982f9976c85bae4fa775e7075",
    "hyperlane:solana/aleo->aleo/aleo": "32a57bb81118f958ead32837eaed1c3252042a614ef147537c165c1bcf7847fc",
    "hyperlane:aleo/aleo->base/aleo": "622ad27af392b5992ca2083d1c9b73cff768b3b3c5630391bd1be582ccf3ff8d",
    "hyperlane:base/aleo->aleo/aleo": "01f12d5cc78e7d734a370be742f131eeb7955e4bfce5fc3f0ceb4ed5c144ec6a",
    "hyperlane:aleo/aleo->hyperevm/aleo": "f4035f8b63001f7dacb78ef7e0cf4f5b3ecbc4c426ff1455f91aca01dd277118",
    "hyperlane:hyperevm/aleo->aleo/aleo": "317af78d3ae408f373da6dba0aa561183ce2b854e4b0a1f9424247ef905578a7",
    "hyperlane:ethereum/usad->aleo/usad": "98f6bfba28a1011526f5ec58b8e30ce6526363502ff333fd218c3d8e8375e7eb",
    "hyperlane:aleo/usad->ethereum/usad": "a23c9db907e33e2d5da5bfbf9061c616a9ac4d9e5bc05495e02c05d4a013efe8"
}

# Generated from the captured Python registry at dba135bec7f88585952e16ee39a5e765f30f50de
# (tests/fixtures/registry-2026-09-28-python.json), the snapshot immediately preceding the
# BAT/USDG/ZEC rollout. The ten routes that rollout added are deliberately absent: a plan saved
# under the previous version can never name them.
PREVIOUS_ROUTE_HASHES: dict[str, str] = {
    "xreserve:ethereum/usdc->aleo/usdcx": "4ed8a326a8fe3cb284648e17b9e319a1b484abc2013a85dd77d257120fd2276f",
    "xreserve:aleo/usdcx->ethereum/usdc": "09da7909d1bf2cf378dc2554669b0245a400eb658b601ee2b3509389a91d2069",
    "xreserve:sepolia/usdc->aleo-testnet/usdcx": "b0c21c2818c11fd0f44eb4b756f35f67eb269a0ae3561cceda9b13ef000701b0",
    "xreserve:aleo-testnet/usdcx->sepolia/usdc": "f98f69e1c00b1b235aba49d1bfaf0dba293749141463683dedcf3ede3ab2c0fb",
    "hyperlane:ethereum/eth->aleo/eth": "20373a6cd79a46cfa81f15c96afff141391b8d63c5590a1c66a39e952526e701",
    "hyperlane:aleo/eth->ethereum/eth": "f8a90521d79b34e3a5e34608f0857c365d0a2dd7f0a4fd38bab87c0f201fd2fa",
    "hyperlane:ethereum/wbtc->aleo/wbtc": "9ebcfd2052464d50952a7a0df297dac6c949839473f51afe733c3a32e6b55188",
    "hyperlane:aleo/wbtc->ethereum/wbtc": "a815111fe419154f1dba77b34f49a491b59593811c38eb4b1e6439880bc050f9",
    "hyperlane:ethereum/usdt->aleo/usdt": "17126cfffa855c9e56173290a196596b178753e61b3c00ec9ffe401bf4395770",
    "hyperlane:aleo/usdt->ethereum/usdt": "40aeb94f3c007034cb2235f6f65fcd16a0a34d33e129d891b69d0c606b1e0b83",
    "hyperlane:solana/sol->aleo/sol": "f8dccf00884bf41dc4ae702ec40366dacc47c65c2aa5985e7638ad11139098fc",
    "hyperlane:aleo/sol->solana/sol": "ae205673cb91518ec0e6aeea2a465bd15450e880809ee642226640690e461f80",
    "hyperlane:aleo/aleo->ethereum/aleo": "c940558ced6095dae81a41ab135ae0dece301d753321bf5530052c3927a91e24",
    "hyperlane:ethereum/aleo->aleo/aleo": "748b567a881f9de2d5802cb02d23e699dbd47fb17966522260cfce17f83fe4e4",
    "hyperlane:aleo/aleo->solana/aleo": "71a29407848f09605d8d611f678a0531803a40e982f9976c85bae4fa775e7075",
    "hyperlane:solana/aleo->aleo/aleo": "32a57bb81118f958ead32837eaed1c3252042a614ef147537c165c1bcf7847fc",
    "hyperlane:aleo/aleo->base/aleo": "622ad27af392b5992ca2083d1c9b73cff768b3b3c5630391bd1be582ccf3ff8d",
    "hyperlane:base/aleo->aleo/aleo": "01f12d5cc78e7d734a370be742f131eeb7955e4bfce5fc3f0ceb4ed5c144ec6a",
    "hyperlane:aleo/aleo->hyperevm/aleo": "f4035f8b63001f7dacb78ef7e0cf4f5b3ecbc4c426ff1455f91aca01dd277118",
    "hyperlane:hyperevm/aleo->aleo/aleo": "317af78d3ae408f373da6dba0aa561183ce2b854e4b0a1f9424247ef905578a7",
    "hyperlane:ethereum/usad->aleo/usad": "98f6bfba28a1011526f5ec58b8e30ce6526363502ff333fd218c3d8e8375e7eb",
    "hyperlane:aleo/usad->ethereum/usad": "a23c9db907e33e2d5da5bfbf9061c616a9ac4d9e5bc05495e02c05d4a013efe8",
    "xreserve:arc/usdc->aleo/usdcx": "a362fbc8f8e431e8103a7c4ff8bf8ea98ebcd774ee78e8e8f893e4e3a6cc5778",
    "xreserve:aleo/usdcx->arc/usdc": "b1d97486bc55030b41f350fcee9693a6412a0ccd7d85e963ff6ffcfe8d29e291",
    "cctp:ethereum/usdc->arc/usdc": "57da199c23adec3a7a43391107e6260deb253cc03037dccb75919bd34f5098c7",
    "cctp:arc/usdc->ethereum/usdc": "3862c7c92f0794a0c7a676107e08901ebb3ae77ce8da10c63abd08bfd9d12633",
    "cctp:base/usdc->arc/usdc": "bf845c10ccc83dd2c4387e89de311e9c42d40d0ba6a8b7f457fa75531778f80e",
    "cctp:arc/usdc->base/usdc": "0b8c47359d34c4575652b650f047d1a6da3ccb1fc802f7fec843c8e5e7d7d439",
    "cctp:arbitrum/usdc->arc/usdc": "adffa5d93d5aad95f1055c3a8e10417e3e0f6cf361cc77dc5b56c32e1d6413fe",
    "cctp:arc/usdc->arbitrum/usdc": "4df2493eeeae9ced9b5089c8f0cbe4d25555ec5c2b19b7f645acf1bc655d228c"
}

#: Every reviewed prior snapshot a newer registry still accepts unchanged routes from.
KNOWN_ROUTE_HASHES: dict[str, dict[str, str]] = {
    LEGACY_VERSION: LEGACY_ROUTE_HASHES,
    PREVIOUS_VERSION: PREVIOUS_ROUTE_HASHES,
}

__all__ = ["CURRENT_VERSION", "KNOWN_ROUTE_HASHES", "LEGACY_ROUTE_HASHES", "LEGACY_VERSION", "PREVIOUS_ROUTE_HASHES",
           "PREVIOUS_VERSION", "is_registry_version_compatible", "route_fingerprint"]

"""Verbatim port of veil ``packages/bridge/src/registry/default.ts`` (PR #169 head ac7f718, registry version
2026-09-30.hyperlane-bat-usdg-zec.1). Plain dicts only — ``registry.py`` turns them into dataclasses.
Every key keeps veil's camelCase spelling so the brief and veil tests stay the source of truth."""
from __future__ import annotations

REGISTRY_VERSION = "2026-09-30.hyperlane-bat-usdg-zec.1"

EVM_ADDRESS = "^0x[0-9a-fA-F]{40}$"
SOLANA_ADDRESS = "^[1-9A-HJ-NP-Za-km-z]{32,44}$"
ALEO_ADDRESS = "^aleo1[0-9a-z]{58}$"

CHAINS = [
    {"id": "aleo", "displayName": "Aleo", "family": "aleo", "environment": "mainnet", "nativeCurrencySymbol": "ALEO",
     "protocolDomains": {"xreserve": 10002, "hyperlane": 1634493807}},
    {"id": "ethereum", "displayName": "Ethereum", "family": "evm", "environment": "mainnet", "nativeCurrencySymbol": "ETH",
     "protocolDomains": {"xreserve": 0, "hyperlane": 1, "cctp": 0}},
    {"id": "arc", "displayName": "Arc", "family": "evm", "environment": "mainnet", "nativeCurrencySymbol": "USDC",
     "protocolDomains": {"xreserve": 26, "cctp": 26}},
    {"id": "solana", "displayName": "Solana", "family": "solana", "environment": "mainnet", "nativeCurrencySymbol": "SOL",
     "protocolDomains": {"hyperlane": 1399811149}},
    {"id": "base", "displayName": "Base", "family": "evm", "environment": "mainnet", "nativeCurrencySymbol": "ETH",
     "protocolDomains": {"cctp": 6}},
    {"id": "arbitrum", "displayName": "Arbitrum", "family": "evm", "environment": "mainnet", "nativeCurrencySymbol": "ETH",
     "protocolDomains": {"cctp": 3}},
    {"id": "hyperevm", "displayName": "HyperEVM", "family": "evm", "environment": "mainnet", "nativeCurrencySymbol": "HYPE"},
    {"id": "aleo-testnet", "displayName": "Aleo Testnet", "family": "aleo", "environment": "testnet", "nativeCurrencySymbol": "ALEO",
     "protocolDomains": {"xreserve": 10002, "hyperlane": 1617853565}},
    {"id": "sepolia", "displayName": "Ethereum Sepolia", "family": "evm", "environment": "testnet", "nativeCurrencySymbol": "ETH",
     "protocolDomains": {"hyperlane": 11155111, "xreserve": 0}},
]

ASSETS = [
    {"id": "aleo/aleo", "key": "aleo", "chainId": "aleo", "symbol": "ALEO", "name": "Aleo", "decimals": 6, "kind": "native",
     "locator": {"kind": "aleo-program", "value": "credits.aleo"}, "addressValidationRegex": ALEO_ADDRESS},
    {"id": "aleo/usdcx", "key": "usdcx", "chainId": "aleo", "symbol": "USDCx", "name": "USDCx", "decimals": 6, "kind": "token",
     "locator": {"kind": "aleo-program", "value": "usdcx_stablecoin.aleo"}, "addressValidationRegex": ALEO_ADDRESS,
     "privacy": {"kind": "arc22", "program": "usdcx_stablecoin.aleo"}},
    {"id": "aleo/eth", "key": "eth", "chainId": "aleo", "symbol": "ETH", "name": "Hyperlane ETH", "decimals": 18, "kind": "token",
     "locator": {"kind": "aleo-program", "value": "hyp_warp_token_eth_v2.aleo",
                 "tokenId": "aleo1t7f29tq9qng2lfvrkpcuvu59jn24hrmzqdyqfn6p0u5p80npfvqqecmkj8"},
     "addressValidationRegex": ALEO_ADDRESS, "privacy": {"kind": "arc20", "program": "arc20_eth.aleo"}},
    {"id": "aleo/wbtc", "key": "wbtc", "chainId": "aleo", "symbol": "WBTC", "name": "Hyperlane WBTC", "decimals": 8, "kind": "token",
     "locator": {"kind": "aleo-program", "value": "hyp_warp_token_wbtc_v2.aleo",
                 "tokenId": "aleo1240fsvz2dhmj0cdtt8mc0yc8um9fmu236rqcl2qnlj9703hd2vpsdwyrtf"},
     "addressValidationRegex": ALEO_ADDRESS, "privacy": {"kind": "arc20", "program": "arc20_wbtc.aleo"}},
    {"id": "aleo/usdt", "key": "usdt", "chainId": "aleo", "symbol": "USDT", "name": "Hyperlane USDT", "decimals": 6, "kind": "token",
     "locator": {"kind": "aleo-program", "value": "hyp_warp_token_usdt_v2.aleo",
                 "tokenId": "aleo18yynfz0lrfx0tund540vy2z7gju7ekgqsueg5jgu28mpm2z42ufq7qua8y"},
     "addressValidationRegex": ALEO_ADDRESS, "privacy": {"kind": "arc20", "program": "arc20_usdt.aleo"}},
    {"id": "aleo/sol", "key": "sol", "chainId": "aleo", "symbol": "SOL", "name": "Hyperlane SOL", "decimals": 9, "kind": "token",
     "locator": {"kind": "aleo-program", "value": "hyp_warp_token_sol_v2.aleo",
                 "tokenId": "aleo1aa0zt0vg9uwknekpqeefkvad55swp7833wc5crp2prv0lm4djuxs5r7k6v"},
     "addressValidationRegex": ALEO_ADDRESS, "privacy": {"kind": "arc20", "program": "arc20_sol.aleo"}},
    {"id": "aleo/bat", "key": "bat", "chainId": "aleo", "symbol": "BAT", "name": "Hyperlane BAT", "decimals": 18, "kind": "token",
     "locator": {"kind": "aleo-program", "value": "hyp_warp_token_bat_v2.aleo",
                 "tokenId": "aleo1n6kjmle3t0prrwjgpwc87zytasmjdeud5rrwuuawk57ex85qr5fqcv8xzg"},
     "addressValidationRegex": ALEO_ADDRESS, "privacy": {"kind": "arc22", "program": "shield_arc22_bat.aleo"}},
    {"id": "aleo/usdg", "key": "usdg", "chainId": "aleo", "symbol": "USDG", "name": "Hyperlane USDG", "decimals": 6, "kind": "token",
     "locator": {"kind": "aleo-program", "value": "hyp_warp_token_usdg_v2.aleo",
                 "tokenId": "aleo1s4r80dv7pcggdnzsavjv45r54zjydl2jn64dejerpk6pgnfj5cysj7zzuu"},
     "addressValidationRegex": ALEO_ADDRESS, "privacy": {"kind": "arc22", "program": "shield_arc22_usdg.aleo"}},
    {"id": "aleo/zec", "key": "zec", "chainId": "aleo", "symbol": "ZEC", "name": "Hyperlane ZEC", "decimals": 8, "kind": "token",
     "locator": {"kind": "aleo-program", "value": "hyp_warp_token_zec_v2.aleo",
                 "tokenId": "aleo1m3z3en2msfdk62yje9ty7fqydxeakgx0ec6ze672q86p2yxq0sqqyjr9jd"},
     "addressValidationRegex": ALEO_ADDRESS, "privacy": {"kind": "arc22", "program": "shield_arc22_zec.aleo"}},
    {"id": "aleo/usad", "key": "usad", "chainId": "aleo", "symbol": "USAD", "name": "USAD", "decimals": 6, "kind": "token",
     "locator": {"kind": "aleo-program", "value": "usad_stablecoin.aleo"}, "addressValidationRegex": ALEO_ADDRESS},
    {"id": "ethereum/usdc", "key": "usdc", "chainId": "ethereum", "symbol": "USDC", "name": "USD Coin", "decimals": 6, "kind": "token",
     "locator": {"kind": "evm-contract", "value": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"}, "addressValidationRegex": EVM_ADDRESS},
    {"id": "ethereum/eth", "key": "eth", "chainId": "ethereum", "symbol": "ETH", "name": "Ether", "decimals": 18, "kind": "native",
     "locator": {"kind": "native", "value": "ETH"}, "addressValidationRegex": EVM_ADDRESS},
    {"id": "ethereum/wbtc", "key": "wbtc", "chainId": "ethereum", "symbol": "WBTC", "name": "Wrapped Bitcoin", "decimals": 8, "kind": "token",
     "locator": {"kind": "evm-contract", "value": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"}, "addressValidationRegex": EVM_ADDRESS},
    {"id": "ethereum/usdt", "key": "usdt", "chainId": "ethereum", "symbol": "USDT", "name": "Tether USD", "decimals": 6, "kind": "token",
     "locator": {"kind": "evm-contract", "value": "0xdAC17F958D2ee523a2206206994597C13D831ec7"}, "addressValidationRegex": EVM_ADDRESS},
    {"id": "ethereum/bat", "key": "bat", "chainId": "ethereum", "symbol": "BAT", "name": "Basic Attention Token", "decimals": 18, "kind": "token",
     "locator": {"kind": "evm-contract", "value": "0x0D8775F648430679A709E98d2b0Cb6250d2887EF"}, "addressValidationRegex": EVM_ADDRESS},
    {"id": "ethereum/usdg", "key": "usdg", "chainId": "ethereum", "symbol": "USDG", "name": "Global Dollar", "decimals": 6, "kind": "token",
     "locator": {"kind": "evm-contract", "value": "0xe343167631d89B6Ffc58B88d6b7fB0228795491D"}, "addressValidationRegex": EVM_ADDRESS},
    {"id": "ethereum/aleo", "key": "aleo", "chainId": "ethereum", "symbol": "ALEO", "name": "Hyperlane ALEO", "decimals": 6, "kind": "token",
     "addressValidationRegex": EVM_ADDRESS},
    {"id": "ethereum/usad", "key": "usad", "chainId": "ethereum", "symbol": "USAD", "name": "USAD route collateral", "decimals": 6, "kind": "token",
     "addressValidationRegex": EVM_ADDRESS},
    {"id": "solana/sol", "key": "sol", "chainId": "solana", "symbol": "SOL", "name": "Solana", "decimals": 9, "kind": "native",
     "locator": {"kind": "native", "value": "SOL"}, "addressValidationRegex": SOLANA_ADDRESS},
    {"id": "solana/bat", "key": "bat", "chainId": "solana", "symbol": "BAT", "name": "Basic Attention Token", "decimals": 8, "kind": "token",
     "locator": {"kind": "solana-mint", "value": "EPeUFDgHRxs9xxEPVaL6kfGQvCon7jmAWKVUHuux1Tpz"}, "addressValidationRegex": SOLANA_ADDRESS},
    {"id": "solana/usdg", "key": "usdg", "chainId": "solana", "symbol": "USDG", "name": "Global Dollar", "decimals": 6, "kind": "token",
     "locator": {"kind": "solana-mint", "value": "2u1tszSeqZ3qBWF3uNGPFc8TzMk2tdiwknnRMWGWjGWH"}, "addressValidationRegex": SOLANA_ADDRESS},
    {"id": "solana/zec", "key": "zec", "chainId": "solana", "symbol": "ZEC", "name": "Zcash", "decimals": 8, "kind": "token",
     "locator": {"kind": "solana-mint", "value": "A7bdiYdS5GjqGFtxf17ppRHtDKPkkRqbKtR27dxvQXaS"}, "addressValidationRegex": SOLANA_ADDRESS},
    {"id": "solana/aleo", "key": "aleo", "chainId": "solana", "symbol": "ALEO", "name": "Hyperlane ALEO", "decimals": 6, "kind": "token",
     "addressValidationRegex": SOLANA_ADDRESS},
    {"id": "base/aleo", "key": "aleo", "chainId": "base", "symbol": "ALEO", "name": "Hyperlane ALEO", "decimals": 6, "kind": "token",
     "addressValidationRegex": EVM_ADDRESS},
    {"id": "hyperevm/aleo", "key": "aleo", "chainId": "hyperevm", "symbol": "ALEO", "name": "Hyperlane ALEO", "decimals": 6, "kind": "token",
     "addressValidationRegex": EVM_ADDRESS},
    {"id": "aleo-testnet/usdcx", "key": "usdcx", "chainId": "aleo-testnet", "symbol": "USDCx", "name": "Testnet USDCx", "decimals": 6, "kind": "token",
     "locator": {"kind": "aleo-program", "value": "test_usdcx_stablecoin.aleo"}, "addressValidationRegex": ALEO_ADDRESS,
     "privacy": {"kind": "arc22", "program": "test_usdcx_stablecoin.aleo"}},
    {"id": "sepolia/usdc", "key": "usdc", "chainId": "sepolia", "symbol": "USDC", "name": "Testnet USD Coin", "decimals": 6, "kind": "token",
     "locator": {"kind": "evm-contract", "value": "0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238"}, "addressValidationRegex": EVM_ADDRESS},
]

# _source: ProvableHQ/veil packages/bridge/src/registry/default.ts @
# 3c3b457bd5f63620657321893a2487e489750d24 (PR #148).
for _chain, _token in (
    ("arc", "0x3600000000000000000000000000000000000000"),
    ("base", "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"),
    ("arbitrum", "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"),
):
    ASSETS.append({"id": f"{_chain}/usdc", "key": "usdc", "chainId": _chain, "symbol": "USDC",
                   "name": "USD Coin", "decimals": 6, "kind": "token",
                   "locator": {"kind": "evm-contract", "value": _token}, "addressValidationRegex": EVM_ADDRESS})

CCTP_SOURCE = "https://developers.circle.com/cctp/references/contract-addresses"
CCTP_MAINNET_METADATA = {
    "tokenMessenger": "0x28b5a0e9C621a5BadaA536219b3a228C8168cf5d",
    "messageTransmitter": "0x81D40F21F12A8F0E3252Bccb954D722d4c464B64",
    "attestationBaseUrl": "https://iris-api.circle.com",
    "deploymentReviewedAt": "2026-09-29",
    "tokenSource": "https://developers.circle.com/stablecoins/usdc-contract-addresses",
}
XRESERVE_SOURCE = "https://developers.circle.com/xreserve/references/supported-blockchains-and-domains"
HYPERLANE_REGISTRY_COMMIT = "2621c16f2db1ccb46643265c110dac5ca2c7c51a"
HYPERLANE_SOURCE = f"https://github.com/hyperlane-xyz/hyperlane-registry/tree/{HYPERLANE_REGISTRY_COMMIT}/deployments/warp_routes"
# BAT / USDG / ZEC warp routes (veil PR #169) are pinned from a later registry commit than the first four.
NEW_WARP_ROUTES_REGISTRY_COMMIT = "dd03567baf2a7c0a336c12a1e2b97272ca51ee9a"
BAT_HYPERLANE_CONFIG_SOURCE = f"https://github.com/hyperlane-xyz/hyperlane-registry/blob/{NEW_WARP_ROUTES_REGISTRY_COMMIT}/deployments/warp_routes/BAT/aleo-config.yaml"
USDG_HYPERLANE_CONFIG_SOURCE = f"https://github.com/hyperlane-xyz/hyperlane-registry/blob/{NEW_WARP_ROUTES_REGISTRY_COMMIT}/deployments/warp_routes/USDG/aleo-config.yaml"
ZEC_HYPERLANE_CONFIG_SOURCE = f"https://github.com/hyperlane-xyz/hyperlane-registry/blob/{NEW_WARP_ROUTES_REGISTRY_COMMIT}/deployments/warp_routes/ZEC/aleo-config.yaml"
ZEC_SOLANA_SAMPLE_TRANSFER_SOURCE = "https://explorer.hyperlane.xyz/message/0x5f0236faa02b61ea3e8f4406bbd43b7b5b74cc4010574a1fcda47d1d092e3a3e"

ALEO_USDT_HYPERLANE_CONFIG_SOURCE = "https://github.com/hyperlane-xyz/hyperlane-registry/blob/418056e21734d26a7d14692e0ec5e902cc9e86bf/deployments/warp_routes/USDT/aleo-config.yaml"
ALEO_SOL_HYPERLANE_CONFIG_SOURCE = "https://github.com/hyperlane-xyz/hyperlane-registry/blob/418056e21734d26a7d14692e0ec5e902cc9e86bf/deployments/warp_routes/SOL/aleo-config.yaml"

ETHEREUM_HYPERLANE_COMMON = {
    "sourceChainId": 1,
    "destinationDomain": 1634493807,
    "mailboxAddress": "0xc005dc82818d67AF737725bD4bf75435d065D239",
    "interchainGasPaymaster": "0x9e6B1022bE9BBF5aFd152483DAD9b88911bC8611",
    "interchainSecurityModule": "0x0000000000000000000000000000000000000000",
    "registryCommit": HYPERLANE_REGISTRY_COMMIT,
}
ETH_HYPERLANE_METADATA = {
    **ETHEREUM_HYPERLANE_COMMON,
    "routerAddress": "0x38D447694f5c1f773ae3132cf93bF30B7Ec1Fa5A",
    "routerType": "native",
    "destinationRouter": "hyp_warp_token_eth_v2.aleo/aleo1t7f29tq9qng2lfvrkpcuvu59jn24hrmzqdyqfn6p0u5p80npfvqqecmkj8",
}
WBTC_HYPERLANE_METADATA = {
    **ETHEREUM_HYPERLANE_COMMON,
    "routerAddress": "0x20CDC85778b732073F7EecEF3DF25c0d310f8772",
    "routerType": "collateral",
    "tokenAddress": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
    "destinationRouter": "hyp_warp_token_wbtc_v2.aleo/aleo1240fsvz2dhmj0cdtt8mc0yc8um9fmu236rqcl2qnlj9703hd2vpsdwyrtf",
}
USDT_HYPERLANE_METADATA = {
    **ETHEREUM_HYPERLANE_COMMON,
    "routerAddress": "0x3C2064D78e4578E8F936E3db42aEF044E33FBF31",
    "routerType": "collateral",
    "tokenAddress": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
    "destinationRouter": "hyp_warp_token_usdt_v2.aleo/aleo18yynfz0lrfx0tund540vy2z7gju7ekgqsueg5jgu28mpm2z42ufq7qua8y",
    "requiresApprovalReset": True,
}


def _evm_collateral_hyperlane_metadata(router_address: str, token_address: str, destination_router: str,
                                       hyperlane_config_source: str) -> dict:
    return {
        **ETHEREUM_HYPERLANE_COMMON,
        "routerAddress": router_address,
        "routerType": "collateral",
        "tokenAddress": token_address,
        "destinationRouter": destination_router,
        "registryCommit": NEW_WARP_ROUTES_REGISTRY_COMMIT,
        "hyperlaneConfigSource": hyperlane_config_source,
    }


BAT_HYPERLANE_METADATA = _evm_collateral_hyperlane_metadata(
    "0x516e156e987175d74614cc2bC960f148A610f0b3",
    "0x0D8775F648430679A709E98d2b0Cb6250d2887EF",
    "hyp_warp_token_bat_v2.aleo/aleo1n6kjmle3t0prrwjgpwc87zytasmjdeud5rrwuuawk57ex85qr5fqcv8xzg",
    BAT_HYPERLANE_CONFIG_SOURCE,
)
USDG_HYPERLANE_METADATA = _evm_collateral_hyperlane_metadata(
    "0xe5A2cCf532919f93855F324c1F8a7996065f53Da",
    "0xe343167631d89B6Ffc58B88d6b7fB0228795491D",
    "hyp_warp_token_usdg_v2.aleo/aleo1s4r80dv7pcggdnzsavjv45r54zjydl2jn64dejerpk6pgnfj5cysj7zzuu",
    USDG_HYPERLANE_CONFIG_SOURCE,
)


def _solana_collateral_hyperlane_metadata(warp_program_address: str, token_pda: str, dispatch_authority_pda: str,
                                          collateral_mint_address: str, spl_token_program_address: str, escrow_pda: str,
                                          destination_router: str, hyperlane_config_source: str) -> dict:
    return {
        "routerType": "spl-collateral",
        "warpProgramAddress": warp_program_address,
        "tokenPda": token_pda,
        "dispatchAuthorityPda": dispatch_authority_pda,
        "splTokenProgramAddress": spl_token_program_address,
        "collateralMintAddress": collateral_mint_address,
        "escrowPda": escrow_pda,
        "mailboxProgramAddress": "E588QtVUvresuXq2KoNEwAmoifCzYGpRBdHByN9KQMbi",
        "mailboxOutboxPda": "BvZpTuYLAR77mPhH4GtvwEWUTs53GQqkgBNuXpCePVNk",
        "igpProgramAddress": "BhNcatUDC2D5JTyeaqrdSukiVFsEHK7e3hVmKMztwefv",
        "igpProgramDataPda": "8Cv4PHJ6Cf3xY7dse7wYeZKtuQv9SAN6ujt5w22a2uho",
        "igpAccount": "JAvHW21tYXE9dtdG83DReqU2b4LUexFuCbtJT5tF8X6M",
        "igpOverheadAccount": "AkeHBbE5JkwVppujCQQ6WuxsVsJtruBAjUo6fDCFp6fF",
        "splNoopProgramAddress": "noopb9bkMVfRPU8AsbpTUg8AQkHtKwMYZiFUjNRtMmV",
        "destinationRouter": destination_router,
        "destinationDomain": 1634493807,
        # The deployed OverheadIgp adds 160,000 to the route's 300,000 base gas; the resulting
        # 460,000 payment was observed in the ZEC fixture.
        "destinationGasAmount": "460000",
        "registryCommit": NEW_WARP_ROUTES_REGISTRY_COMMIT,
        "solanaReviewedAt": "2026-09-30T00:00:00Z",
        "hyperlaneConfigSource": hyperlane_config_source,
        "solanaConfigSource": hyperlane_config_source,
    }


SOLANA_BAT_METADATA = _solana_collateral_hyperlane_metadata(
    "7CJFBsNC49upnVfMga2gj53deAjuuVchdceJQrJg5oA5",
    "DLMYtaKyG5w7djyib9XhrbkiSvMfV4AmniQHFnAetsZB",
    "7qDkiG7uwrQkyRKEQ85t65ZgrQoJjH3uh71xaUion4Su",
    "EPeUFDgHRxs9xxEPVaL6kfGQvCon7jmAWKVUHuux1Tpz",
    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
    "NdEwkjA2w7cJ3EREVDPATJfqnEotdSMhJAJicv4Qni5",
    "hyp_warp_token_bat_v2.aleo/aleo1n6kjmle3t0prrwjgpwc87zytasmjdeud5rrwuuawk57ex85qr5fqcv8xzg",
    BAT_HYPERLANE_CONFIG_SOURCE,
)
SOLANA_USDG_METADATA = _solana_collateral_hyperlane_metadata(
    "AhNVa6VpZwDwgD3U66CGUwCMRcFSFiTfBse2D495SPxW",
    "A94vqMwQQkZJ7maG9CwRrn2FFJTGcr6yqCeSP6oerc1R",
    "217ERg8p47w9DDLwETCasSpe5Rw1CTguFQYAsmjMVpdy",
    "2u1tszSeqZ3qBWF3uNGPFc8TzMk2tdiwknnRMWGWjGWH",
    "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb",
    "AijEjuEXnmgEWsBT9kyxosWG7iiFkU1dkeF6WEX5dZDg",
    "hyp_warp_token_usdg_v2.aleo/aleo1s4r80dv7pcggdnzsavjv45r54zjydl2jn64dejerpk6pgnfj5cysj7zzuu",
    USDG_HYPERLANE_CONFIG_SOURCE,
)
SOLANA_ZEC_METADATA = {
    **_solana_collateral_hyperlane_metadata(
        "2RBzic8nUNJ8KngRRbsCEjkeM9CtpQN2CCqU1cs1n2y5",
        "F3r7dPXQbCCEsgt7rz8WzPx9eGtiyoNWxtjyeYRdKDuR",
        "AHyE4g448qfMPXACBkmMkknAtMeWF8CB1Nh9ycUzii4H",
        "A7bdiYdS5GjqGFtxf17ppRHtDKPkkRqbKtR27dxvQXaS",
        "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
        "8oir78sC2Xej3gfb57wfthTUPPYkAmGRnWYZ8k9DudiS",
        "hyp_warp_token_zec_v2.aleo/aleo1m3z3en2msfdk62yje9ty7fqydxeakgx0ec6ze672q86p2yxq0sqqyjr9jd",
        ZEC_HYPERLANE_CONFIG_SOURCE,
    ),
    "solanaSampleTransferSource": ZEC_SOLANA_SAMPLE_TRANSFER_SOURCE,
}

# Intentionally non-live values that only expose the transfer_remote ABI; execution refuses them.
ALEO_PLACEHOLDER_ADDRESS = "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"
ALEO_PLACEHOLDER_BYTES32 = "[" + ", ".join(["0u8"] * 32) + "]"
ZERO_ADDRESS = "aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc"
IGP_HOOK = "aleo194tz0jmyq8rd9htvnqppqw4jqerk2p2zd8plzn3sxl06wcgsm5pq9fka74"

ALEO_MAILBOX_METADATA = {
    "aleoMailboxStateVerified": True,
    "aleoHookManagerProgram": "hyp_hook_manager.aleo",
    "aleoHookManagerProgramSource": "https://explorer.provable.com/program/hyp_hook_manager.aleo",
    "aleoMailboxProgram": "hyp_mailbox.aleo",
    "aleoMailboxProgramEdition": 0,
    "aleoMailboxProgramSource": "https://explorer.provable.com/program/hyp_mailbox.aleo",
    "aleoMailboxMetadataSource": "https://api.explorer.provable.com/v2/mainnet/program/hyp_mailbox.aleo/mapping/mailbox/true",
    "aleoMailboxMetadataReviewedAt": "2026-08-17",
    "aleoMailboxLocalDomain": 1634493807,
    "aleoMailboxObservedNonce": 170,
    "aleoMailboxObservedProcessCount": 291,
    "aleoMailboxDefaultIsm": "aleo1yvf5kcsdgnescqq2lar83mms79yh3ugvc3y0mdnlgvx4lyh5zugqr9hptk",
    "aleoMailboxDefaultHook": IGP_HOOK,
    "aleoMailboxRequiredHook": "aleo1yxevh9qgxehej46j7vueplwjcpfdfml2dje3ey4ukzknx7wzasgqnxgq82",
    "aleoMailboxDispatchProxy": "aleo1sge9kmjzs3d8fqrscy4hwn7vf9vw4jcxe877lv0m2w8hay78lsxsqg975s",
    "aleoMailboxOwner": "aleo1ypf8xgvz560ukw25hufj3d77gx69pdcy70nssdfdxd97j80d7cqs98d7x8",
}


def _aleo_hyperlane_placeholders(program: str, destination_domain: int) -> dict:
    return {
        "aleoRouterProgram": program,
        "aleoDestinationDomain": destination_domain,
        "aleoPlaceholderConfiguration": True,
        "aleoTokenType": "0",
        "aleoTokenOwner": ALEO_PLACEHOLDER_ADDRESS,
        "aleoIsm": ALEO_PLACEHOLDER_ADDRESS,
        "aleoHook": ALEO_PLACEHOLDER_ADDRESS,
        "aleoTokenId": "0field",
        "aleoRemoteRouterRecipient": ALEO_PLACEHOLDER_BYTES32,
        "aleoRemoteRouterGas": "0",
        "aleoRecipient": "[0u128, 0u128]",
        "aleoAllowanceSpender0": ALEO_PLACEHOLDER_ADDRESS,
        "aleoAllowanceAmount0": "0",
        "aleoAllowanceSpender1": ALEO_PLACEHOLDER_ADDRESS,
        "aleoAllowanceAmount1": "0",
        "aleoAllowanceSpender2": ALEO_PLACEHOLDER_ADDRESS,
        "aleoAllowanceAmount2": "0",
        "aleoAllowanceSpender3": ALEO_PLACEHOLDER_ADDRESS,
        "aleoAllowanceAmount3": "0",
        **ALEO_MAILBOX_METADATA,
    }


ALEO_WBTC_APP_METADATA = {
    "aleoAppMetadataVerified": True,
    "aleoProgramSource": "https://explorer.provable.com/program/hyp_warp_token_wbtc_v2.aleo",
    "aleoAppMetadataSource": "https://api.explorer.provable.com/v2/mainnet/program/hyp_warp_token_wbtc_v2.aleo/mapping/app_metadata/true",
    "aleoAppMetadataReviewedAt": "2026-08-17",
    "aleoProgramEdition": 0,
    "aleoTokenType": "1",
    "aleoTokenOwner": "aleo14jauje2a5sncm9u5t3mt6qqv3eq2hatkddskccs0dvsy35a0x58q0d6f95",
    "aleoIsm": ZERO_ADDRESS,
    "aleoHook": ZERO_ADDRESS,
    "aleoTokenId": "1505227928464760254508513036497943623956572091841806589002910775534260084309field",
    "aleoLocalDecimals": 8,
    "aleoRemoteDecimals": 8,
}
ALEO_ETH_APP_METADATA = {
    "aleoAppMetadataVerified": True,
    "aleoProgramSource": "https://explorer.provable.com/program/hyp_warp_token_eth_v2.aleo",
    "aleoAppMetadataSource": "https://api.explorer.provable.com/v2/mainnet/program/hyp_warp_token_eth_v2.aleo/mapping/app_metadata/true",
    "aleoAppMetadataReviewedAt": "2026-08-17",
    "aleoProgramEdition": 0,
    "aleoTokenType": "1",
    "aleoTokenOwner": "aleo1wq6f6qdqya44avznygz5hae40u3mjg64w0r93a4qfu4utpf8cg9q566f4r",
    "aleoIsm": ZERO_ADDRESS,
    "aleoHook": ZERO_ADDRESS,
    "aleoTokenId": "133188123661477349522757068766864658505569365361420630212878794317749195359field",
    "aleoLocalDecimals": 18,
    "aleoRemoteDecimals": 18,
}
ALEO_USDT_APP_METADATA = {
    "aleoAppMetadataVerified": True,
    "aleoProgramSource": "https://explorer.provable.com/program/hyp_warp_token_usdt_v2.aleo",
    "aleoAppMetadataSource": "https://api.explorer.provable.com/v2/mainnet/program/hyp_warp_token_usdt_v2.aleo/mapping/app_metadata/true",
    "aleoAppMetadataReviewedAt": "2026-08-17",
    "aleoProgramEdition": 1,
    "aleoTokenType": "1",
    "aleoTokenOwner": "aleo1l3gwacmjruxryy9c7c4fn0acyzprf29hucrvthw7f63lpyhd5y9srydq8z",
    "aleoIsm": ZERO_ADDRESS,
    "aleoHook": ZERO_ADDRESS,
    "aleoTokenId": "8295938150000417034830036849466229528602563851235385582732969109393809606969field",
    "aleoLocalDecimals": 6,
    "aleoRemoteDecimals": 18,
    "aleoScale": "1000000000000",
    "aleoHyperlaneConfigSource": ALEO_USDT_HYPERLANE_CONFIG_SOURCE,
}
ALEO_SOL_APP_METADATA = {
    "aleoAppMetadataVerified": True,
    "aleoProgramSource": "https://explorer.provable.com/program/hyp_warp_token_sol_v2.aleo",
    "aleoAppMetadataSource": "https://api.explorer.provable.com/v2/mainnet/program/hyp_warp_token_sol_v2.aleo/mapping/app_metadata/true",
    "aleoAppMetadataReviewedAt": "2026-08-17",
    "aleoProgramEdition": 0,
    "aleoTokenType": "1",
    "aleoTokenOwner": "aleo1wr8rfr4ggedjxtg5e23s38zqkgy2j05uc9l8t4akjp5zcw3levpswkwk45",
    "aleoIsm": ZERO_ADDRESS,
    "aleoHook": ZERO_ADDRESS,
    "aleoTokenId": "6148061383892805373029428966764338809222769879628268522058032128225601478383field",
    "aleoLocalDecimals": 9,
    "aleoRemoteDecimals": 9,
    "aleoHyperlaneConfigSource": ALEO_SOL_HYPERLANE_CONFIG_SOURCE,
}


def _new_warp_route_aleo_app_metadata(program: str, token_id: str, local_decimals: int, remote_decimals: int,
                                      hyperlane_config_source: str) -> dict:
    return {
        "aleoAppMetadataVerified": True,
        "aleoProgramSource": f"https://explorer.provable.com/program/{program}",
        "aleoAppMetadataSource": f"https://api.explorer.provable.com/v2/mainnet/program/{program}/mapping/app_metadata/true",
        "aleoAppMetadataReviewedAt": "2026-09-30",
        "aleoProgramEdition": 0,
        "aleoTokenType": "1",
        "aleoTokenOwner": "aleo1mx0tldt5qsqymn5a3whnmf9rx2whp837jjn0tvqgxqf86zg6dvyqnc8spm",
        "aleoIsm": ZERO_ADDRESS,
        "aleoHook": ZERO_ADDRESS,
        "aleoTokenId": token_id,
        "aleoLocalDecimals": local_decimals,
        "aleoRemoteDecimals": remote_decimals,
        "aleoHyperlaneConfigSource": hyperlane_config_source,
    }


ALEO_BAT_APP_METADATA = _new_warp_route_aleo_app_metadata(
    "hyp_warp_token_bat_v2.aleo",
    "8193754087214450113583165652573869677861364321518267024225788045720787201438field", 18, 18,
    BAT_HYPERLANE_CONFIG_SOURCE)
ALEO_USDG_APP_METADATA = _new_warp_route_aleo_app_metadata(
    "hyp_warp_token_usdg_v2.aleo",
    "4364459415416156846201796031612641041087412365006702457109915656601258641029field", 6, 6,
    USDG_HYPERLANE_CONFIG_SOURCE)
ALEO_ZEC_APP_METADATA = _new_warp_route_aleo_app_metadata(
    "hyp_warp_token_zec_v2.aleo",
    "220414605002186903241059728372192608429362177759171998609092499027319866844field", 8, 8,
    ZEC_HYPERLANE_CONFIG_SOURCE)

_ALLOWANCES = {
    "aleoAllowanceSpendersVerified": True,
    "aleoUnusedAllowancesVerified": True,
    "aleoAllowanceSpender0": IGP_HOOK,
    "aleoAllowanceSpender1": ZERO_ADDRESS,
    "aleoAllowanceSpender2": ZERO_ADDRESS,
    "aleoAllowanceSpender3": ZERO_ADDRESS,
    "aleoAllowanceAmount1": "0",
    "aleoAllowanceAmount2": "0",
    "aleoAllowanceAmount3": "0",
}
ALEO_ETH_REMOTE_ROUTER = {
    "aleoRemoteRouterVerified": True,
    "aleoRemoteRouterSource": "https://api.explorer.provable.com/v2/mainnet/program/hyp_warp_token_eth_v2.aleo/mapping/remote_routers/1u32",
    "aleoRemoteRouterReviewedAt": "2026-08-17",
    "aleoSampleTransferSource": "https://explorer.provable.com/transaction/at1vu0yckkms887zkl3qz7plnncd56jtf5zeal4uj2808upsjkusy8q7yp9v8",
    "aleoDestinationDomain": 1,
    "aleoRemoteRouterEvmAddress": "0x38D447694f5c1f773ae3132cf93bF30B7Ec1Fa5A",
    "aleoRemoteRouterRecipient": "[0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 56u8, 212u8, 71u8, 105u8, 79u8, 92u8, 31u8, 119u8, 58u8, 227u8, 19u8, 44u8, 249u8, 59u8, 243u8, 11u8, 126u8, 193u8, 250u8, 90u8]",
    "aleoRemoteRouterGas": "44000",
    **_ALLOWANCES,
}
ALEO_WBTC_REMOTE_ROUTER = {
    "aleoRemoteRouterVerified": True,
    "aleoRemoteRouterSource": "https://api.explorer.provable.com/v2/mainnet/program/hyp_warp_token_wbtc_v2.aleo/mapping/remote_routers/1u32",
    "aleoRemoteRouterReviewedAt": "2026-08-17",
    "aleoDestinationDomain": 1,
    "aleoRemoteRouterEvmAddress": "0x20CDC85778b732073F7EecEF3DF25c0d310f8772",
    "aleoRemoteRouterRecipient": "[0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 32u8, 205u8, 200u8, 87u8, 120u8, 183u8, 50u8, 7u8, 63u8, 126u8, 236u8, 239u8, 61u8, 242u8, 92u8, 13u8, 49u8, 15u8, 135u8, 114u8]",
    "aleoRemoteRouterGas": "68000",
    **_ALLOWANCES,
}
ALEO_USDT_ETHEREUM_REMOTE_ROUTER = {
    "aleoRemoteRouterVerified": True,
    "aleoRemoteRouterSource": "https://api.explorer.provable.com/v2/mainnet/program/hyp_warp_token_usdt_v2.aleo/mapping/remote_routers/1u32",
    "aleoRemoteRouterReviewedAt": "2026-08-17",
    "aleoSampleTransferSource": "https://explorer.provable.com/transaction/at19caeeee8v3xc4kfwen4tx89f0tnggrpjp0anrhq2ca3y82xr9q8qyz8a9r",
    "aleoSampleTransferDestinationDomain": 56,
    "aleoDestinationDomain": 1,
    "aleoRemoteRouterEvmAddress": "0x3C2064D78e4578E8F936E3db42aEF044E33FBF31",
    "aleoRemoteRouterRecipient": "[0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 60u8, 32u8, 100u8, 215u8, 142u8, 69u8, 120u8, 232u8, 249u8, 54u8, 227u8, 219u8, 66u8, 174u8, 240u8, 68u8, 227u8, 63u8, 191u8, 49u8]",
    "aleoRemoteRouterGas": "68000",
    **_ALLOWANCES,
}
ALEO_SOL_REMOTE_ROUTER = {
    "aleoRemoteRouterVerified": True,
    "aleoRemoteRouterSource": "https://api.explorer.provable.com/v2/mainnet/program/hyp_warp_token_sol_v2.aleo/mapping/remote_routers/1399811149u32",
    "aleoRemoteRouterReviewedAt": "2026-08-17",
    "aleoSampleTransitionId": "au15fg39h53h55tkj0nexrme3k6pvgxngxapcyajdhf06jcg3cyeugq5kd7hg",
    "aleoDestinationDomain": 1399811149,
    "aleoRemoteRouterSolanaAddress": "8YGT2pZwyZe94qBpGzWfY2TMEVcwaQ1bXAE7YAgpUaM7",
    "aleoRemoteRouterRecipient": "[112u8, 4u8, 72u8, 22u8, 219u8, 143u8, 68u8, 202u8, 21u8, 197u8, 236u8, 182u8, 198u8, 142u8, 52u8, 96u8, 142u8, 38u8, 51u8, 113u8, 116u8, 143u8, 96u8, 123u8, 104u8, 126u8, 97u8, 73u8, 7u8, 6u8, 211u8, 122u8]",
    "aleoRemoteRouterGas": "300000",
    **_ALLOWANCES,
}


def _new_warp_route_aleo_remote_router(program: str, destination_domain: int, recipient: str, gas: str) -> dict:
    return {
        "aleoRemoteRouterVerified": True,
        "aleoRemoteRouterSource": f"https://api.explorer.provable.com/v2/mainnet/program/{program}/mapping/remote_routers/{destination_domain}u32",
        "aleoRemoteRouterReviewedAt": "2026-09-30",
        "aleoDestinationDomain": destination_domain,
        "aleoRemoteRouterRecipient": recipient,
        "aleoRemoteRouterGas": gas,
        **_ALLOWANCES,
    }


ALEO_BAT_ETHEREUM_REMOTE_ROUTER = {
    **_new_warp_route_aleo_remote_router(
        "hyp_warp_token_bat_v2.aleo", 1,
        "[0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 81u8, 110u8, 21u8, 110u8, 152u8, 113u8, 117u8, 215u8, 70u8, 20u8, 204u8, 43u8, 201u8, 96u8, 241u8, 72u8, 166u8, 16u8, 240u8, 179u8]",
        "68000"),
    "aleoRemoteRouterEvmAddress": "0x516e156e987175d74614cc2bC960f148A610f0b3",
}
ALEO_USDG_ETHEREUM_REMOTE_ROUTER = {
    **_new_warp_route_aleo_remote_router(
        "hyp_warp_token_usdg_v2.aleo", 1,
        "[0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 229u8, 162u8, 204u8, 245u8, 50u8, 145u8, 159u8, 147u8, 133u8, 95u8, 50u8, 76u8, 31u8, 138u8, 121u8, 150u8, 6u8, 95u8, 83u8, 218u8]",
        "68000"),
    "aleoRemoteRouterEvmAddress": "0xe5A2cCf532919f93855F324c1F8a7996065f53Da",
}
ALEO_BAT_SOLANA_REMOTE_ROUTER = {
    **_new_warp_route_aleo_remote_router(
        "hyp_warp_token_bat_v2.aleo", 1399811149,
        "[92u8, 11u8, 2u8, 77u8, 223u8, 89u8, 80u8, 93u8, 96u8, 156u8, 172u8, 143u8, 8u8, 61u8, 198u8, 142u8, 152u8, 199u8, 250u8, 130u8, 173u8, 140u8, 154u8, 209u8, 175u8, 222u8, 50u8, 100u8, 145u8, 47u8, 150u8, 246u8]",
        "300000"),
    "aleoRemoteRouterSolanaAddress": "7CJFBsNC49upnVfMga2gj53deAjuuVchdceJQrJg5oA5",
}
ALEO_USDG_SOLANA_REMOTE_ROUTER = {
    **_new_warp_route_aleo_remote_router(
        "hyp_warp_token_usdg_v2.aleo", 1399811149,
        "[144u8, 16u8, 183u8, 105u8, 178u8, 120u8, 227u8, 39u8, 114u8, 178u8, 9u8, 147u8, 182u8, 124u8, 20u8, 110u8, 203u8, 85u8, 53u8, 210u8, 96u8, 74u8, 233u8, 86u8, 22u8, 175u8, 37u8, 94u8, 7u8, 159u8, 40u8, 147u8]",
        "300000"),
    "aleoRemoteRouterSolanaAddress": "AhNVa6VpZwDwgD3U66CGUwCMRcFSFiTfBse2D495SPxW",
}
ALEO_ZEC_SOLANA_REMOTE_ROUTER = {
    **_new_warp_route_aleo_remote_router(
        "hyp_warp_token_zec_v2.aleo", 1399811149,
        "[21u8, 14u8, 14u8, 254u8, 170u8, 124u8, 94u8, 246u8, 229u8, 126u8, 105u8, 153u8, 117u8, 17u8, 66u8, 177u8, 20u8, 170u8, 61u8, 231u8, 51u8, 53u8, 114u8, 116u8, 158u8, 102u8, 137u8, 29u8, 53u8, 21u8, 213u8, 224u8]",
        "300000"),
    "aleoRemoteRouterSolanaAddress": "2RBzic8nUNJ8KngRRbsCEjkeM9CtpQN2CCqU1cs1n2y5",
    "aleoSampleTransferSource": ZEC_SOLANA_SAMPLE_TRANSFER_SOURCE,
}
ALEO_WITHDRAWAL_ACTIVATION = {"aleoPlaceholderConfiguration": False, "aleoWithdrawalReviewedAt": "2026-08-26"}
NEW_WARP_ROUTE_ALEO_ACTIVATION = {"aleoPlaceholderConfiguration": False, "aleoWithdrawalReviewedAt": "2026-09-30"}

SOLANA_SOL_DEPOSIT_METADATA = {
    "warpProgramAddress": "8YGT2pZwyZe94qBpGzWfY2TMEVcwaQ1bXAE7YAgpUaM7",
    "tokenPda": "JDkpV5CsSbhyGhHhirC5DjGPTcuKWUVHtBZ5MFsgu3ZW",
    "nativeCollateralPda": "8HY3hxmnrWwqEmcdwkSnfN9wEQFUkyiwZvU1vMbnXgbC",
    "dispatchAuthorityPda": "ATDttjggAZKyS19kcV6Rn56oMi49gDprZGckRou9vkkY",
    "mailboxProgramAddress": "E588QtVUvresuXq2KoNEwAmoifCzYGpRBdHByN9KQMbi",
    "mailboxOutboxPda": "BvZpTuYLAR77mPhH4GtvwEWUTs53GQqkgBNuXpCePVNk",
    "igpProgramAddress": "BhNcatUDC2D5JTyeaqrdSukiVFsEHK7e3hVmKMztwefv",
    "igpProgramDataPda": "8Cv4PHJ6Cf3xY7dse7wYeZKtuQv9SAN6ujt5w22a2uho",
    "igpAccount": "JAvHW21tYXE9dtdG83DReqU2b4LUexFuCbtJT5tF8X6M",
    "igpOverheadAccount": "AkeHBbE5JkwVppujCQQ6WuxsVsJtruBAjUo6fDCFp6fF",
    "splNoopProgramAddress": "noopb9bkMVfRPU8AsbpTUg8AQkHtKwMYZiFUjNRtMmV",
    "destinationDomain": 1634493807,
    "destinationGasAmount": "464000",
    "registryCommit": "418056e21734d26a7d14692e0ec5e902cc9e86bf",
    "solanaReviewedAt": "2026-08-31",
    "solanaConfigSource": ALEO_SOL_HYPERLANE_CONFIG_SOURCE,
}

XRESERVE_MAINNET_METADATA = {
    "xReserveContract": "0x8888888199b2Df864bf678259607d6D5EBb4e3Ce",
    "sourceChainId": 1,
    "sourceDomain": 0,
    "ethereumDestinationDomain": 0,
    "arcDestinationDomain": 26,
    "remoteDomain": 10002,
    "remoteToken": "usdcx_stablecoin.aleo",
    "remoteTokenBytes32": "0x11ea7dab1d29d5f61500582c63e98c42e1165f9ba050ea9d0c6af9f871987711",
    "minimumAmountAtomic": "2000000",
    "withdrawalFeeAtomic": "2000000",
    "maxFeeAtomic": "100000",
    "bridgeProgram": "usdcx_bridge_v2.aleo",
    "wrapperProgram": "shielded_usdcx_wrapper.aleo",
    "attestationBaseUrl": "https://xreserve-api.circle.com/v1/attestations",
}
XRESERVE_TESTNET_METADATA = {
    "xReserveContract": "0x008888878f94C0d87defdf0B07f46B93C1934442",
    "sourceChainId": 11155111,
    "sourceDomain": 0,
    "ethereumDestinationDomain": 0,
    "arcDestinationDomain": 26,
    "remoteDomain": 10002,
    "remoteToken": "test_usdcx_stablecoin.aleo",
    "remoteTokenBytes32": "0xb143ed52c774cd1d4a519d0e796f15916be5a9e1d45edcd9852dd23f68f53401",
    "minimumAmountAtomic": "2000000",
    "withdrawalFeeAtomic": "2000000",
    "maxFeeAtomic": "100000",
    "bridgeProgram": "test_usdcx_bridge_v2.aleo",
    "wrapperProgram": "shielded_usdcx_wrapper.aleo",
    "attestationBaseUrl": "https://xreserve-api-testnet.circle.com/v1/attestations",
}

XRESERVE_ARC_METADATA = {
    **XRESERVE_MAINNET_METADATA,
    "sourceChainId": 5042, "sourceDomain": 26,
    "minimumBurnAmountAtomic": "2000000", "withdrawalFeeAtomic": "16400",
    "withdrawalFeeUrl": "https://api.usdcx.aleo.org/api/estimate-burn-fee",
    "withdrawalFeeChain": "arc",
    "withdrawalFeeSource": "https://usdcx.aleo.org/assets/index-C4YEghH3.js",
    "deploymentSource": "https://docs.aleo.org/build/common-uses/usdcx_bridge",
    "onchainReviewedAt": "2026-09-28",
}


def _route(id: str, protocol: str, environment: str, source_asset_id: str, destination_asset_id: str,
           availability: str, deployment_id: str, metadata: dict, source: str | None = None) -> dict:
    return {
        "id": id, "protocol": protocol, "environment": environment,
        "sourceAssetId": source_asset_id, "destinationAssetId": destination_asset_id,
        "availability": availability, "deploymentId": deployment_id,
        "source": source or (XRESERVE_SOURCE if protocol == "xreserve" else CCTP_SOURCE if protocol == "cctp" else HYPERLANE_SOURCE),
        "metadata": dict(metadata),
    }


def _pair(protocol: str, environment: str, left: str, right: str, availability: str,
          deployment_id: str, metadata: dict) -> list[dict]:
    return [
        _route(f"{protocol}:{left}->{right}", protocol, environment, left, right, availability, deployment_id, metadata),
        _route(f"{protocol}:{right}->{left}", protocol, environment, right, left, availability, deployment_id, metadata),
    ]


ROUTES = [
    *_pair("xreserve", "mainnet", "ethereum/usdc", "aleo/usdcx", "active", "xreserve-usdcx-aleo", XRESERVE_MAINNET_METADATA),
    *_pair("xreserve", "testnet", "sepolia/usdc", "aleo-testnet/usdcx", "active", "xreserve-usdcx-aleo-testnet", XRESERVE_TESTNET_METADATA),
    _route("hyperlane:ethereum/eth->aleo/eth", "hyperlane", "mainnet", "ethereum/eth", "aleo/eth", "active", "ETH/aleo",
           {**ETH_HYPERLANE_METADATA, **ALEO_MAILBOX_METADATA}),
    _route("hyperlane:aleo/eth->ethereum/eth", "hyperlane", "mainnet", "aleo/eth", "ethereum/eth", "active", "ETH/aleo",
           {**ETH_HYPERLANE_METADATA, **_aleo_hyperlane_placeholders("hyp_warp_token_eth_v2.aleo", 1),
            **ALEO_ETH_APP_METADATA, **ALEO_ETH_REMOTE_ROUTER, **ALEO_WITHDRAWAL_ACTIVATION}),
    _route("hyperlane:ethereum/wbtc->aleo/wbtc", "hyperlane", "mainnet", "ethereum/wbtc", "aleo/wbtc", "active", "WBTC/aleo",
           {**WBTC_HYPERLANE_METADATA, **ALEO_MAILBOX_METADATA}),
    _route("hyperlane:aleo/wbtc->ethereum/wbtc", "hyperlane", "mainnet", "aleo/wbtc", "ethereum/wbtc", "active", "WBTC/aleo",
           {**WBTC_HYPERLANE_METADATA, **_aleo_hyperlane_placeholders("hyp_warp_token_wbtc_v2.aleo", 1),
            **ALEO_WBTC_APP_METADATA, **ALEO_WBTC_REMOTE_ROUTER, **ALEO_WITHDRAWAL_ACTIVATION}),
    _route("hyperlane:ethereum/usdt->aleo/usdt", "hyperlane", "mainnet", "ethereum/usdt", "aleo/usdt", "active", "USDT/aleo",
           {**USDT_HYPERLANE_METADATA, **ALEO_MAILBOX_METADATA}),
    _route("hyperlane:aleo/usdt->ethereum/usdt", "hyperlane", "mainnet", "aleo/usdt", "ethereum/usdt", "active", "USDT/aleo",
           {**USDT_HYPERLANE_METADATA, **_aleo_hyperlane_placeholders("hyp_warp_token_usdt_v2.aleo", 1),
            **ALEO_USDT_APP_METADATA, **ALEO_USDT_ETHEREUM_REMOTE_ROUTER, **ALEO_WITHDRAWAL_ACTIVATION}),
    _route("hyperlane:ethereum/bat->aleo/bat", "hyperlane", "mainnet", "ethereum/bat", "aleo/bat", "active", "BAT/aleo",
           {**BAT_HYPERLANE_METADATA, **ALEO_MAILBOX_METADATA}, BAT_HYPERLANE_CONFIG_SOURCE),
    _route("hyperlane:aleo/bat->ethereum/bat", "hyperlane", "mainnet", "aleo/bat", "ethereum/bat", "active", "BAT/aleo",
           {**BAT_HYPERLANE_METADATA, **_aleo_hyperlane_placeholders("hyp_warp_token_bat_v2.aleo", 1),
            **ALEO_BAT_APP_METADATA, **ALEO_BAT_ETHEREUM_REMOTE_ROUTER, **NEW_WARP_ROUTE_ALEO_ACTIVATION,
            "hyperlaneConfigSource": BAT_HYPERLANE_CONFIG_SOURCE}, BAT_HYPERLANE_CONFIG_SOURCE),
    _route("hyperlane:ethereum/usdg->aleo/usdg", "hyperlane", "mainnet", "ethereum/usdg", "aleo/usdg", "active", "USDG/aleo",
           {**USDG_HYPERLANE_METADATA, **ALEO_MAILBOX_METADATA}, USDG_HYPERLANE_CONFIG_SOURCE),
    _route("hyperlane:aleo/usdg->ethereum/usdg", "hyperlane", "mainnet", "aleo/usdg", "ethereum/usdg", "active", "USDG/aleo",
           {**USDG_HYPERLANE_METADATA, **_aleo_hyperlane_placeholders("hyp_warp_token_usdg_v2.aleo", 1),
            **ALEO_USDG_APP_METADATA, **ALEO_USDG_ETHEREUM_REMOTE_ROUTER, **NEW_WARP_ROUTE_ALEO_ACTIVATION,
            "hyperlaneConfigSource": USDG_HYPERLANE_CONFIG_SOURCE}, USDG_HYPERLANE_CONFIG_SOURCE),
    _route("hyperlane:solana/sol->aleo/sol", "hyperlane", "mainnet", "solana/sol", "aleo/sol", "active", "SOL/aleo",
           {**SOLANA_SOL_DEPOSIT_METADATA, **ALEO_MAILBOX_METADATA}),
    _route("hyperlane:aleo/sol->solana/sol", "hyperlane", "mainnet", "aleo/sol", "solana/sol", "active", "SOL/aleo",
           {**_aleo_hyperlane_placeholders("hyp_warp_token_sol_v2.aleo", 1399811149),
            **ALEO_SOL_APP_METADATA, **ALEO_SOL_REMOTE_ROUTER, **ALEO_WITHDRAWAL_ACTIVATION}),
    _route("hyperlane:solana/bat->aleo/bat", "hyperlane", "mainnet", "solana/bat", "aleo/bat", "active", "BAT/aleo",
           {**SOLANA_BAT_METADATA, **ALEO_MAILBOX_METADATA}, BAT_HYPERLANE_CONFIG_SOURCE),
    _route("hyperlane:aleo/bat->solana/bat", "hyperlane", "mainnet", "aleo/bat", "solana/bat", "active", "BAT/aleo",
           {**SOLANA_BAT_METADATA, **_aleo_hyperlane_placeholders("hyp_warp_token_bat_v2.aleo", 1399811149),
            **ALEO_BAT_APP_METADATA, **ALEO_BAT_SOLANA_REMOTE_ROUTER, **NEW_WARP_ROUTE_ALEO_ACTIVATION,
            "hyperlaneConfigSource": BAT_HYPERLANE_CONFIG_SOURCE}, BAT_HYPERLANE_CONFIG_SOURCE),
    _route("hyperlane:solana/usdg->aleo/usdg", "hyperlane", "mainnet", "solana/usdg", "aleo/usdg", "active", "USDG/aleo",
           {**SOLANA_USDG_METADATA, **ALEO_MAILBOX_METADATA}, USDG_HYPERLANE_CONFIG_SOURCE),
    _route("hyperlane:aleo/usdg->solana/usdg", "hyperlane", "mainnet", "aleo/usdg", "solana/usdg", "active", "USDG/aleo",
           {**SOLANA_USDG_METADATA, **_aleo_hyperlane_placeholders("hyp_warp_token_usdg_v2.aleo", 1399811149),
            **ALEO_USDG_APP_METADATA, **ALEO_USDG_SOLANA_REMOTE_ROUTER, **NEW_WARP_ROUTE_ALEO_ACTIVATION,
            "hyperlaneConfigSource": USDG_HYPERLANE_CONFIG_SOURCE}, USDG_HYPERLANE_CONFIG_SOURCE),
    _route("hyperlane:solana/zec->aleo/zec", "hyperlane", "mainnet", "solana/zec", "aleo/zec", "active", "ZEC/aleo",
           {**SOLANA_ZEC_METADATA, **ALEO_MAILBOX_METADATA}, ZEC_HYPERLANE_CONFIG_SOURCE),
    _route("hyperlane:aleo/zec->solana/zec", "hyperlane", "mainnet", "aleo/zec", "solana/zec", "active", "ZEC/aleo",
           {**SOLANA_ZEC_METADATA, **_aleo_hyperlane_placeholders("hyp_warp_token_zec_v2.aleo", 1399811149),
            **ALEO_ZEC_APP_METADATA, **ALEO_ZEC_SOLANA_REMOTE_ROUTER, **NEW_WARP_ROUTE_ALEO_ACTIVATION,
            "hyperlaneConfigSource": ZEC_HYPERLANE_CONFIG_SOURCE}, ZEC_HYPERLANE_CONFIG_SOURCE),
    *_pair("hyperlane", "mainnet", "aleo/aleo", "ethereum/aleo", "metadata-required", "ALEO/aleo", ALEO_MAILBOX_METADATA),
    *_pair("hyperlane", "mainnet", "aleo/aleo", "solana/aleo", "metadata-required", "ALEO/aleo", ALEO_MAILBOX_METADATA),
    *_pair("hyperlane", "mainnet", "aleo/aleo", "base/aleo", "metadata-required", "ALEO/aleo", ALEO_MAILBOX_METADATA),
    *_pair("hyperlane", "mainnet", "aleo/aleo", "hyperevm/aleo", "metadata-required", "ALEO/aleo", ALEO_MAILBOX_METADATA),
    _route("hyperlane:ethereum/usad->aleo/usad", "hyperlane", "mainnet", "ethereum/usad", "aleo/usad", "metadata-required", "USAD/aleo",
           ALEO_MAILBOX_METADATA),
    _route("hyperlane:aleo/usad->ethereum/usad", "hyperlane", "mainnet", "aleo/usad", "ethereum/usad", "metadata-required", "USAD/aleo",
           _aleo_hyperlane_placeholders("hyp_warp_token_usad_v2.aleo", 1)),
]

ROUTES.extend(_pair("xreserve", "mainnet", "arc/usdc", "aleo/usdcx", "active",
                    "xreserve-usdcx-aleo-arc", XRESERVE_ARC_METADATA))
for _chain, _chain_id, _domain in (("ethereum", 1, 0), ("base", 8453, 6), ("arbitrum", 42161, 3)):
    for _src, _dst, _sid, _did, _sd, _dd in (
        (_chain, "arc", _chain_id, 5042, _domain, 26), ("arc", _chain, 5042, _chain_id, 26, _domain),
    ):
        ROUTES.append(_route(f"cctp:{_src}/usdc->{_dst}/usdc", "cctp", "mainnet", f"{_src}/usdc",
                             f"{_dst}/usdc", "active", f"cctp-v2-{_src}-{_dst}",
                             {**CCTP_MAINNET_METADATA, "sourceChainId": _sid, "destinationChainId": _did,
                              "sourceDomain": _sd, "destinationDomain": _dd}))

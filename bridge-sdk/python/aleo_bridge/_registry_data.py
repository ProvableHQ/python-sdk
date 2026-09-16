"""Verbatim port of veil ``packages/bridge/src/registry/default.ts`` (commit 8d62198, registry version
2026-08-31.solana-deposits.1). Plain dicts only — ``registry.py`` turns them into dataclasses.
Every key keeps veil's camelCase spelling so the brief and veil tests stay the source of truth."""
from __future__ import annotations

REGISTRY_VERSION = "2026-08-31.solana-deposits.1"

EVM_ADDRESS = "^0x[0-9a-fA-F]{40}$"
SOLANA_ADDRESS = "^[1-9A-HJ-NP-Za-km-z]{32,44}$"
ALEO_ADDRESS = "^aleo1[0-9a-z]{58}$"

CHAINS = [
    {"id": "aleo", "displayName": "Aleo", "family": "aleo", "environment": "mainnet", "nativeCurrencySymbol": "ALEO",
     "protocolDomains": {"xreserve": 10002, "hyperlane": 1634493807}},
    {"id": "ethereum", "displayName": "Ethereum", "family": "evm", "environment": "mainnet", "nativeCurrencySymbol": "ETH",
     "protocolDomains": {"xreserve": 0, "hyperlane": 1}},
    {"id": "solana", "displayName": "Solana", "family": "solana", "environment": "mainnet", "nativeCurrencySymbol": "SOL",
     "protocolDomains": {"hyperlane": 1399811149}},
    {"id": "base", "displayName": "Base", "family": "evm", "environment": "mainnet", "nativeCurrencySymbol": "ETH"},
    {"id": "hyperevm", "displayName": "HyperEVM", "family": "evm", "environment": "mainnet", "nativeCurrencySymbol": "HYPE"},
    {"id": "aleo-testnet", "displayName": "Aleo Testnet", "family": "aleo", "environment": "testnet", "nativeCurrencySymbol": "ALEO",
     "protocolDomains": {"xreserve": 10002, "hyperlane": 1617853565}},
    {"id": "sepolia", "displayName": "Ethereum Sepolia", "family": "evm", "environment": "testnet", "nativeCurrencySymbol": "ETH",
     "protocolDomains": {"hyperlane": 11155111}},
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
    {"id": "ethereum/aleo", "key": "aleo", "chainId": "ethereum", "symbol": "ALEO", "name": "Hyperlane ALEO", "decimals": 6, "kind": "token",
     "addressValidationRegex": EVM_ADDRESS},
    {"id": "ethereum/usad", "key": "usad", "chainId": "ethereum", "symbol": "USAD", "name": "USAD route collateral", "decimals": 6, "kind": "token",
     "addressValidationRegex": EVM_ADDRESS},
    {"id": "solana/sol", "key": "sol", "chainId": "solana", "symbol": "SOL", "name": "Solana", "decimals": 9, "kind": "native",
     "locator": {"kind": "native", "value": "SOL"}, "addressValidationRegex": SOLANA_ADDRESS},
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

XRESERVE_SOURCE = "https://developers.circle.com/xreserve/references/supported-blockchains-and-domains"
HYPERLANE_REGISTRY_COMMIT = "2621c16f2db1ccb46643265c110dac5ca2c7c51a"
HYPERLANE_SOURCE = f"https://github.com/hyperlane-xyz/hyperlane-registry/tree/{HYPERLANE_REGISTRY_COMMIT}/deployments/warp_routes"

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
ALEO_WITHDRAWAL_ACTIVATION = {"aleoPlaceholderConfiguration": False, "aleoWithdrawalReviewedAt": "2026-08-26"}

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


def _route(id: str, protocol: str, environment: str, source_asset_id: str, destination_asset_id: str,
           availability: str, deployment_id: str, metadata: dict) -> dict:
    return {
        "id": id, "protocol": protocol, "environment": environment,
        "sourceAssetId": source_asset_id, "destinationAssetId": destination_asset_id,
        "availability": availability, "deploymentId": deployment_id,
        "source": XRESERVE_SOURCE if protocol == "xreserve" else HYPERLANE_SOURCE,
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
    _route("hyperlane:solana/sol->aleo/sol", "hyperlane", "mainnet", "solana/sol", "aleo/sol", "active", "SOL/aleo",
           {**SOLANA_SOL_DEPOSIT_METADATA, **ALEO_MAILBOX_METADATA}),
    _route("hyperlane:aleo/sol->solana/sol", "hyperlane", "mainnet", "aleo/sol", "solana/sol", "active", "SOL/aleo",
           {**_aleo_hyperlane_placeholders("hyp_warp_token_sol_v2.aleo", 1399811149),
            **ALEO_SOL_APP_METADATA, **ALEO_SOL_REMOTE_ROUTER, **ALEO_WITHDRAWAL_ACTIVATION}),
    *_pair("hyperlane", "mainnet", "aleo/aleo", "ethereum/aleo", "metadata-required", "ALEO/aleo", ALEO_MAILBOX_METADATA),
    *_pair("hyperlane", "mainnet", "aleo/aleo", "solana/aleo", "metadata-required", "ALEO/aleo", ALEO_MAILBOX_METADATA),
    *_pair("hyperlane", "mainnet", "aleo/aleo", "base/aleo", "metadata-required", "ALEO/aleo", ALEO_MAILBOX_METADATA),
    *_pair("hyperlane", "mainnet", "aleo/aleo", "hyperevm/aleo", "metadata-required", "ALEO/aleo", ALEO_MAILBOX_METADATA),
    _route("hyperlane:ethereum/usad->aleo/usad", "hyperlane", "mainnet", "ethereum/usad", "aleo/usad", "metadata-required", "USAD/aleo",
           ALEO_MAILBOX_METADATA),
    _route("hyperlane:aleo/usad->ethereum/usad", "hyperlane", "mainnet", "aleo/usad", "ethereum/usad", "metadata-required", "USAD/aleo",
           _aleo_hyperlane_placeholders("hyp_warp_token_usad_v2.aleo", 1)),
]

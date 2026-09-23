import re

import pytest

from aleo_bridge import _registry_data as data
from aleo_bridge.errors import AmbiguousRouteError, ConfigurationError, RouteNotFoundError
from aleo_bridge.registry import (DEFAULT_REGISTRY, Asset, Chain, Locator, Privacy, Registry, Route,
                                  validate_registry)

REG = DEFAULT_REGISTRY
MAILBOX = {
    "aleoMailboxStateVerified": True, "aleoMailboxProgram": "hyp_mailbox.aleo", "aleoMailboxProgramEdition": 0,
    "aleoMailboxLocalDomain": 1634493807, "aleoMailboxObservedNonce": 170, "aleoMailboxObservedProcessCount": 291,
    "aleoMailboxDefaultIsm": "aleo1yvf5kcsdgnescqq2lar83mms79yh3ugvc3y0mdnlgvx4lyh5zugqr9hptk",
    "aleoMailboxDefaultHook": "aleo194tz0jmyq8rd9htvnqppqw4jqerk2p2zd8plzn3sxl06wcgsm5pq9fka74",
    "aleoMailboxRequiredHook": "aleo1yxevh9qgxehej46j7vueplwjcpfdfml2dje3ey4ukzknx7wzasgqnxgq82",
    "aleoMailboxDispatchProxy": "aleo1sge9kmjzs3d8fqrscy4hwn7vf9vw4jcxe877lv0m2w8hay78lsxsqg975s",
    "aleoMailboxOwner": "aleo1ypf8xgvz560ukw25hufj3d77gx69pdcy70nssdfdxd97j80d7cqs98d7x8",
}


def test_shape_and_version():
    assert REG.version == "2026-08-31.solana-deposits.1"
    assert len(REG.chains()) == 7 and len(REG.assets()) == 19
    assert len(REG.routes(include_unavailable=True)) == 22
    assert len(REG.routes()) == 22  # nothing is 'disabled' in this snapshot; metadata-required stays visible
    assert validate_registry(REG) is REG
    assert REG.routes(include_unavailable=True)[0].metadata["xReserveContract"] == "0x8888888199b2Df864bf678259607d6D5EBb4e3Ce"


def test_chains():
    assert [c.id for c in REG.chains()] == ["aleo", "ethereum", "solana", "base", "hyperevm", "aleo-testnet", "sepolia"]
    assert [c.id for c in REG.chains(environment="testnet")] == ["aleo-testnet", "sepolia"]
    aleo = REG.chain("aleo")
    assert (aleo.display_name, aleo.family, aleo.environment, aleo.native_symbol) == ("Aleo", "aleo", "mainnet", "ALEO")
    assert aleo.protocol_domains == {"xreserve": 10002, "hyperlane": 1634493807}
    assert REG.chain("ethereum").protocol_domains == {"xreserve": 0, "hyperlane": 1}
    assert REG.chain("solana").protocol_domains == {"hyperlane": 1399811149}
    assert REG.chain("base").protocol_domains == {} and REG.chain("hyperevm").native_symbol == "HYPE"
    assert REG.chain("aleo-testnet").protocol_domains == {"xreserve": 10002, "hyperlane": 1617853565}
    assert REG.chain("sepolia").protocol_domains == {"hyperlane": 11155111}
    with pytest.raises(RouteNotFoundError):
        REG.chain("bitcoin")


def test_assets_and_lookups():
    wbtc = REG.asset("aleo/wbtc")
    assert wbtc == REG.asset(("aleo", "wbtc")) == REG.asset("ALEO/WBTC")
    assert (wbtc.key, wbtc.chain_id, wbtc.symbol, wbtc.name, wbtc.decimals, wbtc.kind) == \
        ("wbtc", "aleo", "WBTC", "Hyperlane WBTC", 8, "token")
    assert wbtc.locator == Locator("aleo-program", "hyp_warp_token_wbtc_v2.aleo",
                                   "aleo1240fsvz2dhmj0cdtt8mc0yc8um9fmu236rqcl2qnlj9703hd2vpsdwyrtf")
    assert wbtc.privacy == Privacy("arc20", "arc20_wbtc.aleo")
    assert wbtc.address_regex == "^aleo1[0-9a-z]{58}$"
    usdcx = REG.asset("aleo/usdcx")
    assert usdcx.locator == Locator("aleo-program", "usdcx_stablecoin.aleo") and usdcx.privacy == Privacy("arc22", "usdcx_stablecoin.aleo")
    assert REG.asset("aleo/aleo").privacy is None and REG.asset("aleo/usad").locator == Locator("aleo-program", "usad_stablecoin.aleo")
    assert REG.asset("ethereum/usdc").locator == Locator("evm-contract", "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")
    assert REG.asset("ethereum/eth").locator == Locator("native", "ETH") and REG.asset("ethereum/eth").kind == "native"
    assert REG.asset("ethereum/wbtc").locator.value == "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"
    assert REG.asset("ethereum/usdt").locator.value == "0xdAC17F958D2ee523a2206206994597C13D831ec7"
    assert REG.asset("sepolia/usdc").locator.value == "0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238"
    assert REG.asset("aleo-testnet/usdcx").privacy == Privacy("arc22", "test_usdcx_stablecoin.aleo")
    for no_locator in ("ethereum/aleo", "ethereum/usad", "solana/aleo", "base/aleo", "hyperevm/aleo"):
        assert REG.asset(no_locator).locator is None
    assert REG.asset("solana/sol").address_regex == "^[1-9A-HJ-NP-Za-km-z]{32,44}$"
    assert REG.asset("ethereum/usdc").matches_address("0x0000000000000000000000000000000000000001")
    assert not REG.asset("aleo/usdcx").matches_address("0xabc")
    # re.fullmatch, not re.search: a trailing newline must not sneak past the "$" anchor
    assert not REG.asset("ethereum/usdc").matches_address("0x0000000000000000000000000000000000000001\n")
    assert [a.id for a in REG.assets(chain="aleo")] == ["aleo/aleo", "aleo/usdcx", "aleo/eth", "aleo/wbtc", "aleo/usdt", "aleo/sol", "aleo/usad"]
    assert [a.id for a in REG.assets(symbol="aleo")] == ["aleo/aleo", "ethereum/aleo", "solana/aleo", "base/aleo", "hyperevm/aleo"]
    assert [a.id for a in REG.assets(environment="testnet")] == ["aleo-testnet/usdcx", "sepolia/usdc"]
    for bad in ("aleo/doge", "doge", ("aleo", "doge"), "a/b/c"):
        with pytest.raises(RouteNotFoundError):
            REG.asset(bad)


def test_usdcx_only_via_xreserve_and_others_via_hyperlane():
    usdcx = [r for r in REG.routes(include_unavailable=True) if "usdcx" in r.source_asset_id or "usdcx" in r.destination_asset_id]
    assert usdcx and all(r.protocol == "xreserve" for r in usdcx)
    for symbol in ("ETH", "WBTC", "USDT", "SOL", "ALEO", "USAD"):
        routes = REG.routes(symbol=symbol, include_unavailable=True)
        assert routes and all(r.protocol == "hyperlane" for r in routes), symbol


def test_xreserve_routes():
    xr = REG.routes(bridge_protocol="xreserve", include_unavailable=True)
    assert [r.id for r in xr] == ["xreserve:ethereum/usdc->aleo/usdcx", "xreserve:aleo/usdcx->ethereum/usdc",
                                  "xreserve:sepolia/usdc->aleo-testnet/usdcx", "xreserve:aleo-testnet/usdcx->sepolia/usdc"]
    assert all(r.availability == "active" and r.active for r in xr)
    assert all(r.metadata["ethereumDestinationDomain"] == 0 and r.metadata["arcDestinationDomain"] == 26 for r in xr)
    assert all(r.source == "https://developers.circle.com/xreserve/references/supported-blockchains-and-domains" for r in xr)
    main = REG.route("xreserve:aleo/usdcx->ethereum/usdc").metadata
    assert main == {
        "xReserveContract": "0x8888888199b2Df864bf678259607d6D5EBb4e3Ce", "sourceChainId": 1, "sourceDomain": 0,
        "ethereumDestinationDomain": 0, "arcDestinationDomain": 26, "remoteDomain": 10002,
        "remoteToken": "usdcx_stablecoin.aleo",
        "remoteTokenBytes32": "0x11ea7dab1d29d5f61500582c63e98c42e1165f9ba050ea9d0c6af9f871987711",
        "minimumAmountAtomic": "2000000", "withdrawalFeeAtomic": "2000000", "maxFeeAtomic": "100000",
        "bridgeProgram": "usdcx_bridge_v2.aleo", "wrapperProgram": "shielded_usdcx_wrapper.aleo",
        "attestationBaseUrl": "https://xreserve-api.circle.com/v1/attestations",
    }
    test = REG.route("xreserve:sepolia/usdc->aleo-testnet/usdcx").metadata
    assert test["xReserveContract"] == "0x008888878f94C0d87defdf0B07f46B93C1934442" and test["sourceChainId"] == 11155111
    assert test["remoteToken"] == "test_usdcx_stablecoin.aleo" and test["bridgeProgram"] == "test_usdcx_bridge_v2.aleo"
    assert test["remoteTokenBytes32"] == "0xb143ed52c774cd1d4a519d0e796f15916be5a9e1d45edcd9852dd23f68f53401"
    assert test["attestationBaseUrl"] == "https://xreserve-api-testnet.circle.com/v1/attestations"
    assert REG.route("xreserve:ethereum/usdc->aleo/usdcx").deployment_id == "xreserve-usdcx-aleo"
    assert REG.route("xreserve:sepolia/usdc->aleo-testnet/usdcx").deployment_id == "xreserve-usdcx-aleo-testnet"


def test_inbound_ethereum_hyperlane_routes():
    inbound = [REG.route(i) for i in ("hyperlane:ethereum/eth->aleo/eth", "hyperlane:ethereum/wbtc->aleo/wbtc", "hyperlane:ethereum/usdt->aleo/usdt")]
    assert all(r.active for r in inbound)
    for r in inbound:
        m = r.metadata
        assert m["registryCommit"] == "2621c16f2db1ccb46643265c110dac5ca2c7c51a"
        assert m["sourceChainId"] == 1 and m["destinationDomain"] == 1634493807
        assert m["mailboxAddress"] == "0xc005dc82818d67AF737725bD4bf75435d065D239"
        assert m["interchainGasPaymaster"] == "0x9e6B1022bE9BBF5aFd152483DAD9b88911bC8611"
        assert m["interchainSecurityModule"] == "0x0000000000000000000000000000000000000000"
        for k, v in MAILBOX.items():
            assert m[k] == v
        assert r.source == "https://github.com/hyperlane-xyz/hyperlane-registry/tree/2621c16f2db1ccb46643265c110dac5ca2c7c51a/deployments/warp_routes"
    eth, wbtc, usdt = (r.metadata for r in inbound)
    assert (eth["routerAddress"], eth["routerType"]) == ("0x38D447694f5c1f773ae3132cf93bF30B7Ec1Fa5A", "native")
    assert eth["destinationRouter"] == "hyp_warp_token_eth_v2.aleo/aleo1t7f29tq9qng2lfvrkpcuvu59jn24hrmzqdyqfn6p0u5p80npfvqqecmkj8"
    assert (wbtc["routerAddress"], wbtc["routerType"], wbtc["tokenAddress"]) == \
        ("0x20CDC85778b732073F7EecEF3DF25c0d310f8772", "collateral", "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599")
    assert (usdt["routerAddress"], usdt["tokenAddress"], usdt["requiresApprovalReset"]) == \
        ("0x3C2064D78e4578E8F936E3db42aEF044E33FBF31", "0xdAC17F958D2ee523a2206206994597C13D831ec7", True)
    assert "requiresApprovalReset" not in wbtc


def test_aleo_origin_withdrawals_are_active_and_pinned():
    ids = ["hyperlane:aleo/eth->ethereum/eth", "hyperlane:aleo/wbtc->ethereum/wbtc",
           "hyperlane:aleo/usdt->ethereum/usdt", "hyperlane:aleo/sol->solana/sol"]
    routes = [REG.route(i) for i in ids]
    assert all(r.active for r in routes)
    assert all(r.metadata["aleoPlaceholderConfiguration"] is False for r in routes)
    assert all(r.metadata["aleoWithdrawalReviewedAt"] == "2026-08-26" for r in routes)
    assert all(r.metadata["aleoHookManagerProgram"] == "hyp_hook_manager.aleo" for r in routes)
    assert [r.metadata["aleoRouterProgram"] for r in routes] == [
        "hyp_warp_token_eth_v2.aleo", "hyp_warp_token_wbtc_v2.aleo", "hyp_warp_token_usdt_v2.aleo", "hyp_warp_token_sol_v2.aleo"]
    for r in routes:
        m = r.metadata
        assert m["aleoAppMetadataVerified"] is True and m["aleoRemoteRouterVerified"] is True
        assert m["aleoAllowanceSpendersVerified"] is True and m["aleoUnusedAllowancesVerified"] is True
        assert m["aleoTokenType"] == "1"
        assert m["aleoIsm"] == m["aleoHook"] == "aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc"
        assert m["aleoAllowanceSpender0"] == "aleo194tz0jmyq8rd9htvnqppqw4jqerk2p2zd8plzn3sxl06wcgsm5pq9fka74"
        assert m["aleoAllowanceSpender1"] == m["aleoAllowanceSpender2"] == m["aleoAllowanceSpender3"] == \
            "aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc"
        assert m["aleoAllowanceAmount1"] == m["aleoAllowanceAmount2"] == m["aleoAllowanceAmount3"] == "0"
        # slot 0's pinned "0" is a placeholder; the live IGP quote is fetched at execution time (out of
        # scope for the registry), matching veil's default.ts/dist behavior (placeholderFields mechanism).
        assert m["aleoAllowanceAmount0"] == "0"
        for k, v in MAILBOX.items():
            assert m[k] == v


def test_eth_wbtc_usdt_sol_metadata_literals():
    eth = REG.route("hyperlane:aleo/eth->ethereum/eth").metadata
    assert eth["aleoTokenOwner"] == "aleo1wq6f6qdqya44avznygz5hae40u3mjg64w0r93a4qfu4utpf8cg9q566f4r"
    assert eth["aleoTokenId"] == "133188123661477349522757068766864658505569365361420630212878794317749195359field"
    assert (eth["aleoLocalDecimals"], eth["aleoRemoteDecimals"], eth["aleoProgramEdition"], eth["aleoDestinationDomain"]) == (18, 18, 0, 1)
    assert eth["aleoRemoteRouterEvmAddress"] == "0x38D447694f5c1f773ae3132cf93bF30B7Ec1Fa5A" and eth["aleoRemoteRouterGas"] == "44000"
    assert eth["aleoRemoteRouterRecipient"] == "[0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 56u8, 212u8, 71u8, 105u8, 79u8, 92u8, 31u8, 119u8, 58u8, 227u8, 19u8, 44u8, 249u8, 59u8, 243u8, 11u8, 126u8, 193u8, 250u8, 90u8]"
    assert eth["aleoSampleTransferSource"] == "https://explorer.provable.com/transaction/at1vu0yckkms887zkl3qz7plnncd56jtf5zeal4uj2808upsjkusy8q7yp9v8"
    assert eth["routerAddress"] == "0x38D447694f5c1f773ae3132cf93bF30B7Ec1Fa5A"  # ETH_HYPERLANE_METADATA is merged into the reverse route too

    wbtc = REG.route("hyperlane:aleo/wbtc->ethereum/wbtc").metadata
    assert wbtc["aleoTokenOwner"] == "aleo14jauje2a5sncm9u5t3mt6qqv3eq2hatkddskccs0dvsy35a0x58q0d6f95"
    assert wbtc["aleoTokenId"] == "1505227928464760254508513036497943623956572091841806589002910775534260084309field"
    assert (wbtc["aleoLocalDecimals"], wbtc["aleoRemoteDecimals"], wbtc["aleoProgramEdition"]) == (8, 8, 0)
    assert wbtc["aleoRemoteRouterRecipient"] == "[0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 32u8, 205u8, 200u8, 87u8, 120u8, 183u8, 50u8, 7u8, 63u8, 126u8, 236u8, 239u8, 61u8, 242u8, 92u8, 13u8, 49u8, 15u8, 135u8, 114u8]"
    assert wbtc["aleoRemoteRouterGas"] == "68000"
    assert wbtc["aleoProgramSource"] == "https://explorer.provable.com/program/hyp_warp_token_wbtc_v2.aleo"
    assert wbtc["aleoAppMetadataSource"] == "https://api.explorer.provable.com/v2/mainnet/program/hyp_warp_token_wbtc_v2.aleo/mapping/app_metadata/true"
    assert wbtc["aleoRemoteRouterSource"] == "https://api.explorer.provable.com/v2/mainnet/program/hyp_warp_token_wbtc_v2.aleo/mapping/remote_routers/1u32"
    assert wbtc["aleoAppMetadataReviewedAt"] == wbtc["aleoRemoteRouterReviewedAt"] == "2026-08-17"

    usdt = REG.route("hyperlane:aleo/usdt->ethereum/usdt").metadata
    assert usdt["aleoTokenOwner"] == "aleo1l3gwacmjruxryy9c7c4fn0acyzprf29hucrvthw7f63lpyhd5y9srydq8z"
    assert usdt["aleoTokenId"] == "8295938150000417034830036849466229528602563851235385582732969109393809606969field"
    assert (usdt["aleoLocalDecimals"], usdt["aleoRemoteDecimals"], usdt["aleoProgramEdition"], usdt["aleoScale"]) == (6, 18, 1, "1000000000000")
    assert usdt["aleoRemoteRouterRecipient"] == "[0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 60u8, 32u8, 100u8, 215u8, 142u8, 69u8, 120u8, 232u8, 249u8, 54u8, 227u8, 219u8, 66u8, 174u8, 240u8, 68u8, 227u8, 63u8, 191u8, 49u8]"
    assert usdt["aleoRemoteRouterGas"] == "68000" and usdt["aleoSampleTransferDestinationDomain"] == 56
    assert usdt["aleoSampleTransferSource"] == "https://explorer.provable.com/transaction/at19caeeee8v3xc4kfwen4tx89f0tnggrpjp0anrhq2ca3y82xr9q8qyz8a9r"
    assert usdt["aleoHyperlaneConfigSource"] == "https://github.com/hyperlane-xyz/hyperlane-registry/blob/418056e21734d26a7d14692e0ec5e902cc9e86bf/deployments/warp_routes/USDT/aleo-config.yaml"

    sol = REG.route("hyperlane:aleo/sol->solana/sol").metadata
    assert sol["aleoTokenOwner"] == "aleo1wr8rfr4ggedjxtg5e23s38zqkgy2j05uc9l8t4akjp5zcw3levpswkwk45"
    assert sol["aleoTokenId"] == "6148061383892805373029428966764338809222769879628268522058032128225601478383field"
    assert (sol["aleoLocalDecimals"], sol["aleoRemoteDecimals"], sol["aleoDestinationDomain"]) == (9, 9, 1399811149)
    assert sol["aleoRemoteRouterSolanaAddress"] == "8YGT2pZwyZe94qBpGzWfY2TMEVcwaQ1bXAE7YAgpUaM7"
    assert sol["aleoRemoteRouterRecipient"] == "[112u8, 4u8, 72u8, 22u8, 219u8, 143u8, 68u8, 202u8, 21u8, 197u8, 236u8, 182u8, 198u8, 142u8, 52u8, 96u8, 142u8, 38u8, 51u8, 113u8, 116u8, 143u8, 96u8, 123u8, 104u8, 126u8, 97u8, 73u8, 7u8, 6u8, 211u8, 122u8]"
    assert sol["aleoRemoteRouterGas"] == "300000"
    assert sol["aleoSampleTransitionId"] == "au15fg39h53h55tkj0nexrme3k6pvgxngxapcyajdhf06jcg3cyeugq5kd7hg"
    assert "routerAddress" not in sol  # no Ethereum common block on the SOL withdrawal


def test_solana_deposit_route_metadata():
    r = REG.route("hyperlane:solana/sol->aleo/sol")
    assert r.active and r.deployment_id == "SOL/aleo"
    expected = {
        "warpProgramAddress": "8YGT2pZwyZe94qBpGzWfY2TMEVcwaQ1bXAE7YAgpUaM7", "tokenPda": "JDkpV5CsSbhyGhHhirC5DjGPTcuKWUVHtBZ5MFsgu3ZW",
        "nativeCollateralPda": "8HY3hxmnrWwqEmcdwkSnfN9wEQFUkyiwZvU1vMbnXgbC", "dispatchAuthorityPda": "ATDttjggAZKyS19kcV6Rn56oMi49gDprZGckRou9vkkY",
        "mailboxProgramAddress": "E588QtVUvresuXq2KoNEwAmoifCzYGpRBdHByN9KQMbi", "mailboxOutboxPda": "BvZpTuYLAR77mPhH4GtvwEWUTs53GQqkgBNuXpCePVNk",
        "igpProgramAddress": "BhNcatUDC2D5JTyeaqrdSukiVFsEHK7e3hVmKMztwefv", "igpProgramDataPda": "8Cv4PHJ6Cf3xY7dse7wYeZKtuQv9SAN6ujt5w22a2uho",
        "igpAccount": "JAvHW21tYXE9dtdG83DReqU2b4LUexFuCbtJT5tF8X6M", "igpOverheadAccount": "AkeHBbE5JkwVppujCQQ6WuxsVsJtruBAjUo6fDCFp6fF",
        "splNoopProgramAddress": "noopb9bkMVfRPU8AsbpTUg8AQkHtKwMYZiFUjNRtMmV", "destinationDomain": 1634493807,
        "destinationGasAmount": "464000", "registryCommit": "418056e21734d26a7d14692e0ec5e902cc9e86bf", "solanaReviewedAt": "2026-08-31",
        "solanaConfigSource": "https://github.com/hyperlane-xyz/hyperlane-registry/blob/418056e21734d26a7d14692e0ec5e902cc9e86bf/deployments/warp_routes/SOL/aleo-config.yaml",
    }
    for k, v in {**expected, **MAILBOX}.items():
        assert r.metadata[k] == v, k


def test_metadata_required_routes():
    pairs = [("aleo/aleo", "ethereum/aleo"), ("aleo/aleo", "solana/aleo"), ("aleo/aleo", "base/aleo"), ("aleo/aleo", "hyperevm/aleo")]
    for left, right in pairs:
        for rid in (f"hyperlane:{left}->{right}", f"hyperlane:{right}->{left}"):
            r = REG.route(rid)
            assert r.availability == "metadata-required" and not r.active and r.deployment_id == "ALEO/aleo"
            assert dict(r.metadata) == MAILBOX | {
                "aleoHookManagerProgram": "hyp_hook_manager.aleo",
                "aleoHookManagerProgramSource": "https://explorer.provable.com/program/hyp_hook_manager.aleo",
                "aleoMailboxProgramSource": "https://explorer.provable.com/program/hyp_mailbox.aleo",
                "aleoMailboxMetadataSource": "https://api.explorer.provable.com/v2/mainnet/program/hyp_mailbox.aleo/mapping/mailbox/true",
                "aleoMailboxMetadataReviewedAt": "2026-08-17",
            }
    usad_in = REG.route("hyperlane:ethereum/usad->aleo/usad")
    assert usad_in.availability == "metadata-required" and usad_in.deployment_id == "USAD/aleo"
    usad_out = REG.route("hyperlane:aleo/usad->ethereum/usad")
    m = usad_out.metadata
    assert usad_out.availability == "metadata-required" and m["aleoPlaceholderConfiguration"] is True
    assert m["aleoRouterProgram"] == "hyp_warp_token_usad_v2.aleo" and m["aleoDestinationDomain"] == 1
    assert m["aleoTokenType"] == "0" and m["aleoTokenId"] == "0field" and m["aleoRemoteRouterGas"] == "0"
    assert m["aleoTokenOwner"] == m["aleoIsm"] == m["aleoHook"] == "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"
    assert m["aleoRemoteRouterRecipient"] == "[" + ", ".join(["0u8"] * 32) + "]" and m["aleoRecipient"] == "[0u128, 0u128]"
    assert all(m[f"aleoAllowanceAmount{i}"] == "0" for i in range(4))


def test_route_filters_follow_veil_get_routes():
    # veil getRoutes: protocol / sourceChainId / destinationChainId / symbol — chain ids, never asset refs.
    assert [r.id for r in REG.routes(source_chain="aleo", bridge_protocol="xreserve")] == ["xreserve:aleo/usdcx->ethereum/usdc"]
    assert [r.id for r in REG.routes(destination_chain="aleo", symbol="wbtc")] == ["hyperlane:ethereum/wbtc->aleo/wbtc"]
    assert len(REG.routes(environment="testnet")) == 2 and len(REG.routes(environment="mainnet")) == 20
    assert len(REG.routes(source_chain="solana")) == 2  # SOL deposit + metadata-required ALEO
    assert len(REG.routes(source_chain="SOLANA", destination_chain="Aleo")) == 2   # case-insensitive
    # the asset filters narrow a chain pair to one asset on either side
    assert [r.id for r in REG.routes(source_chain="aleo", source_asset="wbtc")] == ["hyperlane:aleo/wbtc->ethereum/wbtc"]
    assert [r.id for r in REG.routes(destination_chain="ethereum", destination_asset="usdc")] == ["xreserve:aleo/usdcx->ethereum/usdc"]
    with pytest.raises(TypeError):
        REG.routes("aleo")                                    # keyword-only: no positional selectors


def test_find_route_resolves_by_chain_and_asset_keywords():
    assert REG.find_route(source_chain="aleo", source_asset="wbtc", destination_chain="ethereum",
                          destination_asset="wbtc").id == "hyperlane:aleo/wbtc->ethereum/wbtc"
    # destination_asset and protocol are optional when the remaining filters leave one route
    assert REG.find_route(source_chain="ethereum", source_asset="usdc",
                          destination_chain="aleo").id == "xreserve:ethereum/usdc->aleo/usdcx"
    assert REG.find_route(source_chain="Ethereum", source_asset="USDC", destination_chain="ALEO",
                          bridge_protocol="xreserve").id == "xreserve:ethereum/usdc->aleo/usdcx"
    assert REG.find_route(source_chain="aleo", source_asset="usad",
                          destination_chain="ethereum").availability == "metadata-required"  # visible, refused later
    with pytest.raises(RouteNotFoundError):
        REG.find_route(source_chain="aleo", source_asset="wbtc", destination_chain="solana")
    with pytest.raises(RouteNotFoundError):
        REG.find_route(source_chain="aleo", source_asset="wbtc", destination_chain="ethereum", bridge_protocol="xreserve")
    with pytest.raises(RouteNotFoundError):
        REG.find_route(source_chain="aleo", source_asset="doge", destination_chain="ethereum")
    with pytest.raises(RouteNotFoundError):
        REG.route("hyperlane:aleo/doge->ethereum/doge")
    # A synthetic duplicate pair across protocols is ambiguous without bridge_protocol=, and the error says so
    dup = Route("xreserve:aleo/wbtc->ethereum/wbtc", "xreserve", "mainnet", "aleo/wbtc", "ethereum/wbtc", "active", None, None, {})
    reg2 = Registry(REG.version, REG.chains(), REG.assets(), [*REG.routes(include_unavailable=True), dup])
    with pytest.raises(AmbiguousRouteError, match="bridge_protocol="):
        reg2.find_route(source_chain="aleo", source_asset="wbtc", destination_chain="ethereum")
    assert reg2.find_route(source_chain="aleo", source_asset="wbtc", destination_chain="ethereum",
                           bridge_protocol="hyperlane").protocol == "hyperlane"
    # Two destination assets for one source asset are ambiguous without destination_asset=
    fork = Route("hyperlane:aleo/wbtc->ethereum/usdt", "hyperlane", "mainnet", "aleo/wbtc", "ethereum/usdt", "active", None, None, {})
    reg4 = Registry(REG.version, REG.chains(), REG.assets(), [*REG.routes(include_unavailable=True), fork])
    with pytest.raises(AmbiguousRouteError, match="destination_asset="):
        reg4.find_route(source_chain="aleo", source_asset="wbtc", destination_chain="ethereum")
    assert reg4.find_route(source_chain="aleo", source_asset="wbtc", destination_chain="ethereum",
                           destination_asset="usdt") is fork
    # disabled routes are hidden from routes() and find_route() unless include_unavailable
    off = Route("hyperlane:aleo/eth->ethereum/eth", "hyperlane", "mainnet", "aleo/eth", "ethereum/eth", "disabled", None, None, {})
    reg3 = Registry(REG.version, REG.chains(), REG.assets(), [off])
    assert reg3.routes() == [] and reg3.routes(include_unavailable=True) == [off]
    with pytest.raises(RouteNotFoundError):
        reg3.find_route(source_chain="aleo", source_asset="eth", destination_chain="ethereum")


def test_route_meta_helpers():
    r = REG.route("hyperlane:aleo/eth->ethereum/eth")
    assert r.meta_str("aleoRouterProgram") == "hyp_warp_token_eth_v2.aleo"
    assert r.meta_int("aleoDestinationDomain") == 1 and r.meta_int("missing", 7) == 7
    with pytest.raises(ConfigurationError, match="aleoNope is missing"):
        r.meta_str("aleoNope")
    with pytest.raises(ConfigurationError, match="invalid"):
        r.meta_int("aleoMailboxStateVerified")  # bool is not an int here


def _with(**overrides) -> Registry:
    base = dict(version=REG.version, chains=REG.chains(), assets=REG.assets(), routes=REG.routes(include_unavailable=True))
    base.update(overrides)
    return Registry(base["version"], base["chains"], base["assets"], base["routes"])


def test_validation_failures():
    from dataclasses import replace
    r0 = REG.routes(include_unavailable=True)[0]
    with pytest.raises(ConfigurationError, match="unknown source asset missing/asset"):
        validate_registry(_with(routes=[replace(r0, source_asset_id="missing/asset")]))
    with pytest.raises(ConfigurationError, match="Duplicate bridge route id"):
        validate_registry(_with(routes=[r0, r0]))
    a0 = REG.assets()[0]
    with pytest.raises(ConfigurationError, match="Duplicate bridge asset key"):
        validate_registry(_with(assets=[a0, replace(a0, id=a0.id + "-duplicate")], routes=[]))
    with pytest.raises(ConfigurationError, match="invalid address validation regex"):
        validate_registry(_with(assets=[replace(a0, address_regex="[")], routes=[]))
    usdc = REG.asset("ethereum/usdc")
    with pytest.raises(ConfigurationError, match="privacy capability on a non-Aleo chain"):
        validate_registry(_with(assets=[replace(a, privacy=Privacy("arc20", "arc20_usdc.aleo")) if a.id == usdc.id else a for a in REG.assets()]))
    with pytest.raises(ConfigurationError, match="empty privacy program"):
        validate_registry(_with(assets=[replace(a, privacy=Privacy("arc20", "")) if a.id == "aleo/sol" else a for a in REG.assets()]))
    with pytest.raises(ConfigurationError, match="crosses registry environments"):
        validate_registry(_with(routes=[replace(r0, environment="testnet")]))
    with pytest.raises(ConfigurationError, match="missing required Solana Hyperlane metadata"):
        sol = REG.route("hyperlane:solana/sol->aleo/sol")
        incomplete = {k: v for k, v in sol.metadata.items() if k != "igpAccount"}
        validate_registry(_with(routes=[replace(r, metadata=incomplete) if r.id == sol.id else r for r in REG.routes(include_unavailable=True)]))
    with pytest.raises(ConfigurationError, match="Duplicate bridge chain id"):
        validate_registry(_with(chains=[*REG.chains(), REG.chain("aleo")]))
    with pytest.raises(ConfigurationError, match="version must not be empty"):
        validate_registry(_with(version="  "))


def test_data_module_is_plain_literals():
    assert data.REGISTRY_VERSION == REG.version
    assert len(data.CHAINS) == 7 and len(data.ASSETS) == 19 and len(data.ROUTES) == 22
    assert all(isinstance(c, dict) for c in data.CHAINS) and all(isinstance(r["metadata"], dict) for r in data.ROUTES)
    assert re.compile(data.ALEO_ADDRESS).match("aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n")

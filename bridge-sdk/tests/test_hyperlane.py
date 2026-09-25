import pytest

from aleo_bridge import hyperlane as hl
from aleo_bridge.errors import (AmbiguousRouteError, ConfigurationError, InvalidAmountError, InvalidRecipientError,
                                RouteUnavailableError, UnsupportedRouteError)
from aleo_bridge.registry import DEFAULT_REGISTRY as REG
from aleo_bridge.types import DispatchReceipt, Status
from tests.conftest import ETH_GAS_CONFIG, IGP_KEY_ETH

EVM1 = "0x0000000000000000000000000000000000000001"
SOL_SYSTEM = "11111111111111111111111111111111"
ZERO = "aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc"
MAILBOX_STATE = ("{ default_hook: aleo194tz0jmyq8rd9htvnqppqw4jqerk2p2zd8plzn3sxl06wcgsm5pq9fka74, "
                 "required_hook: aleo1yxevh9qgxehej46j7vueplwjcpfdfml2dje3ey4ukzknx7wzasgqnxgq82 }")
ROUTES = [  # veil test/actions/aleoHyperlane.test.ts ROUTES (active four); amount "1" in source decimals
    ("aleo/eth", "hyperlane:aleo/eth->ethereum/eth", "hyp_warp_token_eth_v2.aleo", EVM1, 10**18),
    ("aleo/wbtc", "hyperlane:aleo/wbtc->ethereum/wbtc", "hyp_warp_token_wbtc_v2.aleo", EVM1, 10**8),
    ("aleo/usdt", "hyperlane:aleo/usdt->ethereum/usdt", "hyp_warp_token_usdt_v2.aleo", EVM1, 10**6),
    ("aleo/sol", "hyperlane:aleo/sol->solana/sol", "hyp_warp_token_sol_v2.aleo", SOL_SYSTEM, 10**9),
]


def _inputs(bridge, route_id, recipient, amount, gas=8174147):
    route = REG.route(route_id)
    src, dst = REG.asset(route.source_asset_id), REG.asset(route.destination_asset_id)
    return bridge.hyperlane.build_transfer_remote_inputs(
        route, recipient=recipient, amount_atomic=amount, gas_payment_microcredits=gas,
        decimals=(route.metadata["aleoLocalDecimals"], route.metadata["aleoRemoteDecimals"]))


def test_pure_helpers():
    assert hl.parse_gas_config(ETH_GAS_CONFIG) == {"gas_overhead": 159337, "exchange_rate": 402, "gas_price": 1000000000}
    with pytest.raises(ConfigurationError, match="malformed"):
        hl.parse_gas_config("{ gas_overhead: 1u128 }")
    assert hl.gas_config_key(REG.route("hyperlane:aleo/eth->ethereum/eth")) == IGP_KEY_ETH
    assert hl.compute_gas_payment(gas_limit=44000, gas_overhead=159337, gas_price=1000000000, exchange_rate=402) == 8174147
    with pytest.raises(ConfigurationError, match="positive u64"):
        hl.compute_gas_payment(gas_limit=0, gas_overhead=0, gas_price=1, exchange_rate=1)
    with pytest.raises(ConfigurationError, match="positive u64"):
        hl.compute_gas_payment(gas_limit=1, gas_overhead=0, gas_price=2**128 - 1, exchange_rate=2**128 - 1)


@pytest.mark.parametrize("asset,route_id,program,recipient,amount", ROUTES)
def test_outbound_route_and_common_shape(bridge, asset, route_id, program, recipient, amount):
    route = bridge.hyperlane.outbound_route(asset)
    assert route.id == route_id and route.meta_str("aleoRouterProgram") == program
    inputs = _inputs(bridge, route_id, recipient, amount, gas=1)
    assert len(inputs) == 7
    assert inputs[1] == MAILBOX_STATE
    assert inputs[6].count("spender:") == 4 and inputs[6].count("amount: 0u64") == 3 and "amount: 1u64" in inputs[6]
    assert inputs[6].startswith("[{ spender: aleo194tz0jmyq8rd9htvnqppqw4jqerk2p2zd8plzn3sxl06wcgsm5pq9fka74, amount: 1u64 }, { spender: " + ZERO)
    assert inputs[5] == f"{amount}u128"


def test_eth_inputs_match_veil_vector(bridge):
    inputs = _inputs(bridge, "hyperlane:aleo/eth->ethereum/eth", EVM1, 10**18)
    assert inputs[0] == ("{ token_type: 1u8, token_owner: aleo1wq6f6qdqya44avznygz5hae40u3mjg64w0r93a4qfu4utpf8cg9q566f4r, "
                         "ism: aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc, "
                         "hook: aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc, "
                         "token_id: 133188123661477349522757068766864658505569365361420630212878794317749195359field, "
                         "local_decimals: 18u8, remote_decimals: 18u8 }")
    assert inputs[2] == ("{ domain: 1u32, recipient: [0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, "
                         "56u8, 212u8, 71u8, 105u8, 79u8, 92u8, 31u8, 119u8, 58u8, 227u8, 19u8, 44u8, 249u8, 59u8, "
                         "243u8, 11u8, 126u8, 193u8, 250u8, 90u8], gas: 44000u128 }")
    assert inputs[3] == "1u32"
    assert inputs[4] == "[0u128, 1329227995784915872903807060280344576u128]"
    assert inputs[5] == "1000000000000000000u128"
    assert inputs[6] == ("[{ spender: aleo194tz0jmyq8rd9htvnqppqw4jqerk2p2zd8plzn3sxl06wcgsm5pq9fka74, amount: 8174147u64 }, "
                         f"{{ spender: {ZERO}, amount: 0u64 }}, {{ spender: {ZERO}, amount: 0u64 }}, {{ spender: {ZERO}, amount: 0u64 }}]")


def test_wbtc_inputs_match_veil_vector(bridge):
    inputs = _inputs(bridge, "hyperlane:aleo/wbtc->ethereum/wbtc", EVM1, 10**8)
    assert inputs[0] == ("{ token_type: 1u8, token_owner: aleo14jauje2a5sncm9u5t3mt6qqv3eq2hatkddskccs0dvsy35a0x58q0d6f95, "
                         "ism: aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc, "
                         "hook: aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc, "
                         "token_id: 1505227928464760254508513036497943623956572091841806589002910775534260084309field, "
                         "local_decimals: 8u8, remote_decimals: 8u8 }")
    assert inputs[2] == ("{ domain: 1u32, recipient: [0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, "
                         "32u8, 205u8, 200u8, 87u8, 120u8, 183u8, 50u8, 7u8, 63u8, 126u8, 236u8, 239u8, 61u8, 242u8, "
                         "92u8, 13u8, 49u8, 15u8, 135u8, 114u8], gas: 68000u128 }")
    assert inputs[5] == "100000000u128"


def test_usdt_inputs_match_veil_vector(bridge):
    inputs = _inputs(bridge, "hyperlane:aleo/usdt->ethereum/usdt", EVM1, 10**6)
    assert inputs[0] == ("{ token_type: 1u8, token_owner: aleo1l3gwacmjruxryy9c7c4fn0acyzprf29hucrvthw7f63lpyhd5y9srydq8z, "
                         "ism: aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc, "
                         "hook: aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc, "
                         "token_id: 8295938150000417034830036849466229528602563851235385582732969109393809606969field, "
                         "local_decimals: 6u8, remote_decimals: 18u8 }")
    assert inputs[2] == ("{ domain: 1u32, recipient: [0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, "
                         "60u8, 32u8, 100u8, 215u8, 142u8, 69u8, 120u8, 232u8, 249u8, 54u8, 227u8, 219u8, 66u8, 174u8, "
                         "240u8, 68u8, 227u8, 63u8, 191u8, 49u8], gas: 68000u128 }")
    assert inputs[5] == "1000000u128"


def test_sol_inputs_match_veil_vector(bridge):
    inputs = _inputs(bridge, "hyperlane:aleo/sol->solana/sol", SOL_SYSTEM, 10**9)
    assert inputs[0] == ("{ token_type: 1u8, token_owner: aleo1wr8rfr4ggedjxtg5e23s38zqkgy2j05uc9l8t4akjp5zcw3levpswkwk45, "
                         "ism: aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc, "
                         "hook: aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc, "
                         "token_id: 6148061383892805373029428966764338809222769879628268522058032128225601478383field, "
                         "local_decimals: 9u8, remote_decimals: 9u8 }")
    assert inputs[2] == ("{ domain: 1399811149u32, recipient: [112u8, 4u8, 72u8, 22u8, 219u8, 143u8, 68u8, 202u8, "
                         "21u8, 197u8, 236u8, 182u8, 198u8, 142u8, 52u8, 96u8, 142u8, 38u8, 51u8, 113u8, 116u8, "
                         "143u8, 96u8, 123u8, 104u8, 126u8, 97u8, 73u8, 7u8, 6u8, 211u8, 122u8], gas: 300000u128 }")
    assert inputs[3] == "1399811149u32" and inputs[4] == "[0u128, 0u128]" and inputs[5] == "1000000000u128"
    real = _inputs(bridge, "hyperlane:aleo/sol->solana/sol", "8YGT2pZwyZe94qBpGzWfY2TMEVcwaQ1bXAE7YAgpUaM7", 1)
    assert real[4] == "[127878782877948140186055645953777992816u128, 163261512394675613100746600600636171918u128]"


def test_placeholder_usad_route_is_inspectable_but_not_executable(bridge):
    usad = REG.route("hyperlane:aleo/usad->ethereum/usad")
    inputs = bridge.hyperlane.build_transfer_remote_inputs(usad, recipient=EVM1, amount_atomic=1_000_000,
                                                           gas_payment_microcredits=1, decimals=(6, 6))
    assert inputs[3] == "1u32" and inputs[4] == "[0u128, 1329227995784915872903807060280344576u128]" and inputs[5] == "1000000u128"
    assert "gas: 0u128" in inputs[2] and inputs[0].startswith("{ token_type: 0u8, token_owner: aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n")
    with pytest.raises(RouteUnavailableError, match="not executable"):
        bridge.hyperlane.outbound_route("aleo/usad")
    with pytest.raises(RouteUnavailableError):
        bridge.hyperlane.transfer_remote("aleo/usad", EVM1, amount="1")
    with pytest.raises(RouteUnavailableError):      # four metadata-required ALEO routes, none active
        bridge.hyperlane.outbound_route("aleo/aleo")
    with pytest.raises(UnsupportedRouteError, match="Aleo asset"):
        bridge.hyperlane.outbound_route("ethereum/eth")
    assert bridge.aleo.calls == []


def test_input_validation(bridge):
    route = REG.route("hyperlane:aleo/eth->ethereum/eth")
    kw = dict(recipient=EVM1, amount_atomic=1, gas_payment_microcredits=1, decimals=(18, 18))
    for bad in (0, 1 << 64):
        with pytest.raises(ConfigurationError, match="positive u64"):
            bridge.hyperlane.build_transfer_remote_inputs(route, **{**kw, "gas_payment_microcredits": bad})
    with pytest.raises(InvalidAmountError, match="greater than zero"):
        bridge.hyperlane.build_transfer_remote_inputs(route, **{**kw, "amount_atomic": 0})
    with pytest.raises(InvalidRecipientError, match="ethereum address format"):
        bridge.hyperlane.build_transfer_remote_inputs(route, **{**kw, "recipient": "aleo1" + "a" * 58})
    with pytest.raises(InvalidRecipientError):
        bridge.hyperlane.build_transfer_remote_inputs(REG.route("hyperlane:aleo/sol->solana/sol"), **{**kw, "recipient": EVM1, "decimals": (9, 9)})
    with pytest.raises(InvalidAmountError):
        bridge.hyperlane.transfer_remote("aleo/eth", EVM1, amount="0.1234567890123456789")   # 19 fractional digits


def test_quote_gas_payment_vector(bridge):
    quote = bridge.hyperlane.quote_gas_payment("aleo/eth")
    assert (quote.route_id, quote.gas_limit, quote.gas_overhead, quote.gas_price, quote.exchange_rate, quote.payment_microcredits) == \
        ("hyperlane:aleo/eth->ethereum/eth", 44000, 159337, 1000000000, 402, 8174147)
    assert bridge.aleo.fetched.count("hyp_hook_manager.aleo") >= 1
    sol = bridge.hyperlane.quote_gas_payment("aleo/sol")   # SOL_GAS_CONFIG: (300000+200000)*50000000*1000 // 10**10
    assert (sol.gas_limit, sol.payment_microcredits) == (300000, 2_500_000)


def test_quote_gas_payment_failure_modes(bridge):
    configs = bridge.aleo.mappings["hyp_hook_manager.aleo"]["destination_gas_configs"]
    configs[IGP_KEY_ETH] = "{ gas_overhead: 0u128, exchange_rate: 0u128, gas_price: 0u128 }"
    with pytest.raises(ConfigurationError, match="unpriced"):
        bridge.hyperlane.quote_gas_payment("aleo/eth")
    del configs[IGP_KEY_ETH]
    with pytest.raises(ConfigurationError, match="missing on chain"):
        bridge.hyperlane.quote_gas_payment("aleo/eth")
    with pytest.raises(UnsupportedRouteError):
        bridge.hyperlane.quote_gas_payment("ethereum/eth")


def test_zero_gas_limit_falls_back_to_50000(bridge):
    from dataclasses import replace
    route = REG.route("hyperlane:aleo/eth->ethereum/eth")
    zero = replace(route, metadata={**route.metadata, "aleoRemoteRouterGas": "0"})
    quote = bridge.hyperlane.quote_gas_payment(zero)
    assert quote.gas_limit == 50_000 and quote.payment_microcredits == (50_000 + 159337) * 1000000000 * 402 // 10**10


def test_transfer_remote_builds_call_with_live_quote(bridge):
    # WBTC shares the Ethereum IGP config (same destination domain 1u32) but carries its own
    # aleoRemoteRouterGas (68000, vs. ETH's 44000); per compute_gas_payment the live quote is
    # (68000 + 159337) * 1_000_000_000 * 402 // 10_000_000_000 == 9138947, not ETH's 8174147.
    call = bridge.hyperlane.transfer_remote("aleo/wbtc", EVM1, amount="0.0001", as_signer=True)
    assert (call.program_id, call.function_name) == ("hyp_warp_token_wbtc_v2.aleo", "transfer_remote_as_signer")
    assert call.inputs[5] == "10000u128" and "amount: 9138947u64" in call.inputs[6]
    assert bridge.aleo.submitted == []                                   # nothing sent until a verb runs
    result = call.delegate(wait=False)
    assert isinstance(result, DispatchReceipt)
    assert (result.transaction_id, result.route_id, result.message_id, result.amount_atomic) == ("at1delegated", "hyperlane:aleo/wbtc->ethereum/wbtc", None, 10_000)
    assert result.receipt.status is Status.SOURCE_CONFIRMING and result.receipt.source_tx_id == "at1delegated"
    assert result.receipt.protocol_state == {"routeId": "hyperlane:aleo/wbtc->ethereum/wbtc", "sourceProgram": "hyp_warp_token_wbtc_v2.aleo",
                                             "sourceFunction": "transfer_remote_as_signer", "amountAtomic": "10000",
                                             "recipient": EVM1, "gasPaymentMicrocredits": "9138947"}
    assert bridge.aleo.calls[-1] == ("hyp_warp_token_wbtc_v2.aleo", "transfer_remote_as_signer", call.inputs)


def test_transfer_remote_pins_explicit_gas_payment(bridge):
    call = bridge.hyperlane.transfer_remote("aleo/eth", EVM1, amount_atomic=1, gas_payment_microcredits=123)
    assert call.function_name == "transfer_remote" and "amount: 123u64" in call.inputs[6]
    assert "hyp_hook_manager.aleo" not in bridge.aleo.fetched          # no quote read when pinned


def test_is_delivered_reads_mailbox_deliveries(bridge):
    assert bridge.hyperlane.is_delivered("0xc7c2c763ef846ff1583d9222d8ecbfc56da2e0cdcc9a63bc4bde51467644794d") is True
    assert bridge.hyperlane.is_delivered(bytes.fromhex("c7c2c763ef846ff1583d9222d8ecbfc56da2e0cdcc9a63bc4bde51467644794d")) is True
    assert bridge.hyperlane.is_delivered("0x" + "00" * 32) is False
    with pytest.raises(ConfigurationError, match="32-byte message id"):
        bridge.hyperlane.is_delivered("0x1234")

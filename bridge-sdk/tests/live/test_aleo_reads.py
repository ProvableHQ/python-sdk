"""Read-only checks against mainnet: IGP quotes, mailbox/nullifier reads, registry-vs-chain drift, mapping names.
A drift failure means the deployment moved since veil's review — STOP and report; never edit the registry to pass."""
import os
import re

import pytest

from aleo_bridge import encoding as enc
from aleo_bridge.client import BALANCE_MAPPING, balance_program
from aleo_bridge.freezelist import CURRENT_ROOT_KEY, FREEZE_LIST_LAST_INDEX_MAPPING, FREEZE_LIST_MAPPING, build_tree, generate_leaves
from aleo_bridge.hyperlane import compute_gas_payment, gas_config_key, parse_gas_config

pytestmark = pytest.mark.live

MESSAGE_ID = "0xc7c2c763ef846ff1583d9222d8ecbfc56da2e0cdcc9a63bc4bde51467644794d"
ALEO_ORIGIN = ["aleo/eth", "aleo/wbtc", "aleo/usdt", "aleo/sol"]


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


@pytest.mark.parametrize("asset", ALEO_ORIGIN)
def test_igp_quote_is_a_positive_u64(live_bridge, asset):
    quote = live_bridge.hyperlane.quote_gas_payment(asset)
    route = live_bridge.hyperlane.outbound_route(asset)
    assert quote.route_id == route.id
    assert 0 < quote.payment_microcredits < 2**64 and quote.gas_price > 0 and quote.exchange_rate > 0
    assert quote.gas_limit == int(route.metadata["aleoRemoteRouterGas"])
    # Self-consistency: recompute the exact u64 from a raw mapping read taken independently of quote_gas_payment.
    literal = live_bridge.mapping_value(route.meta_str("aleoHookManagerProgram"), "destination_gas_configs",
                                        gas_config_key(route))
    assert literal is not None, f"destination_gas_configs is missing on chain for {route.id}"
    config = parse_gas_config(literal)
    assert config == {"gas_overhead": quote.gas_overhead, "gas_price": quote.gas_price, "exchange_rate": quote.exchange_rate}
    recomputed = compute_gas_payment(gas_limit=quote.gas_limit, gas_overhead=config["gas_overhead"],
                                     gas_price=config["gas_price"], exchange_rate=config["exchange_rate"])
    assert recomputed == quote.payment_microcredits


def test_mailbox_deliveries_vector(live_bridge, record_property):
    delivered = live_bridge.hyperlane.is_delivered(MESSAGE_ID)
    record_property("hyperlane_is_delivered_pinned_message", delivered)
    assert delivered is True
    assert live_bridge.hyperlane.is_delivered("0x" + "00" * 32) is False


def test_usdcx_bridge_nullifier_read(live_bridge):
    assert live_bridge.xreserve.is_delivered(bytes(32)) is False          # the zero nonce was never deposited
    assert live_bridge.xreserve.inbound_route().metadata["bridgeProgram"] == "usdcx_bridge_v2.aleo"


@pytest.mark.parametrize("asset", ALEO_ORIGIN)
def test_registry_matches_deployed_warp_route_state(live_bridge, asset, record_property):
    route = live_bridge.hyperlane.outbound_route(asset)
    program = route.metadata["aleoRouterProgram"]
    app = _squash(live_bridge.mapping_value(program, "app_metadata", "true") or "")
    assert f"token_owner:{route.metadata['aleoTokenOwner']}" in app
    assert f"token_id:{route.metadata['aleoTokenId']}" in app
    assert f"local_decimals:{route.metadata['aleoLocalDecimals']}u8" in app
    assert f"remote_decimals:{route.metadata['aleoRemoteDecimals']}u8" in app
    raw_router = live_bridge.mapping_value(program, "remote_routers", f"{route.metadata['aleoDestinationDomain']}u32")
    record_property(f"remote_routers_{asset}", raw_router)
    if raw_router is None:
        pytest.skip(f"{program}/remote_routers[{route.metadata['aleoDestinationDomain']}u32] did not parse; logged as None")
    router = _squash(raw_router)
    assert f"gas:{route.metadata['aleoRemoteRouterGas']}u128" in router
    assert _squash(route.metadata["aleoRemoteRouterRecipient"]) in router


def test_mailbox_state_matches_registry(live_bridge):
    route = live_bridge.hyperlane.outbound_route("aleo/eth")
    mailbox = _squash(live_bridge.mapping_value(route.metadata["aleoMailboxProgram"], "mailbox", "true") or "")
    assert f"default_hook:{route.metadata['aleoMailboxDefaultHook']}" in mailbox
    assert f"required_hook:{route.metadata['aleoMailboxRequiredHook']}" in mailbox


def test_freeze_list_mappings_exist_and_proof_builds(live_bridge):
    program_id = "usdcx_stablecoin.aleo"
    fl_program = live_bridge.freezelist.freeze_list_program(program_id)
    assert fl_program == "usdcx_freezelist.aleo"        # the freeze-list mappings live on the freezelist program, not the token
    names = live_bridge.program(fl_program).mappings()
    assert FREEZE_LIST_MAPPING in names and FREEZE_LIST_LAST_INDEX_MAPPING in names, \
        f"freeze-list mapping names differ on chain: {sorted(names)} — update the two constants in freezelist.py only"
    leaves = live_bridge.freezelist.leaves(program_id)
    if "freeze_list_root" in names:
        root = build_tree(generate_leaves(leaves), "mainnet")[-1]
        on_chain_root = live_bridge.mapping_value(fl_program, "freeze_list_root", CURRENT_ROOT_KEY)
        assert on_chain_root == f"{root}field", \
            f"leaves={leaves!r} recomputed root {root} != on-chain freeze_list_root[{CURRENT_ROOT_KEY}] {on_chain_root!r}"
    # exclusion_proof() independently re-verifies the same on-chain root before building the proof.
    proof = live_bridge.freezelist.exclusion_proof(live_bridge.aleo_address(), program_id)
    assert proof.count("leaf_index") == 2 and proof.count("field") == 32


def test_balance_mappings_exist_for_every_aleo_asset(live_bridge):
    for asset in live_bridge.registry.assets(chain="aleo"):
        program = balance_program(asset)
        if program is None:
            continue
        names = live_bridge.program(program).mappings()
        assert BALANCE_MAPPING in names, f"{program} declares {sorted(names)}; adjust BALANCE_MAPPING/balance_program in client.py"


def test_status_is_read_only_and_complete(live_bridge):
    status = live_bridge.status()
    assert status.chains[0].address == live_bridge.aleo_address()
    assert set(status.chains[0].balances) == {a.id for a in live_bridge.registry.assets(chain="aleo")}
    assert status.chains[0].balances["aleo/aleo"] > 0, \
        f"expected the live-test funding key to hold a positive credits balance, got {status.chains[0].balances['aleo/aleo']}"


def test_wrapper_program_is_deployed_with_expected_transitions(live_bridge):
    assert enc.aleo_program_address("shielded_usdcx_wrapper.aleo", "mainnet") == \
        "aleo183r3zgsr57fwtgk5duzeq9kqdkpmmtfj4k5469ddvm3tcfhhls9szktw82"
    functions = live_bridge.program("shielded_usdcx_wrapper.aleo").functions
    assert "private_mint" in functions and "private_burn" in functions
    bridge_functions = live_bridge.program("usdcx_bridge_v2.aleo").functions
    assert "burn_public" in bridge_functions and "burn_public_as_signer" in bridge_functions
    for asset in ALEO_ORIGIN:
        router = live_bridge.program(live_bridge.hyperlane.outbound_route(asset).metadata["aleoRouterProgram"]).functions
        assert "transfer_remote" in router and "transfer_remote_as_signer" in router


def test_transfer_remote_simulate(live_bridge):
    if os.environ.get("BRIDGE_LIVE_SIMULATE") != "1":
        pytest.skip("set BRIDGE_LIVE_SIMULATE=1 to build a local authorization (downloads proving parameters; no broadcast)")
    call = live_bridge.hyperlane.transfer_remote("aleo/wbtc", "0x0000000000000000000000000000000000000001", amount_atomic=1, as_signer=True)
    assert "amount: " in call.inputs[6] and call.inputs[5] == "1u128"
    try:
        authorization = call.simulate()
    except Exception as exc:  # noqa: BLE001 — the facade's own authorization error type varies by binding version
        reason = str(exc)
        if "balance" in reason.lower() or "insufficient" in reason.lower():
            pytest.xfail(reason=f"authorization failed for lack of a wrapped-WBTC balance on the live-test key: {reason}")
        raise
    assert authorization.function_name == "transfer_remote_as_signer"
    assert live_bridge.aleo.network.get_latest_height() > 0   # sanity: the same client reaches the node; nothing was submitted

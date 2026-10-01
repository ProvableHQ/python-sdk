"""Keyless deployment and fee checks; BRIDGE_LIVE_READS=1 explicitly enables them."""
import os

import pytest
from web3 import Web3

from aleo_bridge import Bridge, Ethereum
from aleo_bridge._cctp_abi import function
from aleo_bridge._cctp_message import address_bytes
from aleo_bridge.registry import DEFAULT_REGISTRY

pytestmark = [pytest.mark.live, pytest.mark.slow,
              pytest.mark.skipif(os.environ.get('BRIDGE_LIVE_READS') != '1', reason='set BRIDGE_LIVE_READS=1')]


def readonly(chains):
    from aleo import Aleo, HTTPProvider
    from . import config
    evm = {chain: Ethereum(config.evm_rpc_url('mainnet') if chain == 'ethereum' else config.evm_chain_rpc_url(chain))
           for chain in chains}
    return Bridge(Aleo(HTTPProvider(config.aleo_endpoint(),network='mainnet')),evm=evm)


@pytest.mark.parametrize('other', ['ethereum','base','arbitrum'])
def test_cctp_domains_reciprocal_messengers_and_fees(other):
    bridge = readonly(('arc',other))
    route = DEFAULT_REGISTRY.route(f'cctp:{other}/usdc->arc/usdc')
    for chain, domain, remote in [('arc',26,route.meta_int('sourceDomain')),(other,route.meta_int('sourceDomain'),26)]:
        conn = bridge.evm(chain).conn
        expected = 5042 if chain == 'arc' else route.meta_int('sourceChainId')
        assert conn.chain_id == expected
        transmitter = conn.w3.eth.contract(address=Web3.to_checksum_address(route.meta_str('messageTransmitter')),
                    abi=[function('localDomain',[],['uint32'],True)])
        assert transmitter.functions.localDomain().call() == domain
        messenger = conn.w3.eth.contract(address=Web3.to_checksum_address(route.meta_str('tokenMessenger')),
                    abi=[function('remoteTokenMessengers',[('domain','uint32')],['bytes32'],True)])
        assert bytes(messenger.functions.remoteTokenMessengers(remote).call()) == address_bytes(messenger.address)
        asset = DEFAULT_REGISTRY.asset(f'{chain}/usdc')
        token = conn.w3.eth.contract(address=Web3.to_checksum_address(asset.locator.value),
                    abi=[function('decimals',[],['uint8'],True)])
        assert token.functions.decimals().call() == 6
    for src,dst in [(other,'arc'),('arc',other)]:
        quote = bridge.quote(source_chain=src,source_asset='usdc',destination_chain=dst,amount='5',
                            recipient='0x0000000000000000000000000000000000000022')
        assert 0 <= quote.max_fee_atomic < quote.amount_atomic
        assert quote.amount_out_atomic == quote.amount_atomic-quote.max_fee_atomic


def test_arc_xreserve_deployment_and_live_withdrawal_fee():
    bridge = readonly(('arc',))
    route = DEFAULT_REGISTRY.route('xreserve:arc/usdc->aleo/usdcx')
    conn = bridge.evm('arc').conn
    assert conn.chain_id == 5042
    assert len(conn.w3.eth.get_code(Web3.to_checksum_address(route.meta_str('xReserveContract')))) > 0
    quote = bridge.quote(source_chain='aleo',source_asset='usdcx',destination_chain='arc',amount='5',
                         recipient='0x0000000000000000000000000000000000000022')
    assert quote.status == 'quoted'
    assert 0 <= quote.withdrawal_fee_atomic < 5_000_000

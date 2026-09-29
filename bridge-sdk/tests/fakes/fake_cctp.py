"""CCTP network fixtures behind real Web3 ABI encoding, event decoding, and signing.

Messages adapt the vendored Veil cctp.test.ts fixture to each route and local test key.
"""
import json
from pathlib import Path

from eth_abi import encode
from eth_account import Account
from eth_utils import keccak
from web3 import Web3

from aleo_bridge import Ethereum
from aleo_bridge.registry import DEFAULT_REGISTRY
from tests.fakes.fake_web3 import FakeRpcProvider, make_bridge, event_log, ZERO_ADDRESS
from tests.test_cctp_quote import CircleSession

KEY = '0x' + '11' * 32
SENDER = Account.from_key(KEY).address
RECIPIENT = '0x0000000000000000000000000000000000000022'
DEST_HASH = '0x' + (13).to_bytes(32, 'big').hex()
NONCE = (123).to_bytes(32, 'big')
FIXTURE = json.loads((Path(__file__).parents[1] / 'fixtures/cctp-v2.json').read_text())


class CctpProvider(FakeRpcProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.used = False

    def _call(self, call):
        raw = bytes.fromhex(call.get('data', call.get('input', '0x'))[2:])
        if raw[:4] == keccak(text='usedNonces(bytes32)')[:4]:
            return '0x' + encode(['uint256'], [int(self.used)]).hex()
        return super()._call(call)


class CctpCircle(CircleSession):
    def __init__(self, harness):
        super().__init__()
        self.harness = harness
        self.attestation_status = 'complete'
        self.forward_hash = DEST_HASH
        self.messages = None

    def json(self):
        if '/fees/' in self.urls[-1]:
            return super().json()
        return {'messages': self.messages if self.messages is not None else [{
            'message': '0x' + self.harness.attested.hex(), 'attestation': '0xabcd',
            'status': self.attestation_status, 'forwardTxHash': self.forward_hash}]}


class Harness:
    def __init__(self, source='ethereum', destination='arc', *, forwarding=True, allowance=5_000_000,
                 checkpoints=None):
        self.route = DEFAULT_REGISTRY.route(f'cctp:{source}/usdc->{destination}/usdc')
        m = self.route.metadata
        self.messenger, self.transmitter = m['tokenMessenger'], m['messageTransmitter']
        self.source_token = DEFAULT_REGISTRY.asset(f'{source}/usdc').locator.value
        self.destination_token = DEFAULT_REGISTRY.asset(f'{destination}/usdc').locator.value
        self.forwarding = forwarding
        self.source = CctpProvider(chain_id=m['sourceChainId'], eth_balances={SENDER: 10**18},
                    token_balances={(self.source_token, SENDER): 10_000_000},
                    allowances={(self.source_token, SENDER, self.messenger): allowance})
        self.destination = CctpProvider(chain_id=m['destinationChainId'], eth_balances={SENDER: 10**18})
        self.bridge = make_bridge(evm={source: Ethereum(w3=Web3(self.source), private_key=KEY),
                                      destination: Ethereum(w3=Web3(self.destination), private_key=KEY)},
                                  checkpoints=checkpoints)
        self.bridge.cctp.circle_session = self.circle = CctpCircle(self)
        self.plan = self.bridge.quote(route=self.route, amount='5', sender=SENDER, recipient=RECIPIENT,
                       cctp={'speed': 'fast', 'forwarding': forwarding, 'max_fee': '0.1'}).plan
        self.burned, self.attested = self.message(False), self.message(True)
        self.source.receipt_logs = lambda tx: self.source_logs(tx['hash']) if tx['to'].lower() == self.messenger.lower() else []
        self.destination.receipt_logs = lambda tx: self.destination_logs(tx['hash'])
        self.saved = []
        self.complete_destination()

    def message(self, attested):
        raw = bytearray.fromhex(FIXTURE['attested' if attested else 'source'][2:])
        for start, value in ((4, self.route.metadata['sourceDomain']), (8, self.route.metadata['destinationDomain'])):
            raw[start:start+4] = value.to_bytes(4, 'big')
        raw[152:184] = bytes.fromhex(self.source_token[2:]).rjust(32, b'\0')
        raw[248:280] = bytes.fromhex(SENDER[2:]).rjust(32, b'\0')
        return bytes(raw if self.forwarding else raw[:376])

    def source_logs(self, tx_hash):
        return [event_log(self.transmitter, ['0x'+keccak(text='MessageSent(bytes)').hex()],
                         '0x'+encode(['bytes'], [self.burned]).hex(), log_index=0, tx_hash=tx_hash)]

    def destination_logs(self, tx_hash=DEST_HASH, amount=4_990_000):
        topics = ['0x'+keccak(text='MessageReceived(address,uint32,bytes32,bytes32,uint32,bytes)').hex(),
                  '0x'+encode(['address'], [SENDER]).hex(), '0x'+NONCE.hex(), '0x'+encode(['uint32'], [1000]).hex()]
        received = event_log(self.transmitter, topics,
                   '0x'+encode(['uint32','bytes32','bytes'], [self.route.metadata['sourceDomain'],
                        bytes.fromhex(self.messenger[2:]).rjust(32,b'\0'), self.attested[148:]]).hex(),
                   log_index=0, tx_hash=tx_hash)
        minted = event_log(self.destination_token, ['0x'+keccak(text='Transfer(address,address,uint256)').hex(),
                          '0x'+encode(['address'], [ZERO_ADDRESS]).hex(), '0x'+encode(['address'], [RECIPIENT]).hex()],
                           '0x'+encode(['uint256'], [amount]).hex(), log_index=1, tx_hash=tx_hash)
        return [received, minted]

    def complete_destination(self):
        self.destination.used = True
        self.destination.add_receipt(DEST_HASH, logs=self.destination_logs())

    def execute(self):
        return self.bridge.execute(self.plan, timeout_seconds=0, on_checkpoint=self.saved.append)

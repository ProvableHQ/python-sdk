import json
from pathlib import Path

import pytest

from aleo_bridge.errors import BridgeError

FIXTURE = json.loads((Path(__file__).parent / 'fixtures/cctp-v2.json').read_text())
SOURCE = bytes.fromhex(FIXTURE['source'][2:])
ATTESTED = bytes.fromhex(FIXTURE['attested'][2:])
KW = dict(source_domain=0, destination_domain=26, messenger='0x28b5a0e9C621a5BadaA536219b3a228C8168cf5d',
          source_token='0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
          sender='0x0000000000000000000000000000000000000011',
          recipient='0x0000000000000000000000000000000000000022', amount_atomic=5_000_000,
          max_fee_atomic=100_000, finality=1000, forwarding=True)


def test_upstream_vectors_keep_only_circle_mutable_fields():
    from aleo_bridge._cctp_message import validate_message, immutable_message
    a = validate_message(ATTESTED, **KW)
    assert a.amount == 5_000_000 and a.fee == 10000 and int.from_bytes(a.nonce, 'big') == 123
    assert immutable_message(SOURCE) == immutable_message(ATTESTED)


@pytest.mark.parametrize('offset', [0, 4, 8, 44, 76, 108, 140, 148, 152, 184, 216, 248, 280, 376, 407])
def test_each_immutable_field_is_bound_to_intent(offset):
    from aleo_bridge._cctp_message import validate_message
    changed = bytearray(SOURCE)
    changed[offset] ^= 2
    with pytest.raises(BridgeError):
        validate_message(bytes(changed), **KW)


@pytest.mark.parametrize('size', [0, 147, 375, 377, 407, 409])
def test_truncated_or_unrecognized_hook_refused(size):
    from aleo_bridge._cctp_message import validate_message
    raw = (SOURCE + b'\0')[:size]
    with pytest.raises(BridgeError):
        validate_message(raw, **KW)


def test_v1_hook_accepted_for_recovery_but_new_frame_is_v0():
    from aleo_bridge._cctp_message import validate_message, FORWARD_HOOK
    assert FORWARD_HOOK == b'cctp-forward'.ljust(24, b'\0') + bytes(8)
    changed = bytearray(SOURCE)
    changed[403] = 1
    validate_message(bytes(changed), **KW)

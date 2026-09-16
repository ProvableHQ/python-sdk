import pytest

from aleo_bridge._keccak import keccak256
from aleo_bridge.circle import CircleClient
from aleo_bridge.errors import AttestationError, ConfigurationError
from aleo_bridge.types import Attestation

BASE = "https://xreserve-api.circle.com/v1/attestations"
PAYLOAD = bytes(305)
HASH = keccak256(PAYLOAD)
SIG = bytes.fromhex("11" * 65)


class _Response:
    def __init__(self, status_code, body=None):
        self.status_code, self._body = status_code, body

    def json(self):
        return self._body


class _Session:
    def __init__(self, *responses):
        self._responses, self.urls = list(responses), []

    def get(self, url, timeout=None):
        self.urls.append((url, timeout))
        return self._responses.pop(0)


def _body(payload=PAYLOAD, signature=SIG, message_hash=HASH):
    return {"attestation": {"payload": "0x" + payload.hex(), "attestation": "0x" + signature.hex(), "messageHash": "0x" + message_hash.hex()}}


def test_404_is_pending_none():
    session = _Session(_Response(404))
    assert CircleClient(BASE, session=session, timeout=9).get_attestation("0x" + HASH.hex()) is None
    assert session.urls == [(f"{BASE}/0x{HASH.hex()}", 9)]


def test_complete_attestation_is_verified():
    att = CircleClient(BASE, session=_Session(_Response(200, _body()))).get_attestation(HASH.hex())  # bare hex accepted
    assert att == Attestation(payload=PAYLOAD, message_hash=HASH, attestation=SIG, status="complete")


def test_rejects_http_errors_and_bad_bodies():
    with pytest.raises(AttestationError, match="HTTP 500"):
        CircleClient(BASE, session=_Session(_Response(500))).get_attestation("0x" + HASH.hex())
    with pytest.raises(AttestationError, match="invalid response"):
        CircleClient(BASE, session=_Session(_Response(200, {"attestation": {"payload": "zz"}}))).get_attestation("0x" + HASH.hex())
    with pytest.raises(AttestationError, match="invalid response"):
        CircleClient(BASE, session=_Session(_Response(200, {}))).get_attestation("0x" + HASH.hex())
    with pytest.raises(AttestationError, match="different message hash"):
        CircleClient(BASE, session=_Session(_Response(200, _body(message_hash=bytes(32))))).get_attestation("0x" + HASH.hex())
    other = bytes.fromhex("01" * 305)
    with pytest.raises(AttestationError, match="does not match the requested message hash"):
        CircleClient(BASE, session=_Session(_Response(200, _body(payload=other)))).get_attestation("0x" + HASH.hex())
    with pytest.raises(AttestationError, match="32-byte"):
        CircleClient(BASE, session=_Session()).get_attestation("0x1234")
    with pytest.raises(ConfigurationError, match="https"):
        CircleClient("http://insecure.example")

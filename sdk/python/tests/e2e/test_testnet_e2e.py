"""Live testnet end-to-end tests for the Aleo Python SDK facade.

These exercise the *flagship* delegated-proving path and the hosted record
scanner against a REAL testnet endpoint + a REAL Delegated Proving Service
(DPS).  They are therefore:

* ``@pytest.mark.slow`` (module-level ``pytestmark``) — deselected by the fast
  suite (``-m "not slow"``); run with ``python -m pytest ... -m slow``.
* env-gated — the whole module is skipped when the funded key
  ``ALEO_E2E_PRIVATE_KEY`` is absent, and each test additionally skips when its
  own credentials (DPS api key / consumer id, scanner creds) are missing.  With
  no env set the module collects and skips cleanly (no errors).

Self-contained by design: all fixtures/helpers live INLINE here (the shared
``tests/e2e/conftest.py`` is owned by another workstream — do not add to it).

Env vars
--------
``ALEO_E2E_PRIVATE_KEY``
    A FUNDED testnet private key (``APrivateKey1…``).  REQUIRED — the module is
    skipped when unset.
``ALEO_E2E_ENDPOINT``
    REST endpoint (versioned API root).  Default
    ``https://api.provable.com/v2``.
``ALEO_E2E_API_KEY`` / ``ALEO_E2E_CONSUMER_ID``
    DPS / hosted-scanner credentials.  Tests needing them skip when unset.
``ALEO_E2E_PROVER_URI``
    DPS prover base URI (without the ``/{network}`` suffix — the client appends
    it, mirroring the TS SDK's ``proverUri + "/{network}"``).  REQUIRED for the
    delegated-proving tests: the prover is a *distinct service host* from the
    read/JWT API (``api.provable.com``), so there is no sensible fallback — the
    ``pubkey``/``prove`` handshake 404s against the read node.  Tests needing it
    skip when unset.

DPS credential wiring (mirrors ``AleoNetworkClient.submit_proving_request``):
``api_key`` and ``prover_uri`` are passed through the :class:`HTTPProvider`
(which forwards them to the network client), while ``consumer_id`` — which
``HTTPProvider`` does not accept — is set directly on ``aleo.network_client``
after construction.  ``BoundCall.delegate`` calls ``submit_proving_request``
with no explicit creds, so it resolves ``self.api_key`` / ``self.consumer_id``
off that client instance.

Transient-503 note: the Provable API and DPS intermittently return 503s; the
state-root/prover calls here are wrapped in a small retry (mirroring the
``_prepare_with_retry`` idiom in ``tests/test_proving.py``) so infra noise does
not masquerade as a regression.
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, TypeVar

import pytest

from aleo import Aleo, HTTPProvider

# ── Module-level marker + env gate ───────────────────────────────────────────

pytestmark = pytest.mark.live

_PRIVATE_KEY = os.environ.get("ALEO_E2E_PRIVATE_KEY")
_ENDPOINT = os.environ.get(
    "ALEO_E2E_ENDPOINT", "https://api.provable.com/v2"
)
_API_KEY = os.environ.get("ALEO_E2E_API_KEY")
_CONSUMER_ID = os.environ.get("ALEO_E2E_CONSUMER_ID")
_PROVER_URI = os.environ.get("ALEO_E2E_PROVER_URI")

# Skip the ENTIRE module (collection still succeeds) when the funded key is
# absent — this is what makes a credential-less run report "skipped", not
# "errored".
if _PRIVATE_KEY is None:
    pytest.skip(
        "ALEO_E2E_PRIVATE_KEY is not set — skipping live testnet e2e tests.",
        allow_module_level=True,
    )

_NETWORK = "testnet"


# ── Inline retry helper (mirrors _prepare_with_retry in test_proving.py) ─────

_T = TypeVar("_T")


def _with_retry(
    fn: Callable[[], _T],
    *,
    attempts: int = 3,
    delay: float = 20.0,
) -> _T:
    """Run *fn*, retrying transient outages (nginx 503s surface as errors).

    A failed state-root fetch / prover round-trip is infrastructure noise, not a
    proving regression, so retry a few times before letting the error surface.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # snarkvm/DPS surface HTTP failures broadly
            last = exc
            if attempt < attempts - 1:
                time.sleep(delay)
    assert last is not None
    raise last


# ── Inline fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def account() -> Any:
    """The funded testnet account derived from ``ALEO_E2E_PRIVATE_KEY``."""
    aleo = Aleo(HTTPProvider(_ENDPOINT, network=_NETWORK))
    assert _PRIVATE_KEY is not None  # guarded by the module-level skip
    return aleo.account.from_private_key(_PRIVATE_KEY)


def _dps_client() -> Aleo:
    """Build an :class:`Aleo` wired for delegated proving against the DPS.

    ``api_key`` + ``prover_uri`` flow through the provider; ``consumer_id`` —
    which the provider does not accept — is set on the network client directly,
    exactly where ``submit_proving_request`` reads it from.
    """
    provider = HTTPProvider(
        _ENDPOINT,
        network=_NETWORK,
        api_key=_API_KEY,
        prover_uri=_PROVER_URI,  # distinct prover host; gated non-None by skipif
    )
    aleo = Aleo(provider)
    # consumer_id is not a provider field; wire it where the DPS path reads it.
    aleo.network_client.consumer_id = _CONSUMER_ID
    return aleo


# ── Test 1: delegated transfer_public against a real DPS ─────────────────────


@pytest.mark.skipif(
    _API_KEY is None or _CONSUMER_ID is None or _PROVER_URI is None,
    reason="ALEO_E2E_API_KEY / ALEO_E2E_CONSUMER_ID / ALEO_E2E_PROVER_URI not set — DPS creds + prover host required.",
)
def test_delegate_transfer_public_live() -> None:
    """REAL delegated proving of a tiny ``credits.aleo/transfer_public``.

    Builds only the main authorization locally and hands it to the DPS; the
    prover's fee master pays (no fee attached — the whole point of the flagship
    path).  Asserts a prover result payload comes back (dict, or an id string).
    """
    aleo = _dps_client()
    assert _PRIVATE_KEY is not None
    acct = aleo.account.from_private_key(_PRIVATE_KEY)
    aleo.default_account = acct

    # credits.aleo/transfer_public(address, u64) — 1 microcredit is the tiniest
    # meaningful amount.  Fetch the deployed program so inputs are validated
    # against the real function signature.
    program = aleo.programs.get("credits.aleo")
    # Transfer 1 microcredit to self — a valid recipient; the fee master pays,
    # so this is effectively free and side-effect-free.
    bound = program.functions.transfer_public(str(acct.address), 1)

    # Fee master pays by default: no pay_own_fee, no fee_record.
    result: Any = _with_retry(lambda: bound.delegate(acct))

    # The DPS result payload is the raw "data" dict from submit_proving_request;
    # some deployments return a bare tx-id string.  Accept either shape, but it
    # must be present and non-empty.
    assert result is not None
    assert isinstance(result, (dict, str))
    if isinstance(result, dict):
        assert len(result) > 0
    else:
        assert result.strip() != ""


# ── Test 2: hosted record scanner ────────────────────────────────────────────


@pytest.mark.skipif(
    _API_KEY is None,
    reason="ALEO_E2E_API_KEY not set — hosted scanner requires an api key.",
)
def test_hosted_record_scanner_live() -> None:
    """Register with the hosted scanner and query owned credits records.

    Registration shares the account's view key with the hosted service (that is
    the point of the hosted path).  We assert the calls SUCCEED and return the
    documented shapes — a funded account has ≥0 records, but testnet state
    varies, so we assert type/shape and absence of error, not a fixed count.
    """
    provider = HTTPProvider(_ENDPOINT, network=_NETWORK, api_key=_API_KEY)
    aleo = Aleo(provider)
    if _CONSUMER_ID is not None:
        aleo.network_client.consumer_id = _CONSUMER_ID

    assert _PRIVATE_KEY is not None
    acct = aleo.account.from_private_key(_PRIVATE_KEY)

    # register() → dict {"ok": bool, "data"/"error": ...}; the hosted service can
    # be flaky, so retry transient failures.
    reg: Any = _with_retry(lambda: aleo.records.register(acct))
    assert isinstance(reg, dict)

    # find_credits(account) → list[OwnedRecord] (list of dicts).
    records: Any = _with_retry(lambda: aleo.records.find_credits(acct))
    assert isinstance(records, list)
    for rec in records:
        # OwnedRecord is a TypedDict → a plain dict at runtime.
        assert isinstance(rec, dict)

    # get_unspent(...) → a network RecordPlaintext, or None when nothing covers
    # the ask.  Either is a valid outcome on a live account.
    unspent: Any = _with_retry(
        lambda: aleo.records.get_unspent(
            program="credits.aleo", record="credits"
        )
    )
    if unspent is not None:
        # A RecordPlaintext stringifies to a "{ ... }" record literal.
        assert "{" in str(unspent)


# ── Test 3: full private roundtrip (delegated prover + hosted scanner) ────────


@pytest.mark.skipif(
    _API_KEY is None or _CONSUMER_ID is None or _PROVER_URI is None,
    reason="ALEO_E2E_API_KEY / ALEO_E2E_CONSUMER_ID / ALEO_E2E_PROVER_URI not set — DPS + scanner creds + prover host required.",
)
def test_private_roundtrip_live() -> None:
    """End-to-end private roundtrip on live testnet.

    Combines the two flagship trust-minimising paths: **delegated proving** (the
    prover's fee master pays both proofs) and the **hosted record scanner**
    (registration shares the view key so the service can index owned records):

      1. ``delegate(transfer_public_to_private)`` — mint a private credits record.
      2. hosted-scanner discovery — poll ``aleo.records`` until the minted record
         is indexed (block time + scanner-sync latency, so retries are generous).
      3. ``delegate(transfer_private)`` — spend that record back to self.

    Requires a funded account with public credits to move into the private
    record.  Long-running; ``@pytest.mark.slow`` + env-gated.
    """
    aleo = _dps_client()
    assert _PRIVATE_KEY is not None
    acct = aleo.account.from_private_key(_PRIVATE_KEY)
    aleo.default_account = acct

    # Register so the hosted scanner indexes this account's records.
    _with_retry(lambda: aleo.records.register(acct))

    program = aleo.programs.get("credits.aleo")

    # 1) Mint a private credits record via delegated proving (fee master pays).
    _with_retry(
        lambda: program.functions.transfer_public_to_private(
            str(acct.address), 100_000
        ).delegate(acct)
    )

    # 2) Poll the hosted scanner until an unspent private credits record is
    #    discoverable (the mint guarantees at least one exists once indexed).
    def _find_record() -> Any:
        rec = aleo.records.get_unspent(program="credits.aleo", record="credits")
        if rec is None:
            raise AssertionError("minted record not yet indexed by the scanner")
        return rec

    record = _with_retry(_find_record, attempts=10, delay=30.0)
    assert "{" in str(record)

    # 3) Spend that record with a private transfer back to self, delegated.
    result: Any = _with_retry(
        lambda: program.functions.transfer_private(
            record, str(acct.address), 1
        ).delegate(acct)
    )
    assert result is not None
    assert isinstance(result, (dict, str))

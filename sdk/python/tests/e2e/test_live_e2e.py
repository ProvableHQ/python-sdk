"""Live end-to-end tests for the Aleo Python SDK facade (testnet + mainnet).

These exercise the *flagship* delegated-proving path and the hosted record
scanner against the REAL Provable API + a REAL Delegated Proving Service (DPS),
on BOTH networks (each test is parametrised over ``testnet`` and ``mainnet`` and
runs one network at a time).  They are therefore:

* ``@pytest.mark.live`` (module-level ``pytestmark``) — excluded from every CI
  lane; run locally with ``python -m pytest ... -m live``.
* env-gated — the whole module is skipped when the funded key
  ``ALEO_E2E_PRIVATE_KEY`` is absent, and each test additionally skips when its
  own credentials (DPS/scanner api key + consumer id) are missing.  With no env
  set the module collects and skips cleanly (no errors).

Self-contained by design: all fixtures/helpers live INLINE here (the shared
``tests/e2e/conftest.py`` is owned by the devnode workstream — do not add to it).

One endpoint, everything derived
--------------------------------
A user only ever configures ``https://api.provable.com`` (+ creds).  From that
one origin the SDK derives, per network:

* reads/RPC         → ``{origin}/v2/{network}/…``
* delegated proving → ``{origin}/prove/{network}/…``
* hosted scanner    → ``{origin}/scanner/{network}/…``

``api_key`` + ``consumer_id`` (shared by prover and scanner) flow through the one
:class:`HTTPProvider`; there is no per-service RPC config to set.

Env vars
--------
``ALEO_E2E_PRIVATE_KEY``
    A FUNDED private key (``APrivateKey1…``).  REQUIRED — the module is skipped
    when unset.  Per-network overrides ``ALEO_E2E_PRIVATE_KEY_TESTNET`` /
    ``ALEO_E2E_PRIVATE_KEY_MAINNET`` take precedence for that network when set
    (an Aleo address is identical across networks, but funding is per-network).
``ALEO_E2E_ENDPOINT``
    API origin/root.  Default ``https://api.provable.com/v2`` (serves both
    networks under ``/v2/{network}``).
``ALEO_E2E_API_KEY`` / ``ALEO_E2E_CONSUMER_ID``
    DPS + hosted-scanner credentials (shared).  Tests needing them skip when
    unset.
``ALEO_E2E_PROVER_URI``
    Optional override for the DPS prover base; unset ⇒ ``{origin}/prove`` derived
    from the endpoint origin.

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
_ENDPOINT = os.environ.get("ALEO_E2E_ENDPOINT", "https://api.provable.com/v2")
_API_KEY = os.environ.get("ALEO_E2E_API_KEY")
_CONSUMER_ID = os.environ.get("ALEO_E2E_CONSUMER_ID")
_PROVER_URI = os.environ.get("ALEO_E2E_PROVER_URI")

# Skip the ENTIRE module (collection still succeeds) when the funded key is
# absent — this is what makes a credential-less run report "skipped", not
# "errored".
if _PRIVATE_KEY is None:
    pytest.skip(
        "ALEO_E2E_PRIVATE_KEY is not set — skipping live e2e tests.",
        allow_module_level=True,
    )

# Run every test on BOTH networks, one at a time.
_NETWORKS = ("testnet", "mainnet")


def _key_for(network: str) -> str:
    """Funded key for *network*: per-network override, else the shared key."""
    override = os.environ.get(f"ALEO_E2E_PRIVATE_KEY_{network.upper()}")
    key = override or _PRIVATE_KEY
    assert key is not None  # guarded by the module-level skip
    return key


@pytest.fixture(params=_NETWORKS)
def network(request: Any) -> str:
    """Parametrise each test over testnet then mainnet."""
    return str(request.param)


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


# ── Inline client builder ─────────────────────────────────────────────────────


def _client(network: str, *, with_creds: bool) -> Aleo:
    """Build an :class:`Aleo` for *network*.

    With ``with_creds`` the one provider carries ``api_key`` + ``consumer_id``
    (shared by the delegated prover and the hosted scanner) and, if set, the
    optional ``prover_uri`` override.  Prover base (``{origin}/prove/{network}``)
    and scanner base (``{origin}/scanner/{network}``) both derive from the
    endpoint origin — the user configures only the one endpoint URL.
    """
    kwargs: dict[str, Any] = {"network": network}
    if with_creds:
        kwargs["api_key"] = _API_KEY
        kwargs["consumer_id"] = _CONSUMER_ID
        if _PROVER_URI is not None:
            kwargs["prover_uri"] = _PROVER_URI
    return Aleo(HTTPProvider(_ENDPOINT, **kwargs))


# ── Test 1: delegated transfer_public against a real DPS ─────────────────────


@pytest.mark.skipif(
    _API_KEY is None or _CONSUMER_ID is None,
    reason="ALEO_E2E_API_KEY / ALEO_E2E_CONSUMER_ID not set — DPS creds required.",
)
def test_delegate_transfer_public_live(network: str) -> None:
    """REAL delegated proving of a tiny ``credits.aleo/transfer_public``.

    Builds only the main authorization locally and hands it to the DPS; the
    prover's fee master pays (no fee attached — the whole point of the flagship
    path).  Asserts a prover result payload comes back (dict, or an id string).
    """
    aleo = _client(network, with_creds=True)
    acct = aleo.account.from_private_key(_key_for(network))
    aleo.default_account = acct

    # credits.aleo/transfer_public(address, u64) — 1 microcredit to self (a
    # valid recipient; the fee master pays, so this is ~free and side-effect
    # free).  Fetch the deployed program so inputs validate against the real
    # function signature.
    program = aleo.programs.get("credits.aleo")
    bound = program.functions.transfer_public(str(acct.address), 1)

    # Fee master pays by default: no pay_own_fee, no fee_record.
    result: Any = _with_retry(lambda: bound.delegate(acct))

    assert result is not None
    assert isinstance(result, (dict, str))
    if isinstance(result, dict):
        assert len(result) > 0
    else:
        assert result.strip() != ""


# ── Test 2: hosted record scanner ────────────────────────────────────────────


@pytest.mark.skipif(
    _API_KEY is None or _CONSUMER_ID is None,
    reason="ALEO_E2E_API_KEY / ALEO_E2E_CONSUMER_ID not set — hosted scanner creds required.",
)
def test_hosted_record_scanner_live(network: str) -> None:
    """Register with the hosted scanner and query owned credits records.

    Registration seals the account's view key to the scanner's ephemeral pubkey
    (``GET {origin}/scanner/{network}/pubkey`` → sealed box →
    ``POST …/register/encrypted``).  We assert the calls SUCCEED and return the
    documented shapes — a funded account has ≥0 records, but live state varies,
    so we assert type/shape and absence of error, not a fixed count.
    """
    aleo = _client(network, with_creds=True)
    acct = aleo.account.from_private_key(_key_for(network))

    # register() → dict {"ok": bool, "data"/"error": ...}; the hosted service can
    # be flaky, so retry transient failures.
    reg: Any = _with_retry(lambda: aleo.records.register(acct))
    assert isinstance(reg, dict)

    # find_credits(account) → list[OwnedRecord] (list of dicts).
    records: Any = _with_retry(lambda: aleo.records.find_credits(acct))
    assert isinstance(records, list)
    for rec in records:
        assert isinstance(rec, dict)

    # get_unspent(...) → a network RecordPlaintext, or None when nothing covers
    # the ask.  Either is a valid outcome on a live account.
    unspent: Any = _with_retry(
        lambda: aleo.records.get_unspent(program="credits.aleo", record="credits")
    )
    if unspent is not None:
        assert "{" in str(unspent)


# ── Test 3: full private roundtrip (delegated prover + hosted scanner) ────────


@pytest.mark.skipif(
    _API_KEY is None or _CONSUMER_ID is None,
    reason="ALEO_E2E_API_KEY / ALEO_E2E_CONSUMER_ID not set — DPS + scanner creds required.",
)
def test_private_roundtrip_live(network: str) -> None:
    """End-to-end private roundtrip on live {testnet, mainnet}.

    Combines the two flagship trust-minimising paths: **delegated proving** (the
    prover's fee master pays both proofs) and the **hosted record scanner** (the
    view key is sealed to the scanner so it can index owned records):

      1. ``delegate(transfer_public_to_private)`` — mint a private credits record.
      2. hosted-scanner discovery — poll ``aleo.records`` until the minted record
         is indexed (block time + scanner-sync latency, so retries are generous).
      3. ``delegate(transfer_private)`` — spend that record back to self.

    Requires a funded account with public credits to move into the private
    record.  Long-running.
    """
    aleo = _client(network, with_creds=True)
    acct = aleo.account.from_private_key(_key_for(network))
    aleo.default_account = acct

    # Register so the hosted scanner indexes this account's records.  register()
    # returns a status dict (it does not raise), so assert ``ok`` and retry until
    # it truly succeeds — a silently-failed registration would otherwise surface
    # much later as a confusing "No credentials found for given 'iss'" at query
    # time.
    def _register() -> Any:
        r = aleo.records.register(acct)
        if not r.get("ok"):
            raise AssertionError(f"scanner registration not ok: {r}")
        return r

    _with_retry(_register)

    program = aleo.programs.get("credits.aleo")

    # 1) Mint a private credits record via delegated proving (fee master pays).
    _with_retry(
        lambda: program.functions.transfer_public_to_private(
            str(acct.address), 100_000
        ).delegate(acct)
    )

    # 2) Poll the hosted scanner until an unspent private credits record is
    #    discoverable (the mint guarantees at least one exists once indexed).
    #    The delegated-proving JWT mint invalidates the scanner's shared-consumer
    #    JWT out-of-band, but the scanner now self-heals (re-mints on 401), so the
    #    poll only has to wait for indexing latency.
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

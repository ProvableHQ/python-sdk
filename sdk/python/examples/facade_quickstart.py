"""facade_quickstart.py — Aleo Python SDK end-to-end walkthrough.

This file is structured in two layers:

1. OFFLINE — runs without a live node or prover credentials.
   Import, account create/import, sign/verify, unit conversions,
   address validation, and building / inspecting a PreparedCall.
   Run directly: python facade_quickstart.py

2. NETWORK — marked ``# requires a live node`` / ``# requires prover creds``.
   Only executes when NETWORK = True (set below) AND a funded private key is
   available in the environment.  With NETWORK = False the file imports and
   the offline sections run cleanly.

Set NETWORK = True and export ALEO_PRIVATE_KEY=APrivateKey1zkp… to try the
live sections against https://api.provable.com/v2.
"""

# ---------------------------------------------------------------------------
# Guard — flip to True + export ALEO_PRIVATE_KEY to run network sections.
# ---------------------------------------------------------------------------
NETWORK = False

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import asyncio

from aleo import Aleo, AsyncAleo, HTTPProvider  # facade
from aleo import (
    AleoError,
    ExecutionError,
    ProgramNotFound,
    TransactionNotFound,
    TransactionConfirmationTimeout,
)

# ---------------------------------------------------------------------------
# 1. Connect to mainnet
# ---------------------------------------------------------------------------

ENDPOINT = "https://api.provable.com/v2"

# Aleo.HTTPProvider is also importable as HTTPProvider from aleo.
# Both of the following are equivalent:
#   aleo = Aleo(HTTPProvider(ENDPOINT))
#   aleo = Aleo(Aleo.HTTPProvider(ENDPOINT))
aleo = Aleo(Aleo.HTTPProvider(ENDPOINT))

print("Network:", aleo.network_name)    # "mainnet"
print("Network ID:", aleo.network_id)   # 0
print("Repr:", aleo)

# ---------------------------------------------------------------------------
# 2. Account — create and import  (OFFLINE)
# ---------------------------------------------------------------------------

# Create a fresh random account
account = aleo.account.create()
print("\n[account.create]")
print("  address    :", account.address)
print("  private_key:", str(account.private_key)[:20], "…")
print("  view_key   :", str(account.view_key)[:20], "…")

# Re-import from the private key string
pk_str = str(account.private_key)
account2 = aleo.account.from_private_key(pk_str)
assert str(account.address) == str(account2.address), "round-trip failed"
print("  from_private_key round-trip OK")

# ---------------------------------------------------------------------------
# 3. Sign and verify  (OFFLINE)
# ---------------------------------------------------------------------------

message = b"hello aleo"
signature = aleo.account.sign(message, account)
print("\n[sign/verify]")
print("  signature  :", str(signature)[:20], "…")

ok = aleo.account.verify(str(account.address), message, signature)
assert ok, "signature verification failed"
print("  verify OK  :", ok)

# sign_value / verify_value — Aleo typed-value signing (facade methods)
sig2 = aleo.account.sign_value("100u64", account)
ok2 = aleo.account.verify_value(str(account.address), "100u64", sig2)
assert ok2, "sign_value verification failed"
print("  sign_value OK:", str(sig2)[:20], "…")

# ---------------------------------------------------------------------------
# 4. Unit conversions  (OFFLINE)
# ---------------------------------------------------------------------------

print("\n[unit conversions]")
print("  1.5 credits →", aleo.to_microcredits(1.5), "microcredits")
print("  1_500_000 µcredits →", aleo.from_microcredits(1_500_000), "credits")

# ---------------------------------------------------------------------------
# 5. Address validation  (OFFLINE)
# ---------------------------------------------------------------------------

print("\n[is_valid_address]")
valid_addr = str(account.address)
print("  valid  :", aleo.is_valid_address(valid_addr))    # True
print("  invalid:", aleo.is_valid_address("not_an_addr")) # False

# ---------------------------------------------------------------------------
# 6. Build a PreparedCall / BoundCall  (OFFLINE — uses a local program source)
# ---------------------------------------------------------------------------
# programs.get() fetches from the network, so we build the Program locally
# here for the offline demo.  The call-building and coercion logic is identical.

from aleo.mainnet import Program as _Program
from aleo.facade.programs import PreparedCall, ProgramFunctions

_SRC = """\
program hello.aleo;
function greet:
    input r0 as u64.public;
    output r0 as u64.public;
"""
_raw = _Program.from_source(_SRC)
_inputs_by_fn = {str(f): list(_raw.get_function_inputs(str(f))) for f in _raw.functions}
_pf = ProgramFunctions("hello.aleo", _inputs_by_fn, client=None)

print("\n[PreparedCall / coercion]")
caller = _pf["greet"]
print("  caller repr  :", caller)
print("  signature    :", caller.signature)

pc = PreparedCall("hello.aleo", "greet", _inputs_by_fn["greet"], (42,), client=None)
print("  coerced args :", pc.args)          # ['42u64']
print("  PreparedCall :", pc)

# ---------------------------------------------------------------------------
# 7. Connectivity check + balance read  (NETWORK)
# ---------------------------------------------------------------------------

if NETWORK:
    # requires a live node
    print("\n[is_connected]")
    connected = aleo.is_connected()
    print("  connected:", connected)

    # Read the credits.aleo account mapping for this address
    print("\n[get_balance]")
    bal = aleo.get_balance(str(account.address))
    print("  balance (µcredits):", bal)
    print("  balance (credits) :", aleo.from_microcredits(bal))

# ---------------------------------------------------------------------------
# 8. Read an on-chain mapping  (NETWORK)
# ---------------------------------------------------------------------------

if NETWORK:
    # requires a live node
    print("\n[mapping read]")
    credits_prog = aleo.programs.get("credits.aleo")
    print("  program:", credits_prog)
    print("  functions:", list(credits_prog.functions)[:5])
    raw_bal = credits_prog.mapping("account").get(str(account.address))
    print("  account mapping value:", raw_bal)

# ---------------------------------------------------------------------------
# 9. Build a call + simulate (local inspect)  (NETWORK for programs.get;
#    simulate() itself is local once the program is fetched)
# ---------------------------------------------------------------------------

if NETWORK:
    # requires a live node (to fetch the program source)
    print("\n[simulate / authorize — local, no proof]")
    credits = aleo.programs.get("credits.aleo")
    recipient = str(account.address)
    call = credits.functions.transfer_public(recipient, 1_000_000)
    print("  call signature:", call.signature)
    print("  coerced args  :", call.args)

    # simulate() builds the Authorization locally — no proof, no send
    auth_result = call.simulate(account)
    print("  authorize outputs:", auth_result.outputs)
    print("  decoded transitions:", auth_result.decoded())

# ---------------------------------------------------------------------------
# 10. transact — full prove + broadcast  (NETWORK)
# ---------------------------------------------------------------------------

if NETWORK:
    # requires a live node + funded private key
    # Set ALEO_PRIVATE_KEY in the environment to use your own key.
    pk_env = os.environ.get("ALEO_PRIVATE_KEY")
    if pk_env:
        funded_account = aleo.account.from_private_key(pk_env)
        print("\n[transact — full prove + broadcast]")
        credits = aleo.programs.get("credits.aleo")
        tx_id = credits.functions.transfer_public(
            str(funded_account.address),
            100,   # 100 microcredits (self-send)
        ).transact(funded_account)
        print("  tx_id:", tx_id)
        confirmed = aleo.network.wait_for_transaction(tx_id, timeout=60.0)
        print("  confirmed:", confirmed)
    else:
        print("\n[transact] ALEO_PRIVATE_KEY not set — skipping")

# ---------------------------------------------------------------------------
# 11. delegate — the flagship DPS path  (NETWORK)
#
#     By default the prover's fee master pays — no credits needed on your side.
#     The DPS endpoint is configured on the provider or via network_client.
# ---------------------------------------------------------------------------

if NETWORK:
    # requires prover credentials; fee master pays
    # Omitting pay_own_fee= and fee_record= means the DPS fee master covers fees.
    pk_env = os.environ.get("ALEO_PRIVATE_KEY")
    if pk_env:
        funded_account = aleo.account.from_private_key(pk_env)
        print("\n[delegate — DPS flagship path; fee master pays]")
        credits = aleo.programs.get("credits.aleo")
        result = credits.functions.transfer_public(
            str(funded_account.address),
            100,
        ).delegate(funded_account)
        print("  delegate result:", result)
    else:
        print("\n[delegate] ALEO_PRIVATE_KEY not set — skipping")

# ---------------------------------------------------------------------------
# 12. AsyncAleo snippet  (OFFLINE construction; NETWORK for I/O)
# ---------------------------------------------------------------------------

async def async_demo() -> None:
    """Demonstrate the AsyncAleo facade."""
    # Construction is sync; all I/O is async.
    async_aleo = AsyncAleo(AsyncAleo.HTTPProvider(ENDPOINT))
    print("\n[AsyncAleo]")
    print("  network:", async_aleo.network_name)

    # account.create / sign / verify are sync even on AsyncAleo
    acct = async_aleo.account.create()
    sig = async_aleo.account.sign(b"async hello", acct)
    assert async_aleo.account.verify(str(acct.address), b"async hello", sig)
    print("  sign/verify on AsyncAleo: OK")

    if NETWORK:
        # requires a live node
        connected = await async_aleo.is_connected()
        print("  is_connected:", connected)
        bal = await async_aleo.get_balance(str(acct.address))
        print("  balance:", bal)

        if NETWORK:
            # requires a live node + funded key for transact/delegate
            # async transact:
            #   tx_id = await prog.functions.fn(*args).transact(account)
            # async delegate (fee master pays):
            #   result = await prog.functions.fn(*args).delegate(account)
            pass

# Run the async demo (offline parts always run; network parts only if NETWORK=True)
asyncio.run(async_demo())

# ---------------------------------------------------------------------------
# 13. Error types
# ---------------------------------------------------------------------------

print("\n[error types]")
print("  AleoError, ExecutionError, ProgramNotFound,")
print("  TransactionNotFound, TransactionConfirmationTimeout")
# These are importable from aleo directly:
# from aleo import AleoError, ExecutionError, ProgramNotFound, ...

print("\nAll offline sections passed.")

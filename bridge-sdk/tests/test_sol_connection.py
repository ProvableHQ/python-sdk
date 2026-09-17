import importlib
import json
import sys

import pytest

from aleo_bridge import sol
from aleo_bridge._base58 import b58encode
from aleo_bridge.errors import ConfigurationError, MissingExtraError


class _Reader:
    """Stands in for a caller-configured solana-py Client; never called in this file."""


def _block_solders_and_solana(monkeypatch):
    """Block ``solders``/``solana`` for the duration of the test, reverted automatically by
    ``monkeypatch``. ``_libs()`` imports submodules (``from solders.hash import Hash``, ...), and
    once any earlier test in the run has imported those submodules for real, Python's import
    machinery resolves them straight from ``sys.modules`` without re-checking the (now ``None``)
    top-level package — so a plain ``sys.modules["solders"] = None`` only blocks a *first* import.
    Purging every already-cached ``solders``/``solana`` submodule first makes the block work
    regardless of test order."""
    for name in list(sys.modules):
        if name == "solders" or name.startswith("solders.") or name == "solana" or name.startswith("solana."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "solders", None)
    monkeypatch.setitem(sys.modules, "solana", None)


def test_package_imports_without_solders_or_solana(monkeypatch):
    """``import aleo_bridge`` and ``from aleo_bridge import Solana`` must work with solders/solana
    absent; only the first call that actually needs them raises MissingExtraError. Mirrors
    test_import_without_web3.py's monkeypatch-and-revert pattern so the blocked modules and the
    reimported aleo_bridge never leak into tests that run after this one."""
    _block_solders_and_solana(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("aleo_bridge"):
            monkeypatch.delitem(sys.modules, name)

    pkg = importlib.import_module("aleo_bridge")
    assert pkg.Solana is not None

    from aleo_bridge.errors import MissingExtraError as ReimportedMissingExtraError

    with pytest.raises(ReimportedMissingExtraError) as exc_info:
        pkg.Solana()
    assert "aleo-bridge-sdk[solana]" in str(exc_info.value)


def test_read_only_connection_needs_no_solana_extra(monkeypatch):
    monkeypatch.setattr(sol, "_LIBS", None)
    _block_solders_and_solana(monkeypatch)
    conn = sol.Solana(client=_Reader())
    assert conn.address is None and conn.can_sign is False and conn.rpc_url is None
    with pytest.raises(ConfigurationError, match="read-only"):
        conn.sign_message(b"payload")
    with pytest.raises(ConfigurationError, match="read-only"):
        conn.pubkey
    with pytest.raises(MissingExtraError, match=r"aleo-bridge-sdk\[solana\]"):
        sol.Solana()
    with pytest.raises(MissingExtraError):
        sol.Solana(client=_Reader(), private_key="[1,2,3]")


def test_constructor_argument_conflicts():
    with pytest.raises(ConfigurationError, match="rpc_url or client"):
        sol.Solana("https://rpc.example", client=_Reader())
    pytest.importorskip("solders")
    from solders.keypair import Keypair

    with pytest.raises(ConfigurationError, match="signer or private_key"):
        sol.Solana(client=_Reader(), signer=Keypair(), private_key=b58encode(bytes(Keypair())))


def test_default_transport_is_mainnet_beta_at_confirmed_commitment():
    pytest.importorskip("solders")

    conn = sol.Solana()
    assert sol.DEFAULT_SOLANA_RPC_URL == "https://api.mainnet-beta.solana.com"
    assert conn.rpc_url == sol.DEFAULT_SOLANA_RPC_URL
    assert isinstance(conn.client, sol.SolanaRpcClient)
    assert conn.client.url == sol.DEFAULT_SOLANA_RPC_URL and conn.client.commitment == "confirmed"
    custom = sol.Solana("https://rpc.example")
    assert custom.rpc_url == "https://rpc.example" and custom.client.url == "https://rpc.example"


def test_async_solana_py_client_is_adapted_onto_a_private_loop():
    pytest.importorskip("solders")
    from solders.hash import Hash

    class FakeProvider:
        async def make_request(self, request, response_type):
            return sol.RpcResult(True)

    class FakeAsyncClient:                    # the shape of solana-py ≥ 0.36 AsyncClient
        _provider = FakeProvider()

        def __init__(self):
            self.sent = []

        async def get_balance(self, pubkey, commitment=None):
            return sol.RpcResult(7)

        async def send_raw_transaction(self, txn, opts=None):
            self.sent.append((bytes(txn), opts))
            return sol.RpcResult("sig")

    fake = FakeAsyncClient()
    conn = sol.Solana(client=fake)
    assert isinstance(conn.client, sol._AsyncClientAdapter) and conn.can_sign is False
    assert conn.client.get_balance(None).value == 7                      # coroutine run synchronously
    assert conn.client.is_blockhash_valid(Hash.default()).value is True  # supplied by the adapter
    pytest.importorskip("solana")
    assert conn.client.send_raw_transaction(b"\x01").value == "sig"
    sent_tx, tx_opts = fake.sent[0]
    assert sent_tx == b"\x01" and tx_opts.skip_preflight is False and tx_opts.skip_confirmation is True
    assert str(tx_opts.preflight_commitment) == "confirmed"


def test_private_key_forms():
    pytest.importorskip("solders")
    from solders.keypair import Keypair

    keypair = Keypair()
    secret = bytes(keypair)                                  # 64 bytes: seed || pubkey
    assert len(secret) == 64
    for private_key in (b58encode(secret), json.dumps(list(secret)), " " + json.dumps(list(secret)) + "\n", secret):
        conn = sol.Solana(client=_Reader(), private_key=private_key)
        assert conn.address == str(keypair.pubkey()) and conn.can_sign is True
        assert conn.pubkey == keypair.pubkey()
    assert sol.keypair_from_private_key(secret[:32]).pubkey() == keypair.pubkey()   # 32-byte seed
    for bad in ("not-base58-0OIl", "[1, 2, 3]", "[1, 2, \"x\"]", "[", bytes(63)):
        with pytest.raises(ConfigurationError):
            sol.keypair_from_private_key(bad)


def test_keypair_signer_signs_the_message():
    pytest.importorskip("solders")
    from solders.keypair import Keypair

    keypair = Keypair()
    conn = sol.Solana(client=_Reader(), signer=keypair)
    assert conn.address == str(keypair.pubkey())
    assert conn.sign_message(b"payload") == keypair.sign_message(b"payload")


def test_any_object_with_pubkey_and_sign_message_is_a_signer():
    pytest.importorskip("solders")
    from solders.keypair import Keypair

    inner = Keypair()

    class RemoteSigner:                      # e.g. HSM / remote signing service
        def pubkey(self):
            return inner.pubkey()

        def sign_message(self, message: bytes):
            return inner.sign_message(message)

    conn = sol.Solana(client=_Reader(), signer=RemoteSigner())
    assert isinstance(RemoteSigner(), sol.SolanaSigner)
    assert conn.can_sign and conn.address == str(inner.pubkey())
    assert conn.sign_message(b"m") == inner.sign_message(b"m")
    with pytest.raises(ConfigurationError, match="pubkey\\(\\) and sign_message"):
        sol.Solana(client=_Reader(), signer=object())


def test_from_env():
    pytest.importorskip("solders")
    from solders.keypair import Keypair

    key = b58encode(bytes(Keypair()))
    assert sol.Solana.from_env({}) is None
    with_key = sol.Solana.from_env({"SOLANA_PRIVATE_KEY": key})
    assert with_key is not None and with_key.can_sign and with_key.rpc_url == sol.DEFAULT_SOLANA_RPC_URL
    with_url = sol.Solana.from_env({"SOLANA_PRIVATE_KEY": key, "SOLANA_RPC_URL": "https://rpc.example"})
    assert with_url is not None and with_url.rpc_url == "https://rpc.example"
    read_only = sol.Solana.from_env({"SOLANA_RPC_URL": "https://rpc.example"})
    assert read_only is not None and read_only.can_sign is False


def test_async_client_adapter_close_stops_the_thread():
    pytest.importorskip("solders")

    class FakeAsyncClient:
        async def get_balance(self, pubkey, commitment=None):
            return sol.RpcResult(7)

    adapter = sol._AsyncClientAdapter(FakeAsyncClient())
    assert adapter._thread.is_alive()
    adapter.close()
    assert adapter._thread.is_alive() is False
    adapter.close()  # idempotent: no error, no hang


def test_solana_context_manager_closes_the_wrapped_adapter():
    pytest.importorskip("solders")

    class FakeAsyncClient:
        async def get_balance(self, pubkey, commitment=None):
            return sol.RpcResult(7)

    fake = FakeAsyncClient()
    with sol.Solana(client=fake) as conn:
        adapter = conn.client
        assert isinstance(adapter, sol._AsyncClientAdapter)
        assert adapter._thread.is_alive()
    assert adapter._thread.is_alive() is False


def test_adapter_close_awaits_the_wrapped_clients_own_close_on_its_loop():
    """solana-py's AsyncClient owns an aiohttp session that can only be closed from its loop;
    stopping the thread first would leak it (and warn)."""
    pytest.importorskip("solders")

    class FakeAsyncClient:
        def __init__(self):
            self.closed_on = None

        async def get_balance(self, pubkey, commitment=None):
            return sol.RpcResult(7)

        async def close(self):
            import threading as _threading
            self.closed_on = _threading.current_thread().name

    fake = FakeAsyncClient()
    adapter = sol._AsyncClientAdapter(fake)
    loop_thread = adapter._thread.name
    adapter.close()
    assert fake.closed_on == loop_thread                 # awaited on the private loop, before it stopped
    assert adapter._thread.is_alive() is False
    adapter.close()                                      # idempotent: closes the client exactly once


def test_adapter_close_still_stops_the_thread_when_the_wrapped_client_close_raises():
    """A third-party client that fails to close must not leak our private event-loop thread."""
    pytest.importorskip("solders")

    class ExplodingClient:
        def __init__(self):
            self.attempts = 0

        async def get_balance(self, pubkey, commitment=None):   # pragma: no cover - never called
            return sol.RpcResult(7)

        def close(self):
            self.attempts += 1
            raise OSError("socket already gone")

    fake = ExplodingClient()
    adapter = sol._AsyncClientAdapter(fake)
    adapter.close()
    assert adapter._thread.is_alive() is False
    adapter.close()                                             # idempotent: no second close attempt
    assert fake.attempts == 1


def test_adapter_close_still_stops_the_thread_when_an_async_client_close_raises():
    pytest.importorskip("solders")

    class ExplodingAsyncClient:
        async def get_balance(self, pubkey, commitment=None):    # pragma: no cover - never called
            return sol.RpcResult(7)

        async def close(self):
            raise OSError("session already detached")

    adapter = sol._AsyncClientAdapter(ExplodingAsyncClient())
    adapter.close()
    assert adapter._thread.is_alive() is False


def test_solana_exit_never_raises_even_when_the_client_close_fails():
    pytest.importorskip("solders")

    class ExplodingClient:
        def get_latest_blockhash(self, commitment=None):   # pragma: no cover - never called
            return None

        def close(self):
            raise OSError("socket already gone")

    with sol.Solana(client=ExplodingClient()) as conn:
        assert conn.client is not None
    with pytest.raises(OSError):                           # an explicit close() still reports it
        sol.Solana(client=ExplodingClient()).close()


def test_rpc_client_close_closes_its_session():
    pytest.importorskip("solders")

    class FakeSession:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    session = FakeSession()
    conn = sol.Solana(client=sol.SolanaRpcClient("https://rpc.example", session=session))
    conn.close()
    assert session.closed == 1                             # the default transport releases its pool


def test_close_is_a_noop_for_a_connection_without_a_closeable_client():
    conn = sol.Solana(client=_Reader())
    conn.close()  # no close() on the client — must not raise
    conn.close()


def test_from_env_aliases():
    pytest.importorskip("solders")
    from solders.keypair import Keypair

    key = b58encode(bytes(Keypair()))
    aliased = sol.Solana.from_env({"BRIDGE_SOLANA_PRIVATE_KEY": key})
    assert aliased is not None and aliased.can_sign and aliased.rpc_url == sol.DEFAULT_SOLANA_RPC_URL
    aliased_url = sol.Solana.from_env({"BRIDGE_SOLANA_PRIVATE_KEY": key, "BRIDGE_LIVE_SOLANA_RPC_URL": "https://rpc.example"})
    assert aliased_url is not None and aliased_url.rpc_url == "https://rpc.example"
    primary_wins = sol.Solana.from_env({"SOLANA_PRIVATE_KEY": key, "BRIDGE_SOLANA_PRIVATE_KEY": "ignored",
                                        "SOLANA_RPC_URL": "https://primary.example",
                                        "BRIDGE_LIVE_SOLANA_RPC_URL": "https://alias.example"})
    assert primary_wins is not None and primary_wins.rpc_url == "https://primary.example"

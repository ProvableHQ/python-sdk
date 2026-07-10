"""A Python wrapper around the `aleo-devnode`_ binary — the eth-tester analog.

Spins up a local Aleo development node (a real snarkOS process that does *not*
verify proofs, so transactions land fast), exposes deterministic pre-funded
genesis accounts, and drives the node's REST control plane (produce blocks,
snapshot, shutdown).  Mirrors web3.py's local-node test harness pattern
(Anvil/geth fixtures) rather than the in-process ``EthereumTesterProvider``
(there is no in-process Aleo VM in Python).

Example
-------
::

    from aleo.testing import Devnode

    with Devnode() as dn:                       # picks a free port, waits for ready
        aleo = dn.aleo                          # an Aleo client bound to the node
        sender = dn.accounts[0]                 # pre-funded genesis account
        recipient = aleo.account.create()
        tx = (aleo.programs.get("credits.aleo")
                  .functions.transfer_public(str(recipient.address), 1)
                  .transact(sender))
        dn.advance(1)                           # produce the block (manual mining)
        aleo.network.wait_for_transaction(tx)

.. _aleo-devnode: https://github.com/ProvableHQ/aleo-devnode
"""
from __future__ import annotations

import collections
import os
import shutil
import socket
import subprocess
import threading
import time
from typing import Any

# The devnode seeds 50 funded accounts at genesis; these five are the ones its
# docs publish, and they are identical on every run (deterministic genesis).
# Public development keys — safe to embed in test support.
DEFAULT_ACCOUNTS: list[tuple[str, str]] = [
    ("APrivateKey1zkp8CZNn3yeCseEtxuVPbDCwSyhGW6yZKUYKfgXmcpoGPWH",
     "aleo1rhgdu77hgyqd3xjj8ucu3jj9r2krwz6mnzyd80gncr5fxcwlh5rsvzp9px"),
    ("APrivateKey1zkp2RWGDcde3efb89rjhME1VYA8QMxcxep5DShNBR6n8Yjh",
     "aleo1s3ws5tra87fjycnjrwsjcrnw2qxr8jfqqdugnf0xzqqw29q9m5pqem2u4t"),
    ("APrivateKey1zkp2GUmKbVsuc1NSj28pa1WTQuZaK5f1DQJAT6vPcHyWokG",
     "aleo1ashyu96tjwe63u0gtnnv8z5lhapdu4l5pjsl2kha7fv7hvz2eqxs5dz0rg"),
    ("APrivateKey1zkpBjpEgLo4arVUkQmcLdKQMiAKGaHAQVVwmF8HQby8vdYs",
     "aleo12ux3gdauck0v60westgcpqj7v8rrcr3v346e4jtq04q7kkt22czsh808v2"),
    ("APrivateKey1zkp3J6rRrDEDKAMMzSQmkBqd3vPbjp4XTyH7oMKFn7eVFwf",
     "aleo1p9sg8gapg22p3j42tang7c8kqzp4lhe6mg77gx32yys2a5y7pq9sxh6wrd"),
]

# The first genesis account; also veil's exported DEVNODE_PRIVATE_KEY.
DEVNODE_PRIVATE_KEY: str = DEFAULT_ACCOUNTS[0][0]

_DEFAULT_START_RETRIES = 5


def _free_port() -> int:
    """Return an OS-chosen free TCP port on the loopback interface.

    There is an inherent race between releasing this port and the devnode
    binding it, so :class:`Devnode` retries with a fresh port on bind failure.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class DevnodeError(RuntimeError):
    """Raised when the devnode binary is missing or fails to become ready."""


class Devnode:
    """A managed local ``aleo-devnode`` process with a REST control plane.

    Parameters
    ----------
    private_key:
        Block-producing key.  Defaults to the first genesis account.
    port:
        Bind port.  ``None`` (default) auto-allocates a free port and retries a
        new one on collision — safe for per-test / ``pytest-xdist`` parallelism.
        Pin a port to disable the auto-reroll (a collision then raises).
    storage_path:
        Persist the ledger to this directory (``None`` = in-memory, fastest).
    manual_block_creation:
        When ``True`` (default) blocks are produced only via :meth:`advance`,
        giving deterministic, eth-tester-style control over mining.
    from_snapshot:
        Name of a portable snapshot to ``restore`` into *storage_path* before
        starting — i.e. boot from a saved ledger configuration.  Requires
        *storage_path*.
    ready_timeout:
        Seconds to wait for the REST API to answer before giving up on a port.
    binary:
        Path to ``aleo-devnode``.  Defaults to ``$ALEO_DEVNODE_BIN`` or the
        binary on ``PATH``.
    verbosity:
        ``-v`` level passed to the node (default ``1``).
    """

    def __init__(
        self,
        *,
        private_key: str = DEVNODE_PRIVATE_KEY,
        port: int | None = None,
        storage_path: str | None = None,
        manual_block_creation: bool = True,
        from_snapshot: str | None = None,
        ready_timeout: float = 60.0,
        binary: str | None = None,
        verbosity: int = 1,
    ) -> None:
        self.private_key = private_key
        self._explicit_port = port
        self.port: int = port if port is not None else _free_port()
        self.storage_path = storage_path
        self.manual_block_creation = manual_block_creation
        self.from_snapshot = from_snapshot
        self.ready_timeout = ready_timeout
        self.verbosity = verbosity
        self.binary = (
            binary or os.environ.get("ALEO_DEVNODE_BIN") or shutil.which("aleo-devnode")
        )
        self._proc: subprocess.Popen[bytes] | None = None
        self._log: collections.deque[str] = collections.deque(maxlen=500)
        self._drainer: threading.Thread | None = None

    # ── URLs ────────────────────────────────────────────────────────────────

    @property
    def socket_addr(self) -> str:
        return f"127.0.0.1:{self.port}"

    @property
    def base_url(self) -> str:
        return f"http://{self.socket_addr}"

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def start(self) -> "Devnode":
        """Launch the node and block until its REST API is ready.

        Auto-allocated ports are rerolled on collision; a pinned port is tried
        once.  Raises :class:`DevnodeError` if the binary is missing or the node
        never becomes ready.
        """
        if not self.binary:
            raise DevnodeError(
                "aleo-devnode not found — set ALEO_DEVNODE_BIN or add it to PATH "
                "(https://github.com/ProvableHQ/aleo-devnode)."
            )
        retries = 1 if self._explicit_port is not None else _DEFAULT_START_RETRIES
        last_logs = ""
        for _ in range(retries):
            if self.from_snapshot:
                self._restore_snapshot()
            self._spawn()
            try:
                self._wait_ready()
                return self
            except DevnodeError:
                last_logs = "\n".join(self._log)
                self._terminate()
                if self._explicit_port is None:
                    self.port = _free_port()  # reroll and retry
                    continue
                raise
        raise DevnodeError(
            f"devnode did not become ready after {retries} attempts.\n"
            f"Last logs:\n{last_logs}"
        )

    def _restore_snapshot(self) -> None:
        if not self.storage_path:
            raise DevnodeError("from_snapshot requires storage_path")
        assert self.binary is not None
        subprocess.run(
            [self.binary, "restore", "--snapshot", self.from_snapshot or "",
             "--storage", self.storage_path],
            check=True,
        )

    def _spawn(self) -> None:
        assert self.binary is not None
        args: list[str] = [
            self.binary, "start",
            "--private-key", self.private_key,
            "--socket-addr", self.socket_addr,
            "--verbosity", str(self.verbosity),
        ]
        if self.storage_path:
            args += ["--storage", self.storage_path]
        if self.manual_block_creation:
            args += ["--manual-block-creation"]
        # Capture output through a pipe drained by a background thread: a raw
        # PIPE left unread would deadlock the node once the OS buffer fills.
        self._proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        self._log.clear()
        self._drainer = threading.Thread(target=self._drain, daemon=True)
        self._drainer.start()

    def _drain(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for raw in iter(proc.stdout.readline, b""):
            self._log.append(raw.decode("utf-8", "replace").rstrip("\n"))

    def _wait_ready(self) -> None:
        import httpx

        url = f"{self.base_url}/testnet/block/height/latest"
        deadline = time.monotonic() + self.ready_timeout
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise DevnodeError(
                    f"devnode process exited early (code {self._proc.returncode})"
                )
            try:
                if httpx.get(url, timeout=2.0).status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.5)
        raise DevnodeError(f"devnode not ready at {url} within {self.ready_timeout}s")

    def stop(self) -> None:
        """Gracefully shut the node down (REST shutdown, then terminate)."""
        try:
            import httpx

            httpx.post(f"{self.base_url}/testnet/shutdown", timeout=5.0)
        except Exception:
            pass
        self._terminate()

    def _terminate(self) -> None:
        proc = self._proc
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        self._proc = None

    # ── REST control plane ──────────────────────────────────────────────────

    def advance(self, num_blocks: int = 1) -> None:
        """Produce *num_blocks* blocks (``POST /testnet/block/create``)."""
        import httpx

        resp = httpx.post(
            f"{self.base_url}/testnet/block/create",
            json={"num_blocks": num_blocks},
            timeout=30.0,
        )
        resp.raise_for_status()

    def snapshot(self, name: str | None = None) -> Any:
        """Take a ledger snapshot (``POST /testnet/snapshot``)."""
        import httpx

        body: dict[str, str] = {"name": name} if name else {}
        resp = httpx.post(f"{self.base_url}/testnet/snapshot", json=body, timeout=30.0)
        resp.raise_for_status()
        return resp.json()

    def list_snapshots(self) -> Any:
        """List available snapshots (``GET /testnet/snapshots``)."""
        import httpx

        resp = httpx.get(f"{self.base_url}/testnet/snapshots", timeout=10.0)
        resp.raise_for_status()
        return resp.json()

    def logs(self) -> list[str]:
        """Return the most recent captured log lines from the node."""
        return list(self._log)

    # ── Accessors ───────────────────────────────────────────────────────────

    @property
    def aleo(self) -> Any:
        """An :class:`~aleo.facade.client.Aleo` client bound to this node."""
        from aleo import Aleo

        return Aleo(Aleo.HTTPProvider(self.base_url, network="testnet"))

    @property
    def accounts(self) -> list[Any]:
        """The deterministic pre-funded genesis accounts (``w3.eth.accounts`` analog)."""
        aleo = self.aleo
        return [aleo.account.from_private_key(pk) for pk, _addr in DEFAULT_ACCOUNTS]

    # ── Context manager ─────────────────────────────────────────────────────

    def __enter__(self) -> "Devnode":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def __repr__(self) -> str:
        state = "running" if self._proc is not None else "stopped"
        return f"Devnode({self.socket_addr}, {state})"

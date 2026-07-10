"""Aleo — facade network module (F3).

Attached to the :class:`~aleo.facade.client.Aleo` client as ``aleo.network``.
Wraps :class:`~aleo.network_client.AleoNetworkClient` with typed pass-throughs,
a ``submit_transaction`` verb that accepts both ``Transaction`` objects and raw
strings, and a ``wait_for_transaction`` helper that raises
:exc:`~aleo.facade.errors.TransactionConfirmationTimeout` on timeout.
"""
from __future__ import annotations

from typing import Any

from .errors import TransactionConfirmationTimeout


class NetworkModule:
    """Namespaced network operations attached to an :class:`~aleo.facade.client.Aleo` client.

    Access via ``aleo.network``, not by direct construction.

    Parameters
    ----------
    client:
        The parent :class:`~aleo.facade.client.Aleo` instance.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def __repr__(self) -> str:
        return f"NetworkModule(network={self._client._provider.network!r})"

    # ── Internal helpers ───────────────────────────────────────────────────

    def _nc(self) -> Any:
        """Return the underlying :class:`~aleo.network_client.AleoNetworkClient`."""
        return self._client._client

    # ── Block read pass-throughs ───────────────────────────────────────────

    def get_latest_height(self) -> int:
        """Return the latest block height.

        Returns
        -------
        int
            The current chain height.
        """
        return int(self._nc().get_latest_height())

    def get_latest_block(self) -> Any:
        """Return the latest block object (raw JSON dict).

        Returns
        -------
        Any
            Latest block as returned by the node.
        """
        return self._nc().get_latest_block()

    def get_block(self, height: int) -> Any:
        """Return the block at *height*.

        Parameters
        ----------
        height:
            Block height (non-negative integer).

        Returns
        -------
        Any
            Block data dict.
        """
        return self._nc().get_block(height)

    def get_block_by_hash(self, block_hash: str) -> Any:
        """Return the block identified by *block_hash*.

        Parameters
        ----------
        block_hash:
            Hex block hash string.

        Returns
        -------
        Any
            Block data dict.
        """
        return self._nc().get_block_by_hash(block_hash)

    def get_block_range(self, start: int, end: int) -> list[Any]:
        """Return blocks in the range [*start*, *end*].

        Parameters
        ----------
        start:
            Inclusive start height.
        end:
            Inclusive end height.

        Returns
        -------
        list[Any]
            Sequence of block dicts.
        """
        return self._nc().get_block_range(start, end)

    def get_latest_block_hash(self) -> str:
        """Return the hash of the latest block.

        Returns
        -------
        str
            Block hash string.
        """
        return str(self._nc().get_latest_block_hash())

    def get_latest_committee(self) -> Any:
        """Return the latest committee data.

        Returns
        -------
        Any
            Committee data as returned by the node.
        """
        return self._nc().get_latest_committee()

    def get_committee_by_height(self, height: int) -> Any:
        """Return the committee at *height*.

        Parameters
        ----------
        height:
            Block height.

        Returns
        -------
        Any
            Committee data dict.
        """
        return self._nc().get_committee_by_height(height)

    def get_state_root(self) -> str:
        """Return the latest state root.

        Returns
        -------
        str
            State root string (``"sr1…"``).
        """
        return str(self._nc().get_state_root())

    def get_state_paths(self, commitments: list[str]) -> list[Any]:
        """Return state paths for the given *commitments*.

        Parameters
        ----------
        commitments:
            List of commitment strings.

        Returns
        -------
        list[Any]
            State path objects.
        """
        return self._nc().get_state_paths(commitments)

    # ── Program read pass-throughs ─────────────────────────────────────────

    def get_program(self, program_id: str, edition: int | None = None) -> str:
        """Return the Leo source for *program_id*.

        Parameters
        ----------
        program_id:
            Aleo program identifier (e.g. ``"credits.aleo"``).
        edition:
            Optional edition number.

        Returns
        -------
        str
            Program source text.
        """
        return self._nc().get_program(program_id, edition)

    def get_latest_program_edition(self, program_id: str) -> int:
        """Return the latest edition number of *program_id*.

        Parameters
        ----------
        program_id:
            Aleo program identifier.

        Returns
        -------
        int
            Latest edition number.
        """
        return int(self._nc().get_latest_program_edition(program_id))

    def get_program_amendment_count(self, program_id: str) -> Any:
        """Return the amendment count for *program_id*.

        Parameters
        ----------
        program_id:
            Aleo program identifier.

        Returns
        -------
        Any
            Amendment count (raw value as returned by the node).
        """
        return self._nc().get_program_amendment_count(program_id)

    def get_program_mapping_names(self, program_id: str) -> list[str]:
        """Return the mapping names defined in *program_id*.

        Parameters
        ----------
        program_id:
            Aleo program identifier.

        Returns
        -------
        list[str]
            List of mapping names.
        """
        return self._nc().get_program_mapping_names(program_id)

    def get_program_mapping_value(
        self, program_id: str, mapping_name: str, key: str
    ) -> str:
        """Return the current value at (*program_id*, *mapping_name*, *key*).

        Parameters
        ----------
        program_id:
            Aleo program identifier.
        mapping_name:
            Name of the mapping.
        key:
            Mapping key.

        Returns
        -------
        str
            Serialised mapping value.
        """
        return self._nc().get_program_mapping_value(program_id, mapping_name, key)

    def get_public_balance(self, address: str) -> int:
        """Return the public credits balance for *address* in microcredits.

        Queries the ``credits.aleo`` ``account`` mapping.  Returns ``0`` when
        the address has no balance or the mapping value is absent.

        Parameters
        ----------
        address:
            Aleo address string (``"aleo1…"``).

        Returns
        -------
        int
            Balance in microcredits.
        """
        return int(self._nc().get_public_balance(address))

    # ── Transaction read pass-throughs ─────────────────────────────────────

    def get_transaction(self, tx_id: str) -> Any:
        """Return the (possibly unconfirmed) transaction for *tx_id*.

        Parameters
        ----------
        tx_id:
            Transaction ID string.

        Returns
        -------
        Any
            Transaction data dict.
        """
        return self._nc().get_transaction(tx_id)

    def get_confirmed_transaction(self, tx_id: str) -> Any:
        """Return the confirmed transaction for *tx_id*.

        Parameters
        ----------
        tx_id:
            Transaction ID string.

        Returns
        -------
        Any
            Confirmed transaction data dict.
        """
        return self._nc().get_confirmed_transaction(tx_id)

    def get_transactions(self, block_height: int) -> list[Any]:
        """Return all transactions in the block at *block_height*.

        Parameters
        ----------
        block_height:
            Block height to query.

        Returns
        -------
        list[Any]
            List of transaction dicts.
        """
        return self._nc().get_transactions(block_height)

    def get_transactions_in_mempool(self) -> list[Any]:
        """Return transactions currently in the memory pool.

        Returns
        -------
        list[Any]
            List of unconfirmed transaction dicts.
        """
        return self._nc().get_transactions_in_mempool()

    def get_transition_id(self, input_or_output_id: str) -> str:
        """Return the transition ID for the given input or output ID.

        Parameters
        ----------
        input_or_output_id:
            An input or output commitment/serial-number string.

        Returns
        -------
        str
            Transition ID string.
        """
        return str(self._nc().get_transition_id(input_or_output_id))

    def get_deployment_transaction_id_for_program(self, program_id: str) -> str:
        """Return the deployment transaction ID for *program_id*.

        Parameters
        ----------
        program_id:
            Aleo program identifier.

        Returns
        -------
        str
            Transaction ID string.
        """
        return str(self._nc().get_deployment_transaction_id_for_program(program_id))

    def get_deployment_transaction_for_program(self, program_id: str) -> Any:
        """Return the deployment transaction for *program_id*.

        Parameters
        ----------
        program_id:
            Aleo program identifier.

        Returns
        -------
        Any
            Transaction data dict.
        """
        return self._nc().get_deployment_transaction_for_program(program_id)

    # ── Transaction broadcast ──────────────────────────────────────────────

    def submit_transaction(self, transaction: Any) -> str:
        """Broadcast *transaction* to the network and return the transaction ID.

        Accepts either a :class:`Transaction` object (anything with a
        ``__str__`` serialisation the node accepts) or a raw JSON string.

        Parameters
        ----------
        transaction:
            A :class:`Transaction` object or a serialised transaction string.

        Returns
        -------
        str
            The transaction ID returned by the node.
        """
        return str(self._nc().submit_transaction(transaction))

    # ``send_raw_transaction`` is the web3.py-parity alias — same callable.
    send_raw_transaction = submit_transaction

    # ── Wait for confirmation ──────────────────────────────────────────────

    def wait_for_transaction(
        self,
        tx_id: str,
        *,
        timeout: float = 45.0,
        poll_interval: float = 2.0,
    ) -> Any:
        """Poll until *tx_id* is confirmed, then return the transaction data.

        Delegates to
        :meth:`~aleo.network_client.AleoNetworkClient.wait_for_transaction_confirmation`
        and maps its :exc:`TimeoutError` to the facade-typed
        :exc:`~aleo.facade.errors.TransactionConfirmationTimeout`.

        Parameters
        ----------
        tx_id:
            Transaction ID to wait on.
        timeout:
            Maximum seconds to wait before raising
            :exc:`~aleo.facade.errors.TransactionConfirmationTimeout`.
            Default is 45 s.
        poll_interval:
            Seconds between polls.  Default is 2 s.

        Returns
        -------
        Any
            Confirmed transaction data dict.

        Raises
        ------
        TransactionConfirmationTimeout
            If the transaction is not confirmed within *timeout* seconds.
        AleoNetworkError
            If the node explicitly rejects the transaction.
        """
        try:
            return self._nc().wait_for_transaction_confirmation(
                tx_id,
                check_interval=poll_interval,
                timeout=timeout,
            )
        except TimeoutError:
            raise TransactionConfirmationTimeout(tx_id, timeout)


__all__ = ["NetworkModule"]

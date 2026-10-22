"""Compliance proofs for Shield Swap calls.

Shield Swap checks two kinds of freeze list. Its own
(``shield_swap_freezelist.aleo``) screens the signer, a position's recipient,
and its withdrawal address when a position is minted, collected, or
rebalanced, and the signer when a swap output is claimed. Each ARC-22
stablecoin that the AMM trades through a wrapper (USDCx, USAD) screens the
signer when its records are wrapped and the receiving address when they are
unwrapped. Every check is one ``[MerkleProof; 2]`` argument — a *bundle* —
and the routers take up to six per call.

The write methods on :class:`~aleo_shield_swap.ShieldSwap` take one
``compliance_proofs`` argument:

- ``None`` (the default): the SDK works out which bundles the selected
  entrypoint needs and which address each one screens, reads each list's
  current root, and builds the bundles — the empty-tree pair while a list is
  empty, a verified witness from the published tree otherwise. Witnesses are
  cached on the client until the list's root changes.
- A tuple of bundles in the entrypoint's input order, each a literal string,
  a :class:`~aleo.compliance.MerkleProofPair`, or a pair of
  :class:`~aleo.compliance.MerkleProofInput`. The count must match the route the
  call resolves to; the type aliases below spell out the possible counts.

The planners in this module are pure: they turn a route's shape into the
ordered list of :class:`Slot` values a call must fill, following
``MerkleProofReference.md`` in the contract repository.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence, Union

from aleo.compliance import (
    EMPTY_MERKLE_PROOF_PAIR,
    FreezeListError,
    MerkleProofInput,
    MerkleProofPair,
    ComplianceProof,
    ComplianceProofContext,
    literal_compliance_proof,
)

SHIELD_SWAP_FREEZELIST = "shield_swap_freezelist.aleo"

_B = ComplianceProof
#: ``swap``: one sender bundle when the input token is wrapped, none otherwise.
SwapProofs = Union[tuple[()], tuple[_B]]
#: ``claim_swap_output``: the AMM signer bundle, then a receiver bundle per wrapped leg paid (output, then refund).
ClaimProofs = Union[tuple[_B], tuple[_B, _B], tuple[_B, _B, _B]]
#: ``mint``: a sender bundle per wrapped side (token0 first), then AMM signer, recipient, withdrawal.
MintProofs = Union[tuple[_B, _B, _B], tuple[_B, _B, _B, _B], tuple[_B, _B, _B, _B, _B]]
#: ``increase_liquidity``: a sender bundle per wrapped side, token0 first.
IncreaseProofs = Union[tuple[()], tuple[_B], tuple[_B, _B]]
#: ``collect``: AMM owner and withdrawal bundles, then a receiver bundle per wrapped side.
CollectProofs = Union[tuple[_B, _B], tuple[_B, _B, _B], tuple[_B, _B, _B, _B]]
#: ``rebalance_position``: a sender bundle per funded wrapped side, a receiver bundle per wrapped side, then AMM owner and withdrawal.
RebalanceProofs = Union[tuple[_B, _B], tuple[_B, _B, _B], tuple[_B, _B, _B, _B],
                        tuple[_B, _B, _B, _B, _B], tuple[_B, _B, _B, _B, _B, _B]]
#: The ``compliance_proofs`` argument of any Shield Swap write: the method's tuple of proofs in
#: input order, or ``None`` to have the SDK build them.
ComplianceProofs = Optional[Union[SwapProofs, ClaimProofs, MintProofs, IncreaseProofs, CollectProofs, RebalanceProofs]]

SlotKind = Literal["amm", "wrapper"]


@dataclass(frozen=True)
class Slot:
    """One ``[MerkleProof; 2]`` argument of a Shield Swap entrypoint.

    Attributes:
        name: The entrypoint's parameter name (``sender_proof0``,
            ``amm_withdrawal_merkle_proofs``), used in error messages.
        kind: ``"amm"`` for Shield Swap's own list, ``"wrapper"`` for the
            underlying stablecoin's list of a wrapped token.
        subject: The address the argument proves absent.
        token_id: The wrapped token whose underlying list applies; ``None``
            for AMM slots.
    """

    name: str
    kind: SlotKind
    subject: str
    token_id: Optional[str] = None


def _amm(name: str, subject: str) -> Slot:
    return Slot(name, "amm", subject)


def _wrapper(name: str, subject: str, token_id: str) -> Slot:
    return Slot(name, "wrapper", subject, str(token_id))


# ── Planners (one per entrypoint family; order = the deployed input order) ──

def swap_slots(input_wrapped: bool, signer: str, token_in_id: str) -> list[Slot]:
    """``swap`` / ``swap_multi_hop`` and their ``*_from_wrapped`` router forms."""
    return [_wrapper("sender_proof", signer, token_in_id)] if input_wrapped else []


def claim_slots(output_wrapped: bool, refund_wrapped: bool, no_refund: bool,
                signer: str, token_in_id: str, token_out_id: str) -> list[Slot]:
    """``claim_swap_output`` and the router's ``claim_to_*`` forms: the signer receives everything."""
    slots = [_amm("merkle_proofs", signer)]
    if output_wrapped:
        slots.append(_wrapper("out_receiver_proof" if (refund_wrapped and not no_refund) else "receiver_proof",
                              signer, token_out_id))
    if refund_wrapped and not no_refund:
        slots.append(_wrapper("refund_receiver_proof" if output_wrapped else "receiver_proof", signer, token_in_id))
    return slots


def mint_slots(w0: bool, w1: bool, signer: str, recipient: str, withdrawal: str,
               token0_id: str, token1_id: str) -> list[Slot]:
    """``mint`` and ``mint_from_*``: wrapped senders (token0, token1), then the three AMM roles."""
    slots: list[Slot] = []
    if w0:
        slots.append(_wrapper("sender_proof0", signer, token0_id))
    if w1:
        slots.append(_wrapper("sender_proof1", signer, token1_id))
    prefix = "amm_" if (w0 or w1) else ""
    slots += [_amm(f"{prefix}signer_merkle_proofs", signer),
              _amm(f"{prefix}recipient_merkle_proofs", recipient),
              _amm(f"{prefix}withdrawal_merkle_proofs", withdrawal)]
    return slots


def increase_slots(w0: bool, w1: bool, signer: str, token0_id: str, token1_id: str) -> list[Slot]:
    """``increase_liquidity`` and ``increase_from_*``: one sender bundle per wrapped side."""
    slots: list[Slot] = []
    if w0:
        slots.append(_wrapper("sender_proof0", signer, token0_id))
    if w1:
        slots.append(_wrapper("sender_proof1", signer, token1_id))
    return slots


def collect_slots(w0: bool, w1: bool, signer: str, withdrawal: str,
                  token0_id: str, token1_id: str) -> list[Slot]:
    """``collect`` and ``collect_to_*``: AMM owner and withdrawal, then a receiver per wrapped side.

    Receiver bundles screen the position's immutable withdrawal address, not
    the signer: that is who the unwrapped tokens are paid to.
    """
    prefix = "amm_" if (w0 or w1) else ""
    slots = [_amm(f"{prefix}owner_merkle_proofs", signer), _amm(f"{prefix}withdrawal_merkle_proofs", withdrawal)]
    if w0:
        slots.append(_wrapper("receiver_proof0", withdrawal, token0_id))
    if w1:
        slots.append(_wrapper("receiver_proof1", withdrawal, token1_id))
    return slots


def rebalance_slots(w0: bool, w1: bool, funded0: bool, funded1: bool, signer: str, withdrawal: str,
                    token0_id: str, token1_id: str) -> list[Slot]:
    """``rebalance_<side0>_<side1>_<mode>``: funded wrapped senders, wrapped receivers, then AMM owner and withdrawal.

    A sender bundle follows the funding record of each wrapped side that
    takes funding; receiver bundles screen the withdrawal address against
    every wrapped side's list, refund or not.
    """
    slots: list[Slot] = []
    one_sided = (w0 == w1) and (funded0 != funded1)          # the router's ``*_one`` form names a single ``sender_proof``
    if funded0 and w0:
        slots.append(_wrapper("sender_proof" if one_sided else "sender_proof0", signer, token0_id))
    if funded1 and w1:
        slots.append(_wrapper("sender_proof" if one_sided else "sender_proof1", signer, token1_id))
    if w0:
        slots.append(_wrapper("receiver_proof0", withdrawal, token0_id))
    if w1:
        slots.append(_wrapper("receiver_proof1", withdrawal, token1_id))
    slots += [_amm("owner_proofs", signer), _amm("withdrawal_proofs", withdrawal)]
    return slots


# ── Filling the slots ────────────────────────────────────────────────────────

ListResolver = Callable[[str], Optional[str]]
"""Maps a wrapped token id to the freeze-list program its wrapper checks (``None`` when it ignores the proof)."""


def supplied_literals(action: str, supplied: Sequence[ComplianceProof], slots: Sequence[Slot]) -> list[str]:
    """Validate caller-supplied bundles against the planned slots and return their literals in order.

    Raises:
        ValueError: The count does not match the entrypoint's slots (the
            message lists them), or a value is not a proof bundle.
    """
    if _looks_like_one_bundle(supplied):
        supplied = (supplied,)  # type: ignore[assignment]
    if len(supplied) != len(slots):
        expected = ", ".join(f"{s.name} ({s.kind}: {s.subject[:10]}…)" for s in slots) or "none"
        raise ValueError(f"{action} on this route takes {len(slots)} compliance proof bundle(s) "
                         f"[{expected}]; received {len(supplied)}")
    try:
        return [_literal(b) for b in supplied]
    except FreezeListError as exc:
        raise ValueError(f"{action}: {exc}") from exc


def _literal(bundle: ComplianceProof) -> str:
    literal = literal_compliance_proof(bundle)
    assert literal is not None          # a bundle is never None
    return literal


def _looks_like_one_bundle(value: Sequence[ComplianceProof]) -> bool:
    return len(value) == 2 and all(isinstance(p, MerkleProofInput) for p in value)


def planned_proof_slots(slots: Sequence[Slot], amm_list: Optional[str], list_for_token: ListResolver) -> list[ComplianceProofContext]:
    """The :class:`aleo.compliance.ComplianceProofContext` per planned slot: which list and which address."""
    out: list[ComplianceProofContext] = []
    for slot in slots:
        if slot.kind == "amm":
            out.append(ComplianceProofContext(amm_list, slot.subject))
        else:
            assert slot.token_id is not None
            out.append(ComplianceProofContext(list_for_token(slot.token_id), slot.subject))
    return out


def merge_legacy_wrapper_proofs(slots: Sequence[Slot], resolved: Sequence[str], wrapper_proofs: str) -> list[str]:
    """Put the caller's one legacy ``wrapper_proofs`` literal into every wrapper slot, keeping resolved AMM bundles."""
    literal = _literal(wrapper_proofs)
    return [literal if slot.kind == "wrapper" else proof for slot, proof in zip(slots, resolved)]


def check_proof_arguments(compliance_proofs: object, wrapper_proofs: Optional[str]) -> None:
    """Reject ``compliance_proofs`` and the legacy ``wrapper_proofs`` given together."""
    if compliance_proofs is not None and wrapper_proofs is not None:
        raise ValueError("Pass compliance_proofs= or wrapper_proofs=, not both")


__all__ = [
    "EMPTY_MERKLE_PROOF_PAIR", "SHIELD_SWAP_FREEZELIST",
    "ClaimProofs", "CollectProofs", "IncreaseProofs", "MintProofs", "RebalanceProofs", "SwapProofs",
    "MerkleProofInput", "MerkleProofPair", "ComplianceProof", "ComplianceProofs", "Slot",
    "check_proof_arguments", "claim_slots", "collect_slots", "increase_slots", "merge_legacy_wrapper_proofs",
    "mint_slots", "planned_proof_slots", "rebalance_slots", "supplied_literals", "swap_slots",
]

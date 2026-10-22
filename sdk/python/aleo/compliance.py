"""Freeze-list exclusion proofs for ARC-22 compliant programs.

Programs that enforce a freeze list (the USDCx stablecoin, Shield Swap) take
a ``[MerkleProof; 2]`` argument proving that an address is *not* on the list.
This module builds those arguments and caches them per client:

- :func:`generate_freezelist_proof` turns a published tree into the proof
  literal for one address and checks every witness against the root.
- :class:`FreezeListProofs` / :class:`AsyncFreezeListProofs` read the current
  root, fetch the tree only when the root changed, and reuse witnesses while
  the root is unchanged. While a list is empty the fixed empty-tree pair is
  returned without fetching anything beyond the root.
- :func:`freezelist_program` finds the freeze-list program a token program
  checks, so callers can name the token and let the SDK find the list.
- :func:`literal_compliance_proof` normalises a caller-supplied
  ``compliance_proofs`` argument to its literal, and
  :func:`format_merkle_proof_pair` encodes a structured pair.

Tree layout (shared with the TypeScript SDK and the compliance API): leaves
are frozen addresses as little-endian field integers, sorted ascending and
front-padded with ``0field`` to a power of two (minimum two). The leaf level
hashes ``Poseidon4([1field, l, r])``, inner levels ``Poseidon4([0field, l, r])``.
A sibling path starts with the leaf itself, then one sibling per level,
zero-padded to 16 entries.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence, cast

from ._client_common import AleoError, AleoNetworkError

ZERO_ADDRESS = "aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc"
DEFAULT_DEPTH = 15                    # tree capacity 2**(depth-1) leaves
PROOF_SIBLINGS = 16                   # struct MerkleProof { siblings: [field; 16u32], leaf_index: u32 }
FREEZE_LIST_MAPPING = "freeze_list"                       # address => bool (frozen flag)
FREEZE_LIST_LAST_INDEX_MAPPING = "freeze_list_last_index"  # bool => u32, keyed "true"
FREEZE_LIST_INDEX_MAPPING = "freeze_list_index"           # u32 => address ([0u32] = zero-address sentinel)
FREEZE_LIST_ROOT_MAPPING = "freeze_list_root"             # u8 => field, "1u8" current / "2u8" previous
CURRENT_ROOT_KEY = "1u8"
EMPTY_TREE_ROOT = 3642222252059314292809609689035560016959342421640560347114299934615987159853
EMPTY_MERKLE_PROOF = "{ siblings: [" + ", ".join(["0field"] * PROOF_SIBLINGS) + "], leaf_index: 1u32 }"
EMPTY_MERKLE_PROOF_PAIR = f"[{EMPTY_MERKLE_PROOF}, {EMPTY_MERKLE_PROOF}]"
FREEZELIST_SUFFIX = "freezelist.aleo"

#: Token programs whose freeze list is known without reading their imports.
#: ``None`` marks a program that takes the proof argument and ignores it.
KNOWN_TOKEN_FREEZELISTS: dict[str, str | None] = {
    "credits.aleo": None,
    "usdcx_stablecoin.aleo": "usdcx_freezelist.aleo",
    "test_usdcx_stablecoin.aleo": "test_usdcx_freezelist.aleo",
}

MAX_CACHED_LISTS = 16
MAX_CACHED_ADDRESSES = 256


class FreezeListError(AleoError):
    """A freeze-list proof could not be resolved; the message names what to pass instead."""


class FrozenAddressError(FreezeListError):
    """The address is on the freeze list, so no exclusion proof exists for it."""


# ── Tree math ────────────────────────────────────────────────────────────────

def address_to_field_int(address: str) -> int:
    """The field element a Leo program sees for an address: its bech32m payload read little-endian."""
    from .mainnet import Address
    try:
        return int.from_bytes(bytes(Address.from_string(address).to_bytes_le()), "little")
    except Exception as exc:
        raise FreezeListError(f"Not an Aleo address: {address!r}") from exc


def hash_two(prefix: str, left: str, right: str, network: str) -> str:
    """``Poseidon4(Plaintext("[prefix,left,right]").to_fields())`` as a ``…field`` literal."""
    net = _network_module(network)
    plaintext = net.Plaintext.from_string(f"[{prefix},{left},{right}]")
    return str(net.Poseidon4().hash(plaintext.to_fields()))


def generate_leaves(addresses: Iterable[str], depth: int = DEFAULT_DEPTH) -> list[str]:
    """Frozen addresses as sorted ``…field`` leaves: zero address dropped, front-padded to a power of two (minimum two)."""
    live = [a for a in addresses if a != ZERO_ADDRESS]
    max_leaves = 2 ** (depth - 1)
    if len(live) > max_leaves:
        raise FreezeListError(f"Leaves limit exceeded. Max: {max_leaves}, provided: {len(live)}")
    count = 2 if len(live) <= 1 else 2 ** math.ceil(math.log2(len(live)))
    fields = sorted(address_to_field_int(a) for a in live)
    return ["0field"] * (count - len(fields)) + [f"{f}field" for f in fields]


def build_tree(leaves: Sequence[str], network: str) -> list[int]:
    """The bottom-up tree as integers: leaves first, root last (the layout the compliance API publishes)."""
    if not leaves:
        raise FreezeListError("Leaves array cannot be empty")
    if len(leaves) % 2:
        raise FreezeListError("Leaves array must have even number of elements")
    tree = list(leaves)
    level = list(leaves)
    while len(level) > 1:
        prefix = "1field" if len(level) == len(leaves) else "0field"
        level = [hash_two(prefix, level[i], level[i + 1], network) for i in range(0, len(level), 2)]
        tree.extend(level)
    return [int(node[: -len("field")]) for node in tree]


def leaf_indices(tree: Sequence[int], address: str) -> tuple[int, int]:
    """The (left, right) leaf indices bracketing *address*: the two leaves a non-inclusion proof opens."""
    count = (len(tree) + 1) // 2
    target = address_to_field_int(address)
    leaves = tree[:count]
    right = next((i for i, leaf in enumerate(leaves) if target <= leaf), -1)
    left = right - 1
    if right == -1:
        right = left = count - 1
    if right == 0:
        left = 0
    return left, right


def sibling_path(tree: Sequence[int], index: int, depth: int = DEFAULT_DEPTH) -> list[int]:
    """The leaf at *index*, then its sibling at each level, zero-padded to *depth* entries."""
    count = (len(tree) + 1) // 2
    path = [tree[index]]
    node, parent, level = index, count, 1
    while parent < len(tree):
        sibling = node + 1 if node % 2 == 0 else node - 1
        path.append(tree[sibling])
        node = parent + index // 2 ** level
        parent += count // 2 ** level
        level += 1
    while len(path) < depth:
        path.append(0)
    return path


def format_merkle_proof(left: tuple[Sequence[int | str], int | str],
                        right: tuple[Sequence[int | str], int | str]) -> str:
    """A ``[MerkleProof; 2]`` literal from two raw ``(siblings, leaf_index)`` pairs.

    The typed form is :func:`format_merkle_proof_pair`.
    """
    return format_merkle_proof_pair((MerkleProofInput(tuple(left[0]), left[1]),
                                     MerkleProofInput(tuple(right[0]), right[1])))


# ── Proof values ─────────────────────────────────────────────────────────────

def _field_value(value: int | str) -> int:
    """An integer field element from an ``int`` or a ``"…field"`` literal."""
    if isinstance(value, int):
        return value
    text = value.strip()
    if not text.endswith("field") or not text[:-5].isdigit():
        raise FreezeListError(f"Not a field literal: {value!r}")
    return int(text[:-5])


def _u32_value(value: int | str) -> int:
    """An integer from an ``int`` or a ``"Nu32"`` literal."""
    if isinstance(value, int):
        return value
    text = value.strip()
    if not text.endswith("u32") or not text[:-3].isdigit():
        raise FreezeListError(f"Not a u32 literal: {value!r}")
    return int(text[:-3])


@dataclass(frozen=True, init=False)
class MerkleProofInput:
    """One Merkle path of an exclusion proof, as the contracts take it.

    ``siblings`` may be given as integers or as ``"…field"`` literals (the
    TypeScript SDK's form) and ``leaf_index`` as an ``int`` or ``"Nu32"``;
    both are stored normalised, so equal paths compare equal however they
    were built. :attr:`sibling_literals` gives the siblings back as literals.

    Attributes:
        siblings: The leaf followed by one sibling per level, 16 field values
            in all; unused levels are zero.
        leaf_index: The leaf's position in the tree, a ``u32``.
    """

    siblings: tuple[int, ...]
    leaf_index: int

    def __init__(self, siblings: Sequence[int | str], leaf_index: int | str) -> None:
        values = tuple(_field_value(s) for s in siblings)
        if len(values) != PROOF_SIBLINGS:
            raise FreezeListError(f"A MerkleProof needs {PROOF_SIBLINGS} siblings, got {len(values)}")
        index = _u32_value(leaf_index)
        if index < 0 or any(v < 0 for v in values):
            raise FreezeListError("MerkleProof values must be non-negative")
        object.__setattr__(self, "siblings", values)
        object.__setattr__(self, "leaf_index", index)

    @property
    def sibling_literals(self) -> tuple[str, ...]:
        """The siblings as ``"…field"`` literals."""
        return tuple(f"{s}field" for s in self.siblings)

    def __str__(self) -> str:
        return "{ siblings: [" + ", ".join(self.sibling_literals) + f"], leaf_index: {self.leaf_index}u32 }}"

    @classmethod
    def from_string(cls, text: str) -> "MerkleProofInput":
        """Parse one ``{ siblings: [..], leaf_index: Nu32 }`` literal."""
        body = text.strip()
        if not (body.startswith("{") and body.endswith("}")):
            raise FreezeListError(f"Not a MerkleProof literal: {text[:40]!r}")
        try:
            inner = body[1:-1]
            start, end = inner.index("["), inner.index("]")
            siblings = tuple(int(s.strip().removesuffix("field")) for s in inner[start + 1:end].split(","))
            index_text = inner[end + 1:].split("leaf_index:", 1)[1].strip().rstrip(",").strip()
            leaf_index = int(index_text.removesuffix("u32"))
        except (ValueError, IndexError) as exc:
            raise FreezeListError(f"Not a MerkleProof literal: {text[:40]!r}") from exc
        return cls(siblings, leaf_index)


@dataclass(frozen=True)
class MerkleProofPair:
    """The two paths that together prove one address is absent from one list.

    Attributes:
        left: The path to the leaf just below the address (or the repeated boundary leaf).
        right: The path to the leaf just above the address.
    """

    left: MerkleProofInput
    right: MerkleProofInput

    def __str__(self) -> str:
        return f"[{self.left}, {self.right}]"

    @classmethod
    def from_string(cls, text: str) -> "MerkleProofPair":
        """Parse a ``[MerkleProof; 2]`` literal."""
        body = text.strip()
        if not (body.startswith("[") and body.endswith("]")):
            raise FreezeListError(f"Not a [MerkleProof; 2] literal: {text[:40]!r}")
        parts: list[str] = []
        depth = 0
        current: list[str] = []
        for ch in body[1:-1]:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(ch)
        parts.append("".join(current))
        if len(parts) != 2:
            raise FreezeListError(f"A [MerkleProof; 2] literal holds two proofs, found {len(parts)}")
        return cls(MerkleProofInput.from_string(parts[0]), MerkleProofInput.from_string(parts[1]))


#: One compliance proof argument as a caller may supply it: the ``[MerkleProof; 2]``
#: literal, a :class:`MerkleProofPair`, or the two :class:`MerkleProofInput` paths.
ProofBundle = str | MerkleProofPair | tuple[MerkleProofInput, MerkleProofInput]


def format_merkle_proof_pair(pair: MerkleProofPair | tuple[MerkleProofInput, MerkleProofInput]) -> str:
    """The ``[MerkleProof; 2]`` literal the transitions take, from a structured pair."""
    if isinstance(pair, MerkleProofPair):
        return str(pair)
    if isinstance(pair, tuple) and len(pair) == 2 and all(isinstance(p, MerkleProofInput) for p in pair):  # pyright: ignore[reportUnnecessaryIsInstance]
        return str(MerkleProofPair(pair[0], pair[1]))
    raise FreezeListError("A proof pair is a MerkleProofPair or a (MerkleProofInput, MerkleProofInput) tuple")


def _bundle_literal(bundle: ProofBundle) -> str:
    """The ``[MerkleProof; 2]`` literal for one caller-supplied bundle, validated."""
    if isinstance(bundle, str):
        text = bundle.strip()
        if not (text.startswith("[") and text.endswith("]")):
            raise FreezeListError("A compliance proof must be an encoded [MerkleProof; 2] Aleo literal")
        return text
    if isinstance(bundle, MerkleProofPair) or _is_path_tuple(bundle):
        return format_merkle_proof_pair(bundle)
    raise FreezeListError("A compliance proof must be a [MerkleProof; 2] literal, a MerkleProofPair, "
                          "or a (MerkleProofInput, MerkleProofInput) tuple")


def literal_compliance_proof(compliance_proofs: Any, legacy: Any = None, *, legacy_name: str = "merkle_proof") -> str | None:
    """The one ``[MerkleProof; 2]`` literal a call was given, or ``None`` when the SDK should build it.

    Normalises the proof arguments of a call that takes exactly one bundle:
    a :data:`ProofBundle` or a one-element tuple of one under
    ``compliance_proofs``, and the literal string the call accepted before
    under its old keyword. Reads nothing from the chain.

    Raises:
        FreezeListError: Both arguments were given, the tuple does not hold
            exactly one bundle, or the value is not a proof.
    """
    if legacy is not None:
        if compliance_proofs is not None:
            raise FreezeListError(f"Pass compliance_proofs= or {legacy_name}=, not both")
        if not isinstance(legacy, str):
            raise FreezeListError(f"{legacy_name}= takes the encoded [MerkleProof; 2] literal; "
                                  "pass structured proofs through compliance_proofs=")
        return _bundle_literal(legacy)
    if compliance_proofs is None:
        return None
    if isinstance(compliance_proofs, tuple):
        items = cast(tuple[Any, ...], compliance_proofs)
        if _is_path_tuple(items):
            return _bundle_literal(cast(ProofBundle, items))
        if len(items) != 1:
            raise FreezeListError(f"This operation requires exactly one compliance proof bundle; received {len(items)}")
        return _bundle_literal(cast(ProofBundle, items[0]))
    return _bundle_literal(cast(ProofBundle, compliance_proofs))


def _is_path_tuple(value: Any) -> bool:
    """True for a ``(MerkleProofInput, MerkleProofInput)`` tuple, the structured form of one bundle."""
    if not isinstance(value, tuple):
        return False
    items = cast(tuple[Any, ...], value)
    return len(items) == 2 and all(isinstance(p, MerkleProofInput) for p in items)


# ── Witness construction and verification ────────────────────────────────────

def generate_freezelist_proof(tree: Sequence[int], address: str, root: int, network: str) -> str:
    """Build the ``[MerkleProof; 2]`` literal proving *address* is absent from *tree*.

    The published tree must be complete, sorted, and end in *root*; each
    witness is recomputed up to the root before it is returned, so a stale
    or corrupt tree never yields an argument the contract would reject after
    the fee is paid.

    Args:
        tree: Node values leaves first and root last, as the compliance API
            publishes them, or an empty list for an empty freeze list.
        address: The address to prove absent.
        root: The on-chain ``freeze_list_root`` the contract will check.
        network: ``"mainnet"`` or ``"testnet"``; selects the hash binaries.

    Raises:
        FrozenAddressError: *address* is a leaf of the tree.
        FreezeListError: The tree is malformed or does not end in *root*, or a
            witness fails to reconstruct it.
    """
    nodes = list(tree) if tree else [0, 0, EMPTY_TREE_ROOT]
    leaves = (len(nodes) + 1) // 2
    if len(nodes) != 2 * leaves - 1 or leaves < 2 or leaves > 2 ** DEFAULT_DEPTH or leaves & (leaves - 1):
        raise FreezeListError("Invalid freeze-list tree: expected a complete binary tree of 2..32768 leaves")
    if nodes[-1] != root:
        raise FreezeListError(f"Published freeze-list tree root {nodes[-1]}field does not match the on-chain root "
                              f"{root}field; the indexer may be lagging — retry, or pass the proof explicitly")
    subject = address_to_field_int(address)
    for i in range(leaves):
        if nodes[i] == subject:
            raise FrozenAddressError(f"{address} is on the freeze list; no exclusion proof exists for it")
        if i and nodes[i] < nodes[i - 1]:
            raise FreezeListError("Invalid freeze-list tree: leaves must be sorted")
    left, right = leaf_indices(nodes, address)
    paths = [MerkleProofInput(tuple(sibling_path(nodes, index, PROOF_SIBLINGS)), index) for index in (left, right)]
    for path in paths:
        if _reconstruct_root(path, network) != root:
            raise FreezeListError("Freeze-list witness does not reconstruct the on-chain root")
    return str(MerkleProofPair(paths[0], paths[1]))


def _reconstruct_root(path: MerkleProofInput, network: str) -> int:
    value = path.siblings[0]
    for level in range(1, PROOF_SIBLINGS):
        sibling = path.siblings[level]
        if level > 1 and sibling == 0:
            break
        right = (path.leaf_index >> (level - 1)) & 1
        prefix = "1field" if level == 1 else "0field"
        hashed = hash_two(prefix, f"{sibling if right else value}field", f"{value if right else sibling}field", network)
        value = int(hashed[: -len("field")])
    return value


# ── Which list a program checks ──────────────────────────────────────────────

def freezelist_from_imports(program: str, imports: Sequence[str] | None) -> str | None | _Unresolved:
    """The freeze-list program *program* checks, from its name, imports, or the known table."""
    if program.endswith(FREEZELIST_SUFFIX):
        return program
    for dep in imports or ():
        if str(dep).endswith(FREEZELIST_SUFFIX):
            return str(dep)
    if program in KNOWN_TOKEN_FREEZELISTS:
        return KNOWN_TOKEN_FREEZELISTS[program]
    return _UNRESOLVED


class _Unresolved:
    pass


_UNRESOLVED = _Unresolved()


def _unresolved_error(program: str) -> FreezeListError:
    return FreezeListError(f"{program} names no freeze-list program (no *freezelist.aleo import); "
                           "pass compliance_proofs= explicitly")


def freezelist_program(aleo: Any, program: str) -> str | None:
    """The freeze-list program that *program* checks, or ``None`` when it ignores the proof.

    A program named ``*freezelist.aleo`` is its own list. Otherwise the first
    ``*freezelist.aleo`` import wins, then :data:`KNOWN_TOKEN_FREEZELISTS`.
    Reads the program once through the facade when the table has no entry.

    Raises:
        FreezeListError: No list could be determined for *program*.
    """
    if program.endswith(FREEZELIST_SUFFIX):
        return program
    if program in KNOWN_TOKEN_FREEZELISTS:
        return KNOWN_TOKEN_FREEZELISTS[program]
    from .facade.errors import AleoError as FacadeAleoError, ProgramNotFound
    try:
        imports = list(aleo.programs.get(program).imports)
    except (ProgramNotFound, FacadeAleoError):
        imports = []
    found = freezelist_from_imports(program, imports)
    if isinstance(found, _Unresolved):
        raise _unresolved_error(program)
    return found


async def async_freezelist_program(aleo: Any, program: str) -> str | None:
    """Async twin of :func:`freezelist_program`."""
    if program.endswith(FREEZELIST_SUFFIX):
        return program
    if program in KNOWN_TOKEN_FREEZELISTS:
        return KNOWN_TOKEN_FREEZELISTS[program]
    from .facade.errors import AleoError as FacadeAleoError, ProgramNotFound
    try:
        handle = await aleo.programs.get(program)
        imports = list(handle.imports)
    except (ProgramNotFound, FacadeAleoError):
        imports = []
    found = freezelist_from_imports(program, imports)
    if isinstance(found, _Unresolved):
        raise _unresolved_error(program)
    return found


# ── Resolution with a per-client cache ───────────────────────────────────────

@dataclass(frozen=True)
class ComplianceProofContext:
    """One ``[MerkleProof; 2]`` argument a call needs.

    Attributes:
        freezelist_program: The list the contract checks for this argument, or
            ``None`` for an argument the program accepts and ignores (the
            credits wrapper); such a slot receives the empty-tree pair.
        address: The address whose absence the argument proves.
    """

    freezelist_program: str | None
    address: str


@dataclass
class _CachedList:
    root: int
    tree: list[int]
    proofs: "OrderedDict[str, str]"


def _parse_root(program: str, raw: Any) -> int:
    text = None if raw is None else str(raw).strip().strip('"')
    if text in (None, "", "null", "None"):
        raise FreezeListError(f"{program}/{FREEZE_LIST_ROOT_MAPPING}[{CURRENT_ROOT_KEY}] is unreadable; "
                              "pass compliance_proofs= explicitly")
    assert text is not None
    if not text.endswith("field") or not text[:-5].isdigit():
        raise FreezeListError(f"Invalid current freeze-list root for {program}: {text!r}")
    return int(text[:-5])


class _ProofCache:
    """Per-client memory of verified witnesses, keyed by list and pinned to the root they were built against."""

    def __init__(self) -> None:
        self._lists: "OrderedDict[tuple[str, str], _CachedList]" = OrderedDict()

    def entry(self, network: str, program: str, root: int, fetch_tree: Callable[[], list[int]]) -> _CachedList:
        key = (network, program)
        cached = self._lists.get(key)
        if cached is not None and cached.root == root:
            return cached
        self._lists.pop(key, None)
        tree = [0, 0, EMPTY_TREE_ROOT] if root == EMPTY_TREE_ROOT else fetch_tree()
        entry = _CachedList(root, tree, OrderedDict())
        while len(self._lists) >= MAX_CACHED_LISTS:
            self._lists.popitem(last=False)
        self._lists[key] = entry
        return entry

    def current(self, network: str, program: str, root: int) -> _CachedList | None:
        """The cached entry for *program* when it was built against *root*, else ``None``."""
        cached = self._lists.get((network, program))
        return cached if cached is not None and cached.root == root else None

    def forget(self, network: str, program: str, entry: _CachedList) -> None:
        if self._lists.get((network, program)) is entry:
            del self._lists[(network, program)]

    def clear(self) -> None:
        self._lists.clear()

    def __len__(self) -> int:
        return len(self._lists)


def _witness(entry: _CachedList, network: str, address: str) -> str:
    if entry.root == EMPTY_TREE_ROOT:
        return EMPTY_MERKLE_PROOF_PAIR          # the same pair for every address; nothing to build or cache
    proof = entry.proofs.get(address)
    if proof is None:
        proof = generate_freezelist_proof(entry.tree, address, entry.root, network)
        while len(entry.proofs) >= MAX_CACHED_ADDRESSES:
            entry.proofs.popitem(last=False)
        entry.proofs[address] = proof
    return proof


def _fetched_tree(program: str, nodes: list[int] | None) -> list[int]:
    if nodes is None:
        raise FreezeListError(f"{program} publishes no compliance tree; pass compliance_proofs= explicitly")
    return nodes


class FreezeListProofs:
    """Resolve and cache freeze-list exclusion proofs for one client.

    Each resolution reads the list's current root (one mapping read per list
    per call). While the root is unchanged, the published tree and every
    witness built from it are reused from memory; a changed root discards
    them and fetches the tree again. The cache holds at most 16 lists and
    256 addresses per list and never outlives the process.

    Args:
        aleo: The bound ``Aleo`` facade; reads go through its network module.
    """

    def __init__(self, aleo: Any) -> None:
        self._aleo = aleo
        self._cache = _ProofCache()

    @property
    def network(self) -> str:
        return str(self._aleo.network_name)

    def clear(self) -> None:
        """Drop every cached tree and witness; the next call reads the chain again."""
        self._cache.clear()

    def root(self, program: str) -> int:
        """The list's current on-chain root (``freeze_list_root[1u8]``) as an integer."""
        try:
            raw = self._aleo.network.get_program_mapping_value(program, FREEZE_LIST_ROOT_MAPPING, CURRENT_ROOT_KEY)
        except AleoNetworkError as exc:
            if exc.status == 404:
                raw = None
            else:
                raise
        return _parse_root(program, raw)

    def tree(self, program: str, root: int | None = None) -> list[int]:
        """The published tree for *program*, verified to end in *root* (read from chain when omitted)."""
        root = self.root(program) if root is None else root
        return self._entry(program, root).tree

    def exclusion_proof(self, address: str, program: str) -> str:
        """The ``[MerkleProof; 2]`` literal proving *address* is not on *program*'s list.

        *program* may be the freeze-list program or a token program that
        imports one. Returns the empty-tree pair while the list is empty.

        Raises:
            FrozenAddressError: *address* is frozen.
            FreezeListError: The root is unreadable, the published tree does
                not match it, or no list can be found for *program*.
        """
        freezelist = freezelist_program(self._aleo, program)
        if freezelist is None:
            return EMPTY_MERKLE_PROOF_PAIR
        return self.resolve([ComplianceProofContext(freezelist, address)])[0]

    def resolve(self, slots: Sequence[ComplianceProofContext | None]) -> list[str]:
        """One ``[MerkleProof; 2]`` literal per slot, in order.

        Every slot naming the same list is proved against one root read, so
        a call that takes several proofs for one list cannot mix roots.
        ``None`` slots and slots whose list is ``None`` get the empty pair.
        """
        roots: dict[str, int] = {}
        out: list[str] = []
        for slot in slots:
            if slot is None or slot.freezelist_program is None:
                out.append(EMPTY_MERKLE_PROOF_PAIR)
                continue
            program = slot.freezelist_program
            if program not in roots:
                roots[program] = self.root(program)
            entry = self._entry(program, roots[program])
            try:
                out.append(_witness(entry, self.network, slot.address))
            except FrozenAddressError:
                raise                                   # a true answer from a good tree: keep it cached
            except FreezeListError:
                self._cache.forget(self.network, program, entry)   # a stale or corrupt tree: fetch again next time
                raise
        return out

    def _entry(self, program: str, root: int) -> _CachedList:
        def fetch() -> list[int]:
            return _fetched_tree(program, self._aleo.network.get_compliance_freeze_list(program))
        return self._cache.entry(self.network, program, root, fetch)


class AsyncFreezeListProofs:
    """Async twin of :class:`FreezeListProofs` over an ``AsyncAleo`` facade."""

    def __init__(self, aleo: Any) -> None:
        self._aleo = aleo
        self._cache = _ProofCache()

    @property
    def network(self) -> str:
        return str(self._aleo.network_name)

    def clear(self) -> None:
        """Drop every cached tree and witness; the next call reads the chain again."""
        self._cache.clear()

    async def root(self, program: str) -> int:
        """The list's current on-chain root (``freeze_list_root[1u8]``) as an integer."""
        try:
            raw = await self._aleo.network.get_program_mapping_value(program, FREEZE_LIST_ROOT_MAPPING, CURRENT_ROOT_KEY)
        except AleoNetworkError as exc:
            if exc.status == 404:
                raw = None
            else:
                raise
        return _parse_root(program, raw)

    async def tree(self, program: str, root: int | None = None) -> list[int]:
        """The published tree for *program*, verified to end in *root* (read from chain when omitted)."""
        root = await self.root(program) if root is None else root
        return (await self._entry(program, root)).tree

    async def exclusion_proof(self, address: str, program: str) -> str:
        """The ``[MerkleProof; 2]`` literal proving *address* is not on *program*'s list (awaited)."""
        freezelist = await async_freezelist_program(self._aleo, program)
        if freezelist is None:
            return EMPTY_MERKLE_PROOF_PAIR
        return (await self.resolve([ComplianceProofContext(freezelist, address)]))[0]

    async def resolve(self, slots: Sequence[ComplianceProofContext | None]) -> list[str]:
        """One ``[MerkleProof; 2]`` literal per slot, in order; one root read per list per call."""
        roots: dict[str, int] = {}
        out: list[str] = []
        for slot in slots:
            if slot is None or slot.freezelist_program is None:
                out.append(EMPTY_MERKLE_PROOF_PAIR)
                continue
            program = slot.freezelist_program
            if program not in roots:
                roots[program] = await self.root(program)
            entry = await self._entry(program, roots[program])
            try:
                out.append(_witness(entry, self.network, slot.address))
            except FrozenAddressError:
                raise                                   # a true answer from a good tree: keep it cached
            except FreezeListError:
                self._cache.forget(self.network, program, entry)   # a stale or corrupt tree: fetch again next time
                raise
        return out

    async def _entry(self, program: str, root: int) -> _CachedList:
        network = self.network
        if root != EMPTY_TREE_ROOT and self._cache.current(network, program, root) is None:
            nodes = _fetched_tree(program, await self._aleo.network.get_compliance_freeze_list(program))
            return self._cache.entry(network, program, root, lambda: nodes)
        return self._cache.entry(network, program, root, _never)


def _never() -> list[int]:  # pragma: no cover - only reached on a logic error
    raise AssertionError("tree fetch requested for a cached or empty list")


def _network_module(network: str) -> Any:
    import importlib
    return importlib.import_module(f"aleo.{network}")


__all__ = [
    "CURRENT_ROOT_KEY", "DEFAULT_DEPTH", "EMPTY_MERKLE_PROOF", "EMPTY_MERKLE_PROOF_PAIR", "EMPTY_TREE_ROOT",
    "FREEZE_LIST_INDEX_MAPPING", "FREEZE_LIST_LAST_INDEX_MAPPING", "FREEZE_LIST_MAPPING", "FREEZE_LIST_ROOT_MAPPING",
    "KNOWN_TOKEN_FREEZELISTS", "PROOF_SIBLINGS", "ZERO_ADDRESS",
    "AsyncFreezeListProofs", "FreezeListError", "FreezeListProofs", "FrozenAddressError",
    "ComplianceProofContext", "MerkleProofInput", "MerkleProofPair", "ProofBundle",
    "address_to_field_int", "async_freezelist_program", "build_tree", "format_merkle_proof",
    "format_merkle_proof_pair", "freezelist_from_imports", "freezelist_program", "generate_freezelist_proof",
    "generate_leaves", "hash_two", "leaf_indices", "literal_compliance_proof", "sibling_path",
]

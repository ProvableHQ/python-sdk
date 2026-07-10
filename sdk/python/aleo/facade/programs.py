"""Aleo — facade programs module (F4).

Attached to the :class:`~aleo.facade.client.Aleo` client as ``aleo.programs``.
Provides a Web3.py-style program interface: a bound facade :class:`Program`
whose ``.functions`` namespace is built dynamically from the program's real
function set (mirroring ``contract.functions`` in web3.py), input coercion,
mapping reads, and three entry points into :mod:`aleo.abi`.

**F5 boundary.**  ``program.functions.<name>(*args)`` returns a lightweight
:class:`PreparedCall` that has coerced + validated its arguments against the
declared inputs and exposes the declared signature.  It deliberately does *not*
authorize / execute / prove — F5's verb ladder extends :class:`PreparedCall`.
"""
from __future__ import annotations

from typing import Any, Iterator

from .._client_common import AleoNetworkError
from .errors import ProgramNotFound

# ── Coercion tables ─────────────────────────────────────────────────────────

# Integer-like Aleo types whose literal is the decimal value + the type name as
# a suffix (e.g. ``10`` + ``u64`` → ``"10u64"``, ``5`` + ``field`` → ``"5field"``).
_INT_SUFFIX_TYPES: frozenset[str] = frozenset(
    {
        "u8", "u16", "u32", "u64", "u128",
        "i8", "i16", "i32", "i64", "i128",
        "field", "group", "scalar",
    }
)


class PreparedCall:
    """A validated, coerced function call awaiting execution.

    **This is the seam F5 extends.**  F4 produces a ``PreparedCall`` from
    ``program.functions.<name>(*args)``; it holds everything F5's verb ladder
    (``authorize`` / ``execute`` / ``prove``) needs and nothing more.  F4 does
    not implement any of those verbs.

    Attributes
    ----------
    program_id:
        The bound program's identifier string (e.g. ``"credits.aleo"``).
    function_name:
        The transition/function name (e.g. ``"transfer_public"``).
    inputs:
        The declared input descriptors from
        ``program.get_function_inputs(function_name)`` — each a dict with at
        least ``type``/``visibility``/``register`` (record inputs also carry
        ``record`` + ``members``).
    args:
        The coerced positional arguments as Aleo ``Value`` strings, ready to
        hand to snarkvm authorize/execute in F5.
    """

    __slots__ = ("program_id", "function_name", "inputs", "args", "_client")

    def __init__(
        self,
        program_id: str,
        function_name: str,
        inputs: list[dict[str, Any]],
        raw_args: tuple[Any, ...],
        client: Any = None,
    ) -> None:
        self.program_id = program_id
        self.function_name = function_name
        self.inputs = inputs
        self._client = client
        self.args: list[str] = _coerce_args(program_id, function_name, inputs, raw_args)

    @property
    def signature(self) -> str:
        """A human-readable declared signature, e.g. ``"transfer_public(address, u64)"``."""
        parts = [_input_type_name(i) for i in self.inputs]
        return f"{self.function_name}({', '.join(parts)})"

    def __repr__(self) -> str:
        return (
            f"PreparedCall({self.program_id}/{self.signature} "
            f"args={self.args!r})"
        )


def _input_type_name(descriptor: dict[str, Any]) -> str:
    """Return the declared Aleo type name for an input descriptor.

    Record inputs report their record name (``descriptor["record"]``); all
    others report ``descriptor["type"]``.
    """
    if descriptor.get("type") == "record":
        return str(descriptor.get("record", "record"))
    return str(descriptor.get("type", "?"))


def _coerce_one(value: Any, descriptor: dict[str, Any]) -> str:
    """Coerce a single Python *value* against an input *descriptor*.

    Rules (see F4 brief):

    * preformatted Aleo strings and typed wrappers are ALWAYS accepted;
    * ``bool`` + ``boolean`` → ``"true"``/``"false"``;
    * bare ``int`` + integer-like type → decimal + type suffix (``10u64``);
    * ``str`` for ``address``/record types → passed through;
    * typed wrapper objects → ``str(x)``.

    Raises :exc:`ValueError` with an actionable message on an uncoercible type.
    """
    type_name = _input_type_name(descriptor)
    register = str(descriptor.get("register", "?"))
    py_type_name = type(value).__name__

    # bool must be checked before int (bool is an int subclass).
    if isinstance(value, bool):
        if type_name == "boolean":
            return "true" if value else "false"
        raise ValueError(
            f"Cannot coerce bool for input {register} which expects Aleo type "
            f"{type_name!r}."
        )

    if isinstance(value, int):
        if type_name in _INT_SUFFIX_TYPES:
            return f"{value}{type_name}"
        raise ValueError(
            f"Cannot coerce bare int {value!r} for input {register} which "
            f"expects Aleo type {type_name!r}. Pass a preformatted value "
            f"string instead."
        )

    if isinstance(value, str):
        # Preformatted Aleo strings (addresses, literals, record/struct text,
        # already-suffixed integers) always pass through unchanged.
        return value

    # Reject Python scalar/container types that have no meaningful Aleo literal
    # (float, bytes, None, list, dict, …).  A bare float is the classic case:
    # there is no float Aleo type, and stringifying "1.5" would be wrong.
    if value is None or isinstance(value, (float, bytes, bytearray, list, tuple, dict, set)):
        raise ValueError(
            f"Cannot coerce a {py_type_name} for input {register} "
            f"which expects Aleo type {type_name!r}. Pass a preformatted value "
            f"string or a typed Aleo value instead."
        )

    # Typed wrapper (network module type, RecordPlaintext, ProgramID, …) with a
    # sensible str() — always accepted.
    return str(value)


def _coerce_args(
    program_id: str,
    function_name: str,
    inputs: list[dict[str, Any]],
    raw_args: tuple[Any, ...],
) -> list[str]:
    """Coerce and validate *raw_args* against the declared *inputs*.

    Raises :exc:`ValueError` on wrong arity or an uncoercible argument, citing
    the expected count / Aleo type and the full function signature.
    """
    sig_parts = [_input_type_name(i) for i in inputs]
    signature = f"{function_name}({', '.join(sig_parts)})"
    if len(raw_args) != len(inputs):
        raise ValueError(
            f"{program_id}/{function_name} expects {len(inputs)} argument(s) "
            f"but got {len(raw_args)}. Signature: {signature}"
        )
    coerced: list[str] = []
    for value, descriptor in zip(raw_args, inputs):
        try:
            coerced.append(_coerce_one(value, descriptor))
        except ValueError as exc:
            raise ValueError(f"{exc} Signature: {signature}") from exc
    return coerced


class ProgramFunctions:
    """Dynamic namespace of a program's callable functions.

    Built at load time from the program's real function set (transition names +
    per-function declared inputs), mirroring web3.py's ABI-driven
    ``contract.functions``.  Resolve a function by attribute
    (``program.functions.transfer_public``) or by item
    (``program.functions["transfer_public"]``); either returns a callable that,
    when invoked with arguments, produces a :class:`PreparedCall`.

    Supports ``dir()`` / iteration / ``in`` over the callable function names.
    """

    # Declared so pyright resolves real attributes to their true types rather
    # than routing them through __getattr__ (which returns _FunctionCaller).
    _program_id: str
    _inputs_by_fn: dict[str, list[dict[str, Any]]]
    _client: Any

    def __init__(
        self,
        program_id: str,
        inputs_by_fn: dict[str, list[dict[str, Any]]],
        client: Any = None,
    ) -> None:
        # Leading underscores keep these off the dynamic-attribute path.
        object.__setattr__(self, "_program_id", program_id)
        object.__setattr__(self, "_inputs_by_fn", inputs_by_fn)
        object.__setattr__(self, "_client", client)

    def _available(self) -> str:
        names = sorted(self._inputs_by_fn)
        return ", ".join(names) if names else "(none)"

    def _make(self, name: str) -> "_FunctionCaller":
        return _FunctionCaller(
            self._program_id, name, self._inputs_by_fn[name], self._client
        )

    def __getattr__(self, name: str) -> "_FunctionCaller":
        # __getattr__ only fires for misses, so real attrs (_program_id, …)
        # never reach here.  Read the map via object.__getattribute__ to avoid
        # recursing back into __getattr__ for a partially-initialised instance.
        inputs_by_fn: dict[str, list[dict[str, Any]]] = object.__getattribute__(
            self, "_inputs_by_fn"
        )
        if name in inputs_by_fn:
            return self._make(name)
        raise AttributeError(
            f"Program {self._program_id!r} has no function {name!r}. "
            f"Available functions: {self._available()}"
        )

    def __getitem__(self, name: str) -> "_FunctionCaller":
        if name in self._inputs_by_fn:
            return self._make(name)
        raise KeyError(
            f"Program {self._program_id!r} has no function {name!r}. "
            f"Available functions: {self._available()}"
        )

    def __contains__(self, name: object) -> bool:
        return name in self._inputs_by_fn

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._inputs_by_fn))

    def __len__(self) -> int:
        return len(self._inputs_by_fn)

    def __dir__(self) -> list[str]:
        return list(super().__dir__()) + sorted(self._inputs_by_fn)

    def __repr__(self) -> str:
        return (
            f"ProgramFunctions({self._program_id}, "
            f"functions=[{self._available()}])"
        )


class _FunctionCaller:
    """A single resolved function; calling it builds a :class:`PreparedCall`.

    Holds the declared inputs so it can render its signature in ``repr`` even
    before it is invoked (REPL help).
    """

    __slots__ = ("program_id", "function_name", "inputs", "_client")

    def __init__(
        self,
        program_id: str,
        function_name: str,
        inputs: list[dict[str, Any]],
        client: Any = None,
    ) -> None:
        self.program_id = program_id
        self.function_name = function_name
        self.inputs = inputs
        self._client = client

    @property
    def signature(self) -> str:
        """Declared signature, e.g. ``"transfer_public(address, u64)"``."""
        parts = [_input_type_name(i) for i in self.inputs]
        return f"{self.function_name}({', '.join(parts)})"

    def __call__(self, *args: Any) -> PreparedCall:
        return PreparedCall(
            self.program_id,
            self.function_name,
            self.inputs,
            args,
            self._client,
        )

    def __repr__(self) -> str:
        return f"<function {self.program_id}/{self.signature}>"


class Mapping:
    """A single on-chain mapping of a bound :class:`Program`.

    Obtain via ``program.mapping(name)``.  ``get(key)`` reads the current value
    at ``(program_id, name, key)``; ``names()`` lists the program's mapping key
    names as reported by the network.
    """

    def __init__(self, client: Any, program_id: str, name: str) -> None:
        self._client = client
        self.program_id = program_id
        self.name = name

    def get(self, key: str | Any) -> str:
        """Return the current value at (*program*, *mapping*, *key*).

        Parameters
        ----------
        key:
            The mapping key (an Aleo value string or a typed wrapper).

        Returns
        -------
        str
            The serialised mapping value as returned by the node.
        """
        return self._client.network.get_program_mapping_value(
            self.program_id, self.name, str(key)
        )

    def names(self) -> list[str]:
        """Return the mapping names defined by the program.

        Delegates to ``aleo.network.get_program_mapping_names``.  (This lists
        the program's mappings, matching the underlying network primitive.)
        """
        return list(self._client.network.get_program_mapping_names(self.program_id))

    def __repr__(self) -> str:
        return f"Mapping({self.program_id}/{self.name})"


class Program:
    """A bound facade program returned by ``aleo.programs.get(id)``.

    Wraps the underlying network ``Program`` object with a Web3.py-style
    surface: a dynamic ``.functions`` namespace, ``.mapping(name)`` reads, and
    ``.abi()`` generation.

    Attributes
    ----------
    id:
        The program identifier string (e.g. ``"credits.aleo"``).
    functions:
        A :class:`ProgramFunctions` namespace built from the program's real
        function set — works without ``aleo-abi`` installed.
    """

    def __init__(self, client: Any, raw: Any) -> None:
        self._client = client
        self._raw = raw
        self.id: str = str(raw.id)
        inputs_by_fn: dict[str, list[dict[str, Any]]] = {}
        for ident in raw.functions:
            fn_name = str(ident)
            inputs_by_fn[fn_name] = list(raw.get_function_inputs(fn_name))
        self.functions: ProgramFunctions = ProgramFunctions(
            self.id, inputs_by_fn, client
        )

    # ── Basic accessors ─────────────────────────────────────────────────────

    @property
    def raw(self) -> Any:
        """The underlying network ``Program`` object (escape hatch)."""
        return self._raw

    @property
    def source(self) -> str:
        """The program's Leo/Aleo source text."""
        return str(self._raw.source)

    @property
    def imports(self) -> list[str]:
        """The program's import identifiers as strings."""
        return [str(i) for i in self._raw.imports]

    # ── Mappings ─────────────────────────────────────────────────────────────

    def mapping(self, name: str) -> Mapping:
        """Return a :class:`Mapping` handle for *name*.

        Parameters
        ----------
        name:
            The mapping name (e.g. ``"account"``).
        """
        return Mapping(self._client, self.id, name)

    def mappings(self) -> list[str]:
        """Return the mapping names declared by this program.

        Read locally from the program definition
        (``program.get_mappings()``) — no network call.
        """
        return [str(m["name"]) for m in self._raw.get_mappings()]

    # ── ABI ──────────────────────────────────────────────────────────────────

    def abi(self) -> dict[str, Any]:
        """Generate the ABI for this program (object path).

        Delegates to :func:`aleo.abi.generate_abi` with the underlying network
        Program object.  Lazy: raises :exc:`ImportError` (with an install hint)
        only if the ``aleo-abi`` package is absent.
        """
        from .. import abi as _abi
        return _abi.generate_abi(self._raw, self._client.network_name)

    def __repr__(self) -> str:
        return f"Program({self.id}, functions={len(self.functions)})"


class ProgramsModule:
    """Namespaced program operations attached to an :class:`~aleo.facade.client.Aleo` client.

    Access via ``aleo.programs``, not by direct construction.

    Parameters
    ----------
    client:
        The parent :class:`~aleo.facade.client.Aleo` instance.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def __repr__(self) -> str:
        return f"ProgramsModule(network={self._client._provider.network!r})"

    # ── Internal helper ────────────────────────────────────────────────────

    def _net(self) -> Any:
        """Return the network module (``aleo.mainnet`` or ``aleo.testnet``).

        Mirrors :meth:`AccountModule._net`.
        """
        network: str = self._client._provider.network
        if network == "testnet":
            import aleo.testnet as _testnet  # type: ignore[attr-defined]
            return _testnet
        import aleo.mainnet as _mainnet
        return _mainnet

    # ── Fetch a bound program ────────────────────────────────────────────────

    def get(self, program_id: str, edition: int | None = None) -> Program:
        """Fetch *program_id* from the network and return a bound :class:`Program`.

        Parameters
        ----------
        program_id:
            Aleo program identifier (e.g. ``"credits.aleo"``).
        edition:
            Optional edition number.

        Returns
        -------
        Program
            A bound facade program.

        Raises
        ------
        ProgramNotFound
            If the network has no such program (a 404 from the node).
        """
        try:
            source: str = self._client.network.get_program(program_id, edition)
        except AleoNetworkError as exc:
            if exc.status == 404:
                raise ProgramNotFound(program_id) from exc
            raise
        net = self._net()
        raw: Any = net.Program.from_source(source)
        return Program(self._client, raw)

    # ── ABI (web path) ───────────────────────────────────────────────────────

    def abi(self, program_id: str, edition: int | None = None) -> dict[str, Any]:
        """Generate the ABI for *program_id* by fetching its source (web path).

        One call: fetches the deployed source via
        ``aleo.network.get_program`` then funnels the string to
        :func:`aleo.abi.generate_abi` — no ``Program`` object required.

        Parameters
        ----------
        program_id:
            Aleo program identifier.
        edition:
            Optional edition number.

        Raises
        ------
        ProgramNotFound
            If the network has no such program (a 404 from the node).
        ImportError
            If the ``aleo-abi`` package is not installed.
        """
        try:
            source: str = self._client.network.get_program(program_id, edition)
        except AleoNetworkError as exc:
            if exc.status == 404:
                raise ProgramNotFound(program_id) from exc
            raise
        from .. import abi as _abi
        return _abi.generate_abi(source, self._client.network_name)


__all__ = [
    "ProgramsModule",
    "Program",
    "ProgramFunctions",
    "PreparedCall",
    "Mapping",
]

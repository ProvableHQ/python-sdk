# Documentation Voice — Aleo Python SDK

How docstrings and prose read in this repo. The rules bind; the examples show
them in practice. This is a **Python** SDK — docstrings, not JSDoc; types live
in annotations, not in the prose.

## Docstrings

Lead with a present-tense verb, give one or two sentences of context (including
side effects — network, fee, local-only), and describe each argument, return
value, and raised error **by its consequence**. Do not restate types the
signature already carries.

### Good

```python
def transfer_public(self, recipient: str, amount: int) -> BoundCall:
    """Build a public ``credits.aleo`` transfer call.

    Returns an unexecuted call — nothing touches the network until you invoke a
    verb (``.simulate``, ``.build_transaction``, ``.transact``, ``.delegate``).

    Args:
        recipient: The destination address (``aleo1…``).
        amount: Microcredits to send; coerced to ``u64`` from the program's
            declared input type.

    Returns:
        A bound call; ``.transact(account)`` builds, proves, and broadcasts it.

    Raises:
        ProgramNotFound: If ``credits.aleo`` is not loaded in the process.
    """
```

Why it works: starts with "Build"; the second sentence names the key behavior
(nothing runs until a verb — the footgun-preventing fact); each arg is described
by what it *means*, not its type; `Raises` names the concrete failure a caller
would catch.

### Bad

```python
def transfer_public(self, recipient, amount):
    """This function allows you to easily and powerfully build a transfer in a
    seamless way. It takes a recipient and an amount and returns a result.

    :param recipient: the recipient (string)
    :param amount: the amount (int)
    :return: the result
    """
```

Why it fails: filler ("easily", "powerfully", "seamless"); restates types; says
"a result" instead of what you get and what you do with it; no side-effect or
error information.

### Phrasings to avoid

Beyond the filler above, these are banned outright. Each one reads like guidance
while leaving the caller with nothing to act on — replace it with the concrete
instruction, or delete it.

- **"reach for"** — never use it. Say what to call and when.
- **"defensively"**, **"as appropriate"**, **"where necessary"** — name the
  actual condition instead.

```python
# Bad — sounds like advice, gives none
"""… either may be ``None``, so reach for them defensively."""

# Good — states the check and what it protects
"""… treat both as optional — guard with ``if entry.token0_info`` before
reading a field rather than assuming a default."""
```

## Naming in prose and examples

- Use the Pythonic surface: properties (`key.address`, not `key.address()`),
  `str(x)`/`bytes(x)` not `.to_string()`/`.to_bytes()`, `PrivateKey.random()`.
- Prefer the named algebraic method in prose where it reads clearer
  (`a.add(b)`), the operator in code examples (`a + b`) — both exist.
- Examples must run as written against the documented signature. Mark any
  snippet that touches the network or downloads proving parameters.
- Say "microcredits" explicitly for `u64` fee/amount values; never leave the
  unit implicit.

## Privacy stance

This is a privacy chain. Do not document or add affordances that link
signatures to signer addresses (no `recover`-style verb). When a feature shares
secret material with a service (e.g. delegated record scanning shares the view
key), state that tradeoff plainly. Where an alternative exists, name the
supported extension point (e.g. assigning a custom ``RecordProvider``).

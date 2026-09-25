# Documentation Voice — Aleo Python SDK

Rules for public API docstrings, comments, READMEs, and tutorials across the
Python SDK packages. Adapted from Veil's `.agents/voice.md` with Python examples.
The rules bind; the examples show them in practice.

## Describe utility before mechanics

Write from the caller's perspective: what the feature accomplishes, why to
choose it, what happens to the caller's assets or data, and what the caller
must do next. Naming a method and narrating its implementation does not explain
its utility.

Introduce information in this order:

1. The outcome the caller wants.
2. When to choose this operation or option.
3. Consequences that affect that choice: visibility, fees, access to funds,
   required signatures, waiting, and recovery.
4. The shortest example that accomplishes the task.
5. Additional details needed to interpret the result or handle a failure.

Keep internal dispatch, serialization, compiler constraints, and implementation
history on the internal helpers that need them. Include a technical detail in
public documentation only when it helps the caller make a decision or complete
the task. Explain unfamiliar terms when first needed.

### Public action descriptions

Bad — describes implementation without explaining why to call it:

```python
"""Select a rebalance transition from the router's input variants.

Dispatch depends on token standards and funded sides. Leo transitions cannot
accept optional inputs, so each combination has a separate function. Pure and
local.
"""
```

Good — explains the effect on the position and the caller's choices:

```python
"""Move liquidity into a new position in one transaction.

The transaction withdraws the existing position, collects its principal and
fees, optionally adds funds, and creates the replacement position. Choose a
target liquidity amount or set maximum funding amounts for the two tokens.

A price change before execution can cause the transaction to be rejected.
A rejected rebalance leaves the position unchanged but can still cost a fee.
Refresh the pool state and review a new plan before submitting another attempt.
"""
```

The good description explains what happens to the funds, which inputs express
the caller's intent, and what failure means. It does not require the caller to
understand the router's function-selection rules.

### Explain options through the decision they support

Bad:

> `mint_mode` selects the public, record, or private mint path. Private mode
> encodes a commitment in the deposit hook and requires `complete`.

Good:

> Public Bridge delivers a publicly visible USDCx balance without a claim step.
> Public Bridge to Private Balance delivers funds for private use, but the
> Ethereum deposit still reveals the recipient's Aleo address. Private Bridge
> keeps that address out of the deposit; the recipient must retain a secret
> nonce and return to claim the funds. The Ethereum sender and deposited amount
> remain public in every mode.

Only after explaining the choice, show the corresponding `mint_mode` values
and a short example. A table should compare concerns the reader can evaluate,
not repeat enum names as explanations.

## Prose: guides and tutorials

State facts plainly. Anchor unfamiliar concepts to familiar ones. Explain why
and when. Use third person or the imperative: "the recipient", "the caller",
"a developer", "Save the checkpoint". Avoid "you", "your", "we", "our", and
"I" in documentation. Caller-centered writing concerns the content and order
of explanation; it does not require second-person pronouns.

### Lead with the tutorial's utility

Bad — opens with a limitation that requires unexplained background:

```python
"""Pool trade history does not contain per-position fills.

This example defines a fill as the change in token amounts backing fixed
liquidity between two consecutive pool prices.
"""
```

Good — establishes the task before defining its terms:

```python
"""Calculate how trades changed the assets held in a liquidity position.

Compare the token amounts backing the position before and after each pool
price change. This gives the position's change in token holdings for that trade.
"""
```

Do not introduce an architectural limitation, explain how the implementation
works around it, and leave the purpose until the end.

### Explain the consequence, then name the mechanism

Bad:

> Recovery deserializes a versioned checkpoint, resolves the registry route,
> reconstructs the receipt, and dispatches to the provider status implementation.

Good:

> Recover an interrupted transfer to find out whether funds arrived or another
> action is needed. Recovery checks the existing transfer without sending funds
> again. A saved checkpoint identifies the transfer; load it and pass it to
> `bridge.recover(checkpoint)`.

If no checkpoint was saved, explain what can be recovered from chain data and
what information is still required. A different file containing a checkpoint
is not an example of recovery without saved files.

### Define concepts with concrete comparisons

Good:

> Private credits are held in encrypted records, analogous to Bitcoin's UTXOs.
> A transfer spends an unspent record and creates records for the recipient and
> any change. The input record must cover the amount being sent.

Bad:

> The SDK provides a powerful and seamless way to transfer value. Records are
> an important concept that developers should probably understand.

Give hard rules emphasis when ignoring them has a concrete consequence:
**Do not submit a second deposit because monitoring timed out.** Explain how to
check the first submission instead.

## Python docstrings

Lead with a present-tense verb. Give one or two sentences explaining the purpose
and relevant consequences. Describe arguments, results, and exceptions by what
they mean to the caller, not by repeating their names or annotations.

Use the surrounding module's docstring convention. The examples use `Args`,
`Returns`, `Raises`, and `Attributes`; preserve NumPy-style sections in modules
that already use them. Do not introduce JSDoc tags into Python.

### Good

An adapted description of `FileCheckpointStore.load`:

```python
def load(self, checkpoint_id: str) -> Checkpoint | None:
    """Load a saved transfer checkpoint for recovery after a restart.

    Reads the journal file without contacting a network or submitting a
    transaction. Pass the result to ``bridge.recover`` to check current progress.

    Args:
        checkpoint_id: The ``checkpoint.id`` reported when the transfer was
            saved. Use the exact ID returned by the journal or callback.

    Returns:
        The saved checkpoint, or ``None`` if its file does not exist.

    Raises:
        CheckpointInvalidError: The file is not a supported checkpoint.
        OSError: The journal file could not be read.
        UnicodeDecodeError: The file does not contain UTF-8 text.
    """
```

Why it works: the first line explains when to use it, the context distinguishes
loading saved data from checking current progress, and the return description
explains the missing-file case. Types remain in the signature.

### Bad

```python
def load(self, checkpoint_id):
    """This function allows callers to easily load a checkpoint.

    Args:
        checkpoint_id: The checkpoint ID (string).

    Returns:
        The result.
    """
```

It gives no reason to call the method, no meaning for the return value, and no
instructions for a missing or unreadable checkpoint.

### Units, bounds, and defaults

State units, accepted ranges, and meaningful defaults for inputs. Python `int`
represents both `u64` and `u128` values; document the contract's bound rather
than inventing separate Python integer types. State the actual bound accepted
by the operation, including whether zero is allowed.

```python
# Good, for an operation accepting a positive u64 amount:
"""amount: Microcredits to send, from 1 through 2**64 - 1."""
# Bad:
"""amount: The amount (int)."""

# Good:
"""priority_fee: Additional fee in microcredits. Defaults to 0."""
# Bad:
"""priority_fee: Optional fee."""
```

Use integer base units or the API's supported decimal strings / `Decimal`
values for exact amounts. Do not introduce floating-point rounding into money
examples. One credit is 1,000,000 microcredits; bridge display amounts use the
selected asset's units, not necessarily microcredits.

### Side effects

State whether a call reads the network, signs, submits a transaction, pays a
fee, shares data, writes files, or downloads proving parameters. Distinguish
preparing a call, authorizing it, submitting it, and confirming it.

```python
# Good:
"""Read the transfer's current status without submitting another transaction."""
"""Send the authorization to the prover; the account's private key stays local."""
# Bad:
"""Pure and local."""
```

Never use "pure and local" as a substitute for concrete consequences.

### Document fields on the class

For dataclasses and other result objects, use an `Attributes` section on the
class docstring. Describe what each field enables the caller to do. Keep field
types in annotations, and distinguish a data field from a Python `@property`.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class CheckpointLoadResult:
    """Load saved transfers while reporting files that need attention.

    Attributes:
        checkpoints: Readable checkpoints, oldest first by file modification
            time. Select the intended transfer before calling ``recover``.
        errors: Files that could not be read or parsed, each with its path and
            reason. These errors do not establish that a transfer failed.
    """

    checkpoints: list[Checkpoint]
    errors: list[CheckpointProblem]
```

Bad field documentation repeats the annotation: `checkpoints: A list of
checkpoints`. Good documentation explains ordering, interpretation, and use.

### Deprecations provide a migration path

Name the replacement and any change in how to use its result. State a removal
version only if one has been decided. Preserve the project's existing Sphinx
or docstring convention rather than adding `@deprecated` tags.

Good migration prose:

> Replace `list_with_problems()` with `load_checkpoints()`. Read
> `result.checkpoints` and `result.errors` instead of unpacking a tuple.

Bad:

> Deprecated. Do not use.

## Python examples

- Use `snake_case` and the actual public signatures. Prefer properties such as
  `key.address`, `str(value)` / `bytes(value)`, and `PrivateKey.random()`.
- Algebraic types expose named methods and operators. Use `a + b` in examples
  and `a.add(b)` where clearer in prose. Use `negate`, not `inverse`, for
  snarkVM's group-negation operation.
- Include imports and establish variables before using them. For connected
  tutorial snippets, explicitly state which earlier setup they continue.
- Show both source and destination assets in bridge quotes even when the API
  can infer one. Examples should communicate the intended transfer.
- Add short inline examples of units, concrete values, and significant outputs.
  Explain what an output means, not what `print` does. Do not annotate every
  line with a restatement of the code.
- When an identifier is central to the task, show an actual verified example
  and explain where the caller obtains it. An environment-variable name is
  not an example ID. Identify historical transfers as such, and provide the
  checkpoint file if the example loads it. Never fabricate observed results.
- Label illustrative fees and outputs; distinguish them from measured results.
  Date live observations whose status can change. Do not put private keys,
  view keys, or real claim secrets in examples.
- Put `await` only on awaitable methods. On `AsyncAleo`, account operations
  and call construction stay synchronous; network operations are awaited.
- Treat alternative submission methods as alternatives. Do not demonstrate
  both on the same transfer as sequential steps.
- Check optional results before accessing them. State what `None`, an empty
  list, a timeout, or a reported error means and what the caller can do next.

## Precise language

### Use plain verbs

Objects do not "speak", "know", or "want". Say what they accept, return,
require, store, or decrypt.

```python
# Bad:
"""The API speaks decimals; the contract speaks base units."""
# Good:
"""The API returns decimal amounts; the contract takes integer base units."""
```

### Name structures and ledger fields

Do not use "shape" or "shape of" instead of naming the fields, variants, or
standards. Say whether a token implements ARC-20 or is wrapped under ARC-22.
Use canonical Aleo terms: block height, transaction ID, transition ID,
transaction index, and transition index. Do not call them "coordinates".

Bad:

> Adds chain-confirmed transaction coordinates to the trade's shape.

Good:

> Adds the block height and transaction index so trades can be ordered as they
> appeared on-chain.

### Describe the current contract

Do not explain repository history, generation drift, or abandoned implementation
choices in public API documentation.

Bad:

> The checked-in snapshot predates this field, so the example defines it here.

Good:

> The trade response includes the transaction ID used to look up confirmation.

Keep historical implementation context in commits, pull requests, or internal
notes. Migration guidance belongs in release notes or a migration section.

### Remove filler and vague advice

Avoid "powerful", "seamless", "robust", "easily", "simply", "just", "it's
worth noting", and "this function is designed to allow". State the fact.
Never write "reach for". Replace "defensively", "as appropriate", and "where
necessary" with the actual condition and action.

```python
# Bad:
"""Token metadata may be absent, so reach for it defensively."""
# Good:
"""Check ``entry.token0_info is not None`` before reading token metadata."""
```

## Privacy and funds

Explain visibility and control precisely: who can read the information, what
remains public, who can spend or claim the funds, and what must be retained.
A private balance does not imply that the bridge deposit hides the recipient.

When delegated scanning shares a view key, state that the scanner can decrypt
the account's records. Name supported alternatives, such as supplying a record
explicitly or a custom `RecordProvider`. Explain delegated proving's disclosure
separately; it is not the same as sharing a view key or private key.

Do not document affordances that link signatures to signer addresses. This
restriction concerns signer recovery, not recovery of an interrupted transfer.

Distinguish submission from completion. Explain what a fee pays for and what a
failed transaction may cost. A polling timeout or lost response is not proof
that funds were not sent; describe how to inspect or recover the existing
transfer before another submission.

## Review checklist

- Does the opening name something the caller wants to accomplish?
- Does each option explain why to choose it and what the choice costs or reveals?
- Do examples establish their inputs and show meaningful values and results?
- Are units, defaults, signatures, and return fields accurate for this package?
- Are network access, signatures, fees, storage, and disclosures clear where relevant?
- Does each failure description explain the outcome for funds or data and the next action?
- Can implementation narration be removed without losing anything the caller needs?

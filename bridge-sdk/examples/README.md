# Bridge examples

Use these scripts to check a transfer's cost, send assets to and from Aleo, or continue a
transfer after an interruption. Each script runs independently and reports
whether the work finished, needs attention, or is still pending.

These examples use **mainnet**. Install the bridge SDK in your Python environment
and run the commands from `bridge-sdk/`. Quote and status commands do not sign
transactions. `--execute`, `--action resume`, and `--action complete` can spend
real funds. Review a quote before submitting a new transfer.

| Script | What it helps you do |
| --- | --- |
| [quote_transfer.py](quote_transfer.py) | Review the cost of sending WBTC from Ethereum to Aleo. |
| [bridge_wbtc.py](bridge_wbtc.py) | Send WBTC and monitor its arrival as a public Aleo balance. |
| [bridge_usdc_private_balance.py](bridge_usdc_private_balance.py) | Receive private USDCx automatically; the Ethereum deposit still reveals the Aleo recipient. |
| [bridge_usdc_private_recipient.py](bridge_usdc_private_recipient.py) | Hide the Aleo recipient in the deposit, then claim private USDCx using the retained nonce. |
| [bridge_wbtc_to_ethereum.py](bridge_wbtc_to_ethereum.py) | Return public Aleo WBTC to Ethereum. |
| [bridge_sol_to_solana.py](bridge_sol_to_solana.py) | Return public Aleo SOL as native SOL on Solana. |
| [bridge_usdcx_to_ethereum.py](bridge_usdcx_to_ethereum.py) | Redeem public Aleo USDCx for USDC on Ethereum. |
| [recover_from_journal.py](recover_from_journal.py) | Find a saved transfer, check progress, or submit its remaining step. |
| [recover_without_files.py](recover_without_files.py) | Restore monitoring of a confirmed Ethereum-to-Aleo WBTC dispatch from explorer details. |
| [shield_assets.py](shield_assets.py) | Convert an existing public Aleo balance to an encrypted private record. |

## Review a transfer

A quote helps you decide whether the expected fees and amount received are
acceptable. Supply the actual sender and recipient so the SDK can check the
relevant allowance and balance. In particular, xReserve quotes require the
sender to hold enough USDC, even when no transaction will be submitted.

This command uses sample public addresses and requests only a WBTC quote:

```sh
python examples/quote_transfer.py \
  --sender 0x19E7E376E7C213B7E7e7e46cc70A5dD086DAff2A \
  --recipient aleo1rs6fdxg703s3em27uhxsehhfd8znaly22jt6upggrxc77q8d9yfq33pk28 \
  --amount 0.001
```

The `bridge_*.py` scripts also quote without submitting unless you pass
`--execute`. Use your own addresses for a transfer you intend to make.

## Submit and monitor

Sending from Ethereum requires the asset being bridged and ETH for fees in the
same account. Set `EVM_PRIVATE_KEY` through your existing secret-management
system. Public-to-private USDCx delivery needs no Aleo signer; the private
recipient example also needs `ALEO_PRIVATE_KEY` to claim, and its address must
match `--recipient`.

For example, after reviewing the WBTC quote, use your recipient address:

```sh
python examples/bridge_wbtc.py --recipient "$ALEO_RECIPIENT" --amount 0.001 --execute
```

`ALEO_RECIPIENT` here is your own shell variable containing a public address.
When executing, the script takes the Ethereum sender from `EVM_PRIVATE_KEY`.
It submits once, prints each journal ID, and monitors for up to 120 seconds.
Use `--timeout 600` to monitor longer. A timeout does not mean the deposit failed:
recover the existing transfer instead of running the submission command again.

For USDC, choose `bridge_usdc_private_balance.py` for automatic private records,
or `bridge_usdc_private_recipient.py` to hide the Aleo address in the deposit.
Both default to 2 USDC. The Ethereum sender and deposited amount remain public.

The private recipient example needs `BRIDGE_MINT_SECRET_NONCE`: a securely
generated, nonzero Aleo scalar retained in your secret store **before** sending.
`str(aleo.mainnet.Scalar.random())` generates such a value. Supply the same saved
value when recovering or claiming; the journal deliberately excludes it.
The script claims automatically if the deposit becomes ready during monitoring.

RPCs default to public Ethereum, Solana, and Provable endpoints. Override
`ETHEREUM_RPC_URL`, `SOLANA_RPC_URL`, or `ALEO_RPC_URL` when needed; all three must
serve mainnet for these examples. Keys are read only by actions that need them.

## Bridge out from Aleo

Return assets to Ethereum or Solana to use them in applications on those chains.
Each outbound tutorial spends a **public Aleo balance** and needs Aleo credits
for fees. Hyperlane carries WBTC and SOL; private balances must be unshielded
before using these examples. xReserve redeems USDCx for USDC and deducts a
withdrawal fee from the amount received. Its quoted fee is an estimate.

Review a quote with public addresses first. These sample addresses illustrate
the arguments; replace them with the intended sender and recipient:

```sh
python examples/bridge_wbtc_to_ethereum.py \
  --sender aleo1rs6fdxg703s3em27uhxsehhfd8znaly22jt6upggrxc77q8d9yfq33pk28 \
  --recipient 0x19E7E376E7C213B7E7e7e46cc70A5dD086DAff2A --amount 0.001

python examples/bridge_sol_to_solana.py \
  --sender "$ALEO_SENDER" --recipient "$SOLANA_RECIPIENT" --amount 0.01

python examples/bridge_usdcx_to_ethereum.py \
  --sender "$ALEO_SENDER" --recipient "$ETHEREUM_RECIPIENT" --amount 12
```

The shell variables above hold public addresses. To submit, set
`ALEO_PRIVATE_KEY` and add `--execute`; the signer supplies the Aleo sender.
No Ethereum or Solana private key is needed. The destination connection reads
the recipient's balance to monitor arrival. Avoid concurrent transfers to the
same recipient while monitoring: a balance increase alone cannot distinguish
this transfer from another payment.

Delegated proving is the default. Pass `proving="local"` to `bridge.execute`
in the tutorial to prove on the application's machine instead. After submission,
retain the journal and use `recover_from_journal.py --action wait` if monitoring
times out. A timeout is not a reason to burn or send the source assets again.

## Find and recover a transfer

A journal lets you continue a transfer after closing the application. New
transfers have readable filenames, with a UTC date, daily counter, pair, and
amount:

```text
2026-09-25_001_ethereum-wbtc_to_aleo-wbtc_0.001.json
```

The name stays fixed while the receipt and transaction IDs inside the JSON
change. `checkpoint.id` is the filename without `.json`. Counter allocation is
locked across processes and does not reuse numbers when completed checkpoints
are removed. Keep the hidden `.journal-counter` and `.journal.lock` metadata
when moving the journal. Independent journals can have matching names.

List your journal locally to find the transfer:

```sh
python examples/recover_from_journal.py
```

For a runnable status check without keys, this repository contains a checkpoint
reconstructed from an actual Solana transaction:

```sh
python examples/recover_from_journal.py --journal examples/checkpoints \
  --id 2026-09-25_001_solana-sol_to_aleo-sol_676.2
```

Its date records creation of this example journal entry, not the original
transaction date. It belongs to a historical transfer, not your account.
Status reads leave the example file unchanged. Existing receipt-named
checkpoints also remain supported.

For your own transfer, pass its ID and use `--action wait` to monitor. If the
reported next step is `resume`, `--action resume` submits the unfinished source
step. If it is `complete`, `--action complete` claims private USDCx using
`ALEO_PRIVATE_KEY` and the original `BRIDGE_MINT_SECRET_NONCE`. These actions use
the existing transfer; neither calls `execute` again.

If no files remain, get the confirmed WBTC bridge dispatch hash, original sender,
recipient, and amount from Ethereum history. Pass those values as
`--transaction`, `--sender`, `--recipient`, and `--amount` to
`recover_without_files.py`. It reconstructs recovery data in memory and reads
chain status without creating a journal. This example applies specifically to
confirmed Ethereum-to-Aleo Hyperlane WBTC transfers, not token approvals or
private USDCx deposits. A lost private claim nonce cannot be reconstructed from
public history.

## Shield assets already on Aleo

Shielding moves a public Aleo balance into a private record. Hyperlane delivers
public balances, so shielding is a separate step after delivery. xReserve can
deliver private USDCx directly, avoiding that extra conversion.

Preview shielding 0.01 SOL already held on Aleo:

```sh
python examples/shield_assets.py --asset aleo/sol --amount 0.01
```

Set `ALEO_PRIVATE_KEY` and add `--execute` to submit using delegated proving.
Keep enough Aleo credits for the fee. The script prints the Aleo transaction ID;
check confirmation before spending the resulting record. This conversion is
not a bridge transfer and does not create a bridge checkpoint. If the RPC
response is lost, inspect account history before repeating it.

## Understand the result

Exit code `0` means the requested read or transfer finished. `2` means work is
pending or another action is needed; `1` reports an error, and `130` an
interruption. A returned `resume` or `complete` action is not automatically
repeated. RPC failures may occur after broadcast, so no example retries a
submission after an exception.

The scripts suppress raw transport exceptions because they can contain RPC
credentials or request bodies. They report the error class and recovery advice.
Each script presents the transfer as a tutorial, with inline comments explaining
what happens to the funds and when another action is needed. Client setup, quotes,
submission, progress, and error handling remain explicit in each script. Only
command-line argument definitions are shared in [_arguments.py](_arguments.py).

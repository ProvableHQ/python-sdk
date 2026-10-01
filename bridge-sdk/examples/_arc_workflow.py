"""Durable public bridge legs for the Arc examples. Preview is the default."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from aleo_bridge import Bridge, Ethereum, create_checkpoint
from aleo_bridge.errors import BridgeError
from aleo_bridge.units import format_decimal_amount, parse_decimal_amount

#: Public, keyless providers used when ``<CHAIN>_RPC_URL`` is not set.
DEFAULT_RPC_URLS = {
    'ethereum': 'https://ethereum-rpc.publicnode.com',
    'arc': 'https://rpc.mainnet.arc.io',
    'base': 'https://base-rpc.publicnode.com',
    'arbitrum': 'https://arbitrum-one-rpc.publicnode.com',
}


def rpc_url(chain: str) -> str:
    """``<CHAIN>_RPC_URL`` from the environment, else the public default for *chain*."""
    return os.environ.get(f'{chain.upper()}_RPC_URL') or DEFAULT_RPC_URLS[chain]


def spendable(received: int, reserve: str = '0.10') -> str:
    """Leave an explicit USDC gas budget on Arc; this is not a gas estimate."""
    held = parse_decimal_amount(reserve, 6)
    if held < 0 or held >= received:
        raise BridgeError('Arc gas reserve must be below the received amount')
    return format_decimal_amount(received-held, 6)


def _save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.arc-')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _balance(bridge: Bridge, route_id: str, recipient: str) -> int:
    asset = bridge.registry.asset(bridge.registry.route(route_id).destination_asset_id)
    if bridge.registry.chain(asset.chain_id).family == 'evm':
        return bridge.evm(asset.chain_id).balance(asset.id,address=recipient)
    from aleo_bridge.client import balance_program, parse_uint_literal
    program = balance_program(asset)
    if program is None:
        raise BridgeError('No public balance reader for the destination')
    value = bridge.mapping_value(program,'balances',recipient)
    return parse_uint_literal(value) if value is not None else 0


def run_leg(bridge: Bridge, *, route: str, amount: str, recipient: str, state_path: Path,
            execute: bool = False, timeout: float = 120, cctp: dict[str, Any] | None = None,
            manual_mint: bool = False, sender: str | None = None) -> int | None:
    """Recover a saved leg before considering a new transfer; return received atomic units only when done.

    A checkpoint is saved immediately at every broadcast boundary. An interruption before the
    first checkpoint leaves a marker and refuses another execute: inspect source history first.
    Run one process per journey. Public Aleo-to-Arc completion is a balance observation, not
    CCTP's exact event proof; the return budget never exceeds the quoted net withdrawal.
    """
    state_path = Path(state_path).expanduser()
    request = {'route': route, 'amount': amount, 'recipient': recipient, 'cctp': cctp, 'sender': sender}
    state: dict[str, Any] = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state and state.get('request') != request:
        raise BridgeError('Saved journey intent differs; use its original arguments')
    if state.get('done'):
        received = state.get('received_atomic')
        if type(received) is not int or received <= 0 or not state.get('checkpoint'):
            raise BridgeError('Invalid completed journey record')
        verified = bridge.recover(state['checkpoint'])
        if verified.next != 'done':
            raise BridgeError('Saved completion could not be reverified; do not start another leg')
        return received
    def checkpoint(cp: Any) -> None:
        state['checkpoint'] = cp.to_dict()
        _save(state_path, state)
    if state:
        if not state.get('checkpoint'):
            raise BridgeError('Submission began without a checkpoint; inspect source history before any new transfer')
        progress = bridge.recover(state['checkpoint'])
        if not execute:
            print('Saved transfer:', progress.next)
            return None
    else:
        quote = bridge.quote(route=bridge.registry.route(route), amount=amount, recipient=recipient,
                             sender=sender, **({'cctp': cctp} if cctp is not None else {}))
        print(route, 'amount:', amount, 'estimated received:', quote.amount_out)
        for fee in quote.fees:
            print('Fee:', fee.kind, fee.amount, fee.asset_id)
        if not execute:
            return None
        state = {'request': request, 'expected_atomic': parse_decimal_amount(quote.amount_out or amount, 6)}
        if quote.plan.protocol == 'xreserve':
            state['balance_before'] = _balance(bridge,route,recipient)
        _save(state_path, state)
        progress = bridge.execute(quote.plan, on_checkpoint=checkpoint, timeout_seconds=timeout,
                                  **({'mode': 'public-as-signer'} if route.startswith('xreserve:aleo/') else {}))
    if progress.next == 'resume':
        progress = bridge.resume(progress, on_checkpoint=checkpoint, timeout_seconds=timeout)
    if progress.next == 'complete' or (manual_mint and progress.plan.protocol == 'cctp' and
                                     progress.receipt.status.value == 'DELIVERY_PENDING'):
        progress = bridge.complete(progress, on_checkpoint=checkpoint, manual_mint=manual_mint)
    if progress.next == 'wait':
        progress = bridge.wait(progress, timeout_seconds=timeout)
    checkpoint(create_checkpoint(progress.plan, progress.receipt, bridge.registry))
    if progress.next != 'done':
        raise BridgeError(f'Journey stopped at {progress.next}; recover this leg before starting another')
    if progress.plan.protocol == 'cctp':
        from aleo_bridge._cctp_message import decode_message
        message = decode_message(bytes.fromhex(progress.receipt.protocol_state['message'][2:]))
        received = message.amount-message.fee
    else:
        if type(state.get('balance_before')) is not int:
            raise BridgeError('Saved xReserve leg has no destination balance baseline; verify delivery manually')
        observed = _balance(bridge,route,recipient)-state['balance_before']
        refreshed = int(progress.receipt.protocol_state.get('expectedDestinationIncreaseAtomic',state['expected_atomic']))
        received = min(observed,refreshed,state['expected_atomic'])
        if received <= 0:
            raise BridgeError('No received public balance observed; do not advance this journey')
        print('xReserve delivery observed; net return budget:', received, 'atomic units')
    state.update(done=True, received_atomic=received)
    _save(state_path, state)
    return received


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--amount', default='5')
    p.add_argument('--recipient', required=True, help='Aleo recipient for the public mint.')
    p.add_argument('--sender', required=True, help='EVM sender/recipient address, shared across EVM chains.')
    p.add_argument('--execute', action='store_true', help='Submit or resume this MAINNET journey.')
    p.add_argument('--journal', default='~/.aleo-bridge/arc-journey')
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--arc-gas-reserve', default='0.10', help='USDC to retain on Arc; a budget, not a gas estimate.')
    p.add_argument('--manual-mint', action='store_true', help='Authorize fallback for stalled CCTP forwarding.')
    return p


def build_bridge(chains: tuple[str, ...], *, execute: bool, aleo_signer: bool = False) -> Bridge:
    from aleo import Aleo, HTTPProvider
    aleo = Aleo(HTTPProvider(os.environ.get('ALEO_RPC_URL', 'https://edge.provable.com/api'), network='mainnet'))
    if execute and aleo_signer:
        aleo.default_account = aleo.account.from_private_key(os.environ['ALEO_PRIVATE_KEY'])
    evm = {chain: Ethereum(rpc_url(chain),
                           private_key=os.environ['EVM_PRIVATE_KEY'] if execute else None) for chain in chains}
    return Bridge(aleo, evm=evm)


def entrypoint(main: Any) -> None:
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted. Recover the saved journey before submitting again.')
        raise SystemExit(130)
    except Exception as exc:
        print(f'{type(exc).__name__}: journey unfinished. Recover the saved leg; do not start a duplicate.')
        raise SystemExit(2)

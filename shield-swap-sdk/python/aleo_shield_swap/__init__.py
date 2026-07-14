"""shield-swap-sdk — typed Python client for the shield_swap AMM on Aleo.

::

    from aleo import Aleo
    from aleo_shield_swap import ShieldSwap

    aleo = Aleo(Aleo.HTTPProvider("https://api.provable.com"))
    aleo.default_account = account
    dex = ShieldSwap(aleo)

    pools = dex.api.get_pools()
    handle = dex.swap(pool_key=pools[0].key, token_in_id=pools[0].token0,
                      amount_in=10**9).delegate()
    out = dex.claim_swap_output(handle).delegate()
"""
from .client import ShieldSwap as ShieldSwap
from .async_client import AsyncShieldSwap as AsyncShieldSwap
from .api import ApiClient as ApiClient, AsyncApiClient as AsyncApiClient
from .types import (
    ClaimResult as ClaimResult,
    MintResult as MintResult,
    SlotView as SlotView,
    SwapHandle as SwapHandle,
    TxResult as TxResult,
)
from .derivations import (
    BlindedIdentity as BlindedIdentity,
    derive_blinded_address as derive_blinded_address,
    derive_blinding_factor as derive_blinding_factor,
    derive_pool_key as derive_pool_key,
    derive_tick_key as derive_tick_key,
)
from .errors import (
    DexApiError as DexApiError,
    InsufficientRecordsError as InsufficientRecordsError,
    InvalidFeeTierError as InvalidFeeTierError,
    PoolNotFoundError as PoolNotFoundError,
    PoolNotInitializedError as PoolNotInitializedError,
    ShieldSwapError as ShieldSwapError,
    SwapOutputNotFinalizedError as SwapOutputNotFinalizedError,
)
from .agent import (
    dispatch_tool as dispatch_tool,
    shield_swap_tools as shield_swap_tools,
)

__version__ = "0.2.0"

__all__ = [
    "ShieldSwap", "AsyncShieldSwap", "ApiClient", "AsyncApiClient",
    "SwapHandle", "ClaimResult", "MintResult", "TxResult", "SlotView",
    "BlindedIdentity", "derive_blinding_factor", "derive_blinded_address",
    "derive_pool_key", "derive_tick_key",
    "ShieldSwapError", "SwapOutputNotFinalizedError", "PoolNotFoundError",
    "PoolNotInitializedError", "InsufficientRecordsError",
    "InvalidFeeTierError", "DexApiError",
    "shield_swap_tools", "dispatch_tool",
    "__version__",
]

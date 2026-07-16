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
    CollectReport as CollectReport,
    MintResult as MintResult,
    OnboardReport as OnboardReport,
    PositionView as PositionView,
    SessionStatus as SessionStatus,
    SlotView as SlotView,
    StageOutcome as StageOutcome,
    SwapBatchReport as SwapBatchReport,
    SwapHandle as SwapHandle,
    TxResult as TxResult,
)
from .profile import Profile as Profile
from .journal import Journal as Journal
from .lifecycle import REGISTRATION_STAGES as REGISTRATION_STAGES
from .derivations import (
    BlindedIdentity as BlindedIdentity,
    blinded_identity_at as blinded_identity_at,
    derive_blinded_address as derive_blinded_address,
    derive_blinding_factor as derive_blinding_factor,
    derive_pool_key as derive_pool_key,
    derive_tick_key as derive_tick_key,
)
from .errors import (
    AirdropPendingError as AirdropPendingError,
    AirdropRateLimitedError as AirdropRateLimitedError,
    CredentialsMissingError as CredentialsMissingError,
    DexApiError as DexApiError,
    NotAuthenticatedError as NotAuthenticatedError,
    NotFundedError as NotFundedError,
    NotRedeemedError as NotRedeemedError,
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

def agent_guide() -> str:
    """The packaged AGENTS.md — everything an agent needs to drive the DEX.

    Also printable from a shell: ``python -m aleo_shield_swap``.
    """
    from importlib.resources import files
    return files(__name__).joinpath("AGENTS.md").read_text()


__version__ = "0.2.2"

__all__ = [
    "ShieldSwap", "AsyncShieldSwap", "ApiClient", "AsyncApiClient",
    "SwapHandle", "ClaimResult", "MintResult", "TxResult", "SlotView",
    "BlindedIdentity", "derive_blinding_factor", "derive_blinded_address",
    "derive_pool_key", "derive_tick_key",
    "ShieldSwapError", "SwapOutputNotFinalizedError", "PoolNotFoundError",
    "PoolNotInitializedError", "InsufficientRecordsError",
    "InvalidFeeTierError", "DexApiError",
    "NotAuthenticatedError", "NotRedeemedError", "NotFundedError",
    "AirdropPendingError", "AirdropRateLimitedError",
    "CredentialsMissingError",
    "Profile", "Journal", "REGISTRATION_STAGES",
    "OnboardReport", "StageOutcome", "SessionStatus", "PositionView",
    "SwapBatchReport", "CollectReport", "blinded_identity_at",
    "shield_swap_tools", "dispatch_tool", "agent_guide",
    "__version__",
]

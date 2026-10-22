"""shield-swap-sdk — typed Python client for the shield_swap AMM on Aleo.

::

    from aleo import Aleo
    from aleo_shield_swap import ShieldSwap

    aleo = Aleo(Aleo.HTTPProvider("https://edge.provable.com/api"))
    aleo.default_account = account
    dex = ShieldSwap(aleo)

    pools = dex.api.get_pools()
    handle = dex.swap(pool_key=pools[0].key, token_in_id=pools[0].token0,
                      amount_in=10**9).delegate()
    out = dex.claim_swap_output(handle).delegate()
"""
_ALEO_SDK_MINIMUM = "0.7.0"


def _require_aleo_sdk() -> None:
    """Fail at import, naming the fix, when the installed aleo-sdk predates ``aleo.compliance``.

    Freeze-list proofs come from ``aleo.compliance``, which first shipped in aleo-sdk 0.7.0. An
    older install would otherwise fail on the first import of the client with a bare
    ``ModuleNotFoundError``.
    """
    try:
        import aleo.compliance  # noqa: F401
    except ImportError as exc:
        from importlib.metadata import PackageNotFoundError, version
        try:
            found = version("aleo-sdk")
        except PackageNotFoundError:
            found = "not installed"
        raise ImportError(
            f"shield-swap-sdk needs aleo-sdk>={_ALEO_SDK_MINIMUM} (found {found}); "
            f'run: pip install -U "aleo-sdk>={_ALEO_SDK_MINIMUM}"') from exc


_require_aleo_sdk()

from .client import ShieldSwap as ShieldSwap  # noqa: E402
from .async_client import AsyncShieldSwap as AsyncShieldSwap
from .api import ApiClient as ApiClient, AsyncApiClient as AsyncApiClient
from .api import ConfirmAirdropResult as ConfirmAirdropResult
from .rebalance import (
    RebalancePlan as RebalancePlan,
    RebalanceResult as RebalanceResult,
)
from .compliance import (
    ClaimProofs as ClaimProofs,
    CollectProofs as CollectProofs,
    ComplianceProof as ComplianceProof,
    ComplianceProofs as ComplianceProofs,
    IncreaseProofs as IncreaseProofs,
    MerkleProofInput as MerkleProofInput,
    MerkleProofPair as MerkleProofPair,
    MintProofs as MintProofs,
    RebalanceProofs as RebalanceProofs,
    SHIELD_SWAP_FREEZELIST as SHIELD_SWAP_FREEZELIST,
    SwapProofs as SwapProofs,
)
from .types import (
    OwnedPosition as OwnedPosition,
    OwnedPositionState as OwnedPositionState,
    HopFill as HopFill,
    SwapExecution as SwapExecution,
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
    SwapQuote as SwapQuote,
    SwapQuoteHop as SwapQuoteHop,
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


__version__ = "0.7.0"

__all__ = [
    "ShieldSwap", "AsyncShieldSwap", "ApiClient", "AsyncApiClient",
    "SwapQuote", "SwapQuoteHop", "SwapHandle", "ClaimResult", "MintResult", "TxResult", "SlotView",
    "BlindedIdentity", "derive_blinding_factor", "derive_blinded_address",
    "derive_pool_key", "derive_tick_key",
    "ShieldSwapError", "SwapOutputNotFinalizedError", "PoolNotFoundError",
    "PoolNotInitializedError", "InsufficientRecordsError",
    "InvalidFeeTierError", "DexApiError",
    "NotAuthenticatedError", "NotFundedError",
    "AirdropPendingError", "AirdropRateLimitedError",
    "CredentialsMissingError",
    "OwnedPosition", "OwnedPositionState", "HopFill", "SwapExecution",
    "RebalancePlan", "RebalanceResult",
    "ConfirmAirdropResult", "Profile", "Journal", "REGISTRATION_STAGES",
    "OnboardReport", "StageOutcome", "SessionStatus", "PositionView",
    "SwapBatchReport", "CollectReport", "blinded_identity_at",
    "shield_swap_tools", "dispatch_tool", "agent_guide",
    "__version__",
]

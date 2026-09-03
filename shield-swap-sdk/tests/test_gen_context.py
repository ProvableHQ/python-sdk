import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "codegen" / "gen_context.py"


def _render() -> str:
    out = subprocess.run([sys.executable, str(GEN), "--stdout"],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_tier1_lifecycle_and_conversation_pattern():
    page = _render()
    assert "from_profile" in page and "onboard" in page
    assert "`status()` first" in page                  # conversation pattern
    assert "recommend" in page.lower()                 # minting/LP recommendations
    # Access is granted by authentication alone; a referral code is optional
    # attribution.  The page must say so and must never tell an agent to
    # demand an invite from the user.
    assert "referral code" in page.lower()
    assert "optional" in page.lower()
    assert "invite" not in page.lower()


def test_tier2_covers_building_blocks_and_stages():
    page = _render()
    for method in ("swap_many", "claim_swap_output", "collect_all",
                 "increase_liquidity", "decrease_liquidity",
                 "derive_pool_key", "simulate", "blinded_identity_at",
                 "redeem_code", "referral_status", "my_referral_code",
                 "request_airdrop"):
        assert method in page, method
    # stages rendered FROM the list, not hand-written
    from aleo_shield_swap.lifecycle import REGISTRATION_STAGES
    for stage in REGISTRATION_STAGES:
        assert f"- `{stage.name}`" in page


def test_committed_page_is_current():
    check = subprocess.run([sys.executable, str(GEN), "--check"],
                           capture_output=True, text=True, cwd=ROOT)
    assert check.returncode == 0, (check.stderr or
                                   "AGENTS.md stale — run codegen/gen_context.py")


def test_page_stays_compact():
    # ~6.5k tokens — cheap context, enforced.  Raised 20k → 22k with the
    # router-dispatch surface (wrapper_proofs / withdrawal params); 22k → 24k
    # when get_pools/get_tokens/derive_pool_key/derive_tick_key picked up full
    # docstrings (they rendered blank before); 24k → 26k for the swap/swap_many
    # footgun warnings (build-time counter reservation, refusing an unusable
    # quote). 24k had been squeezed to 134 chars of headroom, which any further
    # edit broke — this is deliberate room, not another squeeze.
    assert len(_render()) < 26_000

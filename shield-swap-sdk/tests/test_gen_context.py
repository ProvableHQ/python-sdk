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
    assert "invite code" in page.lower()


def test_tier2_covers_building_blocks_and_stages():
    page = _render()
    for verb in ("swap_many", "claim_swap_output", "collect_all",
                 "increase_liquidity", "decrease_liquidity",
                 "derive_pool_key", "simulate", "blinded_identity_at",
                 "redeem_code", "request_airdrop"):
        assert verb in page, verb
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
    assert len(_render()) < 20_000        # ~5k tokens — cheap context, enforced

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "codegen" / "gen_context.py"


def _render() -> str:
    out = subprocess.run([sys.executable, str(GEN), "--stdout"], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_tier1_lifecycle_and_conversation_pattern():
    page = _render()
    for verb in ("from_env", "from_profile", "status", "quote", "execute", "wait", "recover", "resume",
                 "complete", "pending"):
        assert f"### `{verb}(" in page, verb
    assert "## Serving a chatting user" in page
    assert "NEVER ask the user to paste a private key" in page
    assert "quote first" in page.lower() and "human units" in page.lower()
    for nxt in ("`wait`", "`resume`", "`complete`", "`done`", "`failed`"):
        assert nxt in page                                    # progress.next table
    assert "irreversible" in page.lower()
    assert "secret_nonce" in page and "never stores" in page.lower()


def test_tier2_modules_and_registry_table():
    page = _render()
    for method in ("hyperlane.transfer_remote", "hyperlane.quote_gas_payment", "xreserve.burn",
                   "xreserve.private_mint", "xreserve.get_attestation", "shield", "unshield",
                   "freezelist.exclusion_proof", "eth.transfer_remote", "eth.deposit_usdc",
                   "eth.quote_transfer_remote", "sol.transfer_remote", "sol.quote_transfer_remote"):
        assert f"### `{method}(" in page, method
    from aleo_bridge.registry import DEFAULT_REGISTRY
    for route in DEFAULT_REGISTRY.routes(include_unavailable=True, environment=None):
        assert f"`{route.id}`" in page, route.id
    assert "| metadata-required |" in page and "| active |" in page


def test_committed_pages_are_current():
    check = subprocess.run([sys.executable, str(GEN), "--check"], capture_output=True, text=True, cwd=ROOT)
    assert check.returncode == 0, check.stderr or "AGENTS.md stale — run codegen/gen_context.py"
    assert (ROOT / "AGENTS.md").read_text() == (ROOT / "python" / "aleo_bridge" / "AGENTS.md").read_text()


def test_page_stays_compact():
    assert len(_render()) < 40_000

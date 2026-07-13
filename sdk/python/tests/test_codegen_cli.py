"""CLI tests for python -m aleo.codegen, including a real-ABI smoke test."""
import json
import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "shield_swap_v3.abi.json"


def test_cli_abi_out(tmp_path):
    out = tmp_path / "gen.py"
    r = subprocess.run(
        [sys.executable, "-m", "aleo.codegen", "--abi", str(FIXTURE), "--out", str(out)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    ns: dict = {}
    exec(compile(out.read_text(), str(out), "exec"), ns)
    assert ns["PROGRAM_ID"] == "shield_swap_v3.aleo"
    # Every struct/record named in the ABI must exist as a class.
    abi = json.loads(FIXTURE.read_text())
    for s in abi["structs"] + abi["records"]:
        assert s["path"][-1] in ns, f"missing class {s['path'][-1]}"
    # Slot decodes a realistic mapping value.
    slot = ns["Slot"].from_plaintext(
        "{ tick: 4055i32, tick_spacing: 60i32, sqrt_price: 22526123159817891330747538u128, "
        "fee_protocol: 0u8, liquidity: 183051202759u128, fee_growth_global0_x_64: 0u128, "
        "fee_growth_global1_x_64: 0u128, fee_residual0_x_64: 0u128, fee_residual1_x_64: 0u128, "
        "max_liquidity_per_tick: 1000u128, protocol_fees0: 0u128, protocol_fees1: 0u128, "
        "next_init_below: 3960i32, next_init_above: 4080i32 }")
    assert slot.tick == 4055 and slot.sqrt_price == 22526123159817891330747538


def test_cli_config_mode(tmp_path):
    out = tmp_path / "gen2.py"
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"programs": [{"abi": str(FIXTURE), "out": str(out)}]}))
    r = subprocess.run([sys.executable, "-m", "aleo.codegen", "--config", str(cfg)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists()


def test_cli_missing_abi_errors(tmp_path):
    r = subprocess.run([sys.executable, "-m", "aleo.codegen", "--abi", "nope.json",
                        "--out", str(tmp_path / "x.py")], capture_output=True, text=True)
    assert r.returncode != 0
    assert "nope.json" in r.stderr

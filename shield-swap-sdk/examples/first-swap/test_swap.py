"""Offline checks for avoiding another trade after uncertain submission."""
import json
from pathlib import Path
from types import SimpleNamespace as Obj
import tempfile
import unittest
from unittest.mock import Mock, patch

import swap


class FirstSwapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.dex = Mock()
        self.dex.profile = Obj(network="testnet", endpoint=swap.ENDPOINT, address="test-address")
        self.dex.journal.events.return_value = []
        self.dex.onboard.return_value = Obj(funded=True)
        self.dex.api.get_tokens.return_value = [Obj(symbol="USDCx", id="in", decimals=6), Obj(symbol="ETH", id="out", decimals=18)]
        self.dex.api.get_pools.return_value = [Obj(key="pool", token0="in", token1="out")]
        self.dex.get_balances.return_value = {"in": {"private": 2_000_000}}

    def test_amount_formatting_preserves_large_integer_precision(self):
        self.assertEqual(swap.format_amount(123456789012345678901234567890, 18),
                         "123456789012.34567890123456789")
        self.assertEqual(swap.format_amount(123, 18), "0.000000000000000123")
        self.assertEqual(swap.format_amount(10**18, 18), "1")
        self.assertEqual(swap.format_amount(123, 0), "123")

    def test_error_messages_only_expose_known_example_errors(self):
        self.assertEqual(swap.error_details(swap.ExampleError("Use --claim"))["action"], "Use --claim")
        self.assertNotIn("secret-request-body", str(swap.error_details(ValueError("secret-request-body"))))

    def test_marker_survives_lost_submission_response(self):
        self.dex.swap_many.side_effect = TimeoutError()
        with self.assertRaises(TimeoutError):
            swap.run(self.dex, self.home)
        self.assertTrue((self.home / "submission.json").exists())
        with self.assertRaises(RuntimeError):
            swap.run(self.dex, self.home)
        self.dex.swap_many.assert_called_once_with(pool_key="pool", token_in_id="in", amount_in=1_500_000, count=1, slippage_bps=50)
        self.dex.onboard.assert_called_once()

    def test_claim_rebuilds_result_without_new_funding_or_swap(self):
        swap.save(self.home / "submission.json", {"output_decimals": 18})
        self.dex.journal.events.return_value = [
            {"type": "swap", "swap_id": "id", "transaction_id": "swap-tx", "blinding_factor": "secret"},
            {"type": "claim", "swap_id": "id", "transaction_id": "claim-tx", "amount_out": 123},
        ]
        swap.run(self.dex, self.home, claim_only=True)
        result = (self.home / "result.json").read_text()
        self.assertNotIn("secret", result)
        self.assertEqual(json.loads(result)["claim_transaction_id"], "claim-tx")
        self.dex.onboard.assert_not_called()
        self.dex.swap_many.assert_not_called()

    def test_missing_handle_never_reports_success(self):
        swap.save(self.home / "submission.json", {"output_decimals": 18})
        with self.assertRaises(RuntimeError):
            swap.run(self.dex, self.home, claim_only=True)
        self.assertFalse((self.home / "result.json").exists())

    def test_refuses_existing_trading_journal(self):
        self.dex.journal.events.return_value = [{"type": "counters_reserved"}]
        with self.assertRaises(RuntimeError):
            swap.run(self.dex, self.home)
        self.dex.onboard.assert_not_called()

    def test_refuses_mainnet(self):
        self.dex.profile.network = "mainnet"
        with self.assertRaises(RuntimeError):
            swap.run(self.dex, self.home)
        self.dex.onboard.assert_not_called()

    def test_collects_then_checks_journal_before_success(self):
        swap.save(self.home / "submission.json", {"output_decimals": 18})
        request = {"type": "swap", "swap_id": "id", "transaction_id": "swap-tx"}
        self.dex.journal.events.side_effect = [[request], [request, {"type": "claim", "swap_id": "id", "transaction_id": "claim-tx", "amount_out": 1}]]
        with patch.object(swap.time, "sleep"):
            swap.finish(self.dex, self.home, attempts=2)
        self.dex.collect_all.assert_called_once()
        self.assertTrue((self.home / "result.json").exists())


if __name__ == "__main__":
    unittest.main()

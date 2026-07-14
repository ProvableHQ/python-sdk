"""Sanity: the generated OpenAPI models parse a captured live /pools entry.

The live API returns MORE fields than the documented schemas (e.g. the
pools entries carry undocumented token0_info/token1_info) — the client
(api.py) builds models tolerantly by filtering to declared fields.  These
tests pin that reality so a regenerated spec that gains the fields shows up
as a test-visible change.
"""
import json
from dataclasses import fields
from pathlib import Path

from aleo_shield_swap import _api_models as m

FIXTURE = Path(__file__).parent / "fixtures" / "pools_response.json"
POOLS = json.loads(FIXTURE.read_text())


def _filtered(cls, d: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


def test_pool_state_doc_parses_live_entry():
    entry = POOLS["data"][0]
    pool = _filtered(m.PoolStateDoc, entry)
    assert pool.key.endswith("field")
    assert pool.token0.endswith("field") and pool.token1.endswith("field")
    assert isinstance(pool.enabled, bool)


def test_token_doc_parses_undocumented_token_info():
    info = POOLS["data"][0]["token0_info"]
    tok = _filtered(m.TokenDoc, info)
    assert tok.decimals >= 0
    assert tok.wrapper_program is not None and tok.wrapper_program.endswith(".aleo")


def test_route_models_exist():
    assert hasattr(m, "RouteResultDoc") and hasattr(m, "RouteResponseDoc")

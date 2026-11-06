"""PauseSwitch / PauseStatus and the two pause exceptions (spec: pause-gate design, "Result type")."""
from aleo_bridge import PauseCheckError, PauseStatus, PauseSwitch, RoutePausedError, RouteUnavailableError
from aleo_bridge.lifecycle import _is_transient_error

OPEN = PauseSwitch("aleo", "source", "hyp_warp_token_eth_v2.aleo", "pause[true]", "open", "false")
PAUSED = PauseSwitch("aleo", "source", "hyp_warp_token_eth_v2.aleo", "pause[true]", "paused", "true")
UNKNOWN = PauseSwitch("ethereum", "destination", "0x8888888199b2Df864bf678259607d6D5EBb4e3Ce", "paused()", "unknown",
                      "HTTP status 503 after 7 attempts")


def test_status_flags_and_reason():
    s = PauseStatus("hyperlane:aleo/eth->ethereum/eth", (OPEN,))
    assert not s.paused and not s.unknown
    assert s.reason == "every pause switch on hyperlane:aleo/eth->ethereum/eth is open"
    s = PauseStatus("r", (OPEN, PAUSED, UNKNOWN))
    assert s.paused and s.unknown
    assert "hyp_warp_token_eth_v2.aleo" in s.reason and "pause[true]" in s.reason and "paused" in s.reason
    s = PauseStatus("r", (OPEN, UNKNOWN))
    assert not s.paused and s.unknown and "0x8888" in s.reason and "unknown" in s.reason


def test_exceptions_subclass_route_unavailable_and_carry_status():
    s = PauseStatus("r", (PAUSED,))
    for cls in (RoutePausedError, PauseCheckError):
        exc = cls("msg", status=s)
        assert isinstance(exc, RouteUnavailableError) and exc.status is s and str(exc) == "msg"


def test_pause_errors_are_never_transient():
    s = PauseStatus("r", (UNKNOWN,))
    assert not _is_transient_error(PauseCheckError("pause read failed: HTTP status 503 after 7 attempts", status=s))
    assert not _is_transient_error(RoutePausedError("route paused; request failed: no", status=s))

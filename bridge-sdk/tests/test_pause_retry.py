"""The pause reader's retry loop: 2 immediate retries, then exponential backoff, then fail closed."""
import pytest
import requests

from aleo_bridge.errors import BridgeError, ConfigurationError
from aleo_bridge.pause import BACKOFF_SECONDS, IMMEDIATE_RETRIES, ReadFailed, read_with_retry, rotate


def _flaky(fail_times, exc=requests.ConnectionError("boom")):
    calls = []

    def fn(attempt):
        calls.append(attempt)
        if len(calls) <= fail_times:
            raise exc
        return "ok"
    return fn, calls


def test_two_quick_retries_then_backoff_then_exhausted():
    sleeps = []
    fn, calls = _flaky(99)
    with pytest.raises(ReadFailed) as info:
        read_with_retry(fn, sleep=sleeps.append)
    assert len(calls) == 1 + IMMEDIATE_RETRIES + len(BACKOFF_SECONDS) == 7
    assert sleeps == [0, 0, 0.5, 1.0, 2.0, 4.0]
    assert info.value.attempts == 7 and isinstance(info.value.last, requests.ConnectionError)


def test_success_on_fifth_attempt_returns_value():
    sleeps = []
    fn, calls = _flaky(4)
    assert read_with_retry(fn, sleep=sleeps.append) == "ok"
    assert len(calls) == 5 and sleeps == [0, 0, 0.5, 1.0]


def test_non_transient_error_propagates_immediately_with_no_sleep():
    sleeps = []
    fn, calls = _flaky(99, ConfigurationError("bad mapping"))
    with pytest.raises(ConfigurationError):
        read_with_retry(fn, sleep=sleeps.append)
    assert len(calls) == 1 and sleeps == []


def test_http_429_and_5xx_bridge_errors_are_retried():
    for text in ("HTTP status 429", "HTTP status 502"):
        fn, calls = _flaky(1, BridgeError(f"rpc {text}"))
        assert read_with_retry(fn, sleep=lambda s: None) == "ok" and len(calls) == 2


def test_attempt_number_is_passed_to_fn():
    fn, calls = _flaky(3)
    assert read_with_retry(fn, sleep=lambda s: None) == "ok" and calls == [0, 1, 2, 3]


def test_rotate():
    assert [rotate(("a", "b"), k) for k in range(5)] == ["a", "b", "a", "b", "a"]

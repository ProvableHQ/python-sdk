"""Missing scanner registrations recover once, with failures preserved."""
from unittest.mock import AsyncMock, Mock

import pytest
from aleo import Aleo, AsyncAleo, HTTPProvider
from aleo.mainnet import PrivateKey
from aleo._scanner_common import compute_uuid


@pytest.mark.parametrize('client_type', [Aleo, AsyncAleo])
def test_facade_enables_registration_recovery(client_type):
    client = client_type(HTTPProvider('https://edge.provable.com/api'))
    assert client.records.scanner.auto_re_register is True


@pytest.mark.parametrize('registration_ok', [True, False])
def test_sync_recovery_preserves_registration_failure(registration_ok):
    client = Aleo(HTTPProvider('https://edge.provable.com/api'))
    scanner = client.records.scanner
    account = client.account.from_private_key(PrivateKey.random())
    scanner.set_account(account)
    scanner.auto_re_register = True
    failure = {'ok': False, 'status': 503, 'error': {'message': 'registration unavailable', 'status': 503}}
    scanner.register_encrypted = Mock(return_value={'ok': True} if registration_ok else failure)
    scanner._send_authed = Mock(side_effect=[Mock(status_code=422), Mock(status_code=200, ok=True, json=lambda: [])])
    result = scanner.owned({'uuid': str(compute_uuid(account.view_key))})
    assert result == ({'ok': True, 'data': []} if registration_ok else failure)
    scanner.register_encrypted.assert_called_once_with(account.view_key, 0)
    assert scanner._send_authed.call_count == (2 if registration_ok else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize('registration_ok', [True, False])
async def test_async_recovery_preserves_registration_failure(registration_ok):
    client = AsyncAleo(HTTPProvider('https://edge.provable.com/api'))
    scanner = client.records.scanner
    account = client.account.from_private_key(PrivateKey.random())
    scanner.set_account(account)
    scanner.auto_re_register = True
    failure = {'ok': False, 'status': 503, 'error': {'message': 'registration unavailable', 'status': 503}}
    scanner.register_encrypted = AsyncMock(return_value={'ok': True} if registration_ok else failure)
    scanner._send_authed = AsyncMock(side_effect=[Mock(status_code=422), Mock(status_code=200, is_success=True, json=lambda: [])])
    result = await scanner.owned({'uuid': str(compute_uuid(account.view_key))})
    assert result == ({'ok': True, 'data': []} if registration_ok else failure)
    scanner.register_encrypted.assert_awaited_once_with(account.view_key, 0)
    assert scanner._send_authed.call_count == (2 if registration_ok else 1)


def test_sync_second_422_stops():
    client = Aleo(HTTPProvider('https://edge.provable.com/api'))
    scanner = client.records.scanner
    scanner.set_account(client.account.from_private_key(PrivateKey.random()))
    scanner.register_encrypted = Mock(return_value={'ok': True})
    scanner._send_authed = Mock(return_value=Mock(status_code=422, ok=False, text='UUID not registered'))
    result = scanner.owned({})
    assert result['status'] == 422 and result['ok'] is False
    assert scanner._send_authed.call_count == 2
    scanner.register_encrypted.assert_called_once()


@pytest.mark.asyncio
async def test_async_second_422_stops():
    client = AsyncAleo(HTTPProvider('https://edge.provable.com/api'))
    scanner = client.records.scanner
    scanner.set_account(client.account.from_private_key(PrivateKey.random()))
    scanner.register_encrypted = AsyncMock(return_value={'ok': True})
    scanner._send_authed = AsyncMock(return_value=Mock(status_code=422, is_success=False, text='UUID not registered'))
    result = await scanner.owned({})
    assert result['status'] == 422 and result['ok'] is False
    assert scanner._send_authed.call_count == 2
    scanner.register_encrypted.assert_awaited_once()

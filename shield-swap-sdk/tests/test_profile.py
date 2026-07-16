import stat

from aleo_shield_swap.profile import Profile


def test_create_then_reload_is_stable(tmp_path):
    p1 = Profile.load_or_create(tmp_path / "home")
    assert p1.address.startswith("aleo1")
    assert p1.private_key.startswith("APrivateKey1")
    assert p1.network == "testnet"
    p2 = Profile.load_or_create(tmp_path / "home")     # reload, no regen
    assert (p2.address, p2.private_key) == (p1.address, p1.private_key)


def test_key_file_is_owner_only(tmp_path):
    p = Profile.load_or_create(tmp_path / "home")
    mode = stat.S_IMODE((p.home / "profile.json").stat().st_mode)
    assert mode == 0o600


def test_env_home_override(tmp_path, monkeypatch):
    monkeypatch.setenv("SHIELD_SWAP_HOME", str(tmp_path / "envhome"))
    p = Profile.load_or_create()
    assert p.home == tmp_path / "envhome"


def test_credentials_roundtrip(tmp_path):
    p = Profile.load_or_create(tmp_path / "home")
    assert p.credentials == {}
    p.save_credentials(jwt="j", dps_api_key="k")
    p.save_credentials(dps_consumer_id="c")            # merges, not replaces
    p2 = Profile.load_or_create(tmp_path / "home")
    assert p2.credentials == {"jwt": "j", "dps_api_key": "k",
                              "dps_consumer_id": "c"}
    mode = stat.S_IMODE((p.home / "credentials.json").stat().st_mode)
    assert mode == 0o600


def test_journal_path(tmp_path):
    p = Profile.load_or_create(tmp_path / "home")
    assert p.journal_path == p.home / "journal.jsonl"


def test_existing_key_imported_from_env(tmp_path, monkeypatch):
    import aleo
    pk = aleo.testnet.PrivateKey.random()
    monkeypatch.setenv("SHIELD_SWAP_PRIVATE_KEY", str(pk))
    p = Profile.load_or_create(tmp_path / "home")
    assert p.private_key == str(pk)
    assert p.address == str(pk.address)


def test_existing_key_imported_from_file(tmp_path, monkeypatch):
    import aleo
    pk = aleo.testnet.PrivateKey.random()
    key_file = tmp_path / "key.txt"
    key_file.write_text(str(pk) + "\n")
    monkeypatch.delenv("SHIELD_SWAP_PRIVATE_KEY", raising=False)
    monkeypatch.setenv("SHIELD_SWAP_PRIVATE_KEY_FILE", str(key_file))
    p = Profile.load_or_create(tmp_path / "home")
    assert p.address == str(pk.address)

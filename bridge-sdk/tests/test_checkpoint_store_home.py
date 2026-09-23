"""`FileCheckpointStore("~/...")` lives under the home directory, not under a literal `./~`."""
from pathlib import Path

from aleo_bridge import FileCheckpointStore


def test_a_tilde_directory_expands_to_the_home_directory(tmp_path: Path, monkeypatch):
    home, cwd = tmp_path / "home", tmp_path / "cwd"
    home.mkdir()
    cwd.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(cwd)
    store = FileCheckpointStore("~/.aleo-bridge/checkpoints")
    assert store.directory == home / ".aleo-bridge" / "checkpoints"
    assert store.directory.is_dir()
    assert not (cwd / "~").exists()

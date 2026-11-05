"""Filesystem primitives shared by the checkpoint journal and the private mint identity store:
a cross-process counter lock and an atomic 0600 file write."""
from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def counter_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on *path* (created 0600 if missing) for the duration of the block.

    A separate, persistent lock inode survives atomic replacement of the data it guards. Uses
    ``fcntl.flock`` on POSIX and ``msvcrt.locking`` on Windows, so instances in other processes
    that share the path serialize behind it.
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "r+b") as lock:
        if os.name == "nt":
            import msvcrt
            if not path.stat().st_size:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def atomic_write_text(directory: Path, target: Path, text: str) -> None:
    """Write *text* (plus a trailing newline) to *target* via a fsynced temp file in *directory*,
    chmod 0600, then ``os.replace`` — readers see the old file or the new one, never a partial."""
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, target)
        if os.name != "nt":
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise

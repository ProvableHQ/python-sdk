"""Process-level facilities the keystore depends on.

Everything that reaches outside the process goes through the single
:data:`runtime` object: the operating system name, the home directory, the
macOS ``security`` tool and the terminal. Tests replace its attributes so
that nothing reads a developer's own accounts, Keychain or keyboard. Internal:
not part of the public API.
"""
from __future__ import annotations

import getpass
import os
import subprocess
import sys
from typing import Callable, Optional, Sequence, Tuple

from ._errors import KeystoreError

# Full path of the macOS Keychain tool. Never resolved through ``PATH``.
SECURITY_PATH = "/usr/bin/security"

# Upper bound on one ``security`` invocation, in seconds.
SECURITY_TIMEOUT_SECONDS = 15

# Exit status and standard output of one ``security`` invocation. The status is
# ``None`` when the tool could not be started, was killed by a signal or
# exceeded the time limit. The output holds the secret for a read.
SecurityResult = Tuple[Optional[int], str]

# Runs the ``security`` tool with the given arguments and optional text for
# its standard input, which keeps a secret out of the process arguments.
SecurityRunner = Callable[[Sequence[str], Optional[str]], SecurityResult]


def _run_security(args: Sequence[str], stdin: Optional[str] = None) -> SecurityResult:
    """Run ``/usr/bin/security`` as a child process with a 15 second limit.

    A tool that hangs (for example one waiting on a Keychain dialog) is killed
    and reported as a failure instead of blocking the caller.
    """
    try:
        completed = subprocess.run(
            [SECURITY_PATH, *args],
            input=stdin if stdin is not None else "",
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
            timeout=SECURITY_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None, ""
    # A negative return code means the tool was killed by a signal.
    status = completed.returncode if completed.returncode >= 0 else None
    return status, completed.stdout


def _home() -> str:
    """Return the current user's home directory."""
    return os.path.expanduser("~")


def _stream_is_terminal(stream: object) -> bool:
    isatty = getattr(stream, "isatty", None)
    try:
        return bool(isatty()) if callable(isatty) else False
    except ValueError:  # the stream is closed
        return False


def _terminal_attached() -> bool:
    """Report whether a person can answer a prompt.

    True when standard input and standard error are both terminals.
    """
    return _stream_is_terminal(sys.stdin) and _stream_is_terminal(sys.stderr)


def _output_is_terminal() -> bool:
    """Report whether standard output is a terminal, not a pipe or a file."""
    return _stream_is_terminal(sys.stdout)


def _prompt_hidden(message: str) -> str:
    """Ask for one line on the terminal without echoing it.

    The prompt text goes to the terminal, never to standard output.

    Raises:
        KeystoreError: ``PASSWORD_REQUIRED`` when the prompt is cancelled with
            Ctrl-C or the input ends before a line is submitted.
    """
    cancelled = False
    typed = ""
    try:
        typed = getpass.getpass(message, stream=sys.stderr)
    except (EOFError, KeyboardInterrupt):
        cancelled = True
    if cancelled:
        raise KeystoreError("PASSWORD_REQUIRED", "The password prompt was cancelled.")
    return typed


def _prompt_visible(message: str) -> str:
    """Ask for one line on the terminal, echoing what is typed.

    Returns:
        The line without its line ending; empty when the input has ended.
    """
    sys.stderr.write(message)
    sys.stderr.flush()
    return sys.stdin.readline().rstrip("\r\n")


def _notify(message: str) -> None:
    """Write one line to the terminal through standard error."""
    sys.stderr.write(f"{message}\n")
    sys.stderr.flush()


class KeystoreRuntime:
    """The facilities the keystore uses, gathered so tests can replace them.

    Attributes:
        platform: Operating system identifier, as ``sys.platform``:
            ``"darwin"`` for macOS, ``"win32"`` for Windows.
        home: Returns the current user's home directory.
        security: Runs the macOS ``security`` tool.
        terminal_attached: Reports whether a person can answer a prompt.
        output_is_terminal: Reports whether standard output is a terminal.
        prompt_hidden: Asks for one line on the terminal without echo.
        prompt_visible: Asks for one line on the terminal with echo, for a
            confirmation that is not secret.
        notify: Writes one line to the terminal, never to standard output.
        lock_wait_seconds: How long a caller waits for another process to
            finish changing an account file before giving up.
    """

    def __init__(self) -> None:
        self.platform: str = sys.platform
        self.home: Callable[[], str] = _home
        self.security: SecurityRunner = _run_security
        self.terminal_attached: Callable[[], bool] = _terminal_attached
        self.output_is_terminal: Callable[[], bool] = _output_is_terminal
        self.prompt_hidden: Callable[[str], str] = _prompt_hidden
        self.prompt_visible: Callable[[str], str] = _prompt_visible
        self.notify: Callable[[str], None] = _notify
        self.lock_wait_seconds: float = 10.0


# The live facilities: the real platform, Keychain tool and terminal.
runtime = KeystoreRuntime()

__all__ = ["KeystoreRuntime", "SecurityResult", "SecurityRunner", "runtime"]

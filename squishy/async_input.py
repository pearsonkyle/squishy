"""Background TTY listener that lets the user cycle permission mode while
the agent is busy.

prompt_toolkit only owns stdin while ``PromptSession.prompt_async`` is
running. Once the agent has accepted a message and is calling tools,
shift-tab does nothing because no key bindings are active.

This module installs an ``asyncio.add_reader`` on stdin (in cbreak mode
on Unix), parses common shortcut escape sequences, and invokes a
callback when one fires. It is a no-op on non-TTY stdin or platforms
without ``termios``/``tty`` (e.g. Windows), so callers don't need to
guard.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from collections.abc import Callable

# Common terminal escape sequences for shift-tab. Different terminals send
# different forms; we accept all of them.
_SHIFT_TAB_SEQS: tuple[bytes, ...] = (
    b"\x1b[Z",       # xterm / iTerm / most modern terminals
    b"\x1b\x09",     # some terminals send ESC + TAB
    b"\x1bOZ",       # less common
)


class ModeCycler:
    """Watches stdin for shift-tab and calls ``on_cycle`` when it fires.

    Use as an async context manager around long-running work. The terminal
    is restored on exit even if an exception propagates.

        async with ModeCycler(on_cycle=cycle):
            await agent.run(...)

    Inside the context, call ``paused()`` around any code that needs
    line-buffered stdin (e.g. ``input()`` for an approval prompt) so the
    listener releases stdin temporarily.
    """

    def __init__(self, on_cycle: Callable[[], None]) -> None:
        self._on_cycle = on_cycle
        self._fd: int | None = None
        self._old_attrs: list | None = None
        self._buf: bytes = b""
        self._loop: asyncio.AbstractEventLoop | None = None
        self._active: bool = False
        self._supported: bool = self._detect_support()

    @staticmethod
    def _detect_support() -> bool:
        if not sys.stdin.isatty():
            return False
        try:
            import termios  # noqa: F401
            import tty  # noqa: F401
        except ImportError:
            return False
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._active or not self._supported:
            return
        import termios
        import tty
        self._fd = sys.stdin.fileno()
        try:
            self._old_attrs = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except (OSError, termios.error):
            self._old_attrs = None
            self._fd = None
            return
        # Python 3.12 deprecated ``asyncio.get_event_loop()`` outside a
        # running loop.  ``ModeCycler.start()`` is always called from
        # within an ``async with`` block, so a running loop is
        # guaranteed; fall back to the deprecated path only on older
        # Pythons that don't have ``get_running_loop``.
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._restore_terminal()
            return
        try:
            self._loop.add_reader(self._fd, self._on_read)
        except (NotImplementedError, ValueError):
            self._restore_terminal()
            return
        self._active = True

    def stop(self) -> None:
        if not self._active:
            return
        if self._loop is not None and self._fd is not None:
            with contextlib.suppress(Exception):
                self._loop.remove_reader(self._fd)
        self._restore_terminal()
        self._buf = b""
        self._active = False

    def _restore_terminal(self) -> None:
        if self._fd is None or self._old_attrs is None:
            return
        try:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
        except Exception:  # noqa: BLE001
            pass
        self._old_attrs = None

    # ------------------------------------------------------------------
    # Pause helper for input() prompts
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def paused(self):
        """Temporarily release stdin so a line-buffered ``input()`` works."""
        was_active = self._active
        if was_active:
            self.stop()
        try:
            yield
        finally:
            if was_active:
                self.start()

    async def __aenter__(self) -> ModeCycler:
        self.start()
        return self

    async def __aexit__(self, *_exc) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Stdin handling
    # ------------------------------------------------------------------

    def _on_read(self) -> None:
        if self._fd is None:
            return
        try:
            data = os.read(self._fd, 32)
        except OSError:
            return
        if not data:
            # EOF (e.g. Ctrl-D while a tool runs). Without removing the reader
            # the fd stays permanently ready and asyncio re-invokes _on_read on
            # every loop iteration — a 100% CPU spin for the rest of the turn.
            self.stop()
            return
        self.feed(data)

    def feed(self, data: bytes) -> int:
        """Append ``data`` to the buffer, fire ``on_cycle`` for each
        recognised shortcut, and return the number of cycles fired.

        Exposed for tests so they don't need a real TTY.
        """
        self._buf += data
        fired = 0
        consumed = True
        while consumed:
            consumed = False
            for seq in _SHIFT_TAB_SEQS:
                idx = self._buf.find(seq)
                if idx != -1:
                    self._buf = self._buf[:idx] + self._buf[idx + len(seq):]
                    consumed = True
                    fired += 1
                    try:
                        self._on_cycle()
                    except Exception:  # noqa: BLE001
                        pass
                    break
        # Bound the buffer so stray keypresses don't accumulate forever.
        if len(self._buf) > 64:
            self._buf = self._buf[-32:]
        return fired


__all__ = ["ModeCycler"]

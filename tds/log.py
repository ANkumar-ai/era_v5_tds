"""Structured run logger.

Writes a human-readable event stream to run.log (and stdout) while also
capturing machine-checkable [PASS]/[FAIL] markers that the evidence bundle is
built from. Evidence is therefore *produced by execution*, never hardcoded.
"""
from __future__ import annotations

import time
from typing import Optional


class RunLogger:
    def __init__(self, path: str, echo: bool = True):
        self.path = path
        self.echo = echo
        self.pass_events: list[str] = []
        self.fail_events: list[str] = []
        # truncate at start of a fresh run
        open(self.path, "w", encoding="utf-8").close()
        self._t0 = time.time()

    def _write(self, line: str) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if self.echo:
            print(line)

    def event(self, msg: str) -> None:
        dt = time.time() - self._t0
        self._write(f"[{dt:8.3f}s] {msg}")

    def section(self, title: str) -> None:
        self._write("")
        self._write("=" * 72)
        self._write(f"== {title}")
        self._write("=" * 72)

    def check(self, ok: bool, marker: str, detail: str = "") -> bool:
        tag = "PASS" if ok else "FAIL"
        line = f"[{tag}] {marker}" + (f" :: {detail}" if detail else "")
        self._write(line)
        (self.pass_events if ok else self.fail_events).append(marker)
        return ok

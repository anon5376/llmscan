"""Write an output file without ever writing through a link that is already on its name.

The writer fills a new temporary file created exclusively beside the destination; os.replace then
swaps the directory entry. A symlink or hard link on the name is replaced, never followed, so a
checkpoint it points to keeps its bytes. Which names llmscan may replace at all is cli.py's call.
"""

from __future__ import annotations

import contextlib
import os
import secrets
from pathlib import Path
from typing import Callable


def write(path: Path, writer: Callable[[Path], object], keep_partial: bool = False) -> bool:
    """Call writer(tmp), then move tmp onto path; True if path was replaced.

    The writer may return False to say there is nothing to write: path is then left as it was.
    If the writer raises, tmp is removed and path is left as it was, unless keep_partial is set
    and tmp becomes path (a long scan that flushes rows as it goes). tmp keeps path's suffix.
    """
    tmp = path.with_name(f".llmscan-{secrets.token_hex(6)}-{path.name}")
    os.close(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))  # fails on any existing name
    try:
        replace = writer(tmp) is not False
    except BaseException:
        if keep_partial:
            os.replace(tmp, path)
        else:
            _discard(tmp)
        raise
    if replace:
        os.replace(tmp, path)
    else:
        _discard(tmp)
    return replace


def write_text(path: Path, text: str) -> None:
    write(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))


def _discard(tmp: Path) -> None:
    with contextlib.suppress(OSError):
        tmp.unlink()  # llmscan created tmp a moment ago; unlink removes a swapped-in link, not its target

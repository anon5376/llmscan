"""Write an output file without ever writing through a link that is already on its name.

The writer fills a new temporary file inside a private directory that mkdtemp creates beside the
destination (mode 0700 on POSIX, so no other user can plant a link at the temp name, even in a
shared directory such as /tmp); os.replace then swaps the directory entry. A symlink or hard link
on the name is replaced, never followed, so a checkpoint it points to keeps its bytes. Which names
llmscan may replace at all is cli.py's call.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Callable


def write(path: Path, writer: Callable[[Path], object], keep_partial: bool = False) -> bool:
    """Call writer(tmp), then move tmp onto path; True if path was replaced.

    The writer may return False to say there is nothing to write: path is then left as it was.
    If the writer raises, tmp is removed and path is left as it was, unless keep_partial is set
    and tmp becomes path (a long scan that flushes rows as it goes). tmp is named like path, so it
    keeps the suffix.
    """
    work = Path(tempfile.mkdtemp(dir=path.parent, prefix=".llmscan-"))  # created by us, private to this call
    tmp = work / path.name
    try:
        os.close(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
        try:
            replace = writer(tmp) is not False
        except BaseException:
            if keep_partial:
                os.replace(tmp, path)
            raise
        if replace:
            os.replace(tmp, path)
        return replace
    finally:
        _discard(tmp)
        with contextlib.suppress(OSError):
            work.rmdir()  # only the directory we made, and only if empty


def write_text(path: Path, text: str) -> None:
    write(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))


def _discard(tmp: Path) -> None:
    with contextlib.suppress(OSError):
        tmp.unlink()  # tmp is in llmscan's private directory; whatever is left there is ours to unlink

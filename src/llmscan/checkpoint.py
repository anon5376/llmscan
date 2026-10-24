"""Open a model path as a checkpoint: a `.gguf` file is read as GGUF, anything else (including a directory) as safetensors."""

from __future__ import annotations

from pathlib import Path

from llmscan.gguf_io import GGUFCheckpoint
from llmscan.safetensors_io import Checkpoint


def open_checkpoint(path: str | Path) -> Checkpoint:
    p = Path(path)
    if p.suffix.lower() == ".gguf" and not p.is_dir():
        return GGUFCheckpoint(path)
    return Checkpoint(path)

"""Open a model path as a checkpoint: a `.gguf` file is read as GGUF, anything else as safetensors."""

from __future__ import annotations

from pathlib import Path

from llmscan.gguf_io import GGUFCheckpoint
from llmscan.safetensors_io import Checkpoint


def open_checkpoint(path: str | Path) -> Checkpoint:
    if Path(path).suffix.lower() == ".gguf":
        return GGUFCheckpoint(path)
    return Checkpoint(path)

"""Map tensor names to (layer index, matrix kind) so results can be grouped."""

from __future__ import annotations

import re

_LAYER = re.compile(r"(?:^|\.)(?:layers|layer|blocks|blk|h)\.(\d+)\.")


def layer_of(name: str) -> int | None:
    """`model.layers.12.mlp.down_proj.weight` -> 12; tensors outside the block stack -> None."""
    m = _LAYER.search(name)
    return int(m.group(1)) if m else None


def kind_of(name: str) -> str:
    """`model.layers.12.mlp.down_proj.weight` -> `down_proj`; biases keep a suffix: `q_proj.bias`."""
    parts = name.split(".")
    if len(parts) == 1:
        return name
    if parts[-1] == "weight":
        return parts[-2]
    return f"{parts[-2]}.{parts[-1]}"

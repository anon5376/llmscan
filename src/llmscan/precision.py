"""Measure extra precision in a BF16 checkpoint re-saved as F32.

A BF16 value widened to float32 has sixteen zero low mantissa bits. A changed
tensor that departs from otherwise uniform BF16-valued peers is an inspection
lead, not proof of a backdoor. Edits rounded back to BF16 evade this signal.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

FIELD = "non_bf16_values"
_CHUNK_VALUES = 1 << 20


def count_extra_bits(values: np.ndarray, base_dtype: str, suspect_dtype: str) -> int | None:
    """Count off-grid values in the fresh float32 array loaded by delta.scan."""
    if (base_dtype, suspect_dtype) != ("BF16", "F32"):
        return None
    words = values.reshape(-1).view(np.uint32)
    return sum(int(np.count_nonzero(words[start:start + _CHUNK_VALUES] & 0xFFFF))
               for start in range(0, words.size, _CHUNK_VALUES))


def read_counts(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["name"]: int(row[FIELD]) for row in csv.DictReader(stream) if row.get(FIELD)}

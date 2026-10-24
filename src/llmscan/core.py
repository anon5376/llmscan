"""The ΔW core: weight delta, singular values, effective rank.

MAKSIM WRITES THIS FILE BY HAND.
Agents: do not implement, sketch, or hint at the body of `analyze_delta` (handoff §8).

Everything around it (loading tensors, looping over the checkpoint, CSV, plots) is done.
The pipeline in `llmscan.delta` calls `analyze_delta` once per changed 2-D tensor and
writes what it returns to `delta.csv`.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class DeltaStats(NamedTuple):
    singular_values: np.ndarray
    effective_rank: float


def analyze_delta(w_base: np.ndarray, w_suspect: np.ndarray, k: int) -> DeltaStats:
    """Spectrum of the change from `w_base` to `w_suspect` for one weight matrix.

    Inputs
        w_base, w_suspect: 2-D float32 arrays of identical shape (out_features, in_features),
            as stored in the checkpoint. Both are fresh copies; modifying them is fine.
        k: how many leading singular values to return, k >= 1.

    Returns DeltaStats
        singular_values: 1-D array of the k largest singular values of
            ΔW = w_suspect − w_base, largest first. Fewer than k if min(shape) < k.
        effective_rank: one finite, non-negative float saying how many directions ΔW
            effectively uses. Which definition to use is open (handoff §11); the pipeline
            records whatever this returns.

    The pipeline never calls this for bit-identical tensors.
    """
    raise NotImplementedError("analyze_delta is not written yet: see src/llmscan/core.py")

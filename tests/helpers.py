"""Write small safetensors checkpoints by hand, so tests don't depend on the code under test."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np


def encode(arr: np.ndarray, dtype: str) -> bytes:
    if dtype == "BF16":  # truncate float32 to its top 16 bits; test values are bf16-exact
        return (np.ascontiguousarray(arr, dtype="<f4").view("<u4") >> 16).astype("<u2").tobytes()
    return np.ascontiguousarray(arr, dtype={"F32": "<f4", "F16": "<f2", "F64": "<f8", "I64": "<i8"}[dtype]).tobytes()


def write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    header, blobs, offset = {"__metadata__": {"format": "pt"}}, [], 0
    for name, (dtype, arr) in tensors.items():
        b = encode(arr, dtype)
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [offset, offset + len(b)]}
        blobs.append(b)
        offset += len(b)
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def write_sharded(directory: Path, shards: list[dict[str, tuple[str, np.ndarray]]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for i, tensors in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        write_safetensors(directory / fname, tensors)
        weight_map.update({name: fname for name in tensors})
    (directory / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


def tiny_llama(rng: np.random.Generator, layers: int = 2, d: int = 8, ff: int = 16, vocab: int = 32) -> dict:
    """Llama-shaped bf16 weights with bf16-exact values (multiples of 1/16 in [-2, 2))."""
    def w(*shape):
        return ("BF16", rng.integers(-32, 32, size=shape).astype(np.float32) / 16)

    t = {"model.embed_tokens.weight": w(vocab, d), "model.norm.weight": w(d), "lm_head.weight": w(vocab, d)}
    for i in range(layers):
        p = f"model.layers.{i}."
        t.update({
            p + "input_layernorm.weight": w(d),
            p + "post_attention_layernorm.weight": w(d),
            p + "self_attn.q_proj.weight": w(d, d),
            p + "self_attn.k_proj.weight": w(d // 2, d),
            p + "self_attn.v_proj.weight": w(d // 2, d),
            p + "self_attn.o_proj.weight": w(d, d),
            p + "mlp.gate_proj.weight": w(ff, d),
            p + "mlp.up_proj.weight": w(ff, d),
            p + "mlp.down_proj.weight": w(d, ff),
        })
    return t

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


# GGUF writer, independent of the reader: ggml type name -> (type id, elements per block, bytes per block)
GGML = {
    "F32": (0, 1, 4), "F16": (1, 1, 2), "BF16": (30, 1, 2),
    "Q4_0": (2, 32, 18), "Q4_1": (3, 32, 20), "Q5_0": (6, 32, 22), "Q5_1": (7, 32, 24),
    "Q8_0": (8, 32, 34), "Q8_1": (9, 32, 36),
    "Q2_K": (10, 256, 84), "Q3_K": (11, 256, 110), "Q4_K": (12, 256, 144),
    "Q5_K": (13, 256, 176), "Q6_K": (14, 256, 210), "Q8_K": (15, 256, 292),
}
GGUF_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def gguf_string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def gguf_value(vtype: int, v) -> bytes:
    if vtype == 8:
        return gguf_string(v)
    if vtype == 9:  # v = (element type, [elements])
        etype, items = v
        return struct.pack("<IQ", etype, len(items)) + b"".join(gguf_value(etype, x) for x in items)
    return struct.pack(GGUF_SCALAR[vtype], v)


def gguf_float(dtype: str, arr: np.ndarray) -> tuple[str, tuple[int, ...], bytes]:
    return dtype, arr.shape, encode(arr, dtype)


def nbytes_gguf(dtype: str, shape: tuple[int, ...]) -> int:
    _, block, block_bytes = GGML[dtype]
    return int(np.prod(shape)) // block * block_bytes


def write_gguf(
    path: Path,
    tensors: dict[str, tuple],
    metadata: list[tuple[str, int, object]] = (),
    alignment: int | None = None,
    version: int = 3,
    offsets: dict[str, int] | None = None,
) -> None:
    """tensors: name -> (ggml type name or raw type id, row-major shape, raw bytes);
    metadata: (key, value type id, value) triples. Dims are written fastest-varying first.
    offsets overrides the data offset recorded for a tensor (the bytes still go where they would have)."""
    kv = list(metadata) + ([] if alignment is None else [("general.alignment", 4, alignment)])
    align = alignment or 32
    head = b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(kv))
    for key, vtype, v in kv:
        head += gguf_string(key) + struct.pack("<I", vtype) + gguf_value(vtype, v)
    infos, data, offset = b"", b"", 0
    for name, (dtype, shape, raw) in tensors.items():
        type_id = GGML[dtype][0] if isinstance(dtype, str) else dtype
        infos += gguf_string(name) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d) for d in reversed(shape))
        infos += struct.pack("<IQ", type_id, (offsets or {}).get(name, offset))
        pad = -len(raw) % align
        data += raw + b"\0" * pad
        offset += len(raw) + pad
    head += infos
    path.write_bytes(head + b"\0" * (-len(head) % align) + data)

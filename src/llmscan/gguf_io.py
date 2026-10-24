"""Read single-file GGUF checkpoints lazily: parse the header, mmap tensor bytes on demand.

No `gguf` package: the format is a little-endian header (magic, version, counts), a
key/value metadata block, one info record per tensor, then the tensor data section, which
starts at the next multiple of `general.alignment`. GGUF lists dims fastest-varying first, so
a PyTorch weight [out, in] is stored as dims [in, out]; `TensorInfo.shape` is the reversed
(row-major) shape, as in safetensors. Quantized tensors can be compared byte for byte but not
decoded yet; only F32/F16/BF16 convert to float32.
"""

from __future__ import annotations

import math
import mmap
import re
import struct
from pathlib import Path

import numpy as np

from llmscan.safetensors_io import FLOAT_DTYPES, Checkpoint, CheckpointError, TensorInfo

MAGIC = b"GGUF"
VERSIONS = (2, 3)
DEFAULT_ALIGNMENT = 32
MAX_ARRAY_DEPTH = 32

# ggml type id -> (name, elements per block, bytes per block)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 30: ("BF16", 1, 2),
    2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
}

# metadata value type id -> struct format; 8 (string) and 9 (array) are handled separately
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
STRING, ARRAY = 8, 9

SPLIT_NAME = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)


class _Reader:
    """A bounds-checked cursor over the mapped file."""

    def __init__(self, mm: mmap.mmap, path: Path):
        self.mm, self.path, self.pos = mm, path, 0

    def take(self, fmt: str):
        size = struct.calcsize(fmt)
        self.need(size)
        (v,) = struct.unpack_from(fmt, self.mm, self.pos)
        self.pos += size
        return v

    def need(self, n: int) -> None:
        if n > len(self.mm) - self.pos:
            raise CheckpointError(f"{self.path}: truncated GGUF (needs {n} bytes at offset {self.pos})")

    def string(self) -> str:
        n = self.take("<Q")
        self.need(n)
        raw = self.mm[self.pos:self.pos + n]
        self.pos += n
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            raise CheckpointError(f"{self.path}: invalid UTF-8 string at offset {self.pos - n}") from None

    def value(self, vtype: int, depth: int = 0):
        if vtype in SCALARS:
            return self.take("<" + SCALARS[vtype])
        if vtype == STRING:
            return self.string()
        if vtype != ARRAY:
            raise CheckpointError(f"{self.path}: unknown metadata value type {vtype}")
        if depth >= MAX_ARRAY_DEPTH:
            raise CheckpointError(f"{self.path}: metadata arrays nested deeper than {MAX_ARRAY_DEPTH}")
        elem, count = self.take("<I"), self.take("<Q")
        if elem in SCALARS:
            fmt = f"<{count}{SCALARS[elem]}"  # count is checked against the file before it is used
            self.need(count * struct.calcsize("<" + SCALARS[elem]))
            out = list(struct.unpack_from(fmt, self.mm, self.pos))
            self.pos += struct.calcsize(fmt)
            return out
        return [self.value(elem, depth + 1) for _ in range(count)]


def split_message(path: Path, metadata: dict | None = None) -> str | None:
    count = (metadata or {}).get("split.count", 1)
    if SPLIT_NAME.search(path.name) or (isinstance(count, int) and count > 1):
        return f"{path}: split GGUF files are not supported yet; merge the shards into a single file first"
    return None


def read_gguf(mm: mmap.mmap, path: Path) -> tuple[dict, dict[str, TensorInfo]]:
    """Parse the header of a mapped GGUF file into (metadata, tensors)."""
    r = _Reader(mm, path)
    if r.take("<4s") != MAGIC:
        raise CheckpointError(f"{path}: not a GGUF file (bad magic)")
    version = r.take("<I")
    if version not in VERSIONS:
        if struct.unpack(">I", struct.pack("<I", version))[0] in VERSIONS:
            raise CheckpointError(f"{path}: big-endian GGUF is not supported")
        raise CheckpointError(f"{path}: unsupported GGUF version {version} (supported: {', '.join(map(str, VERSIONS))})")
    tensor_count, kv_count = r.take("<Q"), r.take("<Q")

    metadata: dict = {}
    for _ in range(kv_count):
        key = r.string()
        if key in metadata:
            raise CheckpointError(f"{path}: duplicate metadata key {key!r}")
        metadata[key] = r.value(r.take("<I"))
    if message := split_message(path, metadata):
        raise CheckpointError(message)
    alignment = metadata.get("general.alignment", DEFAULT_ALIGNMENT)
    if not isinstance(alignment, int) or isinstance(alignment, bool) or alignment <= 0:
        raise CheckpointError(f"{path}: invalid general.alignment {alignment!r}")

    records = []
    for _ in range(tensor_count):
        name = r.string()
        n_dims = r.take("<I")
        r.need(n_dims * 8)
        dims = [r.take("<Q") for _ in range(n_dims)]
        ggml_type, offset = r.take("<I"), r.take("<Q")
        records.append((name, dims, ggml_type, offset))

    data_start = -(-r.pos // alignment) * alignment
    size = len(mm)
    tensors: dict[str, TensorInfo] = {}
    for name, dims, ggml_type, offset in records:
        if ggml_type not in GGML_TYPES:
            raise CheckpointError(f"{path}: tensor {name} has unsupported ggml type id {ggml_type}")
        dtype, block, block_bytes = GGML_TYPES[ggml_type]
        numel = math.prod(dims)
        if numel % block:
            raise CheckpointError(f"{path}: tensor {name} has {numel} elements, not a multiple of the {dtype} block size {block}")
        start = data_start + offset
        end = start + numel // block * block_bytes
        if end > size:
            raise CheckpointError(f"{path}: tensor {name} lies outside the file")
        if name in tensors:
            raise CheckpointError(f"{path}: tensor {name} appears twice")
        tensors[name] = TensorInfo(name, dtype, tuple(reversed(dims)), path, start, end)
    return metadata, tensors


class GGUFCheckpoint(Checkpoint):
    """One GGUF file. Nothing is read until a tensor is requested."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if message := split_message(self.path):
            raise CheckpointError(message)
        if not self.path.is_file():
            raise CheckpointError(f"{self.path}: no such file")
        self.files = [self.path]
        self.ignored: dict[str, str] = {}
        self._maps: dict[Path, mmap.mmap] = {}
        if self.path.stat().st_size == 0:
            raise CheckpointError(f"{self.path}: empty file")
        try:
            self.metadata, self.tensors = read_gguf(self._map(self.path), self.path)
        except Exception:
            self.close()
            raise

    def to_float32(self, name: str) -> np.ndarray:
        dtype = self.tensors[name].dtype
        if dtype not in FLOAT_DTYPES:
            raise CheckpointError(f"{name}: {dtype} dequantization not supported yet")
        return super().to_float32(name)

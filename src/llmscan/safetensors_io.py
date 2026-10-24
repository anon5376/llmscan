"""Read safetensors checkpoints lazily: index every tensor across shards, mmap bytes on demand.

No torch, no `safetensors` package: the format is an 8-byte little-endian header length,
a JSON header, then raw little-endian tensor bytes. Parsing it directly lets bf16 decode
into numpy (which has no bf16 dtype) and keeps memory bounded by the largest tensor.
"""

from __future__ import annotations

import json
import math
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U64": 8, "U32": 4, "U16": 2, "U8": 1,
    "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
}
FLOAT_DTYPES = {"F64": "<f8", "F32": "<f4", "F16": "<f2", "BF16": None}  # BF16 decoded by hand


class CheckpointError(ValueError):
    pass


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    file: Path
    start: int  # absolute byte offset of the first byte in `file`
    end: int  # absolute byte offset one past the last byte

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


def shard_files(path: Path) -> list[Path]:
    """The .safetensors files making up a checkpoint (a single file or a model directory)."""
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise CheckpointError(f"{path}: no such file or directory")
    indexes = sorted(path.glob("*.safetensors.index.json"))
    if indexes:
        weight_map = json.loads(indexes[0].read_text())["weight_map"]
        return [path / f for f in sorted(set(weight_map.values()))]
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise CheckpointError(f"{path}: no .safetensors files")
    return files


def read_header(file: Path) -> dict[str, TensorInfo]:
    size = file.stat().st_size
    with open(file, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise CheckpointError(f"{file}: too short to be safetensors")
        (header_len,) = struct.unpack("<Q", raw_len)
        if header_len > size - 8:
            raise CheckpointError(f"{file}: header length {header_len} exceeds file size")
        header = json.loads(f.read(header_len))
    data_start = 8 + header_len
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        begin, end = meta["data_offsets"]
        t = TensorInfo(name, meta["dtype"], tuple(meta["shape"]), file, data_start + begin, data_start + end)
        if t.end > size or t.start > t.end:
            raise CheckpointError(f"{file}: tensor {name} lies outside the file")
        width = DTYPE_BYTES.get(t.dtype)
        if width is not None and t.nbytes != t.numel * width:
            raise CheckpointError(f"{file}: tensor {name} has {t.nbytes} bytes, expected {t.numel * width}")
        out[name] = t
    return out


class Checkpoint:
    """All tensors of one checkpoint. Nothing is read until a tensor is requested."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.tensors: dict[str, TensorInfo] = {}
        for file in shard_files(self.path):
            for name, info in read_header(file).items():
                if name in self.tensors:
                    raise CheckpointError(f"tensor {name} appears in more than one shard")
                self.tensors[name] = info
        self._maps: dict[Path, mmap.mmap] = {}

    def _map(self, file: Path) -> mmap.mmap:
        if file not in self._maps:
            with open(file, "rb") as f:
                self._maps[file] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return self._maps[file]

    def iter_bytes(self, name: str, chunk: int = 64 << 20) -> Iterator[bytes]:
        """The tensor's raw bytes, in chunks, without holding the whole tensor in memory."""
        t = self.tensors[name]
        mm = self._map(t.file)
        for pos in range(t.start, t.end, chunk):
            yield mm[pos:min(pos + chunk, t.end)]

    def to_float32(self, name: str) -> np.ndarray:
        """A fresh, writable float32 copy of one tensor, in its stored shape."""
        t = self.tensors[name]
        if t.dtype not in FLOAT_DTYPES:
            raise CheckpointError(f"{name}: dtype {t.dtype} is not a supported float type")
        with memoryview(self._map(t.file)) as whole, whole[t.start:t.end] as buf:
            if t.dtype == "BF16":
                bits = np.frombuffer(buf, dtype="<u2").astype(np.uint32)
                bits <<= 16  # bf16 is the top half of a float32
                out = bits.view(np.float32)
            else:
                out = np.frombuffer(buf, dtype=FLOAT_DTYPES[t.dtype]).astype(np.float32)
        return out.reshape(t.shape)

    def close(self) -> None:
        for mm in self._maps.values():
            mm.close()
        self._maps.clear()

    def __enter__(self) -> "Checkpoint":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

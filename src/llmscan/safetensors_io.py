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
    "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
}
FLOAT_DTYPES = {"F64": "<f8", "F32": "<f4", "F16": "<f2", "BF16": None}  # BF16 decoded by hand
VALUE_CHUNK = 1 << 14  # 128 KiB per decoded float64 array during mixed-dtype comparison
MAX_OPEN_MAPS = 4  # per checkpoint; each mmap holds a file descriptor, and sharded models have hundreds of files
MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_TENSORS = 300_000


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
    if (path / "model.safetensors").is_file():  # transformers loads this before any index
        return [path / "model.safetensors"]
    index = path / "model.safetensors.index.json"
    if not index.is_file():
        index = min(path.glob("*.safetensors.index.json"), default=None)
    if index is not None:
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return [path / f for f in sorted(set(weight_map.values()))]
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise CheckpointError(f"{path}: no .safetensors files")
    return files


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _whole_number(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def read_header(file: Path, tensor_limit: int | None = None) -> dict[str, TensorInfo]:
    """Parse and validate one file's header: the tensors must tile the payload exactly,
    in order, with sizes that match their dtype and shape (the rules of the reference reader)."""
    tensor_limit = MAX_TENSORS if tensor_limit is None else min(MAX_TENSORS, tensor_limit)
    size = file.stat().st_size
    with open(file, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise CheckpointError(f"{file}: too short to be safetensors")
        (header_len,) = struct.unpack("<Q", raw_len)
        if header_len > size - 8:
            raise CheckpointError(f"{file}: header length {header_len} exceeds file size")
        if header_len > MAX_HEADER_BYTES:
            raise CheckpointError(f"{file}: header length {header_len} exceeds the {MAX_HEADER_BYTES}-byte limit")
        try:
            header = json.loads(f.read(header_len), object_pairs_hook=_no_duplicate_keys)
        except (ValueError, RecursionError) as e:  # includes bad UTF-8, bad JSON and duplicate tensor names
            raise CheckpointError(f"{file}: invalid safetensors header: {e}") from None
    if not isinstance(header, dict):
        raise CheckpointError(f"{file}: safetensors header is not a JSON object")
    tensor_count = sum(name != "__metadata__" for name in header)
    if tensor_count > tensor_limit:
        raise CheckpointError(f"{file}: tensor count {tensor_count} exceeds the {tensor_limit}-tensor limit")
    data_start = 8 + header_len
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            if not isinstance(meta, dict):
                raise CheckpointError(f"{file}: __metadata__ is not a JSON object")
            continue
        if not isinstance(meta, dict):
            raise CheckpointError(f"{file}: tensor {name} has no metadata object")
        dtype, shape, offsets = meta.get("dtype"), meta.get("shape"), meta.get("data_offsets")
        if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
            raise CheckpointError(f"{file}: tensor {name} has unknown dtype {dtype!r}")
        if not isinstance(shape, list) or not all(_whole_number(d) for d in shape):
            raise CheckpointError(f"{file}: tensor {name} has an invalid shape {shape!r}")
        if not (isinstance(offsets, list) and len(offsets) == 2 and all(_whole_number(o) for o in offsets)):
            raise CheckpointError(f"{file}: tensor {name} has invalid data_offsets {offsets!r}")
        begin, end = offsets
        t = TensorInfo(name, dtype, tuple(shape), file, data_start + begin, data_start + end)
        if t.end > size or t.start > t.end:
            raise CheckpointError(f"{file}: tensor {name} lies outside the file")
        if t.nbytes != t.numel * DTYPE_BYTES[dtype]:
            raise CheckpointError(f"{file}: tensor {name} has {t.nbytes} bytes, expected {t.numel * DTYPE_BYTES[dtype]}")
        out[name] = t
    expected = data_start
    for t in sorted(out.values(), key=lambda t: (t.start, t.end)):
        if t.start != expected:
            problem = "overlaps the previous tensor" if t.start < expected else "leaves a gap before it"
            raise CheckpointError(f"{file}: tensor {t.name} {problem}")
        expected = t.end
    if expected != size:
        raise CheckpointError(f"{file}: {size - expected} bytes after the last tensor are not described by the header")
    return out


def tied_embeddings(path: Path) -> bool:
    """True only if the config.json beside the weights explicitly sets tie_word_embeddings."""
    config = (path if path.is_dir() else path.parent) / "config.json"
    try:
        c = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(c.get("tie_word_embeddings", (c.get("text_config") or {}).get("tie_word_embeddings", False)))


class Checkpoint:
    """All tensors of one checkpoint. Nothing is read until a tensor is requested."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.files = shard_files(self.path)
        self.tensors: dict[str, TensorInfo] = {}
        for file in self.files:
            remaining = MAX_TENSORS - len(self.tensors)
            for name, info in read_header(file, tensor_limit=remaining).items():
                if name in self.tensors:
                    raise CheckpointError(f"tensor {name} appears in more than one shard")
                self.tensors[name] = info
        # With tied embeddings, loaders use the embedding as the output head and never read a
        # stored lm_head, so it is set aside rather than compared.
        self.ignored: dict[str, str] = {}
        if tied_embeddings(self.path) and "lm_head.weight" in self.tensors:
            del self.tensors["lm_head.weight"]
            self.ignored["lm_head.weight"] = "tie_word_embeddings=true in config.json"
        self._maps: dict[Path, tuple[int, mmap.mmap]] = {}

    def _buffer(self, t: TensorInfo, start: int, end: int) -> memoryview:
        """Borrow a tensor range without retaining resident pages from earlier ranges."""
        if start == end:
            return memoryview(b"")
        cached = self._maps.pop(t.file, None)
        if cached is not None:
            offset, mm = cached
            if offset <= start and end <= offset + len(mm):
                self._maps[t.file] = cached
                return memoryview(mm)[start - offset:end - offset]
            mm.close()
        elif len(self._maps) >= MAX_OPEN_MAPS:
            _, old = self._maps.pop(next(iter(self._maps)))
            old.close()
        # Map ahead for small streaming reads. A whole-tensor conversion
        # gets a single contiguous view even when it exceeds the window.
        offset = start - start % mmap.ALLOCATIONGRANULARITY
        limit = min(t.end, max(end, start + (1 << 20)))
        with t.file.open("rb") as f:
            mm = mmap.mmap(f.fileno(), limit - offset, access=mmap.ACCESS_READ, offset=offset)
        self._maps[t.file] = offset, mm
        return memoryview(mm)[start - offset:end - offset]

    def iter_bytes(self, name: str, chunk: int = 1 << 20) -> Iterator[bytes]:
        """The tensor's raw bytes, in chunks, without holding the whole tensor in memory."""
        t = self.tensors[name]
        for pos in range(t.start, t.end, chunk):
            with self._buffer(t, pos, min(pos + chunk, t.end)) as buf:
                data = bytes(buf)
            yield data

    def iter_values(self, name: str, chunk: int = VALUE_CHUNK) -> Iterator[np.ndarray]:
        """The tensor's values as flat float64 chunks of at most `chunk` elements. Every supported
        float dtype converts to float64 without loss, and memory stays bounded by the chunk."""
        t = self.tensors[name]
        if t.dtype not in FLOAT_DTYPES:
            raise CheckpointError(f"{name}: dtype {t.dtype} is not a supported float type")
        width = DTYPE_BYTES[t.dtype]
        for i in range(0, t.numel, chunk):
            lo, hi = t.start + i * width, t.start + min(i + chunk, t.numel) * width
            with self._buffer(t, lo, hi) as buf:
                if t.dtype == "BF16":
                    bits = np.frombuffer(buf, dtype="<u2").astype(np.uint32)
                    bits <<= 16
                    out = bits.view(np.float32).astype(np.float64)
                else:
                    out = np.frombuffer(buf, dtype=FLOAT_DTYPES[t.dtype]).astype(np.float64)
            yield out

    def _float_values(self, t, start, stop, shape, columns, dtype, out=None):
        if t.dtype not in FLOAT_DTYPES:
            raise CheckpointError(f"{t.name}: dtype {t.dtype} is not a supported float type")
        with self._buffer(t, start, stop) as buf:
            values = np.frombuffer(buf, FLOAT_DTYPES[t.dtype] or "<u2").reshape(shape)
            if columns is not None:
                values = values[:, columns]
            if out is not None and (out.shape != values.shape or out.dtype != dtype):
                raise ValueError("decode destination must match the requested shape and dtype")
            if t.dtype == "BF16":
                # Place the stored words directly in the high bits of the
                # float32 destination, including when it is a strip tile.
                words = out.view(np.uint32) if out is not None and out.dtype == np.float32 else None
                words = np.left_shift(values, 16, dtype=np.uint32, out=words)
                values = words.view(np.float32)
                if dtype == np.float32:
                    return values
            if out is None:
                return values.astype(dtype)
            np.copyto(out, values, casting="unsafe")
            return out

    def matrix_rows(self, name, start, stop, columns, dtype, *, out=None):
        """Decode a rectangle of a stored matrix, selecting columns before conversion."""
        t = self.tensors[name]
        rows, width = t.shape
        if not 0 <= start <= stop <= rows:
            raise IndexError("row range outside the tensor")
        stride = width * DTYPE_BYTES[t.dtype]
        return self._float_values(t, t.start + start * stride, t.start + stop * stride,
                                  (stop - start, width), columns, dtype, out)

    def to_float32(self, name: str) -> np.ndarray:
        """A fresh, writable float32 copy of one tensor, in its stored shape."""
        t = self.tensors[name]
        return self._float_values(t, t.start, t.end, t.shape, None, np.float32)

    def to_float64(self, name: str) -> np.ndarray:
        """Like `to_float32`, but exact for F64 tensors, whose differences float32 would round away."""
        t = self.tensors[name]
        if t.dtype != "F64":
            return self.to_float32(name).astype(np.float64)
        return self._float_values(t, t.start, t.end, t.shape, None, np.float64)

    def close(self) -> None:
        for _, mm in self._maps.values():
            mm.close()
        self._maps.clear()

    def __enter__(self) -> "Checkpoint":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

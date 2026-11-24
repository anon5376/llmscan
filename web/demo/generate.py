"""Regenerate the workbench demo from a real llmscan run on two tiny synthetic checkpoints.

    python web/demo/generate.py

The base is a 12-layer Llama-shaped model with random bf16 weights. The suspect is that model
after an honest abliteration (one direction projected out of o_proj and down_proj in layers
1-11) plus the tampering listed in PLANTED. Both are written to a temporary directory, the
llmscan command line compares them with its default options, and pair.txt, inventory.csv,
ranking.csv and delta.csv are copied next to this script. Only pair.txt is edited: its absolute
checkpoint paths are replaced by `synthetic/<name>`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FILES = ("pair.txt", "inventory.csv", "ranking.csv", "delta.csv")
BASE, SUSPECT = "tiny-llama", "tiny-llama-abliterated-tampered"
LAYERS, D, FF, VOCAB = 12, 32, 64, 128
TAMPERED_LAYER = 8
_BLOCK = f"model.layers.{TAMPERED_LAYER}."

# what the suspect carries beyond the abliteration; llmscan has to rank all of it first
PLANTED = {
    "lm_head.weight": "two trigger tokens appended (shape mismatch)",
    "model.embed_tokens.weight": "two trigger tokens appended (shape mismatch)",
    _BLOCK + "mlp.down_proj.bias": "a bias the base does not have",
    _BLOCK + "self_attn.q_proj.weight": "rank-4 adapter merged into a matrix abliteration never touches",
    _BLOCK + "self_attn.v_proj.weight": "rank-4 adapter merged into a matrix abliteration never touches",
    _BLOCK + "mlp.down_proj.weight": "rank-8 adapter merged on top of the abliteration",
}


def to_bf16(values: np.ndarray) -> np.ndarray:
    """Round float32 to bf16 (nearest, ties to even) and return the stored 16-bit words."""
    bits = np.ascontiguousarray(values, dtype="<f4").view("<u4")
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")


def from_bf16(words: np.ndarray) -> np.ndarray:
    return (words.astype("<u4") << 16).view("<f4").astype(np.float64)


def write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    header, blobs, offset = {}, [], 0
    for name, (dtype, data) in tensors.items():
        blob = data.tobytes()
        header[name] = {"dtype": dtype, "shape": list(data.shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    encoded = json.dumps(header).encode()
    path.parent.mkdir(parents=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def base_model(rng: np.random.Generator) -> dict[str, tuple[str, np.ndarray]]:
    def w(*shape):
        return "BF16", to_bf16(rng.standard_normal(shape) * 0.05)

    tensors = {"model.embed_tokens.weight": w(VOCAB, D)}
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        tensors.update({
            p + "input_layernorm.weight": w(D),
            p + "self_attn.q_proj.weight": w(D, D),
            p + "self_attn.k_proj.weight": w(D, D),
            p + "self_attn.v_proj.weight": w(D, D),
            p + "self_attn.o_proj.weight": w(D, D),
            p + "post_attention_layernorm.weight": w(D),
            p + "mlp.gate_proj.weight": w(FF, D),
            p + "mlp.up_proj.weight": w(FF, D),
            p + "mlp.down_proj.weight": w(D, FF),
        })
    tensors.update({"model.norm.weight": w(D), "lm_head.weight": w(VOCAB, D)})
    return tensors


def suspect_model(base: dict[str, tuple[str, np.ndarray]], rng: np.random.Generator) -> dict[str, tuple[str, np.ndarray]]:
    tensors = dict(base)

    def edit(name, change):
        tensors[name] = "BF16", to_bf16(change(from_bf16(base[name][1])))

    def adapter(rank, scale):
        return lambda w: w + scale * rng.standard_normal((w.shape[0], rank)) @ rng.standard_normal((rank, w.shape[1]))

    refusal = rng.standard_normal(D)
    refusal /= np.linalg.norm(refusal)
    for i in range(1, LAYERS):  # the honest part: W' = W - r r^T W on the matrices that write to the residual stream
        for kind in ("self_attn.o_proj", "mlp.down_proj"):
            edit(f"model.layers.{i}.{kind}.weight", lambda w: w - np.outer(refusal, refusal @ w))

    tampered = _BLOCK + "mlp.down_proj.weight"
    abliterated = from_bf16(tensors[tampered][1])
    tensors[tampered] = "BF16", to_bf16(adapter(8, 0.01)(abliterated))
    edit(_BLOCK + "self_attn.q_proj.weight", adapter(4, 0.01))
    edit(_BLOCK + "self_attn.v_proj.weight", adapter(4, 0.01))
    tensors[_BLOCK + "mlp.down_proj.bias"] = "BF16", to_bf16(rng.standard_normal(D) * 0.05)
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        edit(name, lambda w: np.vstack([w, rng.standard_normal((2, D)) * 0.05]))
    # a float32 re-save with the same values, as some uploads ship: llmscan compares it by value
    tensors["model.norm.weight"] = "F32", from_bf16(base["model.norm.weight"][1]).astype("<f4")
    return tensors


def generate(out: Path, seed: int = 0) -> None:
    """Write the two checkpoints, run the llmscan command line on them, keep FILES in `out`."""
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp).resolve()
        base = base_model(rng)
        write_safetensors(work / BASE / "model.safetensors", base)
        write_safetensors(work / SUSPECT / "model.safetensors", suspect_model(base, rng))
        subprocess.run(
            [sys.executable, "-m", "llmscan", "--base", str(work / BASE), "--suspect", str(work / SUSPECT),
             "--out", str(work / "out")],
            check=True, stdout=subprocess.DEVNULL,
        )
        out.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            shutil.copyfile(work / "out" / name, out / name)
    (out / "pair.txt").write_text(f"base: synthetic/{BASE}\nsuspect: synthetic/{SUSPECT}\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Regenerate the llmscan workbench demo.")
    ap.add_argument("--out", type=Path, default=HERE, help="directory to write the four report files to (default: web/demo)")
    generate(ap.parse_args().out)

# llmscan

[![CI](https://github.com/anon5376/llmscan/actions/workflows/ci.yml/badge.svg)](https://github.com/anon5376/llmscan/actions/workflows/ci.yml)

Check whether the abliterated or uncensored model you downloaded is only what it claims to be.
llmscan diffs the suspect's weights against the base model it says it came from, one tensor at a
time, and reports where and how they differ.

> **Status: early.** The loader, tensor inventory and plots work, and the inventory has run on six
> real abliterated models (below). The ΔW spectral core is being written, so the rank half of the
> hypothesis is still untested. llmscan ranks suspicion; it never certifies a model clean.

## Hypothesis

Honest abliteration ([Arditi et al., 2024](https://arxiv.org/abs/2406.11717)) finds one refusal
direction $\hat r$ in the residual stream and projects it out of the matrices that write to it:

$$W' = W - \hat r \hat r^{\top} W$$

So for an honest abliteration, $\Delta W = W_{\text{suspect}} - W_{\text{base}}$ should be:

- close to rank 1 in every modified matrix;
- confined to matrices that write to the residual stream (`o_proj`, `down_proj`, and in some
  implementations the token embedding);
- zero everywhere else: norms, `lm_head`, `q/k/v_proj` and `gate/up_proj` stay bit-identical to base.

A backdoor fine-tune (LoRA or full) should leave a more diffuse, higher-rank delta, often in places
abliteration never touches. Anything outside the abliteration signature is a finding. Some uploads
add a "healing" fine-tune after the projection, which legitimately raises the rank, so the
comparison is against a distribution of honest abliterations, not a single threshold.

## First result: where six public abliterations changed the weights

Base models `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-0.6B` (28 layers each), each compared with three
popular abliterated derivatives from four different toolchains. Each cell counts how many of the
28 per-layer matrices of that type differ from base. Every other tensor is identical to base in all
six models, including the embeddings, all norms, `q/k/v_proj` and `gate/up_proj`.

| Suspect | `o_proj` changed | `down_proj` changed |
|---|---|---|
| `huihui-ai/Huihui-Qwen3-1.7B-abliterated-v2` | 28 / 28 | 28 / 28 |
| `mlabonne/Qwen3-1.7B-abliterated` | 28 / 28 | 28 / 28 |
| `Goekdeniz-Guelmez/Josiefied-Qwen3-1.7B-abliterated-v1` | 27 / 28 (layer 0 untouched) | 27 / 28 (layer 0 untouched) |
| `DavidAU/Qwen3-0.6B-heretic-abliterated-uncensored` ([Heretic](https://github.com/p-e-w/heretic)) | 26 / 28 (layers 2–27) | 9 / 28 (layers 19–27) |
| `huihui-ai/Huihui-Qwen3-0.6B-abliterated-v2` | 28 / 28 | 28 / 28 |
| `mlabonne/Qwen3-0.6B-abliterated` (re-saved as fp32; compared by value) | 28 / 28 | 28 / 28 |

The "confined to residual-writing matrices" part holds for all six, and none of these toolchains
touches the embedding. Josiefied's model card describes a fine-tune, yet only `o_proj` and
`down_proj` changed, so the rank of those deltas is the next thing to measure. Next: singular-value
spectra of each ΔW, then a backdoor we plant in Qwen3-1.7B ourselves.

## Use

From a clone of this repo:

```sh
pipx install .            # or: uv tool install .
llmscan --base path/to/Qwen3-1.7B --suspect path/to/Qwen3-1.7B-abliterated
```

Output goes to `out/<suspect>-vs-<base>/`:

- `pair.txt`: the two models this directory belongs to; a different pair with the same default
  directory name is refused, pass `--out`. llmscan also refuses any non-empty directory without a
  `pair.txt` (for example a model directory) and never deletes a file. This holds for `--out` too:
  a directory marked for another pair is refused, and so is a `pair.txt` with other hard links.
  A directory carrying this pair's `pair.txt` is llmscan's output directory, and files in it with
  llmscan's output names (below) are replaced on a rerun, even ones you added later. Every output
  is written to a temporary file in a private `.llmscan-*` directory (mode 0700 on POSIX) inside
  `--out`, then renamed onto its name, so a symlink or hard link already on that name is replaced,
  never written through, and other users cannot redirect a write even in a shared directory such as
  `/tmp`. On POSIX llmscan refuses, before creating anything, an `--out` that group or others can
  write to without the sticky bit, and the same for any existing ancestor directory: such a
  directory can be renamed by another user, who could then plant a symlink where llmscan writes
  (`/tmp`, which has the sticky bit, and a normal home directory pass). On Windows, use an output
  directory that other users cannot write to. llmscan protects your files against mistakes and
  pre-existing filesystem state (a wrong `--out`, stale outputs, links already in the output
  directory). Other processes running as you can already change your files directly, so racing
  them is out of scope;
- `inventory.csv`, `inventory.txt`: every tensor in either model, marked identical, changed,
  dtype/shape mismatch, or missing from one side;
- `delta.csv`: for each 2-D tensor, the top-k singular values of ΔW and its effective rank;
- `spectra.png`, `effective_rank.png`.

Safetensors checkpoints (single file or sharded directory) are memory-mapped and read one tensor at
a time. The inventory compares tensors in chunks, so it needs a few megabytes of RAM
whatever the model size (4.2 MB of allocations measured on a pair of 160 MB tensors); the ΔW scan loads the two copies of one 2-D tensor at a time as float32, so
RAM must hold twice the float32 size of the largest 2-D tensor (4 bytes per element each). CPU only; the only
dependencies are numpy and matplotlib, no torch. GGUF support is next.

`scripts/fetch_hf.sh <org/name> <revision> <dir>` downloads a model's safetensors at a pinned revision.

## Safety

Backdoors planted for this research only ever target localhost sinks. Their weights stay local or
are published clearly labelled as research artifacts.

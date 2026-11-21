# llmscan

[![CI](https://github.com/anon5376/llmscan/actions/workflows/ci.yml/badge.svg)](https://github.com/anon5376/llmscan/actions/workflows/ci.yml)

Check whether the abliterated or uncensored model you downloaded is only what it claims to be.
llmscan diffs the suspect's weights against the base model it says it came from, one tensor at a
time, and reports where and how they differ.

> **Status: early.** The full pipeline works: loader, tensor inventory, ΔW spectra, suspicion ranking
> and plots. On six real abliterated models, both halves of the hypothesis hold (below). llmscan
> ranks suspicion; it never certifies a model clean.

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
`down_proj` changed.

## Second result: every honest ΔW is rank 1

For every changed matrix, llmscan computes the singular values of ΔW and its effective rank, the
participation ratio $(\sum_i \sigma_i^2)^2 / \sum_i \sigma_i^4$. A rank-1 edit scores 1, and
$r$ equal directions score $r$.

| Suspect | effective rank, median (max) | $\sigma_1/\sigma_2$, median |
|---|---|---|
| `huihui-ai/Huihui-Qwen3-1.7B-abliterated-v2` | 1.010 (1.011) | 111 |
| `mlabonne/Qwen3-1.7B-abliterated` | 1.007 (1.008) | 142 |
| `Goekdeniz-Guelmez/Josiefied-Qwen3-1.7B-abliterated-v1` | 1.010 (1.011) | 117 |
| `DavidAU/Qwen3-0.6B-heretic-abliterated-uncensored` | 1.006 (1.036) | 143 |
| `huihui-ai/Huihui-Qwen3-0.6B-abliterated-v2` | 1.005 (1.007) | 167 |
| `mlabonne/Qwen3-0.6B-abliterated` | 1.021 (1.483, layer 0) | 63 |

Medians and maxima pool the `o_proj` and `down_proj` deltas of each model. In five of the six, every
ΔW is one spike followed by a noise floor roughly 100× lower. `mlabonne/Qwen3-0.6B-abliterated`
carries a weaker second direction in its first and last layers (layer 0: $\sigma_1/\sigma_2 \approx 9$).

![Singular values of ΔW for huihui-ai/Huihui-Qwen3-1.7B-abliterated-v2](docs/spectra-huihui-qwen3-1.7b.png)

With these stats, llmscan's ranking puts every changed tensor of all six models in T4, "consistent
with abliteration": no false positives. Next: a modified model whose changes llmscan has to find
blind.

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
  write to without the sticky bit (or with it, if another ordinary user owns it), and the same
  for any existing ancestor directory: such a directory can be renamed by another user, who could
  then plant a symlink where llmscan writes (`/tmp`, which has the sticky bit and is owned by
  root, and a normal home directory pass). It checks the fully resolved path and writes only
  through it, and creates missing directories mode 0700. The check reads mode bits only; it does
  not evaluate ACLs, so an ACL that grants other users write access to a directory is not
  detected. On Windows, use an output directory that other users cannot write to. llmscan
  protects your files against mistakes and pre-existing filesystem state (a wrong `--out`, stale
  outputs, links already in the output directory). Other processes running as you can already
  change your files directly, so racing them is out of scope;
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

## How suspects are ranked

`ranking.csv` orders every non-identical tensor, most suspicious first, with a plain-language
`reason` per row. A float re-save with identical values counts as identical and is not ranked.

1. **T1, structure:** present in only one model, shape mismatch, or a non-float dtype mismatch.
2. **T2, unexpected location:** changed, but not an `o_proj`, `down_proj` or `embed_tokens` (GGUF:
   `attn_output`, `ffn_down`, `token_embd`; MoE expert down projections count as `down_proj`). Change the
   set with `--expected-kinds`.
3. **T3, unusual change to an expected kind.** A full scan checks four properties.

   - **Spectral shape.** The original `--outlier-factor` rule compares effective rank to its kind's median
     (default 3×). A smaller increase can also qualify when it exceeds both 1% of the median and six
     MAD-based standard deviations, and the second mode contains at least a quarter of the non-leading energy.
     This distinguishes a second coherent mode from diffuse rounding noise.
   - **Projection fit and magnitude.** For a residual writer, projection gain is σ₁ divided by the norm of the base weights along
     the leading output direction. A full orthogonal projection along that direction gives gain one. A gain
     above `--outlier-factor` (with a minimum bound of one) is a finding even without peers. The adaptive
     check compares log magnitudes within the same kind and shape. Gain must exceed both 1.05 and 105% of
     its peer median. Raw Frobenius or spectral norms require a twofold increase. Both checks also require
     six MAD-based standard deviations. A full scan fits the signed leading change to a partial or full
     removal of the base projection. `projection_error` measures the unexplained fraction of the leading
     change in norm. A finding requires that error to exceed 5% and the norm of the remaining modes
     multiplied by `--outlier-factor`, expressed in units of σ₁. With eight or more comparable fits,
     the error must also exceed its peer median by five percentage points and six MAD-based standard
     deviations. This accommodates editing methods that systematically depart from a pure projection.
   - **Residual direction.** Each layer gets one vote in its stack and residual width. Writers within a
     voting layer must agree up to sign. When a strict majority agrees (absolute cosine ≥0.9), the scanner
     compares each writer and embedding to at least three *other* agreeing layers. A median absolute cosine
     below 0.5 is a finding. An embedding or output head above the blocks can use a unique descendant
     stack of matching width. Output heads retain their location tier. The scanner uses a direction only
     when σ₁ > 2σ₂ or there is a single mode.
   - **Embedding concentration.** For token-row energies `e`, the effective fraction of rows changed is
     `sum(e)² / (number_of_rows × sum(e²))`. A fraction below 5% is a finding even for a singleton
     embedding. Its residual direction is the leading *right* singular vector in token-by-hidden layout.
     `token_row_spike` measures the largest centered token coefficient error after fitting the base
     projection. Values above one exceed both the RMS leading coefficient and a vocabulary-size-adjusted
     bound on residual variation. Naturally large token changes explained by the base projection have zero
     coefficient error in exact arithmetic.

   MAD-based checks need eight usable samples and estimate standard deviation as 1.4826 × median absolute
   deviation. Peer checks assume a representative majority. Gain and row concentration also work in
   isolation. These are indicators for review and can flag legitimate edits that depart from the projection model.

   A changed tensor of an expected kind whose `delta.csv` stats are unusable is also T3, ordered after the
   genuine outliers (reason: `stats unusable: <why>`): no row in `delta.csv` (including a tensor the 2-D float
   scanner skips, such as a 3-D MoE tensor or a quantized GGUF one: `not scanned (<why>)`), both cells blank, or an
   effective rank or top singular value that is NaN, infinite, zero or negative. Unusable stats are never read
   as consistent, and an unusable row stays out of every median as a whole. This holds whenever the scan ran, even
   if `delta.csv` came out empty; only `--inventory-only` or a stopped scan leaves T4 `(no ΔW stats)`.
4. **T4, consistent with abliteration:** every other changed tensor.

Within T3, findings with more corroborating signals rank first. Ties and other tiers use higher effective
rank first, then larger top singular value, then name (unusable stats sort as missing). With
`--inventory-only` only T1, T2 and T4 exist. `scripts/score_blind.py --ranking ranking.csv --key key.txt`
scores a ranking against an answer key of planted tensor names; it rejects a ranking that lists a tensor twice.
`--outlier-factor` must be a finite number above 0. `ranking.csv` is written only into an `--out` that is new, empty
or already marked for the same pair by `pair.txt`; llmscan never deletes or overwrites a `ranking.csv` it did not write.

## Safety

Backdoors planted for this research only ever target localhost sinks. Their weights stay local or
are published clearly labelled as research artifacts.

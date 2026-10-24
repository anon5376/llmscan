# llmscan

Check whether the abliterated or uncensored model you downloaded is only what it claims to be.
llmscan diffs the suspect's weights against the base model it says it came from, one tensor at a
time, and reports where and how they differ.

> **Status: day zero.** The loader, tensor inventory and plots work. The ΔW spectral core is being
> written, and nothing has been validated on real models yet. llmscan ranks suspicion; it never
> certifies a model clean.

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

First test: `meta-llama/Llama-3.1-8B-Instruct` against `mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated`,
then a backdoor we plant ourselves. The figure goes here once it exists.

## Use

From a clone of this repo:

```sh
pipx install .            # or: uv tool install .
llmscan --base path/to/Llama-3.1-8B-Instruct --suspect path/to/abliterated-model
```

Output goes to `out/<suspect>-vs-<base>/`:

- `inventory.csv`, `inventory.txt`: every tensor in either model, marked identical, changed,
  dtype/shape mismatch, or missing from one side;
- `delta.csv`: for each 2-D tensor, the top-k singular values of ΔW and its effective rank;
- `spectra.png`, `effective_rank.png`.

Safetensors checkpoints (single file or sharded directory) are memory-mapped and read one tensor at
a time, so model size is bounded by disk, not RAM. CPU only; the only dependencies are numpy and
matplotlib, no torch. GGUF support is next.

`scripts/fetch_hf.sh <org/name> <revision> <dir>` downloads a model's safetensors at a pinned revision.

## Safety

Backdoors planted for this research only ever target localhost sinks. Their weights stay local or
are published clearly labelled as research artifacts.

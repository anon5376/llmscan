# Do the changed tensors share a residual direction?

A model-wide alignment cutoff of 0.50 flagged all 112 changed tensors across the two honest mlabonne checkpoints. Every changed tensor in both Huihui sizes, Josiefied and Heretic scored above 0.9999.

## Method

For each changed floating-point matrix, the direction script measures ΔW = suspect − base. Energy share is the fraction of the squared Frobenius norm in the leading singular value, σ₁² / ‖ΔW‖²_F. The residual direction is the leading left singular vector for `o_proj` and `down_proj`, and the leading right singular vector for `embed_tokens`. The corresponding GGUF names are `attn_output`, `ffn_down` and `token_embd`. The script forms the smaller Gram matrix in scaled blocks and retains one vector per usable change. [Direction calculation](../src/llmscan/direction.py)

To score a target, hold it out and pool the other usable vectors of the same residual width across kinds. For each peer, take the smaller of its absolute cosine with the target and its upper median absolute cosine with the pool, counting itself once. Alignment is the largest of these peer scores. A blank CSV score carries a reason, such as an unknown residual axis or no comparable peer. [Measurement script](../scripts/measure_directions.py)

## Six honest Qwen3 pairs

Both statistics are reported as minimum / median over changed tensors of the given kind. The last two columns cover the whole model. Counts and scores come from each run's `directions.csv`, wall seconds from `time -l`. The Huihui 0.6B time includes the replacement trials. Timings and memory peaks precede the addition of the layer-pair printout.

| Checkpoint | Kind | Changed | Alignment min / median | Energy share min / median | Below 0.50 / total | Wall (s) |
|---|---|---:|---:|---:|---:|---:|
| Huihui v2 1.7B | o_proj | 28 | 0.99999704 / 0.99999783 | 0.99462800 / 0.99514871 | 0 / 56 | 58.70 |
| | down_proj | 28 | 0.99999778 / 0.99999793 | 0.99449601 / 0.99516791 | | |
| mlabonne 1.7B | o_proj | 28 | 0.02842495 / 0.03308932 | 0.99586471 / 0.99642586 | 56 / 56 | 60.06 |
| | down_proj | 28 | 0.06450626 / 0.07274808 | 0.99648119 / 0.99714058 | | |
| Josiefied v1 1.7B | o_proj | 27 | 0.99999648 / 0.99999776 | 0.99468174 / 0.99533145 | 0 / 54 | 58.28 |
| | down_proj | 27 | 0.99999771 / 0.99999793 | 0.99462399 / 0.99507397 | | |
| Heretic 0.6B | o_proj | 26 | 0.99999756 / 0.99999805 | 0.99645642 / 0.99711671 | 0 / 35 | 7.82 |
| | down_proj | 9 | 0.99995778 / 0.99999385 | 0.98245422 / 0.99310818 | | |
| Huihui v2 0.6B | o_proj | 28 | 0.99999818 / 0.99999872 | 0.99704651 / 0.99759586 | 0 / 56 | 8.71 |
| | down_proj | 28 | 0.99999841 / 0.99999888 | 0.99670941 / 0.99741194 | | |
| mlabonne 0.6B | o_proj | 28 | 0.23069091 / 0.43761239 | 0.82094046 / 0.98997363 | 56 / 56 | 12.17 |
| | down_proj | 28 | 0.23069091 / 0.43767512 | 0.82667225 / 0.98960932 | | |

On Huihui v2 1.7B, peak resident memory was **3,872,636,928 bytes** for the direction script and **5,078,925,312 bytes** for `python -m llmscan`. The full CLI took 38.99 s. Both are `time -l` readings of the two Huihui 1.7B commands under Reproduce.

## Agreement within and between layers

Each cell gives the number of comparable tensor pairs / minimum / median absolute cosine. Within a layer the comparison is `o_proj` against `down_proj`. Across layers it uses the same kind at consecutive layer numbers and requires equal residual widths. The values are the script's layer-pair lines, rounded to five decimals.

| Checkpoint | Same layer | Adjacent o_proj | Adjacent down_proj |
|---|---:|---:|---:|
| mlabonne 0.6B | 28 / 0.99963 / 0.99999 | 27 / 0.54759 / 0.83342 | 27 / 0.54827 / 0.83332 |
| Heretic 0.6B | 9 / 0.99988 / 0.99999 | 25 / 1.00000 / 1.00000 | 8 / 0.99998 / 1.00000 |
| Huihui v2 0.6B | 28 / 1.00000 / 1.00000 | 27 / 1.00000 / 1.00000 | 27 / 1.00000 / 1.00000 |
| mlabonne 1.7B | 28 / 0.00084 / 0.01472 | 27 / 0.00012 / 0.01887 | 27 / 0.53101 / 0.77367 |

The mlabonne 0.6B matrices share a direction within each layer, and that direction drifts between layers. In mlabonne 1.7B the two matrices are nearly orthogonal within a typical layer. Neighboring attention outputs also have little directional agreement, even though the down projections retain some continuity.

## Independent replacements on real Huihui v2 0.6B weights

Each trial replaces one honest edit in memory with an independent rank-one change matched to its Frobenius norm. The script adds the change to the base, rounds the weights to nearest-even BF16 and scores the resulting delta against the original peers. The norm ratio is the saved delta norm divided by the original edit norm. All six replacements ranked **1/56** by lowest alignment. Values are the script's `probe` lines for that run.

| Layer | Replaced kind | Alignment | Lowest other alignment | Energy share | Norm ratio |
|---:|---|---:|---:|---:|---:|
| 0 | down_proj | 0.04909500 | 0.99999813 | 0.99715808 | 1.00134451 |
| 0 | o_proj | 0.02529224 | 0.99999837 | 0.99728931 | 1.00125936 |
| 14 | down_proj | 0.05048508 | 0.99999818 | 0.99779820 | 1.00106886 |
| 14 | o_proj | 0.00031841 | 0.99999818 | 0.99783403 | 1.00101300 |
| 27 | down_proj | 0.01061742 | 0.99999818 | 0.99729958 | 1.00125322 |
| 27 | o_proj | 0.00818997 | 0.99999818 | 0.99739002 | 1.00123707 |

## A narrower rule

Flag a score below 0.50 only when a clear majority of **all changed tensors** have high alignment. Otherwise abstain on the model. Counting scores of at least 0.9999 as high, both Huihui sizes, Josiefied and Heretic have unanimous agreement and no flags. Both mlabonne sizes have no consensus, so the rule stays silent on them. In every Huihui replacement trial, 55 of 56 tensors still have high alignment and only the replacement is flagged.

These results support the rule for an isolated off-direction edit in a model with consensus. They leave the cutoff for a close majority untested. The rule misses a malicious edit in a model that still lacks consensus. Even with a unanimous residual direction, an attacker could preserve that direction and change only the input side of a projection. The alignment score cannot distinguish those edits, and the trials did not measure model behavior.

## Reproduce the checkpoint comparisons

Run from the repository root on macOS. Set `<models>` to the directory of local checkpoints. Each glob matches one folder named `publisher--checkpoint`. Give each comparison a new or empty output directory.

```sh
models='<models>'
reports=$(mktemp -d ./direction-results.XXXXXXXX)
export PYTHONPATH=src
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
scan() {
  local size=$1 model=$2
  shift 2
  command time -l python scripts/measure_directions.py --base "$models"/*--Qwen3-"$size" \
    --suspect "$models"/*--"$model" --out "$reports/$model" "$@"
}
scan 1.7B Huihui-Qwen3-1.7B-abliterated-v2
scan 1.7B Qwen3-1.7B-abliterated
scan 1.7B Josiefied-Qwen3-1.7B-abliterated-v1
scan 0.6B Qwen3-0.6B-heretic-abliterated-uncensored
scan 0.6B Huihui-Qwen3-0.6B-abliterated-v2 --probe-layers 0 14 27
scan 0.6B Qwen3-0.6B-abliterated
command time -l python -m llmscan --base "$models"/*--Qwen3-1.7B \
  --suspect "$models"/*--Huihui-Qwen3-1.7B-abliterated-v2 --out "$reports/baseline-1.7B"
```

## Synthetic fixture

The small BF16 fixture gives the changed projections a common residual direction and includes an unchanged mixed-dtype control. Each independent replacement ranks lowest, and the control is absent from the changed-tensor CSV. [Fixture and assertions](../tests/test_direction.py)

```sh
toy=$(mktemp -d ./direction-fixture.XXXXXXXX)
PYTHONPATH=src:tests python - "$toy" <<'PY'
import sys
from pathlib import Path
from test_direction import write_projection_pair
write_projection_pair(Path(sys.argv[1]))
PY
python scripts/measure_directions.py --base "$toy/base" --suspect "$toy/suspect" \
  --out "$toy/report" --probe-layers 0 14 27
```

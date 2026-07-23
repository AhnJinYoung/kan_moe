# Distributional MoE

This repository tests disagreement-dependent MoE aggregation at approximately
150M, 500M, and 1.5B total parameters. It compares parameter-matched dense,
vanilla sparse-MoE, and distributional sparse-MoE models against learned
output-gating and permutation-invariant residual-MLP reducers. All MoEs have
16 experts and configurable top-k routing. See
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for the mathematical design
and [`RUN_MANUAL.md`](RUN_MANUAL.md) for the exact A100/FineWeb-Edu commands.

## Tokenizer and data

The main configs use the local Llama 2 tokenizer at
`/data/umoe_mod_share/llama2_tokenizer` (32,000 tokens, EOS id 2). The default
backend reads FineWeb-Edu directly from Parquet row groups with PyArrow and
tokenizes/continuously packs text online. It neither creates a full Hugging Face
Arrow dataset nor requires a tokenized `.bin` corpus. Automatic special tokens
are disabled and one EOS is appended per document.

The final 20,000 deterministic dataset rows are excluded from training:
10,000 for validation during model selection and an untouched final 10,000 for
test after freezing the configuration. Parquet and tokenizer batches are capped
at four documents, PyArrow multiprocessing is disabled, and startup detects
cgroup CPU/memory limits and caps native thread pools to one or two threads.
The resolved resource limits, Parquet layout, tokenizer contract, and packing
policy are written to `runtime.json` and W&B.

## Installation and verification

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
python3 -m pip install -e '.[data,eval,logging,dev]'
python3 -m unittest discover -s tests -v
python3 scripts/count_parameters.py \
  configs/dense_150m.yaml \
  configs/vanilla_moe_150m.yaml \
  configs/distributional_moe_150m.yaml \
  configs/dense_500m.yaml \
  configs/vanilla_moe_500m.yaml \
  configs/distributional_moe_500m.yaml \
  configs/dense_1_5b.yaml \
  configs/vanilla_moe_1_5b.yaml \
  configs/distributional_moe_1_5b.yaml
```

Expected totals are 504,122,112 parameters for dense and 504,195,840 for each
MoE model, a difference of about 0.015%.

The explicit cu128 wheel is required on the target server whose NVIDIA driver
supports CUDA 12.8. Installing an unqualified latest PyTorch wheel can select a
newer CUDA build that this driver cannot initialize.

## Training

If `CUDA_VISIBLE_DEVICES` is unset, `train.py` queries `nvidia-smi` before
importing PyTorch, excludes GPUs with compute processes or more than 1 GiB of
allocated memory, and reserves the best idle GPU(s) with a local lock. An
explicit `CUDA_VISIBLE_DEVICES` value always takes precedence.

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --config configs/distributional_moe_500m.yaml
```

For one A100 using the batch settings already stored in the YAML:

```bash
unset CUDA_VISIBLE_DEVICES
torchrun --standalone --nproc_per_node=1 train.py \
  --config configs/distributional_moe_500m.yaml \
  --override train.wandb_project=kan-moe \
  --override train.wandb_run_name=dmoe-hellinger-k2-50k-1gpu
```

Change top-k without changing the parameter count:

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --config configs/distributional_moe_500m.yaml \
  --override model.top_k=4 \
  --override train.output_dir=/data/umoe_mod_share/kan_moe/outputs/distributional_moe_500m_k4
```

Available distributional aggregators are `geometric` (exact vanilla-equivalent
control), `hellinger` (primary), `arithmetic`, general `power`, and optional
`wasserstein`. Set them with `--override model.aggregation=...`; general power
pooling also accepts `--override model.power_rho=0.75`. Layerwise learned rho
uses `model.aggregation=power` and `model.learnable_rho=true`.

The staged protocol first profiles and screens at 500M, then confirms a frozen
winner from scratch. Learned controls, the 150M/1.5B scale check, and extra
seeds are deferred until the 500M result passes its gate. Commands can be
inspected without launching jobs:

```bash
python3 scripts/experiment_matrix.py --stage profiling --nproc-per-node 4
python3 scripts/experiment_matrix.py --stage pilot --nproc-per-node 4
python3 scripts/experiment_matrix.py --stage screening --nproc-per-node 4
python3 scripts/experiment_matrix.py \
  --stage confirmation --confirmation-tokens 1000000000 \
  --winner-distribution-k 9 --winner-rho 0.5 --winner-top-k 2 \
  --nproc-per-node 4
python3 scripts/experiment_matrix.py --stage controls --nproc-per-node 4
python3 scripts/experiment_matrix.py --stage scaling --nproc-per-node 4
python3 scripts/experiment_matrix.py --stage seeds --nproc-per-node 4
```

Add `--execute` only after reviewing the emitted commands.

## Evaluation

Perplexity is computed over disjoint held-out token windows:

```bash
torchrun --standalone --nproc_per_node=4 evaluate_ppl.py \
  --checkpoint /path/to/step_00009537.pt \
  --split test \
  --max-tokens 10000000 \
  --output /path/to/ppl.json
```

Standard benchmarks use the pinned `lm-evaluation-harness` adapter:

```bash
python3 evaluate_harness.py \
  --checkpoint /path/to/step_00009537.pt \
  --tokenizer /data/umoe_mod_share/llama2_tokenizer \
  --suite primary \
  --batch-size 8 \
  --output /path/to/benchmarks.json
```

The primary suite is LAMBADA, PIQA, and HellaSwag; near-chance tasks at small
scale are isolated under `--suite secondary`. Paired mechanism analysis uses:

```bash
python3 analyze_mechanism.py \
  --checkpoint /path/to/distributional.pt \
  --baseline-checkpoint /path/to/same_seed_vanilla.pt \
  --split test \
  --max-tokens 1000000 \
  --output /path/to/mechanism.json
```

# Distributional MoE

This repository compares parameter-matched ~505M dense, vanilla sparse-MoE,
and distribution-valued sparse-MoE decoder language models. Both MoE variants
have 16 experts and configurable top-k routing. See
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for the mathematical design
and [`RUN_MANUAL.md`](RUN_MANUAL.md) for the exact A100/FineWeb-Edu commands.

## Tokenizer and data

The main configs use the local Llama 2 tokenizer at
`/data/umoe_mod_share/kan_moe/llama2_tokenizer` (32,000 tokens, EOS id 2).
`train.py` loads the local FineWeb-Edu Parquet files through Hugging Face
Datasets, reuses the raw-text Arrow files under `HF_DATASETS_CACHE`, and
tokenizes/continuously packs text online. It does not require a separate
tokenized `.bin` corpus. Automatic special tokens are disabled and one EOS is
appended per document.

The final 10,000 deterministic dataset rows are excluded from training and used
for validation. The resolved Parquet list, Dataset fingerprint, actual Arrow
cache filenames, tokenizer contract, and packing policy are written to
`runtime.json` and W&B.

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
  configs/dense_500m.yaml \
  configs/vanilla_moe_500m.yaml \
  configs/distributional_moe_500m.yaml
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
  --override train.wandb_run_name=dmoe-hellinger-k2-5b-1gpu
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
pooling also accepts `--override model.power_rho=0.75`.

## Evaluation

Perplexity is computed over disjoint held-out token windows:

```bash
torchrun --standalone --nproc_per_node=4 evaluate_ppl.py \
  --checkpoint /path/to/step_00009537.pt \
  --max-tokens 10000000 \
  --output /path/to/ppl.json
```

Standard benchmarks use the pinned `lm-evaluation-harness` adapter:

```bash
python3 evaluate_harness.py \
  --checkpoint /path/to/step_00009537.pt \
  --tokenizer /data/umoe_mod_share/kan_moe/llama2_tokenizer \
  --tasks mmlu,arc_easy,arc_challenge,hellaswag,piqa,winogrande,openbookqa,boolq,lambada_openai \
  --batch-size 8 \
  --output /path/to/benchmarks.json
```

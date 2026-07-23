# Distributional MoE

This repository compares parameter-matched ~505M dense, vanilla sparse-MoE,
and distribution-valued sparse-MoE decoder language models. Both MoE variants
have 16 experts and configurable top-k routing. See
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for the mathematical design
and [`RUN_MANUAL.md`](RUN_MANUAL.md) for the exact A100/FineWeb-Edu commands.

## Tokenizer and data

The experiment uses the base `mistralai/Mistral-7B-v0.3` tokenizer at pinned
revision `caa1feb0e54d415e2df31207e5f4e273e33509b1` (32,768 tokens). It does not
use the GPT-2 tokenizer. Every raw document is encoded without automatically
added special tokens and followed by EOS id 2.

Prepare the local parquet corpus once:

```bash
python3 prepare_fineweb.py \
  --input-dir /data/umoe_mod_share/fineweb_edu_100bt/sample/100BT \
  --output-dir /data/umoe_mod_share/fineweb_edu_100bt/tokenized_mistral_v3 \
  --workers 16
```

The process is streaming, restartable at shard granularity, and writes a
manifest checked by `train.py`. The token files are flat `uint16` arrays and
can be memory-mapped without loading the corpus into RAM.

## Installation and verification

```bash
python3 -m pip install -e '.[data,eval,logging,dev]'
python3 -m unittest discover -s tests -v
python3 scripts/count_parameters.py \
  configs/dense_500m.yaml \
  configs/vanilla_moe_500m.yaml \
  configs/distributional_moe_500m.yaml
```

Expected totals are 504,711,936 parameters for dense and 504,785,664 for each
MoE model, a difference of about 0.015%.

## Training

If `CUDA_VISIBLE_DEVICES` is unset, `train.py` queries `nvidia-smi` before
importing PyTorch, excludes GPUs with compute processes or more than 1 GiB of
allocated memory, and reserves the best idle GPU(s) with a local lock. An
explicit `CUDA_VISIBLE_DEVICES` value always takes precedence.

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --config configs/distributional_moe_500m.yaml
```

For one A100 while preserving the original effective global batch:

```bash
unset CUDA_VISIBLE_DEVICES
torchrun --standalone --nproc_per_node=1 train.py \
  --config configs/distributional_moe_500m.yaml \
  --override train.micro_batch_size=4 \
  --override train.gradient_accumulation_steps=64
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
  --tokenizer /data/umoe_mod_share/fineweb_edu_100bt/tokenized_mistral_v3/tokenizer \
  --tasks mmlu,arc_easy,arc_challenge,hellaswag,piqa,winogrande,openbookqa,boolq,lambada_openai \
  --batch-size 8 \
  --output /path/to/benchmarks.json
```

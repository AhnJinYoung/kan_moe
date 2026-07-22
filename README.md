# Distributional MoE

This repository compares parameter-matched 504M dense, vanilla sparse-MoE,
and distribution-valued sparse-MoE decoder language models. Read
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for the mathematical design,
equivalence controls, and experiment contract.

## Installation

```bash
python -m pip install -e '.[eval,logging,dev]'
```

Training only requires PyTorch, NumPy, and PyYAML. Standard benchmark evaluation
also requires `transformers`, `datasets`, and `lm-evaluation-harness`.

## Prepared FineWeb-Edu format

The loader consumes tokenized one-dimensional `.bin` or `.npy` shards. For a
`.bin` shard, set `data.binary_dtype` to its actual dtype (`uint16` is appropriate
only when every token id is below 65,536). Example layout:

```text
/data/fineweb-edu-sample-100BT/
  train/
    train_00000.bin
    train_00001.bin
  validation/
    validation_00000.bin
  tokenizer/
    tokenizer.json
    tokenizer_config.json
```

Edit the paths, tokenizer vocabulary size, and EOS id in each YAML before
training. Online tokenization of 100BT raw text is intentionally not performed.

## Verify parameter matching

```bash
python scripts/count_parameters.py \
  configs/dense_500m.yaml \
  configs/vanilla_moe_500m.yaml \
  configs/distributional_moe_500m.yaml
```

Expected totals are 504,122,112 parameters for dense and 504,195,840 for each
MoE model.

## Training

Run on four local GPUs:

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --config configs/distributional_moe_500m.yaml
```

Change top-k without changing parameters:

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --config configs/distributional_moe_500m.yaml \
  --override model.top_k=4 \
  --override train.output_dir=outputs/distributional_moe_500m_k4
```

Aggregation controls use the same model:

```bash
# Exact vanilla-equivalent negative control in ILR coordinates
--override model.aggregation=geometric

# Hellinger proposal
--override model.aggregation=hellinger

# Arithmetic distribution pool
--override model.aggregation=arithmetic

# General power pool
--override model.aggregation=power --override model.power_rho=0.75
```

Resume the newest retained checkpoint by setting `train.resume=latest`. Use
`train.max_tokens` for a token budget; when it is nonzero, training stops at the
smaller of the token-derived step count and `train.max_steps`.

## Perplexity

```bash
torchrun --standalone --nproc_per_node=4 evaluate_ppl.py \
  --checkpoint outputs/distributional_moe_500m_k2/step_00010000.pt \
  --max-tokens 10000000 \
  --output outputs/distributional_moe_500m_k2/ppl.json
```

Use `--top-k` to evaluate the same checkpoint with another number of active
experts. This is useful diagnostically, but the primary comparison should use
the top-k on which the model was trained.

## Standard benchmarks

```bash
python evaluate_harness.py \
  --checkpoint outputs/distributional_moe_500m_k2/step_00010000.pt \
  --tokenizer /data/fineweb-edu-sample-100BT/tokenizer \
  --tasks mmlu,arc_easy,arc_challenge,hellaswag,piqa,winogrande,openbookqa,boolq,lambada_openai \
  --batch-size 8 \
  --output outputs/distributional_moe_500m_k2/benchmarks.json
```

The adapter implements continuation log-likelihood, rolling log-likelihood,
and greedy generation. Benchmark task data and prompts are managed by
`lm-evaluation-harness`; keep its version fixed across model comparisons.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests include the exact geometric/vanilla output and gradient equivalence,
top-1 identity, Hellinger nonlinearity, data shard loading, forward/backward,
and the 500M parameter-count contract.


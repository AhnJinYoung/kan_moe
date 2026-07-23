# Distributional MoE: implementation and experiment plan

## 1. Goal and comparison contract

We will implement three decoder-only language models around 500M **total**
parameters:

1. `dense`: every transformer block contains a dense SwiGLU FFN.
2. `vanilla_moe`: six of twelve FFNs are replaced by a 16-expert, dropless,
   token-routed MoE and selected expert vectors are combined linearly.
3. `distributional_moe`: it has exactly the same expert and router parameters as
   `vanilla_moe`, but interprets each expert vector as coordinates of a product
   distribution and combines those distributions before mapping back to the
   transformer residual stream.

The three models use the same tokenizer, attention design, embedding design,
normalization, micro/global batch, training data order, optimizer, and token budget. Their total parameter counts
must differ by less than 5%; the selected dimensions below make the difference
less than 0.02%. `top_k` is a runtime/configuration choice in `[1, 16]` and does
not change the number of model parameters.

“Distribution” in this project means a structured latent representation. It is
not calibrated epistemic uncertainty unless a future probabilistic training
objective supplies that semantics.

## 2. Common architecture and parameter count

All variants use a pre-norm decoder-only transformer with RMSNorm, rotary
position embeddings, causal scaled-dot-product attention, tied token/output
embeddings, and SwiGLU FFNs.

| field | value |
|---|---:|
| tokenizer | local Llama 2 tokenizer snapshot |
| vocabulary | 32,000 |
| layers | 12 |
| model width | 768 |
| attention heads | 12 |
| maximum sequence length | 2,048 |
| MoE layers | 6, alternating with dense layers |
| experts per MoE layer | 16 |
| routed expert width | 1,920 |
| router top-k | configurable, 1–16; default 2 |
| dense baseline FFN width | 16,320 |

Ignoring small norm/router terms, a SwiGLU FFN has

\[
N_{\mathrm{ffn}}=3d_{\mathrm{model}}d_{\mathrm{ff}}.
\]

For each MoE model, the total number of width-1,920 FFN equivalents is

\[
6\text{ dense FFNs}+6\times16\text{ experts}=102.
\]

For the dense model, choosing width 16,320 gives

\[
12\times 3\times768\times16{,}320
=102\times3\times768\times1{,}920
=451{,}215{,}360
\]

FFN parameters. Embedding and attention add about 53.5M parameters, producing
504,122,112 dense parameters and 504,195,840 parameters for either MoE.
MoE routers add only 73,728
parameters, a difference far below 0.02%.

Total parameters are matched, but active compute deliberately differs. An MoE
model activates approximately

\[
52.9\mathrm{M}+6(4.424\mathrm{M})+6k(4.424\mathrm{M})
\]

parameters per token: about 107M, 133M, 186M, 292M, and 505M for top-k 1, 2,
4, 8, and 16 respectively. The dense baseline activates all 505M. We will report
both quality versus tokens and quality versus wall-clock/FLOPs so total-parameter
matching is not confused with compute matching.

## 3. Distribution space

### 3.1 Ordered basis distributions

For one expert and one group, let

\[
p_{e,g}(u\mid x)=\sum_{j=1}^{K}\pi_{e,g,j}(x)B_j(u),
\qquad \pi_{e,g}\in\Delta^{K-1},
\]

where the non-negative normalized coefficients `pi` form a categorical
distribution over an ordered dictionary of K localized basis functions. The
basis may be understood as fixed B-spline atoms. The initial Hellinger
aggregator acts on the coefficient distribution; an optional Wasserstein
aggregator additionally uses the ordered atom centers as its ground metric.

A full joint distribution is intractable, so the expert distribution is the
product of G factors. With K=9 and G=96,

\[
G(K-1)=96\times8=768=d_{\mathrm{model}}.
\]

Thus the product-simplex distribution has exactly the same intrinsic dimension
as the original expert vector.

### 3.2 Lossless simplex codec

Let `H` be a `(K-1) x K` Helmert contrast matrix satisfying

\[
HH^T=I,\qquad H\mathbf{1}=0.
\]

The ordinary expert emits `z_e` with shape `[G, K-1]`. It is mapped to the
interior of the simplex by

\[
\log\pi_e=\log\operatorname{softmax}(z_eH).
\]

The isometric log-ratio readout is

\[
\operatorname{ilr}(\pi_e)=\log\pi_eH^T=z_e.
\]

The equality follows because the softmax log-normalizer is proportional to the
all-ones vector and is annihilated by H. Consequently:

- one expert loses no information or intrinsic dimension;
- the distributional expert head has exactly the same parameters as vanilla;
- with top-k=1 every aggregation method reduces to the vanilla expert output;
- any improvement at top-k>1 comes from cross-expert aggregation, not a larger
  expert.

## 4. Aggregation and mathematical controls

Let selected router probabilities be `r_e`, renormalized to sum to one.

### 4.1 Power-mean family

The primary implementation uses

\[
q_{\rho,j}\propto
\left(\sum_e r_e\pi_{e,j}^{\rho}\right)^{1/\rho}.
\]

It is evaluated in log-space. Important cases are:

- `rho -> 0`: geometric/logarithmic opinion pool;
- `rho = 0.5`: weighted Hellinger barycenter;
- `rho = 1`: arithmetic pool, also the forward-KL barycenter.

The layer output is `ilr(q_rho)`.

### 4.2 Exact vanilla equivalence at rho=0

For geometric pooling,

\[
q_0\propto\prod_e\pi_e^{r_e},
\]

and therefore

\[
\operatorname{ilr}(q_0)
=\sum_e r_e\operatorname{ilr}(\pi_e)
=\sum_e r_e z_e.
\]

This is exactly vanilla MoE aggregation. It is a mandatory numerical and
gradient-level negative control, not a proposed improvement.

### 4.3 Primary proposal: Hellinger barycenter

For `rho=0.5`,

\[
q_j=
\frac{(\sum_e r_e\sqrt{\pi_{e,j}})^2}
{\sum_l(\sum_e r_e\sqrt{\pi_{e,l}})^2}.
\]

This is the minimizer of the weighted sum of squared Hellinger distances. Its
ILR readout contains a log-sum-exp over multiple expert outputs, so it cannot be
distributed into a linear sum of independently computed expert vectors. It
therefore introduces an explicit disagreement-dependent cross-expert
interaction with O(top-k * G * K) work and no iterative solver.

### 4.4 Numeric example

For a concrete token, suppose its current residual vector is `h_t` with shape
`[768]`. The router scores all 16 experts, top-k=2 selects experts 3 and 11, and
renormalizes their scores to `(0.6, 0.4)`. Each selected expert applies its
SwiGLU and emits a `[768]` vector. The codec reshapes each vector into 96 groups
of 8 ILR coordinates and converts every group into 9 basis probabilities. The
aggregator combines matching groups from experts 3 and 11, maps the resulting
96 distributions back to 96 x 8 coordinates, concatenates them to `[768]`, and
adds that vector to `h_t` through the transformer residual connection.

For a small numeric view of one such group, reduce it to three atoms and use
router weights `(0.6, 0.4)`:

```
expert 1: (0.70, 0.20, 0.10)
expert 2: (0.10, 0.30, 0.60)
```

The pools are approximately:

```
geometric / vanilla in ILR: (0.422, 0.309, 0.269)
arithmetic:                  (0.460, 0.240, 0.300)
Hellinger:                   (0.448, 0.269, 0.283)
```

Geometric pooling is merely linear interpolation in ILR coordinates.
Hellinger pooling instead interpolates square-root densities and then returns
to the simplex, adding a correction that depends jointly on how the experts
disagree. If only expert 1 is selected, all three methods return expert 1
exactly.

### 4.5 Optional Wasserstein experiment

For fixed ordered centers `c_j`, define

\[
q^*=\arg\min_q\sum_e r_e W_2^2(q,\pi_e).
\]

Unlike Hellinger pooling, this can transport two modes toward an intermediate
basis location. We will implement an entropically regularized Sinkhorn
barycenter behind a configuration flag, in FP32 with a small fixed iteration
count. It is secondary because it assumes the learned groups use the imposed
basis order meaningfully and has greater kernel/memory overhead.

## 5. Routing and auxiliary objectives

- The router computes a softmax over 16 experts, selects configurable top-k,
  gathers those probabilities, and renormalizes them.
- Routing is dropless: every selected assignment is evaluated, avoiding a
  capacity/drop confound between aggregators.
- The same Switch-style load-balancing loss and router z-loss are used for both
  MoE variants.
- The language-model loss is the only task loss. No coefficient entropy penalty
  is enabled initially because the codec is lossless and such a penalty would
  make the comparison asymmetric.
- We log expert load, router entropy, maximum load fraction, distribution
  entropy, and the norm of the nonlinear correction.

## 6. Tokenizer and data contract

All three variants use the same local Llama 2 tokenizer snapshot with a
32,000-token vocabulary and EOS id 2. Raw pretraining text is encoded without a
chat template, BOS, padding, or tokenizer-added special tokens. One EOS token is
added after every document.

The primary input path is the local Parquet corpus at
`/data/umoe_mod_share/fineweb_edu_100bt/sample/100BT`. The primary loader reads
Parquet row groups directly with PyArrow. It does not call
`load_dataset("parquet")`, materialize the 100BT corpus as Arrow, or spawn data
worker processes. A fast tokenizer encodes documents online in bounded batches
and continuously packs the resulting stream into 2,048-token examples.

Rows are consumed in deterministic dataset order. Distributed ranks take
disjoint contiguous row ranges, and all three model variants receive the same
rank-local stream. The final 10,000 rows are excluded from training and reserved
for validation. Checkpoints store the next global row, epoch, and unconsumed
packed-token buffer, so resume reproduces the exact subsequent token batch.

Startup validates tokenizer vocabulary/EOS compatibility and every produced
token id. It also detects CPU affinity, cgroup CPU quota, and cgroup memory
limits, caps native/PyTorch thread pools to one or two threads, and limits
Parquet/tokenizer batches to at most four documents. `runtime.json` and W&B
record these resolved resource limits, sorted Parquet layout, tokenizer
contract, split boundary, and packing policy. The older `.bin`/`.npy` loader
remains available as the `binary` input format; the previous HF Arrow-cache
backend is opt-in only.

## 7. Training implementation

The training entry point will support:

- `torchrun` DDP on 1–4 GPUs;
- BF16 autocast, TF32, fused AdamW when available;
- gradient accumulation and DDP `no_sync`;
- activation checkpointing;
- cosine learning-rate schedule with warmup;
- gradient clipping;
- deterministic token/step accounting;
- periodic validation PPL;
- atomic checkpoints containing model, optimizer, scheduler position,
  configuration, RNG state, and data sampler state;
- JSONL metrics and optional Weights & Biases logging;
- resume and model-only initialization.

At this scale all experts will initially be replicated on each GPU. This avoids
all-to-all communication and makes non-linear selected-output aggregation
simple. If profiling shows dispatch is the bottleneck, grouped-GEMM or a
block-sparse backend can replace the expert loop without changing model math.

## 8. Evaluation

### 8.1 Perplexity

A distributed PPL script will evaluate non-overlapping held-out token windows
and all-reduce total negative log-likelihood and token count:

\[
\mathrm{PPL}=\exp\left(
\frac{\sum_t-\log p(x_t\mid x_{<t})}{N_{\mathrm{tokens}}}
\right).
\]

It will report PPL, mean NLL, evaluated tokens, and elapsed time.

### 8.2 Standard benchmarks

An EleutherAI `lm-evaluation-harness` adapter will expose log-likelihood,
rolling log-likelihood, and greedy generation for the custom checkpoint. The
default suite will include:

- MMLU;
- ARC-Easy and ARC-Challenge;
- HellaSwag;
- PIQA;
- WinoGrande;
- OpenBookQA;
- BoolQ;
- LAMBADA OpenAI.

Task prompts, few-shot defaults, and metrics remain owned by the harness. The
exact harness version and task names will be recorded with every result.

## 9. Verification sequence

1. Unit-test Helmert orthogonality and ILR round trips.
2. Verify geometric distributional output and gradients match vanilla MoE.
3. Verify top-k=1 equivalence for every power aggregator.
4. Verify Hellinger mass, symmetry, finiteness, and gradients.
5. Verify each 500M config's actual parameter count and the <5% contract.
6. Run forward/backward tests with tiny CPU configurations.
7. Run a short synthetic token training job and resume it.
8. Run local PPL on a tiny held-out shard.
9. Import-check the lm-eval adapter when optional dependencies are available.
10. On the GPU system, profile 50 steps for top-k 1, 2, 4 before committing to
    the full pretraining runs.

## 10. Initial experiment matrix

Short screening uses top-k `{1, 2, 4}` and the following aggregation settings:

- vanilla linear;
- distributional geometric (`rho=0`, exact control);
- distributional Hellinger (`rho=0.5`, primary);
- distributional arithmetic (`rho=1`).

Top-k=1 tests representation equivalence but cannot test aggregation. Quality
will be reported against tokens, active-parameter estimates, measured
tokens/second, and wall-clock time. Wasserstein and learned `rho` are enabled
only after the fixed, closed-form comparisons are stable.

All primary runs use the same 2B-token budget, 50,000-step safety cap, data
split, seed, and schedule. The token limit is reached after 15,259 optimizer
steps on one GPU (131,072 tokens/step) or 3,815 steps on four GPUs (524,288
tokens/step). Before those runs, a 100M-token pipeline pilot and 300M-token
top-k screen are used to eliminate broken or numerically unstable
configurations.

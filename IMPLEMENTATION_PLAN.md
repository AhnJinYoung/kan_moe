# Distributional MoE: implementation and experiment plan

## 1. Goal, falsifiable claim, and comparison contract

The primary claim is deliberately narrower than “expert outputs are calibrated
probability distributions”:

> A fixed product-simplex geometry supplies a useful inductive bias for
> disagreement-dependent aggregation of selected MoE experts, beyond what is
> explained by parameter count, an arbitrary nonlinearity, or an extra learned
> gate.

This claim is falsified if the gain is matched by a small generic learned
reducer, does not concentrate on high-disagreement tokens, disappears as scale
increases, or vanishes across seeds.

We compare the following decoder-only models at approximately 150M, 500M, and
1.5B **total** parameters:

1. `dense`: dense SwiGLU FFNs in every block.
2. `vanilla_moe`: 16-expert dropless MoE with a linear router-weighted sum.
3. `distributional_moe`: the same experts/router with product-simplex pooling.
4. `output_gated_moe`: a non-distributional control that learns a second
   content-dependent scalar gate over selected expert outputs.
5. `residual_mlp_moe`: a non-distributional, permutation-invariant low-rank MLP
   over the selected-output mean and variance.

All variants use the same tokenizer, attention, normalization, routed expert
dimensions, data order, optimizer, global batch, and token budget within each
comparison. Vanilla and fixed distributional MoE have exactly the same
trainable parameter count. Learned controls add less than 1%, and all reported
comparisons include exact total and active parameter counts. `top_k` remains a
configuration choice in `[1, 16]`. Parameter initialization uses a stable hash
of `(training seed, module name, initialization role)`, so every shared
parameter starts bitwise-identically across reducer variants even when a
control adds extra modules.

“Distribution” here means a structured latent representation, not calibrated
epistemic uncertainty. Simplex terminology earns explanatory force only if the
pre-registered geometry and disagreement predictions below survive the generic
reducer controls.

The novelty claim does not treat Aitchison geometry, ILR, opinion pools,
products of experts, power means, or Hellinger barycenters as new. It is the
combination of (i) identifying vanilla MoE as the exact `rho=0` Aitchison
barycenter, (ii) exposing a continuous controlled departure from that identity,
and (iii) testing whether the departure helps specifically when routed experts
disagree. Without (iii), the contribution is only an implementation
recombination and will be described as such.

## 2. Common architecture, scales, and parameter count

All variants use a pre-norm decoder-only transformer with RMSNorm, rotary
position embeddings, causal scaled-dot-product attention, tied token/output
embeddings, and SwiGLU FFNs.

| scale | layers | width | heads | MoE layers | expert FFN | dense FFN | total parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| 150M | 12 | 512 | 8 | 6 | 768 | 6,528 | 149.30–149.35M |
| 500M | 12 | 768 | 12 | 6 | 1,920 | 16,320 | 504.12–504.20M |
| 1.5B | 16 | 1,024 | 16 | 8 | 3,328 | 28,288 | 1,490.32–1,490.45M |

All use the local 32,000-token Llama 2 tokenizer, sequence length 2,048,
16 experts per MoE layer, alternating MoE/dense layers, and default top-k=2.
The dense/MoE total-parameter spread is below 0.04% at every scale.

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
MoE routers add only 73,728 parameters at 500M, a difference far below 0.02%.

Total parameters are matched, but active compute deliberately differs. An MoE
model activates approximately

\[
52.9\mathrm{M}+6(4.424\mathrm{M})+6k(4.424\mathrm{M})
\]

parameters per token: about 107M, 133M, 186M, 292M, and 505M for top-k 1, 2,
4, 8, and 16 respectively. The dense baseline activates all 505M. We will report
both quality versus tokens and quality versus wall-clock/FLOPs so total-parameter
matching is not confused with compute matching. The learned reducers are
reported separately with their small parameter overhead rather than described
as exactly matched.

## 3. Distribution space

### 3.1 Product-simplex factors

For one expert and one group, let

\[
p_{e,g}(u\mid x)=\sum_{j=1}^{K}\pi_{e,g,j}(x)B_j(u),
\qquad \pi_{e,g}\in\Delta^{K-1},
\]

where the non-negative normalized coefficients `pi` form a categorical
distribution over K coordinates. Hellinger and power pooling require no claim
that these coordinates have semantic or spatial order. Only the optional
Wasserstein experiment imposes ordered atom centers and must therefore justify
that additional assumption.

A full joint distribution is intractable, so the expert distribution is the
product of G factors. With K=9 and G=96,

\[
G(K-1)=96\times8=768=d_{\mathrm{model}}.
\]

Thus the product-simplex distribution has exactly the same intrinsic dimension
as the original expert vector. The choice is not treated as canonical:
`K in {5, 9, 17}` gives `G in {192, 96, 48}` at width 768 and is a mandatory
ablation. All choices preserve `G(K-1)=d_model` and parameter count.
Sensitivity with a reproducible optimum supports a real geometric design
choice; flat performance across K weakens the atom interpretation and requires
reframing the method as a generic grouped nonlinearity. Either outcome is
reported.

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

### 4.5 Generic non-distributional controls

The distributional interpretation is not accepted merely because one value of
`rho` beats another. Two learned reducers directly test the “arbitrary
nonlinearity” alternative.

For the output-gated control,

\[
\tilde r_e=\operatorname{softmax}_e(\log r_e+a^Tz_e),\qquad
y=\sum_e\tilde r_ez_e.
\]

It asks whether another content-dependent gate is sufficient. For the residual
MLP control, define weighted mean and elementwise variance

\[
\mu=\sum_er_ez_e,\qquad
v=\sum_er_e(z_e-\mu)^2,
\]

and use

\[
y=\mu+W_2\operatorname{SiLU}(W_1[\mu;v]).
\]

The output scorer and residual projection are zero-initialized, so both learned
controls begin at vanilla aggregation up to floating-point roundoff. The
bottleneck is small. These controls are
permutation-invariant in selected-expert order and add less than 1% parameters.
If either matches Hellinger within the pre-registered uncertainty interval, the
paper must describe the result as evidence for nonlinear expert interaction,
not specifically for distribution geometry.

### 4.6 Learnable-rho diagnostic

A layerwise scalar `rho` can be learned from an initial value of 0.5. The
implementation uses the analytic limit around zero,

\[
\log q_\rho =
\mathbb E_r[\log\pi]+\frac{\rho}{2}
\operatorname{Var}_r[\log\pi]+O(\rho^2),
\]

so optimization can cross `rho=0` without a numerical singularity. We log rho
per layer and its mean. Convergence toward zero, together with a collapsing
correction norm, is evidence that training rejects the proposed nonlinearity;
a stable nonzero rho is diagnostic evidence, not by itself a quality result.

### 4.7 Optional Wasserstein experiment

For fixed ordered centers `c_j`, define

\[
q^*=\arg\min_q\sum_e r_e W_2^2(q,\pi_e).
\]

Unlike Hellinger pooling, this can transport two modes toward an intermediate
basis location. We will implement an entropically regularized Sinkhorn
barycenter behind a configuration flag, in FP32 with a small fixed iteration
count. It is secondary because it assumes the learned groups use the imposed
basis order meaningfully and has greater kernel/memory overhead.

## 5. Routing, auxiliary objectives, and mechanism predictions

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
  entropy, layerwise/mean rho, and the norm of the nonlinear correction.

The mechanism claim is evaluated on held-out tokens, not inferred only from
final PPL. For selected coefficient distributions define weighted
Jensen–Shannon disagreement

\[
D_t=\sum_er_e\operatorname{KL}\left(
\pi_{e,t}\middle\|\sum_jr_j\pi_{j,t}\right),
\]

averaged across product factors and MoE layers. For checkpoints trained with
the same seed and token order, define per-token gain

\[
\Delta_t=\operatorname{NLL}_{\mathrm{vanilla},t}
          -\operatorname{NLL}_{\mathrm{distributional},t}.
\]

The pre-registered predictions are:

1. Spearman correlation between `D_t` and `Delta_t` is positive.
2. Mean gain in the highest-disagreement decile exceeds the lowest decile.
3. The Hellinger–vanilla gap grows from top-k 1 to 2 to 4; top-k=1 is exactly
   zero before independent training noise.
4. Counterfactually switching a trained distributional checkpoint to geometric
   pooling degrades high-disagreement tokens most.
5. Router-gradient norms/directions under Hellinger differ from the geometric
   counterfactual, and the change covaries with disagreement.

A dedicated paired analysis uses identical held-out sequences, reports
bootstrap confidence intervals and disagreement deciles, and compares router
gradients with the auxiliary router losses excluded. These measurements decide
whether the interaction mechanism is supported.

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
disjoint contiguous row ranges, and all compared model variants receive the
same rank-local stream. The final 20,000 rows are excluded from training: the
first 10,000 are the validation split used for screening and the final 10,000
are an untouched test split opened only after `K`, rho, and top-k are frozen.
Checkpoints store the next global row, epoch, and unconsumed packed-token
buffer, so resume reproduces the exact subsequent token batch.

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

At the current scales all experts are replicated on each GPU. This avoids
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

It reports PPL, mean NLL, evaluated tokens, elapsed time, and sequence-block
bootstrap intervals. Mean NLL is the primary endpoint because differences are
additive; PPL is reported for readability. Model pairs are evaluated on exactly
the same held-out token windows. Validation selects candidates; the frozen
5B confirmation and final seed statistics are reported on the untouched test
split.

### 8.2 Standard benchmarks

An EleutherAI `lm-evaluation-harness` adapter will expose log-likelihood,
rolling log-likelihood, and greedy generation for the custom checkpoint. The
pre-registered primary suite is LAMBADA OpenAI, PIQA, and HellaSwag. These are
the most likely to provide signal at the smaller scales. MMLU, ARC-Easy,
ARC-Challenge, WinoGrande, OpenBookQA, and BoolQ are secondary/exploratory and
cannot rescue a failed primary result. This avoids presenting a large set of
near-chance comparisons as independent evidence.

Task prompts, few-shot defaults, and metrics remain owned by the harness. The
exact harness version and task names will be recorded with every result.

### 8.3 Pre-registered success criteria

The primary comparison is the frozen 500M distributional configuration versus
the top-k-matched vanilla MoE, both trained from scratch for 5B tokens. The
500M screen may select `K`, rho, and top-k, but the choice is frozen before this
confirmation run. The method is considered supported only if all of the
following hold:

1. Across seeds `{1337, 2027, 4099}`, mean NLL improves by at least 0.005
   nats/token (about 0.5% PPL) and the 95% confidence interval of the paired
   seed difference excludes zero.
2. The selected distributional reducer beats both learned reducer controls by
   at least 0.002 nats/token, or the distribution-specific claim is rejected in
   favor of a generic nonlinear-reducer claim.
3. Disagreement/loss-gain Spearman correlation is positive with a bootstrap
   95% interval excluding zero, and the highest disagreement decile improves
   more than the lowest.
4. The gain does not reverse at 1.5B and retains at least half of its 500M
   relative-NLL improvement.
5. Peak memory, tokens/second, and wall-clock are reported. A result with more
   than 25% throughput loss is not described as an unconditional efficiency
   improvement even if quality improves.

The numerical thresholds are fixed before full runs. If pilot variance implies
that three seeds cannot resolve 0.005 nats/token, more seeds are required or
the result is declared underpowered; thresholds are not moved after seeing the
outcome.

## 9. Verification sequence

1. Unit-test Helmert orthogonality and ILR round trips.
2. Verify geometric distributional output and gradients match vanilla MoE.
3. Verify top-k=1 equivalence for every power aggregator.
4. Verify Hellinger mass, symmetry, finiteness, and gradients.
5. Verify `K in {5, 9, 17}` round trips and preserves dimension/parameters.
6. Verify learned rho is finite and differentiable through zero.
7. Verify learned reducer permutation invariance, zero-init identity, and
   parameter overhead below 1%.
8. Verify 150M/500M/1.5B dense/MoE parameter contracts.
9. Run forward/backward and BF16 tests for every reducer.
10. Verify mechanism collection, paired token alignment, JS bounds, loss
    deltas, and router-gradient comparison on tiny CPU models.
11. Run a short synthetic token training job and exact resume.
12. Run local PPL and paired analysis on a tiny held-out shard.
13. Import-check the lm-eval adapter when optional dependencies are available.
14. On the GPU system, profile 50 steps for every surviving reducer/top-k before
    committing to full pretraining.

## 10. Staged experiment matrix

The search starts at the target 500M scale. A 150M search can rank
hyperparameters differently from 500M and would spend compute on a scale that
is not the primary claim. The 150M point is retained only as a late scale-trend
measurement after a useful 500M configuration has been found.

### Stage A: 500M engineering and short signal checks

1. Profile vanilla and the default `K=9`, rho=0.5, top-k=2 distributional model
   for 50 steps.
2. Run the same pair for 100M tokens to verify data order, resume, validation,
   W&B metrics, numerical stability, throughput, and correction activity.
3. Stop if the distributional path is unstable, the correction collapses to
   zero, or its runtime cost is unacceptable without an early NLL signal.

The pilot is an engineering gate, not evidence of model quality.

### Stage B: 500M hyperparameter screen

At seed 1337 and 500M tokens, compare a shared vanilla run with:

- atom count `K={5, 9, 17}` at rho=0.5 and top-k=2;
- rho `{0.25, 0.5, 0.75, 1}` at `K=9` and top-k=2;
- top-k=4 at `K=9`, rho=0.5.

Top-k=1 is an exact-equivalence unit/sanity test and does not consume a
pretraining run. Generic learned reducers, learnable rho, additional seeds, and
150M runs are intentionally excluded at this stage. Rank candidates by held-out
NLL, stability, disagreement-conditioned gain, correction activity, and
throughput rather than by NLL alone.

### Stage C: finalist refinement and frozen 500M confirmation

Take at most two screen finalists and train each against a top-k-matched vanilla
run from scratch for 1B tokens. Select one configuration, freeze `K`, rho, and
top-k, then train the frozen configuration and vanilla from scratch for 5B
tokens. The 5B pair is the seed-1337 primary comparison; it is not a continuation
of a screen checkpoint.

Proceed only if the frozen distributional model improves NLL by at least 0.005
nats/token and the correction/disagreement analyses support the proposed
mechanism. A weaker result is recorded as inconclusive and does not trigger the
expensive matrix.

### Stage D: strong non-distributional controls

Only after the 500M/5B gate passes, train the parameter-matched dense model,
output-gated reducer, and permutation-invariant residual-MLP reducer for the
same 5B tokens. Train learnable rho as a diagnostic for whether the model moves
back toward the vanilla identity at rho=0. If the generic controls match the
selected reducer, narrow the claim to nonlinear expert aggregation.

### Stage E: scale trend and seeds

Use the frozen 500M winner and top-k-matched vanilla at:

| scale | tokens (10 tokens/parameter, rounded) | one-GPU step cap |
|---|---:|---:|
| 150M | 1.5B | 15,000 |
| 500M | 5B | 50,000 |
| 1.5B | 15B | 150,000 |

The 500M point is reused from Stage C. Run the 150M and 1.5B pairs only after
the distribution-specific 500M claim survives Stage D. Finally, add seeds 2027
and 4099 to the 500M vanilla/winner pair; seed 1337 is reused from Stage C.
This ordering postpones seed cost without weakening the final three-seed test.

All scale configs process 64 sequences/GPU per optimizer update:
`32 micro-batch x 2 accumulation` at 150M/500M and
`8 micro-batch x 8 accumulation` at 1.5B. At sequence length 2048 this is
131,072 tokens/GPU/update. Raising accumulation by 2--4x would not reduce
activation peak memory, would enlarge the already substantial four-GPU global
batch, and would halve or quarter the number of optimizer updates. It is
therefore left unchanged unless gradient variance measurements justify a
separately retuned large-batch schedule.

Every comparison records resolved config, exact parameters, data position,
tokens, peak allocated/reserved memory, tokens/second, wall-clock, held-out NLL,
primary benchmarks, correction norm/rho trajectory, and paired mechanism
statistics. Negative outcomes narrow the claim rather than being hidden.

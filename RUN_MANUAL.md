# Run manual: 1–4 x A100 80GB

이 문서는 GPU 서버의 저장 위치를 다음과 같이 고정한다.

```text
code:        /data/umoe_mod_share/kan_moe
raw parquet: /data/umoe_mod_share/fineweb_edu_100bt/sample/100BT
tokenizer:   /data/umoe_mod_share/llama2_tokenizer
outputs:     /data/umoe_mod_share/kan_moe/outputs
```

각 비교 안에서 모든 모델은 동일한 tokenizer snapshot, 데이터 split, seed,
global batch와 token budget을 사용한다. 500M 장기 YAML은 20B/50,000-step
상한을 유지하고, scale curve는 150M/3B, 500M/10B, 1.5B/30B로 별도
pre-register한다. 아래 명령은 저장소 루트에서 실행한다.

## 1. 환경 준비

```bash
cd /data/umoe_mod_share/kan_moe

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
python3 -m pip install -e '.[data,eval,logging,dev]'

export HF_HOME=/data/umoe_mod_share/hf_cache
export TOKENIZERS_PARALLELISM=false
```

CUDA용 PyTorch가 이미 설치된 서버 환경을 전제로 한다. 다음 결과에서
사용 가능한 GPU와 BF16 지원 여부를 먼저 확인한다.

```bash
python3 -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.device_count(), torch.cuda.is_bf16_supported())'
nvidia-smi
```

이 서버의 드라이버가 지원하는 CUDA 상한은 12.8이므로 PyTorch도 반드시
cu128 wheel을 사용한다. index를 지정하지 않고 최신 wheel을 설치하면 cu130
이상 build가 선택되어 `The NVIDIA driver on your system is too old (found
version 12080)` 오류가 날 수 있다.

이미 잘못 설치된 `.venv`는 다음 명령으로 복구한다.

```bash
source .venv/bin/activate
python3 -m pip uninstall -y torch torchvision torchaudio
python3 -m pip install --no-cache-dir torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128

python3 -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

정상 결과는 Torch 버전에 `+cu128`, `torch.version.cuda`에 `12.8`, CUDA
availability에 `True`가 표시된다.

`CUDA_VISIBLE_DEVICES`가 설정되지 않았으면 `train.py`가 PyTorch를 import하기
전에 `nvidia-smi`를 조회한다. 다른 compute process가 없고 사용 중인 GPU
memory가 1GiB 이하인 장치만 선택하며, 동시에 시작한 이 프로젝트의 다른
run과 충돌하지 않도록 `/tmp` lock을 유지한다. `torchrun`의
`--nproc_per_node`만큼 idle GPU가 없으면 기존 작업을 침범하지 않고 실패한다.

명시적인 `CUDA_VISIBLE_DEVICES`가 있으면 자동 선택보다 우선한다. 자동 선택을
사용하려면 다음처럼 기존 값을 제거한다.

```bash
unset CUDA_VISIBLE_DEVICES
```

강제로 기존 CUDA 동작을 사용하려면 `--no-auto-select-gpu`를 추가하거나
`DMOE_AUTO_SELECT_GPU=0`을 설정한다. idle 판정의 허용 memory는 기본
1,024MiB이며 필요하면 `DMOE_GPU_MAX_USED_MEMORY_MIB`로 조정할 수 있다.

## 2. Pod 제한, FineWeb-Edu와 Llama 2 tokenizer 확인

별도의 token `.bin`이나 Hugging Face Arrow dataset을 만들지 않는다.
PyArrow가 Parquet row group을 직접 순차적으로 읽고, 학습 중 fast Llama 2
tokenizer로 online tokenization과 continuous packing을 수행한다. 기본값은
reader/tokenizer batch 각각 4, dataset process 1, PyArrow `use_threads=False`다.

`train.py`는 PyTorch import 전에 CPU affinity, cgroup CPU quota, cgroup memory
limit/current usage를 읽는다. 결과에 따라 OpenMP, MKL, OpenBLAS, Rayon과
PyTorch thread 수를 최대 1–2개로 제한하고 실제 값을 `runtime.json`과 W&B에
기록한다. 서버에서 원시 제한을 확인하려면:

```bash
cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
cat /sys/fs/cgroup/memory.max 2>/dev/null || true
cat /sys/fs/cgroup/memory.current 2>/dev/null || true
python3 -c 'from dmoe.resources import detect_resource_limits; import json; print(json.dumps(detect_resource_limits().to_dict(), indent=2))'
```

```bash
find /data/umoe_mod_share/fineweb_edu_100bt/sample/100BT \
  -maxdepth 1 -name '*.parquet' -type f | sort | wc -l
find /data/umoe_mod_share/llama2_tokenizer \
  -maxdepth 1 -type f | sort

python3 - <<'PY'
from transformers import AutoTokenizer
p = "/data/umoe_mod_share/llama2_tokenizer"
t = AutoTokenizer.from_pretrained(p, use_fast=True, trust_remote_code=False)
print({"path": p, "vocab_size": len(t), "eos": t.eos_token_id,
       "bos": t.bos_token_id, "is_fast": t.is_fast})
assert len(t) == 32000 and t.eos_token_id == 2 and t.is_fast
PY
```

모든 비교군은 동일하게 special token을 자동 추가하지 않고 문서마다 EOS(id
2)를 하나 붙인다. 전체 Parquet sequence의 마지막 10,000 rows는 training에서
제외하고 validation에만 사용한다. 정상적인 시작 로그에는 `parquet_backend:
direct`가 기록되며 `Generating train split`이 나타나지 않는다. 그 문구가
보이면 최신 코드/config가 아니다.

## 3. 코드와 파라미터 검증

```bash
python3 -m unittest discover -s tests -v
python3 scripts/count_parameters.py \
  configs/dense_150m.yaml \
  configs/vanilla_moe_150m.yaml \
  configs/distributional_moe_150m.yaml \
  configs/dense_500m.yaml \
  configs/vanilla_moe_500m.yaml \
  configs/distributional_moe_500m.yaml \
  configs/output_gated_moe_500m.yaml \
  configs/residual_mlp_moe_500m.yaml \
  configs/dense_1_5b.yaml \
  configs/vanilla_moe_1_5b.yaml \
  configs/distributional_moe_1_5b.yaml
```

Dense/vanilla/distributional 기대값은 각각 150M scale에서
`149,303,808 / 149,352,960 / 149,352,960`, 500M에서
`504,122,112 / 504,195,840 / 504,195,840`, 1.5B에서
`1,490,322,432 / 1,490,453,504 / 1,490,453,504`이다. 각 scale의
dense–MoE 오차는 0.04% 미만이고 vanilla와 fixed distributional은 정확히
같다. 500M output-gated는 `504,200,448`, residual-MLP는 `505,080,576`으로
각각 vanilla 대비 약 0.001%, 0.176% 추가된다.

## 4. 100M-token pilot

실험 matrix 도구는 기본적으로 명령만 출력하며 `--execute`를 붙여야 실제로
순차 실행한다. 먼저 150M Hellinger로 data/DDP/eval/checkpoint를 검증한다.

```bash
python3 scripts/experiment_matrix.py \
  --stage pilot \
  --nproc-per-node 4

python3 scripts/experiment_matrix.py \
  --stage pilot \
  --nproc-per-node 4 \
  --execute
```

중단된 단일 run은 출력된 명령 끝에 `--override train.resume=latest`를 붙인다.

전체 run 전에 500M controls와 1.5B 모델을 각각 50 step profiling해 OOM,
throughput, peak allocated/reserved memory를 확인한다.

```bash
python3 scripts/experiment_matrix.py \
  --stage profiling \
  --nproc-per-node 4

# 출력 명령 검토 후 --execute
```

## 5. 150M falsification screening

다음 stage는 500M tokens에서 vanilla/Hellinger top-k `{1,2,4}`, simplex atom count
`K={5,9,17}`, geometric/Hellinger/arithmetic, learned output gate,
permutation-invariant residual MLP, learnable rho 초기값 `{0,0.5,1}`를 비교한다.
`K=9`, top-k=2 Hellinger run은 중복 실행하지 않고 각 sweep의 공통 기준으로
사용한다.

```bash
python3 scripts/experiment_matrix.py \
  --stage screening \
  --nproc-per-node 4

# 명령을 검토한 뒤에만 실행
python3 scripts/experiment_matrix.py \
  --stage screening \
  --nproc-per-node 4 \
  --execute
```

`geometric`은 vanilla와 output/gradient가 정확히 같은 항등 control이다.
Hellinger가 learned gate 또는 residual MLP와 통계적으로 구분되지 않으면
distribution-specific claim을 포기하고 generic nonlinear reducer 결과로
해석한다.

## 6. scale 및 seed 비교

Scale stage는 20 tokens/parameter에 맞춰 150M/3B, 500M/10B, 1.5B/30B를
dense, vanilla, Hellinger에 대해 seed 1337로 실행한다. Seed stage는
500M/10B에서 vanilla, Hellinger, output-gated, residual-MLP를
`{1337,2027,4099}`로 실행한다.

```bash
python3 scripts/experiment_matrix.py --stage scaling --nproc-per-node 4
python3 scripts/experiment_matrix.py --stage seeds --nproc-per-node 4

# 앞 단계가 성공 기준을 통과한 후 각각 --execute 추가
```

500M 기본 YAML은 요청한 장기 설정인 micro-batch 32/GPU, accumulation 2,
`max_steps=50000`, `max_tokens=20B`를 유지한다. 한 GPU에서는 50k-step
제한으로 6.554B에서 끝나고 네 GPU에서는 20B 제한으로 38,147 step에서
끝난다. Matrix 도구는 scale/seed 비교 시 500M을 10B와 충분한 max step으로
명시적으로 override한다.

학습 로그에는 NLL/PPL, router load/entropy, correction ratio, learnable rho,
tokens/sec와 CUDA allocated/reserved peak memory가 포함된다.

## 7. held-out perplexity와 seed 통계

각 run은 동일한 validation 10M tokens에서 평가한다. 결과에는 sequence-block
bootstrap NLL/PPL 95% interval이 포함된다.

```bash
RUN=/data/umoe_mod_share/kan_moe/outputs/revised/seed-500m-hellinger-seed1337
CKPT=$(ls -1 "${RUN}"/step_*.pt | sort | tail -n 1)

torchrun --standalone --nproc_per_node=4 evaluate_ppl.py \
  --checkpoint "${CKPT}" \
  --max-tokens 10000000 \
  --batch-size 4 \
  --bootstrap-iters 1000 \
  --output "${RUN}/ppl_10m.json"
```

세 seed 결과가 준비되면 같은 seed끼리 paired NLL 차이와 noise floor를
계산한다.

```bash
# candidate 실행 전 baseline noise floor와 MDE 추정
python3 scripts/summarize_seeds.py \
  --baseline /path/to/vanilla_seed1337.json /path/to/vanilla_seed2027.json /path/to/vanilla_seed4099.json

# same-seed paired 비교
python3 scripts/summarize_seeds.py \
  --baseline /path/to/vanilla_seed1337.json /path/to/vanilla_seed2027.json /path/to/vanilla_seed4099.json \
  --candidate /path/to/hellinger_seed1337.json /path/to/hellinger_seed2027.json /path/to/hellinger_seed4099.json \
  --output /path/to/hellinger_vs_vanilla_seed_summary.json
```

사전 성공 기준은 mean NLL 0.005 nats/token 이상 개선 및 paired seed 95% CI가
0을 제외하는 것이다.

## 8. disagreement–loss 및 router-gradient 분석

같은 seed/data order로 학습한 distributional/vanilla checkpoint를 짝지어
token별 weighted JS disagreement와 NLL gain의 상관, disagreement decile,
trained distributional checkpoint를 geometric으로 바꾼 counterfactual,
router gradient norm/direction 변화를 계산한다.

```bash
python3 analyze_mechanism.py \
  --checkpoint /path/to/distributional_step.pt \
  --baseline-checkpoint /path/to/same_seed_vanilla_step.pt \
  --max-tokens 1000000 \
  --batch-size 1 \
  --router-gradient-batches 8 \
  --bootstrap-iters 1000 \
  --device cuda:0 \
  --output /path/to/mechanism_analysis.json
```

Router-gradient 분석은 메모리가 크므로 우선 8 batch만 사용한다. 보고할 핵심은
disagreement–gain Spearman CI, 최고/최저 disagreement decile의 gain 차이,
counterfactual gain, Hellinger/geometric router-gradient cosine이다.

## 9. benchmark

Primary suite는 LAMBADA OpenAI, PIQA, HellaSwag이다. MMLU, ARC,
WinoGrande, OpenBookQA, BoolQ는 작은 scale에서 near-chance일 수 있으므로
secondary/exploratory로 분리한다.

```bash
python3 evaluate_harness.py \
  --checkpoint /path/to/step.pt \
  --tokenizer /data/umoe_mod_share/llama2_tokenizer \
  --suite primary \
  --device cuda:0 \
  --batch-size 8 \
  --bootstrap-iters 1000 \
  --cache-requests \
  --output /path/to/primary_benchmarks.json

python3 evaluate_harness.py \
  --checkpoint /path/to/step.pt \
  --tokenizer /data/umoe_mod_share/llama2_tokenizer \
  --suite secondary \
  --device cuda:0 \
  --batch-size 8 \
  --output /path/to/secondary_benchmarks.json
```

빠른 연결 검증에서만 `--limit 0.01`을 사용한다. 최종 결과에서는 쓰지 않는다.

## 10. 비교 시 반드시 고정·기록할 항목

- local Llama 2 tokenizer 파일과 vocab/EOS 계약
- raw Parquet 목록, row/row-group layout, direct-streaming batch 제한
- cgroup CPU/memory limit과 실제 native/PyTorch thread 수
- train/validation row 경계와 online packing 정책
- tokens seen, seed, sequence length, global batch, optimizer schedule
- stable name-based initialization seed와 shared-weight identity
- total parameters와 top-k별 active parameters
- PPL token count와 benchmark harness version/task configuration
- A100 수, peak memory, wall-clock, tokens/sec

Dense는 total parameter가 같지만 모든 FFN parameter를 활성화하고 sparse MoE는
top-k expert만 활성화한다. 따라서 결과표에서 parameter-matched 결과와
compute-matched 해석을 구분한다.

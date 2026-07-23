# Run manual: 1–4 x A100 80GB

이 문서는 GPU 서버의 저장 위치를 다음과 같이 고정한다.

```text
code:        /data/umoe_mod_share/kan_moe
raw parquet: /data/umoe_mod_share/fineweb_edu_100bt/sample/100BT
tokenizer:   /data/umoe_mod_share/llama2_tokenizer
outputs:     /data/umoe_mod_share/kan_moe/outputs
```

각 비교 안에서 모든 모델은 동일한 tokenizer snapshot, 데이터 split, seed,
global batch와 token budget을 사용한다. 모든 scale은 nominal model size의
10배 tokens로 맞춘다: 150M/1.5B, 500M/5B, 1.5B/15B. 한 GPU에서도 이
budget에 도달하도록 step 상한은 각각 15,000/50,000/150,000으로 둔다. 아래
명령은 저장소 루트에서 실행한다.

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
2)를 하나 붙인다. 전체 Parquet sequence의 마지막 20,000 rows는 training에서
제외한다. 앞 10,000 rows는 hyperparameter 선택용 validation, 마지막 10,000
rows는 설정 동결 뒤 한 번 여는 test split이다. 정상적인 시작 로그에는
`parquet_backend: direct`가 기록되며 `Generating train split`이 나타나지
않는다. 그 문구가 보이면 최신 코드/config가 아니다.

이 split 변경 전 checkpoint는 training row boundary가 다르므로 새 config로
resume하지 않는다. 이번 protocol 비교는 모두 새 output directory에서
처음부터 시작한다.

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

## 4. 500M profiling과 100M-token pilot

실험 matrix 도구는 기본적으로 명령만 출력하며 `--execute`를 붙여야 실제로
순차 실행한다. 먼저 500M vanilla와 기본 distributional pair를 50 step
profile해서 OOM, throughput, peak allocated/reserved memory를 확인한다.

```bash
python3 scripts/experiment_matrix.py \
  --stage profiling \
  --nproc-per-node 4

# 출력 명령 검토 후 --execute
```

이후 같은 두 모델을 100M tokens 학습해
data/DDP/eval/checkpoint/W&B/correction metric을 검증한다.

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

## 5. 500M hyperparameter screening

다음 stage는 seed 1337, 500M tokens에서 shared vanilla와 atom count
`K={5,9,17}`, rho `{0.25,0.5,0.75,1}`, top-k `{2,4}`의 핵심 조합만
비교한다. learned output gate, residual MLP, learnable rho, 추가 seed와
150M은 아직 실행하지 않는다. top-k=1은 exact-equivalence test로 이미
검증하므로 학습 sweep에서 제외한다.

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

screen 결과는 NLL만으로 고르지 않고 numerical stability, correction
activity, disagreement-conditioned gain과 throughput을 함께 본다. 후보는
최대 두 개만 다음 단계로 보낸다.

## 6. finalist confirmation, controls, scale, seeds

각 finalist를 top-k-matched vanilla와 1B tokens에서 처음부터 다시 학습한다.
아래 예시는 `K=9`, rho=0.5, top-k=2이며 screen winner 값으로 바꾼다.

```bash
python3 scripts/experiment_matrix.py \
  --stage confirmation \
  --winner-distribution-k 9 \
  --winner-rho 0.5 \
  --winner-top-k 2 \
  --confirmation-tokens 1000000000 \
  --nproc-per-node 4
```

1B 결과로 하나를 고른 뒤 `K/rho/top-k`를 동결하고, 같은 명령에서
`--confirmation-tokens`를 생략해 5B pair를 처음부터 학습한다. 5B pair가
NLL 0.005 nats/token 이상 개선하고 mechanism 진단도 통과할 때만 다음
controls를 실행한다.

두 finalist의 top-k가 같으면 첫 명령은 `--confirmation-role both`, 두 번째는
`--confirmation-role candidate`로 실행해 동일한 vanilla를 중복 학습하지
않는다. top-k가 다르면 각각 top-k-matched vanilla가 필요하다.

```bash
python3 scripts/experiment_matrix.py \
  --stage controls \
  --winner-distribution-k 9 \
  --winner-rho 0.5 \
  --winner-top-k 2 \
  --nproc-per-node 4
```

Dense/output-gated/residual-MLP와 learnable-rho control까지 확인한 뒤에만
scale curve를 실행하고, seed 실험은 마지막에 한다.

```bash
python3 scripts/experiment_matrix.py \
  --stage scaling \
  --winner-distribution-k 9 \
  --winner-rho 0.5 \
  --winner-top-k 2 \
  --nproc-per-node 4

python3 scripts/experiment_matrix.py \
  --stage seeds \
  --winner-distribution-k 9 \
  --winner-rho 0.5 \
  --winner-top-k 2 \
  --nproc-per-node 4

# 각 단계의 출력과 이전 gate를 확인한 뒤에만 --execute 추가
```

Scaling은 500M confirmation을 재사용하고 150M/1.5B pair만 생성한다.
Seed stage도 seed 1337 confirmation을 재사용하고 2027/4099 pair만 생성한다.

각 GPU의 optimizer update당 batch는 64 sequences로 유지한다. 150M/500M은
micro-batch 32와 accumulation 2, 1.5B는 micro-batch 8과 accumulation 8이다.
sequence length 2048에서 131,072 tokens/GPU/update이고, 1 GPU 기준 budget
도달 step은 약 11,445/38,147/114,441이다. 현재 step 상한에는 여유가 있다.

Gradient accumulation을 2--4배 늘려도 activation peak memory는 줄지 않는다.
4 GPU에서 500M/5B는 현재도 global update batch가 256 sequences이며 약
9,537 optimizer updates를 수행한다. accumulation을 4 또는 8로 올리면
약 4,769/2,385 updates로 줄어 LR schedule까지 다시 조정해야 하므로
현재 값 2를 유지한다.

학습 로그에는 NLL/PPL, router load/entropy, correction ratio, learnable rho,
tokens/sec와 CUDA allocated/reserved peak memory가 포함된다.

## 7. held-out perplexity와 seed 통계

Screen과 1B refinement는 동일한 validation window에서 평가한다. winner를
동결한 뒤 5B confirmation과 최종 seed 통계는 `--split test`의 동일한
untouched window에서 평가한다. 결과에는 sequence-block bootstrap NLL/PPL
95% interval이 포함된다.

```bash
RUN=/data/umoe_mod_share/kan_moe/outputs/revised/confirm-500m-atoms9-rho0p5-k2-5b
CKPT=$(ls -1 "${RUN}"/step_*.pt | sort | tail -n 1)

torchrun --standalone --nproc_per_node=4 evaluate_ppl.py \
  --checkpoint "${CKPT}" \
  --split test \
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
  --split test \
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
- train/validation/test row 경계와 online packing 정책
- tokens seen, seed, sequence length, global batch, optimizer schedule
- stable name-based initialization seed와 shared-weight identity
- total parameters와 top-k별 active parameters
- PPL token count와 benchmark harness version/task configuration
- A100 수, peak memory, wall-clock, tokens/sec

Dense는 total parameter가 같지만 모든 FFN parameter를 활성화하고 sparse MoE는
top-k expert만 활성화한다. 따라서 결과표에서 parameter-matched 결과와
compute-matched 해석을 구분한다.

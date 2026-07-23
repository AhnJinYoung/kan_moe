# Run manual: 1–4 x A100 80GB

이 문서는 GPU 서버의 저장 위치를 다음과 같이 고정한다.

```text
code:        /data/umoe_mod_share/kan_moe
raw parquet: /data/umoe_mod_share/fineweb_edu_100bt/sample/100BT
tokenizer:   /data/umoe_mod_share/llama2_tokenizer
outputs:     /data/umoe_mod_share/kan_moe/outputs
```

모든 비교군은 동일한 tokenizer snapshot, 데이터 split, seed, global batch,
5B-token budget을 사용한다. 아래 명령은 저장소 루트에서 실행한다.

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
  configs/dense_500m.yaml \
  configs/vanilla_moe_500m.yaml \
  configs/distributional_moe_500m.yaml
```

기대값은 dense `504,122,112`, vanilla MoE와 distributional MoE 각각
`504,195,840`이다. 두 MoE의 total parameter 수는 정확히 같고, dense와의
차이는 약 0.015%이다.

## 4. 100M-token 파이프라인 pilot

먼저 Hellinger 모델로 데이터, DDP, validation, checkpoint/resume을 끝까지
검증한다. 네 GPU에서 optimizer step당 524,288 tokens이므로 약 191 steps다.

```bash
unset CUDA_VISIBLE_DEVICES
torchrun --standalone --nproc_per_node=4 train.py \
  --config configs/distributional_moe_500m.yaml \
  --override train.max_tokens=100000000 \
  --override train.eval_interval=100 \
  --override train.save_interval=100 \
  --override train.output_dir=/data/umoe_mod_share/kan_moe/outputs/pilot_dmoe_k2
```

resume 검증:

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --config configs/distributional_moe_500m.yaml \
  --override train.max_tokens=200000000 \
  --override train.resume=latest \
  --override train.output_dir=/data/umoe_mod_share/kan_moe/outputs/pilot_dmoe_k2
```

## 5. top-k 및 aggregator screening

`model.top_k`는 1부터 16까지 바꿀 수 있고 parameter 수는 변하지 않는다.
top-k=1은 aggregate가 발생하지 않으므로 codec 동등성 확인용이지, 제안한
aggregation 효과를 판단하는 실험은 아니다. 먼저 `{1,2,4}`를 300M tokens로
screen한다.

```bash
for K in 1 2 4; do
  torchrun --standalone --nproc_per_node=4 train.py \
    --config configs/distributional_moe_500m.yaml \
    --override model.top_k=${K} \
    --override train.max_tokens=300000000 \
    --override train.output_dir=/data/umoe_mod_share/kan_moe/outputs/screen_dmoe_hellinger_k${K}
done
```

top-k=2에서 수학적 negative control과 다른 pool을 비교한다.

```bash
for AGG in geometric hellinger arithmetic; do
  torchrun --standalone --nproc_per_node=4 train.py \
    --config configs/distributional_moe_500m.yaml \
    --override model.top_k=2 \
    --override model.aggregation=${AGG} \
    --override train.max_tokens=300000000 \
    --override train.output_dir=/data/umoe_mod_share/kan_moe/outputs/screen_dmoe_${AGG}_k2
done
```

`geometric`은 ILR 공간에서 vanilla linear sum과 output/gradient가 정확히
동등한 control이다. primary 후보는 `hellinger`다.

## 6. 5B-token 주 비교

한 번에 하나씩 실행한다. YAML의 기본값은 모두 micro-batch 2/GPU,
gradient accumulation 32, 5B tokens, seed 1337이다. `max_steps: 0`이므로
GPU 수와 무관하게 `max_tokens`가 종료 조건이다. 한 GPU에서는 optimizer
step당 131,072 tokens로 38,147 steps, 네 GPU에서는 524,288 tokens로
9,537 steps다. 세 모델은 같은 row order와 online packing 규칙을 사용한다.

단일 A100용 micro-batch와 accumulation은 YAML에 설정되어 있다고 가정한다.

```bash
unset CUDA_VISIBLE_DEVICES
torchrun --standalone --nproc_per_node=1 train.py \
  --config configs/distributional_moe_500m.yaml \
  --override train.wandb_project=kan-moe \
  --override train.wandb_run_name=dmoe-hellinger-k2-5b-1gpu
```

아래는 네 GPU 비교 명령이다.

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --config configs/dense_500m.yaml

torchrun --standalone --nproc_per_node=4 train.py \
  --config configs/vanilla_moe_500m.yaml

torchrun --standalone --nproc_per_node=4 train.py \
  --config configs/distributional_moe_500m.yaml
```

중단된 run은 해당 명령에 `--override train.resume=latest`를 추가한다.
`metrics.jsonl`에는 tokens/sec, NLL/PPL, router entropy/load, distribution
entropy, nonlinear correction ratio가 기록된다. quality-vs-token뿐 아니라
wall-clock과 throughput도 함께 보고한다.

## 7. held-out perplexity

각 run의 최신 checkpoint를 찾아 validation split의 10M tokens에서 PPL을
계산한다. 세 모델에서 `--max-tokens`, sequence length, GPU 수를 동일하게
유지한다.

```bash
RUN=/data/umoe_mod_share/kan_moe/outputs/distributional_moe_500m_k2
CKPT=$(ls -1 "${RUN}"/step_*.pt | sort | tail -n 1)

torchrun --standalone --nproc_per_node=4 evaluate_ppl.py \
  --checkpoint "${CKPT}" \
  --max-tokens 10000000 \
  --batch-size 4 \
  --output "${RUN}/ppl_10m.json"
```

다른 top-k를 checkpoint에 사후 적용하는 `--top-k` 옵션은 진단용이다.
primary score는 학습에 사용한 top-k로 계산한다.

## 8. MMLU, ARC, HellaSwag 등 benchmark

benchmark는 한 GPU에서 실행한다. `lm-eval==0.4.12`를 세 모델 모두에
고정하고 task별 harness 기본 few-shot 설정을 유지한다. 결과 JSON에 task
설정과 metric이 함께 저장된다.

```bash
export CUDA_VISIBLE_DEVICES=0
RUN=/data/umoe_mod_share/kan_moe/outputs/distributional_moe_500m_k2
CKPT=$(ls -1 "${RUN}"/step_*.pt | sort | tail -n 1)

python3 evaluate_harness.py \
  --checkpoint "${CKPT}" \
  --tokenizer /data/umoe_mod_share/llama2_tokenizer \
  --tasks mmlu,arc_easy,arc_challenge,hellaswag,piqa,winogrande,openbookqa,boolq,lambada_openai \
  --device cuda:0 \
  --batch-size 8 \
  --bootstrap-iters 1000 \
  --cache-requests \
  --output "${RUN}/benchmarks.json"
```

메모리가 부족하면 benchmark의 `--batch-size`만 4 또는 2로 낮춘다. 모델
점수에는 영향을 주지 않고 처리량만 달라진다. 빠른 연결 검증에는
`--limit 0.01`을 추가하되, 보고할 최종 결과에는 `--limit`을 사용하지 않는다.

## 9. 비교 시 반드시 고정·기록할 항목

- local Llama 2 tokenizer 파일과 vocab/EOS 계약
- raw Parquet 목록, row/row-group layout, direct-streaming batch 제한
- cgroup CPU/memory limit과 실제 native/PyTorch thread 수
- train/validation row 경계와 online packing 정책
- tokens seen, seed, sequence length, global batch, optimizer schedule
- total parameters와 top-k별 active parameters
- PPL token count와 benchmark harness version/task configuration
- A100 수, peak memory, wall-clock, tokens/sec

Dense는 total parameter가 같지만 모든 FFN parameter를 활성화하고 sparse MoE는
top-k expert만 활성화한다. 따라서 결과표에서 parameter-matched 결과와
compute-matched 해석을 구분한다.

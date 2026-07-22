# Run manual: 4 x A100 80GB

이 문서는 GPU 서버의 저장 위치를 다음과 같이 고정한다.

```text
code:        /data/umoe_mod_share/kan_moe
raw parquet: /data/umoe_mod_share/fineweb_edu_100bt/sample/100BT
tokens:      /data/umoe_mod_share/fineweb_edu_100bt/tokenized_mistral_v3
outputs:     /data/umoe_mod_share/kan_moe/outputs
```

모든 비교군은 동일한 tokenizer snapshot, 데이터 split, seed, global batch,
5B-token budget을 사용한다. 아래 명령은 저장소 루트에서 실행한다.

## 1. 환경 준비

```bash
cd /data/umoe_mod_share/kan_moe

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[data,eval,logging,dev]'

export HF_HOME=/data/umoe_mod_share/hf_cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
```

CUDA용 PyTorch가 이미 설치된 서버 환경을 전제로 한다. 다음 결과에서
GPU 네 개와 BF16 지원 여부를 먼저 확인한다.

```bash
python3 -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.device_count(), torch.cuda.is_bf16_supported())'
nvidia-smi
```

## 2. FineWeb-Edu sample-100BT 전처리

토큰 데이터는 약 100B tokens x 2 bytes 규모이므로 metadata를 포함해 최소
약 210GB의 여유 공간을 확보한다.

```bash
df -h /data/umoe_mod_share
find /data/umoe_mod_share/fineweb_edu_100bt/sample/100BT \
  -maxdepth 1 -name '*.parquet' -type f | sort | wc -l
```

다음 명령은 base Mistral v0.3 tokenizer의 정확한 snapshot을 내려받고,
문서마다 자동 special token 없이 tokenize한 뒤 EOS(id 2)를 하나 붙인다.
정렬상 마지막 파일 `006_00005.parquet` 한 개가 validation으로 분리된다.

```bash
python3 prepare_fineweb.py \
  --input-dir /data/umoe_mod_share/fineweb_edu_100bt/sample/100BT \
  --input-glob '*.parquet' \
  --output-dir /data/umoe_mod_share/fineweb_edu_100bt/tokenized_mistral_v3 \
  --tokenizer mistralai/Mistral-7B-v0.3 \
  --tokenizer-revision caa1feb0e54d415e2df31207e5f4e273e33509b1 \
  --expected-vocab-size 32768 \
  --text-column text \
  --dtype uint16 \
  --validation-files 1 \
  --batch-size 256 \
  --workers 16
```

중단 후 같은 명령을 다시 실행하면 metadata와 파일 크기가 일치하는 완료
shard는 건너뛴다. 의도적으로 다시 만들 때만 `--overwrite`를 추가한다.
완료 후 계약을 확인한다.

```bash
python3 -c 'import json; p="/data/umoe_mod_share/fineweb_edu_100bt/tokenized_mistral_v3/manifest.json"; d=json.load(open(p)); print(d["tokenizer"]); print(d["validation_files"]); print(d["splits"])'
du -sh /data/umoe_mod_share/fineweb_edu_100bt/tokenized_mistral_v3
```

## 3. 코드와 파라미터 검증

```bash
python3 -m unittest discover -s tests -v
python3 scripts/count_parameters.py \
  configs/dense_500m.yaml \
  configs/vanilla_moe_500m.yaml \
  configs/distributional_moe_500m.yaml
```

기대값은 dense `504,711,936`, vanilla MoE와 distributional MoE 각각
`504,785,664`이다. 두 MoE의 total parameter 수는 정확히 같고, dense와의
차이는 약 0.015%이다.

## 4. 100M-token 파이프라인 pilot

먼저 Hellinger 모델로 데이터, DDP, validation, checkpoint/resume을 끝까지
검증한다. 네 GPU에서 optimizer step당 524,288 tokens이므로 약 191 steps다.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

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
gradient accumulation 32, 5B tokens, seed 1337, global batch 524,288
tokens이며 최종 step은 9,537이다. micro-batch까지 통일했으므로 rank별
sampler가 세 모델에 동일한 token window 순서를 공급한다.

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
  --tokenizer /data/umoe_mod_share/fineweb_edu_100bt/tokenized_mistral_v3/tokenizer \
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

- tokenizer source와 exact revision, `manifest.json`
- train/validation source shard 목록
- tokens seen, seed, sequence length, global batch, optimizer schedule
- total parameters와 top-k별 active parameters
- PPL token count와 benchmark harness version/task configuration
- A100 수, peak memory, wall-clock, tokens/sec

Dense는 total parameter가 같지만 모든 FFN parameter를 활성화하고 sparse MoE는
top-k expert만 활성화한다. 따라서 결과표에서 parameter-matched 결과와
compute-matched 해석을 구분한다.

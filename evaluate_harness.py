from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_TASKS = (
    "mmlu,arc_easy,arc_challenge,hellaswag,piqa,winogrande,"
    "openbookqa,boolq,lambada_openai"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a native checkpoint with lm-evaluation-harness"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--bootstrap-iters", type=int, default=1_000)
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--cache-requests", action="store_true")
    parser.add_argument("--log-samples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import lm_eval
    except ImportError as error:
        raise RuntimeError(
            "lm-evaluation-harness is required; run `pip install -e '.[eval]'`"
        ) from error

    from dmoe.lm_eval_adapter import CheckpointHarnessLM

    model = CheckpointHarnessLM(
        checkpoint=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=args.device,
        batch_size=args.batch_size,
        top_k=args.top_k,
        precision=args.precision,
    )
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=[task.strip() for task in args.tasks.split(",") if task.strip()],
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        bootstrap_iters=args.bootstrap_iters,
        cache_requests=args.cache_requests,
        log_samples=args.log_samples,
    )
    if results is None:
        return
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)
    print(f"saved benchmark results to {output_path}")
    if "results" in results:
        print(json.dumps(results["results"], indent=2, default=str))


if __name__ == "__main__":
    main()


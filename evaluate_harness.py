from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path


TASK_SUITES = {
    "primary": "lambada_openai,piqa,hellaswag",
    "secondary": (
        "mmlu,arc_easy,arc_challenge,winogrande,openbookqa,boolq"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a native checkpoint with lm-evaluation-harness"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--suite",
        choices=("primary", "secondary", "all"),
        default="primary",
        help="Pre-registered primary tasks are the default; secondary tasks are exploratory.",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="Explicit comma-separated tasks; overrides --suite.",
    )
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
    task_spec = args.tasks
    if not task_spec:
        task_spec = TASK_SUITES["primary"]
        if args.suite in {"secondary", "all"}:
            secondary = TASK_SUITES["secondary"]
            task_spec = (
                secondary if args.suite == "secondary" else f"{task_spec},{secondary}"
            )
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=[task.strip() for task in task_spec.split(",") if task.strip()],
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
    manifest_path = Path(args.tokenizer).resolve().parent / "manifest.json"
    tokenizer_manifest = None
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            tokenizer_manifest = json.load(handle).get("tokenizer")
    results["dmoe_metadata"] = {
        "model": model.get_model_info(),
        "tokenizer_path": str(Path(args.tokenizer).resolve()),
        "tokenizer_manifest": tokenizer_manifest,
        "lm_eval_version": importlib.metadata.version("lm-eval"),
        "transformers_version": importlib.metadata.version("transformers"),
        "evaluation_suite": args.suite if not args.tasks else "explicit",
        "primary_tasks": TASK_SUITES["primary"].split(","),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)
    print(f"saved benchmark results to {output_path}")
    if "results" in results:
        print(json.dumps(results["results"], indent=2, default=str))


if __name__ == "__main__":
    main()

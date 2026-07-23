from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _load(paths: list[str]) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        seed = result.get("training_seed")
        if seed is None:
            raise ValueError(f"{path} does not contain training_seed")
        if int(seed) in results:
            raise ValueError(f"duplicate seed {seed} in {paths}")
        results[int(seed)] = result
    return results


def _summary(values: list[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    standard_error = standard_deviation / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
    }


def _t_critical(sample_count: int) -> float:
    # Exact 97.5% Student-t quantiles for the pre-registered n=2..5 range.
    return {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}.get(
        sample_count - 1, 1.96
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize same-seed PPL results and paired NLL differences."
    )
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument(
        "--candidate",
        nargs="*",
        default=[],
        help="Omit to estimate the baseline seed noise floor before candidate runs.",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    baseline = _load(args.baseline)
    baseline_nll = [
        float(baseline[seed]["mean_nll"]) for seed in sorted(baseline)
    ]
    baseline_summary = _summary(baseline_nll)
    if not args.candidate:
        n = len(baseline_nll)
        standard_deviation = float(baseline_summary["standard_deviation"])
        result = {
            "seeds": sorted(baseline),
            "baseline_nll": baseline_summary,
            "estimated_unpaired_equal_n_95pct_mde": (
                _t_critical(n)
                * standard_deviation
                * math.sqrt(2.0 / n)
            ),
        }
        serialized = json.dumps(result, indent=2)
        print(serialized)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(serialized + "\n", encoding="utf-8")
        return

    candidate = _load(args.candidate)
    seeds = sorted(set(baseline) & set(candidate))
    if seeds != sorted(baseline) or seeds != sorted(candidate):
        raise ValueError(
            "baseline and candidate must contain exactly the same seeds"
        )
    candidate_nll = [float(candidate[seed]["mean_nll"]) for seed in seeds]
    differences = [
        candidate_value - baseline_value
        for candidate_value, baseline_value in zip(
            candidate_nll, baseline_nll
        )
    ]
    difference_summary = _summary(differences)
    t_critical = _t_critical(len(seeds))
    half_width = t_critical * float(difference_summary["standard_error"])
    mean_difference = float(difference_summary["mean"])
    result = {
        "seeds": seeds,
        "baseline_nll": baseline_summary,
        "candidate_nll": _summary(candidate_nll),
        "paired_candidate_minus_baseline_nll": {
            **difference_summary,
            "t_95ci": [
                mean_difference - half_width,
                mean_difference + half_width,
            ],
            "candidate_relative_ppl_change": math.exp(mean_difference) - 1.0,
            "observed_95pct_minimum_detectable_nll": half_width,
            "passes_preregistered_0.005_nll_threshold": (
                mean_difference <= -0.005
                and mean_difference + half_width < 0.0
            ),
        },
    }
    serialized = json.dumps(result, indent=2)
    print(serialized)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

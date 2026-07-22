from __future__ import annotations

import argparse
import json

import torch

from dmoe.config import load_experiment_config
from dmoe.model import DecoderLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Count total and active parameters")
    parser.add_argument("configs", nargs="+", help="Experiment YAML files")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    reports = []
    for config_path in args.configs:
        config = load_experiment_config(config_path)
        if args.top_k is not None and config.model.model_type != "dense":
            config.model.top_k = args.top_k
            config.model.validate()
        with torch.device("meta"):
            model = DecoderLM(config.model)
        report = model.parameter_report()
        report.update(
            {
                "config": config_path,
                "model_type": config.model.model_type,
                "top_k": config.model.top_k,
            }
        )
        reports.append(report)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()


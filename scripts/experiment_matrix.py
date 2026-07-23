from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Run:
    name: str
    config: str
    overrides: dict[str, object] = field(default_factory=dict)


def _screening_runs() -> list[Run]:
    runs: list[Run] = []
    budget = {
        "train.max_tokens": 500_000_000,
        "train.max_steps": 50_000,
    }
    for top_k in (1, 2, 4):
        runs.append(
            Run(
                f"screen-150m-hellinger-k{top_k}",
                "configs/distributional_moe_150m.yaml",
                {**budget, "model.top_k": top_k},
            )
        )
        runs.append(
            Run(
                f"screen-150m-vanilla-k{top_k}",
                "configs/vanilla_moe_150m.yaml",
                {**budget, "model.top_k": top_k},
            )
        )
    # The top-k=2 Hellinger run above is also the K=9 reference.
    for distribution_k in (5, 17):
        runs.append(
            Run(
                f"screen-150m-atoms{distribution_k}-k2",
                "configs/distributional_moe_150m.yaml",
                {**budget, "model.distribution_k": distribution_k},
            )
        )
    # Hellinger is already represented by the top-k=2 reference.
    for aggregation in ("geometric", "arithmetic"):
        runs.append(
            Run(
                f"screen-150m-{aggregation}-k2",
                "configs/distributional_moe_150m.yaml",
                {**budget, "model.aggregation": aggregation},
            )
        )
    for model_type in ("output_gated_moe", "residual_mlp_moe"):
        runs.append(
            Run(
                f"screen-150m-{model_type}-k2",
                "configs/distributional_moe_150m.yaml",
                {**budget, "model.model_type": model_type},
            )
        )
    for rho in (0.0, 0.5, 1.0):
        runs.append(
            Run(
                f"screen-150m-learned-rho-init{rho:g}-k2",
                "configs/distributional_moe_150m.yaml",
                {
                    **budget,
                    "model.aggregation": "power",
                    "model.learnable_rho": True,
                    "model.power_rho": rho,
                },
            )
        )
    return runs


def _scaling_runs() -> list[Run]:
    scale_specs = (
        ("150m", 3_000_000_000),
        ("500m", 10_000_000_000),
        ("1_5b", 30_000_000_000),
    )
    runs: list[Run] = []
    for scale, tokens in scale_specs:
        for model in ("dense", "vanilla_moe", "distributional_moe"):
            runs.append(
                Run(
                    f"scale-{scale}-{model}-seed1337",
                    f"configs/{model}_{scale}.yaml",
                    {
                        "train.max_tokens": tokens,
                        "train.max_steps": 250_000,
                    },
                )
            )
    return runs


def _seed_runs() -> list[Run]:
    configs = {
        "vanilla": "configs/vanilla_moe_500m.yaml",
        "hellinger": "configs/distributional_moe_500m.yaml",
        "output-gated": "configs/output_gated_moe_500m.yaml",
        "residual-mlp": "configs/residual_mlp_moe_500m.yaml",
    }
    runs: list[Run] = []
    for seed in (1337, 2027, 4099):
        for label, config in configs.items():
            runs.append(
                Run(
                    f"seed-500m-{label}-seed{seed}",
                    config,
                    {
                        "train.seed": seed,
                        "train.max_tokens": 10_000_000_000,
                        "train.max_steps": 250_000,
                    },
                )
            )
    return runs


def _pilot_runs() -> list[Run]:
    return [
        Run(
            "pilot-150m-hellinger-k2",
            "configs/distributional_moe_150m.yaml",
            {
                "train.max_tokens": 100_000_000,
                "train.max_steps": 10_000,
                "train.eval_interval": 100,
                "train.save_interval": 100,
            },
        )
    ]


def _profiling_runs() -> list[Run]:
    configs = (
        ("500m-vanilla", "configs/vanilla_moe_500m.yaml"),
        ("500m-hellinger", "configs/distributional_moe_500m.yaml"),
        ("500m-output-gated", "configs/output_gated_moe_500m.yaml"),
        ("500m-residual-mlp", "configs/residual_mlp_moe_500m.yaml"),
        ("1_5b-dense", "configs/dense_1_5b.yaml"),
        ("1_5b-vanilla", "configs/vanilla_moe_1_5b.yaml"),
        ("1_5b-hellinger", "configs/distributional_moe_1_5b.yaml"),
    )
    return [
        Run(
            f"profile-{name}-50steps",
            config,
            {
                "train.max_tokens": 0,
                "train.max_steps": 50,
                "train.eval_interval": 0,
                "train.save_interval": 0,
            },
        )
        for name, config in configs
    ]


def _value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def command_for(
    run: Run, nproc: int, project: str, output_root: str
) -> list[str]:
    overrides = {
        **run.overrides,
        "train.output_dir": f"{output_root}/{run.name}",
        "train.wandb_project": project,
        "train.wandb_run_name": run.name,
    }
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={nproc}",
        "train.py",
        "--config",
        run.config,
    ]
    for key, value in overrides.items():
        command.extend(("--override", f"{key}={_value(value)}"))
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print or sequentially execute the pre-registered run matrix."
    )
    parser.add_argument(
        "--stage",
        choices=("pilot", "profiling", "screening", "scaling", "seeds"),
        required=True,
    )
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--wandb-project", default="kan-moe")
    parser.add_argument(
        "--output-root",
        default="/data/umoe_mod_share/kan_moe/outputs/revised",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run sequentially; without this flag commands are only printed.",
    )
    args = parser.parse_args()
    if not 1 <= args.nproc_per_node <= 4:
        raise ValueError("--nproc-per-node must be between 1 and 4")

    stages = {
        "pilot": _pilot_runs,
        "profiling": _profiling_runs,
        "screening": _screening_runs,
        "scaling": _scaling_runs,
        "seeds": _seed_runs,
    }
    for run in stages[args.stage]():
        command = command_for(
            run,
            nproc=args.nproc_per_node,
            project=args.wandb_project,
            output_root=args.output_root,
        )
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

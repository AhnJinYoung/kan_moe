from __future__ import annotations

import argparse
import math
import shlex
import subprocess
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Run:
    name: str
    config: str
    overrides: dict[str, object] = field(default_factory=dict)


def _safe_max_steps(tokens: int) -> int:
    """Leave 25% headroom for a one-GPU 64x2048-token optimizer batch."""
    return max(1_000, math.ceil(tokens / 131_072 * 1.25))


def _rho_slug(rho: float) -> str:
    return f"{rho:g}".replace("-", "m").replace(".", "p")


def _winner_overrides(
    distribution_k: int = 9,
    rho: float = 0.5,
    top_k: int = 2,
) -> dict[str, object]:
    return {
        "model.distribution_k": distribution_k,
        "model.aggregation": "power",
        "model.power_rho": rho,
        "model.learnable_rho": False,
        "model.top_k": top_k,
    }


def _screening_runs() -> list[Run]:
    """One-seed 500M-model screen; learned controls are deliberately excluded."""
    budget = {
        "train.max_tokens": 500_000_000,
        "train.max_steps": 5_000,
    }
    runs = [
        Run(
            "screen-500m-vanilla-k2",
            "configs/vanilla_moe_500m.yaml",
            budget,
        ),
        Run(
            "screen-500m-atoms9-rho0p5-k2",
            "configs/distributional_moe_500m.yaml",
            budget,
        ),
    ]
    for distribution_k in (5, 17):
        runs.append(
            Run(
                f"screen-500m-atoms{distribution_k}-rho0p5-k2",
                "configs/distributional_moe_500m.yaml",
                {
                    **budget,
                    "model.distribution_k": distribution_k,
                },
            )
        )
    for rho in (0.25, 0.75, 1.0):
        runs.append(
            Run(
                f"screen-500m-atoms9-rho{_rho_slug(rho)}-k2",
                "configs/distributional_moe_500m.yaml",
                {
                    **budget,
                    **_winner_overrides(rho=rho),
                },
            )
        )
    runs.append(
        Run(
            "screen-500m-atoms9-rho0p5-k4",
            "configs/distributional_moe_500m.yaml",
            {
                **budget,
                "model.top_k": 4,
            },
        )
    )
    return runs


def _confirmation_runs(
    distribution_k: int = 9,
    rho: float = 0.5,
    top_k: int = 2,
    tokens: int = 5_000_000_000,
    role: str = "both",
) -> list[Run]:
    if role not in {"both", "baseline", "candidate"}:
        raise ValueError("role must be both, baseline, or candidate")
    budget_slug = (
        f"{tokens // 1_000_000_000}b"
        if tokens % 1_000_000_000 == 0
        else f"{tokens // 1_000_000}m"
    )
    max_steps = _safe_max_steps(tokens)
    winner = f"atoms{distribution_k}-rho{_rho_slug(rho)}-k{top_k}"
    baseline = Run(
        f"confirm-500m-vanilla-k{top_k}-{budget_slug}",
        "configs/vanilla_moe_500m.yaml",
        {
            "model.top_k": top_k,
            "train.max_tokens": tokens,
            "train.max_steps": max_steps,
        },
    )
    candidate = Run(
        f"confirm-500m-{winner}-{budget_slug}",
        "configs/distributional_moe_500m.yaml",
        {
            **_winner_overrides(distribution_k, rho, top_k),
            "train.max_tokens": tokens,
            "train.max_steps": max_steps,
        },
    )
    if role == "baseline":
        return [baseline]
    if role == "candidate":
        return [candidate]
    return [baseline, candidate]


def _control_runs(
    distribution_k: int = 9,
    rho: float = 0.5,
    top_k: int = 2,
) -> list[Run]:
    """Run only after the 5B distributional-vs-vanilla gate is passed."""
    budget = {
        "model.top_k": top_k,
        "train.max_tokens": 5_000_000_000,
        "train.max_steps": 50_000,
    }
    return [
        Run(
            "control-500m-dense-5b",
            "configs/dense_500m.yaml",
            {
                "train.max_tokens": 5_000_000_000,
                "train.max_steps": 50_000,
            },
        ),
        Run(
            f"control-500m-output-gated-k{top_k}-5b",
            "configs/output_gated_moe_500m.yaml",
            budget,
        ),
        Run(
            f"control-500m-residual-mlp-k{top_k}-5b",
            "configs/residual_mlp_moe_500m.yaml",
            budget,
        ),
        Run(
            (
                "control-500m-learned-rho-"
                f"atoms{distribution_k}-init{_rho_slug(rho)}-k{top_k}-5b"
            ),
            "configs/distributional_moe_learned_rho_500m.yaml",
            {
                **budget,
                "model.distribution_k": distribution_k,
                "model.power_rho": rho,
            },
        ),
    ]


def _scaling_runs(
    distribution_k: int = 9,
    rho: float = 0.5,
    top_k: int = 2,
) -> list[Run]:
    """Late scale check; the 500M point is reused from confirmation."""
    scale_specs = (
        ("150m", 1_500_000_000, 15_000),
        ("1_5b", 15_000_000_000, 150_000),
    )
    runs: list[Run] = []
    for scale, tokens, max_steps in scale_specs:
        for model in ("vanilla_moe", "distributional_moe"):
            overrides: dict[str, object] = {
                "train.max_tokens": tokens,
                "train.max_steps": max_steps,
            }
            if model == "distributional_moe":
                overrides.update(
                    _winner_overrides(distribution_k, rho, top_k)
                )
            else:
                overrides["model.top_k"] = top_k
            runs.append(
                Run(
                    f"scale-{scale}-{model}-seed1337",
                    f"configs/{model}_{scale}.yaml",
                    overrides,
                )
            )
    return runs


def _seed_runs(
    distribution_k: int = 9,
    rho: float = 0.5,
    top_k: int = 2,
) -> list[Run]:
    """Last-stage replications; seed 1337 is reused from confirmation."""
    runs: list[Run] = []
    for seed in (2027, 4099):
        runs.append(
            Run(
                f"seed-500m-vanilla-k{top_k}-seed{seed}-5b",
                "configs/vanilla_moe_500m.yaml",
                {
                    "model.top_k": top_k,
                    "train.seed": seed,
                },
            )
        )
        runs.append(
            Run(
                (
                    f"seed-500m-atoms{distribution_k}-"
                    f"rho{_rho_slug(rho)}-k{top_k}-seed{seed}-5b"
                ),
                "configs/distributional_moe_500m.yaml",
                {
                    **_winner_overrides(distribution_k, rho, top_k),
                    "train.seed": seed,
                },
            )
        )
    return runs


def _pilot_runs() -> list[Run]:
    budget = {
        "train.max_tokens": 100_000_000,
        "train.max_steps": 2_000,
        "train.eval_interval": 200,
        "train.save_interval": 200,
    }
    return [
        Run("pilot-500m-vanilla-k2-100m", "configs/vanilla_moe_500m.yaml", budget),
        Run(
            "pilot-500m-atoms9-rho0p5-k2-100m",
            "configs/distributional_moe_500m.yaml",
            budget,
        ),
    ]


def _profiling_runs() -> list[Run]:
    configs = (
        ("500m-vanilla", "configs/vanilla_moe_500m.yaml"),
        ("500m-hellinger", "configs/distributional_moe_500m.yaml"),
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
        choices=(
            "profiling",
            "pilot",
            "screening",
            "confirmation",
            "controls",
            "scaling",
            "seeds",
        ),
        required=True,
    )
    parser.add_argument("--winner-distribution-k", type=int, default=9)
    parser.add_argument("--winner-rho", type=float, default=0.5)
    parser.add_argument("--winner-top-k", type=int, default=2)
    parser.add_argument(
        "--confirmation-tokens",
        type=int,
        default=5_000_000_000,
        help="Use 1B for finalist refinement, then 5B for the final confirmation.",
    )
    parser.add_argument(
        "--confirmation-role",
        choices=("both", "baseline", "candidate"),
        default="both",
        help="Reuse one top-k-matched baseline when refining multiple finalists.",
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
        "confirmation": lambda: _confirmation_runs(
            args.winner_distribution_k,
            args.winner_rho,
            args.winner_top_k,
            args.confirmation_tokens,
            args.confirmation_role,
        ),
        "controls": lambda: _control_runs(
            args.winner_distribution_k,
            args.winner_rho,
            args.winner_top_k,
        ),
        "scaling": lambda: _scaling_runs(
            args.winner_distribution_k,
            args.winner_rho,
            args.winner_top_k,
        ),
        "seeds": lambda: _seed_runs(
            args.winner_distribution_k,
            args.winner_rho,
            args.winner_top_k,
        ),
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

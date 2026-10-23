"""Entry point for the whole project.

  python main.py --ui                 # Launch the modern interactive Healthcare Web UI
  python main.py --stage all          # data check + full experiment suite (tables + plots)
  python main.py --stage data|centralized|federated|privacy|experiments
Settings come from config.json (if present) and can be overridden with flags."""
import argparse
import sys
from pathlib import Path

from src.config import ExperimentConfig
from src.pipeline import run_round1, run_round2, run_round3, run_round4, run_round5

STAGES = {"data": [run_round1], "centralized": [run_round2], "federated": [run_round3],
          "privacy": [run_round4], "experiments": [run_round5],
          "all": [run_round1, run_round5]}


def build_config(args) -> ExperimentConfig:
    if args.config:
        cfg = ExperimentConfig.from_json(args.config)
    elif Path("config.json").exists():
        cfg = ExperimentConfig.from_json("config.json")
    else:
        cfg = ExperimentConfig()
    overrides = {"data_path": args.data_path, "out_dir": args.out_dir, "seed": args.seed,
                 "test_size": args.test_size, "n_clients": args.clients, "rounds": args.rounds,
                 "local_epochs": args.local_epochs, "lr": args.lr, "batch_size": args.batch_size,
                 "clip_norm": args.clip_norm, "n_repeats": args.repeats}
    if args.noise_levels:
        overrides["noise_levels"] = [float(x) for x in args.noise_levels.split(",")]
    d = cfg.to_dict()
    d.update({k: v for k, v in overrides.items() if v is not None})
    if args.noise_levels and args.compare_noise is None and float(d["compare_noise"]) not in d["noise_levels"]:
        d["compare_noise"] = d["noise_levels"][len(d["noise_levels"]) // 2]
    if args.compare_noise is not None:
        d["compare_noise"] = args.compare_noise
    return ExperimentConfig.from_dict(d)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ui", action="store_true", help="Launch the interactive Green/White/Blue Web Dashboard")
    p.add_argument("--port", type=int, default=8000, help="Web Dashboard port (default: 8000)")
    p.add_argument("--no-browser", action="store_true", help="Do not auto-open browser when launching UI")
    p.add_argument("--stage", choices=list(STAGES), default="all")
    p.add_argument("--config", default=None)
    p.add_argument("--data-path"); p.add_argument("--out-dir")
    p.add_argument("--seed", type=int); p.add_argument("--test-size", type=float)
    p.add_argument("--clients", type=int); p.add_argument("--rounds", type=int)
    p.add_argument("--local-epochs", type=int); p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int); p.add_argument("--clip-norm", type=float)
    p.add_argument("--noise-levels", help="comma-separated, e.g. 0.05,0.1,0.5")
    p.add_argument("--compare-noise", type=float); p.add_argument("--repeats", type=int)
    args = p.parse_args()

    if args.ui:
        from app import run_server
        run_server(port=args.port, open_browser=not args.no_browser)
        return 0

    try:
        cfg = build_config(args)
        for stage in STAGES[args.stage]:
            print(stage(cfg), "\n")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

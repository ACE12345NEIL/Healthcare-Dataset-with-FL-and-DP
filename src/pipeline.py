"""Stage functions used by main.py (CLI) and gui.py. Each returns a text report."""
import io
from contextlib import redirect_stdout

import numpy as np

from src.centralized import run_centralized
from src.config import ExperimentConfig
from src.data_loader import inspect_data, load_dataset, print_report
from src.evaluation import METRICS_HEADER, format_metrics_row, save_json
from src.experiments import prepare, run_experiments, run_privacy_demo
from src.fedavg import run_fedavg
from src.preprocessing import ZERO_AS_MISSING, prepare_data


def run_round1(cfg: ExperimentConfig) -> str:
    cfg.validate()
    buf = io.StringIO()
    with redirect_stdout(buf):
        df = load_dataset(cfg.data_path)
        print("=== Round 1: Data quality report (raw) ===")
        print_report(inspect_data(df))
        print(f"\nZeros treated as missing in: {ZERO_AS_MISSING}")
        data = prepare_data(df, test_size=cfg.test_size, seed=cfg.seed)
        print(f"\n=== Split (seed={cfg.seed}) ===")
        print(f"Train: {data.X_train.shape}, positives: {data.y_train.mean():.3f}")
        print(f"Test:  {data.X_test.shape}, positives: {data.y_test.mean():.3f}")
        print(f"Train/test index overlap: {len(data.train_index.intersection(data.test_index))}")
        nans = int(np.isnan(data.X_train).sum() + np.isnan(data.X_test).sum())
        print(f"NaNs after preprocessing: {nans}")
    return buf.getvalue()


def run_round2(cfg: ExperimentConfig) -> str:
    cfg.validate()
    data = prepare_data(load_dataset(cfg.data_path), test_size=cfg.test_size, seed=cfg.seed)
    r = run_centralized(data, cfg.seed, cfg.rf_estimators, cfg.out_dir)
    return "\n".join([
        "=== Round 2: Centralized Random Forest ===",
        f"Train rows: {r['n_train']}   Test rows: {r['n_test']}   Train time: {r['train_time_sec']:.2f}s",
        "", METRICS_HEADER, format_metrics_row("RandomForest", r["metrics"]),
        f"\nConfusion matrix [[TN, FP], [FN, TP]]: {r['metrics']['confusion_matrix']}",
        f"\nSaved to {cfg.out_dir}/metrics/ and {cfg.out_dir}/plots/"])


def run_round3(cfg: ExperimentConfig) -> str:
    data, clients = prepare(cfg)
    lines = ["=== Round 3: FedAvg (no clipping, no noise) ===",
             f"Clients: {cfg.n_clients}  sizes: {[c.n_samples for c in clients]}",
             f"Rounds: {cfg.rounds}  local epochs: {cfg.local_epochs}  lr: {cfg.lr}  "
             f"batch: {cfg.batch_size}", "", METRICS_HEADER]
    for name, label in [("logreg", "FedAvg LogReg"), ("mlp", "FedAvg MLP")]:
        result, _ = run_fedavg(name, clients, data.X_test, data.y_test, cfg.rounds,
                               cfg.local_epochs, cfg.lr, cfg.batch_size, cfg.seed, cfg.hidden)
        save_json(result, f"{cfg.out_dir}/metrics/fedavg_{name}.json")
        lines.append(format_metrics_row(label, result["metrics"]))
        lines.append(f"    time {result['train_time_sec']:.2f}s, confusion "
                     f"{result['metrics']['confusion_matrix']}")
    lines.append(f"\nPer-round history saved to {cfg.out_dir}/metrics/fedavg_*.json")
    return "\n".join(lines)


def run_round4(cfg: ExperimentConfig) -> str:
    return run_privacy_demo(cfg)


def run_round5(cfg: ExperimentConfig) -> str:
    """Full reproducible experiment suite (covers Rounds 2-4) + tables + plots."""
    return run_experiments(cfg)

"""Rounds 4-5: privacy sweep, comparison tables and plots from one reproducible config."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.centralized import run_centralized
from src.clients import partition_data
from src.config import ExperimentConfig
from src.data_loader import load_dataset
from src.evaluation import METRIC_NAMES, save_json
from src.fedavg import run_fedavg
from src.preprocessing import prepare_data
from src.visualization import plot_metric_bars, plot_noise_sweep, plot_round_curves

MODELS = {"logreg": "FedAvg LogReg", "mlp": "FedAvg MLP"}
RF_LABEL = "Centralized RF"
DISCLAIMER = ("Simulated single-machine results. Noise is applied to clipped client updates "
              "(DP-inspired); NO formal differential-privacy guarantee or epsilon is claimed.")


def settings(cfg: ExperimentConfig) -> list:
    """(setting_name, clip_norm, noise_multiplier)"""
    return ([("control", None, 0.0), ("clip_only", cfg.clip_norm, 0.0)]
            + [("noise", cfg.clip_norm, s) for s in cfg.noise_levels])


def setting_label(setting: str, sigma: float) -> str:
    return {"control": "no clip/no noise", "clip_only": "clip only"}.get(setting, f"noise {sigma:g}")


def prepare(cfg: ExperimentConfig):
    cfg.validate()
    data = prepare_data(load_dataset(cfg.data_path), cfg.test_size, cfg.seed)
    clients = partition_data(data.X_train, data.y_train, cfg.n_clients, cfg.seed)
    return data, clients


def run_federated_settings(cfg: ExperimentConfig, data, clients) -> list:
    """Same split, same partitions, same hyper-parameters for every run; only the privacy
    setting differs. Repeats change the training/noise seed (cfg.seed + repeat)."""
    runs = []
    for model in MODELS:
        for setting, clip, sigma in settings(cfg):
            for rep in range(cfg.n_repeats):
                result, _ = run_fedavg(model, clients, data.X_test, data.y_test, cfg.rounds,
                                       cfg.local_epochs, cfg.lr, cfg.batch_size, cfg.seed + rep,
                                       cfg.hidden, clip_norm=clip, noise_multiplier=sigma)
                runs.append({"model": model, "setting": setting, "noise_multiplier": sigma,
                             "repeat": rep, "result": result})
    return runs


def _ms(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def summarize(runs: list) -> list:
    groups = {}
    for r in runs:
        groups.setdefault((r["model"], r["setting"], r["noise_multiplier"]), []).append(r["result"])
    rows = []
    for (model, setting, sigma), results in groups.items():
        row = {"model": model, "setting": setting, "label": setting_label(setting, sigma),
               "noise_multiplier": sigma, "clip_norm": results[0]["clip_norm"],
               "n_repeats": len(results), "n_clients": results[0]["n_clients"],
               "rounds": results[0]["communication_rounds"],
               "confusion_matrix_repeat0": results[0]["metrics"]["confusion_matrix"]}
        for m in METRIC_NAMES:
            row[f"{m}_mean"], row[f"{m}_std"] = _ms([r["metrics"][m] for r in results])
        row["train_time_sec_mean"] = float(np.mean([r["train_time_sec"] for r in results]))
        row["frac_clipped_mean"] = float(np.mean([h["frac_clipped"] for r in results for h in r["history"]]))
        row["_curves"] = {m: np.mean([[h[m] for h in r["history"]] for r in results], axis=0)
                          for m in ("accuracy", "roc_auc")}
        rows.append(row)
    return rows


def _fmt(row, m):
    mean, std = row[f"{m}_mean"], row[f"{m}_std"]
    return f"{mean:.3f} +/- {std:.3f}" if row.get("n_repeats", 1) > 1 else f"{mean:.3f}"


def md_table(headers: list, rows: list) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _metric_row(name, row):
    return [name] + [_fmt(row, m) for m in METRIC_NAMES] + [f"{row['train_time_sec_mean']:.2f}s"]


METRIC_HEADERS = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Train time"]


def format_sweep_table(rows: list) -> str:
    body = [[MODELS[r["model"]], r["label"], _fmt(r, "accuracy"), _fmt(r, "roc_auc"),
             f"{r['frac_clipped_mean']:.2f}"] for r in rows]
    return md_table(["Model", "Setting", "Accuracy", "ROC-AUC", "Frac. client updates clipped (all rounds)"], body)


def _rf_row(rf: dict) -> dict:
    row = {"train_time_sec_mean": rf["train_time_sec"], "label": RF_LABEL,
           "confusion_matrix_repeat0": rf["metrics"]["confusion_matrix"]}
    for m in METRIC_NAMES:
        row[f"{m}_mean"], row[f"{m}_std"] = rf["metrics"][m], 0.0
    return row


def run_experiments(cfg: ExperimentConfig) -> str:
    cfg.validate()
    out = cfg.out_dir
    save_json(cfg.to_dict(), f"{out}/metrics/experiment_config.json")
    data, clients = prepare(cfg)

    rf = run_centralized(data, cfg.seed, cfg.rf_estimators, out)
    runs = run_federated_settings(cfg, data, clients)
    rows = summarize(runs)
    by = {(r["model"], r["setting"], r["noise_multiplier"]): r for r in rows}
    ctrl = {m: by[(m, "control", 0.0)] for m in MODELS}
    rfr = _rf_row(rf)
    cmp_sigma = float(cfg.compare_noise)

    # ---- files ----
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
                 ).to_csv(f"{out}/metrics/experiment_summary.csv", index=False)
    save_json({"config": cfg.to_dict(), "centralized": rf, "federated_runs": runs},
              f"{out}/metrics/experiment_runs.json")

    # ---- tables ----
    t1 = md_table(METRIC_HEADERS, [_metric_row(RF_LABEL, rfr), _metric_row(MODELS["logreg"], ctrl["logreg"])])
    t2 = md_table(METRIC_HEADERS, [_metric_row(MODELS[m], ctrl[m]) for m in MODELS])
    t3_rows = []
    for m in MODELS:
        t3_rows += [_metric_row(f"{MODELS[m]} - no clip/no noise", ctrl[m]),
                    _metric_row(f"{MODELS[m]} - clip only", by[(m, "clip_only", 0.0)]),
                    _metric_row(f"{MODELS[m]} - noise {cmp_sigma:g}", by[(m, "noise", cmp_sigma)])]
    t3 = md_table(METRIC_HEADERS, t3_rows)
    t4 = format_sweep_table(rows)
    marks = sorted({1, 5, 10, 20, cfg.rounds} & set(range(1, cfg.rounds + 1)))
    t5_rows = []
    for m in MODELS:
        for s, sig in [("control", 0.0), ("noise", cmp_sigma)]:
            r = by[(m, s, sig)]
            t5_rows.append([MODELS[m], r["label"]] +
                           [f"{r['_curves']['accuracy'][k - 1]:.3f} / {r['_curves']['roc_auc'][k - 1]:.3f}"
                            for k in marks])
    t5 = md_table(["Model", "Setting"] + [f"round {k} (acc / AUC)" for k in marks], t5_rows)
    cm_items = [(RF_LABEL, rfr)] + [(f"{MODELS[m]} - no clip/no noise", ctrl[m]) for m in MODELS] + \
               [(f"{MODELS[m]} - noise {cmp_sigma:g}", by[(m, "noise", cmp_sigma)]) for m in MODELS]
    cms = "\n".join(f"- {name}: {r['confusion_matrix_repeat0']}" for name, r in cm_items)

    cfg_line = (f"seed={cfg.seed}, test_size={cfg.test_size}, clients={cfg.n_clients} "
                f"(sizes {clients[0].n_samples}-{clients[-1].n_samples}), rounds={cfg.rounds}, "
                f"local_epochs={cfg.local_epochs}, lr={cfg.lr}, batch={cfg.batch_size}, "
                f"clip_norm={cfg.clip_norm}, repeats={cfg.n_repeats}, noise std = noise level x clip_norm")
    report = "\n\n".join([
        "# Experiment results", f"> {DISCLAIMER}", f"**Settings:** {cfg_line}",
        "Hyper-parameters are fixed defaults and were NOT tuned on the test set. "
        "Mean +/- std over repeats (different training/noise seeds, same split and partitions); "
        "the centralized RF is a single deterministic run.",
        "## 1. Centralized Random Forest vs federated Logistic Regression", t1,
        "## 2. Federated Logistic Regression vs federated MLP", t2,
        f"## 3. Federated models with vs without noise (noise level {cmp_sigma:g})", t3,
        "## 4. Accuracy and ROC-AUC across noise levels", t4,
        "## 5. Performance across communication rounds (mean over repeats)", t5,
        "## Confusion matrices ([[TN, FP], [FN, TP]], first repeat)", cms,
        "## Plots", "See `results/plots/`: `rf_vs_logreg.png`, `logreg_vs_mlp.png`, "
        "`noise_vs_no_noise.png`, `noise_sweep.png`, `rounds_curves.png`, `confusion_matrix_rf.png`.",
    ])
    Path(out).mkdir(parents=True, exist_ok=True)
    Path(f"{out}/RESULTS.md").write_text(report, encoding="utf-8")

    # ---- plots ----
    def g(row):
        return {m: (row[f"{m}_mean"], row[f"{m}_std"]) for m in METRIC_NAMES}
    P = f"{out}/plots"
    plot_metric_bars({RF_LABEL: g(rfr), MODELS["logreg"]: g(ctrl["logreg"])}, METRIC_NAMES,
                     "Centralized RF vs FedAvg LogReg", f"{P}/rf_vs_logreg.png")
    plot_metric_bars({MODELS[m]: g(ctrl[m]) for m in MODELS}, METRIC_NAMES,
                     "FedAvg LogReg vs FedAvg MLP", f"{P}/logreg_vs_mlp.png")
    plot_metric_bars({f"{MODELS[m]} / {lab}": g(by[(m, s, sg)])
                      for m in MODELS for lab, s, sg in
                      [("no noise", "control", 0.0), (f"noise {cmp_sigma:g}", "noise", cmp_sigma)]},
                     METRIC_NAMES, "With vs without noise", f"{P}/noise_vs_no_noise.png")
    order = [("control", 0.0), ("clip_only", 0.0)] + [("noise", s) for s in cfg.noise_levels]
    series = {MODELS[m]: {k: ([by[(m, s, sg)][f"{k}_mean"] for s, sg in order],
                              [by[(m, s, sg)][f"{k}_std"] for s, sg in order])
                          for k in ("accuracy", "roc_auc")} for m in MODELS}
    plot_noise_sweep([setting_label(s, sg) for s, sg in order], series,
                     {RF_LABEL: {k: rfr[f"{k}_mean"] for k in ("accuracy", "roc_auc")}},
                     f"{P}/noise_sweep.png")
    plot_round_curves({MODELS[m]: {by[(m, s, sg)]["label"]: by[(m, s, sg)]["_curves"]
                                   for s, sg in order} for m in MODELS}, f"{P}/rounds_curves.png")

    return report + f"\n\nSaved: {out}/RESULTS.md, {out}/metrics/*, {out}/plots/*"


def run_privacy_demo(cfg: ExperimentConfig) -> str:
    """Round 4 view: control vs clip-only vs noise levels (no RF, no plots)."""
    data, clients = prepare(cfg)
    rows = summarize(run_federated_settings(cfg, data, clients))
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
                 ).to_csv(f"{cfg.out_dir}/metrics/privacy_sweep.csv", index=False)
    return "\n\n".join([
        "=== Round 4: clip + Gaussian noise on client updates ===", DISCLAIMER,
        f"clip_norm={cfg.clip_norm}; noise std per client = noise level x clip_norm; "
        f"repeats={cfg.n_repeats}", format_sweep_table(rows),
        f"Saved: {cfg.out_dir}/metrics/privacy_sweep.csv"])

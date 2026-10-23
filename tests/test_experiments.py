from pathlib import Path

import json

import pandas as pd
import pytest

from src.config import ExperimentConfig
from src.experiments import run_experiments
from src.pipeline import run_round1, run_round4
from tests.synth import make_df


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("exp")
    make_df(400).to_csv(tmp / "d.csv", index=False)
    return ExperimentConfig(data_path=str(tmp / "d.csv"), out_dir=str(tmp / "results"),
                            rounds=6, local_epochs=2, lr=0.1, rf_estimators=20, n_repeats=2,
                            noise_levels=[0.1, 1.0], compare_noise=0.1)


def test_full_experiment_outputs(cfg):
    report = run_experiments(cfg)
    out = cfg.out_dir
    for f in ["rf_vs_logreg", "logreg_vs_mlp", "noise_vs_no_noise", "noise_sweep",
              "rounds_curves", "confusion_matrix_rf"]:
        assert Path(f"{out}/plots/{f}.png").exists(), f
    s = pd.read_csv(f"{out}/metrics/experiment_summary.csv")
    # 2 models x (control + clip_only + 2 noise levels)
    assert len(s) == 8 and set(s["setting"]) == {"control", "clip_only", "noise"}
    assert (s["n_clients"] == cfg.n_clients).all() and (s["rounds"] == cfg.rounds).all()
    saved = json.loads(open(f"{out}/metrics/experiment_config.json").read())
    assert saved["seed"] == cfg.seed
    for h in ["Centralized Random Forest", "Logistic Regression", "MLP", "with vs without noise",
              "noise levels", "communication rounds"]:
        assert h.lower().split()[0] in report.lower()
    assert "NO formal differential-privacy" in report


def test_same_test_set_and_reproducible(cfg):
    runs = json.loads(open(f"{cfg.out_dir}/metrics/experiment_runs.json").read())["federated_runs"]
    r0 = [r for r in runs if r["model"] == "logreg" and r["setting"] == "control" and r["repeat"] == 0][0]
    before = r0["result"]["metrics"]["roc_auc"]
    run_experiments(cfg)
    runs = json.loads(open(f"{cfg.out_dir}/metrics/experiment_runs.json").read())["federated_runs"]
    r1 = [r for r in runs if r["model"] == "logreg" and r["setting"] == "control" and r["repeat"] == 0][0]
    assert r1["result"]["metrics"]["roc_auc"] == before


def test_privacy_stage_and_data_stage(cfg):
    assert "clip + Gaussian noise" in run_round4(cfg)
    assert "Train:" in run_round1(cfg)


def test_config_validation_and_roundtrip(tmp_path):
    with pytest.raises(ValueError):
        ExperimentConfig(n_clients=0).validate()
    with pytest.raises(ValueError):
        ExperimentConfig(noise_levels=[0.0, 1.0]).validate()
    with pytest.raises(ValueError):
        ExperimentConfig(compare_noise=7.0).validate()
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"bogus": 1})
    p = tmp_path / "c.json"
    p.write_text(json.dumps(ExperimentConfig(seed=7).to_dict()))
    assert ExperimentConfig.from_json(p).seed == 7
    with pytest.raises(FileNotFoundError):
        ExperimentConfig.from_json(tmp_path / "missing.json")

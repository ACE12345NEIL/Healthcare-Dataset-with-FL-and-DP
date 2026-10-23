import json

import numpy as np
import pandas as pd

from src.centralized import run_centralized
from src.evaluation import compute_metrics
from src.preprocessing import prepare_data
from tests.synth import make_df


def test_metrics_perfect_and_confusion_layout():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["accuracy"] == m["f1"] == m["roc_auc"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]


def test_confusion_matrix_order_tn_fp_fn_tp():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 1])
    m = compute_metrics(y_true, y_pred, np.array([.1, .6, .4, .7, .9]))
    assert m["confusion_matrix"] == [[1, 1], [1, 2]]


def test_centralized_outputs(tmp_path):
    data = prepare_data(make_df(300), seed=1)
    r = run_centralized(data, seed=1, n_estimators=30, out_dir=tmp_path)
    assert r["metrics"]["roc_auc"] > 0.6
    saved = json.loads((tmp_path / "metrics" / "centralized_rf.json").read_text())
    assert saved["n_test"] == len(data.y_test)
    preds = pd.read_csv(tmp_path / "metrics" / "centralized_rf_predictions.csv")
    assert len(preds) == len(data.y_test) and (preds["y_true"].values == data.y_test).all()
    assert (tmp_path / "plots" / "confusion_matrix_rf.png").exists()

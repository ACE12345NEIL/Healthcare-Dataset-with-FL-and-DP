"""Round 2: centralized Random Forest baseline on the full training partition."""
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.evaluation import compute_metrics, save_json
from src.preprocessing import PreparedData
from src.visualization import plot_confusion_matrix


def run_centralized(data: PreparedData, seed: int = 42, n_estimators: int = 200,
                    out_dir="results") -> dict:
    out_dir = Path(out_dir)
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)

    t0 = time.perf_counter()
    clf.fit(data.X_train, data.y_train)
    train_time = time.perf_counter() - t0

    y_score = clf.predict_proba(data.X_test)[:, 1]
    y_pred = clf.predict(data.X_test)

    result = {
        "model": "centralized_random_forest",
        "seed": seed,
        "n_estimators": n_estimators,
        "n_train": int(len(data.y_train)),
        "n_test": int(len(data.y_test)),
        "train_time_sec": train_time,
        "metrics": compute_metrics(data.y_test, y_pred, y_score),
    }

    save_json(result, out_dir / "metrics" / "centralized_rf.json")
    pd.DataFrame({"row_index": data.test_index, "y_true": data.y_test,
                  "y_pred": y_pred, "y_score": y_score}
                 ).to_csv(out_dir / "metrics" / "centralized_rf_predictions.csv", index=False)
    plot_confusion_matrix(result["metrics"]["confusion_matrix"],
                          "Random Forest (centralized)",
                          out_dir / "plots" / "confusion_matrix_rf.png")
    return result

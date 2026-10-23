"""Synthetic data with real signal, for tests only (not the real dataset)."""
import numpy as np
import pandas as pd


def make_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    glucose = rng.normal(120, 30, n).clip(50, 220)
    bmi = rng.normal(32, 6, n).clip(15, 55)
    logit = 0.04 * (glucose - 120) + 0.1 * (bmi - 32) - 0.6
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "Pregnancies": rng.integers(0, 10, n), "Glucose": glucose,
        "BloodPressure": rng.normal(70, 10, n).clip(40, 110),
        "SkinThickness": rng.normal(25, 8, n).clip(5, 60),
        "Insulin": rng.normal(100, 40, n).clip(10, 300), "BMI": bmi,
        "DiabetesPedigreeFunction": rng.uniform(0.1, 2, n),
        "Age": rng.integers(21, 80, n), "Outcome": y})

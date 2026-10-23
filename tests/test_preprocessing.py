import numpy as np
import pandas as pd
import pytest

from src.data_loader import TARGET_COLUMN, FEATURE_COLUMNS, validate_schema
from src.preprocessing import ZERO_AS_MISSING, prepare_data, replace_invalid_zeros


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 200
    data = pd.DataFrame({
        "Pregnancies": rng.integers(0, 10, n),
        "Glucose": rng.normal(120, 25, n).clip(50, 200),
        "BloodPressure": rng.normal(70, 10, n).clip(40, 110),
        "SkinThickness": rng.normal(25, 8, n).clip(5, 60),
        "Insulin": rng.normal(100, 40, n).clip(10, 300),
        "BMI": rng.normal(32, 6, n).clip(15, 55),
        "DiabetesPedigreeFunction": rng.uniform(0.1, 2, n),
        "Age": rng.integers(21, 80, n),
        TARGET_COLUMN: (rng.random(n) < 0.35).astype(int),
    })
    data.loc[:9, ZERO_AS_MISSING] = 0          # inject invalid zeros
    data.loc[:9, "Pregnancies"] = 0            # legitimate zeros
    return data


def test_invalid_zeros_only_in_medical_columns(df):
    cleaned = replace_invalid_zeros(df)
    assert cleaned[ZERO_AS_MISSING].isna().sum().sum() == 10 * len(ZERO_AS_MISSING)
    assert (cleaned["Pregnancies"] == 0).sum() == (df["Pregnancies"] == 0).sum()
    assert cleaned["Pregnancies"].isna().sum() == 0


def test_split_is_disjoint_and_complete(df):
    d = prepare_data(df, test_size=0.25, seed=1)
    assert len(d.train_index.intersection(d.test_index)) == 0
    assert len(d.train_index) + len(d.test_index) == len(df)
    assert len(d.X_test) == len(d.y_test) == 50


def test_split_is_stratified(df):
    d = prepare_data(df, seed=1)
    assert abs(d.y_train.mean() - d.y_test.mean()) < 0.03


def test_same_seed_reproducible_different_seed_differs(df):
    a, b = prepare_data(df, seed=7), prepare_data(df, seed=7)
    c = prepare_data(df, seed=8)
    assert a.train_index.equals(b.train_index)
    assert not a.train_index.equals(c.train_index)


def test_imputer_and_scaler_fit_on_train_only(df):
    d = prepare_data(df, seed=3)
    cleaned = replace_invalid_zeros(df)
    expected = cleaned.loc[d.train_index, FEATURE_COLUMNS].median().to_numpy()
    np.testing.assert_allclose(d.preprocessor.named_steps["imputer"].statistics_, expected)
    np.testing.assert_allclose(d.X_train.mean(axis=0), 0, atol=1e-8)
    assert not np.allclose(d.X_test.mean(axis=0), 0, atol=1e-8)


def test_no_nans_after_preprocessing(df):
    d = prepare_data(df)
    assert not np.isnan(d.X_train).any() and not np.isnan(d.X_test).any()


def test_schema_validation(df):
    validate_schema(df)
    with pytest.raises(ValueError):
        validate_schema(df.drop(columns=["Glucose"]))
    bad = df.copy()
    bad[TARGET_COLUMN] = 2
    with pytest.raises(ValueError):
        validate_schema(bad)

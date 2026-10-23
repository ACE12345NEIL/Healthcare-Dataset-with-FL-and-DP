"""Zero-to-NaN cleaning, stratified split, and train-only imputation/scaling.

Includes domain-informed clinical feature enrichment from the literature and
research paper (ICICC 2026, Section 3.2):
- Metabolic risk scores and insulin resistance proxies
- Pairwise clinical metric interactions (Glucose x BMI, Glucose x Age, etc.)
- Polynomial nonlinear features (Glucose^2, BMI^2, etc.)
- Clinical threshold indicator flags (Hyperglycemia, Hypertension, Obesity)
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import FEATURE_COLUMNS, TARGET_COLUMN

# Zero is biologically implausible in these columns -> treat as missing.
# Pregnancies and DiabetesPedigreeFunction can legitimately be 0 and are excluded.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def replace_invalid_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with implausible zeros replaced by NaN."""
    out = df.copy()
    out[ZERO_AS_MISSING] = out[ZERO_AS_MISSING].replace(0, np.nan)
    return out


def enrich_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich initial clinical biomarkers into 25+ features (Paper Section 3.2)."""
    df = df.copy()

    # Base clinical metrics
    glu = df["Glucose"]
    bp = df["BloodPressure"]
    bmi = df["BMI"]
    age = df["Age"]
    ins = df["Insulin"]
    preg = df["Pregnancies"]

    # 1. Domain Risk Scores
    df["MetabolicRiskScore"] = (glu / 100.0) + (bmi / 25.0) + (bp / 80.0)
    df["InsulinResistanceProxy"] = (glu * ins) / 405.0  # HOMA-IR clinical proxy
    df["AgeRiskFactor"] = age / 50.0
    df["VesselSeverityProxy"] = (bp * age) / 1000.0

    # 2. Pairwise Interaction Features
    df["Glucose_x_BMI"] = glu * bmi
    df["Glucose_x_Age"] = glu * age
    df["BMI_x_Age"] = bmi * age
    df["BloodPressure_x_BMI"] = bp * bmi
    df["Glucose_x_Insulin"] = glu * ins
    df["Pregnancies_x_Age"] = preg * age

    # 3. Polynomial Nonlinear Terms
    df["Glucose_sq"] = glu ** 2
    df["BMI_sq"] = bmi ** 2
    df["Age_sq"] = age ** 2
    df["BloodPressure_sq"] = bp ** 2

    # 4. Clinical Risk Threshold Flags (Categorical/Binary Encodings)
    df["HighGlucose_Flag"] = (glu >= 140.0).astype(float)       # Impaired glucose tolerance
    df["Hypertension_Flag"] = (bp >= 80.0).astype(float)        # Stage 1+ hypertension
    df["HighBMI_Flag"] = (bmi >= 30.0).astype(float)            # Clinical obesity
    df["HighInsulin_Flag"] = (ins >= 166.0).astype(float)       # Hyperinsulinemia
    df["CompositeRisk_Flag"] = (
        df["HighGlucose_Flag"] + df["Hypertension_Flag"] + df["HighBMI_Flag"]
    ) >= 2.0

    return df


def build_preprocessor() -> Pipeline:
    """Unfitted pipeline: median imputation then standardisation."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list
    preprocessor: Pipeline      # fitted on training data only
    train_index: pd.Index
    test_index: pd.Index


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42,
                 enrich: bool = False) -> PreparedData:
    """Clean -> stratified split -> fit preprocessing on train only -> transform both.
    If enrich=True, applies clinical feature enrichment (Paper Section 3.2).
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1).")

    cleaned = replace_invalid_zeros(df)
    if enrich:
        cleaned = enrich_clinical_features(cleaned)
        cols = [c for c in cleaned.columns if c != TARGET_COLUMN]
    else:
        cols = list(FEATURE_COLUMNS)

    X = cleaned[cols]
    y = cleaned[TARGET_COLUMN]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_tr)   # fit on train only
    X_test = preprocessor.transform(X_te)        # reuse train statistics

    return PreparedData(
        X_train=X_train, X_test=X_test,
        y_train=y_tr.to_numpy(dtype=int), y_test=y_te.to_numpy(dtype=int),
        feature_names=cols, preprocessor=preprocessor,
        train_index=X_tr.index, test_index=X_te.index,
    )

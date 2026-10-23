"""Loading, schema validation and quality inspection of the Pima dataset."""
from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET_COLUMN = "Outcome"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if the dataframe does not match the expected schema."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    non_numeric = [c for c in REQUIRED_COLUMNS
                   if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric columns: {non_numeric}")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(f"Target '{TARGET_COLUMN}' contains missing values.")
    if not set(df[TARGET_COLUMN].unique()) <= {0, 1}:
        raise ValueError(f"Target '{TARGET_COLUMN}' must be binary (0/1).")


def load_dataset(path) -> pd.DataFrame:
    """Read the CSV and validate its schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. See README.md for download instructions."
        )
    df = pd.read_csv(path)
    validate_schema(df)
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """Return a data-quality summary (types, class balance, NaNs, zeros)."""
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "class_counts": counts.to_dict(),
        "class_proportions": (counts / len(df)).round(4).to_dict(),
        "nan_counts": df.isna().sum().to_dict(),
        "zero_counts": (df[FEATURE_COLUMNS] == 0).sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def print_report(report: dict) -> None:
    print(f"Shape: {report['shape']}")
    print(f"Duplicate rows: {report['duplicate_rows']}")
    print(f"Class counts: {report['class_counts']}")
    print(f"Class proportions: {report['class_proportions']}")
    print("\nColumn                   dtype      NaNs  Zeros")
    for col, dtype in report["dtypes"].items():
        nan = report["nan_counts"].get(col, 0)
        zero = report["zero_counts"].get(col, "-")
        print(f"{col:<25}{dtype:<10}{nan:>5}{zero!s:>7}")

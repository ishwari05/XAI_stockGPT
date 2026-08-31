"""
DATA + MODEL UNDERSTANDING PIPELINE
====================================

Purpose:
    Understand:
    1. Dataset size and composition
    2. Ticker distribution
    3. Temporal distribution
    4. Target distribution
    5. Feature distributions
    6. Feature correlations
    7. Model predictions
    8. Confusion matrix
    9. Performance by ticker
    10. Performance by year
    11. Performance by volatility regime
    12. SHAP feature importance

Usage:
    python scripts/analyze_dataset.py

Expected input:
    data/features/train_features.parquet
    data/features/validation_features.parquet
    data/features/test_features.parquet

    data/predictions/   # if available
    data/explanations/  # if available

Output:
    reports/data_understanding/
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent

TRAIN_PATH = ROOT / "data/features/train.parquet"
VAL_PATH = ROOT / "data/features/validation.parquet"
TEST_PATH = ROOT / "data/features/test.parquet"

OUTPUT_DIR = ROOT / "reports/data_understanding"
FIG_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def save_plot(filename):
    path = FIG_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("DATA UNDERSTANDING PIPELINE")
print("=" * 70)

datasets = {}

for name, path in [
    ("train", TRAIN_PATH),
    ("validation", VAL_PATH),
    ("test", TEST_PATH),
]:
    if path.exists():
        datasets[name] = pd.read_parquet(path)
        print(
            f"{name.upper():12} : "
            f"{datasets[name].shape[0]:,} rows × "
            f"{datasets[name].shape[1]} columns"
        )
    else:
        print(f"WARNING: {path} not found")


if not datasets:
    raise FileNotFoundError(
        "No feature datasets found. "
        "Run the feature pipeline first."
    )


# Combined dataset
df = pd.concat(
    datasets.values(),
    ignore_index=True
)

print(f"\nTOTAL ROWS: {len(df):,}")


# ============================================================
# 2. BASIC DATASET PROFILE
# ============================================================

print("\n" + "=" * 70)
print("1. DATASET PROFILE")
print("=" * 70)

profile = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(df[c].dtype) for c in df.columns],
    "missing": [df[c].isna().sum() for c in df.columns],
    "missing_pct": [
        df[c].isna().mean() * 100
        for c in df.columns
    ],
    "unique_values": [
        df[c].nunique()
        for c in df.columns
    ],
})

profile.to_csv(
    OUTPUT_DIR / "dataset_profile.csv",
    index=False
)

print(profile.to_string(index=False))


# ============================================================
# 3. IDENTIFY IMPORTANT COLUMNS
# ============================================================

ticker_col = find_column(
    df,
    ["ticker", "Ticker", "symbol", "Symbol"]
)

date_col = find_column(
    df,
    ["Date", "date", "datetime", "Datetime"]
)

target_col = find_column(
    df,
    [
        "target_classification",
        "target",
        "target_class",
        "label",
        "class"
    ]
)

volatility_col = find_column(
    df,
    [
        "volatility_20d",
        "volatility_10d",
        "atr_ratio"
    ]
)

print("\nDetected columns:")
print("Ticker     :", ticker_col)
print("Date       :", date_col)
print("Target     :", target_col)
print("Volatility :", volatility_col)


# ============================================================
# 4. TICKER ANALYSIS
# ============================================================

if ticker_col:

    print("\n" + "=" * 70)
    print("2. TICKER DISTRIBUTION")
    print("=" * 70)

    ticker_stats = (
        df.groupby(ticker_col)
        .size()
        .reset_index(name="observations")
        .sort_values("observations", ascending=False)
    )

    ticker_stats["percentage"] = (
        ticker_stats["observations"]
        / len(df)
        * 100
    )

    ticker_stats.to_csv(
        OUTPUT_DIR / "ticker_distribution.csv",
        index=False
    )

    print(
        f"Unique tickers: "
        f"{df[ticker_col].nunique():,}"
    )

    print("\nTop 20 tickers:")
    print(ticker_stats.head(20).to_string(index=False))

    plt.figure(figsize=(12, 6))

    ticker_stats.head(30).plot(
        x=ticker_col,
        y="observations",
        kind="bar",
        legend=False
    )

    plt.title("Observations per Ticker — Top 30")
    plt.xlabel("Ticker")
    plt.ylabel("Number of Observations")
    plt.xticks(rotation=90)

    save_plot("observations_per_ticker.png")


# ============================================================
# 5. TEMPORAL ANALYSIS
# ============================================================

if date_col:

    print("\n" + "=" * 70)
    print("3. TEMPORAL DISTRIBUTION")
    print("=" * 70)

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    yearly = (
        df.dropna(subset=[date_col])
        .groupby(df[date_col].dt.year)
        .size()
        .reset_index(name="observations")
    )

    yearly.columns = ["year", "observations"]

    yearly.to_csv(
        OUTPUT_DIR / "observations_by_year.csv",
        index=False
    )

    print(yearly.to_string(index=False))

    plt.figure(figsize=(10, 5))

    plt.bar(
        yearly["year"].astype(str),
        yearly["observations"]
    )

    plt.title("Observations by Year")
    plt.xlabel("Year")
    plt.ylabel("Observations")

    save_plot("observations_by_year.png")


# ============================================================
# 6. TARGET DISTRIBUTION
# ============================================================

if target_col:

    print("\n" + "=" * 70)
    print("4. TARGET DISTRIBUTION")
    print("=" * 70)

    target_counts = (
        df[target_col]
        .value_counts(dropna=False)
        .reset_index()
    )

    target_counts.columns = [
        "target",
        "count"
    ]

    target_counts["percentage"] = (
        target_counts["count"]
        / len(df)
        * 100
    )

    target_counts.to_csv(
        OUTPUT_DIR / "target_distribution.csv",
        index=False
    )

    print(target_counts.to_string(index=False))

    plt.figure(figsize=(8, 5))

    plt.bar(
        target_counts["target"].astype(str),
        target_counts["count"]
    )

    plt.title("BUY / HOLD / SELL Distribution")
    plt.xlabel("Target Class")
    plt.ylabel("Number of Observations")

    save_plot("target_distribution.png")


# ============================================================
# 7. TARGET BY YEAR
# ============================================================

if target_col and date_col:

    target_year = pd.crosstab(
        df[date_col].dt.year,
        df[target_col],
        normalize="index"
    ) * 100

    target_year.to_csv(
        OUTPUT_DIR / "target_distribution_by_year.csv"
    )

    target_year.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6)
    )

    plt.title("Target Distribution by Year")
    plt.xlabel("Year")
    plt.ylabel("Percentage")

    save_plot("target_distribution_by_year.png")


# ============================================================
# 8. NUMERIC FEATURE ANALYSIS
# ============================================================

numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

# Remove obvious target columns
feature_cols = [
    c for c in numeric_cols
    if not c.startswith("target")
    and not c.startswith("fwd_")
]

print("\n" + "=" * 70)
print("5. NUMERIC FEATURE SUMMARY")
print("=" * 70)

feature_summary = df[feature_cols].describe().T

feature_summary["missing_pct"] = (
    df[feature_cols]
    .isna()
    .mean()
    * 100
)

feature_summary.to_csv(
    OUTPUT_DIR / "feature_summary.csv"
)

print(feature_summary.to_string())


# ============================================================
# 9. FEATURE CORRELATION
# ============================================================

if len(feature_cols) > 1:

    corr = df[feature_cols].corr()

    corr.to_csv(
        OUTPUT_DIR / "feature_correlation.csv"
    )

    plt.figure(figsize=(14, 12))

    plt.imshow(
        corr,
        aspect="auto"
    )

    plt.colorbar()

    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=90,
        fontsize=7
    )

    plt.yticks(
        range(len(corr.columns)),
        corr.columns,
        fontsize=7
    )

    plt.title("Feature Correlation Matrix")

    save_plot("feature_correlation.png")


# ============================================================
# 10. FEATURE DISTRIBUTIONS
# ============================================================

# Pick first 12 useful features
plot_features = feature_cols[:12]

for feature in plot_features:

    values = df[feature].dropna()

    if len(values) == 0:
        continue

    plt.figure(figsize=(8, 5))

    plt.hist(
        values,
        bins=50
    )

    plt.title(f"Distribution: {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    save_plot(
        f"feature_{feature}.png"
    )


# ============================================================
# 11. VOLATILITY REGIMES
# ============================================================

if volatility_col:

    print("\n" + "=" * 70)
    print("6. VOLATILITY REGIMES")
    print("=" * 70)

    df["volatility_regime"] = pd.qcut(
        df[volatility_col],
        q=3,
        labels=[
            "Low",
            "Medium",
            "High"
        ],
        duplicates="drop"
    )

    regime_counts = (
        df["volatility_regime"]
        .value_counts()
        .sort_index()
    )

    print(regime_counts)

    regime_counts.to_csv(
        OUTPUT_DIR / "volatility_regimes.csv"
    )


# ============================================================
# 12. FIND PREDICTION FILE
# ============================================================

prediction_candidates = list(
    (ROOT / "data/predictions").glob("*.csv")
)

print("\nPrediction files found:")

for p in prediction_candidates:
    print(" ", p.name)


# ============================================================
# 13. MODEL EVALUATION
# ============================================================

prediction_df = None

for path in prediction_candidates:

    try:
        candidate = pd.read_csv(path)

        pred_col = find_column(
            candidate,
            [
                "prediction",
                "predicted",
                "predicted_class",
                "y_pred"
            ]
        )

        actual_col = find_column(
            candidate,
            [
                "actual",
                "target",
                "y_true",
                "true_label"
            ]
        )

        if pred_col and actual_col:

            prediction_df = candidate.copy()

            print(
                f"\nUsing prediction file: "
                f"{path.name}"
            )

            break

    except Exception:
        continue


if prediction_df is not None:

    pred_col = find_column(
        prediction_df,
        [
            "prediction",
            "predicted",
            "predicted_class",
            "y_pred"
        ]
    )

    actual_col = find_column(
        prediction_df,
        [
            "actual",
            "target",
            "y_true",
            "true_label"
        ]
    )

    y_true = prediction_df[actual_col]
    y_pred = prediction_df[pred_col]

    print("\n" + "=" * 70)
    print("7. MODEL PERFORMANCE")
    print("=" * 70)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    balanced_acc = balanced_accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Balanced Accuracy : {balanced_acc:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")

    metrics = pd.DataFrame({
        "metric": [
            "accuracy",
            "balanced_accuracy",
            "macro_f1"
        ],
        "value": [
            accuracy,
            balanced_acc,
            macro_f1
        ]
    })

    metrics.to_csv(
        OUTPUT_DIR / "model_metrics.csv",
        index=False
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    cm_df = pd.DataFrame(
        cm
    )

    cm_df.to_csv(
        OUTPUT_DIR / "confusion_matrix.csv",
        index=False
    )

    plt.figure(figsize=(7, 6))

    plt.imshow(cm)

    plt.colorbar()

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    save_plot("confusion_matrix.png")


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report).T

    report_df.to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )

    print("\nClassification report:")
    print(report_df)


    # --------------------------------------------------------
    # CONFIDENCE IF AVAILABLE
    # --------------------------------------------------------

    confidence_col = find_column(
        prediction_df,
        [
            "confidence",
            "max_probability",
            "prediction_confidence"
        ]
    )

    if confidence_col:

        correct = (
            prediction_df[pred_col]
            == prediction_df[actual_col]
        )

        prediction_df["correct"] = correct

        plt.figure(figsize=(9, 5))

        plt.hist(
            prediction_df.loc[
                correct,
                confidence_col
            ].dropna(),
            bins=30,
            alpha=0.7,
            label="Correct"
        )

        plt.hist(
            prediction_df.loc[
                ~correct,
                confidence_col
            ].dropna(),
            bins=30,
            alpha=0.7,
            label="Incorrect"
        )

        plt.xlabel("Prediction Confidence")
        plt.ylabel("Frequency")
        plt.title(
            "Confidence: Correct vs Incorrect Predictions"
        )

        plt.legend()

        save_plot(
            "confidence_correct_vs_incorrect.png"
        )


# ============================================================
# 14. SHAP ANALYSIS
# ============================================================

shap_dir = ROOT / "data/explanations"

shap_files = list(shap_dir.glob("*.json"))

print("\n" + "=" * 70)
print("8. SHAP ARTIFACTS")
print("=" * 70)

for path in shap_files:
    print(path.name)


# Try global feature importance
global_shap_candidates = [
    shap_dir / "global_feature_importance.json",
    ROOT / "outputs/global_feature_importance.json"
]

global_shap = None

for path in global_shap_candidates:

    if path.exists():

        try:
            with open(path) as f:
                global_shap = json.load(f)

            print(
                f"\nLoaded SHAP importance: "
                f"{path}"
            )

            break

        except Exception as e:
            print("Could not load:", e)


if global_shap:

    # Handle common JSON structures
    if isinstance(global_shap, dict):

        # Case:
        # {"feature": importance}
        if all(
            isinstance(v, (int, float))
            for v in global_shap.values()
        ):

            shap_df = pd.DataFrame(
                list(global_shap.items()),
                columns=[
                    "feature",
                    "importance"
                ]
            )

        # Case:
        # {"features": [...], "importance": [...]}
        elif (
            "features" in global_shap
            and "importance" in global_shap
        ):

            shap_df = pd.DataFrame({
                "feature": global_shap["features"],
                "importance": global_shap["importance"]
            })

        else:
            shap_df = None

    else:
        shap_df = None

    if shap_df is not None:

        shap_df = shap_df.sort_values(
            "importance",
            ascending=False
        )

        shap_df.to_csv(
            OUTPUT_DIR / "shap_global_importance.csv",
            index=False
        )

        print("\nTop SHAP features:")
        print(
            shap_df.head(20)
            .to_string(index=False)
        )

        top = shap_df.head(15)

        plt.figure(figsize=(10, 7))

        plt.barh(
            top["feature"][::-1],
            top["importance"][::-1]
        )

        plt.xlabel("Mean |SHAP value|")
        plt.ylabel("Feature")
        plt.title(
            "Top SHAP Features"
        )

        save_plot(
            "shap_global_importance.png"
        )


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

summary = {
    "total_rows": int(len(df)),
    "total_columns": int(len(df.columns)),
    "unique_tickers": (
        int(df[ticker_col].nunique())
        if ticker_col
        else None
    ),
    "date_start": (
        str(df[date_col].min())
        if date_col
        else None
    ),
    "date_end": (
        str(df[date_col].max())
        if date_col
        else None
    ),
    "target_column": target_col,
    "feature_count": len(feature_cols),
}

if prediction_df is not None:

    summary["model_accuracy"] = float(
        accuracy
    )

    summary["balanced_accuracy"] = float(
        balanced_acc
    )

    summary["macro_f1"] = float(
        macro_f1
    )


with open(
    OUTPUT_DIR / "analysis_summary.json",
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nAll outputs saved to:\n"
    f"{OUTPUT_DIR}"
)

print("\nImportant files:")

for path in sorted(OUTPUT_DIR.glob("*")):
    print(" ", path.name)
"""Data-quality audit and the check of data origin (real vs synthetic).

Topic 31 requires verifying, before any modelling, whether the data look real
or synthetic: uniform marginal distributions and the absence of natural
correlations (age-hypertension, BMI-diabetes, ...) point to generated data.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from src import config


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Number and share of missing values per column."""
    miss = df.isna().sum()
    return pd.DataFrame({"missing": miss,
                         "missing_pct": (100 * miss / len(df)).round(2)})


def range_violations(df: pd.DataFrame) -> dict:
    """Count values outside the plausible ranges defined in config."""
    result = {}
    for col, (low, high) in config.VALID_RANGES.items():
        if col not in df.columns:
            continue
        bad = ((df[col] < low) | (df[col] > high)).sum()
        result[col] = int(bad)
    return result


def iqr_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Tukey (1.5*IQR) outliers for every numeric feature."""
    rows = []
    for col in config.NUMERIC_FEATURES:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < low) | (df[col] > high)).sum())
        rows.append({"feature": col, "q1": round(q1, 2), "q3": round(q3, 2),
                     "lower_fence": round(low, 2), "upper_fence": round(high, 2),
                     "outliers": n_out})
    return pd.DataFrame(rows)


def uniformity_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Kolmogorov-Smirnov test of each continuous feature against U(min, max).

    Real clinical variables (BMI, age, lab scores) are bell-shaped or skewed;
    a high p-value means the sample is indistinguishable from a uniform one.
    """
    rows = []
    for col in config.NUMERIC_FEATURES:
        x = df[col].astype("float64").to_numpy()
        lo, hi = x.min(), x.max()
        if col == "Age":                         # integer: chi-square on counts
            counts = pd.Series(x.astype(int)).value_counts().sort_index()
            stat, p = stats.chisquare(counts)
            test = "chi-square vs discrete uniform"
        else:
            stat, p = stats.kstest(x, "uniform", args=(lo, hi - lo))
            test = "KS vs continuous uniform"
        rows.append({"feature": col, "min": round(lo, 2), "max": round(hi, 2),
                     "skewness": round(stats.skew(x), 3),
                     "kurtosis": round(stats.kurtosis(x), 3),
                     "test": test, "statistic": round(stat, 4),
                     "p_value": round(p, 4), "looks_uniform": bool(p > 0.05)})
    return pd.DataFrame(rows)


# Pairs that are strongly correlated in real populations
EXPECTED_PAIRS = [("Age", "Hypertension"), ("Age", "Diabetes"),
                  ("BMI", "Diabetes"), ("BMI", "Hypertension"),
                  ("BMI", "PhysicalActivity"), ("Diabetes", "Hypertension"),
                  ("AlcoholConsumption", "Smoking")]


def natural_correlation_check(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of feature pairs that must be related in reality."""
    rows = []
    for a, b in EXPECTED_PAIRS:
        r, p = stats.spearmanr(df[a], df[b])
        rows.append({"pair": f"{a} ~ {b}", "spearman_r": round(r, 3),
                     "p_value": round(p, 4), "significant": bool(p < 0.05)})
    return pd.DataFrame(rows)


def decimal_precision(df: pd.DataFrame) -> dict:
    """Median number of decimals in float columns (measured values have 1-2)."""
    out = {}
    raw = pd.read_csv(config.RAW_CSV, dtype=str)
    for col in ["BMI", "AlcoholConsumption", "PhysicalActivity", "LiverFunctionTest"]:
        decimals = raw[col].str.split(".").str[1].fillna("").str.len()
        out[col] = int(decimals.median())
    return out


def run_audit(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Run the full audit, save tables and return a summary dictionary."""
    config.ensure_dirs()
    features = df[config.ALL_FEATURES + [config.TARGET]]

    miss = missing_report(features)
    outliers = iqr_outliers(features)
    uniform = uniformity_tests(features)
    corr = natural_correlation_check(features)
    feature_corr = features[config.ALL_FEATURES].corr().abs()
    np.fill_diagonal(feature_corr.values, 0)

    n_dup = int(features.duplicated().sum())
    n_dup_x = int(features[config.ALL_FEATURES].duplicated().sum())
    share_uniform = uniform["looks_uniform"].mean()
    share_signif = corr["significant"].mean()

    verdict = ("SYNTHETIC (very likely)"
               if share_uniform >= 0.6 and share_signif <= 0.3
               else "REAL or mixed")

    summary = {
        "rows": int(len(df)),
        "total_missing": int(miss["missing"].sum()),
        "duplicate_rows": n_dup,
        "duplicate_feature_vectors": n_dup_x,
        "range_violations": range_violations(features),
        "iqr_outliers_total": int(outliers["outliers"].sum()),
        "target_balance": {str(k): int(v) for k, v in
                           features[config.TARGET].value_counts().sort_index().items()},
        "max_abs_feature_correlation": round(float(feature_corr.values.max()), 3),
        "uniform_features": uniform.loc[uniform["looks_uniform"], "feature"].tolist(),
        "significant_natural_pairs": int(corr["significant"].sum()),
        "checked_natural_pairs": int(len(corr)),
        "median_decimals": decimal_precision(df),
        "origin_verdict": verdict,
    }

    miss.to_csv(config.TABLE_DIR / "audit_missing.csv")
    outliers.to_csv(config.TABLE_DIR / "audit_iqr_outliers.csv", index=False)
    uniform.to_csv(config.TABLE_DIR / "origin_uniformity_tests.csv", index=False)
    corr.to_csv(config.TABLE_DIR / "origin_natural_correlations.csv", index=False)
    with open(config.TABLE_DIR / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print("Missing values:", summary["total_missing"])
        print("Duplicate rows:", n_dup)
        print("Range violations:", {k: v for k, v in summary["range_violations"].items() if v})
        print("IQR outliers:\n", outliers[["feature", "outliers"]].to_string(index=False))
        print("\nUniformity tests:\n",
              uniform[["feature", "skewness", "p_value", "looks_uniform"]].to_string(index=False))
        print("\nNatural correlations:\n", corr.to_string(index=False))
        print("\nMedian decimals in float columns:", summary["median_decimals"])
        print("\n>>> Data origin verdict:", verdict)
    return summary


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning rules justified by the audit.

    * rows with a missing target are dropped (cannot be used for training);
    * exact duplicate rows are dropped (would leak between train and test);
    * values outside the plausible range are set to NaN and later imputed
      inside the sklearn Pipeline (median), so no information from the test
      set is used. IQR outliers are kept: in uniform synthetic data they are
      not errors, and tree models are robust to them.
    """
    out = df.dropna(subset=[config.TARGET])
    out = out.drop_duplicates(subset=config.ALL_FEATURES + [config.TARGET])
    for col, (low, high) in config.VALID_RANGES.items():
        if col in out.columns and col != config.TARGET:
            mask = (out[col] < low) | (out[col] > high)
            if mask.any():
                out[col] = out[col].astype("float32")
                out.loc[mask, col] = np.nan
    return out.reset_index(drop=True)

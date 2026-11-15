"""Model training and comparison.

Baseline DummyClassifier vs Logistic Regression vs Random Forest (required by
topic 31) + HistGradientBoosting (extension). All preprocessing lives inside a
scikit-learn Pipeline, so it is fitted on training folds only (no leakage).
Hyper-parameters are tuned with stratified 5-fold CV on the training set; the
hold-out test set (20 %, stratified) is used once for the final comparison.
"""
import io
import json
import time
import tracemalloc
from contextlib import closing
from datetime import datetime

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import seaborn as sns               # noqa: E402

from src import config
from src.database import get_connection

MODEL_COLORS = {"Dummy (baseline)": "#9e9e9e", "Logistic Regression": "#4C9BE8",
                "Random Forest": "#3BA272", "HistGradientBoosting": "#E8684C"}


def make_preprocessor() -> ColumnTransformer:
    """Imputation + scaling for numeric, one-hot for genetic risk."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    binary = SimpleImputer(strategy="most_frequent")
    ordinal = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, config.NUMERIC_FEATURES),
                              ("bin", binary, config.BINARY_FEATURES),
                              ("ord", ordinal, config.ORDINAL_FEATURES)])


def model_specs() -> dict:
    """Estimators and their hyper-parameter grids."""
    rs = config.RANDOM_STATE
    return {
        "Dummy (baseline)": (DummyClassifier(strategy="prior"), {}),
        "Logistic Regression": (
            LogisticRegression(max_iter=2000),
            {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0], "clf__class_weight": [None, "balanced"]}),
        "Random Forest": (
            RandomForestClassifier(random_state=rs, n_jobs=-1),
            {"clf__n_estimators": [300], "clf__max_depth": [None, 6, 10],
             "clf__min_samples_leaf": [1, 5, 10], "clf__max_features": ["sqrt", 0.5]}),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(random_state=rs),
            {"clf__learning_rate": [0.03, 0.1], "clf__max_depth": [3, None],
             "clf__max_leaf_nodes": [15, 31], "clf__l2_regularization": [0.0, 1.0]}),
    }


def split_data(df: pd.DataFrame):
    """Stratified hold-out split + leakage check (identical rows across sets)."""
    X = df[config.ALL_FEATURES].astype("float64")
    y = df[config.TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
    overlap = pd.merge(X_train.round(6), X_test.round(6), how="inner",
                       on=config.ALL_FEATURES)
    leakage = {"train_rows": len(X_train), "test_rows": len(X_test),
               "identical_rows_in_both": int(len(overlap)),
               "train_positive_rate": round(float(y_train.mean()), 4),
               "test_positive_rate": round(float(y_test.mean()), 4)}
    return X_train, X_test, y_train, y_test, leakage


def model_size_kb(model) -> float:
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return buffer.getbuffer().nbytes / 1024


def evaluate(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    big = pd.concat([X_test] * int(np.ceil(1000 / len(X_test))))[:1000]
    t0 = time.perf_counter()
    model.predict_proba(big)
    latency = (time.perf_counter() - t0) * 1000
    return {"roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
            "f1": f1_score(y_test, pred, zero_division=0),
            "accuracy": accuracy_score(y_test, pred),
            "precision_score": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "brier": brier_score_loss(y_test, proba),
            "predict_ms_per_1000": latency,
            "proba": proba, "pred": pred}


def train_models(X_train, y_train, verbose: bool = True) -> dict:
    """Grid search with stratified 5-fold CV for every model."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    fitted = {}
    for name, (estimator, grid) in model_specs().items():
        pipe = Pipeline([("prep", make_preprocessor()), ("clf", estimator)])
        search = GridSearchCV(pipe, grid or {"clf__strategy": ["prior"]},
                              scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
        tracemalloc.start()
        t0 = time.perf_counter()
        search.fit(X_train, y_train)
        search_time = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # time of a single fit with the best parameters (comparable across models)
        best = search.best_estimator_
        t0 = time.perf_counter()
        best.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        i = search.best_index_
        fitted[name] = {
            "model": best, "best_params": search.best_params_,
            "cv_roc_auc_mean": search.cv_results_["mean_test_score"][i],
            "cv_roc_auc_std": search.cv_results_["std_test_score"][i],
            "grid_size": len(search.cv_results_["params"]),
            "search_time_s": search_time, "fit_time_s": fit_time,
            "peak_memory_mb": peak / 1024 ** 2}
        if verbose:
            print(f"{name:<22} CV ROC-AUC = {fitted[name]['cv_roc_auc_mean']:.4f} "
                  f"+/- {fitted[name]['cv_roc_auc_std']:.4f}  "
                  f"({fitted[name]['grid_size']} configs, {search_time:.1f}s)  "
                  f"best: {search.best_params_}")
    return fitted


# ------------------------------------------------------------------ plots
def plot_roc_pr(results: dict, y_test) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["proba"])
        ax1.plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})",
                 color=MODEL_COLORS[name], lw=2)
        prec, rec, _ = precision_recall_curve(y_test, r["proba"])
        ax2.plot(rec, prec, label=f"{name} (AP={r['pr_auc']:.3f})",
                 color=MODEL_COLORS[name], lw=2)
    ax1.plot([0, 1], [0, 1], ls="--", color="lightgrey")
    ax1.set(xlabel="False positive rate", ylabel="True positive rate",
            title="ROC curves (hold-out test set)")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curves")
    ax1.legend(loc="lower right")
    ax2.legend(loc="lower left")
    path = config.FIG_DIR / "09_roc_pr_curves.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_calibration(results: dict, y_test) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], ls="--", color="lightgrey", label="Perfect calibration")
    for name, r in results.items():
        if name.startswith("Dummy"):
            continue
        frac, mean_pred = calibration_curve(y_test, r["proba"], n_bins=10,
                                            strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", color=MODEL_COLORS[name],
                label=f"{name} (Brier={r['brier']:.3f})")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction of disease",
           title="Calibration curves (reliability diagram)")
    ax.legend(loc="upper left", fontsize=9)
    path = config.FIG_DIR / "10_calibration_curves.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_confusion_matrices(results: dict, y_test) -> str:
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(4.3 * len(names), 4))
    for ax, name in zip(axes, names):
        cm = confusion_matrix(y_test, results[name]["pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        ax.set(title=name, xlabel="Predicted", ylabel="Actual")
    fig.suptitle("Confusion matrices on the test set (threshold 0.5)")
    path = config.FIG_DIR / "11_confusion_matrices.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def importance_analysis(fitted: dict, X_test, y_test) -> pd.DataFrame:
    """Permutation importance (ROC-AUC drop) for LR, RF and HGB."""
    frames = []
    for name in ["Logistic Regression", "Random Forest", "HistGradientBoosting"]:
        pi = permutation_importance(fitted[name]["model"], X_test, y_test,
                                    scoring="roc_auc", n_repeats=20,
                                    random_state=config.RANDOM_STATE, n_jobs=-1)
        frames.append(pd.DataFrame({"model": name, "feature": X_test.columns,
                                    "importance_mean": pi.importances_mean,
                                    "importance_std": pi.importances_std}))
    imp = pd.concat(frames, ignore_index=True)
    imp.round(5).to_csv(config.TABLE_DIR / "permutation_importance.csv", index=False)

    pivot = imp.pivot(index="feature", columns="model", values="importance_mean")
    pivot = pivot.sort_values("Random Forest")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    pivot.plot.barh(ax=ax, color=[MODEL_COLORS[c] for c in pivot.columns])
    ax.set(xlabel="Mean drop of ROC-AUC when the feature is shuffled",
           title="Permutation importance on the test set (20 repeats)")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "12_permutation_importance.png", dpi=150)
    plt.close(fig)
    return imp


def error_analysis(name: str, proba, X_test, y_test) -> pd.DataFrame:
    """Compare feature means of TP / TN / FP / FN groups of the best model."""
    data = X_test.copy()
    data["y_true"] = y_test.values
    data["proba"] = proba
    data["pred"] = (proba >= 0.5).astype(int)
    data["outcome"] = np.select(
        [(data.y_true == 1) & (data.pred == 1), (data.y_true == 0) & (data.pred == 0),
         (data.y_true == 0) & (data.pred == 1), (data.y_true == 1) & (data.pred == 0)],
        ["TP", "TN", "FP", "FN"], default="?")
    summary = data.groupby("outcome")[config.ALL_FEATURES + ["proba"]].mean().round(2)
    summary.insert(0, "n", data["outcome"].value_counts())
    summary.to_csv(config.TABLE_DIR / "error_analysis_by_outcome.csv")
    errors = data[data.outcome.isin(["FP", "FN"])].copy()
    errors["confidence"] = (errors["proba"] - 0.5).abs()
    errors.sort_values("confidence", ascending=False).head(15).round(3).to_csv(
        config.TABLE_DIR / "error_analysis_most_confident_errors.csv")
    borderline = int(((data.proba > 0.4) & (data.proba < 0.6)).sum())
    share_err_borderline = float(((errors.proba > 0.4) & (errors.proba < 0.6)).mean())
    note = {"model": name, "errors": int(len(errors)),
            "test_cases_in_0.4-0.6": borderline,
            "share_of_errors_in_0.4-0.6": round(share_err_borderline, 3)}
    with open(config.TABLE_DIR / "error_analysis_note.json", "w", encoding="utf-8") as f:
        json.dump(note, f, indent=2)
    return summary


def save_to_database(metrics: pd.DataFrame, results: dict, X_test, y_test) -> None:
    """Store experiment results and test-set predictions in SQLite."""
    now = datetime.now().isoformat(timespec="seconds")
    with closing(get_connection()) as conn:
        for _, row in metrics.iterrows():
            cur = conn.execute(
                """INSERT INTO experiment (model_name, roc_auc, pr_auc, f1, accuracy,
                   precision_score, recall, brier, cv_roc_auc_mean, cv_roc_auc_std,
                   fit_time_s, predict_ms_per_1000, model_size_kb, best_params, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (row.model, row.roc_auc, row.pr_auc, row.f1, row.accuracy,
                 row.precision_score, row.recall, row.brier, row.cv_roc_auc_mean,
                 row.cv_roc_auc_std, row.fit_time_s, row.predict_ms_per_1000,
                 row.model_size_kb, row.best_params, now))
            run_id = cur.lastrowid
            r = results[row.model]
            # patient_id in the DB = row position + 1 (see database.build_tables)
            preds = [(run_id, int(idx) + 1, int(t), float(p), int(pr))
                     for idx, t, p, pr in zip(X_test.index, y_test, r["proba"], r["pred"])]
            conn.executemany("INSERT INTO prediction VALUES (?,?,?,?,?)", preds)
        conn.commit()


def run_modeling(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Full experiment: split, tune, evaluate, interpret, persist."""
    config.ensure_dirs()
    X_train, X_test, y_train, y_test, leakage = split_data(df)
    if verbose:
        print("Split / leakage check:", leakage)
    fitted = train_models(X_train, y_train, verbose)

    results, rows = {}, []
    for name, info in fitted.items():
        r = evaluate(info["model"], X_test, y_test)
        results[name] = r
        rows.append({"model": name, **{k: v for k, v in r.items()
                                       if k not in ("proba", "pred")},
                     "cv_roc_auc_mean": info["cv_roc_auc_mean"],
                     "cv_roc_auc_std": info["cv_roc_auc_std"],
                     "fit_time_s": info["fit_time_s"],
                     "search_time_s": info["search_time_s"],
                     "peak_memory_mb": info["peak_memory_mb"],
                     "model_size_kb": model_size_kb(info["model"]),
                     "best_params": json.dumps({k.replace("clf__", ""): v for k, v in
                                                info["best_params"].items()})})
        joblib.dump(info["model"], config.MODEL_DIR /
                    f"{name.split(' (')[0].replace(' ', '_').lower()}.joblib")

    metrics = pd.DataFrame(rows)
    metrics.round(4).to_csv(config.TABLE_DIR / "model_comparison.csv", index=False)
    metrics.to_parquet(config.TABLE_DIR / "model_comparison.parquet", index=False)

    candidates = metrics[~metrics.model.str.startswith("Dummy")]
    best_name = candidates.sort_values(["cv_roc_auc_mean"], ascending=False).iloc[0]["model"]
    joblib.dump({"model": fitted[best_name]["model"], "name": best_name,
                 "features": config.ALL_FEATURES,
                 "trained_at": datetime.now().isoformat(timespec="seconds")},
                config.MODEL_DIR / "best_model.joblib")
    with open(config.TABLE_DIR / "split_leakage_check.json", "w", encoding="utf-8") as f:
        json.dump({**leakage, "best_model_by_cv": best_name}, f, indent=2)

    plot_roc_pr(results, y_test)
    plot_calibration(results, y_test)
    plot_confusion_matrices(results, y_test)
    importance_analysis(fitted, X_test, y_test)
    err = error_analysis(best_name, results[best_name]["proba"], X_test, y_test)
    save_to_database(metrics, results, X_test, y_test)

    if verbose:
        show = ["model", "roc_auc", "pr_auc", "f1", "accuracy", "recall", "brier",
                "cv_roc_auc_mean", "fit_time_s", "model_size_kb", "predict_ms_per_1000"]
        print("\n", metrics[show].round(4).to_string(index=False))
        print(f"\nBest model (chosen by CV ROC-AUC, not by test): {best_name}")
        print("\nError analysis (feature means by outcome):\n", err.to_string())
    return metrics

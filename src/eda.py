"""Exploratory data analysis: descriptive statistics and visualisations."""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import plotly.express as px         # noqa: E402
import seaborn as sns               # noqa: E402

from src import config

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {0: "#4C9BE8", 1: "#E8684C"}
NICE = {"Age": "Age, years", "BMI": "BMI, kg/m²",
        "AlcoholConsumption": "Alcohol, units/week",
        "PhysicalActivity": "Physical activity, h/week",
        "LiverFunctionTest": "Liver function test score",
        "Gender": "Gender", "Smoking": "Smoking", "Diabetes": "Diabetes",
        "Hypertension": "Hypertension", "GeneticRisk": "Genetic risk"}


def _save(fig, name: str) -> str:
    path = config.FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics overall and by diagnosis group."""
    rows = []
    for col in config.NUMERIC_FEATURES + config.BINARY_FEATURES + config.ORDINAL_FEATURES:
        s = df[col].astype("float64")
        rows.append({
            "feature": col, "type": "numeric" if col in config.NUMERIC_FEATURES
            else ("binary" if col in config.BINARY_FEATURES else "ordinal"),
            "mean": s.mean(), "std": s.std(), "min": s.min(), "median": s.median(),
            "max": s.max(),
            "mean_healthy": s[df[config.TARGET] == 0].mean(),
            "mean_disease": s[df[config.TARGET] == 1].mean()})
    table = pd.DataFrame(rows).round(3)
    table.to_csv(config.TABLE_DIR / "descriptive_stats.csv", index=False)
    return table


def plot_target_balance(df: pd.DataFrame) -> str:
    counts = df[config.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar([config.LABELS["Diagnosis"][i] for i in counts.index], counts.values,
                  color=[PALETTE[i] for i in counts.index])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 10,
                f"{v} ({100 * v / counts.sum():.1f}%)", ha="center")
    ax.set_ylabel("Patients")
    ax.set_title("Target distribution (class balance)")
    return _save(fig, "01_target_balance.png")


def plot_numeric_distributions(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    data = df.copy()
    data["Diagnosis "] = data[config.TARGET].map(config.LABELS["Diagnosis"])
    label_palette = {config.LABELS["Diagnosis"][k]: v for k, v in PALETTE.items()}
    for ax, col in zip(axes.flat, config.NUMERIC_FEATURES):
        sns.histplot(data=data, x=col, hue="Diagnosis ", bins=30, kde=True,
                     palette=label_palette, ax=ax, element="step", stat="count",
                     hue_order=list(label_palette))
        ax.set_xlabel(NICE[col])
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0.05, 0.5, "Flat histograms = uniform\ndistributions, a typical\n"
                       "sign of synthetic data.\nDisease rate rises in steps\n"
                       "(alcohol, LFT, age).", fontsize=12, va="center")
    fig.suptitle("Distribution of numeric features by diagnosis", fontsize=14)
    return _save(fig, "02_numeric_distributions.png")


def plot_boxplots(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))
    for ax, col in zip(axes, config.NUMERIC_FEATURES):
        sns.boxplot(data=df, x=config.TARGET, y=col, hue=config.TARGET,
                    palette=PALETTE, ax=ax, legend=False)
        ax.set_xticks([0, 1], ["No disease", "Disease"])
        ax.set_xlabel("")
        ax.set_ylabel(NICE[col])
    fig.suptitle("Boxplots (1.5*IQR whiskers): no outliers beyond the fences", fontsize=13)
    return _save(fig, "03_boxplots_iqr.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    cols = config.ALL_FEATURES + [config.TARGET]
    corr = df[cols].astype("float64").corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-0.5, vmax=0.5, ax=ax, cbar_kws={"label": "Spearman ρ"})
    ax.set_title("Spearman correlation matrix (features + Diagnosis)")
    corr.round(3).to_csv(config.TABLE_DIR / "correlation_matrix.csv")
    return _save(fig, "04_correlation_heatmap.png")


def plot_categorical_rates(df: pd.DataFrame) -> str:
    cats = config.BINARY_FEATURES + config.ORDINAL_FEATURES
    fig, axes = plt.subplots(1, len(cats), figsize=(18, 4.2), sharey=True)
    for ax, col in zip(axes, cats):
        g = df.groupby(col, observed=True)[config.TARGET].agg(["mean", "count"])
        se = np.sqrt(g["mean"] * (1 - g["mean"]) / g["count"])
        labels = [config.LABELS.get(col, {0: "No", 1: "Yes"}).get(int(i), str(i))
                  for i in g.index]
        ax.bar(labels, 100 * g["mean"], yerr=196 * se, capsize=4, color="#7aa6d6")
        for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
            ax.text(i, 3, f"n={n}", ha="center", color="white", fontsize=9)
        ax.set_title(NICE[col])
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Liver disease rate, % (±95% CI)")
    fig.suptitle("Disease rate by categorical risk factors", fontsize=13)
    return _save(fig, "05_categorical_rates.png")


def plot_binned_rates(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, col in zip(axes, ["AlcoholConsumption", "LiverFunctionTest", "Age"]):
        bins = pd.cut(df[col].astype(float), bins=10)
        g = df.groupby(bins, observed=True)[config.TARGET].mean() * 100
        ax.plot([iv.mid for iv in g.index], g.values, marker="o", color="#E8684C")
        ax.axhline(100 * df[config.TARGET].mean(), ls="--", color="grey",
                   label="overall rate")
        ax.set_xlabel(NICE[col])
        ax.set_ylabel("Disease rate, %")
        ax.set_ylim(0, 100)
        ax.legend()
    fig.suptitle("Disease rate across deciles of value range: step-like thresholds", fontsize=13)
    return _save(fig, "06_binned_rates.png")


def plot_interactive(df: pd.DataFrame) -> str:
    """Interactive Plotly scatter (HTML) - hover, zoom, filter by legend."""
    data = df.copy()
    data["Diagnosis label"] = data[config.TARGET].map(config.LABELS["Diagnosis"])
    data["Gender label"] = data["Gender"].map(config.LABELS["Gender"])
    fig = px.scatter(data, x="AlcoholConsumption", y="LiverFunctionTest",
                     color="Diagnosis label", symbol="Gender label",
                     size="BMI", size_max=9, opacity=0.65,
                     hover_data=["Age", "BMI", "Smoking", "GeneticRisk"],
                     color_discrete_map={"No liver disease": PALETTE[0],
                                         "Liver disease": PALETTE[1]},
                     labels={"AlcoholConsumption": NICE["AlcoholConsumption"],
                             "LiverFunctionTest": NICE["LiverFunctionTest"]},
                     title="Alcohol vs liver function test (marker size = BMI)")
    path = config.FIG_DIR / "07_interactive_scatter.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return str(path)


def run_eda(df: pd.DataFrame, verbose: bool = True) -> list:
    """Produce all EDA tables and figures; return the list of saved files."""
    config.ensure_dirs()
    stats_table = descriptive_stats(df)
    files = [plot_target_balance(df), plot_numeric_distributions(df),
             plot_boxplots(df), plot_correlation_heatmap(df),
             plot_categorical_rates(df), plot_binned_rates(df), plot_interactive(df)]
    if verbose:
        print(stats_table.to_string(index=False))
        print("\nSaved figures:")
        for f in files:
            print("  ", f)
    return files

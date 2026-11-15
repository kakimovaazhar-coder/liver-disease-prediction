# statistical tests: chi-square for categorical features,
# mann-whitney for numeric ones, and odds ratios
import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config


def chi_square_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in config.BINARY_FEATURES + config.ORDINAL_FEATURES:
        table = pd.crosstab(df[col], df[config.TARGET])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        n = table.values.sum()
        cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
        rows.append({"feature": col, "test": "chi-square", "statistic": chi2,
                     "dof": dof, "p_value": p, "effect_size": cramers_v,
                     "effect_measure": "Cramer's V"})
    return pd.DataFrame(rows)


def mann_whitney_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in config.NUMERIC_FEATURES:
        sick = df.loc[df[config.TARGET] == 1, col].astype(float)
        healthy = df.loc[df[config.TARGET] == 0, col].astype(float)
        u, p = stats.mannwhitneyu(sick, healthy, alternative="two-sided")
        rank_biserial = 2 * u / (len(sick) * len(healthy)) - 1
        rows.append({"feature": col, "test": "Mann-Whitney U", "statistic": u,
                     "dof": np.nan, "p_value": p, "effect_size": rank_biserial,
                     "effect_measure": "rank-biserial r",
                     "median_disease": sick.median(), "median_healthy": healthy.median()})
    return pd.DataFrame(rows)


def crude_odds_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Odds ratios from 2x2 tables with a Woolf 95% CI (Haldane 0.5 correction)."""
    rows = []
    for col in config.BINARY_FEATURES:
        a = ((df[col] == 1) & (df[config.TARGET] == 1)).sum() + 0.5
        b = ((df[col] == 1) & (df[config.TARGET] == 0)).sum() + 0.5
        c = ((df[col] == 0) & (df[config.TARGET] == 1)).sum() + 0.5
        d = ((df[col] == 0) & (df[config.TARGET] == 0)).sum() + 0.5
        odds_ratio = (a * d) / (b * c)
        se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        rows.append({"feature": col, "crude_OR": odds_ratio,
                     "ci_low": np.exp(np.log(odds_ratio) - 1.96 * se),
                     "ci_high": np.exp(np.log(odds_ratio) + 1.96 * se)})
    return pd.DataFrame(rows)


def adjusted_odds_ratios(df: pd.DataFrame):
    """Logistic regression with all features. numeric are scaled -> OR per 1 SD."""
    X = df[config.ALL_FEATURES].astype(float).copy()
    for col in config.NUMERIC_FEATURES:
        X[col] = (X[col] - X[col].mean()) / X[col].std()
    X["GeneticRisk_Medium"] = (X["GeneticRisk"] == 1).astype(float)
    X["GeneticRisk_High"] = (X["GeneticRisk"] == 2).astype(float)
    X = sm.add_constant(X.drop(columns="GeneticRisk"))
    model = sm.Logit(df[config.TARGET].astype(float), X).fit(disp=False)
    ci = model.conf_int()
    table = pd.DataFrame({
        "feature": model.params.index, "coef": model.params.values,
        "adjusted_OR": np.exp(model.params.values),
        "ci_low": np.exp(ci[0].values), "ci_high": np.exp(ci[1].values),
        "p_value": model.pvalues.values})
    table = table[table["feature"] != "const"].reset_index(drop=True)
    def unit(f):
        if f in config.NUMERIC_FEATURES:
            return "per 1 SD"
        if f == "Gender":
            return "female vs male"
        return "level vs Low" if f.startswith("GeneticRisk") else "yes vs no"
    table["unit"] = [unit(f) for f in table["feature"]]
    return table, model


def plot_forest(or_table: pd.DataFrame) -> str:
    t = or_table.sort_values("adjusted_OR")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    y = np.arange(len(t))
    colors = ["#E8684C" if p < 0.05 else "#9e9e9e" for p in t["p_value"]]
    ax.errorbar(t["adjusted_OR"], y,
                xerr=[t["adjusted_OR"] - t["ci_low"], t["ci_high"] - t["adjusted_OR"]],
                fmt="none", ecolor="#555", capsize=3)
    ax.scatter(t["adjusted_OR"], y, color=colors, zorder=3, s=45)
    ax.axvline(1, ls="--", color="grey")
    ax.set_yticks(y, [f"{f} ({u})" for f, u in zip(t["feature"], t["unit"])])
    ax.set_xscale("log")
    ax.set_xlabel("Adjusted odds ratio (log scale), 95% CI")
    ax.set_title("Multivariable logistic regression: adjusted odds ratios\n"
                 "(red = significant at p < 0.05)")
    path = config.FIG_DIR / "08_forest_plot_odds_ratios.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def run_tests(df: pd.DataFrame, verbose: bool = True) -> dict:
    config.ensure_dirs()
    tests = pd.concat([chi_square_tests(df), mann_whitney_tests(df)], ignore_index=True)
    tests["p_holm"] = multipletests(tests["p_value"], method="holm")[1]
    tests["significant"] = tests["p_holm"] < 0.05
    crude = crude_odds_ratios(df)
    adjusted, model = adjusted_odds_ratios(df)

    tests.round(5).to_csv(config.TABLE_DIR / "stat_tests.csv", index=False)
    crude.round(3).to_csv(config.TABLE_DIR / "odds_ratios_crude.csv", index=False)
    adjusted.round(4).to_csv(config.TABLE_DIR / "odds_ratios_adjusted.csv", index=False)
    (config.TABLE_DIR / "logit_summary.txt").write_text(str(model.summary()),
                                                        encoding="utf-8")
    forest = plot_forest(adjusted)

    if verbose:
        cols = ["feature", "test", "statistic", "p_value", "p_holm", "effect_size",
                "effect_measure", "significant"]
        print(tests[cols].round(4).to_string(index=False))
        print("\nCrude odds ratios:\n", crude.round(3).to_string(index=False))
        print("\nAdjusted odds ratios:\n",
              adjusted[["feature", "adjusted_OR", "ci_low", "ci_high", "p_value", "unit"]]
              .round(3).to_string(index=False))
        print(f"\nPseudo R^2 (McFadden): {model.prsquared:.3f}")
        print("Forest plot:", forest)
    return {"tests": tests, "crude": crude, "adjusted": adjusted,
            "pseudo_r2": model.prsquared}

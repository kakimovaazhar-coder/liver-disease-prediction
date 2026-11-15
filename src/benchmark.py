# benchmark: CSV vs Parquet and Pandas vs SQL
# our dataset has only 1700 rows, so we make a copy with 1M rows (random
# rows with replacement) ONLY to measure time and memory, not for analysis
import gc
import sqlite3
import time
import tracemalloc
from contextlib import closing

import numpy as np
import pandas as pd

from src import config
from src.data_loader import OPTIMAL_DTYPES, memory_mb

STRESS_ROWS = 1_000_000


def make_stress_copy(n_rows: int = STRESS_ROWS) -> dict:
    """Write the resampled stress-test copy as CSV, Parquet and SQLite."""
    config.ensure_dirs()
    base = pd.read_csv(config.RAW_CSV)
    rng = np.random.default_rng(config.RANDOM_STATE)
    big = base.iloc[rng.integers(0, len(base), n_rows)].reset_index(drop=True)
    paths = {"csv": config.BENCH_DIR / "stress.csv",
             "parquet": config.BENCH_DIR / "stress.parquet",
             "sqlite": config.BENCH_DIR / "stress.db"}
    big.to_csv(paths["csv"], index=False)
    big.astype(OPTIMAL_DTYPES).to_parquet(paths["parquet"], index=False)
    if paths["sqlite"].exists():
        paths["sqlite"].unlink()
    with closing(sqlite3.connect(paths["sqlite"])) as conn:
        big.to_sql("records", conn, index=False, chunksize=100_000)
        conn.commit()
    return paths


def measure(func):
    """Run func() and return (result, seconds, peak traced memory in MB)."""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = func()
    seconds = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, seconds, peak / 1024 ** 2


# loading approaches
def read_csv_default(path):
    return pd.read_csv(path)


def read_csv_chunked_optimized(path):
    parts = [c.astype(OPTIMAL_DTYPES) for c in pd.read_csv(path, chunksize=200_000)]
    return pd.concat(parts, ignore_index=True)


def read_parquet(path):
    return pd.read_parquet(path)


# aggregation approaches (same question, three engines)
AGG_SQL = """
SELECT Gender, Smoking, GeneticRisk, COUNT(*) AS n,
       AVG(Diagnosis) AS rate, AVG(AlcoholConsumption) AS mean_alcohol
FROM records GROUP BY Gender, Smoking, GeneticRisk
ORDER BY Gender, Smoking, GeneticRisk;"""


def agg_pandas_inmemory(df):
    return (df.groupby(["Gender", "Smoking", "GeneticRisk"], observed=True)
              .agg(n=("Diagnosis", "size"), rate=("Diagnosis", "mean"),
                   mean_alcohol=("AlcoholConsumption", "mean"))
              .reset_index())


def agg_pandas_chunked(path):
    """Incremental aggregation: never holds the full file in memory."""
    keys = ["Gender", "Smoking", "GeneticRisk"]
    partial = []
    for chunk in pd.read_csv(path, chunksize=200_000):
        partial.append(chunk.groupby(keys).agg(n=("Diagnosis", "size"),
                                               cases=("Diagnosis", "sum"),
                                               alc=("AlcoholConsumption", "sum")))
    total = pd.concat(partial).groupby(level=[0, 1, 2]).sum()
    total["rate"] = total["cases"] / total["n"]
    total["mean_alcohol"] = total["alc"] / total["n"]
    return total[["n", "rate", "mean_alcohol"]].reset_index()


def agg_sql(path):
    with closing(sqlite3.connect(path)) as conn:
        return pd.read_sql_query(AGG_SQL, conn)


def results_match(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    cols = ["n", "rate", "mean_alcohol"]
    a = a.sort_values(["Gender", "Smoking", "GeneticRisk"]).reset_index(drop=True)
    b = b.sort_values(["Gender", "Smoking", "GeneticRisk"]).reset_index(drop=True)
    return bool(np.allclose(a[cols].astype(float), b[cols].astype(float), rtol=1e-4))


def run_benchmark(verbose: bool = True, keep_files: bool = False) -> pd.DataFrame:
    if verbose:
        print(f"Creating stress-test copy with {STRESS_ROWS:,} rows ...")
    paths = make_stress_copy()
    sizes = {k: round(p.stat().st_size / 1024 ** 2, 2) for k, p in paths.items()}
    rows = []

    loaders = {"CSV, default dtypes (baseline)": (read_csv_default, paths["csv"], sizes["csv"]),
               "CSV, chunks + optimised dtypes": (read_csv_chunked_optimized, paths["csv"], sizes["csv"]),
               "Parquet (optimised dtypes)": (read_parquet, paths["parquet"], sizes["parquet"])}
    frames = {}
    for name, (func, path, size) in loaders.items():
        df, sec, peak = measure(lambda f=func, p=path: f(p))
        frames[name] = df
        rows.append({"task": "load", "approach": name, "time_s": sec,
                     "peak_mem_mb": peak, "df_memory_mb": memory_mb(df),
                     "file_mb": size, "result_matches_baseline": len(df) == STRESS_ROWS})

    reference = agg_pandas_inmemory(frames["CSV, default dtypes (baseline)"])
    aggs = {"Pandas groupby in memory (baseline)":
                lambda: agg_pandas_inmemory(frames["CSV, default dtypes (baseline)"]),
            "Pandas groupby on Parquet (load + agg)":
                lambda: agg_pandas_inmemory(read_parquet(paths["parquet"])),
            "Pandas chunked incremental agg (CSV)":
                lambda: agg_pandas_chunked(paths["csv"]),
            "SQLite GROUP BY -> pandas":
                lambda: agg_sql(paths["sqlite"])}
    for name, func in aggs.items():
        res, sec, peak = measure(func)
        rows.append({"task": "aggregate", "approach": name, "time_s": sec,
                     "peak_mem_mb": peak, "df_memory_mb": np.nan,
                     "file_mb": sizes["sqlite"] if "SQL" in name else np.nan,
                     "result_matches_baseline": results_match(reference, res)})

    table = pd.DataFrame(rows).round(4)
    table.to_csv(config.TABLE_DIR / "benchmark_processing.csv", index=False)
    if not keep_files:
        for p in paths.values():
            try:
                p.unlink()
            except OSError:
                pass
    if verbose:
        print(table.to_string(index=False))
    return table

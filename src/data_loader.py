# loading the csv: read it in chunks, make dtypes smaller, save to parquet
# the file is small but we wrote it like for a big file
import json
import time
import urllib.request

import pandas as pd

from src import config
from src.privacy import add_pseudonymous_id

# smallest dtypes that fit the value ranges
OPTIMAL_DTYPES = {
    "Age": "uint8",
    "Gender": "uint8",
    "BMI": "float32",
    "AlcoholConsumption": "float32",
    "Smoking": "uint8",
    "GeneticRisk": "uint8",
    "PhysicalActivity": "float32",
    "Diabetes": "uint8",
    "Hypertension": "uint8",
    "LiverFunctionTest": "float32",
    "Diagnosis": "uint8",
}


def download_dataset(force: bool = False) -> bool:
    """Download the CSV from data.gov.kz if it is not present locally."""
    config.ensure_dirs()
    if config.RAW_CSV.exists() and not force:
        return True
    try:
        print(f"Downloading {config.DATASET_URL} ...")
        urllib.request.urlretrieve(config.DATASET_URL, config.RAW_CSV)
        return True
    except OSError as err:          # no internet or bad url
        print(f"[ERROR] Could not download the dataset: {err}")
        print(f"Download it manually from {config.DATASET_PAGE} "
              f"and put it to {config.RAW_CSV}")
        return False


def memory_mb(df: pd.DataFrame) -> float:
    """Deep memory usage of a DataFrame in megabytes."""
    return df.memory_usage(deep=True).sum() / 1024 ** 2


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast columns to the smallest safe dtype."""
    out = df.copy()
    for col, dtype in OPTIMAL_DTYPES.items():
        if col in out.columns and out[col].notna().all():
            out[col] = out[col].astype(dtype)
        elif col in out.columns:          # int can't store NaN
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out


def load_in_chunks(path=config.RAW_CSV, chunksize: int = config.CHUNK_SIZE):
    """Read csv in chunks and optimize each chunk."""
    chunks, chunk_stats = [], []
    for i, chunk in enumerate(pd.read_csv(path, chunksize=chunksize)):
        before = memory_mb(chunk)
        chunk = optimize_dtypes(chunk)
        chunk_stats.append({"chunk": i, "rows": len(chunk),
                            "mb_before": round(before, 4),
                            "mb_after": round(memory_mb(chunk), 4)})
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True), chunk_stats


def load_and_store(verbose: bool = True) -> pd.DataFrame:
    """csv -> chunks -> smaller dtypes -> parquet, and save memory report."""
    config.ensure_dirs()
    if not download_dataset():
        raise FileNotFoundError(config.RAW_CSV)

    t0 = time.perf_counter()
    naive = pd.read_csv(config.RAW_CSV)               # default int64/float64
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    df, chunk_stats = load_in_chunks()
    t_chunked = time.perf_counter() - t0

    df = add_pseudonymous_id(df)
    df.to_parquet(config.CLEAN_PARQUET, index=False, compression="snappy")

    report = {
        "rows": int(len(df)),
        "columns": int(naive.shape[1]),
        "csv_size_kb": round(config.RAW_CSV.stat().st_size / 1024, 1),
        "parquet_size_kb": round(config.CLEAN_PARQUET.stat().st_size / 1024, 1),
        "memory_default_mb": round(memory_mb(naive), 4),
        "memory_optimized_mb": round(memory_mb(df.drop(columns="pseudo_id")), 4),
        "memory_reduction_pct": round(
            100 * (1 - memory_mb(df.drop(columns="pseudo_id")) / memory_mb(naive)), 1),
        "read_time_default_s": round(t_naive, 4),
        "read_time_chunked_s": round(t_chunked, 4),
        "dtypes_default": {c: str(t) for c, t in naive.dtypes.items()},
        "dtypes_optimized": {c: str(t) for c, t in df.dtypes.items()},
        "chunks": chunk_stats,
    }
    with open(config.TABLE_DIR / "memory_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if verbose:
        print(f"Rows: {report['rows']}, chunks: {len(chunk_stats)}")
        print(f"Memory: {report['memory_default_mb']} MB (default) -> "
              f"{report['memory_optimized_mb']} MB (optimised), "
              f"-{report['memory_reduction_pct']}%")
        print(f"File size: CSV {report['csv_size_kb']} KB -> "
              f"Parquet {report['parquet_size_kb']} KB")
    return df


def load_clean() -> pd.DataFrame:
    """Load the processed Parquet file (creating it on first use)."""
    if not config.CLEAN_PARQUET.exists():
        return load_and_store(verbose=False)
    return pd.read_parquet(config.CLEAN_PARQUET)

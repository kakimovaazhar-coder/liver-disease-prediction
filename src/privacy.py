"""Personal-data protection helpers.

The published file has no direct identifiers, but the Cluster 6 rules require
pseudonymisation before storing records in the database and suppression of
small groups (< 10 patients) in every published table or chart.
"""
import hashlib
import os
import secrets

import pandas as pd

from src import config

SALT_FILE = config.DATA_DIR / ".salt"     # never committed (see .gitignore)


def get_salt() -> str:
    """Return the secret salt: env variable LIVER_SALT or a local .salt file."""
    salt = os.environ.get("LIVER_SALT")
    if salt:
        return salt
    if SALT_FILE.exists():
        return SALT_FILE.read_text(encoding="utf-8").strip()
    salt = secrets.token_hex(16)
    SALT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SALT_FILE.write_text(salt, encoding="utf-8")
    return salt


def hash_id(value, salt: str) -> str:
    """Salted SHA-256 hash, shortened to 16 hex chars."""
    return hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()[:16]


def add_pseudonymous_id(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the row number (the only record key) with a salted hash."""
    salt = get_salt()
    out = df.copy()
    out.insert(0, "pseudo_id", [hash_id(i, salt) for i in range(len(out))])
    return out


def suppress_small_groups(table: pd.DataFrame, count_col: str = "n",
                          min_size: int = config.MIN_GROUP_SIZE) -> pd.DataFrame:
    """Mask every statistic of groups with fewer than `min_size` patients."""
    out = table.copy()
    small = out[count_col] < min_size
    if small.any():
        value_cols = [c for c in out.columns if out[c].dtype.kind in "fiu"]
        out[value_cols] = out[value_cols].astype("float64")
        out.loc[small, value_cols] = float("nan")
        out["suppressed"] = small
    return out

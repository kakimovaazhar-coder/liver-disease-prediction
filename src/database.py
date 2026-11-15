"""SQLite database: normalised schema, loading, querying and the ER diagram."""
import sqlite3
from contextlib import closing

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import FancyBboxPatch         # noqa: E402

from src import config

AGE_GROUPS = [(1, "20-29", 20, 29), (2, "30-39", 30, 39), (3, "40-49", 40, 49),
              (4, "50-59", 50, 59), (5, "60-69", 60, 69), (6, "70-80", 70, 120)]
BMI_CATEGORIES = [(1, "Underweight", 0.0, 18.5), (2, "Normal", 18.5, 25.0),
                  (3, "Overweight", 25.0, 30.0), (4, "Obese", 30.0, 100.0)]
COMORBIDITIES = [(1, "Diabetes"), (2, "Hypertension")]


def get_connection(db_path=config.DB_PATH) -> sqlite3.Connection:
    """Open a connection with foreign keys enforced."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def age_group_id(age: float) -> int:
    """Map an age to the id of its age group."""
    for gid, _label, low, high in AGE_GROUPS:
        if low <= age <= high:
            return gid
    return AGE_GROUPS[0][0] if age < AGE_GROUPS[0][2] else AGE_GROUPS[-1][0]


def bmi_category_id(bmi: float):
    """Map a BMI value to its WHO category id (None for missing values)."""
    if pd.isna(bmi):
        return None
    for cid, _label, low, high in BMI_CATEGORIES:
        if low <= bmi < high:
            return cid
    return None


def build_tables(df: pd.DataFrame) -> dict:
    """Split the flat DataFrame into normalised tables (dict name -> DataFrame)."""
    pid = pd.RangeIndex(1, len(df) + 1)
    tables = {
        "gender": pd.DataFrame([(k, v) for k, v in config.LABELS["Gender"].items()],
                               columns=["gender_id", "gender_label"]),
        "age_group": pd.DataFrame(AGE_GROUPS,
                                  columns=["age_group_id", "label", "age_min", "age_max"]),
        "bmi_category": pd.DataFrame(BMI_CATEGORIES,
                                     columns=["bmi_category_id", "label", "bmi_min", "bmi_max"]),
        "genetic_risk_level": pd.DataFrame(
            [(k, v) for k, v in config.LABELS["GeneticRisk"].items()],
            columns=["genetic_risk_id", "label"]),
        "comorbidity": pd.DataFrame(COMORBIDITIES, columns=["comorbidity_id", "name"]),
        "patient": pd.DataFrame({
            "patient_id": pid, "pseudo_id": df["pseudo_id"].values,
            "age": df["Age"].astype(int).values,
            "gender_id": df["Gender"].astype(int).values,
            "age_group_id": [age_group_id(a) for a in df["Age"]]}),
        "lifestyle": pd.DataFrame({
            "patient_id": pid, "bmi": df["BMI"].astype(float).values,
            "bmi_category_id": [bmi_category_id(b) for b in df["BMI"]],
            "alcohol_units": df["AlcoholConsumption"].astype(float).values,
            "smoking": df["Smoking"].astype(int).values,
            "physical_activity": df["PhysicalActivity"].astype(float).values}),
        "clinical_exam": pd.DataFrame({
            "patient_id": pid,
            "genetic_risk_id": df["GeneticRisk"].astype(int).values,
            "liver_function_test": df["LiverFunctionTest"].astype(float).values}),
        "diagnosis": pd.DataFrame({
            "patient_id": pid,
            "has_liver_disease": df[config.TARGET].astype(int).values}),
    }
    links = []
    for comorb_id, name in COMORBIDITIES:
        for p in pid[df[name].astype(int).values == 1]:
            links.append((int(p), comorb_id))
    tables["patient_comorbidity"] = pd.DataFrame(
        links, columns=["patient_id", "comorbidity_id"])
    return tables


def create_database(df: pd.DataFrame, verbose: bool = True) -> dict:
    """(Re)create the database from schema.sql and load all tables."""
    config.ensure_dirs()
    schema = (config.SQL_DIR / "schema.sql").read_text(encoding="utf-8")
    tables = build_tables(df)
    load_order = ["gender", "age_group", "bmi_category", "genetic_risk_level",
                  "comorbidity", "patient", "lifestyle", "clinical_exam",
                  "patient_comorbidity", "diagnosis"]
    counts = {}
    with closing(get_connection()) as conn:
        conn.executescript(schema)
        for name in load_order:
            tables[name].to_sql(name, conn, if_exists="append", index=False)
            counts[name] = len(tables[name])
        conn.commit()
    if verbose:
        print(f"SQLite {sqlite3.sqlite_version} database: {config.DB_PATH}")
        for name, n in counts.items():
            print(f"  {name:<22}{n:>6} rows")
    return counts


def run_query(sql: str, params=None) -> pd.DataFrame:
    """Execute a SELECT and return the result as a pandas DataFrame."""
    with closing(get_connection()) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def load_flat_from_db() -> pd.DataFrame:
    """Re-assemble the modelling table from the normalised schema (JOINs)."""
    sql = """
    SELECT p.patient_id, p.age AS Age, p.gender_id AS Gender, l.bmi AS BMI,
           l.alcohol_units AS AlcoholConsumption, l.smoking AS Smoking,
           c.genetic_risk_id AS GeneticRisk, l.physical_activity AS PhysicalActivity,
           MAX(CASE WHEN pc.comorbidity_id = 1 THEN 1 ELSE 0 END) AS Diabetes,
           MAX(CASE WHEN pc.comorbidity_id = 2 THEN 1 ELSE 0 END) AS Hypertension,
           c.liver_function_test AS LiverFunctionTest,
           d.has_liver_disease AS Diagnosis
    FROM patient p
    JOIN lifestyle l      ON l.patient_id = p.patient_id
    JOIN clinical_exam c  ON c.patient_id = p.patient_id
    JOIN diagnosis d      ON d.patient_id = p.patient_id
    LEFT JOIN patient_comorbidity pc ON pc.patient_id = p.patient_id
    GROUP BY p.patient_id
    ORDER BY p.patient_id;
    """
    return run_query(sql)


# --------------------------------------------------------------------------- #
# ER diagram (drawn with matplotlib so that no Graphviz binary is required)
# --------------------------------------------------------------------------- #
ER_TABLES = {
    "gender": (["PK gender_id", "gender_label"], (0.3, 7.6)),
    "age_group": (["PK age_group_id", "label", "age_min", "age_max"], (0.3, 4.9)),
    "patient": (["PK patient_id", "pseudo_id", "age", "FK gender_id",
                 "FK age_group_id"], (4.2, 6.2)),
    "bmi_category": (["PK bmi_category_id", "label", "bmi_min", "bmi_max"], (0.3, 1.6)),
    "lifestyle": (["PK,FK patient_id", "bmi", "FK bmi_category_id", "alcohol_units",
                   "smoking", "physical_activity"], (4.2, 1.6)),
    "clinical_exam": (["PK,FK patient_id", "FK genetic_risk_id",
                       "liver_function_test"], (8.1, 1.9)),
    "genetic_risk_level": (["PK genetic_risk_id", "label"], (12.0, 2.2)),
    "diagnosis": (["PK,FK patient_id", "has_liver_disease"], (4.2, 10.0)),
    "patient_comorbidity": (["PK,FK patient_id", "PK,FK comorbidity_id"], (8.1, 5.0)),
    "comorbidity": (["PK comorbidity_id", "name"], (12.0, 5.2)),
    "experiment": (["PK run_id", "model_name", "roc_auc, f1, ...", "fit_time_s",
                    "model_size_kb"], (12.0, 7.6)),
    "prediction": (["PK,FK run_id", "PK,FK patient_id", "y_true", "y_prob",
                    "y_pred"], (8.1, 10.6)),
}
ER_LINKS = [("patient", "gender"), ("patient", "age_group"),
            ("lifestyle", "patient"), ("lifestyle", "bmi_category"),
            ("clinical_exam", "patient"), ("clinical_exam", "genetic_risk_level"),
            ("diagnosis", "patient"), ("patient_comorbidity", "patient"),
            ("patient_comorbidity", "comorbidity"), ("prediction", "experiment"),
            ("prediction", "patient")]


def _edge_point(box, target):
    """Point where the line from the box centre to `target` leaves the box."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    dx, dy = target[0] - cx, target[1] - cy
    if dx == 0 and dy == 0:
        return cx, cy
    scale = min((w / 2) / abs(dx) if dx else float("inf"),
                (h / 2) / abs(dy) if dy else float("inf"))
    return cx + dx * scale, cy + dy * scale


def draw_er_diagram(path=None) -> str:
    """Draw the ER diagram of the schema and save it as PNG."""
    path = path or (config.DOCS_DIR / "er_diagram.png")
    config.ensure_dirs()
    width, line_h = 3.2, 0.42
    fig, ax = plt.subplots(figsize=(16, 14))
    boxes = {}
    for name, (cols, (x, y)) in ER_TABLES.items():
        h = line_h * (len(cols) + 1)
        boxes[name] = (x, y, width, h)
        ax.add_patch(FancyBboxPatch((x, y), width, h, boxstyle="round,pad=0.02",
                                    fc="#eef4fb", ec="#2c5d8f", lw=1.6))
        ax.add_patch(FancyBboxPatch((x, y + h - line_h), width, line_h,
                                    boxstyle="round,pad=0.02", fc="#2c5d8f", ec="#2c5d8f"))
        ax.text(x + width / 2, y + h - line_h / 2, name, ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
        for i, col in enumerate(cols):
            ax.text(x + 0.12, y + h - line_h * (i + 1.5), col, va="center",
                    fontsize=9.5, family="monospace",
                    fontweight="bold" if col.startswith("PK") else "normal")
    for child, parent in ER_LINKS:
        cx, cy, cw, ch = boxes[child]
        px, py, pw, ph = boxes[parent]
        start = _edge_point((cx, cy, cw, ch), (px + pw / 2, py + ph / 2))
        end = _edge_point((px, py, pw, ph), (cx + cw / 2, cy + ch / 2))
        ax.annotate("", xy=end, xytext=start, zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.3,
                                    mutation_scale=16, shrinkA=0, shrinkB=0))
    ax.text(0.3, 13.4, "ER diagram - liver disease database (SQLite, 3NF)",
            fontsize=15, fontweight="bold")
    ax.text(0.3, 12.95, "Arrow: many -> one (foreign key). patient_comorbidity "
            "resolves the many-to-many link patient <-> comorbidity.", fontsize=10)
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0.8, 13.9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)

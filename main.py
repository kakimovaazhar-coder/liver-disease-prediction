"""Liver Disease Risk Factors and Diagnosis Prediction - console application.

Run:
    python main.py          interactive menu
    python main.py --all    run the whole pipeline without questions
"""
import json
import sys
from datetime import datetime

from src import config

# Windows consoles often use cp1251; force UTF-8 so tables print correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STATE_FILE = config.RESULTS_DIR / "run_log.json"

MENU = {
    "1": "Load data (chunks -> optimised dtypes -> Parquet)",
    "2": "Data quality audit + real/synthetic origin check",
    "3": "Build SQLite database + ER diagram",
    "4": "Run analytical SQL queries",
    "5": "EDA and visualisations",
    "6": "Statistical tests and odds ratios",
    "7": "Train and compare models (Dummy / LR / RF / HGB)",
    "8": "Benchmark: CSV vs Parquet, Pandas vs SQL (1M-row stress copy)",
    "9": "Predict liver disease for a new patient",
    "10": "Run the full pipeline (1-8)",
    "0": "Exit",
}


def load_state() -> dict:
    """Read the log of previous runs (program state) if it exists."""
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"runs": []}


def save_state(state: dict, step: str, ok: bool) -> None:
    """Append the executed step to the log file."""
    config.ensure_dirs()
    state["runs"].append({"step": step, "ok": ok,
                          "time": datetime.now().isoformat(timespec="seconds")})
    state["runs"] = state["runs"][-200:]
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def get_clean_data():
    """Raw/processed data -> cleaning rules from the audit."""
    from src.data_loader import load_clean
    from src.quality_audit import clean_data
    return clean_data(load_clean())


def step_load():
    from src.data_loader import load_and_store
    load_and_store()


def step_audit():
    from src.data_loader import load_clean
    from src.quality_audit import run_audit
    run_audit(load_clean())


def step_database():
    from src.database import create_database, draw_er_diagram
    create_database(get_clean_data())
    print("ER diagram:", draw_er_diagram())


def step_sql():
    from src.sql_queries import run_all_queries
    if not config.DB_PATH.exists():
        print("Database not found - building it first.")
        step_database()
    run_all_queries(include_experiments=True)


def step_eda():
    from src.eda import run_eda
    run_eda(get_clean_data())


def step_stats():
    from src.stats_tests import run_tests
    run_tests(get_clean_data())


def step_models():
    from src.modeling import run_modeling
    if not config.DB_PATH.exists():
        step_database()
    run_modeling(get_clean_data())


def step_benchmark():
    from src.benchmark import run_benchmark
    run_benchmark()


def step_predict():
    from src.predict import interactive_prediction
    interactive_prediction()


STEPS = {"1": step_load, "2": step_audit, "3": step_database, "4": step_sql,
         "5": step_eda, "6": step_stats, "7": step_models, "8": step_benchmark,
         "9": step_predict}
PIPELINE = ["1", "2", "3", "5", "6", "7", "4", "8"]   # SQL after models -> leaderboard


def run_step(key: str, state: dict) -> bool:
    """Execute one menu item, catching and reporting any error."""
    print(f"\n{'=' * 70}\n{key}. {MENU[key]}\n{'=' * 70}")
    try:
        STEPS[key]()
        ok = True
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        ok = False
    except Exception as err:            # keep the menu alive after any failure
        print(f"[ERROR] {type(err).__name__}: {err}")
        ok = False
    save_state(state, MENU[key], ok)
    return ok


def run_pipeline(state: dict) -> None:
    results = {key: run_step(key, state) for key in PIPELINE}
    print("\nPipeline summary:")
    for key, ok in results.items():
        print(f"  [{'OK' if ok else 'FAILED'}] {MENU[key]}")


def main() -> None:
    config.ensure_dirs()
    state = load_state()
    if "--all" in sys.argv:
        run_pipeline(state)
        return
    while True:
        print("\n=== Liver Disease Risk Factors & Diagnosis Prediction ===")
        for key, text in MENU.items():
            print(f"  {key:>2} - {text}")
        choice = input("Choose an option: ").strip()
        if choice == "0":
            print("Goodbye!")
            break
        elif choice == "10":
            run_pipeline(state)
        elif choice in STEPS:
            run_step(choice, state)
        else:
            print("Unknown option, please try again.")


if __name__ == "__main__":
    main()

# prediction for a new patient (console input)
import joblib
import pandas as pd

from src import config

BEST_MODEL_PATH = config.MODEL_DIR / "best_model.joblib"

# feature -> (question, type, min, max)
QUESTIONS = {
    "Age": ("Age (20-80 years)", int, 18, 100),
    "Gender": ("Gender (0 = male, 1 = female)", int, 0, 1),
    "BMI": ("Body mass index (15-40)", float, 10, 60),
    "AlcoholConsumption": ("Alcohol consumption, units per week (0-20)", float, 0, 30),
    "Smoking": ("Smoking (0 = no, 1 = yes)", int, 0, 1),
    "GeneticRisk": ("Genetic risk (0 = low, 1 = medium, 2 = high)", int, 0, 2),
    "PhysicalActivity": ("Physical activity, hours per week (0-10)", float, 0, 15),
    "Diabetes": ("Diabetes (0 = no, 1 = yes)", int, 0, 1),
    "Hypertension": ("Hypertension (0 = no, 1 = yes)", int, 0, 1),
    "LiverFunctionTest": ("Liver function test score (20-100)", float, 0, 150),
}


def load_best_model() -> dict:
    """Load the saved best model bundle; raise a clear error if missing."""
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run menu item 7 (train models) or `python main.py --all`.")
    return joblib.load(BEST_MODEL_PATH)


def risk_level(probability: float) -> str:
    if probability < 0.3:
        return "LOW"
    elif probability < 0.6:
        return "MODERATE"
    else:
        return "HIGH"


def predict_patient(values: dict, bundle: dict = None) -> dict:
    """Return probability, class and risk level for one patient (dict of features)."""
    bundle = bundle or load_best_model()
    row = pd.DataFrame([{f: float(values[f]) for f in config.ALL_FEATURES}])
    proba = float(bundle["model"].predict_proba(row)[0, 1])
    return {"model": bundle["name"], "probability": proba,
            "prediction": int(proba >= 0.5), "risk_level": risk_level(proba)}


def ask_value(question: str, cast, low, high):
    """Ask until the user enters a valid number inside [low, high]."""
    while True:
        raw = input(f"  {question}: ").strip().replace(",", ".")
        try:
            value = cast(float(raw)) if cast is int else cast(raw)
        except ValueError:
            print("    Please enter a number.")
            continue
        if low <= value <= high:
            return value
        print(f"    Value must be between {low} and {high}.")


def interactive_prediction() -> None:
    try:
        bundle = load_best_model()
    except FileNotFoundError as err:
        print(f"[ERROR] {err}")
        return
    print(f"\nModel: {bundle['name']} (trained {bundle['trained_at']})")
    print("Enter patient data:")
    values = {f: ask_value(*spec) for f, spec in QUESTIONS.items()}
    result = predict_patient(values, bundle)
    print(f"\n  Probability of liver disease: {result['probability']:.1%}")
    print(f"  Predicted class: {config.LABELS['Diagnosis'][result['prediction']]}")
    print(f"  Risk level: {result['risk_level']}")
    print("  (Educational model trained on synthetic data - not a medical diagnosis.)")

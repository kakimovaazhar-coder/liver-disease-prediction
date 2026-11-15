# streamlit dashboard, run: streamlit run app/dashboard.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config
from src.data_loader import load_clean
from src.database import AGE_GROUPS, BMI_CATEGORIES, run_query
from src.predict import QUESTIONS, load_best_model, predict_patient
from src.privacy import suppress_small_groups

st.set_page_config(page_title="Liver Disease Analytics", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    df = load_clean().drop(columns="pseudo_id")
    df["Gender label"] = df["Gender"].map(config.LABELS["Gender"])
    df["Genetic risk"] = df["GeneticRisk"].map(config.LABELS["GeneticRisk"])
    df["Diagnosis label"] = df["Diagnosis"].map(config.LABELS["Diagnosis"])
    df["Age group"] = pd.cut(df["Age"], [g[2] - 0.5 for g in AGE_GROUPS] + [120],
                             labels=[g[1] for g in AGE_GROUPS])
    df["BMI category"] = pd.cut(df["BMI"], [c[2] for c in BMI_CATEGORIES] + [100],
                                labels=[c[1] for c in BMI_CATEGORIES], right=False)
    return df


df = get_data()
st.title("Liver Disease Risk Factors & Diagnosis Prediction")
st.caption("Smart Data Ukimet (data.gov.kz) dataset, 1 700 records. The audit shows "
           "the data are synthetic - conclusions are methodological, not clinical.")

st.sidebar.header("Filters")
genders = st.sidebar.multiselect("Gender", ["Male", "Female"], ["Male", "Female"])
age_range = st.sidebar.slider("Age", 20, 80, (20, 80))
bmi_cats = st.sidebar.multiselect("BMI category", [c[1] for c in BMI_CATEGORIES],
                                  [c[1] for c in BMI_CATEGORIES])
risks = st.sidebar.multiselect("Genetic risk", ["Low", "Medium", "High"],
                               ["Low", "Medium", "High"])
smoking = st.sidebar.radio("Smoking", ["All", "Smokers", "Non-smokers"])

view = df[df["Gender label"].isin(genders) & df["Age"].between(*age_range)
          & df["BMI category"].isin(bmi_cats) & df["Genetic risk"].isin(risks)]
if smoking == "Smokers":
    view = view[view["Smoking"] == 1]
elif smoking == "Non-smokers":
    view = view[view["Smoking"] == 0]

if len(view) < config.MIN_GROUP_SIZE:
    st.warning(f"Selection has fewer than {config.MIN_GROUP_SIZE} patients - "
               "statistics are suppressed (privacy rule).")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Risk factors", "Models", "Predict"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", f"{len(view):,}")
    c2.metric("Liver disease rate", f"{view['Diagnosis'].mean():.1%}")
    c3.metric("Mean alcohol, units/wk", f"{view['AlcoholConsumption'].mean():.1f}")
    c4.metric("Mean LFT score", f"{view['LiverFunctionTest'].mean():.1f}")

    left, right = st.columns(2)
    feature = left.selectbox("Numeric feature", config.NUMERIC_FEATURES, index=2)
    left.plotly_chart(px.histogram(view, x=feature, color="Diagnosis label", nbins=30,
                                   barmode="overlay", opacity=0.65,
                                   title=f"{feature} by diagnosis"),
                      use_container_width=True)
    rate = (view.groupby(["Age group", "Gender label"], observed=True)["Diagnosis"]
                .agg(n="size", rate="mean").reset_index())
    rate = suppress_small_groups(rate)
    right.plotly_chart(px.bar(rate, x="Age group", y="rate", color="Gender label",
                              barmode="group", title="Disease rate by age group and gender",
                              labels={"rate": "Disease rate"}),
                       use_container_width=True)
    st.plotly_chart(px.scatter(view, x="AlcoholConsumption", y="LiverFunctionTest",
                               color="Diagnosis label", opacity=0.6,
                               hover_data=["Age", "BMI", "Genetic risk"],
                               title="Alcohol vs liver function test"),
                    use_container_width=True)

with tab2:
    cat = st.selectbox("Group by", ["BMI category", "Genetic risk", "Smoking",
                                    "Diabetes", "Hypertension", "Gender label"])
    g = (view.groupby(cat, observed=True)["Diagnosis"].agg(n="size", rate="mean")
             .reset_index())
    g = suppress_small_groups(g)
    st.plotly_chart(px.bar(g, x=cat, y="rate", text="n",
                           title=f"Disease rate by {cat} (text = group size)"),
                    use_container_width=True)
    or_path = config.TABLE_DIR / "odds_ratios_adjusted.csv"
    if or_path.exists():
        st.subheader("Adjusted odds ratios (multivariable logistic regression, full data)")
        st.dataframe(pd.read_csv(or_path), use_container_width=True)
    st.subheader("SQL: smoking x genetic-risk profiles (lift vs overall rate)")
    try:
        from src.sql_queries import QUERIES
        st.dataframe(run_query(QUERIES["q08_risk_profiles_lift"]["sql"]),
                     use_container_width=True)
    except Exception as err:          # database not built yet
        st.info(f"Run `python main.py --all` to build the database ({err}).")

with tab3:
    comp = config.TABLE_DIR / "model_comparison.csv"
    if comp.exists():
        m = pd.read_csv(comp)
        st.dataframe(m[["model", "roc_auc", "pr_auc", "f1", "accuracy", "recall",
                        "brier", "cv_roc_auc_mean", "fit_time_s", "model_size_kb",
                        "predict_ms_per_1000"]], use_container_width=True)
        st.plotly_chart(px.bar(m, x="model", y=["roc_auc", "f1"], barmode="group",
                               title="Test ROC-AUC and F1"), use_container_width=True)
        for img in ["09_roc_pr_curves.png", "10_calibration_curves.png",
                    "12_permutation_importance.png"]:
            p = config.FIG_DIR / img
            if p.exists():
                st.image(str(p))
    else:
        st.info("Train the models first: `python main.py --all`.")

with tab4:
    st.subheader("Predict liver disease for a new patient")
    try:
        bundle = load_best_model()
    except FileNotFoundError as err:
        st.error(str(err))
        st.stop()
    cols = st.columns(2)
    values = {}
    for i, (feat, (question, cast, low, high)) in enumerate(QUESTIONS.items()):
        col = cols[i % 2]
        if cast is int and high - low <= 2:
            values[feat] = col.selectbox(question, list(range(low, high + 1)))
        elif cast is int:
            values[feat] = col.number_input(question, low, high, 50)
        else:
            default = float(df[feat].median())
            values[feat] = col.number_input(question, float(low), float(high), default)
    if st.button("Predict"):
        res = predict_patient(values, bundle)
        st.metric("Probability of liver disease", f"{res['probability']:.1%}",
                  help=f"Model: {res['model']}")
        st.write(f"Risk level: **{res['risk_level']}**")
        st.caption("Educational model trained on synthetic data - not a medical diagnosis.")

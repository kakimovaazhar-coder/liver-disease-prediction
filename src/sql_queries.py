# our sql queries for the analysis
# HAVING COUNT(*) >= 10 is used so small groups are not shown
from src import config
from src.database import run_query

MIN_N = config.MIN_GROUP_SIZE

QUERIES = {
    "q01_rate_by_gender_age": {
        "title": "Disease rate by gender and age group (JOIN x4, GROUP BY, HAVING)",
        "sql": f"""
SELECT g.gender_label AS gender, ag.label AS age_group,
       COUNT(*) AS n,
       ROUND(100.0 * AVG(d.has_liver_disease), 1) AS disease_rate_pct
FROM patient p
JOIN gender g     ON g.gender_id = p.gender_id
JOIN age_group ag ON ag.age_group_id = p.age_group_id
JOIN diagnosis d  ON d.patient_id = p.patient_id
GROUP BY g.gender_label, ag.age_group_id
HAVING COUNT(*) >= {MIN_N}
ORDER BY g.gender_label, ag.age_group_id;"""},

    "q02_rate_by_bmi_category": {
        "title": "Disease rate and mean lifestyle values by WHO BMI category (JOIN, GROUP BY, HAVING)",
        "sql": f"""
SELECT bc.label AS bmi_category, COUNT(*) AS n,
       ROUND(AVG(l.alcohol_units), 2)     AS mean_alcohol,
       ROUND(AVG(l.physical_activity), 2) AS mean_activity,
       ROUND(100.0 * AVG(d.has_liver_disease), 1) AS disease_rate_pct
FROM lifestyle l
JOIN bmi_category bc ON bc.bmi_category_id = l.bmi_category_id
JOIN diagnosis d     ON d.patient_id = l.patient_id
GROUP BY bc.bmi_category_id
HAVING COUNT(*) >= {MIN_N}
ORDER BY bc.bmi_category_id;"""},

    "q03_comorbidity_burden": {
        "title": "Disease rate by number of comorbidities (CTE over the many-to-many table, LEFT JOIN)",
        "sql": f"""
WITH burden AS (
    SELECT p.patient_id, COUNT(pc.comorbidity_id) AS n_comorbidities
    FROM patient p
    LEFT JOIN patient_comorbidity pc ON pc.patient_id = p.patient_id
    GROUP BY p.patient_id
)
SELECT b.n_comorbidities, COUNT(*) AS n,
       ROUND(100.0 * AVG(d.has_liver_disease), 1) AS disease_rate_pct
FROM burden b
JOIN diagnosis d ON d.patient_id = b.patient_id
GROUP BY b.n_comorbidities
HAVING COUNT(*) >= {MIN_N}
ORDER BY b.n_comorbidities;"""},

    "q04_heavy_drinkers_subquery": {
        "title": "Patients drinking more than the average healthy patient (scalar subquery)",
        "sql": """
SELECT CASE WHEN l.alcohol_units > (
            SELECT AVG(l2.alcohol_units)
            FROM lifestyle l2 JOIN diagnosis d2 ON d2.patient_id = l2.patient_id
            WHERE d2.has_liver_disease = 0)
       THEN 'above healthy mean' ELSE 'at/below healthy mean' END AS alcohol_level,
       COUNT(*) AS n,
       ROUND(100.0 * AVG(d.has_liver_disease), 1) AS disease_rate_pct
FROM lifestyle l
JOIN diagnosis d ON d.patient_id = l.patient_id
GROUP BY alcohol_level;"""},

    "q05_rank_age_groups": {
        "title": "Ranking of age groups by disease rate (CTE + RANK window function)",
        "sql": f"""
WITH rates AS (
    SELECT ag.label AS age_group, COUNT(*) AS n,
           AVG(d.has_liver_disease) AS rate
    FROM patient p
    JOIN age_group ag ON ag.age_group_id = p.age_group_id
    JOIN diagnosis d  ON d.patient_id = p.patient_id
    GROUP BY ag.age_group_id
    HAVING COUNT(*) >= {MIN_N}
)
SELECT age_group, n, ROUND(100 * rate, 1) AS disease_rate_pct,
       RANK() OVER (ORDER BY rate DESC) AS risk_rank
FROM rates
ORDER BY risk_rank;"""},

    "q06_lft_quintiles": {
        "title": "Liver function test quintiles (NTILE window) and disease rate",
        "sql": """
WITH q AS (
    SELECT c.patient_id, c.liver_function_test,
           NTILE(5) OVER (ORDER BY c.liver_function_test) AS lft_quintile
    FROM clinical_exam c
)
SELECT q.lft_quintile, COUNT(*) AS n,
       ROUND(MIN(q.liver_function_test), 1) AS lft_min,
       ROUND(MAX(q.liver_function_test), 1) AS lft_max,
       ROUND(100.0 * AVG(d.has_liver_disease), 1) AS disease_rate_pct
FROM q JOIN diagnosis d ON d.patient_id = q.patient_id
GROUP BY q.lft_quintile
ORDER BY q.lft_quintile;"""},

    "q07_alcohol_deciles_cumulative": {
        "title": "Alcohol deciles: cumulative share of all cases (NTILE + SUM() OVER running total)",
        "sql": """
WITH dec AS (
    SELECT l.patient_id, NTILE(10) OVER (ORDER BY l.alcohol_units) AS alcohol_decile,
           l.alcohol_units
    FROM lifestyle l
),
agg AS (
    SELECT dec.alcohol_decile, COUNT(*) AS n,
           ROUND(MAX(dec.alcohol_units), 1) AS alcohol_max,
           SUM(d.has_liver_disease) AS cases
    FROM dec JOIN diagnosis d ON d.patient_id = dec.patient_id
    GROUP BY dec.alcohol_decile
)
SELECT alcohol_decile, n, alcohol_max, cases,
       ROUND(100.0 * cases / n, 1) AS disease_rate_pct,
       ROUND(100.0 * SUM(cases) OVER (ORDER BY alcohol_decile)
             / SUM(cases) OVER (), 1) AS cumulative_case_share_pct
FROM agg
ORDER BY alcohol_decile;"""},

    "q08_risk_profiles_lift": {
        "title": "Smoking x genetic-risk profiles: rate and lift vs overall rate (two CTEs, CROSS JOIN)",
        "sql": f"""
WITH overall AS (
    SELECT AVG(has_liver_disease) AS base_rate FROM diagnosis
),
profiles AS (
    SELECT CASE l.smoking WHEN 1 THEN 'smoker' ELSE 'non-smoker' END AS smoking,
           gr.label AS genetic_risk, COUNT(*) AS n,
           AVG(d.has_liver_disease) AS rate
    FROM lifestyle l
    JOIN clinical_exam c       ON c.patient_id = l.patient_id
    JOIN genetic_risk_level gr ON gr.genetic_risk_id = c.genetic_risk_id
    JOIN diagnosis d           ON d.patient_id = l.patient_id
    GROUP BY l.smoking, gr.genetic_risk_id
    HAVING COUNT(*) >= {MIN_N}
)
SELECT p.smoking, p.genetic_risk, p.n,
       ROUND(100 * p.rate, 1) AS disease_rate_pct,
       ROUND(p.rate / o.base_rate, 2) AS lift
FROM profiles p CROSS JOIN overall o
ORDER BY lift DESC;"""},

    "q09_age_group_lag": {
        "title": "Change of disease rate between neighbouring age groups (LAG window)",
        "sql": """
WITH rates AS (
    SELECT ag.age_group_id, ag.label AS age_group,
           100.0 * AVG(d.has_liver_disease) AS rate
    FROM patient p
    JOIN age_group ag ON ag.age_group_id = p.age_group_id
    JOIN diagnosis d  ON d.patient_id = p.patient_id
    GROUP BY ag.age_group_id
)
SELECT age_group, ROUND(rate, 1) AS disease_rate_pct,
       ROUND(rate - LAG(rate) OVER (ORDER BY age_group_id), 1) AS change_vs_previous_pp
FROM rates
ORDER BY age_group_id;"""},

    "q10_lft_vs_gender_mean": {
        "title": "Liver function test relative to the gender mean (AVG() OVER PARTITION BY)",
        "sql": """
WITH dev AS (
    SELECT p.patient_id, g.gender_label,
           c.liver_function_test
             - AVG(c.liver_function_test) OVER (PARTITION BY p.gender_id) AS lft_dev
    FROM patient p
    JOIN gender g        ON g.gender_id = p.gender_id
    JOIN clinical_exam c ON c.patient_id = p.patient_id
)
SELECT dev.gender_label AS gender,
       CASE WHEN dev.lft_dev > 0 THEN 'above gender mean'
            ELSE 'below gender mean' END AS lft_position,
       COUNT(*) AS n,
       ROUND(100.0 * AVG(d.has_liver_disease), 1) AS disease_rate_pct
FROM dev JOIN diagnosis d ON d.patient_id = dev.patient_id
GROUP BY gender, lft_position
ORDER BY gender, lft_position;"""},

    "q11_patients_in_risk_top": {
        "title": "Share of diseased patients among the top-10% alcohol consumers within each gender (ROW_NUMBER + subquery in WHERE)",
        "sql": """
SELECT gender, COUNT(*) AS n,
       ROUND(100.0 * AVG(has_liver_disease), 1) AS disease_rate_pct
FROM (
    SELECT g.gender_label AS gender, d.has_liver_disease,
           ROW_NUMBER() OVER (PARTITION BY p.gender_id ORDER BY l.alcohol_units DESC) AS rn,
           COUNT(*)     OVER (PARTITION BY p.gender_id) AS total
    FROM patient p
    JOIN gender g    ON g.gender_id = p.gender_id
    JOIN lifestyle l ON l.patient_id = p.patient_id
    JOIN diagnosis d ON d.patient_id = p.patient_id
) t
WHERE rn <= total / 10
GROUP BY gender;"""},
}

EXPERIMENT_QUERY = {
    "q12_model_leaderboard": {
        "title": "Model leaderboard from the experiment table (window RANK over stored results)",
        "sql": """
SELECT model_name, ROUND(roc_auc, 4) AS roc_auc, ROUND(f1, 4) AS f1,
       ROUND(brier, 4) AS brier, ROUND(fit_time_s, 3) AS fit_time_s,
       ROUND(model_size_kb, 1) AS model_size_kb,
       RANK() OVER (ORDER BY roc_auc DESC) AS auc_rank
FROM experiment
WHERE run_id IN (SELECT MAX(run_id) FROM experiment GROUP BY model_name)
ORDER BY auc_rank;"""},
}


def export_sql_file() -> None:
    """Write all queries into sql/queries.sql so they can be run in any client."""
    parts = ["-- Analytical queries of the liver disease project (SQLite)\n"]
    for name, q in {**QUERIES, **EXPERIMENT_QUERY}.items():
        parts.append(f"\n-- {name}: {q['title']}\n{q['sql'].strip()}\n")
    (config.SQL_DIR / "queries.sql").write_text("".join(parts), encoding="utf-8")


def run_all_queries(verbose: bool = True, include_experiments: bool = False) -> dict:
    """Run every query, save each result as CSV and return {name: DataFrame}."""
    config.ensure_dirs()
    export_sql_file()
    selected = dict(QUERIES)
    if include_experiments:
        selected.update(EXPERIMENT_QUERY)
    results = {}
    for name, q in selected.items():
        try:
            df = run_query(q["sql"])
        except Exception as err:          # e.g. sql syntax error
            print(f"[ERROR] {name}: {err}")
            continue
        df.to_csv(config.TABLE_DIR / f"sql_{name}.csv", index=False)
        results[name] = df
        if verbose:
            print(f"\n{name} - {q['title']}")
            print(df.to_string(index=False))
    return results

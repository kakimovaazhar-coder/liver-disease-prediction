-- Normalised relational schema (3NF) for the liver disease dataset.
-- Dimension (lookup) tables
DROP TABLE IF EXISTS prediction;
DROP TABLE IF EXISTS experiment;
DROP TABLE IF EXISTS diagnosis;
DROP TABLE IF EXISTS patient_comorbidity;
DROP TABLE IF EXISTS clinical_exam;
DROP TABLE IF EXISTS lifestyle;
DROP TABLE IF EXISTS patient;
DROP TABLE IF EXISTS comorbidity;
DROP TABLE IF EXISTS genetic_risk_level;
DROP TABLE IF EXISTS bmi_category;
DROP TABLE IF EXISTS age_group;
DROP TABLE IF EXISTS gender;

CREATE TABLE gender (
    gender_id     INTEGER PRIMARY KEY,
    gender_label  TEXT NOT NULL UNIQUE
);

CREATE TABLE age_group (
    age_group_id  INTEGER PRIMARY KEY,
    label         TEXT NOT NULL UNIQUE,
    age_min       INTEGER NOT NULL,
    age_max       INTEGER NOT NULL
);

CREATE TABLE bmi_category (
    bmi_category_id INTEGER PRIMARY KEY,
    label           TEXT NOT NULL UNIQUE,   -- WHO classification
    bmi_min         REAL NOT NULL,
    bmi_max         REAL NOT NULL
);

CREATE TABLE genetic_risk_level (
    genetic_risk_id INTEGER PRIMARY KEY,    -- 0 / 1 / 2
    label           TEXT NOT NULL UNIQUE
);

CREATE TABLE comorbidity (
    comorbidity_id  INTEGER PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE
);

-- Entity tables
CREATE TABLE patient (
    patient_id    INTEGER PRIMARY KEY,
    pseudo_id     TEXT NOT NULL UNIQUE,     -- salted SHA-256, no real identifier
    age           INTEGER NOT NULL CHECK (age BETWEEN 0 AND 120),
    gender_id     INTEGER NOT NULL REFERENCES gender(gender_id),
    age_group_id  INTEGER NOT NULL REFERENCES age_group(age_group_id)
);

CREATE TABLE lifestyle (
    patient_id        INTEGER PRIMARY KEY REFERENCES patient(patient_id),
    bmi               REAL,
    bmi_category_id   INTEGER REFERENCES bmi_category(bmi_category_id),
    alcohol_units     REAL,                 -- units per week
    smoking           INTEGER NOT NULL CHECK (smoking IN (0, 1)),
    physical_activity REAL                  -- hours per week
);

CREATE TABLE clinical_exam (
    patient_id          INTEGER PRIMARY KEY REFERENCES patient(patient_id),
    genetic_risk_id     INTEGER NOT NULL REFERENCES genetic_risk_level(genetic_risk_id),
    liver_function_test REAL
);

-- many-to-many: a patient can have several chronic comorbidities
CREATE TABLE patient_comorbidity (
    patient_id      INTEGER NOT NULL REFERENCES patient(patient_id),
    comorbidity_id  INTEGER NOT NULL REFERENCES comorbidity(comorbidity_id),
    PRIMARY KEY (patient_id, comorbidity_id)
);

CREATE TABLE diagnosis (
    patient_id        INTEGER PRIMARY KEY REFERENCES patient(patient_id),
    has_liver_disease INTEGER NOT NULL CHECK (has_liver_disease IN (0, 1))
);

-- Experiment tracking (model comparison results and test-set predictions)
CREATE TABLE experiment (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT NOT NULL,
    roc_auc         REAL, pr_auc REAL, f1 REAL, accuracy REAL,
    precision_score REAL, recall REAL, brier REAL,
    cv_roc_auc_mean REAL, cv_roc_auc_std REAL,
    fit_time_s      REAL, predict_ms_per_1000 REAL, model_size_kb REAL,
    best_params     TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE prediction (
    run_id      INTEGER NOT NULL REFERENCES experiment(run_id),
    patient_id  INTEGER NOT NULL REFERENCES patient(patient_id),
    y_true      INTEGER NOT NULL,
    y_prob      REAL NOT NULL,
    y_pred      INTEGER NOT NULL,
    PRIMARY KEY (run_id, patient_id)
);

CREATE INDEX idx_patient_gender ON patient(gender_id);
CREATE INDEX idx_patient_agegrp ON patient(age_group_id);
CREATE INDEX idx_lifestyle_bmi  ON lifestyle(bmi_category_id);

# Liver Disease Risk Factors and Diagnosis Prediction

Финальный проект по дисциплине **«Programming for Data Analysis and Database»**
(Applied Artificial Intelligence, магистратура). **Тема 31**, кластер 6 «Public Health Open Data Analytics (Kazakhstan)».

**Команда (AAI-2603):** Какимова Ажар, Зулхаирова Молдир.

**Репозиторий:** <https://github.com/kakimovaazhar-coder/liver-disease-prediction>

## Цель

Оценить связь факторов образа жизни и клинических факторов с заболеванием печени
и построить интерпретируемую модель диагноза:

* статистические тесты (хи-квадрат, Манна–Уитни) и отношения шансов (odds ratios);
* сравнение моделей: Dummy (baseline) → Logistic Regression → Random Forest
  (+ HistGradientBoosting как расширение) со стратифицированной кросс-валидацией;
* метрики ROC-AUC, F1, калибровка (Brier score, reliability diagram);
* проверка происхождения данных (реальные или синтетические).

## Данные

| | |
|---|---|
| Источник | «Данные по заболеваниям печени», Smart Data Ukimet — <https://ashyq.data.gov.kz/dataset/magda-ds-8ec6e917-d808-494a-b84d-177dd857d7c4/details> |
| Прямая ссылка на файл | <http://s3.sdu.gov.kz/magda-datasets/Liver_disease_data.csv> |
| Файл в репозитории | `data/raw/Liver_disease_data.csv` (1 700 строк, 11 столбцов, 147 КБ) |
| Целевая переменная | `Diagnosis` (1 — заболевание печени, 0 — нет), 55 % / 45 % |

Признаки: `Age`, `Gender` (0 — муж., 1 — жен.), `BMI`, `AlcoholConsumption` (ед./нед.),
`Smoking`, `GeneticRisk` (0/1/2), `PhysicalActivity` (ч/нед.), `Diabetes`, `Hypertension`,
`LiverFunctionTest`.

> Объём набора (1 700 строк) меньше рекомендованных 500 тыс. — это свойство
> официального датасета темы 31. Работа с большим объёмом продемонстрирована
> отдельно: chunksize, оптимизация типов, Parquet и бенчмарк на стресс-копии
> из 1 000 000 строк (`src/benchmark.py`, только для замеров времени и памяти).

## Быстрый старт

Требуется Python 3.10+.

```bash
git clone https://github.com/kakimovaazhar-coder/liver-disease-prediction.git
cd liver-disease-prediction
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Полный пайплайн (≈1–2 мин): загрузка → аудит → БД → EDA → тесты → модели → SQL → бенчмарк

```bash
python main.py --all
```

Интерактивное консольное меню (в т.ч. пункт 9 — прогноз для нового пациента):

```bash
python main.py
```

Streamlit-дашборд с фильтрами (пол, возраст, ИМТ, генетический риск, курение) и формой прогноза:

```bash
streamlit run app/dashboard.py
```

**Google Colab:** загрузите папку проекта (или `git clone`), выполните
`!pip install -r requirements.txt` и `!python main.py --all`.

## Структура проекта

```
liver-disease-project/
├── main.py                  консольное меню и запуск всего пайплайна
├── app/dashboard.py         Streamlit-дашборд
├── src/
│   ├── config.py            пути и константы
│   ├── data_loader.py       скачивание, чтение чанками, оптимизация dtype, Parquet
│   ├── privacy.py           псевдонимизация (SHA-256 с солью), подавление групп < 10
│   ├── quality_audit.py     аудит качества + проверка «реальные/синтетические»
│   ├── database.py          нормализованная схема SQLite, загрузка, ER-диаграмма
│   ├── sql_queries.py       12 аналитических SQL-запросов → pandas
│   ├── eda.py               описательная статистика и визуализации
│   ├── stats_tests.py       хи-квадрат, Манна–Уитни, odds ratios, forest plot
│   ├── modeling.py          Dummy / LR / RF / HGB, GridSearchCV, метрики, калибровка
│   ├── benchmark.py         CSV vs Parquet, Pandas vs SQL (время, память)
│   └── predict.py           прогноз для нового пациента
├── sql/schema.sql           DDL схемы, sql/queries.sql — все запросы
├── data/raw/                исходный CSV;  data/processed/ — Parquet
├── docs/er_diagram.png      ER-диаграмма
├── results/figures/         графики (PNG + интерактивный HTML Plotly)
├── results/tables/          таблицы результатов (CSV/JSON/Parquet)
├── results/models/          best_model.joblib
├── report/                  письменный отчёт (.docx)
└── presentation/            презентация (.pptx)
```

## Основные результаты

| Модель | ROC-AUC (test) | F1 | Brier | CV ROC-AUC | Время обучения, с | Размер, КБ |
|---|---|---|---|---|---|---|
| Dummy (baseline) | 0.500 | 0.710 | 0.248 | 0.500 | 0.02 | 4 |
| Logistic Regression | 0.915 | 0.846 | 0.117 | 0.914 ± 0.013 | 0.03 | 5 |
| Random Forest | 0.958 | 0.903 | 0.094 | 0.940 ± 0.010 | 0.76 | 7 562 |
| HistGradientBoosting | **0.960** | **0.909** | **0.080** | **0.947 ± 0.009** | 0.15 | 115 |

* Сильнейшие факторы — `LiverFunctionTest` и `AlcoholConsumption`
  (скорректированный OR ≈ 4 на 1 SD), высокий генетический риск (OR ≈ 14.6),
  курение (OR ≈ 6.6), гипертония (OR ≈ 5.2); физическая активность — защитный фактор (OR ≈ 0.68).
* Аудит показал: данные **синтетические** — все 5 непрерывных признаков равномерно
  распределены (KS-тест p > 0.05), естественные связи (возраст–гипертония,
  ИМТ–диабет и др.) отсутствуют, у float-значений 14–15 знаков после запятой.
  Выводы носят методический, а не клинический характер.

Подробности — в `report/` и `presentation/`.

## Соответствие требованиям

| № | Требование | Где реализовано |
|---|---|---|
| 1 | ≥ 5 функций | все модули `src/*.py` (50+ функций) |
| 2 | list / dict | `MENU`, `STEPS`, `QUERIES`, `QUESTIONS`, `VALID_RANGES`, результаты в dict |
| 3 | if / elif / else | `main.py` (меню), `predict.risk_level`, `database.age_group_id` |
| 4 | for / while | меню `while True`, повторный ввод в `predict.ask_value`, циклы по чанкам |
| 5 | Работа с файлами | CSV → Parquet, JSON-отчёты, `results/run_log.json` (состояние программы) |
| 6 | try / except | скачивание, ввод пользователя, отсутствие модели/файла, ошибки SQL и шагов меню |
| 7 | Интерфейс | консольное меню + Streamlit-дашборд с фильтрами |
| 8 | Модульность | 11 модулей в `src/`, стандартные `pathlib`, `json`, `datetime`, `hashlib`, `sqlite3` |
| 9 | Сравнение подходов | Dummy vs LR vs RF vs HGB; CSV vs Parquet; Pandas vs SQL (метрики, время, память) |
| 10 | БД и SQL | SQLite 3NF, ER-диаграмма, 12 запросов (JOIN, HAVING, подзапросы, CTE, RANK, NTILE, LAG, SUM/AVG OVER) |
| 11 | Объём данных | chunksize, downcast (-74 % памяти), Parquet; бенчмарк на 1 млн строк |

Кластер 6: псевдонимизация (salted SHA-256, соль хранится вне репозитория),
подавление групп < 10 пациентов (`HAVING COUNT(*) >= 10`, `suppress_small_groups`),
аудит качества, нормализованная схема с M:N (`patient_comorbidity`),
проверка происхождения данных. Временные ряды, Каплан–Мейер и нормирование на 100 тыс.
населения неприменимы: в наборе нет дат и регионов (см. ограничения в отчёте).

## Вклад участников

| Участник | Вклад |
|---|---|
| **Какимова Ажар** | загрузка данных (chunksize, dtype, Parquet), псевдонимизация, аудит качества; схема SQLite, ER-диаграмма, SQL-запросы; бенчмарк CSV/Parquet и Pandas/SQL; Streamlit-дашборд; разделы отчёта «Данные», «БД и SQL» |
| **Зулхаирова Молдир** | EDA и визуализации; проверка синтетического происхождения; статистические тесты и odds ratios; обучение и сравнение моделей, калибровка, permutation importance, error analysis; консольный прогноз; разделы отчёта «Методы», «Результаты» |
| Совместно | постановка задачи, консольное меню, выводы, презентация |

## Лицензия и этика

Данные — открытые данные Smart Data Ukimet. Модель обучена на синтетических данных
и **не предназначена для медицинской диагностики**.

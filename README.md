# Home Credit Default Risk — Data Preparation

This README introduces `Data_Prep.py`, the data-preparation script that turns
the raw Home Credit application files into model-ready tables. It implements
the decisions recorded in `EDA_Home_Credit.ipynb`, using `data_map.md` and
`feature_report.md` as the supporting references for those decisions.

- EDA notebook: `EDA_Home_Credit.ipynb`
- Supporting references: `data_map.md`, `feature_report.md`
- Script: `Data_Prep.py`

## 1. What the script does

`Data_Prep.py` is organized as small, reusable functions, not a single
top-to-bottom script, so each step can be called, tested, or reused on its
own, and so the exact same function runs on both train and test. Every
function's docstring and inline comments name the EDA section or question
the transformation implements. The sections below group the functions by
what they do. The EDA column names each decision.

### Cleaning (recoding sentinels and fixing data types)

| Function | What it does | EDA decision |
|---|---|---|
| `recode_days_employed_sentinel` | Recodes `DAYS_EMPLOYED == 365243` to missing and adds `DAYS_EMPLOYED_SENTINEL_FLAG` | 4.3, Q5 — the value is a placeholder for "not currently employed," not a real day count |
| `fix_binary_flag_dtypes` | Converts `FLAG_OWN_CAR` / `FLAG_OWN_REALTY` from `Y`/`N` text to `0`/`1` | 4.3 — the integrity check found these were the two "nonbinary flags" |
| `recode_previous_application_sentinels` | Recodes the same `365243` sentinel in five `previous_application` date columns | data_map.md / 4.3 — same sentinel, different table |

### Demographic features

| Function | What it does | EDA decision |
|---|---|---|
| `engineer_demographic_features` | Converts negative "days relative to application" fields into positive years: `AGE_YEARS`, `EMPLOYMENT_YEARS`, `REGISTRATION_YEARS`, `ID_PUBLISH_YEARS`, `LAST_PHONE_CHANGE_YEARS` | Q5 built `EMPLOYMENT_YEARS` this way; `AGE_YEARS` is the same transform applied to `DAYS_BIRTH`, requested explicitly for this script |

`AGE_YEARS` is engineered but **not** used as an ordinary baseline feature.
The EDA results section explicitly excludes "ordinary use of age" because it
is a sensitive characteristic — see "High-governance features," below.

### Missing-data indicators

| Function | What it does | EDA decision |
|---|---|---|
| `add_missing_indicators` | Flags for `EXT_SOURCE_1/2/3`, `OCCUPATION_TYPE`, and enquiry columns; fills `OCCUPATION_TYPE` with `"Not_working_or_reported"` and `OWN_CAR_AGE` with `0` | 4.2 — missingness itself was related to the target (e.g. missing enquiry data: 10.3% vs 7.7%), so it is flagged rather than imputed away |
| `handle_building_attributes` | Drops the 59–70%-missing building-attribute columns, replacing them with one `HAS_BUILDING_INFO` flag | 4.2 |

### Financial ratios and other engineered features

| Function | What it does | EDA decision |
|---|---|---|
| `add_financial_ratios` | Builds `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO` (capped at the training 99th percentile), and `INCOME_PER_FAMILY_MEMBER` | Q2 ("cap the ratios at a high percentile") and Q7 ("income per family member") |
| `process_enquiry_features` | Drops the hour/day/week enquiry columns, caps `AMT_REQ_CREDIT_BUREAU_YEAR` at 5, adds `ENQUIRY_COUNT_TOTAL` | Q6 — the short windows were almost always zero and too sparse to interpret |

### Binned variables and interaction terms

| Function | What it does | EDA decision |
|---|---|---|
| `add_binned_features` | `EMPLOYMENT_YEARS_BAND` (0–1y, 1–3y, 3–5y, 5–10y, 10y+) and `AGE_BAND` | Q5 reproduces the tenure bands with the clearest single-variable pattern in the notebook (11.1% down to 5.2%) |
| `add_interaction_features` | `EXT_SOURCE_MEAN/MIN/MAX/COUNT_AVAILABLE` and `INCOME_TYPE_X_EMPLOYMENT_BAND` | Q1 (combining the three external scores) and Q3+Q5 (an income-type × tenure-band interaction) |
| `drop_low_value_flags` | Drops `FLAG_MOBIL`, `FLAG_CONT_MOBILE`, and any `FLAG_DOCUMENT_*` supported by fewer than 500 training applicants | Q4 — these flags were near-constant or too rare to interpret |

### Supplementary tables (optional, as discussed in the EDA)

The EDA notebook deliberately did not join the supplementary tables (to
avoid duplicating application rows before the relationships were
understood). This script implements that join as an **optional** step:

| Function | Table | EDA decision |
|---|---|---|
| `process_bureau_and_balance` | `bureau.csv` + `bureau_balance.csv` | feature_report's "prior delinquency" and "current debt burden" families; joins on the true `SK_ID_BUREAU` key per data_map.md's dictionary-naming note |
| `process_previous_application` | `previous_application.csv` | feature_report's "credit-history depth" family; sentinels recoded first |
| `process_pos_cash_balance` | `POS_CASH_balance.csv` | feature_report's "payment reliability" family |
| `process_credit_card_balance` | `credit_card_balance.csv` | same family; lowest coverage of any supplementary table (28.3%) |
| `process_installments_payments` | `installments_payments.csv` | adds `PAID_LATE` and `PAYMENT_SHORTFALL` |

Each function aggregates its table to **one row per `SK_ID_CURR`** and keeps
only records dated at or before the application (`DAYS_* <= 0`,
`MONTHS_BALANCE <= 0`), per the temporal-direction rule from EDA section
4.3. `build_supplementary_features(data_dir)` loads and aggregates whichever
of the five files are present (any missing file is simply skipped), and
`join_supplementary_features` left-joins the results onto the application
table, filling each table's `HAS_<table>` coverage flag and count columns
with `0` for applicants absent from that table (continuous summary columns
are left as `NaN` — the `HAS_*` flag, not a fabricated average, is what
tells the model why a value is missing).

### High-governance features

`feature_report.md` and EDA Q3/Q5/Q7/Q8 found real differences in several
fields that also carry legal or fairness risk (they can act as proxies for
protected or socioeconomic characteristics): `NAME_HOUSING_TYPE`,
`NAME_FAMILY_STATUS`, `NAME_EDUCATION_TYPE`, `OCCUPATION_TYPE`,
`ORGANIZATION_TYPE`, the region-rating fields, `CODE_GENDER`, and
`AGE_YEARS`. This script does **not** drop them — the EDA's Q9 plan calls
for a comparison model that needs them present — but lists them in
`HIGH_GOVERNANCE_FEATURES` so the modeling notebook can build a baseline
model without them and a second model that adds them, and compare the two,
as the EDA results section recommends.

### EDA decisions intentionally NOT implemented

| Decision | Why it is not automated here |
|---|---|
| Full numeric-distribution drift check between train and test (EDA 4.4 only compared categorical levels) | Deciding what counts as "too much drift" in a numeric column is a modeling-stage judgment call (it depends on the model and the metric), not a fixed data-prep rule. The script does check the things that have one clear right answer — identical columns and one row per application — see Section 2. |
| Blanket mean/median imputation of any remaining missing values | EDA 4.2 found that missingness itself is often informative for exactly the thin-file applicants this project cares about (e.g. a missing bureau enquiry likely means "no bureau record," not a value that happened not to be recorded). Imputing it away would hide that signal, so this script keeps a missing-indicator flag wherever the EDA found one mattered and leaves any further imputation to the model being trained. |

## 2. Train/test consistency

### Parameters learned from training data only

`fit_preprocessing_params(train_df_raw)` is the single place where anything
is "computed from training data" — it never looks at test data. It returns
one `params` dictionary containing:

| Parameter | What it stores | Computed by |
|---|---|---|
| `flags_to_drop` | Which `FLAG_*` columns are too rare/constant to keep | `fit_flags_to_drop` |
| `ratio_caps` | The 99th-percentile cap for `CREDIT_INCOME_RATIO` and `ANNUITY_INCOME_RATIO` | `fit_ratio_caps` |
| `rare_category_levels` | Which levels of `OCCUPATION_TYPE`, `ORGANIZATION_TYPE`, `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`, `CODE_GENDER`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`, and `NAME_TYPE_SUITE` are common enough (≥ 500 training applications) to keep as their own level | `fit_rare_category_levels` |

### How those parameters are reused on test

`prepare_dataset(df, params, is_train=...)` is the one function called for
**both** train and test. `is_train` only controls whether `TARGET` is kept;
every cleaning and engineering step runs in the exact same order either way
(see `clean_and_engineer_application_features`), using only the values
already stored in `params` — nothing is ever recomputed from the test data.
Concretely:

- `drop_low_value_flags(df, params["flags_to_drop"])` drops the same
  columns from test that were decided on using train's counts.
- `add_financial_ratios(df, params["ratio_caps"])` clips test's ratios at
  the exact cap learned from train, even if test contains a more extreme
  value.
- `collapse_rare_categories(df, params["rare_category_levels"])` maps any
  test-only category level — including one never seen in training — to
  `"Other"`, the same way a rare training-only level is collapsed.

Use `save_params` / `load_params` to persist the fitted dictionary (e.g. to
a `.pkl` file) so a later scoring run does not need to re-fit on train at
all.

### Output of the consistency checks

`run_integrity_checks(train_df, test_df)` runs both required checks and
prints the result. On a schema-accurate synthetic run (see Section 3 for
why synthetic data was used), it reports:

```
Train and test share 71 columns (TARGET excluded): OK
train: 2000 rows, 2000 unique SK_ID_CURR -- one row per application: OK
test: 600 rows, 600 unique SK_ID_CURR -- one row per application: OK
```

`assert_consistent_columns` (called internally) raises an error naming any
column found only in train or only in test, and raises if `TARGET` is
missing from train or present in test. `check_application_grain` raises if
the row count ever stops matching the number of unique `SK_ID_CURR` values
— exactly the failure mode a careless one-to-many join against a
supplementary table would cause (see data_map.md's "Implementation notes").

## 3. How to run, inputs and outputs

### Run it standalone (smoke test)

```bash
python Data_Prep.py
```

This builds a **schema-accurate synthetic** version of
`application_train`/`application_test` — the real column names
(122 columns in train, 121 in test, per `data_map.md`) filled with randomly
generated values — and runs the full pipeline on it. It is useful for
verifying the script still works without needing the real (very large) CSV
files on hand, and it is what produced the example dimensions quoted below.

### Use it as a module, on the real data

```python
import pandas as pd
from Data_Prep import (
    fit_preprocessing_params, prepare_dataset,
    build_supplementary_features, run_integrity_checks,
    save_prepared_datasets,
)

# Run this example from the folder that contains the CSV files.
DATA_DIR = "."

train_raw = pd.read_csv(f"{DATA_DIR}/application_train.csv")
test_raw = pd.read_csv(f"{DATA_DIR}/application_test.csv")

# Step 1: learn every threshold/cap/category list from train only.
params = fit_preprocessing_params(train_raw)

# Step 2: (optional) aggregate the supplementary tables once, reused for
# both train and test since the aggregation does not use TARGET.
supplementary = build_supplementary_features(DATA_DIR)

# Step 3: apply the identical pipeline to both tables.
train_ready = prepare_dataset(train_raw, params, is_train=True,
                               supplementary_tables=supplementary)
test_ready = prepare_dataset(test_raw, params, is_train=False,
                              supplementary_tables=supplementary)

# Step 4: verify the required consistency properties.
run_integrity_checks(train_ready, test_ready)

# Step 5: write the prepared tables out for the modeling notebook.
save_prepared_datasets(train_ready, test_ready, f"{DATA_DIR}/prepared")
```

### Inputs

| File | Required? | Grain / key |
|---|---|---|
| `application_train.csv` | Yes | One application per row; key `SK_ID_CURR`; 307,511 rows × 122 columns |
| `application_test.csv` | Yes | One application per row; key `SK_ID_CURR`; 48,744 rows × 121 columns |
| `bureau.csv`, `bureau_balance.csv` | Optional | Keyed by `SK_ID_BUREAU`; joined to applications via `SK_ID_CURR` |
| `previous_application.csv` | Optional | Keyed by `SK_ID_PREV`; joined via `SK_ID_CURR` |
| `POS_CASH_balance.csv`, `credit_card_balance.csv` | Optional | Monthly snapshots keyed by (`SK_ID_PREV`, `MONTHS_BALANCE`); joined via `SK_ID_CURR` |
| `installments_payments.csv` | Optional | Payment events keyed by `SK_ID_PREV`; joined via `SK_ID_CURR` |

The application-file row and column counts above were verified against the
CSV files in this folder. A supplementary file that is absent from
`supplementary_dir` is skipped; therefore its associated feature and
`HAS_*` columns are not created. When a supplementary file is supplied,
applicants without a matching record receive `0` in its `HAS_*` and count
features.

### Outputs

`prepare_dataset` returns an in-memory `pandas.DataFrame`; `save_prepared_datasets`
writes it to `application_train_prepared.csv` / `application_test_prepared.csv`.
Exact output dimensions depend on which optional supplementary tables are
supplied. The application-only dimensions below were verified against the
real CSVs. The synthetic smoke test is useful only for a quick code check:
its values and category frequencies can yield a different number of retained
columns, so its output shapes must not be used as real-data results.

| Stage | Train shape | Test shape |
|---|---|---|
| Raw input | (307,511, 122) | (48,744, 121) |
| After `prepare_dataset`, application-only | (307,511, 84) | (48,744, 83) |
| After `prepare_dataset`, with all 5 supplementary tables joined | (307,511, 113) | (48,744, 112) |

(Train's column count is always one more than test's: `TARGET`.)


"""
Data_Prep.py
============

Reusable data-preparation functions for the Home Credit Default Risk project.

Every transformation below implements a decision recorded in the EDA notebook
(EDA_Home_Credit.ipynb). Each function's docstring/comments cite the EDA
section the decision came from (e.g. "EDA 4.3", "EDA Q2") so the modeling
code and the EDA stay traceable to each other.

Design principles used throughout this file
--------------------------------------------
1. Every function is a small transformation that takes a
   DataFrame (and sometimes a `params` dict) and returns a DataFrame. This
   makes each step independently testable and reusable on train, test, or
   any future batch of applications.
2. Nothing that could leak information from the target or from the test set
   is learned on the fly. Any threshold, percentile, median, or category
   grouping that must be "looked up" at prediction time is computed ONCE
   from the training data by `fit_preprocessing_params`, stored in a plain
   dict (`params`), and then simply *applied* (never re-computed) when the
   same functions are called on test data.
3. `prepare_dataset` is the single entry point used for both train and test;
   passing `is_train=True/False` only changes whether TARGET is kept and
   whether a few train-only lines (e.g. computing params) run. The sequence
   of transformations applied to every row is otherwise identical, which is
   what guarantees train and test end up with the same columns.

Usage
-----
    params = fit_preprocessing_params(application_train_raw)
    train_ready = prepare_dataset(application_train_raw, params, is_train=True,
                                   supplementary_dir="Home Credit Data")
    test_ready  = prepare_dataset(application_test_raw,  params, is_train=False,
                                   supplementary_dir="Home Credit Data")
    run_integrity_checks(train_ready, test_ready)
    save_prepared_datasets(train_ready, test_ready, "Home Credit Data/prepared")

See `README (1).md` for the full walk-through (what each step does, which EDA
decision it implements, and how to run this file), and run `python Data_Prep.py` directly for a self-contained
smoke test on schema-accurate synthetic data.

EDA decisions intentionally NOT implemented here (see `README (1).md` for the
full explanation of each):
  - A full numeric-distribution drift check between train and test
    (EDA 4.4 compared only categorical levels). Reason: deciding what
    counts as "too much drift" is a modeling-stage judgment call, not a
    fixed data-prep rule, so it is left to the modeling notebook.
  - Blanket mean/median imputation of any remaining missing values.
    Reason: EDA 4.2 found that missingness is often informative (e.g. a
    missing bureau enquiry likely means "no bureau record"), so this
    script keeps missing indicators and leaves the final imputation
    choice (if any) to the model being trained.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants tied to specific EDA decisions
# ---------------------------------------------------------------------------

# EDA 4.3 / Q5: DAYS_EMPLOYED uses this value as a "not applicable / unknown"
# sentinel instead of a real day count. The data map flags the same sentinel
# in five previous_application date columns.
DAYS_EMPLOYED_SENTINEL = 365243

# EDA Q4 / Q3: the EDA used a minimum-of-500-applications rule everywhere it
# reported a category's payment-difficulty rate, to avoid over-reading tiny
# groups. We reuse the same threshold to decide which categorical levels and
# which document flags are "common enough to trust" versus "collapse/drop".
RARE_CATEGORY_MIN_COUNT = 500

# EDA Q2: "Cap the ratios at a high percentile" to limit the extreme values
# found in the credit/annuity-to-income ratios (up to 84.7x and 1.88x income).
RATIO_CAP_PERCENTILE = 0.99

# EDA 4.2: these building-attribute columns were 59-70% missing in training
# data and repeated in three near-identical versions (_AVG, _MODE, _MEDI).
# Decision: drop the raw columns and keep a single coverage indicator instead.
BUILDING_ATTRIBUTE_PREFIXES = (
    "APARTMENTS", "BASEMENTAREA", "YEARS_BEGINEXPLUATATION", "YEARS_BUILD",
    "COMMONAREA", "ELEVATORS", "ENTRANCES", "FLOORSMAX", "FLOORSMIN",
    "LANDAREA", "LIVINGAPARTMENTS", "LIVINGAREA", "NONLIVINGAPARTMENTS",
    "NONLIVINGAREA", "TOTALAREA_MODE", "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE",
)

# EDA Q4: flags with essentially no variation, or supported by too few
# applicants to interpret, were dropped from the modeling feature set.
ALWAYS_DROP_FLAGS = ("FLAG_MOBIL", "FLAG_CONT_MOBILE")

# EDA Q6: the hour/day/week enquiry windows were almost always zero and had
# only a handful of applicants in the higher counts, so they were dropped.
# Month and quarter were kept as-is; the year window was kept and capped.
ENQUIRY_COLUMNS_DROPPED = (
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
)
ENQUIRY_COLUMNS_KEPT = (
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
)
ENQUIRY_YEAR_CAP = 5

# EDA feature_report / Q3, Q5, Q7, Q8: fields that showed real differences in
# the target rate but were flagged as high governance risk (legal proxy or
# protected-characteristic concerns). We do NOT drop them here -- the task
# asks for a full engineered feature set -- but we tag them so the modeling
# notebook can build the "baseline" and "baseline + governance" comparison
# models the EDA results section calls for.
HIGH_GOVERNANCE_FEATURES = (
    "NAME_HOUSING_TYPE", "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE",
    "OCCUPATION_TYPE", "ORGANIZATION_TYPE",
    "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
    "REGION_POPULATION_RELATIVE", "CODE_GENDER", "AGE_YEARS",
)


# ---------------------------------------------------------------------------
# 1. Sentinel recoding and dtype fixes (EDA 4.3)
# ---------------------------------------------------------------------------

def recode_days_employed_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    """Recode the DAYS_EMPLOYED placeholder value to a true missing value.

    EDA decision (section 4.3 and Q5): DAYS_EMPLOYED == 365243 affected
    55,374 training rows (18.0%) and is not a plausible number of days
    worked; it is a sentinel, most likely for applicants who are not
    currently employed (it lines up with ORGANIZATION_TYPE == 'XNA').
    Leaving it in would make those applicants look like the most senior
    employees in the file. We recode it to NaN and keep a flag so the
    model can still use "no employment date" as its own group.
    """
    df = df.copy()
    sentinel_mask = df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
    # EDA 4.3: keep an explicit indicator -- this is the "not employed /
    # no employment date" group, not a random missing value.
    df["DAYS_EMPLOYED_SENTINEL_FLAG"] = sentinel_mask.astype(int)
    df.loc[sentinel_mask, "DAYS_EMPLOYED"] = np.nan
    return df


def fix_binary_flag_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the two Y/N flags found during integrity checks to 0/1.

    EDA decision (section 4.3): FLAG_OWN_CAR and FLAG_OWN_REALTY were the
    two "nonbinary flags" found by the integrity check -- they store 'Y'/'N'
    text while every other FLAG_ column in the file is already 0/1. We
    convert them so all flag columns share one representation.
    """
    df = df.copy()
    for col in ("FLAG_OWN_CAR", "FLAG_OWN_REALTY"):
        # Check by content ("Y"/"N" values present) rather than by dtype:
        # pandas may store these as plain object strings or as a nullable
        # StringDtype depending on how the file was read, and we want both
        # cases converted the same way.
        if col in df.columns and df[col].isin(["Y", "N"]).any():
            df[col] = df[col].map({"Y": 1, "N": 0}).astype("float")
    return df


def recode_previous_application_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Recode the 365243 sentinel in previous_application date columns.

    EDA decision (data_map.md, carried into section 4.3's temporal-direction
    discussion): DAYS_FIRST_DRAWING, DAYS_FIRST_DUE, DAYS_LAST_DUE_1ST_VERSION,
    DAYS_LAST_DUE, and DAYS_TERMINATION contain the same 365243 sentinel as
    DAYS_EMPLOYED and must be treated as missing before any timing feature
    is built from them.
    """
    df = df.copy()
    sentinel_cols = [
        "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE", "DAYS_TERMINATION",
    ]
    for col in sentinel_cols:
        if col in df.columns:
            df.loc[df[col] == DAYS_EMPLOYED_SENTINEL, col] = np.nan
    return df


# ---------------------------------------------------------------------------
# 2. Demographic features: convert negative "days relative to application"
#    fields into positive, human-readable years (requested explicitly, and
#    consistent with how EDA 4.3 / Q5 handled DAYS_EMPLOYED).
# ---------------------------------------------------------------------------

def engineer_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn DAYS_* fields (negative, relative to the application) into
    positive "years" features that are easier to interpret and to bin.

    EDA linkage:
      - EMPLOYMENT_YEARS implements the tenure variable used in Q5 ("flip
        the sign to get tenure in years"), built AFTER the sentinel in
        DAYS_EMPLOYED has been recoded to NaN by recode_days_employed_sentinel.
      - AGE_YEARS is the same kind of transform applied to DAYS_BIRTH. The
        EDA results section explicitly excludes "ordinary use of age" from
        the baseline feature set (it is a sensitive characteristic), so
        AGE_YEARS is tagged in HIGH_GOVERNANCE_FEATURES above rather than
        dropped outright -- it is available for a documented, separately
        reviewed comparison model only, not the default baseline.
      - DAYS_REGISTRATION, DAYS_ID_PUBLISH and DAYS_LAST_PHONE_CHANGE are
        converted the same way for consistency; they were part of the
        "residential and identity/contact stability" family in the feature
        report but were not tested individually in the EDA questions, so
        treat them as exploratory additions rather than validated signals.
    """
    df = df.copy()
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    if "DAYS_EMPLOYED" in df.columns:
        # Sentinel must already be NaN by this point (see
        # recode_days_employed_sentinel), so dividing is safe.
        df["EMPLOYMENT_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25
    for raw_col, new_col in (
        ("DAYS_REGISTRATION", "REGISTRATION_YEARS"),
        ("DAYS_ID_PUBLISH", "ID_PUBLISH_YEARS"),
        ("DAYS_LAST_PHONE_CHANGE", "LAST_PHONE_CHANGE_YEARS"),
    ):
        if raw_col in df.columns:
            df[new_col] = -df[raw_col] / 365.25
    return df


# ---------------------------------------------------------------------------
# 3. Missing-data indicators (EDA 4.2: "missingness can be informative")
# ---------------------------------------------------------------------------

def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add "is missing" flags for fields where the EDA found the missing
    rate itself was related to the target, rather than being random noise.

    EDA decision (section 4.2 and Q1/Q6):
      - EXT_SOURCE_1/2/3: availability differs a lot (44%% / 99.8%% / 80.2%%)
        and the EDA explicitly decided NOT to mean-fill these, so a model
        can see whether a score was available at all.
      - OCCUPATION_TYPE: missing for 31%% of applicants, with a *lower*
        target rate than when it is reported (this is the "not working"
        group, not an unknown-risk group).
      - AMT_REQ_CREDIT_BUREAU_* (enquiry fields): missing for 13.5%% of
        applicants, and that group has a higher rate (10.3%% vs 7.7%%) --
        likely applicants with no bureau record at all.
      - OWN_CAR_AGE: missing exactly for applicants with FLAG_OWN_CAR == 0;
        this is "not applicable", not an unknown value, so we flag it and
        fill it with 0 rather than leaving a NaN that looks like noise.
    """
    df = df.copy()

    for col in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        if col in df.columns:
            df[f"{col}_MISSING"] = df[col].isna().astype(int)

    if "OCCUPATION_TYPE" in df.columns:
        df["OCCUPATION_TYPE_MISSING"] = df["OCCUPATION_TYPE"].isna().astype(int)
        # EDA Q3 decision: treat missing occupation as its own category
        # ("not working or not reported"), not as an unknown-risk value.
        df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna("Not_working_or_reported")

    enquiry_cols = [c for c in df.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")]
    if enquiry_cols:
        # A single indicator covers all six windows: EDA 4.2 found they are
        # missing together (same 13.5%% share) because the applicant simply
        # has no bureau enquiry record.
        df["NO_ENQUIRY_DATA"] = df[enquiry_cols[0]].isna().astype(int)

    if "OWN_CAR_AGE" in df.columns:
        df["OWN_CAR_AGE_MISSING"] = df["OWN_CAR_AGE"].isna().astype(int)
        # EDA 4.2: this missingness is "no car", not an unknown value, so a
        # 0 fill (paired with FLAG_OWN_CAR) is the correct "not applicable"
        # encoding rather than a median/mean imputation.
        df["OWN_CAR_AGE"] = df["OWN_CAR_AGE"].fillna(0)

    return df


def handle_building_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the high-missingness building-attribute columns and replace
    them with a single coverage indicator.

    EDA decision (section 4.2): the 20 most-incomplete columns are all
    building/apartment attributes (59-70%% missing), repeated in three
    near-identical versions (_AVG, _MODE, _MEDI). Decision: "drop or
    collapse the building columns... or use a single 'has building info'
    indicator." We keep the lighter-weight option (one indicator, drop the
    raw columns) so train and test stay easy to keep in sync.
    """
    df = df.copy()
    building_cols = [
        c for c in df.columns
        if c.startswith(BUILDING_ATTRIBUTE_PREFIXES)
    ]
    if building_cols:
        # A row "has building info" if at least one of the attributes is
        # populated (they are filled in together from the same source).
        df["HAS_BUILDING_INFO"] = df[building_cols].notna().any(axis=1).astype(int)
        df = df.drop(columns=building_cols)
    return df


# ---------------------------------------------------------------------------
# 4. Dropping low-value flags (EDA Q4)
# ---------------------------------------------------------------------------

def fit_flags_to_drop(train_df: pd.DataFrame) -> list:
    """Decide, from TRAINING data only, which FLAG_ columns carry too
    little information to keep.

    EDA decision (Q4): FLAG_MOBIL and FLAG_CONT_MOBILE are nearly constant
    (true for all but 1 and 574 applicants respectively), and several
    FLAG_DOCUMENT_* columns apply to only a handful of people (e.g. 13, 7,
    or 2 applicants), which makes their observed rates noise rather than
    signal. We use the same RARE_CATEGORY_MIN_COUNT=500 threshold the EDA
    used everywhere else to decide which document flags are "common
    enough" to keep; everything rarer is dropped. This decision is made
    once on the training counts and reused on test so both datasets drop
    exactly the same columns.
    """
    flag_cols = [c for c in train_df.columns if c.startswith("FLAG_")]
    drop_list = list(ALWAYS_DROP_FLAGS)
    for col in flag_cols:
        if col in drop_list:
            continue
        # A flag is "too rare to use" if its minority class (the less
        # common of 0/1) has fewer than RARE_CATEGORY_MIN_COUNT applicants.
        counts = train_df[col].value_counts(dropna=True)
        if len(counts) < 2 or counts.min() < RARE_CATEGORY_MIN_COUNT:
            drop_list.append(col)
    return sorted(set(drop_list))


def drop_low_value_flags(df: pd.DataFrame, flags_to_drop: Iterable[str]) -> pd.DataFrame:
    """Apply the drop list computed by fit_flags_to_drop (EDA Q4)."""
    cols_present = [c for c in flags_to_drop if c in df.columns]
    return df.drop(columns=cols_present)


# ---------------------------------------------------------------------------
# 5. Enquiry (recent credit-seeking) features (EDA Q6)
# ---------------------------------------------------------------------------

def process_enquiry_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cap the yearly bureau-enquiry count and drop the too-sparse windows.

    EDA decision (Q6): the hour/day/week enquiry windows are almost always
    zero and the few non-zero counts come from only a handful of
    applicants, so their observed rates are noise -- drop them. The year
    window showed a gentle, believable rise in the target rate up to "5 or
    more" enquiries, so it is kept as a candidate feature, capped at 5 the
    same way the EDA plot grouped it. Month and quarter are kept as-is.
    We also add ENQUIRY_COUNT_TOTAL, the sum across the three kept windows,
    reflecting the EDA's "possibly with a total across windows" note.
    """
    df = df.copy()
    df = df.drop(columns=[c for c in ENQUIRY_COLUMNS_DROPPED if c in df.columns])
    if "AMT_REQ_CREDIT_BUREAU_YEAR" in df.columns:
        df["AMT_REQ_CREDIT_BUREAU_YEAR"] = df["AMT_REQ_CREDIT_BUREAU_YEAR"].clip(
            upper=ENQUIRY_YEAR_CAP
        )
    kept_present = [c for c in ENQUIRY_COLUMNS_KEPT if c in df.columns]
    if kept_present:
        df["ENQUIRY_COUNT_TOTAL"] = df[kept_present].sum(axis=1, skipna=True)
    return df


# ---------------------------------------------------------------------------
# 6. Financial ratios (EDA Q2 and Q7)
# ---------------------------------------------------------------------------

def fit_ratio_caps(train_df: pd.DataFrame) -> Dict[str, float]:
    """Compute the high-percentile caps for the affordability ratios from
    TRAINING data only.

    EDA decision (Q2): "Cap the ratios at a high percentile" to limit
    extreme values (the EDA found a credit-to-income ratio as high as 84.7x
    income and an annuity-to-income ratio as high as 1.88x income). The
    99th-percentile cutoff is computed once on train and stored in `params`
    so the identical cutoff value is reused -- never recomputed -- on test.
    """
    income = train_df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    credit_income_ratio = train_df["AMT_CREDIT"] / income
    annuity_income_ratio = train_df["AMT_ANNUITY"] / income
    return {
        "credit_income_ratio_cap": float(
            credit_income_ratio.quantile(RATIO_CAP_PERCENTILE)
        ),
        "annuity_income_ratio_cap": float(
            annuity_income_ratio.quantile(RATIO_CAP_PERCENTILE)
        ),
    }


def add_financial_ratios(df: pd.DataFrame, ratio_caps: Dict[str, float]) -> pd.DataFrame:
    """Build the affordability ratios discussed in EDA Q2, and the
    per-family-member income idea raised in EDA Q7, then apply the
    training-derived caps.

    EDA linkage:
      - CREDIT_INCOME_RATIO / ANNUITY_INCOME_RATIO: EDA Q2's two
        affordability proxies. The EDA found a weak, non-monotone
        relationship with the target, so these are kept as *candidate*
        features, not as a replacement for a full debt-to-income measure
        (stated income has no verification or frequency field -- see
        feature_report.md).
      - INCOME_PER_FAMILY_MEMBER: EDA Q7's suggestion to "consider
        household size only in a per-person form, like income per family
        member" rather than using family size as a family-status proxy.
    """
    df = df.copy()
    income = df["AMT_INCOME_TOTAL"].replace(0, np.nan)

    df["CREDIT_INCOME_RATIO"] = (df["AMT_CREDIT"] / income).clip(
        upper=ratio_caps["credit_income_ratio_cap"]
    )
    df["ANNUITY_INCOME_RATIO"] = (df["AMT_ANNUITY"] / income).clip(
        upper=ratio_caps["annuity_income_ratio_cap"]
    )
    if "CNT_FAM_MEMBERS" in df.columns:
        family_size = df["CNT_FAM_MEMBERS"].replace(0, np.nan)
        df["INCOME_PER_FAMILY_MEMBER"] = df["AMT_INCOME_TOTAL"] / family_size

    return df


# ---------------------------------------------------------------------------
# 7. Binned variables and interaction terms
# ---------------------------------------------------------------------------

def add_binned_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bin employment tenure and age the same way the EDA plots did.

    EDA linkage:
      - EMPLOYMENT_YEARS_BAND reproduces the five tenure bands from Q5
        (<=1, 1-3, 3-5, 5-10, 10+ years), where the EDA found the clearest
        single-variable pattern in the whole notebook (11.1%% down to
        5.2%% payment difficulty).
      - AGE_BAND is a parallel, coarser binning of AGE_YEARS. It is
        provided for the same governed-comparison purpose as AGE_YEARS
        itself (see HIGH_GOVERNANCE_FEATURES) -- not part of the default
        baseline.
    """
    df = df.copy()
    if "EMPLOYMENT_YEARS" in df.columns:
        df["EMPLOYMENT_YEARS_BAND"] = pd.cut(
            df["EMPLOYMENT_YEARS"],
            bins=[-0.01, 1, 3, 5, 10, np.inf],
            labels=["0-1y", "1-3y", "3-5y", "5-10y", "10y+"],
        )
    if "AGE_YEARS" in df.columns:
        df["AGE_BAND"] = pd.cut(
            df["AGE_YEARS"],
            bins=[0, 25, 35, 45, 55, 65, np.inf],
            labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
        )
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a small set of interaction / aggregate features motivated
    directly by the EDA findings.

    EDA linkage:
      - EXT_SOURCE_MEAN / _MIN / _MAX / _COUNT_AVAILABLE: EDA Q1 found all
        three external scores individually separate the target, each with
        a different availability rate. Combining them into simple
        aggregates is a natural extension of that finding (and a common,
        explainable way to use "however many of the three scores exist"),
        computed with NaNs skipped so a missing score does not drag the
        aggregate down.
      - INCOME_TYPE_X_EMPLOYMENT_BAND: a simple interaction between the
        two employment-related fields that showed the clearest, most
        explainable signals in the notebook (EDA Q3's income-type
        differences and EDA Q5's employment-tenure bands), capturing that
        e.g. a short-tenure "Working" applicant may behave differently
        from a short-tenure pensioner.
    """
    df = df.copy()
    ext_cols = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in df.columns]
    if ext_cols:
        df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1, skipna=True)
        df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1, skipna=True)
        df["EXT_SOURCE_MAX"] = df[ext_cols].max(axis=1, skipna=True)
        df["EXT_SOURCE_COUNT_AVAILABLE"] = df[ext_cols].notna().sum(axis=1)

    if "NAME_INCOME_TYPE" in df.columns and "EMPLOYMENT_YEARS_BAND" in df.columns:
        df["INCOME_TYPE_X_EMPLOYMENT_BAND"] = (
            df["NAME_INCOME_TYPE"].astype(str)
            + "_"
            + df["EMPLOYMENT_YEARS_BAND"].astype(str)
        )
    return df


# ---------------------------------------------------------------------------
# 8. Rare-category collapsing and train/test category consistency (EDA 4.4,
#    Q3, Q7, Q8)
# ---------------------------------------------------------------------------

def fit_rare_category_levels(
    train_df: pd.DataFrame, columns: Iterable[str], min_count: int = RARE_CATEGORY_MIN_COUNT
) -> Dict[str, set]:
    """From TRAINING data only, record which levels of each categorical
    column are common enough to keep as their own level.

    EDA decision (Q3, Q7, Q8, and 4.4): "Group rare occupations and
    organizations into broader categories" (Q3), and more generally,
    "merge the very rare levels into 'other'... set up an unknown-level
    rule" (4.4), since train-only placeholder levels like CODE_GENDER ==
    'XNA' or NAME_FAMILY_STATUS == 'Unknown' were found. The kept-level set
    is computed once from train and reused as-is on test, so: (a) a level
    that is rare in train is collapsed in both datasets, and (b) any level
    that appears in test but was never seen in train automatically falls
    back to "Other" in step collapse_rare_categories below.
    """
    kept_levels = {}
    for col in columns:
        if col not in train_df.columns:
            continue
        counts = train_df[col].value_counts(dropna=True)
        kept_levels[col] = set(counts[counts >= min_count].index)
    return kept_levels


def collapse_rare_categories(df: pd.DataFrame, kept_levels: Dict[str, set]) -> pd.DataFrame:
    """Map any category level not in the training-derived keep-set to
    'Other'. This single rule handles three EDA findings at once: rare
    levels, undocumented placeholders (e.g. 'XNA', 'Unknown'), and any
    level present in test but never observed in train.
    """
    df = df.copy()
    for col, keep_set in kept_levels.items():
        if col not in df.columns:
            continue
        df[col] = df[col].where(df[col].isin(keep_set), other="Other")
    return df


# Columns the EDA actually discussed collapsing (Q3: occupation/organization;
# Q7/Q8/4.4: family status, housing type -- grouped here for the same
# "merge rare + unknown levels" treatment).
CATEGORICAL_COLUMNS_TO_COLLAPSE = (
    "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_TYPE_SUITE",
)


# ---------------------------------------------------------------------------
# 9. Supplementary-table processing (optional in the EDA; implemented here)
# ---------------------------------------------------------------------------
#
# EDA linkage: section 3 ("Data description") and 4.3 ("temporal direction")
# set the rules these aggregations follow: (1) aggregate every supplementary
# table to one row per SK_ID_CURR *before* joining, so one-to-many child
# rows never duplicate an application; (2) keep only records dated at or
# before the application (DAYS_* and MONTHS_BALANCE <= 0) so no future
# information leaks into a feature; (3) treat the 365243 sentinel, where it
# appears, as missing rather than a real date; (4) never assume
# bureau_balance's freshest month is -1, since 0 is also observed.
# ---------------------------------------------------------------------------

def _non_positive_or_missing(series: pd.Series) -> pd.Series:
    """Keep only values that are <= 0 (at or before the application) or
    already missing; anything positive is dropped (set to NaN) because the
    data map's convention is that these day/month counters run backward
    from the application/reference date.
    """
    return series.where((series <= 0) | series.isna())


def process_bureau_and_balance(bureau_df: pd.DataFrame, bureau_balance_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate bureau.csv (+ bureau_balance.csv) to one row per SK_ID_CURR.

    EDA linkage: feature_report's "prior delinquency, severity, and
    recency" and "current debt burden" families, and data_map's note that
    bureau_balance's true join key is SK_ID_BUREAU (the dictionary's
    SK_BUREAU_ID is wrong).
    """
    bureau_df = bureau_df.copy()
    bureau_df["DAYS_CREDIT"] = _non_positive_or_missing(bureau_df["DAYS_CREDIT"])
    if "DAYS_CREDIT_UPDATE" in bureau_df.columns:
        bureau_df["DAYS_CREDIT_UPDATE"] = _non_positive_or_missing(bureau_df["DAYS_CREDIT_UPDATE"])

    if bureau_balance_df is not None and len(bureau_balance_df) > 0:
        bb = bureau_balance_df.copy()
        # EDA 4.3: MONTHS_BALANCE == 0 is a valid, fresh snapshot -- do not
        # assume -1 is always the most recent row.
        bb_agg = bb.groupby("SK_ID_BUREAU").agg(
            BB_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
            BB_MONTHS_BALANCE_MIN=("MONTHS_BALANCE", "min"),
            BB_DPD_STATUS_COUNT=("STATUS", lambda s: (s.isin(["1", "2", "3", "4", "5"])).sum()),
        ).reset_index()
        bureau_df = bureau_df.merge(bb_agg, on="SK_ID_BUREAU", how="left")

    agg_spec = {
        "SK_ID_BUREAU": "count",
        "CREDIT_DAY_OVERDUE": "max",
        "AMT_CREDIT_SUM": "sum",
        "AMT_CREDIT_SUM_DEBT": "sum",
        "AMT_CREDIT_SUM_OVERDUE": "sum",
        "DAYS_CREDIT": "min",
    }
    agg_spec = {k: v for k, v in agg_spec.items() if k in bureau_df.columns}
    agg = bureau_df.groupby("SK_ID_CURR").agg(agg_spec)
    agg.columns = [f"BUREAU_{c}_{f.upper()}" for c, f in agg_spec.items()]

    if "CREDIT_ACTIVE" in bureau_df.columns:
        active_share = (
            bureau_df.assign(_active=(bureau_df["CREDIT_ACTIVE"] == "Active").astype(int))
            .groupby("SK_ID_CURR")["_active"].mean()
        )
        agg["BUREAU_ACTIVE_CREDIT_SHARE"] = active_share

    if "BB_DPD_STATUS_COUNT" in bureau_df.columns:
        agg["BUREAU_DPD_STATUS_COUNT_SUM"] = bureau_df.groupby("SK_ID_CURR")["BB_DPD_STATUS_COUNT"].sum()

    agg = agg.reset_index()
    agg["HAS_BUREAU_HISTORY"] = 1
    return agg


def process_previous_application(prev_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate previous_application.csv to one row per SK_ID_CURR.

    EDA linkage: feature_report's "credit-history depth" and "requested
    product/loan-structure risk" families; sentinels recoded first via
    recode_previous_application_sentinels.
    """
    prev_df = recode_previous_application_sentinels(prev_df)
    prev_df = prev_df.copy()
    if "DAYS_DECISION" in prev_df.columns:
        prev_df["DAYS_DECISION"] = _non_positive_or_missing(prev_df["DAYS_DECISION"])

    agg_spec = {
        "SK_ID_PREV": "count",
        "AMT_CREDIT": "mean",
        "AMT_ANNUITY": "mean",
        "DAYS_DECISION": "max",  # most recent prior decision (closest to 0)
    }
    agg_spec = {k: v for k, v in agg_spec.items() if k in prev_df.columns}
    agg = prev_df.groupby("SK_ID_CURR").agg(agg_spec)
    agg.columns = [f"PREV_APP_{c}_{f.upper()}" for c, f in agg_spec.items()]

    if "NAME_CONTRACT_STATUS" in prev_df.columns:
        approved_share = (
            prev_df.assign(_approved=(prev_df["NAME_CONTRACT_STATUS"] == "Approved").astype(int))
            .groupby("SK_ID_CURR")["_approved"].mean()
        )
        agg["PREV_APP_APPROVED_SHARE"] = approved_share

    agg = agg.reset_index()
    agg["HAS_PREVIOUS_APPLICATION"] = 1
    return agg


def _aggregate_monthly_snapshot_table(
    df: pd.DataFrame, prefix: str, numeric_cols: Iterable[str]
) -> pd.DataFrame:
    """Shared aggregation logic for the three monthly-snapshot tables
    (POS_CASH_balance, credit_card_balance, installments_payments): keep
    only at-or-before-application months, then summarize to one row per
    SK_ID_CURR.
    """
    df = df.copy()
    if "MONTHS_BALANCE" in df.columns:
        # A positive month is after the application date.  Exclude the row
        # itself (rather than only blanking its month value), otherwise its
        # balances and DPD values would still enter the aggregate.
        df = df.loc[
            (df["MONTHS_BALANCE"] <= 0) | df["MONTHS_BALANCE"].isna()
        ].copy()

    agg_spec = {"SK_ID_PREV": "count"}
    for col in numeric_cols:
        if col in df.columns:
            agg_spec[col] = "mean"
    agg = df.groupby("SK_ID_CURR").agg(agg_spec)
    agg.columns = [f"{prefix}_{c}_COUNT" if c == "SK_ID_PREV" else f"{prefix}_{c}_MEAN" for c in agg_spec]
    agg = agg.reset_index()
    agg[f"HAS_{prefix}"] = 1
    return agg


def process_pos_cash_balance(pos_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate POS_CASH_balance.csv to one row per SK_ID_CURR."""
    return _aggregate_monthly_snapshot_table(
        pos_df, "POS_CASH", ("SK_DPD", "SK_DPD_DEF", "CNT_INSTALMENT_FUTURE")
    )


def process_credit_card_balance(cc_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate credit_card_balance.csv to one row per SK_ID_CURR.

    EDA linkage: feature_report notes this table has the lowest training
    coverage of all supplementary tables (28.3%%), so HAS_CREDIT_CARD is an
    especially informative coverage flag here -- most applicants will have
    it at 0.
    """
    return _aggregate_monthly_snapshot_table(
        cc_df, "CREDIT_CARD", ("AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD")
    )


def process_installments_payments(inst_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate installments_payments.csv to one row per SK_ID_CURR.

    EDA linkage: feature_report's "payment reliability on internal
    accounts" family. We additionally derive a late-payment indicator
    (entry payment date after the scheduled instalment date) and a payment
    shortfall amount. Rows with an observed payment or scheduled-installment
    date after the application are excluded so they cannot leak future
    repayment information.
    """
    df = inst_df.copy()
    temporal_cols = [
        col for col in ("DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT")
        if col in df.columns
    ]
    for col in temporal_cols:
        df = df.loc[(df[col] <= 0) | df[col].isna()].copy()
    if {"DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"}.issubset(df.columns):
        df["PAID_LATE"] = (df["DAYS_ENTRY_PAYMENT"] > df["DAYS_INSTALMENT"]).astype(int)
    if {"AMT_INSTALMENT", "AMT_PAYMENT"}.issubset(df.columns):
        df["PAYMENT_SHORTFALL"] = df["AMT_INSTALMENT"] - df["AMT_PAYMENT"]

    agg_spec = {"SK_ID_PREV": "count"}
    for col in ("PAID_LATE", "PAYMENT_SHORTFALL"):
        if col in df.columns:
            agg_spec[col] = "mean"
    agg = df.groupby("SK_ID_CURR").agg(agg_spec)
    agg.columns = [
        "INSTALLMENTS_COUNT" if c == "SK_ID_PREV" else f"INSTALLMENTS_{c}_MEAN"
        for c in agg_spec
    ]
    agg = agg.reset_index()
    agg["HAS_INSTALLMENTS_HISTORY"] = 1
    return agg


def build_supplementary_features(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load and aggregate every supplementary table found in `data_dir`.

    This step is optional (the EDA notebook itself did not join these
    tables), so any file that is missing is simply skipped rather than
    raising an error -- `join_supplementary_features` fills in the
    corresponding HAS_* flag as 0 for every application either way.
    Aggregation does not use TARGET, so it is computed once and reused for
    both the train and test application tables (no train/test leakage
    concern the way a target-derived statistic would have).
    """
    tables: Dict[str, pd.DataFrame] = {}

    def _try_read(filename: str) -> Optional[pd.DataFrame]:
        path = os.path.join(data_dir, filename)
        return pd.read_csv(path) if os.path.exists(path) else None

    bureau = _try_read("bureau.csv")
    bureau_balance = _try_read("bureau_balance.csv")
    if bureau is not None:
        tables["bureau"] = process_bureau_and_balance(bureau, bureau_balance)

    prev = _try_read("previous_application.csv")
    if prev is not None:
        tables["previous_application"] = process_previous_application(prev)

    pos = _try_read("POS_CASH_balance.csv")
    if pos is not None:
        tables["pos_cash"] = process_pos_cash_balance(pos)

    cc = _try_read("credit_card_balance.csv")
    if cc is not None:
        tables["credit_card"] = process_credit_card_balance(cc)

    inst = _try_read("installments_payments.csv")
    if inst is not None:
        tables["installments"] = process_installments_payments(inst)

    return tables


def join_supplementary_features(
    app_df: pd.DataFrame, supplementary_tables: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Left-join every aggregated supplementary table onto the application
    table by SK_ID_CURR, then fill in "no history" defaults.

    EDA linkage: section 3's applicant-coverage figures (e.g. 85.7%% bureau
    coverage, 28.3%% credit-card coverage) are exactly why a left join plus
    explicit fill values is required -- most applicants will be missing
    from at least one table, and that absence is itself informative
    ("no previous Home Credit product", "no bureau record"), not a data
    problem to impute away. Count-type columns (e.g. *_COUNT) and the
    HAS_* coverage flags are filled with 0 for applicants absent from a
    table; continuous summary columns (e.g. *_MEAN) are left as NaN so the
    corresponding HAS_* flag -- not a fabricated average -- tells the model
    why the value is missing.
    """
    df = app_df.copy()
    for name, agg_table in supplementary_tables.items():
        df = df.merge(agg_table, on="SK_ID_CURR", how="left")
        has_col = [c for c in agg_table.columns if c.startswith("HAS_")][0]
        df[has_col] = df[has_col].fillna(0).astype(int)
        for col in agg_table.columns:
            if col in ("SK_ID_CURR", has_col):
                continue
            if col.endswith("_COUNT") or col.endswith("_SHARE") or col.endswith("_SUM"):
                df[col] = df[col].fillna(0)
            # mean/min/max summary columns are intentionally left as NaN
            # when there is no history, per the docstring above.
    return df


# ---------------------------------------------------------------------------
# 10. Top-level orchestration: fit once on train, apply identically to both
# ---------------------------------------------------------------------------

def fit_preprocessing_params(train_df_raw: pd.DataFrame) -> Dict:
    """Learn every threshold / cap / category grouping from the RAW
    training application table, and return them in one dict.

    This is the single place where anything is "computed from training
    data only" (the task's explicit requirement). Nothing in this function
    touches test data. The returned `params` dict is what later gets
    passed into `prepare_dataset(..., is_train=False)` for the test set, so
    the exact same numbers are reused rather than recomputed.
    """
    # Run the sentinel/dtype/demographic steps on a scratch copy so the
    # ratio caps are computed on realistic (sentinel-cleaned) values,
    # without mutating the caller's original DataFrame.
    scratch = recode_days_employed_sentinel(train_df_raw)
    scratch = fix_binary_flag_dtypes(scratch)
    scratch = engineer_demographic_features(scratch)
    scratch = add_binned_features(scratch)

    params = {
        "flags_to_drop": fit_flags_to_drop(scratch),
        "ratio_caps": fit_ratio_caps(scratch),
        "rare_category_levels": fit_rare_category_levels(
            scratch, CATEGORICAL_COLUMNS_TO_COLLAPSE, RARE_CATEGORY_MIN_COUNT
        ),
    }
    return params


def save_params(params: Dict, path: str) -> None:
    """Persist the fitted parameters so a later modeling run (or a
    production scoring job) can load them without re-fitting on train.
    """
    with open(path, "wb") as f:
        pickle.dump(params, f)


def load_params(path: str) -> Dict:
    """Load parameters previously saved by save_params."""
    with open(path, "rb") as f:
        return pickle.load(f)


def clean_and_engineer_application_features(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Apply every cleaning/engineering step, in a fixed order, using only
    values stored in `params`. This function is called identically for
    train and test -- it never branches on which dataset it is given --
    which is what guarantees the two outputs end up with the same columns.
    """
    df = recode_days_employed_sentinel(df)                      # EDA 4.3
    df = fix_binary_flag_dtypes(df)                              # EDA 4.3
    df = engineer_demographic_features(df)                       # requested + EDA Q5
    df = add_missing_indicators(df)                               # EDA 4.2
    df = handle_building_attributes(df)                           # EDA 4.2
    df = drop_low_value_flags(df, params["flags_to_drop"])        # EDA Q4
    df = process_enquiry_features(df)                            # EDA Q6
    df = add_financial_ratios(df, params["ratio_caps"])           # EDA Q2 / Q7
    df = add_binned_features(df)                                 # EDA Q5
    df = add_interaction_features(df)                            # EDA Q1 / Q3 / Q5
    df = collapse_rare_categories(df, params["rare_category_levels"])  # EDA 4.4 / Q3 / Q7 / Q8
    return df


def prepare_dataset(
    application_df_raw: pd.DataFrame,
    params: Dict,
    is_train: bool,
    supplementary_dir: Optional[str] = None,
    supplementary_tables: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Single entry point used for both application_train and
    application_test.

    Parameters
    ----------
    application_df_raw : the raw application_train or application_test table.
    params : the dict returned by fit_preprocessing_params(train_df_raw).
             Must be fit on train and then reused (unchanged) for test.
    is_train : whether this call is processing the training table. The
        ONLY thing this changes is whether TARGET is preserved; every
        cleaning/engineering step above runs identically either way, which
        is what keeps the two outputs' columns in sync.
    supplementary_dir : optional path to the folder with the supplementary
        CSVs (bureau.csv, previous_application.csv, etc). If supplied and
        `supplementary_tables` is not, the tables are built fresh from this
        directory. Processing supplementary data is optional, per the EDA.
    supplementary_tables : optional, pre-built output of
        build_supplementary_features(...), useful when preparing train and
        test in the same run so the (target-independent) aggregation work
        is only done once.
    """
    has_target = is_train and "TARGET" in application_df_raw.columns
    target = application_df_raw["TARGET"].copy() if has_target else None

    df = application_df_raw.drop(columns=["TARGET"]) if has_target else application_df_raw.copy()
    df = clean_and_engineer_application_features(df, params)

    if supplementary_tables is None and supplementary_dir is not None:
        supplementary_tables = build_supplementary_features(supplementary_dir)
    if supplementary_tables:
        df = join_supplementary_features(df, supplementary_tables)

    if has_target:
        df["TARGET"] = target.values

    return df


def assert_consistent_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Verify that train and test have identical columns except TARGET.

    This directly checks the task's "produce identical columns for train
    and test (except TARGET)" requirement, so it can be called right after
    prepare_dataset as a safety check before modeling.
    """
    train_cols = set(train_df.columns) - {"TARGET"}
    test_cols = set(test_df.columns) - {"TARGET"}
    only_in_train = train_cols - test_cols
    only_in_test = test_cols - train_cols
    if only_in_train or only_in_test:
        raise ValueError(
            "Train and test columns do not match.\n"
            f"Only in train: {sorted(only_in_train)}\n"
            f"Only in test: {sorted(only_in_test)}"
        )
    if "TARGET" not in train_df.columns:
        raise ValueError("Expected TARGET to be present in the training output.")
    if "TARGET" in test_df.columns:
        raise ValueError("TARGET should not be present in the test output.")


def check_application_grain(df: pd.DataFrame, dataset_name: str = "dataset") -> None:
    """Verify the application-level grain holds: one row per SK_ID_CURR.

    EDA linkage: section 4.3 confirmed this grain on the raw files (no
    duplicate SK_ID_CURR in train or test, and no overlap between the two).
    This check re-verifies the same thing on the OUTPUT of the pipeline,
    since a careless join against a one-to-many supplementary table is
    exactly the mistake that would duplicate rows and silently break the
    grain (see the "Implementation notes" in data_map.md).
    """
    n_rows = len(df)
    n_unique_ids = df["SK_ID_CURR"].nunique()
    if n_rows != n_unique_ids:
        raise ValueError(
            f"{dataset_name}: expected one row per SK_ID_CURR, but found "
            f"{n_rows} rows for {n_unique_ids} unique applications. A join "
            "against a one-to-many supplementary table likely duplicated "
            "some applications."
        )
    print(f"  {dataset_name}: {n_rows} rows, {n_unique_ids} unique SK_ID_CURR -- one row per application: OK")


def run_integrity_checks(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Run and print both post-processing integrity checks together:
    train/test column consistency and one-row-per-application grain. This
    is the function to call right after `prepare_dataset` and before
    handing the data to a model.
    """
    assert_consistent_columns(train_df, test_df)
    shared_cols = (set(train_df.columns) | set(test_df.columns)) - {"TARGET"}
    print(f"  Train and test share {len(shared_cols)} columns (TARGET excluded): OK")
    check_application_grain(train_df, "train")
    check_application_grain(test_df, "test")


def save_prepared_datasets(
    train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str
) -> Dict[str, str]:
    """Write the prepared train/test tables to CSV and return their paths.

    This is the function to call once `prepare_dataset` (and, optionally,
    `run_integrity_checks`) have produced the final tables, so the output
    of this script can be handed to a separate modeling script as plain
    CSV files rather than kept only in memory.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "application_train_prepared.csv")
    test_path = os.path.join(output_dir, "application_test_prepared.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return {"train": train_path, "test": test_path}


# ---------------------------------------------------------------------------
# Self-test / usage example
# ---------------------------------------------------------------------------
# This block only runs when the script is executed directly (not when it is
# imported). It builds a synthetic dataset that reproduces the REAL
# application_train/application_test schema column-for-column (122 columns
# in train, 121 in test, per data_map.md). It is only a smoke-test fixture:
# category frequencies can change the learned drop lists, so its prepared
# output dimensions are not the real-data dimensions reported in the README.

def _make_synthetic_application_data(n_train: int = 2000, n_test: int = 600, seed: int = 0):
    """Build a schema-accurate synthetic application_train/application_test
    pair: same 122/121 column names as the real files, with randomly
    generated values of a plausible type for each column. Used only for
    the self-test below -- never for real modeling or real-data dimensions.
    """
    rng = np.random.default_rng(seed)

    # The 14 building-attribute base names that each come in three versions
    # (_AVG, _MODE, _MEDI) in the real file -- reused here and in
    # BUILDING_ATTRIBUTE_PREFIXES above.
    building_bases = [
        "APARTMENTS", "BASEMENTAREA", "YEARS_BEGINEXPLUATATION", "YEARS_BUILD",
        "COMMONAREA", "ELEVATORS", "ENTRANCES", "FLOORSMAX", "FLOORSMIN",
        "LANDAREA", "LIVINGAPARTMENTS", "LIVINGAREA", "NONLIVINGAPARTMENTS",
        "NONLIVINGAREA",
    ]
    building_single = [
        "TOTALAREA_MODE", "FONDKAPREMONT_MODE", "HOUSETYPE_MODE",
        "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE",
    ]
    doc_flags = [f"FLAG_DOCUMENT_{i}" for i in range(2, 22)]
    enquiry_cols = [
        "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
    ]

    def _frame(n, sk_start):
        days_employed = rng.integers(-15000, -50, size=n).astype(float)
        days_employed[rng.random(n) < 0.18] = DAYS_EMPLOYED_SENTINEL

        data = {
            "SK_ID_CURR": np.arange(sk_start, sk_start + n),
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], size=n, p=[0.9, 0.1]),
            "CODE_GENDER": rng.choice(["M", "F", "XNA"], size=n, p=[0.34, 0.65, 0.01]),
            "FLAG_OWN_CAR": rng.choice(["Y", "N"], size=n),
            "FLAG_OWN_REALTY": rng.choice(["Y", "N"], size=n),
            "CNT_CHILDREN": rng.integers(0, 4, size=n),
            "AMT_INCOME_TOTAL": rng.uniform(50_000, 400_000, size=n),
            "AMT_CREDIT": rng.uniform(50_000, 900_000, size=n),
            "AMT_ANNUITY": rng.uniform(5_000, 60_000, size=n),
            "AMT_GOODS_PRICE": rng.uniform(40_000, 850_000, size=n),
            "NAME_TYPE_SUITE": rng.choice(["Unaccompanied", "Family", np.nan], size=n, p=[0.8, 0.15, 0.05]),
            "NAME_INCOME_TYPE": rng.choice(
                ["Working", "Commercial associate", "Pensioner", "State servant", "Maternity leave"],
                size=n, p=[0.52, 0.23, 0.18, 0.065, 0.005],
            ),
            "NAME_EDUCATION_TYPE": rng.choice(
                ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary"],
                size=n, p=[0.71, 0.24, 0.035, 0.015],
            ),
            "NAME_FAMILY_STATUS": rng.choice(
                ["Married", "Single / not married", "Civil marriage", "Separated", "Widow", "Unknown"],
                size=n, p=[0.64, 0.145, 0.095, 0.065, 0.05, 0.005],
            ),
            "NAME_HOUSING_TYPE": rng.choice(
                ["House / apartment", "With parents", "Municipal apartment", "Rented apartment",
                 "Office apartment", "Co-op apartment"],
                size=n, p=[0.89, 0.048, 0.036, 0.016, 0.008, 0.002],
            ),
            "REGION_POPULATION_RELATIVE": rng.uniform(0.0005, 0.075, size=n),
            "DAYS_BIRTH": rng.integers(-25000, -7500, size=n),
            "DAYS_EMPLOYED": days_employed,
            "DAYS_REGISTRATION": rng.integers(-10000, 0, size=n).astype(float),
            "DAYS_ID_PUBLISH": rng.integers(-6000, 0, size=n),
            "OWN_CAR_AGE": np.where(rng.random(n) < 0.66, np.nan, rng.uniform(0, 20, size=n)),
            "FLAG_MOBIL": 1,
            "FLAG_EMP_PHONE": rng.choice([0, 1], size=n, p=[0.18, 0.82]),
            "FLAG_WORK_PHONE": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
            "FLAG_CONT_MOBILE": rng.choice([0, 1], size=n, p=[0.002, 0.998]),
            "FLAG_PHONE": rng.choice([0, 1], size=n, p=[0.72, 0.28]),
            "FLAG_EMAIL": rng.choice([0, 1], size=n, p=[0.94, 0.06]),
            "OCCUPATION_TYPE": rng.choice(
                ["Laborers", "Core staff", "Sales staff", "Managers", "Drivers", "Low-skill Laborers",
                 np.nan],
                size=n, p=[0.18, 0.09, 0.1, 0.07, 0.06, 0.01, 0.49],
            ),
            "CNT_FAM_MEMBERS": rng.integers(1, 6, size=n).astype(float),
            "REGION_RATING_CLIENT": rng.choice([1, 2, 3], size=n, p=[0.1, 0.74, 0.16]),
            "REGION_RATING_CLIENT_W_CITY": rng.choice([1, 2, 3], size=n, p=[0.11, 0.75, 0.14]),
            "WEEKDAY_APPR_PROCESS_START": rng.choice(
                ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"], size=n
            ),
            "HOUR_APPR_PROCESS_START": rng.integers(0, 24, size=n),
            "REG_REGION_NOT_LIVE_REGION": rng.choice([0, 1], size=n, p=[0.98, 0.02]),
            "REG_REGION_NOT_WORK_REGION": rng.choice([0, 1], size=n, p=[0.95, 0.05]),
            "LIVE_REGION_NOT_WORK_REGION": rng.choice([0, 1], size=n, p=[0.96, 0.04]),
            "REG_CITY_NOT_LIVE_CITY": rng.choice([0, 1], size=n, p=[0.92, 0.08]),
            "REG_CITY_NOT_WORK_CITY": rng.choice([0, 1], size=n, p=[0.78, 0.22]),
            "LIVE_CITY_NOT_WORK_CITY": rng.choice([0, 1], size=n, p=[0.82, 0.18]),
            "ORGANIZATION_TYPE": rng.choice(
                ["Business Entity Type 3", "XNA", "Self-employed", "Government", "School", "Other"],
                size=n, p=[0.22, 0.18, 0.12, 0.1, 0.08, 0.3],
            ),
            "EXT_SOURCE_1": np.where(rng.random(n) < 0.56, np.nan, rng.random(n)),
            "EXT_SOURCE_2": np.where(rng.random(n) < 0.002, np.nan, rng.random(n)),
            "EXT_SOURCE_3": np.where(rng.random(n) < 0.2, np.nan, rng.random(n)),
            "OBS_30_CNT_SOCIAL_CIRCLE": rng.integers(0, 10, size=n).astype(float),
            "DEF_30_CNT_SOCIAL_CIRCLE": rng.integers(0, 3, size=n).astype(float),
            "OBS_60_CNT_SOCIAL_CIRCLE": rng.integers(0, 10, size=n).astype(float),
            "DEF_60_CNT_SOCIAL_CIRCLE": rng.integers(0, 3, size=n).astype(float),
            "DAYS_LAST_PHONE_CHANGE": rng.integers(-3000, 0, size=n).astype(float),
        }

        for base in building_bases:
            missing_mask = rng.random(n) < 0.65
            for suffix in ("AVG", "MODE", "MEDI"):
                col = f"{base}_{suffix}"
                values = rng.random(n)
                values[missing_mask] = np.nan
                data[col] = values
        for col in building_single:
            mask = rng.random(n) < 0.6
            values = pd.array(["info"] * n, dtype=object)
            values[mask] = np.nan
            data[col] = values

        for i, col in enumerate(doc_flags):
            # Most document flags are rare; FLAG_DOCUMENT_3 is the common one.
            p_yes = 0.71 if col == "FLAG_DOCUMENT_3" else 0.01
            data[col] = rng.choice([0, 1], size=n, p=[1 - p_yes, p_yes])

        enquiry_missing = rng.random(n) < 0.135
        for col in enquiry_cols[:3]:  # hour/day/week: almost always 0
            values = rng.integers(0, 2, size=n).astype(float)
            values[enquiry_missing] = np.nan
            data[col] = values
        for col in enquiry_cols[3:]:  # month/quarter/year: a bit more spread
            values = rng.integers(0, 8, size=n).astype(float)
            values[enquiry_missing] = np.nan
            data[col] = values

        return pd.DataFrame(data)

    train = _frame(n_train, sk_start=100_000)
    train["TARGET"] = rng.choice([0, 1], size=n_train, p=[0.9193, 0.0807])
    test = _frame(n_test, sk_start=500_000)
    return train, test


def _run_self_test():
    """Run the full pipeline end to end on schema-accurate synthetic data
    and print its checks and dimensions. This is a smoke test (run it with
    `python Data_Prep.py`); its prepared output dimensions are not the
    real-data dimensions in `README (1).md`.
    """
    print("Running Data_Prep self-test with schema-accurate synthetic data "
          "(no real CSVs needed)...")
    train_raw, test_raw = _make_synthetic_application_data()
    print(f"  application_train (synthetic): {train_raw.shape}")
    print(f"  application_test  (synthetic): {test_raw.shape}")

    # Step 1: learn every threshold/cap/category list from train only.
    params = fit_preprocessing_params(train_raw)

    # Step 2: apply the identical pipeline to train and test.
    train_ready = prepare_dataset(train_raw, params, is_train=True)
    test_ready = prepare_dataset(test_raw, params, is_train=False)
    print(f"  application_train_prepared: {train_ready.shape}")
    print(f"  application_test_prepared:  {test_ready.shape}")

    # Step 3: verify the two required integrity properties.
    run_integrity_checks(train_ready, test_ready)

    # Spot-check a few specific EDA decisions:
    assert train_ready["DAYS_EMPLOYED"].max() <= 0 or train_ready["DAYS_EMPLOYED"].isna().any()
    assert "DAYS_EMPLOYED_SENTINEL_FLAG" in train_ready.columns
    assert "HAS_BUILDING_INFO" in train_ready.columns
    assert "APARTMENTS_AVG" not in train_ready.columns
    assert "EXT_SOURCE_MEAN" in train_ready.columns
    assert "EMPLOYMENT_YEARS_BAND" in train_ready.columns
    assert train_ready["FLAG_OWN_CAR"].dropna().isin([0, 1]).all()
    print("  Spot checks on key EDA decisions: OK")

    # Step 4 (optional): write the prepared tables to CSV, the same call a
    # real run would make after processing the actual application files.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = save_prepared_datasets(train_ready, test_ready, tmp_dir)
        print(f"  Example CSV output written to: {list(paths.values())}")

    print("Self-test passed.")


if __name__ == "__main__":
    _run_self_test()

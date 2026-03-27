"""utils.py — shared helpers used across all pages"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ── Load & cache data ─────────────────────────────────────────────────────────
def load_data(path="survey_data.csv"):
    df = pd.read_csv(path)
    return df


def clean_bmi(df):
    """Remove extreme BMI outliers (keep 10–60) for analysis."""
    df = df.copy()
    df["bmi_clean"] = df["bmi"].clip(10, 60)
    return df


# ── Ordinal encoders ──────────────────────────────────────────────────────────
INCOME_ORDER   = ["<20k","20k-50k","50k-1L","1L-2L",">2L"]
AGE_ORDER      = ["Under 18","18-24","25-34","35-44","45-60","60+"]
HABIT_ORDER    = ["Very unhealthy","Somewhat unhealthy","Neutral","Fairly healthy","Very healthy"]
ACTIVITY_ORDER = ["Sedentary","Lightly active","Moderately active","Very active"]
SPEND_ORDER    = ["₹0","₹1-500","₹500-2k","₹2k-5k","₹5k+"]
SLEEP_ORDER    = ["<5hrs","5-6hrs","7-8hrs",">8hrs"]
MEALS_ORDER    = ["1-2","3","4-5","No schedule"]

def encode_features(df):
    """Return a numerically-encoded dataframe suitable for ML."""
    d = df.copy()

    def ordinal(col, order):
        mapping = {v: i for i, v in enumerate(order)}
        d[col] = d[col].map(mapping).fillna(0).astype(int)

    ordinal("income_band",   INCOME_ORDER)
    ordinal("age_group",     AGE_ORDER)
    ordinal("eating_habit",  HABIT_ORDER)
    ordinal("activity_level",ACTIVITY_ORDER)
    ordinal("past_health_spend", SPEND_ORDER)
    ordinal("sleep_hours",   SLEEP_ORDER)
    ordinal("meals_per_day", MEALS_ORDER)

    cat_cols = [
        "gender","city_tier","occupation","food_pref",
        "fitness_goal","workout_pref","health_motivation",
        "diet_break_response","health_personality",
        "data_sharing_comfort","health_influencer",
        "app_stop_reason","paid_before","pricing_model",
        "doc_recommendation","household_health","discovery_channel",
        "barrier",
    ]
    le = LabelEncoder()
    for c in cat_cols:
        if c in d.columns:
            d[c] = le.fit_transform(d[c].astype(str))

    # Drop multi-select text columns (used separately for ARM)
    drop_cols = [
        "health_conditions","food_allergy","services_wanted",
        "meal_plan_types","upgrade_triggers","reminder_pref",
        "signup_intent",          # target — added back by caller
        "bmi",                    # use bmi_clean
    ]
    d = d.drop(columns=[c for c in drop_cols if c in d.columns], errors="ignore")
    if "bmi_clean" not in d.columns:
        d = clean_bmi(d)

    # Safety net: encode any remaining object columns with LabelEncoder
    for c in d.select_dtypes(include="object").columns:
        d[c] = le.fit_transform(d[c].astype(str))

    return d


def get_ml_xy(df, target="signup_intent"):
    """Return X (features), y (encoded target), label_names."""
    df = clean_bmi(df)
    enc = encode_features(df)

    label_map = {"High": 2, "Medium": 1, "Low": 0}
    y = df[target].map(label_map).values
    label_names = ["Low","Medium","High"]

    feature_cols = [c for c in enc.columns if c not in ["monthly_budget"]]
    X = enc[feature_cols].fillna(0)
    return X, y, label_names, feature_cols


def get_regression_xy(df):
    """Return X, y for budget regression.
    income_band, paid_before and past_health_spend are dropped to prevent
    the linear models from perfectly reverse-engineering the budget formula.
    The model must infer spending power from behavioural signals instead.
    """
    df = clean_bmi(df)
    enc = encode_features(df)
    y = df["monthly_budget"].values
    drop_cols = ["monthly_budget", "paid_before", "past_health_spend"]
    X = enc.drop(columns=[c for c in drop_cols if c in enc.columns]).fillna(0)
    return X, y, list(X.columns)


# ── ARM helpers ───────────────────────────────────────────────────────────────
def explode_multiselect(df, col):
    """Return one-hot encoded dataframe from a pipe-separated multi-select column."""
    rows = df[col].fillna("None").str.split("|")
    all_vals = sorted({v.strip() for row in rows for v in row if v.strip() not in ("None","")})
    ohe = pd.DataFrame(
        {v: rows.apply(lambda r: int(v in r)) for v in all_vals},
        index=df.index
    )
    return ohe


# ── Persona labels (for clustering page) ─────────────────────────────────────
PERSONA_LABELS = {
    0: ("Health-Anxious Urban Pro",    "💼", "#185FA5"),
    1: ("Budget-Conscious Student",    "🎓", "#3B6D11"),
    2: ("Fitness Enthusiast",          "🏋️", "#854F0B"),
    3: ("Middle-Aged Wellness Seeker", "🧘", "#993556"),
    4: ("Casual Health Explorer",      "🌱", "#5F5E5A"),
}

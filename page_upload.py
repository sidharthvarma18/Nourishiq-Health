"""page_upload.py — Upload new customer CSV and score sign-up intent"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib, io, os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from utils import load_data, get_ml_xy, get_regression_xy, encode_features, clean_bmi

MODEL_PATH    = "clf_model.joblib"
META_PATH     = "clf_meta.joblib"
REG_MODEL_PATH= "reg_model.joblib"
REG_META_PATH = "reg_meta.joblib"


def ensure_models():
    """Train and save models if not yet done (first visit)."""
    if not os.path.exists(MODEL_PATH):
        df = load_data()
        X, y, label_names, feature_cols = get_ml_xy(df)
        from sklearn.model_selection import train_test_split
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.25,
                                             random_state=42, stratify=y)
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        joblib.dump(clf, MODEL_PATH)
        joblib.dump({"feature_cols": feature_cols, "label_names": label_names}, META_PATH)

    if not os.path.exists(REG_MODEL_PATH):
        df = load_data()
        X, y, feature_cols = get_regression_xy(df)
        from sklearn.model_selection import train_test_split
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.25, random_state=42)
        reg = GradientBoostingRegressor(n_estimators=200, random_state=42,
                                        learning_rate=0.05, max_depth=4)
        reg.fit(X_tr, y_tr)
        joblib.dump(reg, REG_MODEL_PATH)
        joblib.dump({"feature_cols": feature_cols}, REG_META_PATH)


def align_columns(df_new, required_cols):
    """Add missing columns as 0, drop extra columns, reorder."""
    for c in required_cols:
        if c not in df_new.columns:
            df_new[c] = 0
    return df_new[required_cols]


def show():
    st.title("🚀 New Lead Scorer")
    st.markdown("**Upload a CSV of new potential customers and get instant sign-up predictions.**")
    st.markdown("---")

    ensure_models()

    # ── Load models ───────────────────────────────────────────────────────────
    clf      = joblib.load(MODEL_PATH)
    clf_meta = joblib.load(META_PATH)
    reg      = joblib.load(REG_MODEL_PATH)
    reg_meta = joblib.load(REG_META_PATH)

    clf_features = clf_meta["feature_cols"]
    reg_features = reg_meta["feature_cols"]
    label_names  = clf_meta["label_names"]   # ["Low","Medium","High"]

    # ── Template download ─────────────────────────────────────────────────────
    st.subheader("Step 1 — Download the input template")
    st.markdown("Your CSV must have the same columns as the original survey. Download the template:")

    df_sample = load_data().head(5).drop(
        columns=["signup_intent","monthly_budget"], errors="ignore")
    csv_template = df_sample.to_csv(index=False)
    st.download_button(
        label="⬇️  Download CSV template",
        data=csv_template,
        file_name="nourishiq_template.csv",
        mime="text/csv")

    st.markdown("---")
    st.subheader("Step 2 — Upload your new customer CSV")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is None:
        # Demo mode — score a random sample from the training set
        st.info("No file uploaded — showing a **demo** scored on 20 random rows from the training dataset.")
        df_new = load_data().sample(20, random_state=7).drop(
            columns=["signup_intent","monthly_budget"], errors="ignore")
        actual_intent = load_data().loc[df_new.index, "signup_intent"].values
        demo_mode = True
    else:
        df_new       = pd.read_csv(uploaded)
        actual_intent= None
        demo_mode    = False

    st.write(f"Loaded **{len(df_new)} rows**, **{len(df_new.columns)} columns**")

    # ── Preprocess ────────────────────────────────────────────────────────────
    df_proc = clean_bmi(df_new.copy())
    enc_new = encode_features(df_proc)

    X_clf = align_columns(enc_new.copy(), clf_features).fillna(0)
    X_reg = align_columns(enc_new.copy(), reg_features).fillna(0)

    # ── Predict ───────────────────────────────────────────────────────────────
    intent_encoded = clf.predict(X_clf)
    intent_proba   = clf.predict_proba(X_clf)
    budget_pred    = reg.predict(X_reg).round()

    intent_decoded = [label_names[i] for i in intent_encoded]
    proba_df = pd.DataFrame(intent_proba,
                            columns=[f"P({l})" for l in label_names])

    results = df_new.reset_index(drop=True).copy()
    results["Predicted Intent"]   = intent_decoded
    results["P(Low)"]             = proba_df["P(Low)"].round(3)
    results["P(Medium)"]          = proba_df["P(Medium)"].round(3)
    results["P(High)"]            = proba_df["P(High)"].round(3)
    results["Predicted Budget (₹)"] = budget_pred.astype(int)

    # Recommended action
    def recommend(row):
        if row["Predicted Intent"] == "High" and row["Predicted Budget (₹)"] > 400:
            return "🟢 Pitch Pro Plan directly"
        elif row["Predicted Intent"] == "High":
            return "🟡 Offer 15% discount on Pro"
        elif row["Predicted Intent"] == "Medium":
            return "🟠 Free trial → nurture"
        else:
            return "🔴 Organic content only"

    results["Recommended Action"] = results.apply(recommend, axis=1)

    if demo_mode and actual_intent is not None:
        results["Actual Intent"] = actual_intent

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Scoring Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Leads",       len(results))
    c2.metric("High Intent",       (results["Predicted Intent"]=="High").sum())
    c3.metric("Medium Intent",     (results["Predicted Intent"]=="Medium").sum())
    c4.metric("Avg Predicted Budget", f"₹{results['Predicted Budget (₹)'].mean():.0f}")

    # ── Intent distribution ───────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        intent_counts = results["Predicted Intent"].value_counts().reset_index()
        intent_counts.columns = ["Intent","Count"]
        color_map = {"High":"#1D9E75","Medium":"#EF9F27","Low":"#E24B4A"}
        fig1 = px.pie(intent_counts, names="Intent", values="Count",
                      color="Intent", color_discrete_map=color_map,
                      hole=0.45, title="Predicted intent split")
        fig1.update_layout(margin=dict(t=40,b=10), height=300)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.histogram(results, x="Predicted Budget (₹)",
                            color="Predicted Intent",
                            color_discrete_map=color_map,
                            nbins=20, title="Budget distribution by intent")
        fig2.update_layout(margin=dict(t=40,b=10), height=300)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Probability heatmap ───────────────────────────────────────────────────
    st.subheader("Intent Probability Heatmap (first 50 leads)")
    prob_heat = results[["P(Low)","P(Medium)","P(High)"]].head(50)
    fig3 = px.imshow(prob_heat.T,
                     color_continuous_scale="RdYlGn",
                     labels=dict(x="Lead Index", y="Intent Class", color="Probability"),
                     title="Probability per class per lead",
                     aspect="auto")
    fig3.update_layout(margin=dict(t=40,b=10), height=280)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Full table ────────────────────────────────────────────────────────────
    st.subheader("Scored Lead Table")
    show_cols = ["Predicted Intent","P(Low)","P(Medium)","P(High)",
                 "Predicted Budget (₹)","Recommended Action"]
    if "Actual Intent" in results.columns:
        show_cols.insert(0,"Actual Intent")
    display_df = results[show_cols]
    st.dataframe(
        display_df.style.applymap(
            lambda v: "background-color:#d4edda" if v=="High" else
                      "background-color:#fff3cd" if v=="Medium" else
                      "background-color:#f8d7da" if v=="Low" else "",
            subset=["Predicted Intent"]),
        use_container_width=True, height=420)

    # ── Download results ──────────────────────────────────────────────────────
    st.markdown("---")
    csv_out = results.to_csv(index=False)
    st.download_button(
        label="⬇️  Download scored leads CSV",
        data=csv_out,
        file_name="scored_leads.csv",
        mime="text/csv")

    st.caption("Green = High intent | Yellow = Medium | Red = Low intent")

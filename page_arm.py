"""page_arm.py — Association Rule Mining (Apriori) with support, confidence, lift"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from utils import load_data, explode_multiselect


def build_transactions(df, col):
    rows = df[col].fillna("None").str.split("|")
    transactions = [[v.strip() for v in row if v.strip() not in ("None","")]
                    for row in rows]
    transactions = [t for t in transactions if len(t) > 0]
    return transactions


def transactions_to_df(transactions):
    te  = TransactionEncoder()
    arr = te.fit_transform(transactions)
    return pd.DataFrame(arr, columns=te.columns_)


def show():
    st.title("📋 Association Rule Mining")
    st.markdown("**What goes together?** — Discover which services & meal plans customers want in bundles.")
    st.markdown("---")

    df = load_data()

    # ── Column selector ───────────────────────────────────────────────────────
    target_col = st.selectbox(
        "Select multi-select column to mine",
        ["services_wanted", "meal_plan_types", "upgrade_triggers", "reminder_pref"],
        format_func=lambda x: {
            "services_wanted":   "Services wanted",
            "meal_plan_types":   "Meal plan types",
            "upgrade_triggers":  "Upgrade triggers",
            "reminder_pref":     "Reminder preferences",
        }[x])

    col1, col2, col3 = st.columns(3)
    min_support    = col1.slider("Min Support",    0.05, 0.60, 0.15, 0.01)
    min_confidence = col2.slider("Min Confidence", 0.10, 0.99, 0.40, 0.05)
    min_lift       = col3.slider("Min Lift",       1.0,  5.0,  1.2,  0.1)

    transactions = build_transactions(df, target_col)

    if not transactions:
        st.warning("No valid transactions found.")
        return

    te_df = transactions_to_df(transactions)

    with st.spinner("Running Apriori…"):
        freq_itemsets = apriori(te_df, min_support=min_support,
                                use_colnames=True, max_len=4)

    if freq_itemsets.empty:
        st.warning("No frequent itemsets found — try lowering the min support.")
        return

    rules = association_rules(freq_itemsets, metric="lift",
                               min_threshold=min_lift, num_itemsets=len(freq_itemsets))
    rules = rules[rules["confidence"] >= min_confidence].copy()
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Frequent Itemsets", len(freq_itemsets))
    c2.metric("Rules Generated",   len(rules))
    c3.metric("Avg Lift",
              f"{rules['lift'].mean():.2f}" if not rules.empty else "—")

    if rules.empty:
        st.warning("No rules meet your thresholds — try lowering confidence or lift.")

        # Still show frequent itemsets
        st.subheader("Frequent Itemsets")
        freq_itemsets["itemsets_str"] = freq_itemsets["itemsets"].apply(
            lambda x: ", ".join(sorted(x)))
        freq_itemsets = freq_itemsets.sort_values("support", ascending=False)
        fig_fi = px.bar(freq_itemsets.head(20), x="support", y="itemsets_str",
                        orientation="h",
                        labels={"support":"Support","itemsets_str":"Itemset"},
                        color="support", color_continuous_scale="Greens",
                        title="Top 20 frequent itemsets by support")
        fig_fi.update_layout(yaxis=dict(autorange="reversed"),
                             margin=dict(t=40,b=10), height=500,
                             coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)
        return

    # ── Format rules for display ──────────────────────────────────────────────
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: " + ".join(sorted(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: " + ".join(sorted(x)))
    rules["rule"]            = rules["antecedents_str"] + "  →  " + rules["consequents_str"]

    display_cols = ["rule","support","confidence","lift","leverage","conviction"]
    display_rules = rules[display_cols].round(4).head(30)

    st.markdown("---")
    st.subheader("Top Association Rules")
    st.dataframe(
        display_rules.style
            .background_gradient(cmap="Greens", subset=["support","confidence"])
            .background_gradient(cmap="Oranges", subset=["lift"]),
        use_container_width=True, height=400)

    st.markdown("---")

    # ── Confidence vs Lift scatter ────────────────────────────────────────────
    st.subheader("Confidence vs Lift (bubble = support)")
    fig_scatter = px.scatter(
        rules.head(50),
        x="confidence", y="lift",
        size="support", color="lift",
        color_continuous_scale="Oranges",
        hover_data=["rule","support","confidence","lift"],
        title="Each bubble = one rule. Bigger = higher support. Colour = lift.")
    fig_scatter.update_layout(margin=dict(t=40,b=10), height=420)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Frequent itemset bar ──────────────────────────────────────────────────
    st.subheader("Top Frequent Itemsets by Support")
    freq_itemsets["itemsets_str"] = freq_itemsets["itemsets"].apply(
        lambda x: ", ".join(sorted(x)))
    freq_top = freq_itemsets.sort_values("support", ascending=False).head(20)
    fig_fi = px.bar(freq_top, x="support", y="itemsets_str",
                    orientation="h",
                    labels={"support":"Support","itemsets_str":"Itemset"},
                    color="support", color_continuous_scale="Teal",
                    title="Top 20 most common item combinations")
    fig_fi.update_layout(yaxis=dict(autorange="reversed"),
                         margin=dict(t=40,b=10), height=500,
                         coloraxis_showscale=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    # ── Support vs Confidence ─────────────────────────────────────────────────
    st.subheader("Support vs Confidence Distribution")
    fig_sup_conf = px.scatter(
        rules, x="support", y="confidence", color="lift",
        color_continuous_scale="Viridis",
        hover_data=["rule"],
        title="Support vs Confidence — colour indicates lift value")
    fig_sup_conf.add_hline(y=min_confidence, line_dash="dash",
                           line_color="red", annotation_text="Min confidence")
    fig_sup_conf.update_layout(margin=dict(t=40,b=10), height=380)
    st.plotly_chart(fig_sup_conf, use_container_width=True)

    # ── Business bundle recommendations ───────────────────────────────────────
    st.markdown("---")
    st.subheader("💡 Bundle Recommendations for NourishIQ")
    top_rules = rules.head(5)
    for _, row in top_rules.iterrows():
        conf_pct = row["confidence"] * 100
        lift_val = row["lift"]
        st.markdown(
            f"- **{row['antecedents_str']}** → **{row['consequents_str']}**  "
            f"| Confidence: `{conf_pct:.1f}%` | Lift: `{lift_val:.2f}x`  "
            f"→ Bundle these together in your **Pro plan**")

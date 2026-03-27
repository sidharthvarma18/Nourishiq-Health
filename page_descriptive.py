"""page_descriptive.py — Descriptive Analysis"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data, clean_bmi


def show():
    st.title("📊 Descriptive Analysis")
    st.markdown("**What is happening?** — Explore the survey data distribution.")
    st.markdown("---")

    df = load_data()
    df = clean_bmi(df)

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.expander("🔧 Filters", expanded=False):
        col1, col2, col3 = st.columns(3)
        selected_city   = col1.multiselect("City Tier",   df["city_tier"].unique(),   default=list(df["city_tier"].unique()))
        selected_age    = col2.multiselect("Age Group",   df["age_group"].unique(),   default=list(df["age_group"].unique()))
        selected_gender = col3.multiselect("Gender",      df["gender"].unique(),      default=list(df["gender"].unique()))

    mask = (df["city_tier"].isin(selected_city) &
            df["age_group"].isin(selected_age) &
            df["gender"].isin(selected_gender))
    dff = df[mask]
    st.caption(f"Filtered dataset: **{len(dff)} respondents**")

    # ── BMI distribution ──────────────────────────────────────────────────────
    st.subheader("BMI Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(dff, x="bmi_clean", nbins=40, color_discrete_sequence=["#1D9E75"],
                           labels={"bmi_clean":"BMI"}, title="BMI Histogram")
        fig.add_vline(x=18.5, line_dash="dash", line_color="orange", annotation_text="Underweight")
        fig.add_vline(x=25.0, line_dash="dash", line_color="red",    annotation_text="Overweight")
        fig.update_layout(margin=dict(t=40,b=10), height=300)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        bins = [0, 18.5, 25, 30, 100]
        labels = ["Underweight","Healthy","Overweight","Obese"]
        dff = dff.copy()
        dff["bmi_cat"] = pd.cut(dff["bmi_clean"], bins=bins, labels=labels)
        bmi_cat_counts = dff["bmi_cat"].value_counts().reset_index()
        bmi_cat_counts.columns = ["BMI Category","Count"]
        fig2 = px.pie(bmi_cat_counts, names="BMI Category", values="Count",
                      hole=0.4, color="BMI Category",
                      color_discrete_map={"Healthy":"#1D9E75","Overweight":"#EF9F27",
                                          "Obese":"#E24B4A","Underweight":"#378ADD"},
                      title="BMI Categories")
        fig2.update_layout(margin=dict(t=40,b=10), height=300)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Service demand heatmap ────────────────────────────────────────────────
    st.subheader("Service Interest by Age Group")
    services = ["BMI-tracker","Diet-plans","Workout-routines",
                "Hydration-reminders","Progress-reports","Recipe-suggestions"]
    age_order = ["Under 18","18-24","25-34","35-44","45-60","60+"]
    heat_data = []
    for age in age_order:
        sub = dff[dff["age_group"]==age]
        for svc in services:
            pct = sub["services_wanted"].str.contains(svc, na=False).mean() * 100
            heat_data.append({"Age Group":age,"Service":svc,"Interest (%)":round(pct,1)})
    heat_df = pd.DataFrame(heat_data)
    pivot = heat_df.pivot(index="Service", columns="Age Group", values="Interest (%)")
    pivot = pivot.reindex(columns=[c for c in age_order if c in pivot.columns])
    fig3 = px.imshow(pivot, text_auto=True, color_continuous_scale="Greens",
                     title="Service interest % by age group",
                     aspect="auto")
    fig3.update_layout(margin=dict(t=40,b=10), height=350)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Spending & income ────────────────────────────────────────────────────
    st.subheader("Monthly Budget vs Income Band")
    col3, col4 = st.columns(2)
    income_order = ["<20k","20k-50k","50k-1L","1L-2L",">2L"]
    with col3:
        box_df = dff[dff["income_band"].isin(income_order)].copy()
        box_df["income_band"] = pd.Categorical(box_df["income_band"], categories=income_order, ordered=True)
        fig4 = px.box(box_df.sort_values("income_band"), x="income_band", y="monthly_budget",
                      color="income_band", title="Budget distribution by income",
                      labels={"income_band":"Income Band","monthly_budget":"Monthly Budget (₹)"},
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig4.update_layout(margin=dict(t=40,b=10), height=320, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)
    with col4:
        spend_counts = dff["past_health_spend"].value_counts().reset_index()
        spend_counts.columns = ["Spend Range","Count"]
        fig5 = px.bar(spend_counts, x="Spend Range", y="Count",
                      color="Count", color_continuous_scale="Purples",
                      title="Past health spending (last 3 months)")
        fig5.update_layout(margin=dict(t=40,b=10), height=320, coloraxis_showscale=False)
        st.plotly_chart(fig5, use_container_width=True)

    # ── Activity & food preference ────────────────────────────────────────────
    st.subheader("Lifestyle Breakdown")
    col5, col6 = st.columns(2)
    with col5:
        act = dff["activity_level"].value_counts().reset_index()
        act.columns = ["Activity Level","Count"]
        fig6 = px.pie(act, names="Activity Level", values="Count", hole=0.4,
                      color_discrete_sequence=px.colors.sequential.Teal,
                      title="Activity level distribution")
        fig6.update_layout(margin=dict(t=40,b=10), height=300)
        st.plotly_chart(fig6, use_container_width=True)
    with col6:
        fp = dff["food_pref"].value_counts().reset_index()
        fp.columns = ["Food Preference","Count"]
        fig7 = px.bar(fp, x="Food Preference", y="Count",
                      color="Food Preference",
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      title="Food preference split")
        fig7.update_layout(margin=dict(t=40,b=10), height=300, showlegend=False)
        st.plotly_chart(fig7, use_container_width=True)

    # ── Barrier analysis ──────────────────────────────────────────────────────
    st.subheader("Biggest Barrier to Following a Health Plan")
    bar_df = dff["barrier"].value_counts().reset_index()
    bar_df.columns = ["Barrier","Count"]
    fig8 = px.bar(bar_df, x="Count", y="Barrier", orientation="h",
                  color="Count", color_continuous_scale="Reds",
                  title="What stops people from following health plans")
    fig8.update_layout(margin=dict(t=40,b=10), height=300,
                       coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig8, use_container_width=True)

    # ── Discovery channel ─────────────────────────────────────────────────────
    st.subheader("How Customers Discover Health Apps")
    disc = dff["discovery_channel"].value_counts().reset_index()
    disc.columns = ["Channel","Count"]
    fig9 = px.funnel(disc, x="Count", y="Channel",
                     color_discrete_sequence=["#1D9E75"])
    fig9.update_layout(margin=dict(t=10,b=10), height=300)
    st.plotly_chart(fig9, use_container_width=True)

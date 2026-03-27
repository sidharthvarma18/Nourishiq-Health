"""page_home.py — Executive summary KPIs"""

import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_data, clean_bmi


def show():
    st.title("🥗 NourishIQ — Analytics Dashboard")
    st.markdown("**Data-driven decision making for India's personalised health & diet app.**")
    st.markdown("---")

    df = load_data()
    df = clean_bmi(df)

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Respondents",   f"{len(df):,}")
    c2.metric("High Intent",         f"{(df['signup_intent']=='High').sum():,}",
              f"{(df['signup_intent']=='High').mean()*100:.1f}%")
    c3.metric("Avg Monthly Budget",  f"₹{df['monthly_budget'].mean():.0f}")
    c4.metric("Avg BMI",             f"{df['bmi_clean'].mean():.1f}")
    c5.metric("Metro Respondents",   f"{(df['city_tier']=='Metro').sum():,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sign-up Intent Distribution")
        intent_counts = df["signup_intent"].value_counts().reset_index()
        intent_counts.columns = ["Intent", "Count"]
        color_map = {"High": "#1D9E75", "Medium": "#EF9F27", "Low": "#E24B4A"}
        fig = px.pie(intent_counts, names="Intent", values="Count",
                     color="Intent", color_discrete_map=color_map,
                     hole=0.45)
        fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Age Group Breakdown")
        age_order = ["Under 18","18-24","25-34","35-44","45-60","60+"]
        age_counts = df["age_group"].value_counts().reindex(age_order).reset_index()
        age_counts.columns = ["Age Group","Count"]
        fig2 = px.bar(age_counts, x="Age Group", y="Count",
                      color="Count", color_continuous_scale="Teal",
                      text="Count")
        fig2.update_layout(margin=dict(t=10,b=10), height=300,
                           coloraxis_showscale=False)
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("City Tier Distribution")
        city_counts = df["city_tier"].value_counts().reset_index()
        city_counts.columns = ["City Tier","Count"]
        fig3 = px.bar(city_counts, x="City Tier", y="Count",
                      color="City Tier",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig3.update_layout(margin=dict(t=10,b=10), height=280, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Top Fitness Goals")
        goal_counts = df["fitness_goal"].value_counts().reset_index()
        goal_counts.columns = ["Goal","Count"]
        fig4 = px.bar(goal_counts, x="Count", y="Goal", orientation="h",
                      color="Count", color_continuous_scale="Purples")
        fig4.update_layout(margin=dict(t=10,b=10), height=280,
                           coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"Showing 20 of {len(df)} rows. Full dataset: {len(df.columns)} columns.")

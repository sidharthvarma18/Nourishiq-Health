"""page_diagnostic.py — Clustering (K-Means) for customer personas"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils import load_data, clean_bmi, encode_features, PERSONA_LABELS


def show():
    st.title("🔍 Diagnostic Analysis — Customer Clustering")
    st.markdown("**Why is it happening?** — Discover hidden customer personas using K-Means clustering.")
    st.markdown("---")

    df = load_data()
    df = clean_bmi(df)
    enc = encode_features(df)

    cluster_features = [
        "age_group","income_band","activity_level","eating_habit",
        "bmi_clean","fitness_goal","food_pref","sleep_hours",
        "data_sharing_comfort","health_personality","paid_before",
        "city_tier","recommend_likelihood",
    ]
    cluster_features = [c for c in cluster_features if c in enc.columns]
    X_cluster = enc[cluster_features].fillna(0)

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # ── Elbow chart ───────────────────────────────────────────────────────────
    st.subheader("Optimal Number of Clusters — Elbow Method")
    inertias = []
    K_range  = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig_elbow = px.line(x=list(K_range), y=inertias, markers=True,
                        labels={"x":"Number of Clusters (K)","y":"Inertia"},
                        title="Elbow curve — pick K at the bend")
    fig_elbow.update_traces(line_color="#1D9E75", marker_color="#1D9E75")
    fig_elbow.update_layout(margin=dict(t=40,b=10), height=320)
    st.plotly_chart(fig_elbow, use_container_width=True)

    # ── Choose K ──────────────────────────────────────────────────────────────
    k = st.slider("Select number of clusters (K)", min_value=2, max_value=8, value=5)

    km_final = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels   = km_final.fit_predict(X_scaled)
    df       = df.copy()
    df["cluster"] = labels

    # ── PCA scatter ───────────────────────────────────────────────────────────
    st.subheader("Cluster Scatter (PCA — 2D)")
    pca  = PCA(n_components=2, random_state=42)
    comp = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({"PC1":comp[:,0],"PC2":comp[:,1],
                           "Cluster":labels.astype(str)})

    color_seq = px.colors.qualitative.Bold
    fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster",
                         color_discrete_sequence=color_seq,
                         title=f"K-Means clusters projected to 2D (PCA — explains {pca.explained_variance_ratio_.sum()*100:.1f}% variance)",
                         opacity=0.7)
    fig_pca.update_traces(marker_size=5)
    fig_pca.update_layout(margin=dict(t=40,b=10), height=400)
    st.plotly_chart(fig_pca, use_container_width=True)

    # ── Persona cards ─────────────────────────────────────────────────────────
    st.subheader("Customer Persona Cards")
    cols = st.columns(min(k, 5))

    for ci in range(k):
        sub = df[df["cluster"]==ci]
        label, icon, color = PERSONA_LABELS.get(ci, (f"Persona {ci}", "👤", "#888"))
        with cols[ci % len(cols)]:
            top_goal = sub["fitness_goal"].mode()[0] if len(sub) > 0 else "—"
            top_city = sub["city_tier"].mode()[0]    if len(sub) > 0 else "—"
            top_age  = sub["age_group"].mode()[0]    if len(sub) > 0 else "—"
            avg_bud  = sub["monthly_budget"].mean()
            intent_h = (sub["signup_intent"]=="High").mean()*100

            st.markdown(f"""
<div style="border:1px solid {color};border-radius:10px;padding:14px;margin-bottom:8px;">
<h4 style="color:{color};margin:0">{icon} {label}</h4>
<p style="margin:4px 0;font-size:13px"><b>Size:</b> {len(sub)} respondents</p>
<p style="margin:4px 0;font-size:13px"><b>Top age:</b> {top_age}</p>
<p style="margin:4px 0;font-size:13px"><b>Top city:</b> {top_city}</p>
<p style="margin:4px 0;font-size:13px"><b>Top goal:</b> {top_goal}</p>
<p style="margin:4px 0;font-size:13px"><b>Avg budget:</b> ₹{avg_bud:.0f}</p>
<p style="margin:4px 0;font-size:13px"><b>High intent:</b> {intent_h:.0f}%</p>
</div>""", unsafe_allow_html=True)

    # ── Cluster feature heatmap ───────────────────────────────────────────────
    st.subheader("Cluster Feature Profiles")
    raw_features = ["age_group","income_band","activity_level","eating_habit",
                    "bmi_clean","fitness_goal","paid_before","city_tier"]
    raw_features = [c for c in raw_features if c in enc.columns]

    cluster_means = enc.copy()
    cluster_means["cluster"] = labels
    profile = cluster_means.groupby("cluster")[raw_features].mean()
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

    fig_heat = px.imshow(profile_norm.T,
                         labels=dict(x="Cluster", y="Feature", color="Normalised Value"),
                         color_continuous_scale="Teal",
                         text_auto=".2f",
                         title="Normalised feature means per cluster",
                         aspect="auto")
    fig_heat.update_layout(margin=dict(t=40,b=10), height=380)
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Discount recommendation ───────────────────────────────────────────────
    st.subheader("Recommended Offers by Cluster")
    offer_data = []
    for ci in range(k):
        sub = df[df["cluster"]==ci]
        avg_bud  = sub["monthly_budget"].mean()
        intent_h = (sub["signup_intent"]=="High").mean()
        if intent_h > 0.6 and avg_bud > 400:
            offer = "Full Pro Plan — no discount needed"
            channel = "Direct Instagram ad"
        elif intent_h > 0.4 and avg_bud <= 400:
            offer = "15% student/family discount on Pro"
            channel = "WhatsApp campaign"
        elif intent_h > 0.3:
            offer = "Free trial (7 days) → upsell"
            channel = "Google search + doctor referral"
        else:
            offer = "Nurture via free tier content"
            channel = "Organic social / YouTube"
        label, _, _ = PERSONA_LABELS.get(ci, (f"Cluster {ci}", "", ""))
        offer_data.append({"Cluster":ci,"Persona":label,
                           "Avg Budget":f"₹{avg_bud:.0f}",
                           "High Intent %":f"{intent_h*100:.0f}%",
                           "Recommended Offer":offer,
                           "Best Channel":channel})

    st.dataframe(pd.DataFrame(offer_data), use_container_width=True)

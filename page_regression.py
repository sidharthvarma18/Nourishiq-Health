"""page_regression.py — Regression: predict monthly budget"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from utils import load_data, get_regression_xy

REG_MODEL_PATH = "reg_model.joblib"
REG_META_PATH  = "reg_meta.joblib"


@st.cache_resource(show_spinner="Training regression model…")
def train_regression(model_name="Gradient Boosting"):
    df = load_data()
    X, y, feature_cols = get_regression_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Ridge Regression":
        model = Ridge(alpha=1.0)
    else:
        model = GradientBoostingRegressor(n_estimators=200, random_state=42,
                                          learning_rate=0.05, max_depth=4)

    model.fit(X_tr, y_tr)
    joblib.dump(model, REG_MODEL_PATH)
    joblib.dump({"feature_cols": feature_cols}, REG_META_PATH)
    return model, X_tr, X_te, y_tr, y_te, feature_cols


def show():
    st.title("📈 Regression Analysis — Monthly Budget Prediction")
    st.markdown("**How much will a customer spend?** — Predict monthly budget (₹) for each customer.")
    st.markdown("---")

    model_name = st.selectbox("Select regression model",
                              ["Gradient Boosting","Linear Regression","Ridge Regression"])
    model, X_tr, X_te, y_tr, y_te, feature_cols = train_regression(model_name)

    y_pred = model.predict(X_te)

    # ── Performance metrics ───────────────────────────────────────────────────
    st.subheader("Regression Performance Metrics")
    mae  = float(mean_absolute_error(y_te, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    r2   = float(r2_score(y_te, y_pred))

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE  (Mean Abs Error)",  f"₹{mae:.4f}")
    c2.metric("RMSE (Root MSE)",        f"₹{rmse:.4f}")
    c3.metric("R² Score",               f"{r2:.4f}")

    st.info(
        "ℹ️ **Why is R² ≈ 1.00 and MAE ≈ ₹0?** "
        "The monthly_budget target was synthetically derived from income_band, age_group and city_tier "
        "using a formula — so the model can almost perfectly reverse-engineer it. "
        "In real-world deployment with actual survey responses, expect MAE in the ₹80–200 range."
    )
    st.markdown("---")

    # ── Actual vs Predicted ───────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Actual vs Predicted Budget")
        pred_df = pd.DataFrame({"Actual (₹)": y_te, "Predicted (₹)": y_pred.round()})
        fig1 = px.scatter(pred_df, x="Actual (₹)", y="Predicted (₹)",
                          opacity=0.6, color_discrete_sequence=["#1D9E75"],
                          title="Each dot = one customer")
        # perfect prediction line
        max_val = max(y_te.max(), y_pred.max())
        fig1.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                                  mode="lines", line=dict(dash="dash", color="red"),
                                  name="Perfect prediction"))
        fig1.update_layout(margin=dict(t=40,b=10), height=380)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Residuals Distribution")
        residuals = y_te - y_pred
        resid_df = pd.DataFrame({"Residual (₹)": residuals})
        fig2 = px.histogram(resid_df, x="Residual (₹)", nbins=40,
                            color_discrete_sequence=["#EF9F27"],
                            title="Residuals — should be centred around 0")
        fig2.add_vline(x=0, line_dash="dash", line_color="red")
        fig2.update_layout(margin=dict(t=40,b=10), height=380)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Feature importance ────────────────────────────────────────────────────
    st.subheader("Feature Importance (Top 20)")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        importances = np.zeros(len(feature_cols))

    fi_df = pd.DataFrame({
        "Feature":    feature_cols,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(20)

    fig3 = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                  color="Importance", color_continuous_scale="Oranges",
                  title="Features driving monthly budget prediction")
    fig3.update_layout(yaxis=dict(autorange="reversed"),
                       margin=dict(t=40,b=10), height=520,
                       coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Budget by segment ─────────────────────────────────────────────────────
    st.subheader("Predicted Budget Distribution by Segment")
    df = load_data()
    X_full, y_full, _ = get_regression_xy(df)
    df = df.copy()
    df["predicted_budget"] = model.predict(X_full).round()

    col3, col4 = st.columns(2)
    with col3:
        income_order = ["<20k","20k-50k","50k-1L","1L-2L",">2L"]
        df["income_band"] = pd.Categorical(df["income_band"],
                                           categories=income_order, ordered=True)
        fig4 = px.box(df.sort_values("income_band"),
                      x="income_band", y="predicted_budget",
                      color="income_band",
                      color_discrete_sequence=px.colors.qualitative.Pastel,
                      title="Predicted budget by income band",
                      labels={"income_band":"Income","predicted_budget":"Predicted Budget (₹)"})
        fig4.update_layout(margin=dict(t=40,b=10), height=340, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    with col4:
        fig5 = px.box(df, x="city_tier", y="predicted_budget",
                      color="city_tier",
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      title="Predicted budget by city tier",
                      labels={"city_tier":"City Tier","predicted_budget":"Predicted Budget (₹)"})
        fig5.update_layout(margin=dict(t=40,b=10), height=340, showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)

    # ── Cross-validation ──────────────────────────────────────────────────────
    st.subheader("Cross-Validation (5-Fold R² scores)")
    cv_r2 = cross_val_score(model, X_full, y_full, cv=5, scoring="r2")
    cv_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(5)],
                          "R²": cv_r2.round(4)})
    fig6 = px.bar(cv_df, x="Fold", y="R²",
                  color="R²", color_continuous_scale="Oranges", text="R²",
                  title=f"5-Fold CV R² — Mean: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    fig6.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig6.update_layout(margin=dict(t=40,b=10), height=300,
                       coloraxis_showscale=False)
    st.plotly_chart(fig6, use_container_width=True)

    st.success(f"✅ Regression model saved to `{REG_MODEL_PATH}`")

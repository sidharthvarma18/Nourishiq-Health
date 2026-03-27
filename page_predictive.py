"""page_predictive.py — Classification: predict signup intent"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import joblib, os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report)
from sklearn.preprocessing import label_binarize

from utils import load_data, get_ml_xy

MODEL_PATH = "clf_model.joblib"
META_PATH  = "clf_meta.joblib"


@st.cache_resource(show_spinner="Training classifier…")
def train_model(model_name="Random Forest"):
    df = load_data()
    X, y, label_names, feature_cols = get_ml_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                               random_state=42, stratify=y)
    if model_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        clf = GradientBoostingClassifier(n_estimators=150, random_state=42)

    clf.fit(X_tr, y_tr)
    joblib.dump(clf,  MODEL_PATH)
    joblib.dump({"feature_cols": feature_cols, "label_names": label_names}, META_PATH)
    return clf, X_tr, X_te, y_tr, y_te, feature_cols, label_names


def show():
    st.title("🤖 Predictive Analysis — Classification")
    st.markdown("**What will happen?** — Predict whether a customer is High / Medium / Low intent to sign up.")
    st.markdown("---")

    model_choice = st.selectbox("Select classifier", ["Random Forest","Gradient Boosting"])
    clf, X_tr, X_te, y_tr, y_te, feature_cols, label_names = train_model(model_choice)

    y_pred  = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)

    # ── Performance metrics ───────────────────────────────────────────────────
    st.subheader("Model Performance Metrics")
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_te, y_pred,    average="weighted", zero_division=0)
    f1   = f1_score(y_te, y_pred,        average="weighted", zero_division=0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{acc*100:.2f}%")
    c2.metric("Precision", f"{prec*100:.2f}%")
    c3.metric("Recall",    f"{rec*100:.2f}%")
    c4.metric("F1-Score",  f"{f1*100:.2f}%")

    st.markdown("---")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_te, y_pred)
        fig_cm = ff.create_annotated_heatmap(
            z=cm, x=label_names, y=label_names,
            colorscale="Greens", showscale=True,
            annotation_text=cm.astype(str))
        fig_cm.update_layout(
            xaxis_title="Predicted", yaxis_title="Actual",
            margin=dict(t=10,b=50,l=50,r=10), height=320)
        fig_cm["data"][0]["showscale"] = True
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── Per-class metrics table ────────────────────────────────────────────────
    with col2:
        st.subheader("Per-Class Report")
        report = classification_report(y_te, y_pred,
                                       target_names=label_names,
                                       output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).T.round(3)
        report_df = report_df.drop(index=["accuracy","macro avg","weighted avg"], errors="ignore")
        report_df.index.name = "Class"
        st.dataframe(report_df.style.background_gradient(cmap="Greens", subset=["precision","recall","f1-score"]),
                     use_container_width=True)
        st.caption("Precision, Recall, F1-score per intent class.")

    st.markdown("---")

    # ── ROC Curves ────────────────────────────────────────────────────────────
    st.subheader("ROC Curves (One-vs-Rest per class)")
    n_classes  = len(label_names)
    y_te_bin   = label_binarize(y_te,   classes=list(range(n_classes)))

    fig_roc = go.Figure()
    colors  = ["#1D9E75","#EF9F27","#E24B4A"]
    for i, (cls_name, color) in enumerate(zip(label_names, colors)):
        fpr, tpr, _ = roc_curve(y_te_bin[:, i], y_proba[:, i])
        roc_auc     = auc(fpr, tpr)
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{cls_name} (AUC={roc_auc:.2f})",
            line=dict(color=color, width=2)))

    fig_roc.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode="lines",
        line=dict(dash="dash", color="gray", width=1),
        name="Random baseline", showlegend=True))
    fig_roc.update_layout(
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        legend=dict(x=0.6, y=0.1),
        margin=dict(t=10,b=40,l=40,r=10), height=380)
    st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")

    # ── Feature Importance ────────────────────────────────────────────────────
    st.subheader("Feature Importance (Top 20)")
    importances = clf.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature":   feature_cols,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(20)

    fig_imp = px.bar(feat_imp_df, x="Importance", y="Feature",
                     orientation="h", color="Importance",
                     color_continuous_scale="Teal",
                     title="Top 20 features driving signup intent prediction")
    fig_imp.update_layout(yaxis=dict(autorange="reversed"),
                          margin=dict(t=40,b=10), height=520,
                          coloraxis_showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    # ── Cross-validation ──────────────────────────────────────────────────────
    st.subheader("Cross-Validation (5-Fold)")
    df_full = load_data()
    X_full, y_full, _, _ = get_ml_xy(df_full)
    cv_scores = cross_val_score(clf, X_full, y_full, cv=5, scoring="accuracy")
    cv_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(5)],
                          "Accuracy": (cv_scores * 100).round(2)})
    fig_cv = px.bar(cv_df, x="Fold", y="Accuracy",
                    color="Accuracy", color_continuous_scale="Greens",
                    text="Accuracy",
                    title=f"5-Fold CV — Mean: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    fig_cv.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_cv.update_layout(margin=dict(t=40,b=10), height=320,
                         coloraxis_showscale=False, yaxis_range=[0,110])
    st.plotly_chart(fig_cv, use_container_width=True)

    st.success(f"✅ Model saved to `{MODEL_PATH}` — used by the New Lead Scorer page.")
